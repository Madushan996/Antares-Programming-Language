//! Tensor maths on the graphics card (GPU).
//!
//! Antares talks to the GPU through Direct3D 11 compute shaders: small
//! programs, written in HLSL, that the GPU runs on thousands of numbers at
//! once. Every Windows PC with a graphics card (or integrated graphics) has
//! Direct3D 11, and Windows itself includes the shader compiler, so nothing
//! needs installing. The libraries are loaded when first needed, so a
//! computer without them simply has no GPU: Antares keeps using the CPU.
//!
//! The GPU works in 32-bit numbers (about 7 significant digits), where the
//! CPU uses 64-bit ones (about 16), so results agree to roughly 6 digits.
//! That is why the GPU is something a program turns on rather than a
//! change underneath existing programs.
//!
//! macOS and Linux have no GPU support yet: `available()` is `None` there.

#[cfg(windows)]
pub use win::{available, matmul};

#[cfg(not(windows))]
pub fn available() -> Option<String> {
    None
}

#[cfg(not(windows))]
pub fn matmul(_: &[f64], _: &[f64], _: usize, _: usize, _: usize) -> Result<Vec<f64>, String> {
    Err("the GPU isn't available on this computer".into())
}

/// gpu(): the name of the graphics card Antares would use, or nothing.
pub fn b_gpu(_: &mut crate::interp::Interp, a: Vec<crate::value::Value>) -> Result<crate::value::Value, String> {
    crate::builtins::want(&a, 0, "gpu")?;
    Ok(match available() {
        Some(name) => crate::value::Value::Str(name.into()),
        None => crate::value::Value::Nothing,
    })
}

/// The matrix-multiply shader: each group of 16 x 16 threads works out a
/// 16 x 16 tile of the result, reading tiles of A and B into fast shared
/// memory once instead of every thread reading them separately.
#[cfg_attr(not(windows), allow(dead_code))]
const MATMUL_HLSL: &str = r#"
cbuffer Dims : register(b0) { uint M; uint N; uint P; uint pad; };
StructuredBuffer<float> A : register(t0);
StructuredBuffer<float> B : register(t1);
RWStructuredBuffer<float> C : register(u0);
#define T 16
groupshared float As[T][T];
groupshared float Bs[T][T];
[numthreads(T, T, 1)]
void main(uint3 g : SV_GroupID, uint3 t : SV_GroupThreadID) {
    uint row = g.y * T + t.y;
    uint col = g.x * T + t.x;
    float acc = 0;
    for (uint k0 = 0; k0 < N; k0 += T) {
        uint ka = k0 + t.x;
        uint kb = k0 + t.y;
        As[t.y][t.x] = (row < M && ka < N) ? A[row * N + ka] : 0;
        Bs[t.y][t.x] = (kb < N && col < P) ? B[kb * P + col] : 0;
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint k = 0; k < T; k++) acc += As[t.y][k] * Bs[k][t.x];
        GroupMemoryBarrierWithGroupSync();
    }
    if (row < M && col < P) C[row * P + col] = acc;
}
"#;

#[cfg(windows)]
mod win {
    use super::MATMUL_HLSL;
    use std::cell::RefCell;
    use std::ffi::c_void;

    type Ptr = *mut c_void;
    type HResult = i32;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryW(name: *const u16) -> Ptr;
        fn GetProcAddress(module: Ptr, name: *const u8) -> Ptr;
    }

    #[repr(C)]
    struct Guid(u32, u16, u16, [u8; 8]);

    const IID_DXGI_FACTORY1: Guid = Guid(0x770aae78, 0xf26f, 0x4dba, [0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87]);

    #[repr(C)]
    struct AdapterDesc1 {
        description: [u16; 128],
        vendor: u32,
        device: u32,
        subsys: u32,
        revision: u32,
        dedicated_video: usize,
        dedicated_system: usize,
        shared_system: usize,
        luid: [u32; 2],
        flags: u32,
    }

    #[repr(C)]
    struct BufferDesc {
        byte_width: u32,
        usage: u32,
        bind_flags: u32,
        cpu_access: u32,
        misc: u32,
        stride: u32,
    }

    #[repr(C)]
    struct SubresourceData {
        mem: *const c_void,
        pitch: u32,
        slice_pitch: u32,
    }

    #[repr(C)]
    struct Mapped {
        data: Ptr,
        row_pitch: u32,
        depth_pitch: u32,
    }

    const USAGE_DEFAULT: u32 = 0;
    const USAGE_STAGING: u32 = 3;
    const BIND_CONSTANT: u32 = 0x4;
    const BIND_SHADER_RESOURCE: u32 = 0x8;
    const BIND_UNORDERED_ACCESS: u32 = 0x80;
    const CPU_READ: u32 = 0x20000;
    const MISC_STRUCTURED: u32 = 0x40;
    const MAP_READ: u32 = 1;

    /// Calls method `index` of a COM object's table of functions.
    unsafe fn method<F: Copy>(obj: Ptr, index: usize) -> F {
        let table = unsafe { *(obj as *const *const Ptr) };
        let f = unsafe { *table.add(index) };
        unsafe { std::mem::transmute_copy(&f) }
    }

    unsafe fn release(obj: Ptr) {
        if !obj.is_null() {
            let f: unsafe extern "system" fn(Ptr) -> u32 = unsafe { method(obj, 2) };
            unsafe { f(obj) };
        }
    }

    /// A COM object, released when dropped.
    struct Com(Ptr);

    impl Drop for Com {
        fn drop(&mut self) {
            unsafe { release(self.0) }
        }
    }

    type CompileFn = unsafe extern "system" fn(*const c_void, usize, *const u8, Ptr, Ptr, *const u8, *const u8, u32, u32, *mut Ptr, *mut Ptr) -> HResult;

    struct Gpu {
        name: String,
        device: Com,
        context: Com,
        compile: CompileFn,
        matmul: Com,
    }

    thread_local! {
        /// The GPU, set up the first time it is asked for (None: there isn't one).
        static GPU: RefCell<Option<Option<Gpu>>> = const { RefCell::new(None) };
    }

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain([0]).collect()
    }

    fn load(dll: &str, func: &str) -> Option<Ptr> {
        let m = unsafe { LoadLibraryW(wide(dll).as_ptr()) };
        if m.is_null() {
            return None;
        }
        let name: Vec<u8> = func.bytes().chain([0]).collect();
        let f = unsafe { GetProcAddress(m, name.as_ptr()) };
        (!f.is_null()).then_some(f)
    }

    /// The graphics card with the most memory of its own, skipping
    /// Windows' software renderer. Gives the adapter and its name.
    fn best_adapter() -> Option<(Com, String)> {
        let create: unsafe extern "system" fn(*const Guid, *mut Ptr) -> HResult = unsafe { std::mem::transmute(load("dxgi.dll", "CreateDXGIFactory1")?) };
        let mut factory: Ptr = std::ptr::null_mut();
        if unsafe { create(&IID_DXGI_FACTORY1, &mut factory) } < 0 || factory.is_null() {
            return None;
        }
        let factory = Com(factory);
        let enum_adapters: unsafe extern "system" fn(Ptr, u32, *mut Ptr) -> HResult = unsafe { method(factory.0, 12) };
        let mut best: Option<(Com, String, usize)> = None;
        for i in 0..16 {
            let mut a: Ptr = std::ptr::null_mut();
            if unsafe { enum_adapters(factory.0, i, &mut a) } < 0 || a.is_null() {
                break;
            }
            let a = Com(a);
            let get_desc: unsafe extern "system" fn(Ptr, *mut AdapterDesc1) -> HResult = unsafe { method(a.0, 10) };
            let mut d: AdapterDesc1 = unsafe { std::mem::zeroed() };
            if unsafe { get_desc(a.0, &mut d) } < 0 || d.flags & 2 != 0 {
                continue; // a software adapter
            }
            let len = d.description.iter().position(|c| *c == 0).unwrap_or(128);
            let name = String::from_utf16_lossy(&d.description[..len]).trim().to_string();
            if best.as_ref().is_none_or(|b| d.dedicated_video > b.2) {
                best = Some((a, name, d.dedicated_video));
            }
        }
        best.map(|(a, n, _)| (a, n))
    }

    fn setup() -> Option<Gpu> {
        let compile: CompileFn = unsafe { std::mem::transmute(load("d3dcompiler_47.dll", "D3DCompile")?) };
        type CreateFn = unsafe extern "system" fn(Ptr, u32, Ptr, u32, *const u32, u32, u32, *mut Ptr, *mut u32, *mut Ptr) -> HResult;
        let create: CreateFn = unsafe { std::mem::transmute(load("d3d11.dll", "D3D11CreateDevice")?) };
        // ANTARES_GPU=warp uses Windows' software GPU (for tests on machines without one)
        let warp = std::env::var("ANTARES_GPU").is_ok_and(|v| v.eq_ignore_ascii_case("warp"));
        let (adapter, name) = if warp { (Com(std::ptr::null_mut()), "Microsoft Basic Render Driver (software)".to_string()) } else { best_adapter()? };
        let levels = [0xb000u32]; // feature level 11_0: compute shaders 5.0
        let (mut device, mut context, mut level): (Ptr, Ptr, u32) = (std::ptr::null_mut(), std::ptr::null_mut(), 0);
        let driver = if warp { 5 } else { 0 }; // D3D_DRIVER_TYPE_WARP, or UNKNOWN with an adapter
        let hr = unsafe { create(adapter.0, driver, std::ptr::null_mut(), 0, levels.as_ptr(), 1, 7, &mut device, &mut level, &mut context) };
        if hr < 0 || device.is_null() || context.is_null() {
            return None;
        }
        let (device, context) = (Com(device), Com(context));
        let mut g = Gpu { name, device, context, compile, matmul: Com(std::ptr::null_mut()) };
        g.matmul = g.shader(MATMUL_HLSL).ok()?;
        Some(g)
    }

    impl Gpu {
        /// Compiles HLSL source (entry point `main`) into a compute shader.
        fn shader(&self, src: &str) -> Result<Com, String> {
            let (mut code, mut errors): (Ptr, Ptr) = (std::ptr::null_mut(), std::ptr::null_mut());
            let hr = unsafe {
                (self.compile)(src.as_ptr().cast(), src.len(), c"antares".as_ptr().cast(), std::ptr::null_mut(), std::ptr::null_mut(), c"main".as_ptr().cast(), c"cs_5_0".as_ptr().cast(), 1 << 15, 0, &mut code, &mut errors)
            };
            let (code, errors) = (Com(code), Com(errors));
            if hr < 0 || code.0.is_null() {
                let msg = if errors.0.is_null() { String::new() } else { unsafe { blob_text(errors.0) } };
                return Err(format!("the GPU program didn't compile: {}", msg.trim()));
            }
            let ptr: unsafe extern "system" fn(Ptr) -> *const c_void = unsafe { method(code.0, 3) };
            let size: unsafe extern "system" fn(Ptr) -> usize = unsafe { method(code.0, 4) };
            let create: unsafe extern "system" fn(Ptr, *const c_void, usize, Ptr, *mut Ptr) -> HResult = unsafe { method(self.device.0, 18) };
            let mut cs: Ptr = std::ptr::null_mut();
            if unsafe { create(self.device.0, ptr(code.0), size(code.0), std::ptr::null_mut(), &mut cs) } < 0 {
                return Err("the GPU didn't accept the program".into());
            }
            Ok(Com(cs))
        }

        fn buffer(&self, desc: BufferDesc, init: Option<&[u8]>) -> Result<Com, String> {
            let create: unsafe extern "system" fn(Ptr, *const BufferDesc, *const SubresourceData, *mut Ptr) -> HResult = unsafe { method(self.device.0, 3) };
            let data = init.map(|b| SubresourceData { mem: b.as_ptr().cast(), pitch: 0, slice_pitch: 0 });
            let mut buf: Ptr = std::ptr::null_mut();
            let hr = unsafe { create(self.device.0, &desc, data.as_ref().map_or(std::ptr::null(), |d| d as *const _), &mut buf) };
            if hr < 0 || buf.is_null() {
                return Err(format!("the GPU couldn't make room for {} numbers", desc.byte_width / 4));
            }
            Ok(Com(buf))
        }

        /// Numbers for the GPU to read.
        fn input(&self, xs: &[f32]) -> Result<(Com, Com), String> {
            let bytes: &[u8] = unsafe { std::slice::from_raw_parts(xs.as_ptr().cast(), xs.len().max(1) * 4) };
            let pad = [0u8; 4];
            let init = if xs.is_empty() { &pad[..] } else { bytes };
            let buf = self.buffer(BufferDesc { byte_width: init.len() as u32, usage: USAGE_DEFAULT, bind_flags: BIND_SHADER_RESOURCE, cpu_access: 0, misc: MISC_STRUCTURED, stride: 4 }, Some(init))?;
            let create: unsafe extern "system" fn(Ptr, Ptr, *const c_void, *mut Ptr) -> HResult = unsafe { method(self.device.0, 7) };
            let mut view: Ptr = std::ptr::null_mut();
            if unsafe { create(self.device.0, buf.0, std::ptr::null(), &mut view) } < 0 {
                return Err("the GPU couldn't read the numbers".into());
            }
            Ok((buf, Com(view)))
        }

        /// Room for `n` numbers for the GPU to write.
        fn output(&self, n: usize) -> Result<(Com, Com), String> {
            let buf = self.buffer(BufferDesc { byte_width: (n.max(1) * 4) as u32, usage: USAGE_DEFAULT, bind_flags: BIND_UNORDERED_ACCESS, cpu_access: 0, misc: MISC_STRUCTURED, stride: 4 }, None)?;
            let create: unsafe extern "system" fn(Ptr, Ptr, *const c_void, *mut Ptr) -> HResult = unsafe { method(self.device.0, 8) };
            let mut view: Ptr = std::ptr::null_mut();
            if unsafe { create(self.device.0, buf.0, std::ptr::null(), &mut view) } < 0 {
                return Err("the GPU couldn't write the numbers".into());
            }
            Ok((buf, Com(view)))
        }

        /// Small numbers every thread can see (a constant buffer).
        fn constants(&self, words: [u32; 4]) -> Result<Com, String> {
            let bytes: &[u8] = unsafe { std::slice::from_raw_parts(words.as_ptr().cast(), 16) };
            self.buffer(BufferDesc { byte_width: 16, usage: USAGE_DEFAULT, bind_flags: BIND_CONSTANT, cpu_access: 0, misc: 0, stride: 0 }, Some(bytes))
        }

        /// Runs `shader` with these inputs and outputs, in (x, y, z) groups.
        fn run(&self, shader: &Com, consts: &Com, inputs: &[&Com], outputs: &[&Com], groups: (u32, u32, u32)) {
            let c = self.context.0;
            unsafe {
                let set_shader: unsafe extern "system" fn(Ptr, Ptr, *const Ptr, u32) = method(c, 69);
                let set_srvs: unsafe extern "system" fn(Ptr, u32, u32, *const Ptr) = method(c, 67);
                let set_uavs: unsafe extern "system" fn(Ptr, u32, u32, *const Ptr, *const u32) = method(c, 68);
                let set_cbs: unsafe extern "system" fn(Ptr, u32, u32, *const Ptr) = method(c, 71);
                let dispatch: unsafe extern "system" fn(Ptr, u32, u32, u32) = method(c, 41);
                set_shader(c, shader.0, std::ptr::null(), 0);
                let srvs: Vec<Ptr> = inputs.iter().map(|v| v.0).collect();
                let uavs: Vec<Ptr> = outputs.iter().map(|v| v.0).collect();
                set_srvs(c, 0, srvs.len() as u32, srvs.as_ptr());
                set_uavs(c, 0, uavs.len() as u32, uavs.as_ptr(), std::ptr::null());
                set_cbs(c, 0, 1, &consts.0);
                dispatch(c, groups.0, groups.1, groups.2);
                // unbind, so the buffers can be read back or reused
                let none = [std::ptr::null_mut::<c_void>(); 8];
                set_srvs(c, 0, srvs.len() as u32, none.as_ptr());
                set_uavs(c, 0, uavs.len() as u32, none.as_ptr(), std::ptr::null());
            }
        }

        /// Copies `n` numbers back from the GPU.
        fn read(&self, buf: &Com, n: usize) -> Result<Vec<f32>, String> {
            let staging = self.buffer(BufferDesc { byte_width: (n.max(1) * 4) as u32, usage: USAGE_STAGING, bind_flags: 0, cpu_access: CPU_READ, misc: MISC_STRUCTURED, stride: 4 }, None)?;
            let c = self.context.0;
            unsafe {
                let copy: unsafe extern "system" fn(Ptr, Ptr, Ptr) = method(c, 47);
                let map: unsafe extern "system" fn(Ptr, Ptr, u32, u32, u32, *mut Mapped) -> HResult = method(c, 14);
                let unmap: unsafe extern "system" fn(Ptr, Ptr, u32) = method(c, 15);
                copy(c, staging.0, buf.0);
                let mut m: Mapped = std::mem::zeroed();
                if map(c, staging.0, 0, MAP_READ, 0, &mut m) < 0 || m.data.is_null() {
                    return Err("couldn't read the GPU's answer".into());
                }
                let out = std::slice::from_raw_parts(m.data as *const f32, n).to_vec();
                unmap(c, staging.0, 0);
                Ok(out)
            }
        }
    }

    unsafe fn blob_text(blob: Ptr) -> String {
        let ptr: unsafe extern "system" fn(Ptr) -> *const u8 = unsafe { method(blob, 3) };
        let size: unsafe extern "system" fn(Ptr) -> usize = unsafe { method(blob, 4) };
        let bytes = unsafe { std::slice::from_raw_parts(ptr(blob), size(blob)) };
        String::from_utf8_lossy(bytes).into_owned()
    }

    fn with_gpu<R>(f: impl FnOnce(&Gpu) -> R) -> Option<R> {
        GPU.with(|g| {
            let mut g = g.borrow_mut();
            if g.is_none() {
                *g = Some(setup());
            }
            g.as_ref().unwrap().as_ref().map(f)
        })
    }

    /// The graphics card's name, if there is one Antares can use.
    pub fn available() -> Option<String> {
        with_gpu(|g| g.name.clone())
    }

    /// The m x n matrix `a` times the n x p matrix `b`, worked out on the
    /// GPU in 32-bit numbers.
    pub fn matmul(a: &[f64], b: &[f64], m: usize, n: usize, p: usize) -> Result<Vec<f64>, String> {
        with_gpu(|g| {
            let fa: Vec<f32> = a.iter().map(|&x| x as f32).collect();
            let fb: Vec<f32> = b.iter().map(|&x| x as f32).collect();
            let (_ab, av) = g.input(&fa)?;
            let (_bb, bv) = g.input(&fb)?;
            let (cb, cv) = g.output(m * p)?;
            let consts = g.constants([m as u32, n as u32, p as u32, 0])?;
            g.run(&g.matmul, &consts, &[&av, &bv], &[&cv], (p.div_ceil(16) as u32, m.div_ceil(16) as u32, 1));
            Ok(g.read(&cb, m * p)?.into_iter().map(|x| x as f64).collect())
        })
        .unwrap_or_else(|| Err("the GPU isn't available on this computer".into()))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// cargo test --release --lib benchmark -- --ignored --nocapture
        #[test]
        #[ignore]
        fn benchmark_matmul() {
            let Some(name) = available() else { return };
            println!("GPU: {}", name);
            for &n in &[128usize, 256, 512, 1024, 1536] {
                let a = crate::tensor::new(vec![n, n], (0..n * n).map(|i| (i % 17) as f64 / 17.0).collect());
                let b = crate::tensor::new(vec![n, n], (0..n * n).map(|i| (i % 13) as f64 / 13.0).collect());
                let _ = matmul(&a.data.borrow(), &b.data.borrow(), n, n, n); // warm up
                let reps = if n <= 512 { 5 } else { 2 };
                let t = std::time::Instant::now();
                for _ in 0..reps {
                    let _ = crate::tensor::matmul(&a, &b).unwrap();
                }
                let cpu = t.elapsed().as_secs_f64() / reps as f64;
                let t = std::time::Instant::now();
                for _ in 0..reps {
                    let _ = matmul(&a.data.borrow(), &b.data.borrow(), n, n, n).unwrap();
                }
                let gpu = t.elapsed().as_secs_f64() / reps as f64;
                println!("{0} x {0}: CPU {1:.1} ms, GPU {2:.1} ms (with copying), {3:.1}x", n, cpu * 1e3, gpu * 1e3, cpu / gpu);
            }
        }

        #[test]
        fn matmul_on_the_gpu_matches_the_cpu() {
            if available().is_none() {
                eprintln!("no GPU here (set ANTARES_GPU=warp to test with the software one)");
                return;
            }
            // odd sizes, so the edges of the 16 x 16 tiles are exercised
            for &(m, n, p) in &[(1, 1, 1), (3, 5, 2), (17, 33, 31), (64, 64, 64), (100, 7, 129)] {
                let a: Vec<f64> = (0..m * n).map(|i| ((i * 7 % 13) as f64 - 6.0) / 3.0).collect();
                let b: Vec<f64> = (0..n * p).map(|i| ((i * 5 % 11) as f64 - 5.0) / 2.0).collect();
                let got = matmul(&a, &b, m, n, p).unwrap();
                for i in 0..m {
                    for j in 0..p {
                        let want: f64 = (0..n).map(|k| a[i * n + k] * b[k * p + j]).sum();
                        let g = got[i * p + j];
                        assert!((g - want).abs() <= 1e-4 * (1.0 + want.abs()), "{}x{}x{} at ({}, {}): {} vs {}", m, n, p, i, j, g, want);
                    }
                }
            }
        }
    }
}
