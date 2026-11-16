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
pub use win::{available, binary, download, matmul, matmul_bufs, unary, upload, Buf};

#[cfg(not(windows))]
pub use none::*;

/// Without GPU support, nothing ever reaches the GPU.
#[cfg(not(windows))]
mod none {
    const NO: &str = "the GPU isn't available on this computer";

    /// Numbers on the GPU (there are never any here).
    #[derive(Clone)]
    pub struct Buf(());

    pub fn available() -> Option<String> {
        None
    }
    pub fn matmul(_: &[f64], _: &[f64], _: usize, _: usize, _: usize) -> Result<Vec<f64>, String> {
        Err(NO.into())
    }
    pub fn upload(_: &[f64]) -> Result<Buf, String> {
        Err(NO.into())
    }
    pub fn download(_: &Buf, _: usize) -> Result<Vec<f64>, String> {
        Err(NO.into())
    }
    pub fn matmul_bufs(_: &Buf, _: &Buf, _: usize, _: usize, _: usize, _: bool, _: bool) -> Result<Buf, String> {
        Err(NO.into())
    }
    pub fn binary(_: u32, _: u32, _: &Buf, _: &Buf, _: usize, _: usize) -> Result<Buf, String> {
        Err(NO.into())
    }
    pub fn unary(_: u32, _: &Buf, _: usize) -> Result<Buf, String> {
        Err(NO.into())
    }
}

/// Two-input operations, by number: + - * / ^ and %.
pub const ADD: u32 = 0;
pub const SUB: u32 = 1;
pub const MUL: u32 = 2;
pub const DIV: u32 = 3;
pub const POW: u32 = 4;
pub const MOD: u32 = 5;

/// How the second (or first) input of a two-input operation lines up:
/// the same shape, a single number, or a row repeated for every row.
pub const SAME: u32 = 0;
pub const B_ONE: u32 = 1;
pub const A_ONE: u32 = 2;
pub const B_ROW: u32 = 3;
pub const A_ROW: u32 = 4;

/// gpu(): the name of the graphics card Antares would use, or nothing.
pub fn b_gpu(_: &mut crate::interp::Interp, a: Vec<crate::value::Value>) -> Result<crate::value::Value, String> {
    crate::builtins::want(&a, 0, "gpu")?;
    Ok(match available() {
        Some(name) => crate::value::Value::Str(name.into()),
        None => crate::value::Value::Nothing,
    })
}

/// use_gpu(yes or no): what `use gpu` and `use cpu` do. With the GPU on,
/// big matrix multiplications (and their gradients) run there. Without a
/// GPU, `use gpu` says so once and everything stays on the CPU.
pub fn b_use_gpu(_: &mut crate::interp::Interp, a: Vec<crate::value::Value>) -> Result<crate::value::Value, String> {
    crate::builtins::want(&a, 1, "use_gpu")?;
    let on = matches!(a[0], crate::value::Value::Bool(true));
    if on && available().is_none() {
        static TOLD: std::sync::Once = std::sync::Once::new();
        TOLD.call_once(|| eprintln!("Note: `use gpu` found no graphics card Antares can use here, so tensor maths stays on the CPU."));
    }
    crate::tensor::set_gpu(on && available().is_some());
    Ok(crate::value::Value::Nothing)
}

/// The matrix-multiply shader: each group of 16 x 16 threads works out a
/// 16 x 16 tile of the result, reading tiles of A and B into fast shared
/// memory once instead of every thread reading them separately. It is
/// compiled four times: with FA and/or FB set it reads A or B as if
/// flipped, so gradients need no copies. (A choice made while running would
/// be slower: HLSL's `?:` reads memory for both of its sides.)
#[cfg_attr(not(windows), allow(dead_code))]
const MATMUL_HLSL: &str = r#"
cbuffer Dims : register(b0) { uint M; uint N; uint P; uint FLAGS; uint4 unused; };
StructuredBuffer<float> A : register(t0);
StructuredBuffer<float> B : register(t1);
RWStructuredBuffer<float> C : register(u0);
#define T 16
groupshared float As[T][T];
groupshared float Bs[T][T];
#if FA
float a_at(uint i, uint k) { return A[k * M + i]; }
#else
float a_at(uint i, uint k) { return A[i * N + k]; }
#endif
#if FB
float b_at(uint k, uint j) { return B[j * N + k]; }
#else
float b_at(uint k, uint j) { return B[k * P + j]; }
#endif
[numthreads(T, T, 1)]
void main(uint3 g : SV_GroupID, uint3 t : SV_GroupThreadID) {
    uint row = g.y * T + t.y;
    uint col = g.x * T + t.x;
    float acc = 0;
    for (uint k0 = 0; k0 < N; k0 += T) {
        uint ka = k0 + t.x;
        uint kb = k0 + t.y;
        As[t.y][t.x] = (row < M && ka < N) ? a_at(row, ka) : 0;
        Bs[t.y][t.x] = (kb < N && col < P) ? b_at(kb, col) : 0;
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint k = 0; k < T; k++) acc += As[t.y][k] * Bs[k][t.x];
        GroupMemoryBarrierWithGroupSync();
    }
    if (row < M && col < P) C[row * P + col] = acc;
}
"#;

/// Element by element: two inputs (+ - * / ^ %), lined up as MODE says.
/// Threads are numbered across up to 65535 groups in x, then rows in y.
#[cfg_attr(not(windows), allow(dead_code))]
const BINARY_HLSL: &str = r#"
cbuffer C : register(b0) { uint OP; uint MODE; uint N; uint INNER; uint GX; uint3 unused; };
StructuredBuffer<float> A : register(t0);
StructuredBuffer<float> B : register(t1);
RWStructuredBuffer<float> O : register(u0);
float power(float a, float b) {
    if (b == 2) return a * a;
    float r = pow(abs(a), b);
    if (a < 0) {
        if (frac(b) != 0) return asfloat(0x7fc00000);
        if (fmod(abs(b), 2) == 1) r = -r;
    }
    if (a == 0 && b == 0) r = 1;
    return r;
}
[numthreads(256, 1, 1)]
void main(uint3 g : SV_GroupID, uint3 t : SV_GroupThreadID) {
    uint i = (g.y * GX + g.x) * 256 + t.x;
    if (i >= N) return;
    float a;
    float b;
    if (MODE == 2) a = A[0]; else if (MODE == 4) a = A[i % INNER]; else a = A[i];
    if (MODE == 1) b = B[0]; else if (MODE == 3) b = B[i % INNER]; else b = B[i];
    float r;
    if (OP == 0) r = a + b;
    else if (OP == 1) r = a - b;
    else if (OP == 2) r = a * b;
    else if (OP == 3) r = a / b;
    else if (OP == 4) r = power(a, b);
    else { r = fmod(a, b); if (r < 0) r += abs(b); }
    O[i] = r;
}
"#;

/// Element by element: one input through a function (the numbers match
/// the order of UFn in tensor.rs, then 13 for minus).
#[cfg_attr(not(windows), allow(dead_code))]
const UNARY_HLSL: &str = r#"
cbuffer C : register(b0) { uint OP; uint MODE; uint N; uint INNER; uint GX; uint3 unused; };
StructuredBuffer<float> A : register(t0);
RWStructuredBuffer<float> O : register(u0);
float safe_tanh(float x) { return 1 - 2 / (exp(2 * x) + 1); }
[numthreads(256, 1, 1)]
void main(uint3 g : SV_GroupID, uint3 t : SV_GroupThreadID) {
    uint i = (g.y * GX + g.x) * 256 + t.x;
    if (i >= N) return;
    float x = A[i];
    float r;
    if (OP == 0) r = exp(x);
    else if (OP == 1) r = log(x);
    else if (OP == 2) r = safe_tanh(x);
    else if (OP == 3) r = 1 / (1 + exp(-x));
    else if (OP == 4) r = max(x, 0);
    else if (OP == 5) r = sqrt(x);
    else if (OP == 6) r = abs(x);
    else if (OP == 7) r = sin(x);
    else if (OP == 8) r = cos(x);
    else if (OP == 9) r = 0.5 * x * (1 + safe_tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
    else if (OP == 10) r = x / (1 + exp(-x));
    else if (OP == 11) r = (x > 30) ? x : log(1 + exp(x));
    else if (OP == 12) r = (x > 0) ? x : 0.01 * x;
    else r = -x;
    O[i] = r;
}
"#;

#[cfg(windows)]
mod win {
    use super::{BINARY_HLSL, MATMUL_HLSL, UNARY_HLSL};
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
        /// The matrix multiply, by [flip A][flip B].
        matmul: [[Com; 2]; 2],
        binary: Com,
        unary: Com,
    }

    struct BufInner {
        buf: Com,
        srv: Com,
        uav: Com,
    }

    /// Numbers on the GPU, shared between the tensors that use them.
    #[derive(Clone)]
    pub struct Buf(std::rc::Rc<BufInner>);

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
        // ANTARES_GPU=warp uses Windows' software GPU (for tests on machines
        // without one); ANTARES_GPU=none behaves as if there were no GPU
        let setting = std::env::var("ANTARES_GPU").unwrap_or_default().to_ascii_lowercase();
        if setting == "none" {
            return None;
        }
        let warp = setting == "warp";
        let (adapter, name) = if warp { (Com(std::ptr::null_mut()), "Microsoft Basic Render Driver (software)".to_string()) } else { best_adapter()? };
        let levels = [0xb000u32]; // feature level 11_0: compute shaders 5.0
        let (mut device, mut context, mut level): (Ptr, Ptr, u32) = (std::ptr::null_mut(), std::ptr::null_mut(), 0);
        let driver = if warp { 5 } else { 0 }; // D3D_DRIVER_TYPE_WARP, or UNKNOWN with an adapter
        let hr = unsafe { create(adapter.0, driver, std::ptr::null_mut(), 0, levels.as_ptr(), 1, 7, &mut device, &mut level, &mut context) };
        if hr < 0 || device.is_null() || context.is_null() {
            return None;
        }
        let (device, context) = (Com(device), Com(context));
        let null = || Com(std::ptr::null_mut());
        let mut g = Gpu { name, device, context, compile, matmul: [[null(), null()], [null(), null()]], binary: null(), unary: null() };
        for fa in 0..2 {
            for fb in 0..2 {
                g.matmul[fa][fb] = g.shader(&format!("#define FA {}
#define FB {}
{}", fa, fb, MATMUL_HLSL)).ok()?;
            }
        }
        g.binary = g.shader(BINARY_HLSL).ok()?;
        g.unary = g.shader(UNARY_HLSL).ok()?;
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

        /// Small numbers every thread can see (a constant buffer).
        fn constants(&self, words: [u32; 8]) -> Result<Com, String> {
            let bytes: &[u8] = unsafe { std::slice::from_raw_parts(words.as_ptr().cast(), 32) };
            self.buffer(BufferDesc { byte_width: 32, usage: USAGE_DEFAULT, bind_flags: BIND_CONSTANT, cpu_access: 0, misc: 0, stride: 0 }, Some(bytes))
        }

        /// Room for `n` numbers that shaders can write and later read,
        /// starting with `init` if given.
        fn rw(&self, n: usize, init: Option<&[f32]>) -> Result<Buf, String> {
            let bytes = init.map(|xs| unsafe { std::slice::from_raw_parts(xs.as_ptr().cast::<u8>(), xs.len() * 4) });
            let size = (n.max(1) * 4) as u32;
            let pad;
            let init = match bytes {
                Some(b) if b.len() == size as usize => Some(b),
                Some(b) => {
                    pad = { let mut v = b.to_vec(); v.resize(size as usize, 0); v };
                    Some(&pad[..])
                }
                None => None,
            };
            let buf = self.buffer(BufferDesc { byte_width: size, usage: USAGE_DEFAULT, bind_flags: BIND_SHADER_RESOURCE | BIND_UNORDERED_ACCESS, cpu_access: 0, misc: MISC_STRUCTURED, stride: 4 }, init)?;
            let srv_of: unsafe extern "system" fn(Ptr, Ptr, *const c_void, *mut Ptr) -> HResult = unsafe { method(self.device.0, 7) };
            let uav_of: unsafe extern "system" fn(Ptr, Ptr, *const c_void, *mut Ptr) -> HResult = unsafe { method(self.device.0, 8) };
            let (mut srv, mut uav): (Ptr, Ptr) = (std::ptr::null_mut(), std::ptr::null_mut());
            if unsafe { srv_of(self.device.0, buf.0, std::ptr::null(), &mut srv) } < 0 || unsafe { uav_of(self.device.0, buf.0, std::ptr::null(), &mut uav) } < 0 {
                return Err("the GPU couldn't make room for the numbers".into());
            }
            Ok(Buf(std::rc::Rc::new(BufInner { buf, srv: Com(srv), uav: Com(uav) })))
        }

        /// Groups of 256 threads covering n numbers: up to 65535 across, then rows.
        fn spread(n: usize) -> (u32, u32) {
            let groups = n.div_ceil(256).max(1);
            let gx = groups.min(65535);
            (gx as u32, groups.div_ceil(gx) as u32)
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

    const NO_GPU: &str = "the GPU isn't available on this computer";

    /// Numbers sent to the GPU (as 32-bit numbers).
    pub fn upload(xs: &[f64]) -> Result<Buf, String> {
        with_gpu(|g| {
            let f: Vec<f32> = xs.iter().map(|&x| x as f32).collect();
            g.rw(xs.len(), Some(&f))
        })
        .unwrap_or_else(|| Err(NO_GPU.into()))
    }

    /// The first `n` numbers of a GPU buffer, brought back.
    pub fn download(b: &Buf, n: usize) -> Result<Vec<f64>, String> {
        with_gpu(|g| Ok(g.read(&b.0.buf, n)?.into_iter().map(|x| x as f64).collect())).unwrap_or_else(|| Err(NO_GPU.into()))
    }

    /// The m x n matrix `a` times the n x p matrix `b`, both already on the
    /// GPU, with the answer left there. `ta` / `tb`: the buffer holds that
    /// matrix flipped (n x m, or p x n).
    pub fn matmul_bufs(a: &Buf, b: &Buf, m: usize, n: usize, p: usize, ta: bool, tb: bool) -> Result<Buf, String> {
        with_gpu(|g| {
            let out = g.rw(m * p, None)?;
            let consts = g.constants([m as u32, n as u32, p as u32, ta as u32 | (tb as u32) << 1, 0, 0, 0, 0])?;
            g.run(&g.matmul[ta as usize][tb as usize], &consts, &[&a.0.srv, &b.0.srv], &[&out.0.uav], (p.div_ceil(16) as u32, m.div_ceil(16) as u32, 1));
            Ok(out)
        })
        .unwrap_or_else(|| Err(NO_GPU.into()))
    }

    /// `n` results of a two-input operation (see ADD.. and SAME.. in gpu.rs);
    /// `inner` is the length of a repeated row.
    pub fn binary(op: u32, mode: u32, a: &Buf, b: &Buf, n: usize, inner: usize) -> Result<Buf, String> {
        with_gpu(|g| {
            let out = g.rw(n, None)?;
            let (gx, gy) = Gpu::spread(n);
            let consts = g.constants([op, mode, n as u32, inner.max(1) as u32, gx, 0, 0, 0])?;
            g.run(&g.binary, &consts, &[&a.0.srv, &b.0.srv], &[&out.0.uav], (gx, gy, 1));
            Ok(out)
        })
        .unwrap_or_else(|| Err(NO_GPU.into()))
    }

    /// `n` results of a one-input function (numbered as in UNARY_HLSL).
    pub fn unary(op: u32, a: &Buf, n: usize) -> Result<Buf, String> {
        with_gpu(|g| {
            let out = g.rw(n, None)?;
            let (gx, gy) = Gpu::spread(n);
            let consts = g.constants([op, 0, n as u32, 1, gx, 0, 0, 0])?;
            g.run(&g.unary, &consts, &[&a.0.srv], &[&out.0.uav], (gx, gy, 1));
            Ok(out)
        })
        .unwrap_or_else(|| Err(NO_GPU.into()))
    }

    /// The m x n matrix `a` times the n x p matrix `b`, worked out on the
    /// GPU in 32-bit numbers (sent there and brought back).
    pub fn matmul(a: &[f64], b: &[f64], m: usize, n: usize, p: usize) -> Result<Vec<f64>, String> {
        let (ab, bb) = (upload(a)?, upload(b)?);
        download(&matmul_bufs(&ab, &bb, m, n, p, false, false)?, m * p)
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
                let _ = matmul(&a.vals(), &b.vals(), n, n, n); // warm up
                let reps = if n <= 512 { 5 } else { 2 };
                let t = std::time::Instant::now();
                for _ in 0..reps {
                    let _ = crate::tensor::matmul(&a, &b).unwrap();
                }
                let cpu = t.elapsed().as_secs_f64() / reps as f64;
                let t = std::time::Instant::now();
                for _ in 0..reps {
                    let _ = matmul(&a.vals(), &b.vals(), n, n, n).unwrap();
                }
                let gpu = t.elapsed().as_secs_f64() / reps as f64;
                println!("{0} x {0}: CPU {1:.1} ms, GPU {2:.1} ms (with copying), {3:.1}x", n, cpu * 1e3, gpu * 1e3, cpu / gpu);
            }
        }

        #[test]
        fn element_wise_operations_match_the_cpu() {
            if available().is_none() {
                return;
            }
            let xs: Vec<f64> = (0..1000).map(|i| (i as f64 - 500.0) / 97.0).collect();
            let ys: Vec<f64> = (0..1000).map(|i| ((i * 7 % 23) as f64 - 11.0) / 5.0 + 0.1).collect();
            let (a, b) = (upload(&xs).unwrap(), upload(&ys).unwrap());
            let row = upload(&ys[..10]).unwrap();
            let one = upload(&[3.0]).unwrap();
            let close = |g: f64, w: f64| (g - w).abs() <= 1e-4 * (1.0 + w.abs()) || (g.is_nan() && w.is_nan());
            for (op, f) in [(super::super::ADD, (|x: f64, y: f64| x + y) as fn(f64, f64) -> f64), (super::super::SUB, |x, y| x - y), (super::super::MUL, |x, y| x * y), (super::super::DIV, |x, y| x / y), (super::super::MOD, |x: f64, y: f64| x.rem_euclid(y))] {
                let got = download(&binary(op, super::super::SAME, &a, &b, 1000, 1000).unwrap(), 1000).unwrap();
                for i in 0..1000 {
                    assert!(close(got[i], f(xs[i], ys[i])), "op {} at {}: {} vs {}", op, i, got[i], f(xs[i], ys[i]));
                }
                let got = download(&binary(op, super::super::B_ROW, &a, &row, 1000, 10).unwrap(), 1000).unwrap();
                assert!(close(got[37], f(xs[37], ys[7])));
                let got = download(&binary(op, super::super::B_ONE, &a, &one, 1000, 1).unwrap(), 1000).unwrap();
                assert!(close(got[5], f(xs[5], 3.0)));
            }
            // powers of negative numbers, as the CPU does them
            let got = download(&binary(super::super::POW, super::super::B_ONE, &a, &upload(&[2.0]).unwrap(), 1000, 1).unwrap(), 1000).unwrap();
            assert!(close(got[0], xs[0] * xs[0]));
            let got = download(&binary(super::super::POW, super::super::B_ONE, &a, &upload(&[3.0]).unwrap(), 1000, 1).unwrap(), 1000).unwrap();
            assert!(close(got[0], xs[0].powi(3)));
            use crate::tensor::UFn::*;
            for (op, u) in [Exp, Log, Tanh, Sigmoid, Relu, Sqrt, Abs, Sin, Cos, Gelu, Silu, Softplus, LeakyRelu].into_iter().enumerate() {
                let got = download(&unary(op as u32, &a, 1000).unwrap(), 1000).unwrap();
                for i in (0..1000).step_by(7) {
                    let want = crate::tensor::apply(u, xs[i]);
                    assert!(close(got[i], want) || (want.is_nan() && got[i].is_nan()) || (want.is_infinite() && got[i].is_infinite()), "unary {} at x = {}: {} vs {}", op, xs[i], got[i], want);
                }
            }
        }

        #[test]
        fn matmul_with_flipped_inputs() {
            if available().is_none() {
                return;
            }
            let (m, n, p) = (19, 23, 17);
            let a: Vec<f64> = (0..m * n).map(|i| (i % 7) as f64 - 3.0).collect();
            let b: Vec<f64> = (0..n * p).map(|i| (i % 5) as f64 - 2.0).collect();
            let flip = |x: &[f64], r: usize, c: usize| (0..r * c).map(|k| x[(k % r) * c + k / r]).collect::<Vec<f64>>();
            let want = matmul(&a, &b, m, n, p).unwrap();
            let at = upload(&flip(&a, m, n)).unwrap();
            let bt = upload(&flip(&b, n, p)).unwrap();
            let got = download(&matmul_bufs(&at, &bt, m, n, p, true, true).unwrap(), m * p).unwrap();
            for i in 0..m * p {
                assert!((got[i] - want[i]).abs() < 1e-3, "{} vs {}", got[i], want[i]);
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
