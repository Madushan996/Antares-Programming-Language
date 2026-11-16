//! The GPU on Linux, through NVIDIA's CUDA: what Google Colab, Kaggle and
//! Linux PCs with an NVIDIA graphics card have.
//!
//! Nothing is linked at build time. The driver's library (libcuda) and the
//! run-time compiler (libnvrtc, part of the CUDA toolkit that Colab and
//! Kaggle include) are opened when a program first asks for the GPU; without
//! them there is simply no GPU and Antares keeps using the CPU. The kernels
//! below are written in CUDA C, compiled once by libnvrtc into PTX for this
//! GPU, and kept in a cache folder so later programs don't wait.
//!
//! The functions are the same as the Direct3D ones in gpu.rs, so the rest of
//! Antares doesn't know which GPU it is using.

use std::cell::RefCell;
use std::ffi::{c_char, c_int, c_void};
use std::rc::Rc;

unsafe extern "C" {
    fn dlopen(file: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
}

const RTLD_NOW: c_int = 2;
const RTLD_GLOBAL: c_int = 0x100;

type CuResult = c_int;
type DevPtr = u64;

/// The CUDA driver and run-time compiler functions Antares uses.
struct Api {
    init: unsafe extern "C" fn(u32) -> CuResult,
    device_count: unsafe extern "C" fn(*mut c_int) -> CuResult,
    device_get: unsafe extern "C" fn(*mut c_int, c_int) -> CuResult,
    device_name: unsafe extern "C" fn(*mut c_char, c_int, c_int) -> CuResult,
    device_mem: unsafe extern "C" fn(*mut usize, c_int) -> CuResult,
    device_attr: unsafe extern "C" fn(*mut c_int, c_int, c_int) -> CuResult,
    ctx_retain: unsafe extern "C" fn(*mut *mut c_void, c_int) -> CuResult,
    ctx_set: unsafe extern "C" fn(*mut c_void) -> CuResult,
    module_load: unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> CuResult,
    module_fn: unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> CuResult,
    mem_alloc: unsafe extern "C" fn(*mut DevPtr, usize) -> CuResult,
    mem_free: unsafe extern "C" fn(DevPtr) -> CuResult,
    to_dev: unsafe extern "C" fn(DevPtr, *const c_void, usize) -> CuResult,
    to_host: unsafe extern "C" fn(*mut c_void, DevPtr, usize) -> CuResult,
    #[allow(clippy::type_complexity)]
    launch: unsafe extern "C" fn(*mut c_void, u32, u32, u32, u32, u32, u32, u32, *mut c_void, *mut *mut c_void, *mut *mut c_void) -> CuResult,
    rtc_create: unsafe extern "C" fn(*mut *mut c_void, *const c_char, *const c_char, c_int, *const *const c_char, *const *const c_char) -> c_int,
    rtc_compile: unsafe extern "C" fn(*mut c_void, c_int, *const *const c_char) -> c_int,
    rtc_log_size: unsafe extern "C" fn(*mut c_void, *mut usize) -> c_int,
    rtc_log: unsafe extern "C" fn(*mut c_void, *mut c_char) -> c_int,
    rtc_ptx_size: unsafe extern "C" fn(*mut c_void, *mut usize) -> c_int,
    rtc_ptx: unsafe extern "C" fn(*mut c_void, *mut c_char) -> c_int,
    rtc_destroy: unsafe extern "C" fn(*mut *mut c_void) -> c_int,
}

fn open(names: &[String]) -> Option<*mut c_void> {
    for n in names {
        let c = std::ffi::CString::new(n.as_str()).ok()?;
        let h = unsafe { dlopen(c.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
        if !h.is_null() {
            return Some(h);
        }
    }
    None
}

/// libnvrtc's possible names: the plain one, versioned ones, and the files
/// in the usual CUDA folders.
fn nvrtc_names() -> Vec<String> {
    let mut names = vec!["libnvrtc.so".to_string()];
    for v in ["13", "12", "11.2", "11.1", "11.0"] {
        names.push(format!("libnvrtc.so.{}", v));
    }
    let mut dirs = vec!["/usr/local/cuda/lib64".to_string(), "/usr/local/cuda/targets/x86_64-linux/lib".to_string(), "/usr/lib/x86_64-linux-gnu".to_string()];
    if let Ok(home) = std::env::var("CUDA_HOME").or_else(|_| std::env::var("CUDA_PATH")) {
        dirs.insert(0, format!("{}/lib64", home));
    }
    if let Ok(paths) = std::env::var("LD_LIBRARY_PATH") {
        dirs.extend(paths.split(':').filter(|p| !p.is_empty()).map(str::to_string));
    }
    if let Ok(entries) = std::fs::read_dir("/usr/local") {
        for e in entries.flatten() {
            let n = e.file_name().to_string_lossy().into_owned();
            if n.starts_with("cuda-") {
                dirs.push(format!("/usr/local/{}/lib64", n));
            }
        }
    }
    for d in dirs {
        if let Ok(entries) = std::fs::read_dir(&d) {
            let mut found: Vec<String> = entries.flatten().map(|e| e.file_name().to_string_lossy().into_owned()).filter(|n| n.starts_with("libnvrtc.so")).collect();
            found.sort();
            found.reverse(); // the newest version first
            names.extend(found.into_iter().map(|f| format!("{}/{}", d, f)));
        }
    }
    names
}

fn sym<T: Copy>(lib: *mut c_void, name: &str) -> Option<T> {
    let c = std::ffi::CString::new(name).ok()?;
    let p = unsafe { dlsym(lib, c.as_ptr()) };
    (!p.is_null()).then(|| unsafe { std::mem::transmute_copy(&p) })
}

fn load_api() -> Option<Api> {
    let cu = open(&["libcuda.so.1".to_string(), "libcuda.so".to_string()])?;
    let rtc = open(&nvrtc_names())?;
    Some(Api {
        init: sym(cu, "cuInit")?,
        device_count: sym(cu, "cuDeviceGetCount")?,
        device_get: sym(cu, "cuDeviceGet")?,
        device_name: sym(cu, "cuDeviceGetName")?,
        device_mem: sym(cu, "cuDeviceTotalMem_v2")?,
        device_attr: sym(cu, "cuDeviceGetAttribute")?,
        ctx_retain: sym(cu, "cuDevicePrimaryCtxRetain")?,
        ctx_set: sym(cu, "cuCtxSetCurrent")?,
        module_load: sym(cu, "cuModuleLoadData")?,
        module_fn: sym(cu, "cuModuleGetFunction")?,
        mem_alloc: sym(cu, "cuMemAlloc_v2")?,
        mem_free: sym(cu, "cuMemFree_v2")?,
        to_dev: sym(cu, "cuMemcpyHtoD_v2")?,
        to_host: sym(cu, "cuMemcpyDtoH_v2")?,
        launch: sym(cu, "cuLaunchKernel")?,
        rtc_create: sym(rtc, "nvrtcCreateProgram")?,
        rtc_compile: sym(rtc, "nvrtcCompileProgram")?,
        rtc_log_size: sym(rtc, "nvrtcGetProgramLogSize")?,
        rtc_log: sym(rtc, "nvrtcGetProgramLog")?,
        rtc_ptx_size: sym(rtc, "nvrtcGetPTXSize")?,
        rtc_ptx: sym(rtc, "nvrtcGetPTX")?,
        rtc_destroy: sym(rtc, "nvrtcDestroyProgram")?,
    })
}

/// All of Antares's kernels: ports of the Direct3D shaders in gpu.rs, with
/// the same operation and mode numbers.
const KERNELS: &str = r#"
template <bool FA, bool FB>
__device__ void mm(const float* A, const float* B, float* C, unsigned M, unsigned N, unsigned P) {
    __shared__ float As[16][65];
    __shared__ float Bs[16][65];
    unsigned tx = threadIdx.x, ty = threadIdx.y, id = ty * 16 + tx;
    unsigned row0 = blockIdx.y * 64, col0 = blockIdx.x * 64;
    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.f;
    }
    for (unsigned k0 = 0; k0 < N; k0 += 16) {
        for (int l = 0; l < 4; l++) {
            unsigned e = id + l * 256;
            unsigned r = e / 16, ka = e % 16;
            unsigned gr = row0 + r, gk = k0 + ka;
            As[ka][r] = (gr < M && gk < N) ? (FA ? A[gk * M + gr] : A[gr * N + gk]) : 0.f;
            unsigned kb = e / 64, c = e % 64;
            unsigned gkb = k0 + kb, gc = col0 + c;
            Bs[kb][c] = (gkb < N && gc < P) ? (FB ? B[gc * N + gkb] : B[gkb * P + gc]) : 0.f;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < 16; k++) {
            float a[4], b[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) a[i] = As[k][ty + 16 * i];
            #pragma unroll
            for (int j = 0; j < 4; j++) b[j] = Bs[k][tx + 16 * j];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                #pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] += a[i] * b[j];
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        unsigned row = row0 + ty + 16 * i;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            unsigned col = col0 + tx + 16 * j;
            if (row < M && col < P) C[row * P + col] = acc[i][j];
        }
    }
}

extern "C" __global__ void mm00(const float* A, const float* B, float* C, unsigned M, unsigned N, unsigned P) { mm<false, false>(A, B, C, M, N, P); }
extern "C" __global__ void mm01(const float* A, const float* B, float* C, unsigned M, unsigned N, unsigned P) { mm<false, true>(A, B, C, M, N, P); }
extern "C" __global__ void mm10(const float* A, const float* B, float* C, unsigned M, unsigned N, unsigned P) { mm<true, false>(A, B, C, M, N, P); }
extern "C" __global__ void mm11(const float* A, const float* B, float* C, unsigned M, unsigned N, unsigned P) { mm<true, true>(A, B, C, M, N, P); }

__device__ float pick(const float* X, unsigned i, unsigned mode, unsigned one, unsigned row, unsigned inner) {
    if (mode == one) return X[0];
    if (mode == row) return X[i % inner];
    return X[i];
}

extern "C" __global__ void binary(unsigned op, unsigned mode, unsigned n, unsigned inner, const float* A, const float* B, float* O) {
    unsigned i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    float a = pick(A, i, mode, 2, 4, inner);
    float b = pick(B, i, mode, 1, 3, inner);
    float r;
    if (op == 0) r = a + b;
    else if (op == 1) r = a - b;
    else if (op == 2) r = a * b;
    else if (op == 3) r = a / b;
    else if (op == 4) r = powf(a, b);
    else { r = fmodf(a, b); if (r < 0) r += fabsf(b); }
    O[i] = r;
}

extern "C" __global__ void unary(unsigned op, unsigned n, const float* A, float* O) {
    unsigned i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    float x = A[i];
    float r;
    if (op == 0) r = expf(x);
    else if (op == 1) r = logf(x);
    else if (op == 2) r = tanhf(x);
    else if (op == 3) r = 1.f / (1.f + expf(-x));
    else if (op == 4) r = fmaxf(x, 0.f);
    else if (op == 5) r = sqrtf(x);
    else if (op == 6) r = fabsf(x);
    else if (op == 7) r = sinf(x);
    else if (op == 8) r = cosf(x);
    else if (op == 9) r = 0.5f * x * (1.f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    else if (op == 10) r = x / (1.f + expf(-x));
    else if (op == 11) r = (x > 30.f) ? x : log1pf(expf(x));
    else if (op == 12) r = (x > 0.f) ? x : 0.01f * x;
    else r = -x;
    O[i] = r;
}

extern "C" __global__ void bin_back(unsigned op, unsigned mode, unsigned n, unsigned inner, unsigned need, const float* G, const float* A, const float* B, float* GA, float* GB) {
    unsigned i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    float a = pick(A, i, mode, 2, 4, inner);
    float b = pick(B, i, mode, 1, 3, inner);
    float d = G[i];
    float ga = d, gb = 0.f;
    if (op == 0) gb = d;
    else if (op == 1) gb = -d;
    else if (op == 2) { ga = d * b; gb = d * a; }
    else if (op == 3) { ga = d / b; gb = -d * a / (b * b); }
    else if (op == 4) ga = d * b * powf(a, b - 1.f);
    if (need & 1) GA[i] = ga;
    if (need & 2) GB[i] = gb;
}

extern "C" __global__ void un_back(unsigned op, unsigned n, const float* G, const float* X, const float* Y, float* O) {
    unsigned i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    float x = X[i], y = Y[i], d;
    if (op == 0) d = y;
    else if (op == 1) d = 1.f / x;
    else if (op == 2) d = 1.f - y * y;
    else if (op == 3) d = y * (1.f - y);
    else if (op == 4) d = (x > 0.f) ? 1.f : 0.f;
    else if (op == 5) d = 0.5f / y;
    else if (op == 6) d = (x >= 0.f) ? 1.f : -1.f;
    else if (op == 7) d = cosf(x);
    else if (op == 8) d = -sinf(x);
    else if (op == 9) {
        float u = 0.7978845608f * (x + 0.044715f * x * x * x);
        float th = tanhf(u);
        d = 0.5f * (1.f + th) + 0.5f * x * (1.f - th * th) * 0.7978845608f * (1.f + 3.f * 0.044715f * x * x);
    }
    else if (op == 10) { float sg = 1.f / (1.f + expf(-x)); d = sg * (1.f + x * (1.f - sg)); }
    else if (op == 11) d = 1.f / (1.f + expf(-x));
    else if (op == 12) d = (x > 0.f) ? 1.f : 0.01f;
    else d = -1.f;
    O[i] = G[i] * d;
}

extern "C" __global__ void reduce(unsigned n, unsigned inner, unsigned rows, unsigned square, const float* T, float* O) {
    unsigned j = blockIdx.x * 256 + threadIdx.x;
    if (j >= inner) return;
    float s = 0.f;
    for (unsigned r = 0; r < rows; r++) {
        unsigned k = r * inner + j;
        if (k < n) { float v = T[k]; s += square ? v * v : v; }
    }
    O[j] = s;
}

extern "C" __global__ void optimise(unsigned method, unsigned n, float rate, float c1, float c2, float decay, float scale, const float* P, const float* G, float* PN, float* M, float* V) {
    unsigned i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    float d = G[i] * scale;
    float p = P[i];
    if (method == 1) {
        float m = 0.9f * M[i] + 0.1f * d;
        float v = 0.999f * V[i] + 0.001f * d * d;
        M[i] = m;
        V[i] = v;
        float step = rate * (m / c1) / (sqrtf(v / c2) + 1e-8f);
        p -= step + rate * decay * p;
    } else if (method == 3) {
        float m = 0.9f * M[i] + d;
        M[i] = m;
        p -= rate * m;
    } else {
        p -= rate * d;
    }
    PN[i] = p;
}
"#;

const KERNEL_NAMES: [&str; 10] = ["mm00", "mm01", "mm10", "mm11", "binary", "unary", "bin_back", "un_back", "reduce", "optimise"];

struct Gpu {
    api: Rc<Api>,
    name: String,
    /// The kernels, by KERNEL_NAMES position.
    kernels: Vec<*mut c_void>,
}

thread_local! {
    static GPU: RefCell<Option<Option<Gpu>>> = const { RefCell::new(None) };
}

/// Memory on the GPU, freed when the last tensor using it is gone.
struct Mem {
    ptr: DevPtr,
    api: Rc<Api>,
}

impl Drop for Mem {
    fn drop(&mut self) {
        unsafe { (self.api.mem_free)(self.ptr) };
    }
}

/// Numbers on the GPU, shared between the tensors that use them.
#[derive(Clone)]
pub struct Buf(Rc<Mem>);

fn check(r: CuResult, what: &str) -> Result<(), String> {
    if r == 0 { Ok(()) } else { Err(format!("the GPU couldn't {} (CUDA error {})", what, r)) }
}

/// Where compiled kernels are kept: antares/cuda in the cache folder, named
/// by a hash of the source, the GPU's architecture and the Antares version.
fn ptx_cache_file(arch: &str) -> Option<std::path::PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME").filter(|v| !v.is_empty()).map(std::path::PathBuf::from).or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache")))?;
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in KERNELS.bytes().chain(arch.bytes()).chain(env!("CARGO_PKG_VERSION").bytes()) {
        h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
    }
    Some(base.join("antares").join("cuda").join(format!("{:016x}.ptx", h)))
}

/// The kernels compiled into PTX for `arch` (like "compute_75").
fn compile(api: &Api, arch: &str) -> Result<Vec<u8>, String> {
    let src = std::ffi::CString::new(KERNELS).unwrap();
    let mut prog: *mut c_void = std::ptr::null_mut();
    if unsafe { (api.rtc_create)(&mut prog, src.as_ptr(), c"antares.cu".as_ptr(), 0, std::ptr::null(), std::ptr::null()) } != 0 {
        return Err("the CUDA compiler couldn't start".into());
    }
    let opt = std::ffi::CString::new(format!("--gpu-architecture={}", arch)).unwrap();
    let opts = [opt.as_ptr()];
    let ok = unsafe { (api.rtc_compile)(prog, 1, opts.as_ptr()) } == 0;
    let result = if ok {
        let mut size = 0usize;
        unsafe { (api.rtc_ptx_size)(prog, &mut size) };
        let mut ptx = vec![0u8; size];
        unsafe { (api.rtc_ptx)(prog, ptx.as_mut_ptr().cast()) };
        Ok(ptx)
    } else {
        let mut size = 0usize;
        unsafe { (api.rtc_log_size)(prog, &mut size) };
        let mut log = vec![0u8; size.max(1)];
        unsafe { (api.rtc_log)(prog, log.as_mut_ptr().cast()) };
        Err(format!("the GPU programs didn't compile: {}", String::from_utf8_lossy(&log).trim_end_matches('\0').trim()))
    };
    unsafe { (api.rtc_destroy)(&mut prog) };
    result
}

fn setup() -> Option<Gpu> {
    if std::env::var("ANTARES_GPU").is_ok_and(|v| v.eq_ignore_ascii_case("none")) {
        return None;
    }
    let api = Rc::new(load_api()?);
    unsafe {
        check((api.init)(0), "start").ok()?;
        let mut count = 0;
        if (api.device_count)(&mut count) != 0 || count < 1 {
            return None;
        }
        // the GPU with the most memory
        let mut best = (0, 0usize);
        for i in 0..count {
            let mut dev = 0;
            let mut mem = 0usize;
            if (api.device_get)(&mut dev, i) == 0 && (api.device_mem)(&mut mem, dev) == 0 && mem > best.1 {
                best = (dev, mem);
            }
        }
        let dev = best.0;
        let mut name = [0 as c_char; 256];
        (api.device_name)(name.as_mut_ptr(), 256, dev);
        let name = std::ffi::CStr::from_ptr(name.as_ptr()).to_string_lossy().trim().to_string();
        let (mut major, mut minor) = (0, 0);
        (api.device_attr)(&mut major, 75, dev); // compute capability
        (api.device_attr)(&mut minor, 76, dev);
        let mut ctx: *mut c_void = std::ptr::null_mut();
        check((api.ctx_retain)(&mut ctx, dev), "start").ok()?;
        check((api.ctx_set)(ctx), "start").ok()?;
        let arch = format!("compute_{}{}", major, minor);
        let cached = ptx_cache_file(&arch);
        let mut ptx = match cached.as_ref().and_then(|f| std::fs::read(f).ok()) {
            Some(p) if !p.is_empty() => p,
            _ => {
                let p = match compile(&api, &arch) {
                    Ok(p) => p,
                    Err(e) => {
                        eprintln!("Note: {}", e);
                        return None;
                    }
                };
                if let Some(f) = &cached {
                    if let Some(dir) = f.parent() {
                        let _ = std::fs::create_dir_all(dir);
                    }
                    let tmp = f.with_extension(format!("tmp{}", std::process::id()));
                    if std::fs::write(&tmp, &p).is_ok() && std::fs::rename(&tmp, f).is_err() {
                        let _ = std::fs::remove_file(&tmp);
                    }
                }
                p
            }
        };
        if ptx.last() != Some(&0) {
            ptx.push(0);
        }
        let mut module: *mut c_void = std::ptr::null_mut();
        if (api.module_load)(&mut module, ptx.as_ptr().cast()) != 0 {
            // a damaged or out-of-date cache file: compile afresh next time
            if let Some(f) = &cached {
                let _ = std::fs::remove_file(f);
            }
            return None;
        }
        let mut kernels = vec![];
        for k in KERNEL_NAMES {
            let mut f: *mut c_void = std::ptr::null_mut();
            let cname = std::ffi::CString::new(k).unwrap();
            check((api.module_fn)(&mut f, module, cname.as_ptr()), "find its programs").ok()?;
            kernels.push(f);
        }
        Some(Gpu { api, name, kernels })
    }
}

fn with_gpu<R>(f: impl FnOnce(&Gpu) -> Result<R, String>) -> Result<R, String> {
    GPU.with(|g| {
        let mut g = g.borrow_mut();
        if g.is_none() {
            *g = Some(setup());
        }
        match g.as_ref().unwrap() {
            Some(gpu) => f(gpu),
            None => Err("the GPU isn't available on this computer".into()),
        }
    })
}

impl Gpu {
    fn alloc(&self, n: usize) -> Result<Buf, String> {
        let mut ptr: DevPtr = 0;
        check(unsafe { (self.api.mem_alloc)(&mut ptr, n.max(1) * 4) }, &format!("make room for {} numbers", n))?;
        Ok(Buf(Rc::new(Mem { ptr, api: self.api.clone() })))
    }

    /// Runs kernel `k` on `blocks` blocks of `threads` threads with these arguments.
    fn run(&self, k: usize, blocks: (u32, u32), threads: (u32, u32), args: &mut [*mut c_void]) -> Result<(), String> {
        check(
            unsafe { (self.api.launch)(self.kernels[k], blocks.0, blocks.1, 1, threads.0, threads.1, 1, 0, std::ptr::null_mut(), args.as_mut_ptr(), std::ptr::null_mut()) },
            "run its program",
        )
    }

    /// Enough blocks of 256 threads for n numbers.
    fn blocks(n: usize) -> (u32, u32) {
        (n.div_ceil(256).max(1) as u32, 1)
    }
}

/// A kernel argument: a pointer to the value, as cuLaunchKernel wants.
fn arg<T>(x: &mut T) -> *mut c_void {
    x as *mut T as *mut c_void
}

pub fn available() -> Option<String> {
    with_gpu(|g| Ok(g.name.clone())).ok()
}

pub fn upload(xs: &[f64]) -> Result<Buf, String> {
    with_gpu(|g| {
        let f: Vec<f32> = xs.iter().map(|&x| x as f32).collect();
        let b = g.alloc(xs.len())?;
        if !f.is_empty() {
            check(unsafe { (g.api.to_dev)(b.0.ptr, f.as_ptr().cast(), f.len() * 4) }, "receive the numbers")?;
        }
        Ok(b)
    })
}

pub fn download(b: &Buf, n: usize) -> Result<Vec<f64>, String> {
    with_gpu(|g| {
        let mut f = vec![0f32; n];
        if n > 0 {
            check(unsafe { (g.api.to_host)(f.as_mut_ptr().cast(), b.0.ptr, n * 4) }, "send back its answer")?;
        }
        Ok(f.into_iter().map(|x| x as f64).collect())
    })
}

pub fn matmul_bufs(a: &Buf, b: &Buf, m: usize, n: usize, p: usize, ta: bool, tb: bool) -> Result<Buf, String> {
    with_gpu(|g| {
        let out = g.alloc(m * p)?;
        let (mut pa, mut pb, mut pc) = (a.0.ptr, b.0.ptr, out.0.ptr);
        let (mut mm, mut nn, mut pp) = (m as u32, n as u32, p as u32);
        let k = (ta as usize) * 2 + tb as usize;
        g.run(k, (p.div_ceil(64) as u32, m.div_ceil(64) as u32), (16, 16), &mut [arg(&mut pa), arg(&mut pb), arg(&mut pc), arg(&mut mm), arg(&mut nn), arg(&mut pp)])?;
        Ok(out)
    })
}

pub fn binary(op: u32, mode: u32, a: &Buf, b: &Buf, n: usize, inner: usize) -> Result<Buf, String> {
    with_gpu(|g| {
        let out = g.alloc(n)?;
        let (mut op, mut mode, mut nn, mut inner) = (op, mode, n as u32, inner.max(1) as u32);
        let (mut pa, mut pb, mut po) = (a.0.ptr, b.0.ptr, out.0.ptr);
        g.run(4, Gpu::blocks(n), (256, 1), &mut [arg(&mut op), arg(&mut mode), arg(&mut nn), arg(&mut inner), arg(&mut pa), arg(&mut pb), arg(&mut po)])?;
        Ok(out)
    })
}

pub fn unary(op: u32, a: &Buf, n: usize) -> Result<Buf, String> {
    with_gpu(|g| {
        let out = g.alloc(n)?;
        let (mut op, mut nn) = (op, n as u32);
        let (mut pa, mut po) = (a.0.ptr, out.0.ptr);
        g.run(5, Gpu::blocks(n), (256, 1), &mut [arg(&mut op), arg(&mut nn), arg(&mut pa), arg(&mut po)])?;
        Ok(out)
    })
}

pub fn fill(x: f64, n: usize) -> Result<Buf, String> {
    upload(&vec![x; n])
}

#[allow(clippy::too_many_arguments)]
pub fn bin_back(op: u32, mode: u32, g: &Buf, a: &Buf, b: &Buf, _y: Option<&Buf>, n: usize, inner: usize, need_a: bool, need_b: bool) -> Result<(Option<Buf>, Option<Buf>), String> {
    with_gpu(|gpu| {
        let ga = if need_a { Some(gpu.alloc(n)?) } else { None };
        let gb = if need_b { Some(gpu.alloc(n)?) } else { None };
        let (mut op, mut mode, mut nn, mut inner, mut need) = (op, mode, n as u32, inner.max(1) as u32, need_a as u32 | (need_b as u32) << 1);
        let (mut pg, mut pa, mut pb) = (g.0.ptr, a.0.ptr, b.0.ptr);
        let mut pga = ga.as_ref().map_or(0, |x| x.0.ptr);
        let mut pgb = gb.as_ref().map_or(0, |x| x.0.ptr);
        gpu.run(6, Gpu::blocks(n), (256, 1), &mut [arg(&mut op), arg(&mut mode), arg(&mut nn), arg(&mut inner), arg(&mut need), arg(&mut pg), arg(&mut pa), arg(&mut pb), arg(&mut pga), arg(&mut pgb)])?;
        Ok((ga, gb))
    })
}

pub fn un_back(op: u32, g: &Buf, x: &Buf, y: &Buf, n: usize) -> Result<Buf, String> {
    with_gpu(|gpu| {
        let out = gpu.alloc(n)?;
        let (mut op, mut nn) = (op, n as u32);
        let (mut pg, mut px, mut py, mut po) = (g.0.ptr, x.0.ptr, y.0.ptr, out.0.ptr);
        gpu.run(7, Gpu::blocks(n), (256, 1), &mut [arg(&mut op), arg(&mut nn), arg(&mut pg), arg(&mut px), arg(&mut py), arg(&mut po)])?;
        Ok(out)
    })
}

pub fn col_sums(t: &Buf, n: usize, inner: usize, square: bool) -> Result<Buf, String> {
    with_gpu(|g| {
        let inner = inner.max(1);
        let out = g.alloc(inner)?;
        let (mut nn, mut ii, mut rows, mut sq) = (n as u32, inner as u32, n.div_ceil(inner) as u32, square as u32);
        let (mut pt, mut po) = (t.0.ptr, out.0.ptr);
        g.run(8, Gpu::blocks(inner), (256, 1), &mut [arg(&mut nn), arg(&mut ii), arg(&mut rows), arg(&mut sq), arg(&mut pt), arg(&mut po)])?;
        Ok(out)
    })
}

pub fn sum_all(t: &Buf, n: usize, square: bool) -> Result<f64, String> {
    let width = n.clamp(1, 1024);
    let partial = col_sums(t, n, width, square)?;
    let total = col_sums(&partial, width, 1, false)?;
    Ok(download(&total, 1)?[0])
}

pub fn optimise(method: u32, p: &Buf, g: &Buf, m: &Buf, v: &Buf, n: usize, k: [f32; 5]) -> Result<Buf, String> {
    with_gpu(|gpu| {
        let out = gpu.alloc(n)?;
        let (mut method, mut nn) = (method, n as u32);
        let (mut rate, mut c1, mut c2, mut decay, mut scale) = (k[0], k[1], k[2], k[3], k[4]);
        let (mut pp, mut pg, mut po, mut pm, mut pv) = (p.0.ptr, g.0.ptr, out.0.ptr, m.0.ptr, v.0.ptr);
        gpu.run(
            9,
            Gpu::blocks(n),
            (256, 1),
            &mut [arg(&mut method), arg(&mut nn), arg(&mut rate), arg(&mut c1), arg(&mut c2), arg(&mut decay), arg(&mut scale), arg(&mut pp), arg(&mut pg), arg(&mut po), arg(&mut pm), arg(&mut pv)],
        )?;
        Ok(out)
    })
}

pub fn matmul(a: &[f64], b: &[f64], m: usize, n: usize, p: usize) -> Result<Vec<f64>, String> {
    let (ab, bb) = (upload(a)?, upload(b)?);
    download(&matmul_bufs(&ab, &bb, m, n, p, false, false)?, m * p)
}
