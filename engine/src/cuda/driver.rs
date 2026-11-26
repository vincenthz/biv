//! The few CUDA driver API calls the backend makes, from the driver library
//! loaded at run time (`nvcuda.dll`, `libcuda.so.1`): nothing to link, and a
//! binary built with the `cuda` feature starts on a machine without one.

use core::cell::Cell;
use core::ffi::{c_char, c_int, c_uint, c_void};
use core::sync::atomic::{AtomicUsize, Ordering};
use std::ffi::CStr;
use std::sync::OnceLock;

use crate::Error;
use crate::prelude::*;

pub(crate) type DevPtr = u64;
type CuResult = c_int;
type Device = c_int;
type Context = *mut c_void;
type Module = *mut c_void;
type Function = *mut c_void;
type RawStream = *mut c_void;
type RawGraph = *mut c_void;
type RawGraphExec = *mut c_void;
type RawEvent = *mut c_void;

const ATTR_MULTIPROCESSOR_COUNT: c_int = 16;
const ATTR_CAN_MAP_HOST_MEMORY: c_int = 19;
const ATTR_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
const ATTR_COMPUTE_CAPABILITY_MINOR: c_int = 76;
const JIT_ERROR_LOG_BUFFER: c_int = 5;
const JIT_ERROR_LOG_BUFFER_SIZE_BYTES: c_int = 6;
const FUNC_MAX_DYNAMIC_SHARED_SIZE_BYTES: c_int = 8;
const MEMHOSTALLOC_PORTABLE: c_uint = 1;
const MEMHOSTALLOC_DEVICEMAP: c_uint = 2;
const MEMHOSTALLOC_WRITECOMBINED: c_uint = 4;
const EVENT_DISABLE_TIMING: c_uint = 2;
const STREAM_CAPTURE_MODE_THREAD_LOCAL: c_int = 1;

macro_rules! api {
    ($($name:ident: fn($($arg:ty),*) = $sym:literal;)*) => {
        #[allow(non_snake_case)]
        struct Api {
            $($name: unsafe extern "system" fn($($arg),*) -> CuResult,)*
        }

        impl Api {
            /// # Safety
            /// `lib` is the CUDA driver library
            unsafe fn load(lib: *mut c_void) -> Result<Self, String> {
                Ok(Api {
                    $($name: unsafe {
                        let f = os::symbol(lib, concat!($sym, "\0"));
                        if f.is_null() {
                            return Err(format!("the driver has no {}", $sym));
                        }
                        core::mem::transmute::<*mut c_void, unsafe extern "system" fn($($arg),*) -> CuResult>(f)
                    },)*
                })
            }
        }
    };
}

api! {
    init: fn(c_uint) = "cuInit";
    driver_version: fn(*mut c_int) = "cuDriverGetVersion";
    device_get: fn(*mut Device, c_int) = "cuDeviceGet";
    device_name: fn(*mut c_char, c_int, Device) = "cuDeviceGetName";
    device_attribute: fn(*mut c_int, c_int, Device) = "cuDeviceGetAttribute";
    device_total_mem: fn(*mut usize, Device) = "cuDeviceTotalMem_v2";
    primary_ctx_retain: fn(*mut Context, Device) = "cuDevicePrimaryCtxRetain";
    ctx_set_current: fn(Context) = "cuCtxSetCurrent";
    ctx_synchronize: fn() = "cuCtxSynchronize";
    module_load: fn(*mut Module, *const c_void, c_uint, *mut c_int, *mut *mut c_void) = "cuModuleLoadDataEx";
    module_function: fn(*mut Function, Module, *const c_char) = "cuModuleGetFunction";
    func_set_attribute: fn(Function, c_int, c_int) = "cuFuncSetAttribute";
    launch_kernel: fn(Function, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, *mut c_void, *mut *mut c_void, *mut *mut c_void) = "cuLaunchKernel";
    mem_alloc: fn(*mut DevPtr, usize) = "cuMemAlloc_v2";
    mem_free: fn(DevPtr) = "cuMemFree_v2";
    mem_info: fn(*mut usize, *mut usize) = "cuMemGetInfo_v2";
    memcpy_htod: fn(DevPtr, *const c_void, usize) = "cuMemcpyHtoD_v2";
    memcpy_dtoh: fn(*mut c_void, DevPtr, usize) = "cuMemcpyDtoH_v2";
    memset_d8: fn(DevPtr, u8, usize) = "cuMemsetD8_v2";
    host_alloc: fn(*mut *mut c_void, usize, c_uint) = "cuMemHostAlloc";
    host_free: fn(*mut c_void) = "cuMemFreeHost";
    host_device_pointer: fn(*mut DevPtr, *mut c_void, c_uint) = "cuMemHostGetDevicePointer_v2";
    error_name: fn(CuResult, *mut *const c_char) = "cuGetErrorName";
    stream_create: fn(*mut RawStream, c_uint) = "cuStreamCreate";
    stream_destroy: fn(RawStream) = "cuStreamDestroy_v2";
    stream_synchronize: fn(RawStream) = "cuStreamSynchronize";
    begin_capture: fn(RawStream, c_int) = "cuStreamBeginCapture_v2";
    end_capture: fn(RawStream, *mut RawGraph) = "cuStreamEndCapture";
    graph_instantiate: fn(*mut RawGraphExec, RawGraph, u64) = "cuGraphInstantiateWithFlags";
    graph_launch: fn(RawGraphExec, RawStream) = "cuGraphLaunch";
    graph_exec_destroy: fn(RawGraphExec) = "cuGraphExecDestroy";
    graph_destroy: fn(RawGraph) = "cuGraphDestroy";
    memcpy_htod_async: fn(DevPtr, *const c_void, usize, RawStream) = "cuMemcpyHtoDAsync_v2";
    memcpy_dtoh_async: fn(*mut c_void, DevPtr, usize, RawStream) = "cuMemcpyDtoHAsync_v2";
    event_create: fn(*mut RawEvent, c_uint) = "cuEventCreate";
    event_destroy: fn(RawEvent) = "cuEventDestroy_v2";
    event_record: fn(RawEvent, RawStream) = "cuEventRecord";
    stream_wait_event: fn(RawStream, RawEvent, c_uint) = "cuStreamWaitEvent";
}

std::thread_local! {
    /// The stream this thread's launches and async copies go to: the legacy
    /// default stream, or the one of [`Stream::run`]
    static STREAM: Cell<RawStream> = const { Cell::new(core::ptr::null_mut()) };
}

fn current_stream() -> RawStream {
    STREAM.with(Cell::get)
}

#[cfg(windows)]
mod os {
    use core::ffi::{c_char, c_void};

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryA(name: *const c_char) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
    }

    pub(super) fn open() -> *mut c_void {
        unsafe { LoadLibraryA(c"nvcuda.dll".as_ptr()) }
    }

    /// # Safety
    /// `name` ends with a nul
    pub(super) unsafe fn symbol(lib: *mut c_void, name: &str) -> *mut c_void {
        unsafe { GetProcAddress(lib, name.as_ptr().cast()) }
    }
}

#[cfg(unix)]
mod os {
    use core::ffi::{c_char, c_int, c_void};

    #[cfg_attr(all(target_os = "linux", target_env = "gnu"), link(name = "dl"))]
    unsafe extern "C" {
        fn dlopen(name: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(lib: *mut c_void, name: *const c_char) -> *mut c_void;
    }

    const RTLD_NOW: c_int = 2;

    pub(super) fn open() -> *mut c_void {
        for name in [c"libcuda.so.1", c"libcuda.so"] {
            let lib = unsafe { dlopen(name.as_ptr(), RTLD_NOW) };
            if !lib.is_null() {
                return lib;
            }
        }
        core::ptr::null_mut()
    }

    /// # Safety
    /// `name` ends with a nul
    pub(super) unsafe fn symbol(lib: *mut c_void, name: &str) -> *mut c_void {
        unsafe { dlsym(lib, name.as_ptr().cast()) }
    }
}

/// The driver, initialized, and the primary context of device 0
pub(crate) struct Driver {
    api: Api,
    ctx: Context,
    device: Device,
    /// bytes this process allocated on the device, and in host memory mapped to it
    pub(crate) device_bytes: AtomicUsize,
    pub(crate) host_bytes: AtomicUsize,
}

// the driver API is thread safe; the context is made current on each thread
// that calls it
unsafe impl Send for Driver {}
unsafe impl Sync for Driver {}

static DRIVER: OnceLock<Result<Driver, String>> = OnceLock::new();

/// The driver, loaded on first use
pub(crate) fn driver() -> Result<&'static Driver, Error> {
    DRIVER
        .get_or_init(Driver::load)
        .as_ref()
        .map_err(|e| Error::Backend(format!("cuda: {e}")))
}

/// [`driver`], once it loaded
fn loaded() -> &'static Driver {
    match DRIVER.get() {
        Some(Ok(d)) => d,
        _ => unreachable!("cuda: the driver is loaded before anything is allocated"),
    }
}

impl Driver {
    fn load() -> Result<Self, String> {
        let lib = os::open();
        if lib.is_null() {
            return Err("no NVIDIA driver library (nvcuda.dll / libcuda.so)".into());
        }
        let api = unsafe { Api::load(lib)? };
        let d = Driver {
            api,
            ctx: core::ptr::null_mut(),
            device: 0,
            device_bytes: AtomicUsize::new(0),
            host_bytes: AtomicUsize::new(0),
        };
        d.check(unsafe { (d.api.init)(0) }, "cuInit")
            .map_err(|e| e.to_string())?;
        let mut device = 0;
        d.check(unsafe { (d.api.device_get)(&mut device, 0) }, "cuDeviceGet")
            .map_err(|e| e.to_string())?;
        let mut ctx = core::ptr::null_mut();
        d.check(
            unsafe { (d.api.primary_ctx_retain)(&mut ctx, device) },
            "cuDevicePrimaryCtxRetain",
        )
        .map_err(|e| e.to_string())?;
        Ok(Driver { ctx, device, ..d })
    }

    fn check(&self, r: CuResult, what: &str) -> Result<(), Error> {
        if r == 0 {
            return Ok(());
        }
        let mut name: *const c_char = core::ptr::null();
        unsafe { (self.api.error_name)(r, &mut name) };
        let name = if name.is_null() {
            format!("error {r}")
        } else {
            unsafe { CStr::from_ptr(name) }
                .to_string_lossy()
                .into_owned()
        };
        Err(Error::Backend(format!("cuda: {what}: {name}")))
    }

    /// Panic on a failed call where an error can't be returned: a pass
    fn must(&self, r: CuResult, what: &str) {
        if let Err(e) = self.check(r, what) {
            panic!("{e}");
        }
    }

    /// Make the context current on this thread
    pub(crate) fn bind(&self) {
        self.must(
            unsafe { (self.api.ctx_set_current)(self.ctx) },
            "cuCtxSetCurrent",
        );
    }

    pub(crate) fn name(&self) -> String {
        let mut buf = [0 as c_char; 256];
        self.must(
            unsafe { (self.api.device_name)(buf.as_mut_ptr(), buf.len() as c_int, self.device) },
            "cuDeviceGetName",
        );
        unsafe { CStr::from_ptr(buf.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    }

    fn attribute(&self, attr: c_int) -> i32 {
        let mut v = 0;
        self.must(
            unsafe { (self.api.device_attribute)(&mut v, attr, self.device) },
            "cuDeviceGetAttribute",
        );
        v
    }

    pub(crate) fn compute_capability(&self) -> (i32, i32) {
        (
            self.attribute(ATTR_COMPUTE_CAPABILITY_MAJOR),
            self.attribute(ATTR_COMPUTE_CAPABILITY_MINOR),
        )
    }

    pub(crate) fn multiprocessors(&self) -> usize {
        self.attribute(ATTR_MULTIPROCESSOR_COUNT) as usize
    }

    pub(crate) fn can_map_host(&self) -> bool {
        self.attribute(ATTR_CAN_MAP_HOST_MEMORY) != 0
    }

    pub(crate) fn version(&self) -> i32 {
        let mut v = 0;
        self.must(
            unsafe { (self.api.driver_version)(&mut v) },
            "cuDriverGetVersion",
        );
        v
    }

    pub(crate) fn total_memory(&self) -> usize {
        let mut v = 0;
        self.must(
            unsafe { (self.api.device_total_mem)(&mut v, self.device) },
            "cuDeviceTotalMem",
        );
        v
    }

    /// (free, total) bytes of the device
    pub(crate) fn memory_info(&self) -> (usize, usize) {
        self.bind();
        let (mut free, mut total) = (0, 0);
        self.must(
            unsafe { (self.api.mem_info)(&mut free, &mut total) },
            "cuMemGetInfo",
        );
        (free, total)
    }

    /// A module of kernels: machine code (a cubin), or PTX for the driver
    /// to compile
    pub(crate) fn module(&self, image: &[u8], ptx: bool) -> Result<Module, Error> {
        self.bind();
        let mut src = Vec::with_capacity(image.len() + 1);
        src.extend_from_slice(image);
        if ptx {
            src.push(0);
        }
        let mut log = vec![0u8; 16384];
        let mut opts = [JIT_ERROR_LOG_BUFFER, JIT_ERROR_LOG_BUFFER_SIZE_BYTES];
        let mut vals = [log.as_mut_ptr().cast::<c_void>(), log.len() as *mut c_void];
        let mut m = core::ptr::null_mut();
        let r = unsafe {
            (self.api.module_load)(
                &mut m,
                src.as_ptr().cast(),
                opts.len() as c_uint,
                opts.as_mut_ptr(),
                vals.as_mut_ptr(),
            )
        };
        if r != 0 {
            let end = log.iter().position(|&b| b == 0).unwrap_or(log.len());
            let msg = String::from_utf8_lossy(&log[..end]).trim().to_string();
            let e = self.check(r, "loading the kernels").unwrap_err();
            return Err(if msg.is_empty() {
                e
            } else {
                Error::Backend(format!("{e}: {msg}"))
            });
        }
        Ok(m)
    }

    pub(crate) fn function(&self, m: Module, name: &str) -> Result<Func, Error> {
        let cname = std::ffi::CString::new(name).expect("a kernel name");
        let mut f = core::ptr::null_mut();
        self.check(
            unsafe { (self.api.module_function)(&mut f, m, cname.as_ptr()) },
            &format!("kernel {name}"),
        )?;
        Ok(Func(f))
    }

    /// Let `f` use up to `bytes` of dynamic shared memory
    pub(crate) fn max_shared(&self, f: Func, bytes: usize) -> Result<(), Error> {
        self.check(
            unsafe {
                (self.api.func_set_attribute)(
                    f.0,
                    FUNC_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    bytes as c_int,
                )
            },
            "cuFuncSetAttribute",
        )
    }

    /// Launch `f` on the current stream with its one argument, `args`
    pub(crate) fn launch<T>(
        &self,
        f: Func,
        grid: [u32; 3],
        block: [u32; 3],
        shared: u32,
        args: &T,
    ) {
        if grid.contains(&0) {
            return;
        }
        let mut params = [(args as *const T).cast_mut().cast::<c_void>()];
        self.must(
            unsafe {
                (self.api.launch_kernel)(
                    f.0,
                    grid[0],
                    grid[1],
                    grid[2],
                    block[0],
                    block[1],
                    block[2],
                    shared,
                    current_stream(),
                    params.as_mut_ptr(),
                    core::ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        );
    }

    /// Wait for the work queued, panicking with the error of a kernel that failed
    pub(crate) fn sync(&self) {
        self.must(unsafe { (self.api.ctx_synchronize)() }, "a kernel");
    }

    pub(crate) fn upload(&self, dst: DevPtr, src: &[u8]) {
        if !src.is_empty() {
            self.must(
                unsafe { (self.api.memcpy_htod)(dst, src.as_ptr().cast(), src.len()) },
                "cuMemcpyHtoD",
            );
        }
    }

    /// Waits for the work queued before
    pub(crate) fn download(&self, dst: &mut [u8], src: DevPtr) {
        if !dst.is_empty() {
            self.must(
                unsafe { (self.api.memcpy_dtoh)(dst.as_mut_ptr().cast(), src, dst.len()) },
                "cuMemcpyDtoH",
            );
        }
    }
}

/// A stream of the device's work: what runs on it, captured or not
pub(crate) struct Stream(RawStream);

unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

impl Stream {
    /// A stream that synchronizes with the legacy default stream, so that the
    /// blocking copies there wait for its work
    pub(crate) fn new() -> Result<Stream, Error> {
        let d = driver()?;
        d.bind();
        let mut s = core::ptr::null_mut();
        d.check(
            unsafe { (d.api.stream_create)(&mut s, 0) },
            "cuStreamCreate",
        )?;
        Ok(Stream(s))
    }

    /// `f` with the launches and async copies of this thread on this stream
    pub(crate) fn run<R>(&self, f: impl FnOnce() -> R) -> R {
        let prev = STREAM.with(|c| c.replace(self.0));
        let r = f();
        STREAM.with(|c| c.set(prev));
        r
    }

    /// The work `f` queues on this stream, as a graph to launch instead
    pub(crate) fn capture(&self, f: impl FnOnce()) -> Result<Graph, Error> {
        let d = loaded();
        d.check(
            unsafe { (d.api.begin_capture)(self.0, STREAM_CAPTURE_MODE_THREAD_LOCAL) },
            "cuStreamBeginCapture",
        )?;
        self.run(f);
        let mut graph = core::ptr::null_mut();
        d.check(
            unsafe { (d.api.end_capture)(self.0, &mut graph) },
            "cuStreamEndCapture",
        )?;
        let mut exec = core::ptr::null_mut();
        let r = unsafe { (d.api.graph_instantiate)(&mut exec, graph, 0) };
        if let Err(e) = d.check(r, "cuGraphInstantiate") {
            unsafe { (d.api.graph_destroy)(graph) };
            return Err(e);
        }
        Ok(Graph { graph, exec })
    }

    /// Launch `g` on this stream and wait for it, panicking with the error of
    /// a kernel that failed
    pub(crate) fn launch_and_wait(&self, g: &Graph) {
        let d = loaded();
        d.must(
            unsafe { (d.api.graph_launch)(g.exec, self.0) },
            "cuGraphLaunch",
        );
        d.must(unsafe { (d.api.stream_synchronize)(self.0) }, "a kernel");
    }
}

/// A point in a stream's work that another stream waits for
pub(crate) struct Event(RawEvent);

unsafe impl Send for Event {}
unsafe impl Sync for Event {}

impl Event {
    pub(crate) fn new() -> Result<Event, Error> {
        let d = driver()?;
        d.bind();
        let mut e = core::ptr::null_mut();
        d.check(
            unsafe { (d.api.event_create)(&mut e, EVENT_DISABLE_TIMING) },
            "cuEventCreate",
        )?;
        Ok(Event(e))
    }

    /// Mark the work queued so far on the current stream
    pub(crate) fn record(&self) {
        let d = loaded();
        d.must(
            unsafe { (d.api.event_record)(self.0, current_stream()) },
            "cuEventRecord",
        );
    }

    /// Make the current stream's next work wait for the last mark
    pub(crate) fn wait(&self) {
        let d = loaded();
        d.must(
            unsafe { (d.api.stream_wait_event)(current_stream(), self.0, 0) },
            "cuStreamWaitEvent",
        );
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        let d = loaded();
        d.bind();
        unsafe { (d.api.event_destroy)(self.0) };
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        let d = loaded();
        d.bind();
        unsafe { (d.api.stream_destroy)(self.0) };
    }
}

/// Work captured from a stream, instantiated
pub(crate) struct Graph {
    graph: RawGraph,
    exec: RawGraphExec,
}

unsafe impl Send for Graph {}
unsafe impl Sync for Graph {}

impl Drop for Graph {
    fn drop(&mut self) {
        let d = loaded();
        d.bind();
        unsafe {
            (d.api.graph_exec_destroy)(self.exec);
            (d.api.graph_destroy)(self.graph);
        }
    }
}

/// Pinned host memory, for copies queued on a stream
pub(crate) struct HostBuf {
    ptr: *mut u8,
    bytes: usize,
}

unsafe impl Send for HostBuf {}
unsafe impl Sync for HostBuf {}

impl HostBuf {
    pub(crate) fn new(bytes: usize) -> Result<HostBuf, Error> {
        let d = driver()?;
        d.bind();
        let n = bytes.max(4);
        let mut host = core::ptr::null_mut();
        if unsafe { (d.api.host_alloc)(&mut host, n, MEMHOSTALLOC_PORTABLE) } != 0 {
            return Err(Error::OutOfMemory { bytes: n });
        }
        unsafe { core::ptr::write_bytes(host.cast::<u8>(), 0, n) };
        Ok(HostBuf {
            ptr: host.cast(),
            bytes: n,
        })
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self.ptr, self.bytes) }
    }

    pub(crate) fn bytes_mut(&mut self) -> &mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(self.ptr, self.bytes) }
    }

    /// Queue the copy of `len` bytes from `offset` to `dst`, on the current stream
    pub(crate) fn upload(&self, offset: usize, len: usize, dst: DevPtr) {
        assert!(offset + len <= self.bytes);
        let d = loaded();
        let src = unsafe { self.ptr.add(offset) };
        d.must(
            unsafe { (d.api.memcpy_htod_async)(dst, src.cast(), len, current_stream()) },
            "cuMemcpyHtoDAsync",
        );
    }

    /// Queue the copy of `len` bytes from `src` to `offset`, on the current stream
    pub(crate) fn download(&self, offset: usize, len: usize, src: DevPtr) {
        assert!(offset + len <= self.bytes);
        let d = loaded();
        let dst = unsafe { self.ptr.add(offset) };
        d.must(
            unsafe { (d.api.memcpy_dtoh_async)(dst.cast(), src, len, current_stream()) },
            "cuMemcpyDtoHAsync",
        );
    }
}

impl Drop for HostBuf {
    fn drop(&mut self) {
        let d = loaded();
        d.bind();
        unsafe { (d.api.host_free)(self.ptr.cast()) };
    }
}

/// A kernel of the loaded module
#[derive(Clone, Copy)]
pub(crate) struct Func(Function);

unsafe impl Send for Func {}
unsafe impl Sync for Func {}

/// Memory the device reads: its own, or pinned host memory mapped to it
pub(crate) struct Buf {
    pub(crate) ptr: DevPtr,
    pub(crate) bytes: usize,
    /// the host address of mapped host memory
    host: *mut c_void,
    /// a part of `arena` (see [`Buf::carve`]), or of nothing: not freed
    view: bool,
    arena: Option<alloc::sync::Arc<Buf>>,
}

unsafe impl Send for Buf {}
unsafe impl Sync for Buf {}

impl Buf {
    /// `bytes` of device memory, zeroed
    pub(crate) fn zeroed(bytes: usize) -> Result<Buf, Error> {
        let d = driver()?;
        d.bind();
        let mut ptr = 0;
        // a buffer of nothing still gets an address to bind
        let n = bytes.max(4);
        let r = unsafe { (d.api.mem_alloc)(&mut ptr, n) };
        if r != 0 {
            return Err(Error::OutOfMemory { bytes });
        }
        d.must(unsafe { (d.api.memset_d8)(ptr, 0, n) }, "cuMemsetD8");
        d.device_bytes.fetch_add(n, Ordering::Relaxed);
        Ok(Buf {
            ptr,
            bytes: n,
            host: core::ptr::null_mut(),
            view: false,
            arena: None,
        })
    }

    /// Device memory holding `data`
    pub(crate) fn from_bytes(data: &[u8]) -> Result<Buf, Error> {
        let b = Buf::zeroed(data.len())?;
        loaded().upload(b.ptr, data);
        Ok(b)
    }

    pub(crate) fn from_f32(data: &[f32]) -> Result<Buf, Error> {
        Buf::from_bytes(f32_bytes(data))
    }

    /// Pinned host memory mapped to the device, holding `data`: the device
    /// reads it across the bus
    pub(crate) fn mapped(data: &[u8]) -> Result<Buf, Error> {
        let b = Buf::mapped_uninit(data.len())?;
        unsafe { core::ptr::copy_nonoverlapping(data.as_ptr(), b.host.cast::<u8>(), data.len()) };
        Ok(b)
    }

    /// `bytes` of pinned host memory mapped to the device, not initialized
    pub(crate) fn mapped_uninit(bytes: usize) -> Result<Buf, Error> {
        let d = driver()?;
        d.bind();
        let n = bytes.max(4);
        let mut host = core::ptr::null_mut();
        let flags = MEMHOSTALLOC_PORTABLE | MEMHOSTALLOC_DEVICEMAP | MEMHOSTALLOC_WRITECOMBINED;
        if unsafe { (d.api.host_alloc)(&mut host, n, flags) } != 0 {
            return Err(Error::OutOfMemory { bytes: n });
        }
        let mut ptr = 0;
        let r = d.check(
            unsafe { (d.api.host_device_pointer)(&mut ptr, host, 0) },
            "cuMemHostGetDevicePointer",
        );
        if let Err(e) = r {
            unsafe { (d.api.host_free)(host) };
            return Err(e);
        }
        d.host_bytes.fetch_add(n, Ordering::Relaxed);
        Ok(Buf {
            ptr,
            bytes: n,
            host,
            view: false,
            arena: None,
        })
    }

    /// `data` at `offset` of the mapped host memory `arena`, a buffer of its
    /// own that keeps the arena
    pub(crate) fn carve(arena: &alloc::sync::Arc<Buf>, offset: usize, data: &[u8]) -> Buf {
        assert!(arena.is_host() && offset + data.len() <= arena.bytes);
        let host = unsafe { arena.host.cast::<u8>().add(offset) };
        unsafe { core::ptr::copy_nonoverlapping(data.as_ptr(), host, data.len()) };
        Buf {
            ptr: arena.ptr + offset as u64,
            bytes: data.len(),
            host: host.cast(),
            view: true,
            arena: Some(arena.clone()),
        }
    }

    /// A buffer of nothing, at no address
    pub(crate) fn none() -> Buf {
        Buf {
            ptr: 0,
            bytes: 0,
            host: core::ptr::null_mut(),
            view: true,
            arena: None,
        }
    }

    /// Whether this is host memory mapped to the device
    pub(crate) fn is_host(&self) -> bool {
        !self.host.is_null()
    }

    /// Queue the copy of this mapped host memory's first `bytes` to `dst`, on
    /// the current stream: the copy engine at the bus's speed
    pub(crate) fn copy_to_device(&self, dst: DevPtr, bytes: usize) {
        assert!(self.is_host() && bytes <= self.bytes);
        let d = loaded();
        d.must(
            unsafe {
                (d.api.memcpy_htod_async)(dst, self.host.cast_const(), bytes, current_stream())
            },
            "cuMemcpyHtoDAsync",
        );
    }

    /// Address `bytes` into the buffer
    pub(crate) fn at(&self, bytes: usize) -> DevPtr {
        debug_assert!(bytes <= self.bytes);
        self.ptr + bytes as u64
    }

    /// Address `n` f32 into the buffer
    pub(crate) fn f32_at(&self, n: usize) -> DevPtr {
        self.at(4 * n)
    }

    /// Copy `dst.len()` bytes from `offset` out, after the work queued
    pub(crate) fn read(&self, offset: usize, dst: &mut [u8]) {
        debug_assert!(offset + dst.len() <= self.bytes);
        loaded().download(dst, self.at(offset));
    }

    pub(crate) fn read_f32(&self, offset: usize, n: usize) -> Vec<f32> {
        let mut v = vec![0f32; n];
        self.read(4 * offset, f32_bytes_mut(&mut v));
        v
    }

    pub(crate) fn write(&self, offset: usize, src: &[u8]) {
        debug_assert!(offset + src.len() <= self.bytes);
        loaded().upload(self.at(offset), src);
    }
}

impl Drop for Buf {
    fn drop(&mut self) {
        if self.view {
            return;
        }
        let d = loaded();
        d.bind();
        if self.host.is_null() {
            unsafe { (d.api.mem_free)(self.ptr) };
            d.device_bytes.fetch_sub(self.bytes, Ordering::Relaxed);
        } else {
            unsafe { (d.api.host_free)(self.host) };
            d.host_bytes.fetch_sub(self.bytes, Ordering::Relaxed);
        }
    }
}

pub(crate) fn f32_bytes(v: &[f32]) -> &[u8] {
    unsafe { core::slice::from_raw_parts(v.as_ptr().cast(), 4 * v.len()) }
}

/// Bytes as f32, 4-aligned
pub(crate) fn f32_slice(b: &[u8]) -> &[f32] {
    assert!(b.as_ptr().cast::<f32>().is_aligned());
    unsafe { core::slice::from_raw_parts(b.as_ptr().cast(), b.len() / 4) }
}

pub(crate) fn f32_slice_mut(b: &mut [u8]) -> &mut [f32] {
    assert!(b.as_ptr().cast::<f32>().is_aligned());
    unsafe { core::slice::from_raw_parts_mut(b.as_mut_ptr().cast(), b.len() / 4) }
}

pub(crate) fn f32_bytes_mut(v: &mut [f32]) -> &mut [u8] {
    unsafe { core::slice::from_raw_parts_mut(v.as_mut_ptr().cast(), 4 * v.len()) }
}
