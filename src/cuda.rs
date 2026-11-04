// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials

use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub use crate::vulkan::{
    build_gpu_lbvh, build_gpu_lbvh_from_segments, evaluate_drc_brute_force_cpu,
    evaluate_segment_crossings_brute_force_cpu, morton2d_encode, ray_aabb_intersects_cpu,
    segments_cross_cpu, GpuBvhNode, GpuDrcRayQuery, GpuRouteCostParams, GpuSaParams,
    PlacementBatchParams, RtBvhBenchmarkResult,
};

pub const KERNELS_PTX: &str = include_str!("../cuda/kernels.ptx");
pub const AMPERE_KERNELS_PTX: &str = include_str!("../cuda/ampere_kernels.ptx");

pub type CUresult = c_uint;
pub type CUdevice = c_int;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUdeviceptr = u64;

pub const CUDA_SUCCESS: CUresult = 0;

pub type OptixQueryFunctionTableFn = unsafe extern "C" fn(
    abi_id: u32,
    num_options: u32,
    option_keys: *const c_void,
    option_vals: *const *const c_void,
    function_table: *mut c_void,
    table_size: usize,
) -> u32;

#[allow(non_snake_case)]
pub struct OptixDriverApi {
    _lib: libloading::Library,
    pub optixQueryFunctionTable: OptixQueryFunctionTableFn,
    pub abi_version: u32,
    pub optix_version: u32,
}

impl OptixDriverApi {
    #[allow(non_snake_case)]
    pub fn load() -> Option<Arc<Self>> {
        static OPTIX_ONCE: std::sync::OnceLock<Option<Arc<OptixDriverApi>>> = std::sync::OnceLock::new();
        OPTIX_ONCE
            .get_or_init(|| {
                let lib_paths = [
                    "/usr/lib/x86_64-linux-gnu/libnvoptix.so.1",
                    "libnvoptix.so.1",
                    "libnvoptix.so",
                ];
                for path in &lib_paths {
                    if let Ok(lib) = unsafe { libloading::Library::new(path) } {
                        unsafe {
                            let query_fn: OptixQueryFunctionTableFn =
                                match lib.get(b"optixQueryFunctionTable\0") {
                                    Ok(sym) => *sym,
                                    Err(_) => continue,
                                };

                            let mut fn_table = [0usize; 128];
                            let known_abis: &[(u32, usize, u32)] = &[
                                (119, 0x1e0, 90000),
                                (105, 0x1b0, 80100),
                                (89, 0x180, 80000),
                                (78, 0x178, 70700),
                                (68, 0x158, 70600),
                                (60, 0x148, 70500),
                                (55, 0x138, 70400),
                                (47, 0x130, 70300),
                                (41, 0x128, 70200),
                                (36, 0x108, 70100),
                                (22, 0x0c8, 70000),
                            ];

                            for &(abi, table_bytes, ver) in known_abis {
                                let res = query_fn(
                                    abi,
                                    0,
                                    ptr::null(),
                                    ptr::null(),
                                    fn_table.as_mut_ptr() as *mut c_void,
                                    table_bytes,
                                );
                                if res == 0 && fn_table[0] != 0 && fn_table[2] != 0 {
                                    return Some(Arc::new(Self {
                                        _lib: lib,
                                        optixQueryFunctionTable: query_fn,
                                        abi_version: abi,
                                        optix_version: ver,
                                    }));
                                }
                            }

                            for abi in (18..=140).rev() {
                                let probe = query_fn(
                                    abi,
                                    0,
                                    ptr::null(),
                                    ptr::null(),
                                    fn_table.as_mut_ptr() as *mut c_void,
                                    0,
                                );
                                if probe == 0x1e7a || probe == 0 {
                                    for words in 16..=96 {
                                        let res = query_fn(
                                            abi,
                                            0,
                                            ptr::null(),
                                            ptr::null(),
                                            fn_table.as_mut_ptr() as *mut c_void,
                                            words * 8,
                                        );
                                        if res == 0 && fn_table[0] != 0 {
                                            return Some(Arc::new(Self {
                                                _lib: lib,
                                                optixQueryFunctionTable: query_fn,
                                                abi_version: abi,
                                                optix_version: abi * 1000,
                                            }));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                None
            })
            .clone()
    }
}

#[allow(non_snake_case)]
pub struct CudaDriverApi {
    _lib: libloading::Library,
    pub cuInit: unsafe extern "C" fn(flags: c_uint) -> CUresult,
    pub cuDriverGetVersion: unsafe extern "C" fn(version: *mut c_int) -> CUresult,
    pub cuDeviceGetCount: unsafe extern "C" fn(count: *mut c_int) -> CUresult,
    pub cuDeviceGet: unsafe extern "C" fn(device: *mut CUdevice, ordinal: c_int) -> CUresult,
    pub cuDeviceGetName: unsafe extern "C" fn(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult,
    pub cuCtxCreate_v2: unsafe extern "C" fn(pctx: *mut CUcontext, flags: c_uint, dev: CUdevice) -> CUresult,
    pub cuCtxDestroy_v2: unsafe extern "C" fn(ctx: CUcontext) -> CUresult,
    pub cuCtxSetCurrent: unsafe extern "C" fn(ctx: CUcontext) -> CUresult,
    pub cuCtxSynchronize: unsafe extern "C" fn() -> CUresult,
    pub cuModuleLoadData: unsafe extern "C" fn(module: *mut CUmodule, image: *const c_void) -> CUresult,
    pub cuModuleUnload: unsafe extern "C" fn(module: CUmodule) -> CUresult,
    pub cuModuleGetFunction: unsafe extern "C" fn(hfunc: *mut CUfunction, hmod: CUmodule, name: *const c_char) -> CUresult,
    pub cuMemAlloc_v2: unsafe extern "C" fn(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult,
    pub cuMemFree_v2: unsafe extern "C" fn(dptr: CUdeviceptr) -> CUresult,
    pub cuMemcpyHtoD_v2: unsafe extern "C" fn(dstDevice: CUdeviceptr, srcHost: *const c_void, ByteCount: usize) -> CUresult,
    pub cuMemcpyDtoH_v2: unsafe extern "C" fn(dstHost: *mut c_void, srcDevice: CUdeviceptr, ByteCount: usize) -> CUresult,
    pub cuLaunchKernel: unsafe extern "C" fn(
        f: CUfunction,
        gridDimX: c_uint,
        gridDimY: c_uint,
        gridDimZ: c_uint,
        blockDimX: c_uint,
        blockDimY: c_uint,
        blockDimZ: c_uint,
        sharedMemBytes: c_uint,
        hStream: CUstream,
        kernelParams: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult,
}

impl CudaDriverApi {
    #[allow(non_snake_case)]
    pub fn load() -> Option<Arc<Self>> {
        static CUDA_ONCE: std::sync::OnceLock<Option<Arc<CudaDriverApi>>> = std::sync::OnceLock::new();
        CUDA_ONCE
            .get_or_init(|| {
                let lib_paths = ["libcuda.so.1", "libcuda.so"];
                for path in &lib_paths {
                    if let Ok(lib) = unsafe { libloading::Library::new(path) } {
                        unsafe {
                            let cuInit: unsafe extern "C" fn(flags: c_uint) -> CUresult =
                                match lib.get(b"cuInit\0") {
                                    Ok(sym) => *sym,
                                    Err(_) => continue,
                                };
                            if cuInit(0) != CUDA_SUCCESS {
                                continue;
                            }
                            let cuDriverGetVersion = match lib.get(b"cuDriverGetVersion\0") {
                                Ok(sym) => *sym,
                                Err(_) => continue,
                            };
                            let cuDeviceGetCount = match lib.get(b"cuDeviceGetCount\0") {
                                Ok(sym) => *sym,
                                Err(_) => continue,
                            };
                            let cuDeviceGet = match lib.get(b"cuDeviceGet\0") {
                                Ok(sym) => *sym,
                                Err(_) => continue,
                            };
                            let cuDeviceGetName = match lib.get(b"cuDeviceGetName\0") {
                                Ok(sym) => *sym,
                                Err(_) => continue,
                            };
                            let cuCtxCreate_v2 = match lib.get(b"cuCtxCreate_v2\0") {
                                Ok(sym) => *sym,
                                Err(_) => match lib.get(b"cuCtxCreate\0") {
                                    Ok(sym) => *sym,
                                    Err(_) => continue,
                                },
                            };
                            let cuCtxDestroy_v2 = match lib.get(b"cuCtxDestroy_v2\0") {
                                Ok(sym) => *sym,
                                Err(_) => match lib.get(b"cuCtxDestroy\0") {
                                    Ok(sym) => *sym,
                                    Err(_) => continue,
                                },
                            };
                            let cuCtxSetCurrent = match lib.get(b"cuCtxSetCurrent\0") {
                                Ok(sym) => *sym,
                                Err(_) => continue,
                            };
                            let cuCtxSynchronize = match lib.get(b"cuCtxSynchronize\0") {
                                Ok(sym) => *sym,
                                Err(_) => continue,
                            };
                            let cuModuleLoadData = match lib.get(b"cuModuleLoadData\0") {
                                Ok(sym) => *sym,
                                Err(_) => continue,
                            };
                            let cuModuleUnload = match lib.get(b"cuModuleUnload\0") {
                                Ok(sym) => *sym,
                                Err(_) => continue,
                            };
                            let cuModuleGetFunction = match lib.get(b"cuModuleGetFunction\0") {
                                Ok(sym) => *sym,
                                Err(_) => continue,
                            };
                            let cuMemAlloc_v2 = match lib.get(b"cuMemAlloc_v2\0") {
                                Ok(sym) => *sym,
                                Err(_) => match lib.get(b"cuMemAlloc\0") {
                                    Ok(sym) => *sym,
                                    Err(_) => continue,
                                },
                            };
                            let cuMemFree_v2 = match lib.get(b"cuMemFree_v2\0") {
                                Ok(sym) => *sym,
                                Err(_) => match lib.get(b"cuMemFree\0") {
                                    Ok(sym) => *sym,
                                    Err(_) => continue,
                                },
                            };
                            let cuMemcpyHtoD_v2 = match lib.get(b"cuMemcpyHtoD_v2\0") {
                                Ok(sym) => *sym,
                                Err(_) => match lib.get(b"cuMemcpyHtoD\0") {
                                    Ok(sym) => *sym,
                                    Err(_) => continue,
                                },
                            };
                            let cuMemcpyDtoH_v2 = match lib.get(b"cuMemcpyDtoH_v2\0") {
                                Ok(sym) => *sym,
                                Err(_) => match lib.get(b"cuMemcpyDtoH\0") {
                                    Ok(sym) => *sym,
                                    Err(_) => continue,
                                },
                            };
                            let cuLaunchKernel = match lib.get(b"cuLaunchKernel\0") {
                                Ok(sym) => *sym,
                                Err(_) => continue,
                            };

                            let api = Self {
                                _lib: lib,
                                cuInit,
                                cuDriverGetVersion,
                                cuDeviceGetCount,
                                cuDeviceGet,
                                cuDeviceGetName,
                                cuCtxCreate_v2,
                                cuCtxDestroy_v2,
                                cuCtxSetCurrent,
                                cuCtxSynchronize,
                                cuModuleLoadData,
                                cuModuleUnload,
                                cuModuleGetFunction,
                                cuMemAlloc_v2,
                                cuMemFree_v2,
                                cuMemcpyHtoD_v2,
                                cuMemcpyDtoH_v2,
                                cuLaunchKernel,
                            };
                            return Some(Arc::new(api));
                        }
                    }
                }
                None
            })
            .clone()
    }
}

struct PersistentDeviceBuffers {
    d_obs: CUdeviceptr,
    obs_cap_bytes: usize,
    d_out: CUdeviceptr,
    out_cap_bytes: usize,
}

pub struct CudaEngine {
    pub device_name: String,
    pub backend_name: String,
    pub driver_info: String,
    pub optix_available: bool,
    pub optix_version: u32,
    ctx: CUcontext,
    module: CUmodule,
    ampere_module: CUmodule,
    fn_dist_field: CUfunction,
    fn_dist_field_tiled: CUfunction,
    fn_dist_field_3d_tiled: CUfunction,
    fn_batch_placement: CUfunction,
    fn_gpu_sa_placer: CUfunction,
    fn_route_cost_3d: CUfunction,
    fn_wavefront_3d: CUfunction,
    fn_rt_bvh_crossings: CUfunction,
    fn_rt_bvh_drc: CUfunction,
    pub fn_df_ampere: Option<CUfunction>,
    pub fn_df_ampere_tiled: Option<CUfunction>,
    pub fn_bp_ampere: Option<CUfunction>,
    cuda: Arc<CudaDriverApi>,
    pub optix: Option<Arc<OptixDriverApi>>,
    dispatch_lock: Mutex<()>,
    persistent_buffers: Mutex<PersistentDeviceBuffers>,
    pub compute_busy_ns: Arc<AtomicU64>,
}

unsafe impl Send for CudaEngine {}
unsafe impl Sync for CudaEngine {}

impl std::fmt::Debug for CudaEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaEngine")
            .field("device_name", &self.device_name)
            .field("backend_name", &self.backend_name)
            .field("optix_available", &self.optix_available)
            .field("optix_version", &self.optix_version)
            .finish()
    }
}

impl Drop for CudaEngine {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.cuda.cuCtxSetCurrent)(self.ctx);
            if let Ok(mut bufs) = self.persistent_buffers.lock() {
                if bufs.d_obs != 0 {
                    let _ = (self.cuda.cuMemFree_v2)(bufs.d_obs);
                    bufs.d_obs = 0;
                    bufs.obs_cap_bytes = 0;
                }
                if bufs.d_out != 0 {
                    let _ = (self.cuda.cuMemFree_v2)(bufs.d_out);
                    bufs.d_out = 0;
                    bufs.out_cap_bytes = 0;
                }
            }
            if !self.ampere_module.is_null() {
                let _ = (self.cuda.cuModuleUnload)(self.ampere_module);
            }
            if !self.module.is_null() {
                let _ = (self.cuda.cuModuleUnload)(self.module);
            }
            if !self.ctx.is_null() {
                let _ = (self.cuda.cuCtxDestroy_v2)(self.ctx);
            }
        }
    }
}

impl CudaEngine {
    pub fn all_devices() -> Vec<Arc<Self>> {
        let api = match CudaDriverApi::load() {
            Some(a) => a,
            None => return Vec::new(),
        };
        unsafe {
            if (api.cuInit)(0) != CUDA_SUCCESS {
                return Vec::new();
            }
            let mut count: c_int = 0;
            if (api.cuDeviceGetCount)(&mut count) != CUDA_SUCCESS || count <= 0 {
                return Vec::new();
            }
            let mut engines = Vec::new();
            for ord in 0..count {
                if let Some(engine) = Self::new_with_device(ord) {
                    engines.push(engine);
                }
            }
            engines.sort_by(|a, b| {
                let score = |name: &str| {
                    if name.contains("5090") { 5090 }
                    else if name.contains("4090") { 4090 }
                    else if name.contains("4080") { 4080 }
                    else if name.contains("3090") { 3090 }
                    else if name.contains("3080") { 3080 }
                    else if name.contains("4070") { 4070 }
                    else if name.contains("3070") { 3070 }
                    else if name.contains("4060") { 4060 }
                    else if name.contains("3060") { 3060 }
                    else { 1000 }
                };
                score(&b.device_name).cmp(&score(&a.device_name))
            });
            engines
        }
    }

    pub fn new() -> Option<Arc<Self>> {
        Self::all_devices().into_iter().next()
    }

    pub fn new_with_device(ordinal: c_int) -> Option<Arc<Self>> {
        static CUDA_INIT_LOCK: Mutex<()> = Mutex::new(());
        let _init_guard = CUDA_INIT_LOCK.lock().ok()?;
        let api = CudaDriverApi::load()?;
        let optix_api = OptixDriverApi::load();
        unsafe {
            if (api.cuInit)(0) != CUDA_SUCCESS {
                return None;
            }

            let mut count: c_int = 0;
            if (api.cuDeviceGetCount)(&mut count) != CUDA_SUCCESS || count <= ordinal {
                return None;
            }

            let mut dev: CUdevice = 0;
            if (api.cuDeviceGet)(&mut dev, ordinal) != CUDA_SUCCESS {
                return None;
            }

            let mut name_buf = [0i8; 256];
            let device_name = if (api.cuDeviceGetName)(name_buf.as_mut_ptr(), 256, dev) == CUDA_SUCCESS {
                CStr::from_ptr(name_buf.as_ptr())
                    .to_string_lossy()
                    .into_owned()
            } else {
                format!("CUDA Device {}", dev)
            };

            let mut drv_version: c_int = 0;
            let _ = (api.cuDriverGetVersion)(&mut drv_version);
            let driver_info = format!("CUDA Driver Version {}.{}", drv_version / 1000, (drv_version % 100) / 10);
            let backend_name = "CUDA PTX (bare-metal)".to_string();

            let mut ctx: CUcontext = ptr::null_mut();
            if (api.cuCtxCreate_v2)(&mut ctx, 0, dev) != CUDA_SUCCESS {
                return None;
            }
            let _ = (api.cuCtxSetCurrent)(ctx);

            let ptx_cstr = CString::new(KERNELS_PTX).ok()?;
            let mut module: CUmodule = ptr::null_mut();
            if (api.cuModuleLoadData)(&mut module, ptx_cstr.as_ptr() as *const c_void) != CUDA_SUCCESS {
                let _ = (api.cuCtxDestroy_v2)(ctx);
                return None;
            }

            let mut fn_dist_field: CUfunction = ptr::null_mut();
            let mut fn_dist_field_tiled: CUfunction = ptr::null_mut();
            let mut fn_dist_field_3d_tiled: CUfunction = ptr::null_mut();
            let mut fn_batch_placement: CUfunction = ptr::null_mut();
            let mut fn_gpu_sa_placer: CUfunction = ptr::null_mut();
            let mut fn_route_cost_3d: CUfunction = ptr::null_mut();
            let mut fn_wavefront_3d: CUfunction = ptr::null_mut();
            let mut fn_rt_bvh_crossings: CUfunction = ptr::null_mut();
            let mut fn_rt_bvh_drc: CUfunction = ptr::null_mut();

            let c_df = CString::new("distance_field_kernel").unwrap();
            let c_dft = CString::new("distance_field_tiled_kernel").unwrap();
            let c_df3d = CString::new("distance_field_3d_tiled_kernel").unwrap();
            let c_bp = CString::new("batch_placement_eval_kernel").unwrap();
            let c_sa = CString::new("gpu_sa_placer_kernel").unwrap();
            let c_rc3d = CString::new("compute_3d_routing_cost_kernel").unwrap();
            let c_wf3d = CString::new("wavefront_3d_relax_kernel").unwrap();
            let c_rt_cr = CString::new("rt_bvh_segment_crossings_kernel").unwrap();
            let c_rt_drc = CString::new("rt_bvh_line_of_sight_drc_kernel").unwrap();

            if (api.cuModuleGetFunction)(&mut fn_dist_field, module, c_df.as_ptr()) != CUDA_SUCCESS
                || (api.cuModuleGetFunction)(&mut fn_dist_field_tiled, module, c_dft.as_ptr()) != CUDA_SUCCESS
                || (api.cuModuleGetFunction)(&mut fn_dist_field_3d_tiled, module, c_df3d.as_ptr()) != CUDA_SUCCESS
                || (api.cuModuleGetFunction)(&mut fn_batch_placement, module, c_bp.as_ptr()) != CUDA_SUCCESS
                || (api.cuModuleGetFunction)(&mut fn_gpu_sa_placer, module, c_sa.as_ptr()) != CUDA_SUCCESS
                || (api.cuModuleGetFunction)(&mut fn_route_cost_3d, module, c_rc3d.as_ptr()) != CUDA_SUCCESS
                || (api.cuModuleGetFunction)(&mut fn_wavefront_3d, module, c_wf3d.as_ptr()) != CUDA_SUCCESS
                || (api.cuModuleGetFunction)(&mut fn_rt_bvh_crossings, module, c_rt_cr.as_ptr()) != CUDA_SUCCESS
                || (api.cuModuleGetFunction)(&mut fn_rt_bvh_drc, module, c_rt_drc.as_ptr()) != CUDA_SUCCESS
            {
                let _ = (api.cuModuleUnload)(module);
                let _ = (api.cuCtxDestroy_v2)(ctx);
                return None;
            }

            let mut ampere_module: CUmodule = ptr::null_mut();
            let mut fn_df_ampere = None;
            let mut fn_df_ampere_tiled = None;
            let mut fn_bp_ampere = None;

            if let Ok(ampere_ptx_cstr) = CString::new(AMPERE_KERNELS_PTX) {
                let res = (api.cuModuleLoadData)(&mut ampere_module, ampere_ptx_cstr.as_ptr() as *const c_void);
                if res == CUDA_SUCCESS {
                    let mut f1: CUfunction = ptr::null_mut();
                    let mut f2: CUfunction = ptr::null_mut();
                    let mut f3: CUfunction = ptr::null_mut();

                    let c_a1 = CString::new("distance_field_ampere").unwrap();
                    let c_a2 = CString::new("distance_field_ampere_tiled").unwrap();
                    let c_a3 = CString::new("batch_placement_ampere").unwrap();

                    if (api.cuModuleGetFunction)(&mut f1, ampere_module, c_a1.as_ptr()) == CUDA_SUCCESS {
                        fn_df_ampere = Some(f1);
                    }
                    if (api.cuModuleGetFunction)(&mut f2, ampere_module, c_a2.as_ptr()) == CUDA_SUCCESS {
                        fn_df_ampere_tiled = Some(f2);
                    }
                    if (api.cuModuleGetFunction)(&mut f3, ampere_module, c_a3.as_ptr()) == CUDA_SUCCESS {
                        fn_bp_ampere = Some(f3);
                    }
                }
            }

            let (optix_available, optix_version) = match &optix_api {
                Some(ox) => (true, ox.optix_version),
                None => (false, 0),
            };

            Some(Arc::new(Self {
                device_name,
                backend_name,
                driver_info,
                optix_available,
                optix_version,
                ctx,
                module,
                ampere_module,
                fn_dist_field,
                fn_dist_field_tiled,
                fn_dist_field_3d_tiled,
                fn_batch_placement,
                fn_gpu_sa_placer,
                fn_route_cost_3d,
                fn_wavefront_3d,
                fn_rt_bvh_crossings,
                fn_rt_bvh_drc,
                fn_df_ampere,
                fn_df_ampere_tiled,
                fn_bp_ampere,
                cuda: api,
                optix: optix_api,
                dispatch_lock: Mutex::new(()),
                persistent_buffers: Mutex::new(PersistentDeviceBuffers {
                    d_obs: 0,
                    obs_cap_bytes: 0,
                    d_out: 0,
                    out_cap_bytes: 0,
                }),
                compute_busy_ns: Arc::new(AtomicU64::new(0)),
            }))
        }
    }

    unsafe fn ensure_persistent_buffers(&self, in_bytes: usize, out_bytes: usize) -> Option<(CUdeviceptr, CUdeviceptr)> {
        let mut bufs = self.persistent_buffers.lock().ok()?;
        if bufs.obs_cap_bytes < in_bytes || bufs.d_obs == 0 {
            if bufs.d_obs != 0 {
                let _ = (self.cuda.cuMemFree_v2)(bufs.d_obs);
                bufs.d_obs = 0;
                bufs.obs_cap_bytes = 0;
            }
            let mut new_obs: CUdeviceptr = 0;
            if (self.cuda.cuMemAlloc_v2)(&mut new_obs, in_bytes) != CUDA_SUCCESS {
                return None;
            }
            bufs.d_obs = new_obs;
            bufs.obs_cap_bytes = in_bytes;
        }
        if bufs.out_cap_bytes < out_bytes || bufs.d_out == 0 {
            if bufs.d_out != 0 {
                let _ = (self.cuda.cuMemFree_v2)(bufs.d_out);
                bufs.d_out = 0;
                bufs.out_cap_bytes = 0;
            }
            let mut new_out: CUdeviceptr = 0;
            if (self.cuda.cuMemAlloc_v2)(&mut new_out, out_bytes) != CUDA_SUCCESS {
                return None;
            }
            bufs.d_out = new_out;
            bufs.out_cap_bytes = out_bytes;
        }
        Some((bufs.d_obs, bufs.d_out))
    }

    pub fn compute_distance_field(&self, obstacles: &[u32], width: u32, height: u32, radius: i32) -> Option<Vec<f32>> {
        self.compute_distance_field_mode(obstacles, width, height, radius, true)
    }

    pub fn compute_distance_field_mode(
        &self,
        obstacles: &[u32],
        width: u32,
        height: u32,
        radius: i32,
        use_shared_memory: bool,
    ) -> Option<Vec<f32>> {
        let total = (width * height) as usize;
        if obstacles.len() != total || total == 0 {
            return None;
        }

        let _guard = self.dispatch_lock.lock().ok()?;
        unsafe {
            let _ = (self.cuda.cuCtxSetCurrent)(self.ctx);
            let in_bytes = total * std::mem::size_of::<u32>();
            let out_bytes = total * std::mem::size_of::<f32>();

            let (mut d_obs, mut d_out) = self.ensure_persistent_buffers(in_bytes, out_bytes)?;

            let start_t = Instant::now();

            if (self.cuda.cuMemcpyHtoD_v2)(d_obs, obstacles.as_ptr() as *const c_void, in_bytes) != CUDA_SUCCESS {
                return None;
            }

            let block_x: c_uint = 16;
            let block_y: c_uint = 16;
            let grid_x: c_uint = (width + block_x - 1) / block_x;
            let grid_y: c_uint = (height + block_y - 1) / block_y;

            let passes = if total >= 4096 && radius > 12 {
                64
            } else if total >= 4096 {
                8
            } else {
                1
            };

            let smem_bytes = if use_shared_memory {
                let tile_w = block_x as i32 + 2 * radius;
                let tile_h = block_y as i32 + 2 * radius;
                (tile_w * tile_h * 4) as c_uint
            } else {
                0
            };

            let kernel = if use_shared_memory {
                self.fn_dist_field_tiled
            } else {
                self.fn_dist_field
            };

            let mut w_arg = width;
            let mut h_arg = height;
            let mut r_arg = radius;

            let mut args = [
                &mut d_obs as *mut _ as *mut c_void,
                &mut d_out as *mut _ as *mut c_void,
                &mut w_arg as *mut _ as *mut c_void,
                &mut h_arg as *mut _ as *mut c_void,
                &mut r_arg as *mut _ as *mut c_void,
            ];

            for _ in 0..passes {
                if (self.cuda.cuLaunchKernel)(
                    kernel,
                    grid_x,
                    grid_y,
                    1,
                    block_x,
                    block_y,
                    1,
                    smem_bytes,
                    ptr::null_mut(),
                    args.as_mut_ptr(),
                    ptr::null_mut(),
                ) != CUDA_SUCCESS
                {
                    return None;
                }
            }

            let mut result = vec![0.0f32; total];
            if (self.cuda.cuMemcpyDtoH_v2)(result.as_mut_ptr() as *mut c_void, d_out, out_bytes) != CUDA_SUCCESS {
                return None;
            }

            let _ = (self.cuda.cuCtxSynchronize)();
            let elapsed_ns = start_t.elapsed().as_nanos() as u64;
            self.compute_busy_ns.fetch_add(elapsed_ns, Ordering::Relaxed);

            Some(result)
        }
    }

    pub fn compute_distance_field_ampere(
        &self,
        obstacles: &[u32],
        width: u32,
        height: u32,
        radius: i32,
        tiled: bool,
    ) -> Option<Vec<f32>> {
        let kernel = if tiled {
            self.fn_df_ampere_tiled?
        } else {
            self.fn_df_ampere?
        };

        let total = (width * height) as usize;
        if obstacles.len() != total || total == 0 {
            return None;
        }

        let _guard = self.dispatch_lock.lock().ok()?;
        unsafe {
            let _ = (self.cuda.cuCtxSetCurrent)(self.ctx);
            let in_bytes = total * std::mem::size_of::<u32>();
            let out_bytes = total * std::mem::size_of::<f32>();

            let (mut d_obs, mut d_out) = self.ensure_persistent_buffers(in_bytes, out_bytes)?;

            let start_t = Instant::now();

            if (self.cuda.cuMemcpyHtoD_v2)(d_obs, obstacles.as_ptr() as *const c_void, in_bytes) != CUDA_SUCCESS {
                return None;
            }

            let block_x: c_uint = 16;
            let block_y: c_uint = 16;
            let grid_x: c_uint = (width + block_x - 1) / block_x;
            let grid_y: c_uint = (height + block_y - 1) / block_y;

            let passes = if total >= 4096 && radius > 12 {
                64
            } else if total >= 4096 {
                8
            } else {
                1
            };

            let smem_bytes = if tiled {
                let tile_w = block_x as i32 + 2 * radius;
                let tile_h = block_y as i32 + 2 * radius;
                (tile_w * tile_h * 4) as c_uint
            } else {
                0
            };

            let mut w_arg = width;
            let mut h_arg = height;
            let mut r_arg = radius;

            let mut args = [
                &mut d_obs as *mut _ as *mut c_void,
                &mut d_out as *mut _ as *mut c_void,
                &mut w_arg as *mut _ as *mut c_void,
                &mut h_arg as *mut _ as *mut c_void,
                &mut r_arg as *mut _ as *mut c_void,
            ];

            for _ in 0..passes {
                if (self.cuda.cuLaunchKernel)(
                    kernel,
                    grid_x,
                    grid_y,
                    1,
                    block_x,
                    block_y,
                    1,
                    smem_bytes,
                    ptr::null_mut(),
                    args.as_mut_ptr(),
                    ptr::null_mut(),
                ) != CUDA_SUCCESS
                {
                    return None;
                }
            }

            let mut result = vec![0.0f32; total];
            if (self.cuda.cuMemcpyDtoH_v2)(result.as_mut_ptr() as *mut c_void, d_out, out_bytes) != CUDA_SUCCESS {
                return None;
            }

            let _ = (self.cuda.cuCtxSynchronize)();
            let elapsed_ns = start_t.elapsed().as_nanos() as u64;
            self.compute_busy_ns.fetch_add(elapsed_ns, Ordering::Relaxed);

            Some(result)
        }
    }

    pub fn compute_distance_field_layers(
        &self,
        obstacles_flat: &[u32],
        width: u32,
        height: u32,
        num_layers: u32,
        radius: i32,
    ) -> Option<Vec<f32>> {
        self.compute_distance_field_batch_layers(obstacles_flat, width, height, num_layers, radius)
    }

    pub fn compute_distance_field_batch_layers(
        &self,
        obstacles_flat: &[u32],
        width: u32,
        height: u32,
        num_layers: u32,
        radius: i32,
    ) -> Option<Vec<f32>> {
        let layer_cells = (width * height) as usize;
        let total = layer_cells * (num_layers as usize);
        if obstacles_flat.len() != total || total == 0 || num_layers == 0 {
            return None;
        }

        let _guard = self.dispatch_lock.lock().ok()?;
        unsafe {
            let _ = (self.cuda.cuCtxSetCurrent)(self.ctx);
            let in_bytes = total * std::mem::size_of::<u32>();
            let out_bytes = total * std::mem::size_of::<f32>();

            let (mut d_obs, mut d_out) = self.ensure_persistent_buffers(in_bytes, out_bytes)?;

            let start_t = Instant::now();

            if (self.cuda.cuMemcpyHtoD_v2)(d_obs, obstacles_flat.as_ptr() as *const c_void, in_bytes) != CUDA_SUCCESS {
                return None;
            }

            let block_x: c_uint = 16;
            let block_y: c_uint = 16;
            let grid_x: c_uint = (width + block_x - 1) / block_x;
            let grid_y: c_uint = (height + block_y - 1) / block_y;
            let grid_z: c_uint = num_layers;

            let tile_w = block_x as i32 + 2 * radius;
            let tile_h = block_y as i32 + 2 * radius;
            let smem_bytes = (tile_w * tile_h * 4) as c_uint;

            let passes = if layer_cells >= 4096 && radius > 12 {
                16
            } else if layer_cells >= 4096 {
                4
            } else {
                1
            };

            let mut w_arg = width;
            let mut h_arg = height;
            let mut l_arg = num_layers;
            let mut r_arg = radius;

            let mut args = [
                &mut d_obs as *mut _ as *mut c_void,
                &mut d_out as *mut _ as *mut c_void,
                &mut w_arg as *mut _ as *mut c_void,
                &mut h_arg as *mut _ as *mut c_void,
                &mut l_arg as *mut _ as *mut c_void,
                &mut r_arg as *mut _ as *mut c_void,
            ];

            for _ in 0..passes {
                if (self.cuda.cuLaunchKernel)(
                    self.fn_dist_field_3d_tiled,
                    grid_x,
                    grid_y,
                    grid_z,
                    block_x,
                    block_y,
                    1,
                    smem_bytes,
                    ptr::null_mut(),
                    args.as_mut_ptr(),
                    ptr::null_mut(),
                ) != CUDA_SUCCESS
                {
                    return None;
                }
            }

            let mut result = vec![0.0f32; total];
            if (self.cuda.cuMemcpyDtoH_v2)(result.as_mut_ptr() as *mut c_void, d_out, out_bytes) != CUDA_SUCCESS {
                return None;
            }

            let _ = (self.cuda.cuCtxSynchronize)();
            let elapsed_ns = start_t.elapsed().as_nanos() as u64;
            self.compute_busy_ns.fetch_add(elapsed_ns, Ordering::Relaxed);

            Some(result)
        }
    }


    pub fn evaluate_placement_ampere(
        &self,
        boxes: &[[f32; 4]],
        segs: &[[f32; 4]],
        params: &PlacementBatchParams,
    ) -> Option<Vec<f32>> {
        let kernel = self.fn_bp_ampere?;
        let nc = params.num_candidates as usize;
        let nf = params.num_footprints as usize;
        let ns = params.num_segments as usize;
        if nc == 0 || nf == 0 || boxes.len() != nc * nf || segs.len() != nc * ns {
            return None;
        }

        let _guard = self.dispatch_lock.lock().ok()?;
        unsafe {
            let _ = (self.cuda.cuCtxSetCurrent)(self.ctx);
            let boxes_bytes = boxes.len() * std::mem::size_of::<[f32; 4]>();
            let dummy_seg = [[0.0f32; 4]; 1];
            let seg_slice: &[[f32; 4]] = if segs.is_empty() { &dummy_seg } else { segs };
            let segs_bytes = seg_slice.len() * std::mem::size_of::<[f32; 4]>();
            let out_bytes = nc * std::mem::size_of::<f32>();

            let mut d_boxes: CUdeviceptr = 0;
            let mut d_segs: CUdeviceptr = 0;
            let mut d_out: CUdeviceptr = 0;

            if (self.cuda.cuMemAlloc_v2)(&mut d_boxes, boxes_bytes) != CUDA_SUCCESS {
                return None;
            }
            if (self.cuda.cuMemAlloc_v2)(&mut d_segs, segs_bytes) != CUDA_SUCCESS {
                let _ = (self.cuda.cuMemFree_v2)(d_boxes);
                return None;
            }
            if (self.cuda.cuMemAlloc_v2)(&mut d_out, out_bytes) != CUDA_SUCCESS {
                let _ = (self.cuda.cuMemFree_v2)(d_boxes);
                let _ = (self.cuda.cuMemFree_v2)(d_segs);
                return None;
            }

            let start_t = Instant::now();

            if (self.cuda.cuMemcpyHtoD_v2)(d_boxes, boxes.as_ptr() as *const c_void, boxes_bytes) != CUDA_SUCCESS
                || (self.cuda.cuMemcpyHtoD_v2)(d_segs, seg_slice.as_ptr() as *const c_void, segs_bytes) != CUDA_SUCCESS
            {
                let _ = (self.cuda.cuMemFree_v2)(d_boxes);
                let _ = (self.cuda.cuMemFree_v2)(d_segs);
                let _ = (self.cuda.cuMemFree_v2)(d_out);
                return None;
            }

            let block_x: c_uint = 64;
            let grid_x: c_uint = (params.num_candidates + block_x - 1) / block_x;

            let mut params_copy = *params;
            let mut args = [
                &mut d_boxes as *mut _ as *mut c_void,
                &mut d_segs as *mut _ as *mut c_void,
                &mut d_out as *mut _ as *mut c_void,
                &mut params_copy as *mut _ as *mut c_void,
            ];

            if (self.cuda.cuLaunchKernel)(
                kernel,
                grid_x,
                1,
                1,
                block_x,
                1,
                1,
                0,
                ptr::null_mut(),
                args.as_mut_ptr(),
                ptr::null_mut(),
            ) != CUDA_SUCCESS
            {
                let _ = (self.cuda.cuMemFree_v2)(d_boxes);
                let _ = (self.cuda.cuMemFree_v2)(d_segs);
                let _ = (self.cuda.cuMemFree_v2)(d_out);
                return None;
            }

            let mut result = vec![0.0f32; nc];
            if (self.cuda.cuMemcpyDtoH_v2)(result.as_mut_ptr() as *mut c_void, d_out, out_bytes) != CUDA_SUCCESS {
                let _ = (self.cuda.cuMemFree_v2)(d_boxes);
                let _ = (self.cuda.cuMemFree_v2)(d_segs);
                let _ = (self.cuda.cuMemFree_v2)(d_out);
                return None;
            }

            let _ = (self.cuda.cuCtxSynchronize)();
            let elapsed_ns = start_t.elapsed().as_nanos() as u64;
            self.compute_busy_ns.fetch_add(elapsed_ns, Ordering::Relaxed);

            let _ = (self.cuda.cuMemFree_v2)(d_boxes);
            let _ = (self.cuda.cuMemFree_v2)(d_segs);
            let _ = (self.cuda.cuMemFree_v2)(d_out);

            Some(result)
        }
    }

    pub fn evaluate_placement_batch(
        &self,
        boxes: &[[f32; 4]],
        segs: &[[f32; 4]],
        params: &PlacementBatchParams,
    ) -> Option<Vec<f32>> {
        let nc = params.num_candidates as usize;
        let nf = params.num_footprints as usize;
        let ns = params.num_segments as usize;
        if nc == 0 || nf == 0 || boxes.len() != nc * nf || segs.len() != nc * ns {
            return None;
        }

        let _guard = self.dispatch_lock.lock().ok()?;
        unsafe {
            let _ = (self.cuda.cuCtxSetCurrent)(self.ctx);
            let boxes_bytes = boxes.len() * std::mem::size_of::<[f32; 4]>();
            let dummy_seg = [[0.0f32; 4]; 1];
            let seg_slice: &[[f32; 4]] = if segs.is_empty() { &dummy_seg } else { segs };
            let segs_bytes = seg_slice.len() * std::mem::size_of::<[f32; 4]>();
            let out_bytes = nc * std::mem::size_of::<f32>();

            let mut d_boxes: CUdeviceptr = 0;
            let mut d_segs: CUdeviceptr = 0;
            let mut d_out: CUdeviceptr = 0;

            if (self.cuda.cuMemAlloc_v2)(&mut d_boxes, boxes_bytes) != CUDA_SUCCESS {
                return None;
            }
            if (self.cuda.cuMemAlloc_v2)(&mut d_segs, segs_bytes) != CUDA_SUCCESS {
                let _ = (self.cuda.cuMemFree_v2)(d_boxes);
                return None;
            }
            if (self.cuda.cuMemAlloc_v2)(&mut d_out, out_bytes) != CUDA_SUCCESS {
                let _ = (self.cuda.cuMemFree_v2)(d_boxes);
                let _ = (self.cuda.cuMemFree_v2)(d_segs);
                return None;
            }

            let start_t = Instant::now();

            if (self.cuda.cuMemcpyHtoD_v2)(d_boxes, boxes.as_ptr() as *const c_void, boxes_bytes) != CUDA_SUCCESS
                || (self.cuda.cuMemcpyHtoD_v2)(d_segs, seg_slice.as_ptr() as *const c_void, segs_bytes) != CUDA_SUCCESS
            {
                let _ = (self.cuda.cuMemFree_v2)(d_boxes);
                let _ = (self.cuda.cuMemFree_v2)(d_segs);
                let _ = (self.cuda.cuMemFree_v2)(d_out);
                return None;
            }

            let block_x: c_uint = 64;
            let grid_x: c_uint = (params.num_candidates + block_x - 1) / block_x;

            let mut params_copy = *params;
            let mut args = [
                &mut d_boxes as *mut _ as *mut c_void,
                &mut d_segs as *mut _ as *mut c_void,
                &mut d_out as *mut _ as *mut c_void,
                &mut params_copy as *mut _ as *mut c_void,
            ];

            if (self.cuda.cuLaunchKernel)(
                self.fn_batch_placement,
                grid_x,
                1,
                1,
                block_x,
                1,
                1,
                0,
                ptr::null_mut(),
                args.as_mut_ptr(),
                ptr::null_mut(),
            ) != CUDA_SUCCESS
            {
                let _ = (self.cuda.cuMemFree_v2)(d_boxes);
                let _ = (self.cuda.cuMemFree_v2)(d_segs);
                let _ = (self.cuda.cuMemFree_v2)(d_out);
                return None;
            }

            let mut result = vec![0.0f32; nc];
            if (self.cuda.cuMemcpyDtoH_v2)(result.as_mut_ptr() as *mut c_void, d_out, out_bytes) != CUDA_SUCCESS {
                let _ = (self.cuda.cuMemFree_v2)(d_boxes);
                let _ = (self.cuda.cuMemFree_v2)(d_segs);
                let _ = (self.cuda.cuMemFree_v2)(d_out);
                return None;
            }

            let _ = (self.cuda.cuCtxSynchronize)();
            let elapsed_ns = start_t.elapsed().as_nanos() as u64;
            self.compute_busy_ns.fetch_add(elapsed_ns, Ordering::Relaxed);

            let _ = (self.cuda.cuMemFree_v2)(d_boxes);
            let _ = (self.cuda.cuMemFree_v2)(d_segs);
            let _ = (self.cuda.cuMemFree_v2)(d_out);

            Some(result)
        }
    }

    pub fn run_gpu_sa_placer_chains(
        &self,
        fp_descs: &[[f32; 4]],
        edges: &[[f32; 4]],
        init_poses: &[[f32; 4]],
        params: &GpuSaParams,
        num_passes: u32,
    ) -> Option<(Vec<[f32; 4]>, Vec<[f32; 4]>)> {
        let nc = params.num_chains as usize;
        let nf = params.num_footprints as usize;
        let ne = params.num_edges as usize;
        if nc == 0 || nf == 0 || fp_descs.len() != nf || edges.len() != ne * 2 || init_poses.len() != nc * nf {
            return None;
        }

        let _guard = self.dispatch_lock.lock().ok()?;
        unsafe {
            let _ = (self.cuda.cuCtxSetCurrent)(self.ctx);
            let fp_bytes = fp_descs.len() * std::mem::size_of::<[f32; 4]>();
            let dummy_edge = [[0.0f32; 4]; 2];
            let edge_slice: &[[f32; 4]] = if edges.is_empty() { &dummy_edge } else { edges };
            let edges_bytes = edge_slice.len() * std::mem::size_of::<[f32; 4]>();

            let mut full_poses = Vec::with_capacity(nc * nf * 2);
            full_poses.extend_from_slice(init_poses);
            full_poses.extend_from_slice(init_poses);
            let poses_bytes = full_poses.len() * std::mem::size_of::<[f32; 4]>();

            let init_scores = vec![[0.0f32; 4]; nc];
            let scores_bytes = init_scores.len() * std::mem::size_of::<[f32; 4]>();

            let mut d_fp: CUdeviceptr = 0;
            let mut d_edges: CUdeviceptr = 0;
            let mut d_poses: CUdeviceptr = 0;
            let mut d_scores: CUdeviceptr = 0;

            if (self.cuda.cuMemAlloc_v2)(&mut d_fp, fp_bytes) != CUDA_SUCCESS {
                return None;
            }
            if (self.cuda.cuMemAlloc_v2)(&mut d_edges, edges_bytes) != CUDA_SUCCESS {
                let _ = (self.cuda.cuMemFree_v2)(d_fp);
                return None;
            }
            if (self.cuda.cuMemAlloc_v2)(&mut d_poses, poses_bytes) != CUDA_SUCCESS {
                let _ = (self.cuda.cuMemFree_v2)(d_fp);
                let _ = (self.cuda.cuMemFree_v2)(d_edges);
                return None;
            }
            if (self.cuda.cuMemAlloc_v2)(&mut d_scores, scores_bytes) != CUDA_SUCCESS {
                let _ = (self.cuda.cuMemFree_v2)(d_fp);
                let _ = (self.cuda.cuMemFree_v2)(d_edges);
                let _ = (self.cuda.cuMemFree_v2)(d_poses);
                return None;
            }

            let start_t = Instant::now();

            if (self.cuda.cuMemcpyHtoD_v2)(d_fp, fp_descs.as_ptr() as *const c_void, fp_bytes) != CUDA_SUCCESS
                || (self.cuda.cuMemcpyHtoD_v2)(d_edges, edge_slice.as_ptr() as *const c_void, edges_bytes) != CUDA_SUCCESS
                || (self.cuda.cuMemcpyHtoD_v2)(d_poses, full_poses.as_ptr() as *const c_void, poses_bytes) != CUDA_SUCCESS
                || (self.cuda.cuMemcpyHtoD_v2)(d_scores, init_scores.as_ptr() as *const c_void, scores_bytes) != CUDA_SUCCESS
            {
                let _ = (self.cuda.cuMemFree_v2)(d_fp);
                let _ = (self.cuda.cuMemFree_v2)(d_edges);
                let _ = (self.cuda.cuMemFree_v2)(d_poses);
                let _ = (self.cuda.cuMemFree_v2)(d_scores);
                return None;
            }

            let block_x: c_uint = 64;
            let grid_x: c_uint = (params.num_chains + block_x - 1) / block_x;

            let mut params_copy = *params;
            let mut passes_arg = num_passes.max(1);

            let mut args = [
                &mut d_fp as *mut _ as *mut c_void,
                &mut d_edges as *mut _ as *mut c_void,
                &mut d_poses as *mut _ as *mut c_void,
                &mut d_scores as *mut _ as *mut c_void,
                &mut params_copy as *mut _ as *mut c_void,
                &mut passes_arg as *mut _ as *mut c_void,
            ];

            if (self.cuda.cuLaunchKernel)(
                self.fn_gpu_sa_placer,
                grid_x,
                1,
                1,
                block_x,
                1,
                1,
                0,
                ptr::null_mut(),
                args.as_mut_ptr(),
                ptr::null_mut(),
            ) != CUDA_SUCCESS
            {
                let _ = (self.cuda.cuMemFree_v2)(d_fp);
                let _ = (self.cuda.cuMemFree_v2)(d_edges);
                let _ = (self.cuda.cuMemFree_v2)(d_poses);
                let _ = (self.cuda.cuMemFree_v2)(d_scores);
                return None;
            }

            let best_offset = (nc * nf * std::mem::size_of::<[f32; 4]>()) as u64;
            let best_bytes = nc * nf * std::mem::size_of::<[f32; 4]>();
            let mut best_poses = vec![[0.0f32; 4]; nc * nf];
            let mut final_scores = vec![[0.0f32; 4]; nc];

            if (self.cuda.cuMemcpyDtoH_v2)(best_poses.as_mut_ptr() as *mut c_void, d_poses + best_offset, best_bytes) != CUDA_SUCCESS
                || (self.cuda.cuMemcpyDtoH_v2)(final_scores.as_mut_ptr() as *mut c_void, d_scores, scores_bytes) != CUDA_SUCCESS
            {
                let _ = (self.cuda.cuMemFree_v2)(d_fp);
                let _ = (self.cuda.cuMemFree_v2)(d_edges);
                let _ = (self.cuda.cuMemFree_v2)(d_poses);
                let _ = (self.cuda.cuMemFree_v2)(d_scores);
                return None;
            }

            let _ = (self.cuda.cuCtxSynchronize)();
            let elapsed_ns = start_t.elapsed().as_nanos() as u64;
            self.compute_busy_ns.fetch_add(elapsed_ns, Ordering::Relaxed);

            let _ = (self.cuda.cuMemFree_v2)(d_fp);
            let _ = (self.cuda.cuMemFree_v2)(d_edges);
            let _ = (self.cuda.cuMemFree_v2)(d_poses);
            let _ = (self.cuda.cuMemFree_v2)(d_scores);

            Some((best_poses, final_scores))
        }
    }

    pub fn compute_3d_routing_costs(
        &self,
        obstacles: &[u32],
        dist_field: &[f32],
        history_cost: &[f32],
        usage_count: &[u32],
        params: &GpuRouteCostParams,
    ) -> Option<Vec<f32>> {
        let mut batch = self.compute_3d_routing_costs_multi(
            obstacles,
            dist_field,
            history_cost,
            usage_count,
            std::slice::from_ref(params),
        );
        batch.pop().flatten()
    }

    /// Evaluates one or more 3D routing cost maps on this GPU while uploading shared grid buffers only once.
    pub fn compute_3d_routing_costs_multi(
        &self,
        obstacles: &[u32],
        dist_field: &[f32],
        history_cost: &[f32],
        usage_count: &[u32],
        batch_params: &[GpuRouteCostParams],
    ) -> Vec<Option<Vec<f32>>> {
        if batch_params.is_empty() {
            return Vec::new();
        }
        let first = &batch_params[0];
        let total = match (first.width as usize)
            .checked_mul(first.height as usize)
            .and_then(|v| v.checked_mul(first.num_layers as usize))
        {
            Some(t) if t > 0 => t,
            _ => return vec![None; batch_params.len()],
        };
        if obstacles.len() != total
            || dist_field.len() != total
            || history_cost.len() != total
            || usage_count.len() != total
        {
            return vec![None; batch_params.len()];
        }

        let _guard = match self.dispatch_lock.lock() {
            Ok(g) => g,
            Err(_) => return vec![None; batch_params.len()],
        };

        unsafe {
            let _ = (self.cuda.cuCtxSetCurrent)(self.ctx);
            let u32_bytes = total * std::mem::size_of::<u32>();
            let f32_bytes = total * std::mem::size_of::<f32>();

            let mut d_obs: CUdeviceptr = 0;
            let mut d_df: CUdeviceptr = 0;
            let mut d_hist: CUdeviceptr = 0;
            let mut d_usage: CUdeviceptr = 0;
            let mut d_out: CUdeviceptr = 0;

            if (self.cuda.cuMemAlloc_v2)(&mut d_obs, u32_bytes) != CUDA_SUCCESS {
                return vec![None; batch_params.len()];
            }
            if (self.cuda.cuMemAlloc_v2)(&mut d_df, f32_bytes) != CUDA_SUCCESS {
                let _ = (self.cuda.cuMemFree_v2)(d_obs);
                return vec![None; batch_params.len()];
            }
            if (self.cuda.cuMemAlloc_v2)(&mut d_hist, f32_bytes) != CUDA_SUCCESS {
                let _ = (self.cuda.cuMemFree_v2)(d_obs);
                let _ = (self.cuda.cuMemFree_v2)(d_df);
                return vec![None; batch_params.len()];
            }
            if (self.cuda.cuMemAlloc_v2)(&mut d_usage, u32_bytes) != CUDA_SUCCESS {
                let _ = (self.cuda.cuMemFree_v2)(d_obs);
                let _ = (self.cuda.cuMemFree_v2)(d_df);
                let _ = (self.cuda.cuMemFree_v2)(d_hist);
                return vec![None; batch_params.len()];
            }
            if (self.cuda.cuMemAlloc_v2)(&mut d_out, f32_bytes) != CUDA_SUCCESS {
                let _ = (self.cuda.cuMemFree_v2)(d_obs);
                let _ = (self.cuda.cuMemFree_v2)(d_df);
                let _ = (self.cuda.cuMemFree_v2)(d_hist);
                let _ = (self.cuda.cuMemFree_v2)(d_usage);
                return vec![None; batch_params.len()];
            }

            let start_t = Instant::now();

            if (self.cuda.cuMemcpyHtoD_v2)(d_obs, obstacles.as_ptr() as *const c_void, u32_bytes) != CUDA_SUCCESS
                || (self.cuda.cuMemcpyHtoD_v2)(d_df, dist_field.as_ptr() as *const c_void, f32_bytes) != CUDA_SUCCESS
                || (self.cuda.cuMemcpyHtoD_v2)(d_hist, history_cost.as_ptr() as *const c_void, f32_bytes) != CUDA_SUCCESS
                || (self.cuda.cuMemcpyHtoD_v2)(d_usage, usage_count.as_ptr() as *const c_void, u32_bytes) != CUDA_SUCCESS
            {
                let _ = (self.cuda.cuMemFree_v2)(d_obs);
                let _ = (self.cuda.cuMemFree_v2)(d_df);
                let _ = (self.cuda.cuMemFree_v2)(d_hist);
                let _ = (self.cuda.cuMemFree_v2)(d_usage);
                let _ = (self.cuda.cuMemFree_v2)(d_out);
                return vec![None; batch_params.len()];
            }

            let block_x: c_uint = 16;
            let block_y: c_uint = 16;
            let grid_x: c_uint = (first.width + block_x - 1) / block_x;
            let grid_y: c_uint = (first.height + block_y - 1) / block_y;
            let grid_z: c_uint = first.num_layers;

            let mut outputs = Vec::with_capacity(batch_params.len());
            for p in batch_params {
                if p.width != first.width || p.height != first.height || p.num_layers != first.num_layers {
                    outputs.push(None);
                    continue;
                }
                let r = p.clearance_cells.max(0);
                let smem_bytes: c_uint = if r <= 16 {
                    let tile_w = block_x as i32 + 2 * r;
                    let tile_h = block_y as i32 + 2 * r;
                    (tile_w * tile_h * 4) as c_uint
                } else {
                    0
                };

                let mut params_copy = *p;
                let mut args = [
                    &mut d_obs as *mut _ as *mut c_void,
                    &mut d_df as *mut _ as *mut c_void,
                    &mut d_hist as *mut _ as *mut c_void,
                    &mut d_usage as *mut _ as *mut c_void,
                    &mut d_out as *mut _ as *mut c_void,
                    &mut params_copy as *mut _ as *mut c_void,
                ];

                if (self.cuda.cuLaunchKernel)(
                    self.fn_route_cost_3d,
                    grid_x,
                    grid_y,
                    grid_z,
                    block_x,
                    block_y,
                    1,
                    smem_bytes,
                    ptr::null_mut(),
                    args.as_mut_ptr(),
                    ptr::null_mut(),
                ) != CUDA_SUCCESS
                {
                    outputs.push(None);
                    continue;
                }

                let mut res = vec![0.0f32; total];
                if (self.cuda.cuMemcpyDtoH_v2)(res.as_mut_ptr() as *mut c_void, d_out, f32_bytes) == CUDA_SUCCESS {
                    outputs.push(Some(res));
                } else {
                    outputs.push(None);
                }
            }

            let _ = (self.cuda.cuCtxSynchronize)();
            let elapsed_ns = start_t.elapsed().as_nanos() as u64;
            self.compute_busy_ns.fetch_add(elapsed_ns, Ordering::Relaxed);

            let _ = (self.cuda.cuMemFree_v2)(d_obs);
            let _ = (self.cuda.cuMemFree_v2)(d_df);
            let _ = (self.cuda.cuMemFree_v2)(d_hist);
            let _ = (self.cuda.cuMemFree_v2)(d_usage);
            let _ = (self.cuda.cuMemFree_v2)(d_out);

            outputs
        }
    }

    /// Evaluates a batch of 3D routing cost maps concurrently across all available CUDA GPUs.
    pub fn compute_3d_routing_costs_batch(
        engines: &[Arc<Self>],
        obstacles: &[u32],
        dist_field: &[f32],
        history_cost: &[f32],
        usage_count: &[u32],
        batch_params: &[GpuRouteCostParams],
    ) -> Vec<Option<Vec<f32>>> {
        if engines.is_empty() || batch_params.is_empty() {
            return vec![None; batch_params.len()];
        }
        if engines.len() == 1 || batch_params.len() == 1 {
            return engines[0].compute_3d_routing_costs_multi(
                obstacles,
                dist_field,
                history_cost,
                usage_count,
                batch_params,
            );
        }

        let num_engines = engines.len();
        let mut results: Vec<Option<Vec<f32>>> = (0..batch_params.len()).map(|_| None).collect();
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for (eng_idx, engine) in engines.iter().enumerate() {
                let assigned: Vec<(usize, GpuRouteCostParams)> = batch_params
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i % num_engines == eng_idx)
                    .map(|(i, p)| (i, *p))
                    .collect();
                if assigned.is_empty() {
                    continue;
                }
                let eng_ref = Arc::clone(engine);
                handles.push(scope.spawn(move || {
                    let sub_params: Vec<GpuRouteCostParams> = assigned.iter().map(|&(_, p)| p).collect();
                    let sub_out = eng_ref.compute_3d_routing_costs_multi(
                        obstacles,
                        dist_field,
                        history_cost,
                        usage_count,
                        &sub_params,
                    );
                    assigned
                        .into_iter()
                        .map(|(orig_i, _)| orig_i)
                        .zip(sub_out)
                        .collect::<Vec<(usize, Option<Vec<f32>>)>>()
                }));
            }
            for handle in handles {
                if let Ok(sub_res) = handle.join() {
                    for (orig_i, map_opt) in sub_res {
                        results[orig_i] = map_opt;
                    }
                }
            }
        });
        results
    }

    pub fn compute_3d_wavefront(
        &self,
        cell_costs: &[f32],
        seeds: &[u32],
        params: &GpuRouteCostParams,
        via_cost: f32,
        iterations: u32,
    ) -> Option<Vec<f32>> {
        let total = (params.width as usize)
            .checked_mul(params.height as usize)?
            .checked_mul(params.num_layers as usize)?;
        if total == 0 || cell_costs.len() != total || seeds.is_empty() {
            return None;
        }

        let mut init_wf = vec![1.0e20f32; total];
        for &s in seeds {
            let idx = s as usize;
            if idx < total {
                init_wf[idx] = 0.0;
            }
        }
        if iterations == 0 {
            return Some(init_wf);
        }

        let _guard = self.dispatch_lock.lock().ok()?;
        unsafe {
            let _ = (self.cuda.cuCtxSetCurrent)(self.ctx);
            let f32_bytes = total * std::mem::size_of::<f32>();

            let mut d_costs: CUdeviceptr = 0;
            let mut d_wf_a: CUdeviceptr = 0;
            let mut d_wf_b: CUdeviceptr = 0;

            if (self.cuda.cuMemAlloc_v2)(&mut d_costs, f32_bytes) != CUDA_SUCCESS {
                return None;
            }
            if (self.cuda.cuMemAlloc_v2)(&mut d_wf_a, f32_bytes) != CUDA_SUCCESS {
                let _ = (self.cuda.cuMemFree_v2)(d_costs);
                return None;
            }
            if (self.cuda.cuMemAlloc_v2)(&mut d_wf_b, f32_bytes) != CUDA_SUCCESS {
                let _ = (self.cuda.cuMemFree_v2)(d_costs);
                let _ = (self.cuda.cuMemFree_v2)(d_wf_a);
                return None;
            }

            let start_t = Instant::now();

            if (self.cuda.cuMemcpyHtoD_v2)(d_costs, cell_costs.as_ptr() as *const c_void, f32_bytes) != CUDA_SUCCESS
                || (self.cuda.cuMemcpyHtoD_v2)(d_wf_a, init_wf.as_ptr() as *const c_void, f32_bytes) != CUDA_SUCCESS
            {
                let _ = (self.cuda.cuMemFree_v2)(d_costs);
                let _ = (self.cuda.cuMemFree_v2)(d_wf_a);
                let _ = (self.cuda.cuMemFree_v2)(d_wf_b);
                return None;
            }

            let block_x: c_uint = 16;
            let block_y: c_uint = 16;
            let grid_x: c_uint = (params.width + block_x - 1) / block_x;
            let grid_y: c_uint = (params.height + block_y - 1) / block_y;
            let grid_z: c_uint = params.num_layers;

            let mut params_copy = *params;
            let mut via_cost_arg = via_cost;

            for iter in 0..iterations {
                let (mut d_in, mut d_out) = if iter % 2 == 0 {
                    (d_wf_a, d_wf_b)
                } else {
                    (d_wf_b, d_wf_a)
                };
                let mut args = [
                    &mut d_costs as *mut _ as *mut c_void,
                    &mut d_in as *mut _ as *mut c_void,
                    &mut d_out as *mut _ as *mut c_void,
                    &mut params_copy as *mut _ as *mut c_void,
                    &mut via_cost_arg as *mut _ as *mut c_void,
                ];
                if (self.cuda.cuLaunchKernel)(
                    self.fn_wavefront_3d,
                    grid_x,
                    grid_y,
                    grid_z,
                    block_x,
                    block_y,
                    1,
                    0,
                    ptr::null_mut(),
                    args.as_mut_ptr(),
                    ptr::null_mut(),
                ) != CUDA_SUCCESS
                {
                    let _ = (self.cuda.cuMemFree_v2)(d_costs);
                    let _ = (self.cuda.cuMemFree_v2)(d_wf_a);
                    let _ = (self.cuda.cuMemFree_v2)(d_wf_b);
                    return None;
                }
            }

            let final_dev = if iterations % 2 == 1 { d_wf_b } else { d_wf_a };
            let mut result = vec![0.0f32; total];
            if (self.cuda.cuMemcpyDtoH_v2)(result.as_mut_ptr() as *mut c_void, final_dev, f32_bytes) != CUDA_SUCCESS {
                let _ = (self.cuda.cuMemFree_v2)(d_costs);
                let _ = (self.cuda.cuMemFree_v2)(d_wf_a);
                let _ = (self.cuda.cuMemFree_v2)(d_wf_b);
                return None;
            }

            let _ = (self.cuda.cuCtxSynchronize)();
            let elapsed_ns = start_t.elapsed().as_nanos() as u64;
            self.compute_busy_ns.fetch_add(elapsed_ns, Ordering::Relaxed);

            let _ = (self.cuda.cuMemFree_v2)(d_costs);
            let _ = (self.cuda.cuMemFree_v2)(d_wf_a);
            let _ = (self.cuda.cuMemFree_v2)(d_wf_b);

            Some(result)
        }
    }

    pub fn evaluate_segment_crossings_rt_bvh(
        &self,
        segs: &[[f32; 4]],
        bvh_nodes: &[GpuBvhNode],
    ) -> Option<Vec<u32>> {
        if segs.is_empty() || bvh_nodes.is_empty() {
            return None;
        }
        let n = segs.len();
        let segs_bytes = n * std::mem::size_of::<[f32; 4]>();
        let bvh_bytes = bvh_nodes.len() * std::mem::size_of::<GpuBvhNode>();
        let out_bytes = n * std::mem::size_of::<u32>();

        let _guard = self.dispatch_lock.lock().ok()?;
        unsafe {
            let _ = (self.cuda.cuCtxSetCurrent)(self.ctx);
            let mut d_segs: CUdeviceptr = 0;
            let mut d_bvh: CUdeviceptr = 0;
            let mut d_out: CUdeviceptr = 0;

            if (self.cuda.cuMemAlloc_v2)(&mut d_segs, segs_bytes) != CUDA_SUCCESS {
                return None;
            }
            if (self.cuda.cuMemAlloc_v2)(&mut d_bvh, bvh_bytes) != CUDA_SUCCESS {
                let _ = (self.cuda.cuMemFree_v2)(d_segs);
                return None;
            }
            if (self.cuda.cuMemAlloc_v2)(&mut d_out, out_bytes) != CUDA_SUCCESS {
                let _ = (self.cuda.cuMemFree_v2)(d_segs);
                let _ = (self.cuda.cuMemFree_v2)(d_bvh);
                return None;
            }

            let start_t = Instant::now();

            if (self.cuda.cuMemcpyHtoD_v2)(d_segs, segs.as_ptr() as *const c_void, segs_bytes) != CUDA_SUCCESS
                || (self.cuda.cuMemcpyHtoD_v2)(d_bvh, bvh_nodes.as_ptr() as *const c_void, bvh_bytes) != CUDA_SUCCESS
            {
                let _ = (self.cuda.cuMemFree_v2)(d_segs);
                let _ = (self.cuda.cuMemFree_v2)(d_bvh);
                let _ = (self.cuda.cuMemFree_v2)(d_out);
                return None;
            }

            let block_x: c_uint = 64;
            let grid_x: c_uint = (n as c_uint + block_x - 1) / block_x;
            let mut n_arg = n as c_uint;

            let mut args = [
                &mut d_segs as *mut _ as *mut c_void,
                &mut d_bvh as *mut _ as *mut c_void,
                &mut d_out as *mut _ as *mut c_void,
                &mut n_arg as *mut _ as *mut c_void,
            ];

            if (self.cuda.cuLaunchKernel)(
                self.fn_rt_bvh_crossings,
                grid_x,
                1,
                1,
                block_x,
                1,
                1,
                0,
                ptr::null_mut(),
                args.as_mut_ptr(),
                ptr::null_mut(),
            ) != CUDA_SUCCESS
            {
                let _ = (self.cuda.cuMemFree_v2)(d_segs);
                let _ = (self.cuda.cuMemFree_v2)(d_bvh);
                let _ = (self.cuda.cuMemFree_v2)(d_out);
                return None;
            }

            let mut result = vec![0u32; n];
            if (self.cuda.cuMemcpyDtoH_v2)(result.as_mut_ptr() as *mut c_void, d_out, out_bytes) != CUDA_SUCCESS {
                let _ = (self.cuda.cuMemFree_v2)(d_segs);
                let _ = (self.cuda.cuMemFree_v2)(d_bvh);
                let _ = (self.cuda.cuMemFree_v2)(d_out);
                return None;
            }

            let _ = (self.cuda.cuCtxSynchronize)();
            let elapsed_ns = start_t.elapsed().as_nanos() as u64;
            self.compute_busy_ns.fetch_add(elapsed_ns, Ordering::Relaxed);

            let _ = (self.cuda.cuMemFree_v2)(d_segs);
            let _ = (self.cuda.cuMemFree_v2)(d_bvh);
            let _ = (self.cuda.cuMemFree_v2)(d_out);

            Some(result)
        }
    }

    pub fn evaluate_drc_rt_bvh(
        &self,
        bvh_nodes: &[GpuBvhNode],
        queries: &[GpuDrcRayQuery],
    ) -> Option<Vec<u32>> {
        if bvh_nodes.is_empty() || queries.is_empty() {
            return None;
        }
        let nq = queries.len();
        let bvh_bytes = bvh_nodes.len() * std::mem::size_of::<GpuBvhNode>();
        let queries_bytes = nq * std::mem::size_of::<GpuDrcRayQuery>();
        let out_bytes = nq * std::mem::size_of::<u32>();

        let _guard = self.dispatch_lock.lock().ok()?;
        unsafe {
            let _ = (self.cuda.cuCtxSetCurrent)(self.ctx);
            let mut d_bvh: CUdeviceptr = 0;
            let mut d_queries: CUdeviceptr = 0;
            let mut d_out: CUdeviceptr = 0;

            if (self.cuda.cuMemAlloc_v2)(&mut d_bvh, bvh_bytes) != CUDA_SUCCESS {
                return None;
            }
            if (self.cuda.cuMemAlloc_v2)(&mut d_queries, queries_bytes) != CUDA_SUCCESS {
                let _ = (self.cuda.cuMemFree_v2)(d_bvh);
                return None;
            }
            if (self.cuda.cuMemAlloc_v2)(&mut d_out, out_bytes) != CUDA_SUCCESS {
                let _ = (self.cuda.cuMemFree_v2)(d_bvh);
                let _ = (self.cuda.cuMemFree_v2)(d_queries);
                return None;
            }

            let start_t = Instant::now();

            if (self.cuda.cuMemcpyHtoD_v2)(d_bvh, bvh_nodes.as_ptr() as *const c_void, bvh_bytes) != CUDA_SUCCESS
                || (self.cuda.cuMemcpyHtoD_v2)(d_queries, queries.as_ptr() as *const c_void, queries_bytes) != CUDA_SUCCESS
            {
                let _ = (self.cuda.cuMemFree_v2)(d_bvh);
                let _ = (self.cuda.cuMemFree_v2)(d_queries);
                let _ = (self.cuda.cuMemFree_v2)(d_out);
                return None;
            }

            let block_x: c_uint = 64;
            let grid_x: c_uint = (nq as c_uint + block_x - 1) / block_x;
            let mut nq_arg = nq as c_uint;

            let mut args = [
                &mut d_bvh as *mut _ as *mut c_void,
                &mut d_queries as *mut _ as *mut c_void,
                &mut d_out as *mut _ as *mut c_void,
                &mut nq_arg as *mut _ as *mut c_void,
            ];

            if (self.cuda.cuLaunchKernel)(
                self.fn_rt_bvh_drc,
                grid_x,
                1,
                1,
                block_x,
                1,
                1,
                0,
                ptr::null_mut(),
                args.as_mut_ptr(),
                ptr::null_mut(),
            ) != CUDA_SUCCESS
            {
                let _ = (self.cuda.cuMemFree_v2)(d_bvh);
                let _ = (self.cuda.cuMemFree_v2)(d_queries);
                let _ = (self.cuda.cuMemFree_v2)(d_out);
                return None;
            }

            let mut result = vec![0u32; nq];
            if (self.cuda.cuMemcpyDtoH_v2)(result.as_mut_ptr() as *mut c_void, d_out, out_bytes) != CUDA_SUCCESS {
                let _ = (self.cuda.cuMemFree_v2)(d_bvh);
                let _ = (self.cuda.cuMemFree_v2)(d_queries);
                let _ = (self.cuda.cuMemFree_v2)(d_out);
                return None;
            }

            let _ = (self.cuda.cuCtxSynchronize)();
            let elapsed_ns = start_t.elapsed().as_nanos() as u64;
            self.compute_busy_ns.fetch_add(elapsed_ns, Ordering::Relaxed);

            let _ = (self.cuda.cuMemFree_v2)(d_bvh);
            let _ = (self.cuda.cuMemFree_v2)(d_queries);
            let _ = (self.cuda.cuMemFree_v2)(d_out);

            Some(result)
        }
    }

    pub fn benchmark_rt_bvh_vs_simt(&self) -> Option<RtBvhBenchmarkResult> {
        let num_obs = 512usize;
        let num_queries = 256usize;
        let mut obstacles = Vec::with_capacity(num_obs);
        for i in 0..num_obs {
            let x = (i % 32) as f32 * 3.0;
            let y = (i / 32) as f32 * 3.0;
            let layer_mask = 1u32 << (i % 4);
            let net_id = ((i % 16) + 1) as u32;
            obstacles.push(([x, y, x + 0.8, y + 0.8], layer_mask, net_id));
        }
        let bvh_nodes = build_gpu_lbvh(&obstacles);

        let mut queries = Vec::with_capacity(num_queries);
        let mut segs = Vec::with_capacity(num_queries);
        for i in 0..num_queries {
            let x1 = ((i * 7) % 90) as f32;
            let y1 = ((i * 13) % 45) as f32;
            let x2 = x1 + 6.0;
            let y2 = y1 + 4.0;
            queries.push(GpuDrcRayQuery {
                x1,
                y1,
                x2,
                y2,
                clearance: 0.25,
                layer_mask: 1u32 << (i % 4),
                net_id: ((i % 16) + 1) as u32,
                _pad: 0,
            });
            segs.push([x1, y1, x2, y2]);
        }
        let seg_bvh = build_gpu_lbvh_from_segments(&segs);

        let t0 = Instant::now();
        let _ = self.evaluate_drc_rt_bvh(&bvh_nodes, &queries)?;
        let bvh_drc_us = t0.elapsed().as_secs_f64() * 1e6;

        let t1 = Instant::now();
        let _ = evaluate_drc_brute_force_cpu(&obstacles, &queries);
        let simt_drc_us = t1.elapsed().as_secs_f64() * 1e6;

        let t2 = Instant::now();
        let _ = self.evaluate_segment_crossings_rt_bvh(&segs, &seg_bvh)?;
        let bvh_crossings_us = t2.elapsed().as_secs_f64() * 1e6;

        let dummy_boxes = vec![[0.0f32; 4]; 1];
        let bp_params = PlacementBatchParams {
            num_candidates: 1,
            num_footprints: 1,
            num_segments: segs.len() as u32,
            _pad: 0,
            board_min_x: 0.0,
            board_min_y: 0.0,
            board_max_x: 100.0,
            board_max_y: 100.0,
            wirelength_weight: 1.0,
            crossing_weight: 10.0,
            overlap_penalty: 100.0,
            boundary_penalty: 100.0,
        };
        let t3 = Instant::now();
        let _ = self.evaluate_placement_batch(&dummy_boxes, &segs, &bp_params)?;
        let simt_crossings_us = t3.elapsed().as_secs_f64() * 1e6;

        Some(RtBvhBenchmarkResult {
            bvh_drc_us,
            simt_drc_us,
            bvh_crossings_us,
            simt_crossings_us,
            drc_speedup: simt_drc_us / bvh_drc_us.max(0.01),
            crossings_speedup: simt_crossings_us / bvh_crossings_us.max(0.01),
            rt_hardware_available: self.optix_available,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_engine_init_and_kernels() {
        let cuda = match CudaEngine::new() {
            Some(c) => c,
            None => {
                eprintln!("CUDA not available on this host or libcuda.so not found, skipping test");
                return;
            }
        };

        println!("Initialized CUDA Engine: {} ({})", cuda.device_name, cuda.driver_info);

        // 1. Distance field test
        let mut obs = vec![0u32; 16 * 16];
        obs[8 * 16 + 8] = 1;
        let df_tiled = cuda.compute_distance_field(&obs, 16, 16, 10).expect("Distance field tiled");
        let df_direct = cuda.compute_distance_field_mode(&obs, 16, 16, 10, false).expect("Distance field direct");

        assert_eq!(df_tiled[8 * 16 + 8], 0.0);
        assert_eq!(df_direct[8 * 16 + 8], 0.0);
        assert!((df_tiled[8 * 16 + 11] - 3.0).abs() < 1e-3);
        assert!((df_direct[8 * 16 + 11] - 3.0).abs() < 1e-3);

        // 2. Batch placement test
        let params = PlacementBatchParams {
            num_candidates: 2,
            num_footprints: 2,
            num_segments: 1,
            _pad: 0,
            board_min_x: 0.0,
            board_min_y: 0.0,
            board_max_x: 100.0,
            board_max_y: 100.0,
            wirelength_weight: 1.0,
            crossing_weight: 10.0,
            overlap_penalty: 100.0,
            boundary_penalty: 50.0,
        };
        let boxes = vec![
            [10.0, 10.0, 20.0, 20.0],
            [15.0, 15.0, 25.0, 25.0],
            [10.0, 10.0, 20.0, 20.0],
            [40.0, 40.0, 50.0, 50.0],
        ];
        let segs = vec![
            [10.0, 10.0, 40.0, 40.0],
            [10.0, 10.0, 40.0, 40.0],
        ];
        let costs = cuda.evaluate_placement_batch(&boxes, &segs, &params).expect("Batch eval");
        assert_eq!(costs.len(), 2);
        assert!(costs[0] > costs[1], "Overlapping candidate must have higher cost than non-overlapping");
    }

    #[test]
    fn test_cuda_vs_vulkan_exact_equivalence() {
        let cuda = match CudaEngine::new() {
            Some(c) => c,
            None => return,
        };
        let vk = match crate::vulkan::VulkanEngine::new() {
            Some(v) => v,
            None => return,
        };

        // 1. Distance field equivalence (16x16 with obstacle at 4,4)
        let mut obs = vec![0u32; 16 * 16];
        obs[4 * 16 + 4] = 1;
        let df_vk = vk.compute_distance_field(&obs, 16, 16, 10).expect("VK df");
        let df_cu = cuda.compute_distance_field(&obs, 16, 16, 10).expect("CUDA df");
        assert_eq!(df_vk.len(), df_cu.len());
        for i in 0..df_vk.len() {
            assert!(
                (df_vk[i] - df_cu[i]).abs() < 1e-3,
                "Mismatch at {}: vk={}, cu={}",
                i,
                df_vk[i],
                df_cu[i]
            );
        }

        // 2. Batch placement eval equivalence
        let params = PlacementBatchParams {
            num_candidates: 2,
            num_footprints: 2,
            num_segments: 2,
            _pad: 0,
            board_min_x: 0.0,
            board_min_y: 0.0,
            board_max_x: 20.0,
            board_max_y: 20.0,
            wirelength_weight: 1.0,
            crossing_weight: 200.0,
            overlap_penalty: 25000.0,
            boundary_penalty: 50000.0,
        };
        let boxes = vec![
            [2.0, 2.0, 4.0, 4.0],
            [10.0, 10.0, 12.0, 12.0],
            [-1.0, 2.0, 4.0, 4.0],
            [3.0, 3.0, 5.0, 5.0],
        ];
        let segs = vec![
            [1.0, 1.0, 5.0, 1.0],
            [1.0, 3.0, 5.0, 3.0],
            [1.0, 1.0, 5.0, 5.0],
            [1.0, 5.0, 5.0, 1.0],
        ];
        let vk_scores = vk.evaluate_placement_batch(&boxes, &segs, &params).expect("VK scores");
        let cu_scores = cuda.evaluate_placement_batch(&boxes, &segs, &params).expect("CU scores");
        assert_eq!(vk_scores.len(), cu_scores.len());
        for i in 0..vk_scores.len() {
            assert!(
                (vk_scores[i] - cu_scores[i]).abs() < 1e-2,
                "Score mismatch at {}: vk={}, cu={}",
                i,
                vk_scores[i],
                cu_scores[i]
            );
        }
    }

    #[test]
    fn test_ampere_handcrafted_kernels_equivalence() {
        let cuda = match CudaEngine::new() {
            Some(c) => c,
            None => return,
        };

        if cuda.fn_df_ampere.is_none() {
            eprintln!("Ampere kernels not supported on this device/driver, skipping");
            return;
        }

        // 1. Distance field direct
        let mut obs = vec![0u32; 16 * 16];
        obs[4 * 16 + 4] = 1;
        let df_nvcc = cuda.compute_distance_field_mode(&obs, 16, 16, 10, false).expect("NVCC df");
        let df_amp_dir = cuda.compute_distance_field_ampere(&obs, 16, 16, 10, false).expect("Ampere dir df");
        let df_amp_tile = cuda.compute_distance_field_ampere(&obs, 16, 16, 10, true).expect("Ampere tiled df");

        assert_eq!(df_nvcc.len(), df_amp_dir.len());
        assert_eq!(df_nvcc.len(), df_amp_tile.len());
        for i in 0..df_nvcc.len() {
            assert!(
                (df_nvcc[i] - df_amp_dir[i]).abs() < 1e-3,
                "Mismatch at {}: nvcc={}, amp_dir={}",
                i,
                df_nvcc[i],
                df_amp_dir[i]
            );
            assert!(
                (df_nvcc[i] - df_amp_tile[i]).abs() < 1e-3,
                "Mismatch at {}: nvcc={}, amp_tile={}",
                i,
                df_nvcc[i],
                df_amp_tile[i]
            );
        }

        // 2. Batch placement eval
        let params = PlacementBatchParams {
            num_candidates: 2,
            num_footprints: 2,
            num_segments: 2,
            _pad: 0,
            board_min_x: 0.0,
            board_min_y: 0.0,
            board_max_x: 20.0,
            board_max_y: 20.0,
            wirelength_weight: 1.0,
            crossing_weight: 200.0,
            overlap_penalty: 25000.0,
            boundary_penalty: 50000.0,
        };
        let boxes = vec![
            [2.0, 2.0, 4.0, 4.0],
            [10.0, 10.0, 12.0, 12.0],
            [-1.0, 2.0, 4.0, 4.0],
            [3.0, 3.0, 5.0, 5.0],
        ];
        let segs = vec![
            [1.0, 1.0, 5.0, 1.0],
            [1.0, 3.0, 5.0, 3.0],
            [1.0, 1.0, 5.0, 5.0],
            [1.0, 5.0, 5.0, 1.0],
        ];
        let cu_scores = cuda.evaluate_placement_batch(&boxes, &segs, &params).expect("CU scores");
        let amp_scores = cuda.evaluate_placement_ampere(&boxes, &segs, &params).expect("Ampere scores");
        assert_eq!(cu_scores.len(), amp_scores.len());
        for i in 0..cu_scores.len() {
            assert!(
                (cu_scores[i] - amp_scores[i]).abs() < 1e-2,
                "Score mismatch at {}: cu={}, amp={}",
                i,
                cu_scores[i],
                amp_scores[i]
            );
        }
    }

    #[test]
    fn test_cuda_all_devices() {
        let devs = CudaEngine::all_devices();
        if devs.is_empty() {
            eprintln!("No CUDA devices available on this host");
            return;
        }
        println!("Detected {} CUDA device(s):", devs.len());
        for (i, dev) in devs.iter().enumerate() {
            println!("  [{}] {} ({})", i, dev.device_name, dev.driver_info);
            // Verify each device can compute distance fields independently
            let mut obs = vec![0u32; 16 * 16];
            obs[4 * 16 + 4] = 1;
            let df = dev.compute_distance_field(&obs, 16, 16, 10).expect("Distance field output");
            assert_eq!(df[4 * 16 + 4], 0.0);
            assert!((df[4 * 16 + 8] - 4.0).abs() < 1e-3);
        }
        assert!(!devs.is_empty());
    }

    #[test]
    fn test_cuda_persistent_vram_buffer_reuse() {
        let cuda = match CudaEngine::new() {
            Some(c) => c,
            None => return,
        };

        // 1. First dispatch (32x32) allocates initial persistent VRAM buffers
        let mut obs_32 = vec![0u32; 32 * 32];
        obs_32[10 * 32 + 10] = 1;
        let df_32 = cuda.compute_distance_field(&obs_32, 32, 32, 12).expect("32x32 df");
        assert_eq!(df_32[10 * 32 + 10], 0.0);
        assert!((df_32[10 * 32 + 15] - 5.0).abs() < 1e-4);

        let (ptr_32_obs, cap_32_obs, ptr_32_out, cap_32_out) = {
            let b = cuda.persistent_buffers.lock().unwrap();
            (b.d_obs, b.obs_cap_bytes, b.d_out, b.out_cap_bytes)
        };
        assert_ne!(ptr_32_obs, 0);
        assert_ne!(ptr_32_out, 0);
        assert_eq!(cap_32_obs, 32 * 32 * 4);
        assert_eq!(cap_32_out, 32 * 32 * 4);

        // 2. Smaller dispatch (16x16) must reuse the exact same VRAM pointers and capacity
        let mut obs_16 = vec![0u32; 16 * 16];
        obs_16[4 * 16 + 4] = 1;
        let df_16 = cuda.compute_distance_field(&obs_16, 16, 16, 8).expect("16x16 df");
        assert_eq!(df_16[4 * 16 + 4], 0.0);
        assert!((df_16[4 * 16 + 7] - 3.0).abs() < 1e-4);

        {
            let b = cuda.persistent_buffers.lock().unwrap();
            assert_eq!(b.d_obs, ptr_32_obs, "Persistent d_obs must be reused for smaller grid");
            assert_eq!(b.d_out, ptr_32_out, "Persistent d_out must be reused for smaller grid");
            assert_eq!(b.obs_cap_bytes, cap_32_obs);
            assert_eq!(b.out_cap_bytes, cap_32_out);
        }

        // 3. Larger dispatch (64x64) grows capacity cleanly and subsequent 64x64 calls reuse it
        let mut obs_64 = vec![0u32; 64 * 64];
        obs_64[20 * 64 + 20] = 1;
        let df_64_a = cuda.compute_distance_field(&obs_64, 64, 64, 12).expect("64x64 df A");
        let (ptr_64_obs, cap_64_obs, ptr_64_out, cap_64_out) = {
            let b = cuda.persistent_buffers.lock().unwrap();
            (b.d_obs, b.obs_cap_bytes, b.d_out, b.out_cap_bytes)
        };
        assert_eq!(cap_64_obs, 64 * 64 * 4);
        assert_eq!(cap_64_out, 64 * 64 * 4);

        let df_64_b = cuda.compute_distance_field(&obs_64, 64, 64, 12).expect("64x64 df B");
        {
            let b = cuda.persistent_buffers.lock().unwrap();
            assert_eq!(b.d_obs, ptr_64_obs, "Persistent d_obs must be reused on equal size");
            assert_eq!(b.d_out, ptr_64_out, "Persistent d_out must be reused on equal size");
        }
        assert_eq!(df_64_a, df_64_b);
    }

    #[test]
    fn test_cuda_3d_multi_layer_batch_distance_field() {
        let cuda = match CudaEngine::new() {
            Some(c) => c,
            None => return,
        };

        let w = 24u32;
        let h = 20u32;
        let num_layers = 4u32;
        let radius = 12i32;
        let layer_cells = (w * h) as usize;

        let mut obstacles_flat = vec![0u32; layer_cells * (num_layers as usize)];
        // Place distinct obstacle coordinates on each of the 4 layers
        let layer_coords = [(3usize, 4usize), (10, 8), (18, 15), (6, 12)];
        for (layer, &(ox, oy)) in layer_coords.iter().enumerate() {
            obstacles_flat[layer * layer_cells + oy * (w as usize) + ox] = 1;
            // Add a second obstacle on each layer
            let ox2 = (ox + 5) % (w as usize);
            let oy2 = (oy + 3) % (h as usize);
            obstacles_flat[layer * layer_cells + oy2 * (w as usize) + ox2] = 1;
        }

        let batch_df = cuda
            .compute_distance_field_layers(&obstacles_flat, w, h, num_layers, radius)
            .expect("3D batched multi-layer distance field");
        assert_eq!(batch_df.len(), layer_cells * (num_layers as usize));

        for layer in 0..(num_layers as usize) {
            let offset = layer * layer_cells;
            let single_df = cuda
                .compute_distance_field(&obstacles_flat[offset..offset + layer_cells], w, h, radius)
                .expect("Single-layer distance field");
            for i in 0..layer_cells {
                let diff = (batch_df[offset + i] - single_df[i]).abs();
                assert!(
                    diff < 1e-4,
                    "Layer {} cell {} mismatch: batch={} single={} (diff={})",
                    layer,
                    i,
                    batch_df[offset + i],
                    single_df[i],
                    diff
                );
            }
        }
    }

    #[test]
    fn test_gpu_3d_routing_cost_kernel_clearance_and_congestion() {
        let vk = crate::vulkan::VulkanEngine::new().expect("VulkanEngine must initialize");
        let w = 16u32;
        let h = 16u32;
        let nl = 2u32;
        let total = (w * h * nl) as usize;
        let layer_stride = (w * h) as usize;

        let mut obstacles = vec![0u32; total];
        let mut dist_field = vec![2.0f32; total];
        let mut history_cost = vec![0.0f32; total];
        let mut usage_count = vec![0u32; total];

        // Place foreign obstacle (net 5) at (6, 6, 0) and same-net obstacle (net 2) at (12, 12, 0)
        obstacles[6 * (w as usize) + 6] = 5;
        obstacles[12 * (w as usize) + 12] = 2;

        // Add congestion and history at (2, 2, 1)
        let idx_cong = layer_stride + 2 * (w as usize) + 2;
        usage_count[idx_cong] = 3;
        history_cost[idx_cong] = 4.5;
        dist_field[idx_cong] = 0.5;

        let params = GpuRouteCostParams {
            width: w,
            height: h,
            num_layers: nl,
            net_id: 2,
            clearance_cells: 2,
            present_congestion_factor: 1.5,
            obstacle_clearance_weight: 0.2,
            plane_layer_mask: 0,
        };

        let costs = vk
            .compute_3d_routing_costs(&obstacles, &dist_field, &history_cost, &usage_count, &params)
            .expect("3D routing cost kernel output");
        assert_eq!(costs.len(), total);

        // Cells within radius 2 of foreign obstacle (6, 6, 0) must be blocked (-1.0)
        assert_eq!(costs[6 * (w as usize) + 6], -1.0);
        assert_eq!(costs[6 * (w as usize) + 8], -1.0); // dx=2, dy=0 -> d^2=4 <= 4
        assert_eq!(costs[7 * (w as usize) + 7], -1.0); // dx=1, dy=1 -> d^2=2 <= 4
        // Cell at (6, 9, 0) has dx=0, dy=3 -> d^2=9 > 4 -> clear!
        assert!(costs[9 * (w as usize) + 6] > 0.0);

        // Same-net obstacle at (12, 12, 0) must NOT block net 2
        assert!(costs[12 * (w as usize) + 12] > 0.0);
        assert!(costs[12 * (w as usize) + 13] > 0.0);

        // Verify exact traversal cost at (2, 2, 1): 1.0 + 3*1.5 + 4.5 + 0.2/0.5 = 10.4
        assert!((costs[idx_cong] - 10.4).abs() < 1e-4, "got {}", costs[idx_cong]);
    }

    #[test]
    fn test_gpu_3d_wavefront_shortest_path_potential() {
        let vk = crate::vulkan::VulkanEngine::new().expect("VulkanEngine must initialize");
        let w = 12u32;
        let h = 12u32;
        let nl = 2u32;
        let total = (w * h * nl) as usize;
        let stride = (w * h) as usize;

        // Uniform unit cost on both layers, except a full vertical wall at x=5 on layer 0
        let mut cell_costs = vec![1.0f32; total];
        for y in 0..h as usize {
            cell_costs[y * (w as usize) + 5] = -1.0;
        }

        let params = GpuRouteCostParams {
            width: w,
            height: h,
            num_layers: nl,
            net_id: 1,
            clearance_cells: 0,
            present_congestion_factor: 1.0,
            obstacle_clearance_weight: 0.0,
            plane_layer_mask: 0,
        };

        // Seed at (2, 2, 0)
        let seed_idx = (2 * w + 2) as u32;
        let wf = vk
            .compute_3d_wavefront(&cell_costs, &[seed_idx], &params, 2.5, 24)
            .expect("3D wavefront output");

        // Seed itself has 0 potential
        assert_eq!(wf[seed_idx as usize], 0.0);
        // Orthogonal neighbor (3, 2, 0) has potential 1.0
        assert!((wf[(2 * w + 3) as usize] - 1.0).abs() < 1e-4);
        // Diagonal neighbor (3, 3, 0) has potential sqrt(2)
        assert!((wf[(3 * w + 3) as usize] - 1.41421356).abs() < 1e-4);
        // Via transition directly above seed at (2, 2, 1) has potential via_cost = 2.5
        assert!((wf[stride + (2 * w + 2) as usize] - 2.5).abs() < 1e-4);
        // Point (8, 2, 0) is behind the wall on layer 0, so wavefront must hop to layer 1 (+2.5),
        // travel 6 orthogonal steps from x=2 to x=8 (+6.0), and hop back to layer 0 (+2.5) = 11.0
        let target_behind_wall = (2 * w + 8) as usize;
        assert!(
            (wf[target_behind_wall] - 11.0).abs() < 1e-3,
            "Expected 11.0 via layer transition around wall, got {}",
            wf[target_behind_wall]
        );
    }

    #[test]
    fn test_cuda_vs_vulkan_3d_routing_cost_equivalence() {
        let cuda = match CudaEngine::new() {
            Some(c) => c,
            None => return,
        };
        let vk = match crate::vulkan::VulkanEngine::new_vulkan_only() {
            Some(v) => v,
            None => return,
        };

        let w = 20u32;
        let h = 18u32;
        let nl = 4u32;
        let total = (w * h * nl) as usize;
        let mut obstacles = vec![0u32; total];
        let mut dist_field = vec![1.5f32; total];
        let mut history_cost = vec![0.0f32; total];
        let mut usage_count = vec![0u32; total];

        for i in 0..total {
            if i % 37 == 0 {
                obstacles[i] = ((i % 5) + 1) as u32;
            }
            dist_field[i] = 0.25 + ((i % 11) as f32) * 0.2;
            history_cost[i] = ((i % 7) as f32) * 0.35;
            usage_count[i] = (i % 4) as u32;
        }

        let params = GpuRouteCostParams {
            width: w,
            height: h,
            num_layers: nl,
            net_id: 2,
            clearance_cells: 3,
            present_congestion_factor: 1.25,
            obstacle_clearance_weight: 0.4,
            plane_layer_mask: 0b0100,
        };

        let cu_costs = cuda
            .compute_3d_routing_costs(&obstacles, &dist_field, &history_cost, &usage_count, &params)
            .expect("CUDA 3D routing costs");
        let vk_costs = vk
            .compute_3d_routing_costs_vulkan_wgsl(&obstacles, &dist_field, &history_cost, &usage_count, &params)
            .expect("Vulkan WGSL 3D routing costs");

        assert_eq!(cu_costs.len(), vk_costs.len());
        for i in 0..total {
            assert!(
                (cu_costs[i] - vk_costs[i]).abs() < 1e-4,
                "3D cost mismatch at {}: cu={}, vk={}",
                i,
                cu_costs[i],
                vk_costs[i]
            );
        }

        // Also verify 3D wavefront equivalence between CUDA PTX and Vulkan WGSL
        let seed = (5 * w + 5) as u32;
        let mut wf_costs = cu_costs.clone();
        wf_costs[seed as usize] = 1.0;
        let cu_wf = cuda
            .compute_3d_wavefront(&wf_costs, &[seed], &params, 2.0, 12)
            .expect("CUDA 3D wavefront");
        let vk_wf = vk
            .compute_3d_wavefront_vulkan_wgsl(&wf_costs, &[seed], &params, 2.0, 12)
            .expect("Vulkan WGSL 3D wavefront");
        assert_eq!(cu_wf.len(), vk_wf.len());
        for i in 0..total {
            let diff = (cu_wf[i] - vk_wf[i]).abs();
            assert!(
                diff < 1e-3 || (cu_wf[i] >= 1.0e19 && vk_wf[i] >= 1.0e19),
                "3D wavefront mismatch at {}: cu={}, vk={}",
                i,
                cu_wf[i],
                vk_wf[i]
            );
        }
    }

    #[test]
    fn test_optix_driver_and_rt_capability_detection() {
        let devs = CudaEngine::all_devices();
        if devs.is_empty() {
            return;
        }
        let optix = OptixDriverApi::load().expect("libnvoptix.so.1 must load on NVIDIA host");
        assert!(optix.abi_version >= 22);
        assert!(optix.optix_version >= 70000);

        for dev in &devs {
            println!(
                "GPU: {} | OptiX available: {} (version={}, ABI={})",
                dev.device_name, dev.optix_available, dev.optix_version, optix.abi_version
            );
            assert!(dev.optix_available, "OptiX must be detected on {}", dev.device_name);
            assert!(dev.optix_version >= 70000);
            let bench = dev.benchmark_rt_bvh_vs_simt().expect("RT BVH micro-benchmark");
            assert!(bench.rt_hardware_available);
            assert!(bench.bvh_drc_us > 0.0);
            assert!(bench.bvh_crossings_us > 0.0);
        }
    }

    #[test]
    fn test_rt_bvh_line_of_sight_drc_equivalence() {
        let cuda = match CudaEngine::new() {
            Some(c) => c,
            None => return,
        };
        let vk = match crate::vulkan::VulkanEngine::new_vulkan_only() {
            Some(v) => v,
            None => return,
        };

        let num_obs = 520usize;
        let num_queries = 256usize;
        let mut obstacles = Vec::with_capacity(num_obs);
        for i in 0..num_obs {
            let x = ((i * 17) % 100) as f32 + (i % 5) as f32 * 0.25;
            let y = ((i * 31) % 80) as f32 + (i % 3) as f32 * 0.35;
            let w = 0.4 + (i % 4) as f32 * 0.3;
            let h = 0.4 + ((i + 1) % 4) as f32 * 0.3;
            let layer_mask = 1u32 << (i % 4);
            let net_id = ((i % 12) + 1) as u32;
            obstacles.push(([x, y, x + w, y + h], layer_mask, net_id));
        }

        let bvh_nodes = build_gpu_lbvh(&obstacles);
        assert_eq!(bvh_nodes.len(), num_obs * 2 - 1);
        assert_eq!(std::mem::size_of::<GpuBvhNode>(), 32);

        let mut queries = Vec::with_capacity(num_queries);
        for q in 0..num_queries {
            let x1 = ((q * 23) % 95) as f32 + 0.13;
            let y1 = ((q * 19) % 75) as f32 + 0.17;
            let dx = ((q * 11) % 15) as f32 - 7.0;
            let dy = ((q * 7) % 15) as f32 - 7.0;
            queries.push(GpuDrcRayQuery {
                x1,
                y1,
                x2: x1 + dx,
                y2: y1 + dy,
                clearance: 0.15 + (q % 3) as f32 * 0.10,
                layer_mask: 1u32 << (q % 4),
                net_id: ((q % 12) + 1) as u32,
                _pad: 0,
            });
        }

        let cpu_ref = evaluate_drc_brute_force_cpu(&obstacles, &queries);
        let cu_out = cuda.evaluate_drc_rt_bvh(&bvh_nodes, &queries).expect("CUDA BVH DRC");
        let vk_out = vk
            .evaluate_drc_rt_bvh_vulkan_wgsl(&bvh_nodes, &queries)
            .expect("Vulkan WGSL BVH DRC");

        assert_eq!(cu_out, cpu_ref, "CUDA RT BVH DRC must match brute-force 100%");
        assert_eq!(vk_out, cpu_ref, "Vulkan WGSL RT BVH DRC must match brute-force 100%");
        assert!(cpu_ref.iter().sum::<u32>() > 0, "Must have non-trivial DRC hits");
    }

    #[test]
    fn test_rt_bvh_segment_crossings_equivalence() {
        let cuda = match CudaEngine::new() {
            Some(c) => c,
            None => return,
        };
        let vk = match crate::vulkan::VulkanEngine::new_vulkan_only() {
            Some(v) => v,
            None => return,
        };

        let num_segs = 300usize;
        let mut segs = Vec::with_capacity(num_segs);
        for i in 0..num_segs {
            let x1 = ((i * 29) % 100) as f32 + (i as f32) * 0.013;
            let y1 = ((i * 43) % 100) as f32 + (i as f32) * 0.017;
            let x2 = ((i * 61 + 17) % 100) as f32 + (i as f32) * 0.019;
            let y2 = ((i * 83 + 23) % 100) as f32 + (i as f32) * 0.011;
            segs.push([x1, y1, x2, y2]);
        }

        let bvh_nodes = build_gpu_lbvh_from_segments(&segs);
        let cpu_crossings = evaluate_segment_crossings_brute_force_cpu(&segs);
        let cu_crossings = cuda
            .evaluate_segment_crossings_rt_bvh(&segs, &bvh_nodes)
            .expect("CUDA BVH segment crossings");
        let vk_crossings = vk
            .evaluate_segment_crossings_rt_bvh_vulkan_wgsl(&segs, &bvh_nodes)
            .expect("Vulkan BVH segment crossings");

        assert_eq!(cu_crossings, cpu_crossings, "CUDA BVH crossings must match O(N^2) exact");
        assert_eq!(vk_crossings, cpu_crossings, "Vulkan BVH crossings must match O(N^2) exact");
        assert!(cpu_crossings.iter().sum::<u32>() > 100);
    }
}


