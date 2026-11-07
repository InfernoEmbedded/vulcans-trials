// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials

use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

pub use crate::vulkan::{
    build_gpu_lbvh, build_gpu_lbvh_from_segments, evaluate_corridor_cuts_brute_force_cpu,
    evaluate_corridor_cuts_lbvh_cpu, evaluate_drc_brute_force_cpu,
    evaluate_segment_crossings_brute_force_cpu, morton2d_encode, query_lbvh_crossing_segments,
    ray_aabb_intersects_cpu, segments_cross_cpu, DrcBatchParams, GpuBvhNode, GpuDrcCandidate,
    GpuDrcPad, GpuDrcRayQuery, GpuDrcTrack, GpuDrcVia, GpuRouteCostParams, GpuSaParams,
    PlacementBatchParams, RtBvhBenchmarkResult,
};

static ALL_CUDA_ENGINES: OnceLock<Vec<Arc<CudaEngine>>> = OnceLock::new();

pub const KERNELS_PTX: &str = include_str!("../cuda/kernels.ptx");
pub const AMPERE_KERNELS_PTX: &str = include_str!("../cuda/ampere_kernels.ptx");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceFieldVariant {
    SimtDirect,
    SimtSharedTiled,
    AmperePtxDirect,
    AmperePtxTiled,
    TensorCore,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementEvalVariant {
    SimtStandard,
    AmperePtx,
    TensorCore,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingCost3DVariant {
    SimtSharedTiled,
    TensorCore,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaPlacerVariant {
    SimtFullEval,
    SimtIncrementalDeltaE,
    TensorCore,
}

#[derive(Debug, Clone)]
pub struct KernelDispatchProfile {
    pub device_name: String,
    pub best_distance_field: DistanceFieldVariant,
    pub best_placement_eval: PlacementEvalVariant,
    pub best_routing_cost_3d: RoutingCost3DVariant,
    pub best_sa_placer: SaPlacerVariant,
    pub df_timings_us: Vec<(DistanceFieldVariant, f64)>,
    pub pe_timings_us: Vec<(PlacementEvalVariant, f64)>,
    pub rc_timings_us: Vec<(RoutingCost3DVariant, f64)>,
    pub sa_timings_us: Vec<(SaPlacerVariant, f64)>,
}

static DISPATCH_PROFILE_CACHE: OnceLock<Mutex<HashMap<String, KernelDispatchProfile>>> = OnceLock::new();

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
    pub dispatch_profile: KernelDispatchProfile,
    ctx: CUcontext,
    module: CUmodule,
    ampere_module: CUmodule,
    fn_dist_field: CUfunction,
    fn_dist_field_tiled: CUfunction,
    fn_dist_field_3d_tiled: CUfunction,
    pub fn_dist_field_tensor_core: CUfunction,
    pub fn_dist_field_3d_tensor_core: CUfunction,
    fn_batch_placement: CUfunction,
    pub fn_batch_placement_tensor_core: CUfunction,
    fn_gpu_sa_placer: CUfunction,
    pub fn_gpu_sa_placer_delta_e: CUfunction,
    pub fn_gpu_sa_placer_tensor_core: CUfunction,
    fn_route_cost_3d: CUfunction,
    pub fn_route_cost_3d_tensor_core: CUfunction,
    fn_wavefront_3d: CUfunction,
    fn_wavefront_3d_backtrack: CUfunction,
    fn_rt_bvh_crossings: CUfunction,
    fn_rt_bvh_corridor_cuts: CUfunction,
    fn_rt_bvh_drc: CUfunction,
    fn_batch_drc: CUfunction,
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
            .field("dispatch_profile", &self.dispatch_profile)
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

fn median_us(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    samples[samples.len() / 2]
}

impl CudaEngine {
    pub fn all_devices() -> Vec<Arc<Self>> {
        ALL_CUDA_ENGINES
            .get_or_init(|| {
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
            })
            .clone()
    }

    pub fn new() -> Option<Arc<Self>> {
        Self::all_devices().into_iter().next()
    }

    pub fn new_with_device(ordinal: c_int) -> Option<Arc<Self>> {
        static CUDA_CTX_INIT_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let init_mutex = CUDA_CTX_INIT_LOCK.get_or_init(|| Mutex::new(()));
        let _init_guard = match init_mutex.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
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

            let mut name_buf = [0 as std::ffi::c_char; 256];
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

            let ptx_ascii: String = KERNELS_PTX
                .chars()
                .map(|c| if c.is_ascii() { c } else { '-' })
                .collect();
            let ptx_cstr = CString::new(ptx_ascii).ok()?;
            let mut module: CUmodule = ptr::null_mut();
            if (api.cuModuleLoadData)(&mut module, ptx_cstr.as_ptr() as *const c_void) != CUDA_SUCCESS {
                let _ = (api.cuCtxDestroy_v2)(ctx);
                return None;
            }

            let mut fn_dist_field: CUfunction = ptr::null_mut();
            let mut fn_dist_field_tiled: CUfunction = ptr::null_mut();
            let mut fn_dist_field_3d_tiled: CUfunction = ptr::null_mut();
            let mut fn_dist_field_tensor_core: CUfunction = ptr::null_mut();
            let mut fn_dist_field_3d_tensor_core: CUfunction = ptr::null_mut();
            let mut fn_batch_placement: CUfunction = ptr::null_mut();
            let mut fn_batch_placement_tensor_core: CUfunction = ptr::null_mut();
            let mut fn_gpu_sa_placer: CUfunction = ptr::null_mut();
            let mut fn_gpu_sa_placer_delta_e: CUfunction = ptr::null_mut();
            let mut fn_gpu_sa_placer_tensor_core: CUfunction = ptr::null_mut();
            let mut fn_route_cost_3d: CUfunction = ptr::null_mut();
            let mut fn_route_cost_3d_tensor_core: CUfunction = ptr::null_mut();
            let mut fn_wavefront_3d: CUfunction = ptr::null_mut();
            let mut fn_wavefront_3d_backtrack: CUfunction = ptr::null_mut();
            let mut fn_rt_bvh_crossings: CUfunction = ptr::null_mut();
            let mut fn_rt_bvh_corridor_cuts: CUfunction = ptr::null_mut();
            let mut fn_rt_bvh_drc: CUfunction = ptr::null_mut();
            let mut fn_batch_drc: CUfunction = ptr::null_mut();

            let c_df = CString::new("distance_field_kernel").unwrap();
            let c_dft = CString::new("distance_field_tiled_kernel").unwrap();
            let c_df3d = CString::new("distance_field_3d_tiled_kernel").unwrap();
            let c_dftc = CString::new("distance_field_tensor_core_kernel").unwrap();
            let c_df3dtc = CString::new("distance_field_3d_tensor_core_kernel").unwrap();
            let c_bp = CString::new("batch_placement_eval_kernel").unwrap();
            let c_bptc = CString::new("batch_placement_tensor_core_kernel").unwrap();
            let c_sa = CString::new("gpu_sa_placer_kernel").unwrap();
            let c_sade = CString::new("gpu_sa_placer_delta_e_kernel").unwrap();
            let c_satc = CString::new("gpu_sa_placer_tensor_core_kernel").unwrap();
            let c_rc3d = CString::new("compute_3d_routing_cost_kernel").unwrap();
            let c_rc3dtc = CString::new("compute_3d_routing_cost_tensor_core_kernel").unwrap();
            let c_wf3d = CString::new("wavefront_3d_relax_kernel").unwrap();
            let c_wf3d_bt = CString::new("wavefront_3d_backtrack_kernel").unwrap();
            let c_rt_cr = CString::new("rt_bvh_segment_crossings_kernel").unwrap();
            let c_rt_cc = CString::new("rt_bvh_corridor_cuts_kernel").unwrap();
            let c_rt_drc = CString::new("rt_bvh_line_of_sight_drc_kernel").unwrap();
            let c_drc = CString::new("batch_drc_eval_kernel").unwrap();

            if (api.cuModuleGetFunction)(&mut fn_dist_field, module, c_df.as_ptr()) != CUDA_SUCCESS
                || (api.cuModuleGetFunction)(&mut fn_dist_field_tiled, module, c_dft.as_ptr()) != CUDA_SUCCESS
                || (api.cuModuleGetFunction)(&mut fn_dist_field_3d_tiled, module, c_df3d.as_ptr()) != CUDA_SUCCESS
                || (api.cuModuleGetFunction)(&mut fn_dist_field_tensor_core, module, c_dftc.as_ptr()) != CUDA_SUCCESS
                || (api.cuModuleGetFunction)(&mut fn_dist_field_3d_tensor_core, module, c_df3dtc.as_ptr()) != CUDA_SUCCESS
                || (api.cuModuleGetFunction)(&mut fn_batch_placement, module, c_bp.as_ptr()) != CUDA_SUCCESS
                || (api.cuModuleGetFunction)(&mut fn_batch_placement_tensor_core, module, c_bptc.as_ptr()) != CUDA_SUCCESS
                || (api.cuModuleGetFunction)(&mut fn_gpu_sa_placer, module, c_sa.as_ptr()) != CUDA_SUCCESS
                || (api.cuModuleGetFunction)(&mut fn_gpu_sa_placer_delta_e, module, c_sade.as_ptr()) != CUDA_SUCCESS
                || (api.cuModuleGetFunction)(&mut fn_gpu_sa_placer_tensor_core, module, c_satc.as_ptr()) != CUDA_SUCCESS
                || (api.cuModuleGetFunction)(&mut fn_route_cost_3d, module, c_rc3d.as_ptr()) != CUDA_SUCCESS
                || (api.cuModuleGetFunction)(&mut fn_route_cost_3d_tensor_core, module, c_rc3dtc.as_ptr()) != CUDA_SUCCESS
                || (api.cuModuleGetFunction)(&mut fn_wavefront_3d, module, c_wf3d.as_ptr()) != CUDA_SUCCESS
                || (api.cuModuleGetFunction)(&mut fn_wavefront_3d_backtrack, module, c_wf3d_bt.as_ptr()) != CUDA_SUCCESS
                || (api.cuModuleGetFunction)(&mut fn_rt_bvh_crossings, module, c_rt_cr.as_ptr()) != CUDA_SUCCESS
                || (api.cuModuleGetFunction)(&mut fn_rt_bvh_corridor_cuts, module, c_rt_cc.as_ptr()) != CUDA_SUCCESS
                || (api.cuModuleGetFunction)(&mut fn_rt_bvh_drc, module, c_rt_drc.as_ptr()) != CUDA_SUCCESS
                || (api.cuModuleGetFunction)(&mut fn_batch_drc, module, c_drc.as_ptr()) != CUDA_SUCCESS
            {
                let _ = (api.cuModuleUnload)(module);
                let _ = (api.cuCtxDestroy_v2)(ctx);
                return None;
            }

            let mut ampere_module: CUmodule = ptr::null_mut();
            let mut fn_df_ampere = None;
            let mut fn_df_ampere_tiled = None;
            let mut fn_bp_ampere = None;

            let ampere_ptx_ascii: String = AMPERE_KERNELS_PTX
                .chars()
                .map(|c| if c.is_ascii() { c } else { '-' })
                .collect();
            if let Ok(ampere_ptx_cstr) = CString::new(ampere_ptx_ascii) {
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

            let default_profile = KernelDispatchProfile {
                device_name: device_name.clone(),
                best_distance_field: DistanceFieldVariant::SimtSharedTiled,
                best_placement_eval: PlacementEvalVariant::SimtStandard,
                best_routing_cost_3d: RoutingCost3DVariant::SimtSharedTiled,
                best_sa_placer: SaPlacerVariant::SimtIncrementalDeltaE,
                df_timings_us: Vec::new(),
                pe_timings_us: Vec::new(),
                rc_timings_us: Vec::new(),
                sa_timings_us: Vec::new(),
            };

            let mut engine = Self {
                device_name: device_name.clone(),
                backend_name,
                driver_info,
                optix_available,
                optix_version,
                dispatch_profile: default_profile,
                ctx,
                module,
                ampere_module,
                fn_dist_field,
                fn_dist_field_tiled,
                fn_dist_field_3d_tiled,
                fn_dist_field_tensor_core,
                fn_dist_field_3d_tensor_core,
                fn_batch_placement,
                fn_batch_placement_tensor_core,
                fn_gpu_sa_placer,
                fn_gpu_sa_placer_delta_e,
                fn_gpu_sa_placer_tensor_core,
                fn_route_cost_3d,
                fn_route_cost_3d_tensor_core,
                fn_wavefront_3d,
                fn_wavefront_3d_backtrack,
                fn_rt_bvh_crossings,
                fn_rt_bvh_corridor_cuts,
                fn_rt_bvh_drc,
                fn_batch_drc,
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
            };

            let cache_key = format!("{}:{}", ordinal, device_name);
            engine.calibrate_dispatch_profile(&cache_key);
            let _ = (engine.cuda.cuCtxSetCurrent)(engine.ctx);
            if let Ok(mut bufs) = engine.persistent_buffers.lock() {
                if bufs.d_obs != 0 {
                    let _ = (engine.cuda.cuMemFree_v2)(bufs.d_obs);
                    bufs.d_obs = 0;
                    bufs.obs_cap_bytes = 0;
                }
                if bufs.d_out != 0 {
                    let _ = (engine.cuda.cuMemFree_v2)(bufs.d_out);
                    bufs.d_out = 0;
                    bufs.out_cap_bytes = 0;
                }
            }
            engine.compute_busy_ns.store(0, Ordering::Relaxed);

            Some(Arc::new(engine))
        }
    }

    fn calibrate_dispatch_profile(&mut self, cache_key: &str) {
        let cache = DISPATCH_PROFILE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let mut cache_guard = match cache.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Some(existing) = cache_guard.get(cache_key) {
            self.dispatch_profile = existing.clone();
            return;
        }

        // 1. Calibrate Distance Field (128x128, radius=10)
        let df_w = 128u32;
        let df_h = 128u32;
        let df_r = 10i32;
        let mut df_obs = vec![0u32; (df_w * df_h) as usize];
        for y in (8..df_h as usize).step_by(24) {
            for x in (8..df_w as usize).step_by(24) {
                df_obs[y * (df_w as usize) + x] = 1;
            }
        }
        let ref_df = self
            .compute_distance_field_variant(&df_obs, df_w, df_h, df_r, DistanceFieldVariant::SimtSharedTiled)
            .unwrap_or_default();

        let df_candidates = [
            DistanceFieldVariant::SimtDirect,
            DistanceFieldVariant::SimtSharedTiled,
            DistanceFieldVariant::AmperePtxDirect,
            DistanceFieldVariant::AmperePtxTiled,
            DistanceFieldVariant::TensorCore,
        ];
        let mut df_timings_us = Vec::new();
        let mut best_df = DistanceFieldVariant::SimtSharedTiled;
        let mut best_df_us = f64::INFINITY;

        if !ref_df.is_empty() {
            for &variant in &df_candidates {
                if let Some(out) = self.compute_distance_field_variant(&df_obs, df_w, df_h, df_r, variant) {
                    let valid = out.len() == ref_df.len()
                        && out.iter().zip(&ref_df).all(|(a, b)| (a - b).abs() < 1e-3);
                    if valid {
                        let mut samples = Vec::with_capacity(3);
                        for _ in 0..3 {
                            let t0 = Instant::now();
                            let _ = self.compute_distance_field_variant(&df_obs, df_w, df_h, df_r, variant);
                            samples.push(t0.elapsed().as_secs_f64() * 1e6);
                        }
                        let med = median_us(samples);
                        df_timings_us.push((variant, med));
                        if med < best_df_us {
                            best_df_us = med;
                            best_df = variant;
                        }
                    }
                }
            }
        }

        // 2. Calibrate Placement Evaluation (64 candidates, 16 footprints, 24 segments)
        let pe_nc = 64u32;
        let pe_nf = 16u32;
        let pe_ns = 24u32;
        let pe_params = PlacementBatchParams {
            num_candidates: pe_nc,
            num_footprints: pe_nf,
            num_segments: pe_ns,
            _pad: 0,
            board_min_x: 0.0,
            board_min_y: 0.0,
            board_max_x: 80.0,
            board_max_y: 60.0,
            wirelength_weight: 1.0,
            crossing_weight: 200.0,
            overlap_penalty: 25000.0,
            boundary_penalty: 50000.0,
        };
        let mut pe_boxes = Vec::with_capacity((pe_nc * pe_nf) as usize);
        for c in 0..pe_nc {
            for f in 0..pe_nf {
                let bx = ((f * 11 + c * 3) % 70) as f32 + 2.0;
                let by = ((f * 7 + c * 5) % 50) as f32 + 2.0;
                pe_boxes.push([bx, by, bx + 4.0, by + 3.0]);
            }
        }
        let mut pe_segs = Vec::with_capacity((pe_nc * pe_ns) as usize);
        for c in 0..pe_nc {
            for s in 0..pe_ns {
                let x1 = ((s * 9 + c * 2) % 65) as f32 + 3.0;
                let y1 = ((s * 5 + c * 3) % 45) as f32 + 3.0;
                let x2 = ((s * 13 + c * 7 + 15) % 65) as f32 + 3.0;
                let y2 = ((s * 11 + c * 5 + 10) % 45) as f32 + 3.0;
                pe_segs.push([x1, y1, x2, y2]);
            }
        }
        let ref_pe = self
            .evaluate_placement_variant(&pe_boxes, &pe_segs, &pe_params, PlacementEvalVariant::SimtStandard)
            .unwrap_or_default();

        let pe_candidates = [
            PlacementEvalVariant::SimtStandard,
            PlacementEvalVariant::AmperePtx,
            PlacementEvalVariant::TensorCore,
        ];
        let mut pe_timings_us = Vec::new();
        let mut best_pe = PlacementEvalVariant::SimtStandard;
        let mut best_pe_us = f64::INFINITY;

        if !ref_pe.is_empty() {
            for &variant in &pe_candidates {
                if let Some(out) = self.evaluate_placement_variant(&pe_boxes, &pe_segs, &pe_params, variant) {
                    let valid = out.len() == ref_pe.len()
                        && out.iter().zip(&ref_pe).all(|(a, b)| (a - b).abs() <= 1e-2 * b.abs().max(1.0));
                    if valid {
                        let mut samples = Vec::with_capacity(3);
                        for _ in 0..3 {
                            let t0 = Instant::now();
                            let _ = self.evaluate_placement_variant(&pe_boxes, &pe_segs, &pe_params, variant);
                            samples.push(t0.elapsed().as_secs_f64() * 1e6);
                        }
                        let med = median_us(samples);
                        pe_timings_us.push((variant, med));
                        if med < best_pe_us {
                            best_pe_us = med;
                            best_pe = variant;
                        }
                    }
                }
            }
        }

        // 3. Calibrate 3D Routing Cost (64x64x4, clearance_cells=3)
        let rc_w = 64u32;
        let rc_h = 64u32;
        let rc_nl = 4u32;
        let rc_total = (rc_w * rc_h * rc_nl) as usize;
        let mut rc_obs = vec![0u32; rc_total];
        let mut rc_df = vec![2.0f32; rc_total];
        let mut rc_hist = vec![0.0f32; rc_total];
        let mut rc_usage = vec![0u32; rc_total];
        for i in 0..rc_total {
            if i % 53 == 0 {
                rc_obs[i] = ((i % 6) + 1) as u32;
            }
            rc_df[i] = 0.5 + ((i % 9) as f32) * 0.25;
            rc_hist[i] = ((i % 5) as f32) * 0.2;
            rc_usage[i] = (i % 3) as u32;
        }
        let rc_params = GpuRouteCostParams {
            width: rc_w,
            height: rc_h,
            num_layers: rc_nl,
            net_id: 2,
            clearance_cells: 3,
            present_congestion_factor: 1.25,
            obstacle_clearance_weight: 0.3,
            plane_layer_mask: 0b0100,
        };
        let ref_rc = self
            .compute_3d_routing_costs_variant(
                &rc_obs,
                &rc_df,
                &rc_hist,
                &rc_usage,
                &rc_params,
                RoutingCost3DVariant::SimtSharedTiled,
            )
            .unwrap_or_default();

        let rc_candidates = [RoutingCost3DVariant::SimtSharedTiled, RoutingCost3DVariant::TensorCore];
        let mut rc_timings_us = Vec::new();
        let mut best_rc = RoutingCost3DVariant::SimtSharedTiled;
        let mut best_rc_us = f64::INFINITY;

        if !ref_rc.is_empty() {
            for &variant in &rc_candidates {
                if let Some(out) = self.compute_3d_routing_costs_variant(
                    &rc_obs,
                    &rc_df,
                    &rc_hist,
                    &rc_usage,
                    &rc_params,
                    variant,
                ) {
                    let valid = out.len() == ref_rc.len()
                        && out.iter().zip(&ref_rc).all(|(a, b)| (a - b).abs() < 1e-4);
                    if valid {
                        let mut samples = Vec::with_capacity(3);
                        for _ in 0..3 {
                            let t0 = Instant::now();
                            let _ = self.compute_3d_routing_costs_variant(
                                &rc_obs,
                                &rc_df,
                                &rc_hist,
                                &rc_usage,
                                &rc_params,
                                variant,
                            );
                            samples.push(t0.elapsed().as_secs_f64() * 1e6);
                        }
                        let med = median_us(samples);
                        rc_timings_us.push((variant, med));
                        if med < best_rc_us {
                            best_rc_us = med;
                            best_rc = variant;
                        }
                    }
                }
            }
        }

        // 4. Calibrate GPU SA MCMC Placer (512 chains, 24 footprints, 48 edges, 18 steps/pass, 2 passes)
        let sa_nc = 512u32;
        let sa_nf = 24u32;
        let sa_ne = 48u32;
        let sa_params = GpuSaParams {
            num_chains: sa_nc,
            num_footprints: sa_nf,
            num_edges: sa_ne,
            steps_per_pass: 18,
            seed: 0xC0FFEE42,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            board_min_x: 0.0,
            board_min_y: 0.0,
            board_max_x: 80.0,
            board_max_y: 60.0,
            grid_step: 0.5,
            wirelength_weight: 1.0,
            crossing_weight: 200.0,
            overlap_penalty: 25000.0,
            boundary_penalty: 50000.0,
            passive_affinity_weight: 80.0,
            passive_max_dist: 3.5,
            base_temp: 5.0,
        };
        let mut sa_fp_descs = Vec::with_capacity(sa_nf as usize);
        for f in 0..sa_nf {
            let locked = if f == 0 { 1.0 } else { 0.0 };
            sa_fp_descs.push([1.5, 1.2, locked, 0.0]);
        }
        let mut sa_edges = Vec::with_capacity((sa_ne * 2) as usize);
        for e in 0..sa_ne {
            let fa = (e % sa_nf) as f32;
            let fb = ((e * 5 + 1) % sa_nf) as f32;
            let fa_safe = if (fa - fb).abs() < 0.5 { ((e + 2) % sa_nf) as f32 } else { fa };
            let tag = if e >= 40 { 2.0 } else if e >= 32 { 1.0 } else { 0.0 };
            sa_edges.push([fa_safe, -0.5, 0.2, tag]);
            sa_edges.push([fb, 0.5, -0.2, tag]);
        }
        let mut sa_init_poses = Vec::with_capacity((sa_nc * sa_nf) as usize);
        for _ in 0..sa_nc {
            for f in 0..sa_nf {
                let px = 6.0 + (f % 6) as f32 * 12.0;
                let py = 6.0 + (f / 6) as f32 * 12.0;
                sa_init_poses.push([px, py, 0.0, 0.0]);
            }
        }

        let sa_candidates = [
            SaPlacerVariant::SimtFullEval,
            SaPlacerVariant::SimtIncrementalDeltaE,
            SaPlacerVariant::TensorCore,
        ];
        let mut sa_timings_us = Vec::new();
        let mut best_sa = SaPlacerVariant::SimtIncrementalDeltaE;
        let mut best_sa_us = f64::INFINITY;

        for &variant in &sa_candidates {
            if let Some((poses, scores)) = self.run_gpu_sa_placer_chains_variant(
                &sa_fp_descs,
                &sa_edges,
                &sa_init_poses,
                &sa_params,
                2,
                variant,
            ) {
                let valid = poses.len() == (sa_nc * sa_nf) as usize
                    && scores.len() == sa_nc as usize
                    && scores.iter().all(|s| s[0].is_finite() && s[1] > 0.0 && s[1].is_finite() && s[2] >= 0.0)
                    && poses.iter().all(|p| p[0].is_finite() && p[1].is_finite() && p[2].is_finite());
                if valid {
                    let mut samples = Vec::with_capacity(3);
                    for _ in 0..3 {
                        let t0 = Instant::now();
                        let _ = self.run_gpu_sa_placer_chains_variant(
                            &sa_fp_descs,
                            &sa_edges,
                            &sa_init_poses,
                            &sa_params,
                            2,
                            variant,
                        );
                        samples.push(t0.elapsed().as_secs_f64() * 1e6);
                    }
                    let med = median_us(samples);
                    sa_timings_us.push((variant, med));
                    if med < best_sa_us {
                        best_sa_us = med;
                        best_sa = variant;
                    }
                }
            }
        }

        let profile = KernelDispatchProfile {
            device_name: self.device_name.clone(),
            best_distance_field: best_df,
            best_placement_eval: best_pe,
            best_routing_cost_3d: best_rc,
            best_sa_placer: best_sa,
            df_timings_us,
            pe_timings_us,
            rc_timings_us,
            sa_timings_us,
        };
        cache_guard.insert(cache_key.to_string(), profile.clone());
        self.dispatch_profile = profile;
    }

    unsafe fn ensure_persistent_buffers(&self, in_bytes: usize, out_bytes: usize) -> Option<(CUdeviceptr, CUdeviceptr)> {
        let mut bufs = self.persistent_buffers.lock().unwrap_or_else(|e| e.into_inner());
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
        self.compute_distance_field_variant(
            obstacles,
            width,
            height,
            radius,
            self.dispatch_profile.best_distance_field,
        )
        .or_else(|| self.compute_distance_field_mode(obstacles, width, height, radius, true))
    }

    pub fn compute_distance_field_variant(
        &self,
        obstacles: &[u32],
        width: u32,
        height: u32,
        radius: i32,
        variant: DistanceFieldVariant,
    ) -> Option<Vec<f32>> {
        match variant {
            DistanceFieldVariant::SimtDirect => self.compute_distance_field_mode(obstacles, width, height, radius, false),
            DistanceFieldVariant::SimtSharedTiled => self.compute_distance_field_mode(obstacles, width, height, radius, true),
            DistanceFieldVariant::AmperePtxDirect => self.compute_distance_field_ampere(obstacles, width, height, radius, false),
            DistanceFieldVariant::AmperePtxTiled => self.compute_distance_field_ampere(obstacles, width, height, radius, true),
            DistanceFieldVariant::TensorCore => self.compute_distance_field_tensor_core(obstacles, width, height, radius),
        }
    }

    pub fn compute_distance_field_tensor_core(
        &self,
        obstacles: &[u32],
        width: u32,
        height: u32,
        radius: i32,
    ) -> Option<Vec<f32>> {
        self.launch_distance_field_kernel(self.fn_dist_field_tensor_core, obstacles, width, height, radius, true)
    }

    pub fn compute_distance_field_mode(
        &self,
        obstacles: &[u32],
        width: u32,
        height: u32,
        radius: i32,
        use_shared_memory: bool,
    ) -> Option<Vec<f32>> {
        let kernel = if use_shared_memory {
            self.fn_dist_field_tiled
        } else {
            self.fn_dist_field
        };
        self.launch_distance_field_kernel(kernel, obstacles, width, height, radius, use_shared_memory)
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
        self.launch_distance_field_kernel(kernel, obstacles, width, height, radius, tiled)
    }

    fn launch_distance_field_kernel(
        &self,
        kernel: CUfunction,
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

        let _guard = self.dispatch_lock.lock().unwrap_or_else(|e| e.into_inner());
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

        let kernel = if self.dispatch_profile.best_distance_field == DistanceFieldVariant::TensorCore {
            self.fn_dist_field_3d_tensor_core
        } else {
            self.fn_dist_field_3d_tiled
        };

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
                    kernel,
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

    pub fn evaluate_placement_batch(
        &self,
        boxes: &[[f32; 4]],
        segs: &[[f32; 4]],
        params: &PlacementBatchParams,
    ) -> Option<Vec<f32>> {
        self.evaluate_placement_variant(boxes, segs, params, self.dispatch_profile.best_placement_eval)
            .or_else(|| self.evaluate_placement_variant(boxes, segs, params, PlacementEvalVariant::SimtStandard))
    }

    pub fn evaluate_placement_variant(
        &self,
        boxes: &[[f32; 4]],
        segs: &[[f32; 4]],
        params: &PlacementBatchParams,
        variant: PlacementEvalVariant,
    ) -> Option<Vec<f32>> {
        let kernel = match variant {
            PlacementEvalVariant::SimtStandard => self.fn_batch_placement,
            PlacementEvalVariant::AmperePtx => self.fn_bp_ampere?,
            PlacementEvalVariant::TensorCore => self.fn_batch_placement_tensor_core,
        };
        self.launch_placement_kernel(kernel, boxes, segs, params)
    }

    pub fn evaluate_placement_ampere(
        &self,
        boxes: &[[f32; 4]],
        segs: &[[f32; 4]],
        params: &PlacementBatchParams,
    ) -> Option<Vec<f32>> {
        self.evaluate_placement_variant(boxes, segs, params, PlacementEvalVariant::AmperePtx)
    }

    pub fn evaluate_placement_tensor_core(
        &self,
        boxes: &[[f32; 4]],
        segs: &[[f32; 4]],
        params: &PlacementBatchParams,
    ) -> Option<Vec<f32>> {
        self.evaluate_placement_variant(boxes, segs, params, PlacementEvalVariant::TensorCore)
    }

    fn launch_placement_kernel(
        &self,
        kernel: CUfunction,
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

    pub fn run_gpu_sa_placer_chains(
        &self,
        fp_descs: &[[f32; 4]],
        edges: &[[f32; 4]],
        init_poses: &[[f32; 4]],
        params: &GpuSaParams,
        num_passes: u32,
    ) -> Option<(Vec<[f32; 4]>, Vec<[f32; 4]>)> {
        self.run_gpu_sa_placer_chains_variant(
            fp_descs,
            edges,
            init_poses,
            params,
            num_passes,
            self.dispatch_profile.best_sa_placer,
        )
        .or_else(|| {
            self.run_gpu_sa_placer_chains_variant(
                fp_descs,
                edges,
                init_poses,
                params,
                num_passes,
                SaPlacerVariant::SimtFullEval,
            )
        })
    }

    pub fn run_gpu_sa_placer_chains_variant(
        &self,
        fp_descs: &[[f32; 4]],
        edges: &[[f32; 4]],
        init_poses: &[[f32; 4]],
        params: &GpuSaParams,
        num_passes: u32,
        variant: SaPlacerVariant,
    ) -> Option<(Vec<[f32; 4]>, Vec<[f32; 4]>)> {
        let nc = params.num_chains as usize;
        let nf = params.num_footprints as usize;
        let ne = params.num_edges as usize;
        if nc == 0 || nf == 0 || fp_descs.len() != nf || edges.len() != ne * 2 || init_poses.len() != nc * nf {
            return None;
        }

        let kernel = match variant {
            SaPlacerVariant::SimtFullEval => self.fn_gpu_sa_placer,
            SaPlacerVariant::SimtIncrementalDeltaE => self.fn_gpu_sa_placer_delta_e,
            SaPlacerVariant::TensorCore => self.fn_gpu_sa_placer_tensor_core,
        };

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
        self.compute_3d_routing_costs_variant(
            obstacles,
            dist_field,
            history_cost,
            usage_count,
            params,
            self.dispatch_profile.best_routing_cost_3d,
        )
    }

    pub fn compute_3d_routing_costs_tensor_core(
        &self,
        obstacles: &[u32],
        dist_field: &[f32],
        history_cost: &[f32],
        usage_count: &[u32],
        params: &GpuRouteCostParams,
    ) -> Option<Vec<f32>> {
        self.compute_3d_routing_costs_variant(
            obstacles,
            dist_field,
            history_cost,
            usage_count,
            params,
            RoutingCost3DVariant::TensorCore,
        )
    }

    pub fn compute_3d_routing_costs_variant(
        &self,
        obstacles: &[u32],
        dist_field: &[f32],
        history_cost: &[f32],
        usage_count: &[u32],
        params: &GpuRouteCostParams,
        variant: RoutingCost3DVariant,
    ) -> Option<Vec<f32>> {
        let mut batch = self.compute_3d_routing_costs_multi_variant(
            obstacles,
            dist_field,
            history_cost,
            usage_count,
            std::slice::from_ref(params),
            variant,
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
        self.compute_3d_routing_costs_multi_variant(
            obstacles,
            dist_field,
            history_cost,
            usage_count,
            batch_params,
            self.dispatch_profile.best_routing_cost_3d,
        )
    }

    pub fn compute_3d_routing_costs_multi_variant(
        &self,
        obstacles: &[u32],
        dist_field: &[f32],
        history_cost: &[f32],
        usage_count: &[u32],
        batch_params: &[GpuRouteCostParams],
        variant: RoutingCost3DVariant,
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

        let kernel = match variant {
            RoutingCost3DVariant::SimtSharedTiled => self.fn_route_cost_3d,
            RoutingCost3DVariant::TensorCore => self.fn_route_cost_3d_tensor_core,
        };

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
                    kernel,
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

    pub fn evaluate_corridor_cuts_rt_bvh(
        &self,
        segs: &[[f32; 4]],
        bvh_nodes: &[GpuBvhNode],
        cuts: &[[f32; 4]],
    ) -> Option<Vec<u32>> {
        if segs.is_empty() || bvh_nodes.is_empty() || cuts.is_empty() {
            return None;
        }
        let num_cuts = cuts.len();
        let segs_bytes = segs.len() * std::mem::size_of::<[f32; 4]>();
        let bvh_bytes = bvh_nodes.len() * std::mem::size_of::<GpuBvhNode>();
        let cuts_bytes = num_cuts * std::mem::size_of::<[f32; 4]>();
        let out_bytes = num_cuts * std::mem::size_of::<u32>();

        let _guard = self.dispatch_lock.lock().ok()?;
        unsafe {
            let _ = (self.cuda.cuCtxSetCurrent)(self.ctx);
            let mut d_segs: CUdeviceptr = 0;
            let mut d_bvh: CUdeviceptr = 0;
            let mut d_cuts: CUdeviceptr = 0;
            let mut d_out: CUdeviceptr = 0;

            if (self.cuda.cuMemAlloc_v2)(&mut d_segs, segs_bytes) != CUDA_SUCCESS {
                return None;
            }
            if (self.cuda.cuMemAlloc_v2)(&mut d_bvh, bvh_bytes) != CUDA_SUCCESS {
                let _ = (self.cuda.cuMemFree_v2)(d_segs);
                return None;
            }
            if (self.cuda.cuMemAlloc_v2)(&mut d_cuts, cuts_bytes) != CUDA_SUCCESS {
                let _ = (self.cuda.cuMemFree_v2)(d_segs);
                let _ = (self.cuda.cuMemFree_v2)(d_bvh);
                return None;
            }
            if (self.cuda.cuMemAlloc_v2)(&mut d_out, out_bytes) != CUDA_SUCCESS {
                let _ = (self.cuda.cuMemFree_v2)(d_segs);
                let _ = (self.cuda.cuMemFree_v2)(d_bvh);
                let _ = (self.cuda.cuMemFree_v2)(d_cuts);
                return None;
            }

            let start_t = Instant::now();

            if (self.cuda.cuMemcpyHtoD_v2)(d_segs, segs.as_ptr() as *const c_void, segs_bytes) != CUDA_SUCCESS
                || (self.cuda.cuMemcpyHtoD_v2)(d_bvh, bvh_nodes.as_ptr() as *const c_void, bvh_bytes) != CUDA_SUCCESS
                || (self.cuda.cuMemcpyHtoD_v2)(d_cuts, cuts.as_ptr() as *const c_void, cuts_bytes) != CUDA_SUCCESS
            {
                let _ = (self.cuda.cuMemFree_v2)(d_segs);
                let _ = (self.cuda.cuMemFree_v2)(d_bvh);
                let _ = (self.cuda.cuMemFree_v2)(d_cuts);
                let _ = (self.cuda.cuMemFree_v2)(d_out);
                return None;
            }

            let block_x: c_uint = 64;
            let grid_x: c_uint = (num_cuts as c_uint + block_x - 1) / block_x;
            let mut num_cuts_arg = num_cuts as c_uint;

            let mut args = [
                &mut d_segs as *mut _ as *mut c_void,
                &mut d_bvh as *mut _ as *mut c_void,
                &mut d_cuts as *mut _ as *mut c_void,
                &mut d_out as *mut _ as *mut c_void,
                &mut num_cuts_arg as *mut _ as *mut c_void,
            ];

            if (self.cuda.cuLaunchKernel)(
                self.fn_rt_bvh_corridor_cuts,
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
                let _ = (self.cuda.cuMemFree_v2)(d_cuts);
                let _ = (self.cuda.cuMemFree_v2)(d_out);
                return None;
            }

            let mut result = vec![0u32; num_cuts];
            if (self.cuda.cuMemcpyDtoH_v2)(result.as_mut_ptr() as *mut c_void, d_out, out_bytes) != CUDA_SUCCESS {
                let _ = (self.cuda.cuMemFree_v2)(d_segs);
                let _ = (self.cuda.cuMemFree_v2)(d_bvh);
                let _ = (self.cuda.cuMemFree_v2)(d_cuts);
                let _ = (self.cuda.cuMemFree_v2)(d_out);
                return None;
            }

            let _ = (self.cuda.cuCtxSynchronize)();
            let elapsed_ns = start_t.elapsed().as_nanos() as u64;
            self.compute_busy_ns.fetch_add(elapsed_ns, Ordering::Relaxed);

            let _ = (self.cuda.cuMemFree_v2)(d_segs);
            let _ = (self.cuda.cuMemFree_v2)(d_bvh);
            let _ = (self.cuda.cuMemFree_v2)(d_cuts);
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

    pub fn evaluate_drc_candidates(
        &self,
        pads: &[GpuDrcPad],
        tracks: &[GpuDrcTrack],
        vias: &[GpuDrcVia],
        candidates: &[GpuDrcCandidate],
    ) -> Option<Vec<u32>> {
        let nc = candidates.len();
        if nc == 0 {
            return Some(Vec::new());
        }

        let devs = Self::all_devices();
        if devs.len() > 1 {
            if nc >= 32 {
                let num_devs = devs.len();
                let chunk_size = (nc + num_devs - 1) / num_devs;
                let mut chunks_out: Vec<Option<Vec<u32>>> = (0..num_devs).map(|_| None).collect();
                std::thread::scope(|scope| {
                    let mut handles = Vec::with_capacity(num_devs);
                    for (idx, dev) in devs.iter().enumerate() {
                        let start = idx * chunk_size;
                        let end = (start + chunk_size).min(nc);
                        if start >= end {
                            continue;
                        }
                        let dev_ref = Arc::clone(dev);
                        let cand_slice = &candidates[start..end];
                        handles.push((
                            idx,
                            scope.spawn(move || {
                                dev_ref.evaluate_drc_candidates_single(pads, tracks, vias, cand_slice)
                            }),
                        ));
                    }
                    for (idx, handle) in handles {
                        if let Ok(res) = handle.join() {
                            chunks_out[idx] = res;
                        }
                    }
                });

                let mut combined = Vec::with_capacity(nc);
                for opt in chunks_out {
                    if let Some(part) = opt {
                        combined.extend_from_slice(&part);
                    }
                }
                if combined.len() == nc {
                    return Some(combined);
                }
            } else {
                static DRC_DEVICE_RR: std::sync::atomic::AtomicUsize =
                    std::sync::atomic::AtomicUsize::new(0);
                let rr_idx =
                    DRC_DEVICE_RR.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % devs.len();
                if let Some(res) =
                    devs[rr_idx].evaluate_drc_candidates_single(pads, tracks, vias, candidates)
                {
                    return Some(res);
                }
            }
        }

        self.evaluate_drc_candidates_single(pads, tracks, vias, candidates)
    }

    pub fn evaluate_drc_candidates_single(
        &self,
        pads: &[GpuDrcPad],
        tracks: &[GpuDrcTrack],
        vias: &[GpuDrcVia],
        candidates: &[GpuDrcCandidate],
    ) -> Option<Vec<u32>> {
        let nc = candidates.len();
        if nc == 0 {
            return Some(Vec::new());
        }

        let _guard = self.dispatch_lock.lock().ok()?;
        unsafe {
            let _ = (self.cuda.cuCtxSetCurrent)(self.ctx);

            let dummy_pad = [bytemuck::Zeroable::zeroed(); 1];
            let pad_slice: &[GpuDrcPad] = if pads.is_empty() { &dummy_pad } else { pads };
            let pads_bytes = pad_slice.len() * std::mem::size_of::<GpuDrcPad>();

            let dummy_trk = [bytemuck::Zeroable::zeroed(); 1];
            let trk_slice: &[GpuDrcTrack] = if tracks.is_empty() { &dummy_trk } else { tracks };
            let tracks_bytes = trk_slice.len() * std::mem::size_of::<GpuDrcTrack>();

            let dummy_via = [bytemuck::Zeroable::zeroed(); 1];
            let via_slice: &[GpuDrcVia] = if vias.is_empty() { &dummy_via } else { vias };
            let vias_bytes = via_slice.len() * std::mem::size_of::<GpuDrcVia>();

            let cands_bytes = nc * std::mem::size_of::<GpuDrcCandidate>();
            let out_bytes = nc * std::mem::size_of::<u32>();

            let mut d_pads: CUdeviceptr = 0;
            let mut d_tracks: CUdeviceptr = 0;
            let mut d_vias: CUdeviceptr = 0;
            let mut d_cands: CUdeviceptr = 0;
            let mut d_out: CUdeviceptr = 0;

            if (self.cuda.cuMemAlloc_v2)(&mut d_pads, pads_bytes) != CUDA_SUCCESS {
                return None;
            }
            if (self.cuda.cuMemAlloc_v2)(&mut d_tracks, tracks_bytes) != CUDA_SUCCESS {
                let _ = (self.cuda.cuMemFree_v2)(d_pads);
                return None;
            }
            if (self.cuda.cuMemAlloc_v2)(&mut d_vias, vias_bytes) != CUDA_SUCCESS {
                let _ = (self.cuda.cuMemFree_v2)(d_pads);
                let _ = (self.cuda.cuMemFree_v2)(d_tracks);
                return None;
            }
            if (self.cuda.cuMemAlloc_v2)(&mut d_cands, cands_bytes) != CUDA_SUCCESS {
                let _ = (self.cuda.cuMemFree_v2)(d_pads);
                let _ = (self.cuda.cuMemFree_v2)(d_tracks);
                let _ = (self.cuda.cuMemFree_v2)(d_vias);
                return None;
            }
            if (self.cuda.cuMemAlloc_v2)(&mut d_out, out_bytes) != CUDA_SUCCESS {
                let _ = (self.cuda.cuMemFree_v2)(d_pads);
                let _ = (self.cuda.cuMemFree_v2)(d_tracks);
                let _ = (self.cuda.cuMemFree_v2)(d_vias);
                let _ = (self.cuda.cuMemFree_v2)(d_cands);
                return None;
            }

            let start_t = Instant::now();

            if (self.cuda.cuMemcpyHtoD_v2)(d_pads, pad_slice.as_ptr() as *const c_void, pads_bytes) != CUDA_SUCCESS
                || (self.cuda.cuMemcpyHtoD_v2)(d_tracks, trk_slice.as_ptr() as *const c_void, tracks_bytes) != CUDA_SUCCESS
                || (self.cuda.cuMemcpyHtoD_v2)(d_vias, via_slice.as_ptr() as *const c_void, vias_bytes) != CUDA_SUCCESS
                || (self.cuda.cuMemcpyHtoD_v2)(d_cands, candidates.as_ptr() as *const c_void, cands_bytes) != CUDA_SUCCESS
            {
                let _ = (self.cuda.cuMemFree_v2)(d_pads);
                let _ = (self.cuda.cuMemFree_v2)(d_tracks);
                let _ = (self.cuda.cuMemFree_v2)(d_vias);
                let _ = (self.cuda.cuMemFree_v2)(d_cands);
                let _ = (self.cuda.cuMemFree_v2)(d_out);
                return None;
            }

            let block_x: c_uint = 64;
            let grid_x: c_uint = ((nc as c_uint) + block_x - 1) / block_x;

            let mut params = DrcBatchParams {
                num_pads: pads.len() as u32,
                num_tracks: tracks.len() as u32,
                num_vias: vias.len() as u32,
                num_candidates: nc as u32,
            };

            let mut args = [
                &mut d_pads as *mut _ as *mut c_void,
                &mut d_tracks as *mut _ as *mut c_void,
                &mut d_vias as *mut _ as *mut c_void,
                &mut d_cands as *mut _ as *mut c_void,
                &mut d_out as *mut _ as *mut c_void,
                &mut params as *mut _ as *mut c_void,
            ];

            if (self.cuda.cuLaunchKernel)(
                self.fn_batch_drc,
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
                let _ = (self.cuda.cuMemFree_v2)(d_pads);
                let _ = (self.cuda.cuMemFree_v2)(d_tracks);
                let _ = (self.cuda.cuMemFree_v2)(d_vias);
                let _ = (self.cuda.cuMemFree_v2)(d_cands);
                let _ = (self.cuda.cuMemFree_v2)(d_out);
                return None;
            }

            let mut result = vec![0u32; nc];
            if (self.cuda.cuMemcpyDtoH_v2)(result.as_mut_ptr() as *mut c_void, d_out, out_bytes) != CUDA_SUCCESS {
                let _ = (self.cuda.cuMemFree_v2)(d_pads);
                let _ = (self.cuda.cuMemFree_v2)(d_tracks);
                let _ = (self.cuda.cuMemFree_v2)(d_vias);
                let _ = (self.cuda.cuMemFree_v2)(d_cands);
                let _ = (self.cuda.cuMemFree_v2)(d_out);
                return None;
            }

            let _ = (self.cuda.cuCtxSynchronize)();
            let elapsed_ns = start_t.elapsed().as_nanos() as u64;
            self.compute_busy_ns.fetch_add(elapsed_ns, Ordering::Relaxed);

            let _ = (self.cuda.cuMemFree_v2)(d_pads);
            let _ = (self.cuda.cuMemFree_v2)(d_tracks);
            let _ = (self.cuda.cuMemFree_v2)(d_vias);
            let _ = (self.cuda.cuMemFree_v2)(d_cands);
            let _ = (self.cuda.cuMemFree_v2)(d_out);

            Some(result)
        }
    }

    /// Solves a single 3D shortest path on the GPU using parallel Bellman-Ford / Lee wavefront
    /// relaxation (`wavefront_3d_relax_kernel`) followed by GPU backtracking (`wavefront_3d_backtrack_kernel`).
    pub fn solve_3d_wavefront_path(
        &self,
        obstacle_grid: &[u32],
        gw: usize,
        gh: usize,
        layers: usize,
        src: (usize, usize, usize),
        dst: (usize, usize, usize),
        net_id: u32,
        via_cost: f32,
    ) -> Option<Vec<(usize, usize, usize)>> {
        self.solve_3d_wavefront_path_with_clearance(
            obstacle_grid,
            gw,
            gh,
            layers,
            src,
            dst,
            net_id,
            via_cost,
            0,
        )
    }

    /// Solves a single 3D shortest path on the GPU with explicit net clearance radius (`clearance_cells`).
    pub fn solve_3d_wavefront_path_with_clearance(
        &self,
        obstacle_grid: &[u32],
        gw: usize,
        gh: usize,
        layers: usize,
        src: (usize, usize, usize),
        dst: (usize, usize, usize),
        net_id: u32,
        via_cost: f32,
        clearance_cells: i32,
    ) -> Option<Vec<(usize, usize, usize)>> {
        self.solve_batch_3d_wavefront_paths_with_clearance(
            obstacle_grid,
            gw,
            gh,
            layers,
            &[(src, dst, net_id, via_cost, clearance_cells)],
        )
        .into_iter()
        .next()
        .flatten()
    }

    /// Solves a batch of 3D shortest paths on this CUDA GPU, uploading `obstacle_grid` once.
    pub fn solve_batch_3d_wavefront_paths(
        &self,
        obstacle_grid: &[u32],
        gw: usize,
        gh: usize,
        layers: usize,
        requests: &[((usize, usize, usize), (usize, usize, usize), u32, f32)],
    ) -> Vec<Option<Vec<(usize, usize, usize)>>> {
        let reqs_with_clearance: Vec<((usize, usize, usize), (usize, usize, usize), u32, f32, i32)> =
            requests
                .iter()
                .map(|&(src, dst, net_id, via_cost)| (src, dst, net_id, via_cost, 0))
                .collect();
        self.solve_batch_3d_wavefront_paths_with_clearance(
            obstacle_grid,
            gw,
            gh,
            layers,
            &reqs_with_clearance,
        )
    }

    /// Solves a batch of 3D shortest paths with per-request clearance radii on this CUDA GPU.
    pub fn solve_batch_3d_wavefront_paths_with_clearance(
        &self,
        obstacle_grid: &[u32],
        gw: usize,
        gh: usize,
        layers: usize,
        requests: &[((usize, usize, usize), (usize, usize, usize), u32, f32, i32)],
    ) -> Vec<Option<Vec<(usize, usize, usize)>>> {
        if requests.is_empty() || gw == 0 || gh == 0 || layers == 0 {
            return vec![None; requests.len()];
        }
        let total = match gw.checked_mul(gh).and_then(|v| v.checked_mul(layers)) {
            Some(t) if t > 0 && obstacle_grid.len() == t => t,
            _ => return vec![None; requests.len()],
        };

        let _guard = match self.dispatch_lock.lock() {
            Ok(g) => g,
            Err(_) => return vec![None; requests.len()],
        };

        let max_path_len = total.min(65536).max(64) as u32;
        let u32_bytes = total * std::mem::size_of::<u32>();
        let f32_bytes = total * std::mem::size_of::<f32>();
        let path_bytes = (max_path_len as usize) * 3 * std::mem::size_of::<u32>();
        let len_bytes = std::mem::size_of::<u32>();

        let zero_f32 = vec![0.0f32; total];
        let zero_u32 = vec![0u32; total];
        let mut init_wf = vec![1.0e20f32; total];

        unsafe {
            let _ = (self.cuda.cuCtxSetCurrent)(self.ctx);

            let mut d_obs: CUdeviceptr = 0;
            let mut d_df: CUdeviceptr = 0;
            let mut d_hist: CUdeviceptr = 0;
            let mut d_usage: CUdeviceptr = 0;
            let mut d_costs: CUdeviceptr = 0;
            let mut d_wf_a: CUdeviceptr = 0;
            let mut d_wf_b: CUdeviceptr = 0;
            let mut d_path_out: CUdeviceptr = 0;
            let mut d_path_len: CUdeviceptr = 0;

            let alloc_ok = (self.cuda.cuMemAlloc_v2)(&mut d_obs, u32_bytes) == CUDA_SUCCESS
                && (self.cuda.cuMemAlloc_v2)(&mut d_df, f32_bytes) == CUDA_SUCCESS
                && (self.cuda.cuMemAlloc_v2)(&mut d_hist, f32_bytes) == CUDA_SUCCESS
                && (self.cuda.cuMemAlloc_v2)(&mut d_usage, u32_bytes) == CUDA_SUCCESS
                && (self.cuda.cuMemAlloc_v2)(&mut d_costs, f32_bytes) == CUDA_SUCCESS
                && (self.cuda.cuMemAlloc_v2)(&mut d_wf_a, f32_bytes) == CUDA_SUCCESS
                && (self.cuda.cuMemAlloc_v2)(&mut d_wf_b, f32_bytes) == CUDA_SUCCESS
                && (self.cuda.cuMemAlloc_v2)(&mut d_path_out, path_bytes) == CUDA_SUCCESS
                && (self.cuda.cuMemAlloc_v2)(&mut d_path_len, len_bytes) == CUDA_SUCCESS;

            let free_all = |api: &CudaDriverApi, ptrs: [CUdeviceptr; 9]| {
                for ptr in ptrs {
                    if ptr != 0 {
                        let _ = (api.cuMemFree_v2)(ptr);
                    }
                }
            };

            if !alloc_ok {
                free_all(
                    &self.cuda,
                    [
                        d_obs, d_df, d_hist, d_usage, d_costs, d_wf_a, d_wf_b, d_path_out,
                        d_path_len,
                    ],
                );
                return vec![None; requests.len()];
            }

            let start_t = Instant::now();

            if (self.cuda.cuMemcpyHtoD_v2)(d_obs, obstacle_grid.as_ptr() as *const c_void, u32_bytes) != CUDA_SUCCESS
                || (self.cuda.cuMemcpyHtoD_v2)(d_df, zero_f32.as_ptr() as *const c_void, f32_bytes) != CUDA_SUCCESS
                || (self.cuda.cuMemcpyHtoD_v2)(d_hist, zero_f32.as_ptr() as *const c_void, f32_bytes) != CUDA_SUCCESS
                || (self.cuda.cuMemcpyHtoD_v2)(d_usage, zero_u32.as_ptr() as *const c_void, u32_bytes) != CUDA_SUCCESS
            {
                free_all(
                    &self.cuda,
                    [
                        d_obs, d_df, d_hist, d_usage, d_costs, d_wf_a, d_wf_b, d_path_out,
                        d_path_len,
                    ],
                );
                return vec![None; requests.len()];
            }

            let block_x: c_uint = 16;
            let block_y: c_uint = 16;
            let grid_x: c_uint = (gw as c_uint + block_x - 1) / block_x;
            let grid_y: c_uint = (gh as c_uint + block_y - 1) / block_y;
            let grid_z: c_uint = layers as c_uint;
            let iterations = ((gw + gh + (gw * gh / 2).min(512) + layers * 4)
                .max(32)
                .min(4096)) as u32;

            let cost_kernel = match self.dispatch_profile.best_routing_cost_3d {
                RoutingCost3DVariant::TensorCore => self.fn_route_cost_3d_tensor_core,
                RoutingCost3DVariant::SimtSharedTiled => self.fn_route_cost_3d,
            };

            let layer_stride = gw * gh;
            let mut outputs = Vec::with_capacity(requests.len());

            for &(src, dst, net_id, via_cost, clearance_cells) in requests {
                if src.0 >= gw
                    || src.1 >= gh
                    || src.2 >= layers
                    || dst.0 >= gw
                    || dst.1 >= gh
                    || dst.2 >= layers
                {
                    outputs.push(None);
                    continue;
                }
                let src_idx = src.2 * layer_stride + src.1 * gw + src.0;
                let dst_idx = dst.2 * layer_stride + dst.1 * gw + dst.0;
                let src_obs = obstacle_grid[src_idx];
                let dst_obs = obstacle_grid[dst_idx];
                if (src_obs != 0 && src_obs != net_id) || (dst_obs != 0 && dst_obs != net_id) {
                    outputs.push(None);
                    continue;
                }
                if src == dst {
                    outputs.push(Some(vec![src]));
                    continue;
                }

                let r = clearance_cells.max(0);
                let mut params_copy = GpuRouteCostParams {
                    width: gw as u32,
                    height: gh as u32,
                    num_layers: layers as u32,
                    net_id,
                    clearance_cells: r,
                    present_congestion_factor: 0.0,
                    obstacle_clearance_weight: 0.0,
                    plane_layer_mask: 0,
                };
                let smem_bytes: c_uint = if r <= 16 {
                    let tile_w = block_x as i32 + 2 * r;
                    let tile_h = block_y as i32 + 2 * r;
                    (tile_w * tile_h * 4) as c_uint
                } else {
                    0
                };

                let mut cost_args = [
                    &mut d_obs as *mut _ as *mut c_void,
                    &mut d_df as *mut _ as *mut c_void,
                    &mut d_hist as *mut _ as *mut c_void,
                    &mut d_usage as *mut _ as *mut c_void,
                    &mut d_costs as *mut _ as *mut c_void,
                    &mut params_copy as *mut _ as *mut c_void,
                ];
                if (self.cuda.cuLaunchKernel)(
                    cost_kernel,
                    grid_x,
                    grid_y,
                    grid_z,
                    block_x,
                    block_y,
                    1,
                    smem_bytes,
                    ptr::null_mut(),
                    cost_args.as_mut_ptr(),
                    ptr::null_mut(),
                ) != CUDA_SUCCESS
                {
                    outputs.push(None);
                    continue;
                }

                init_wf[src_idx] = 0.0;
                let copy_ok = (self.cuda.cuMemcpyHtoD_v2)(
                    d_wf_a,
                    init_wf.as_ptr() as *const c_void,
                    f32_bytes,
                ) == CUDA_SUCCESS;
                init_wf[src_idx] = 1.0e20;
                if !copy_ok {
                    outputs.push(None);
                    continue;
                }

                let mut via_cost_arg = via_cost;
                let mut relax_ok = true;
                for iter in 0..iterations {
                    let (mut d_in, mut d_out) = if iter % 2 == 0 {
                        (d_wf_a, d_wf_b)
                    } else {
                        (d_wf_b, d_wf_a)
                    };
                    let mut wf_args = [
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
                        wf_args.as_mut_ptr(),
                        ptr::null_mut(),
                    ) != CUDA_SUCCESS
                    {
                        relax_ok = false;
                        break;
                    }
                }
                if !relax_ok {
                    outputs.push(None);
                    continue;
                }

                let mut final_dev = if iterations % 2 == 1 { d_wf_b } else { d_wf_a };
                let mut src_x_arg = src.0 as u32;
                let mut src_y_arg = src.1 as u32;
                let mut src_l_arg = src.2 as u32;
                let mut dst_x_arg = dst.0 as u32;
                let mut dst_y_arg = dst.1 as u32;
                let mut dst_l_arg = dst.2 as u32;
                let mut max_len_arg = max_path_len;

                let mut bt_args = [
                    &mut final_dev as *mut _ as *mut c_void,
                    &mut d_costs as *mut _ as *mut c_void,
                    &mut params_copy as *mut _ as *mut c_void,
                    &mut via_cost_arg as *mut _ as *mut c_void,
                    &mut src_x_arg as *mut _ as *mut c_void,
                    &mut src_y_arg as *mut _ as *mut c_void,
                    &mut src_l_arg as *mut _ as *mut c_void,
                    &mut dst_x_arg as *mut _ as *mut c_void,
                    &mut dst_y_arg as *mut _ as *mut c_void,
                    &mut dst_l_arg as *mut _ as *mut c_void,
                    &mut max_len_arg as *mut _ as *mut c_void,
                    &mut d_path_out as *mut _ as *mut c_void,
                    &mut d_path_len as *mut _ as *mut c_void,
                ];
                if (self.cuda.cuLaunchKernel)(
                    self.fn_wavefront_3d_backtrack,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    bt_args.as_mut_ptr(),
                    ptr::null_mut(),
                ) != CUDA_SUCCESS
                {
                    outputs.push(None);
                    continue;
                }

                let mut path_len: u32 = 0;
                if (self.cuda.cuMemcpyDtoH_v2)(
                    &mut path_len as *mut u32 as *mut c_void,
                    d_path_len,
                    len_bytes,
                ) != CUDA_SUCCESS
                    || path_len == 0
                    || path_len > max_path_len
                {
                    outputs.push(None);
                    continue;
                }

                let n_words = (path_len as usize) * 3;
                let mut raw_coords = vec![0u32; n_words];
                if (self.cuda.cuMemcpyDtoH_v2)(
                    raw_coords.as_mut_ptr() as *mut c_void,
                    d_path_out,
                    n_words * std::mem::size_of::<u32>(),
                ) != CUDA_SUCCESS
                {
                    outputs.push(None);
                    continue;
                }

                let mut path = Vec::with_capacity(path_len as usize);
                for c in raw_coords.chunks_exact(3) {
                    path.push((c[0] as usize, c[1] as usize, c[2] as usize));
                }
                outputs.push(Some(path));
            }

            let _ = (self.cuda.cuCtxSynchronize)();
            let elapsed_ns = start_t.elapsed().as_nanos() as u64;
            self.compute_busy_ns.fetch_add(elapsed_ns, Ordering::Relaxed);
            free_all(
                &self.cuda,
                [
                    d_obs, d_df, d_hist, d_usage, d_costs, d_wf_a, d_wf_b, d_path_out, d_path_len,
                ],
            );
            outputs
        }
    }
}

/// Multi-GPU router pool distributing batched 3D wavefront pathfinding across all available
/// CUDA GPUs (`RTX 3080` + `RTX 3060`) in parallel, with automatic fallback to `VulkanEngine`.
pub struct MultiGpuRouterPool {
    pub cuda_engines: Vec<Arc<CudaEngine>>,
    pub vulkan_fallback: Option<Arc<crate::vulkan::VulkanEngine>>,
}

impl MultiGpuRouterPool {
    pub fn new() -> Self {
        let cuda_engines = CudaEngine::all_devices();
        let vulkan_fallback = crate::vulkan::VulkanEngine::new();
        Self {
            cuda_engines,
            vulkan_fallback,
        }
    }

    pub fn with_vulkan(vulkan: Option<Arc<crate::vulkan::VulkanEngine>>) -> Self {
        let cuda_engines = CudaEngine::all_devices();
        Self {
            cuda_engines,
            vulkan_fallback: vulkan,
        }
    }

    pub fn device_count(&self) -> usize {
        if !self.cuda_engines.is_empty() {
            self.cuda_engines.len()
        } else if self.vulkan_fallback.is_some() {
            1
        } else {
            0
        }
    }

    pub fn solve_batch_3d_wavefront_paths(
        &self,
        obstacle_grid: &[u32],
        gw: usize,
        gh: usize,
        layers: usize,
        requests: &[((usize, usize, usize), (usize, usize, usize), u32, f32)],
    ) -> Vec<Option<Vec<(usize, usize, usize)>>> {
        let reqs_with_clearance: Vec<((usize, usize, usize), (usize, usize, usize), u32, f32, i32)> =
            requests
                .iter()
                .map(|&(src, dst, net_id, via_cost)| (src, dst, net_id, via_cost, 0))
                .collect();
        self.solve_batch_3d_wavefront_paths_with_clearance(
            obstacle_grid,
            gw,
            gh,
            layers,
            &reqs_with_clearance,
        )
    }

    pub fn solve_batch_3d_wavefront_paths_with_clearance(
        &self,
        obstacle_grid: &[u32],
        gw: usize,
        gh: usize,
        layers: usize,
        requests: &[((usize, usize, usize), (usize, usize, usize), u32, f32, i32)],
    ) -> Vec<Option<Vec<(usize, usize, usize)>>> {
        if requests.is_empty() {
            return Vec::new();
        }
        if !self.cuda_engines.is_empty() {
            if self.cuda_engines.len() == 1 || requests.len() == 1 {
                return self.cuda_engines[0].solve_batch_3d_wavefront_paths_with_clearance(
                    obstacle_grid,
                    gw,
                    gh,
                    layers,
                    requests,
                );
            }
            let num_engines = self.cuda_engines.len();
            let mut results: Vec<Option<Vec<(usize, usize, usize)>>> =
                (0..requests.len()).map(|_| None).collect();
            std::thread::scope(|scope| {
                let mut handles = Vec::new();
                for (eng_idx, engine) in self.cuda_engines.iter().enumerate() {
                    let assigned: Vec<(
                        usize,
                        ((usize, usize, usize), (usize, usize, usize), u32, f32, i32),
                    )> = requests
                        .iter()
                        .enumerate()
                        .filter(|&(i, _)| i % num_engines == eng_idx)
                        .map(|(i, r)| (i, *r))
                        .collect();
                    if assigned.is_empty() {
                        continue;
                    }
                    let eng_ref = Arc::clone(engine);
                    handles.push(scope.spawn(move || {
                        let sub_reqs: Vec<(
                            (usize, usize, usize),
                            (usize, usize, usize),
                            u32,
                            f32,
                            i32,
                        )> = assigned.iter().map(|&(_, r)| r).collect();
                        let sub_out = eng_ref.solve_batch_3d_wavefront_paths_with_clearance(
                            obstacle_grid,
                            gw,
                            gh,
                            layers,
                            &sub_reqs,
                        );
                        assigned
                            .into_iter()
                            .map(|(orig_i, _)| orig_i)
                            .zip(sub_out)
                            .collect::<Vec<(usize, Option<Vec<(usize, usize, usize)>>)>>()
                    }));
                }
                for handle in handles {
                    if let Ok(sub_res) = handle.join() {
                        for (orig_i, path_opt) in sub_res {
                            results[orig_i] = path_opt;
                        }
                    }
                }
            });
            return results;
        }
        if let Some(ref vk) = self.vulkan_fallback {
            return vk.solve_batch_3d_wavefront_paths_with_clearance(
                obstacle_grid,
                gw,
                gh,
                layers,
                requests,
            );
        }
        vec![None; requests.len()]
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
        let cuda = match CudaEngine::new_with_device(0) {
            Some(c) => c,
            None => return,
        };

        // 1. First dispatch (32x32) ensures persistent VRAM buffers are allocated
        let mut obs_32 = vec![0u32; 32 * 32];
        obs_32[10 * 32 + 10] = 1;
        let df_32 = cuda.compute_distance_field(&obs_32, 32, 32, 12).expect("32x32 df");
        assert_eq!(df_32[10 * 32 + 10], 0.0);
        assert!((df_32[10 * 32 + 15] - 5.0).abs() < 1e-4);

        let (ptr_32_obs, cap_32_obs, ptr_32_out, cap_32_out) = {
            let b = cuda.persistent_buffers.lock().unwrap_or_else(|e| e.into_inner());
            (b.d_obs, b.obs_cap_bytes, b.d_out, b.out_cap_bytes)
        };
        assert_ne!(ptr_32_obs, 0);
        assert_ne!(ptr_32_out, 0);
        assert!(cap_32_obs >= 32 * 32 * 4);
        assert!(cap_32_out >= 32 * 32 * 4);

        // 2. Smaller dispatch (16x16) must reuse the exact same VRAM pointers and capacity
        let mut obs_16 = vec![0u32; 16 * 16];
        obs_16[4 * 16 + 4] = 1;
        let df_16 = cuda.compute_distance_field(&obs_16, 16, 16, 8).expect("16x16 df");
        assert_eq!(df_16[4 * 16 + 4], 0.0);
        assert!((df_16[4 * 16 + 7] - 3.0).abs() < 1e-4);

        let (ptr_16_obs, cap_16_obs, ptr_16_out, cap_16_out) = {
            let b = cuda.persistent_buffers.lock().unwrap_or_else(|e| e.into_inner());
            (b.d_obs, b.obs_cap_bytes, b.d_out, b.out_cap_bytes)
        };
        assert_eq!(ptr_16_obs, ptr_32_obs, "Persistent d_obs must be reused for smaller grid");
        assert_eq!(ptr_16_out, ptr_32_out, "Persistent d_out must be reused for smaller grid");
        assert_eq!(cap_16_obs, cap_32_obs);
        assert_eq!(cap_16_out, cap_32_out);

        // 3. Larger dispatch (256x256) grows capacity cleanly and subsequent 256x256 calls reuse it
        let mut obs_256 = vec![0u32; 256 * 256];
        obs_256[20 * 256 + 20] = 1;
        let df_256_a = cuda.compute_distance_field(&obs_256, 256, 256, 12).expect("256x256 df A");
        let (ptr_256_obs, cap_256_obs, _ptr_256_out, cap_256_out) = {
            let b = cuda.persistent_buffers.lock().unwrap_or_else(|e| e.into_inner());
            (b.d_obs, b.obs_cap_bytes, b.d_out, b.out_cap_bytes)
        };
        assert_eq!(cap_256_obs, 256 * 256 * 4);
        assert_eq!(cap_256_out, 256 * 256 * 4);

        let df_256_b = cuda.compute_distance_field(&obs_256, 256, 256, 12).expect("256x256 df B");
        let (ptr_256b_obs, cap_256b_obs, cap_256b_out) = {
            let b = cuda.persistent_buffers.lock().unwrap_or_else(|e| e.into_inner());
            (b.d_obs, b.obs_cap_bytes, b.out_cap_bytes)
        };
        assert_eq!(ptr_256b_obs, ptr_256_obs, "Persistent d_obs must be reused on equal size");
        assert_eq!(cap_256b_obs, 256 * 256 * 4);
        assert_eq!(cap_256b_out, 256 * 256 * 4);
        assert_eq!(df_256_a, df_256_b);
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
        let vk = match crate::vulkan::VulkanEngine::new() {
            Some(v) => v,
            None => return,
        };
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
        let vk = match crate::vulkan::VulkanEngine::new() {
            Some(v) => v,
            None => return,
        };
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

    #[test]
    fn test_tensor_core_kernels_exact_equivalence() {
        let devs = CudaEngine::all_devices();
        if devs.is_empty() {
            return;
        }

        for cuda in &devs {
            // 1. Distance Field Tensor Core vs SIMT Tiled (both empty and populated tiles)
            let w = 48u32;
            let h = 48u32;
            let r = 10i32;
            let mut obs = vec![0u32; (w * h) as usize];
            obs[6 * (w as usize) + 6] = 1;
            obs[35 * (w as usize) + 40] = 2;

            let df_ref = cuda
                .compute_distance_field_variant(&obs, w, h, r, DistanceFieldVariant::SimtSharedTiled)
                .expect("SIMT tiled DF");
            let df_tc = cuda
                .compute_distance_field_tensor_core(&obs, w, h, r)
                .expect("Tensor Core DF");
            assert_eq!(df_ref.len(), df_tc.len());
            for i in 0..df_ref.len() {
                assert!(
                    (df_ref[i] - df_tc[i]).abs() < 1e-3,
                    "[{}] DF TensorCore mismatch at {}: ref={}, tc={}",
                    cuda.device_name,
                    i,
                    df_ref[i],
                    df_tc[i]
                );
            }

            // 2. Batch Placement Tensor Core vs SIMT Standard
            let pe_params = PlacementBatchParams {
                num_candidates: 16,
                num_footprints: 8,
                num_segments: 12,
                _pad: 0,
                board_min_x: 0.0,
                board_min_y: 0.0,
                board_max_x: 50.0,
                board_max_y: 40.0,
                wirelength_weight: 1.0,
                crossing_weight: 200.0,
                overlap_penalty: 25000.0,
                boundary_penalty: 50000.0,
            };
            let mut boxes = Vec::with_capacity(16 * 8);
            for c in 0..16u32 {
                for f in 0..8u32 {
                    let bx = ((f * 9 + c * 3) % 45) as f32 - 1.0;
                    let by = ((f * 7 + c * 5) % 35) as f32 + 1.0;
                    boxes.push([bx, by, bx + 4.5, by + 3.5]);
                }
            }
            let mut segs = Vec::with_capacity(16 * 12);
            for c in 0..16u32 {
                for s in 0..12u32 {
                    let x1 = ((s * 7 + c * 2) % 40) as f32 + 2.0;
                    let y1 = ((s * 5 + c * 3) % 30) as f32 + 2.0;
                    let x2 = ((s * 11 + c * 7 + 12) % 40) as f32 + 2.0;
                    let y2 = ((s * 13 + c * 5 + 8) % 30) as f32 + 2.0;
                    segs.push([x1, y1, x2, y2]);
                }
            }

            let pe_ref = cuda
                .evaluate_placement_variant(&boxes, &segs, &pe_params, PlacementEvalVariant::SimtStandard)
                .expect("SIMT Placement Eval");
            let pe_tc = cuda
                .evaluate_placement_tensor_core(&boxes, &segs, &pe_params)
                .expect("Tensor Core Placement Eval");
            assert_eq!(pe_ref.len(), pe_tc.len());
            for i in 0..pe_ref.len() {
                assert!(
                    (pe_ref[i] - pe_tc[i]).abs() <= 1e-2 * pe_ref[i].abs().max(1.0),
                    "[{}] Placement TensorCore mismatch at {}: ref={}, tc={}",
                    cuda.device_name,
                    i,
                    pe_ref[i],
                    pe_tc[i]
                );
            }

            // 3. 3D Routing Cost Tensor Core vs SIMT Shared Tiled
            let rw = 32u32;
            let rh = 32u32;
            let rnl = 4u32;
            let rtot = (rw * rh * rnl) as usize;
            let mut robs = vec![0u32; rtot];
            let mut rdf = vec![1.5f32; rtot];
            let mut rhist = vec![0.0f32; rtot];
            let mut ruse = vec![0u32; rtot];
            for i in 0..rtot {
                if i % 41 == 0 {
                    robs[i] = ((i % 5) + 1) as u32;
                }
                rdf[i] = 0.3 + ((i % 9) as f32) * 0.2;
                rhist[i] = ((i % 6) as f32) * 0.3;
                ruse[i] = (i % 4) as u32;
            }
            let rparams = GpuRouteCostParams {
                width: rw,
                height: rh,
                num_layers: rnl,
                net_id: 2,
                clearance_cells: 3,
                present_congestion_factor: 1.25,
                obstacle_clearance_weight: 0.4,
                plane_layer_mask: 0b0100,
            };

            let rc_ref = cuda
                .compute_3d_routing_costs_variant(
                    &robs,
                    &rdf,
                    &rhist,
                    &ruse,
                    &rparams,
                    RoutingCost3DVariant::SimtSharedTiled,
                )
                .expect("SIMT 3D Routing Cost");
            let rc_tc = cuda
                .compute_3d_routing_costs_tensor_core(&robs, &rdf, &rhist, &ruse, &rparams)
                .expect("Tensor Core 3D Routing Cost");
            assert_eq!(rc_ref.len(), rc_tc.len());
            for i in 0..rtot {
                assert!(
                    (rc_ref[i] - rc_tc[i]).abs() < 1e-4,
                    "[{}] 3D Routing Cost TensorCore mismatch at {}: ref={}, tc={}",
                    cuda.device_name,
                    i,
                    rc_ref[i],
                    rc_tc[i]
                );
            }
        }
    }

    #[test]
    fn test_startup_kernel_benchmark_and_auto_dispatcher() {
        let devs = CudaEngine::all_devices();
        if devs.is_empty() {
            return;
        }

        for dev in &devs {
            let p = &dev.dispatch_profile;
            println!(
                "Auto-Dispatch Profile for {}: DF={:?} {:?}, PE={:?} {:?}, RC3D={:?} {:?}, SA={:?} {:?}",
                p.device_name,
                p.best_distance_field,
                p.df_timings_us,
                p.best_placement_eval,
                p.pe_timings_us,
                p.best_routing_cost_3d,
                p.rc_timings_us,
                p.best_sa_placer,
                p.sa_timings_us
            );
            assert_eq!(p.device_name, dev.device_name);
            assert_eq!(p.df_timings_us.len(), 5, "All 5 DistanceFieldVariants must be benchmarked");
            assert_eq!(p.pe_timings_us.len(), 3, "All 3 PlacementEvalVariants must be benchmarked");
            assert_eq!(p.rc_timings_us.len(), 2, "Both RoutingCost3DVariants must be benchmarked");
            assert_eq!(p.sa_timings_us.len(), 3, "All 3 SaPlacerVariants must be benchmarked");

            for &(_, us) in &p.df_timings_us {
                assert!(us > 0.0 && us.is_finite());
            }
            for &(_, us) in &p.pe_timings_us {
                assert!(us > 0.0 && us.is_finite());
            }
            for &(_, us) in &p.rc_timings_us {
                assert!(us > 0.0 && us.is_finite());
            }
            for &(_, us) in &p.sa_timings_us {
                assert!(us > 0.0 && us.is_finite());
            }

            // Verify selected best variant matches the minimum recorded median time
            let min_df = p
                .df_timings_us
                .iter()
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();
            assert_eq!(p.best_distance_field, min_df.0);

            let min_pe = p
                .pe_timings_us
                .iter()
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();
            assert_eq!(p.best_placement_eval, min_pe.0);

            let min_rc = p
                .rc_timings_us
                .iter()
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();
            assert_eq!(p.best_routing_cost_3d, min_rc.0);

            let min_sa = p
                .sa_timings_us
                .iter()
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();
            assert_eq!(p.best_sa_placer, min_sa.0);
        }

        // Verify process-wide cache hits immediately on subsequent initialization
        let devs_cached = CudaEngine::all_devices();
        assert_eq!(devs.len(), devs_cached.len());
        for (a, b) in devs.iter().zip(devs_cached.iter()) {
            assert_eq!(a.dispatch_profile.best_distance_field, b.dispatch_profile.best_distance_field);
            assert_eq!(a.dispatch_profile.best_placement_eval, b.dispatch_profile.best_placement_eval);
            assert_eq!(a.dispatch_profile.best_routing_cost_3d, b.dispatch_profile.best_routing_cost_3d);
            assert_eq!(a.dispatch_profile.best_sa_placer, b.dispatch_profile.best_sa_placer);
            assert_eq!(a.dispatch_profile.df_timings_us, b.dispatch_profile.df_timings_us);
            assert_eq!(a.dispatch_profile.sa_timings_us, b.dispatch_profile.sa_timings_us);
        }
    }

    #[test]
    fn test_gpu_sa_placer_variants_equivalence_and_speedup() {
        let devs = CudaEngine::all_devices();
        if devs.is_empty() {
            return;
        }

        let sa_nc = 64u32;
        let sa_nf = 16u32;
        let sa_ne = 24u32;
        let sa_params = GpuSaParams {
            num_chains: sa_nc,
            num_footprints: sa_nf,
            num_edges: sa_ne,
            steps_per_pass: 16,
            seed: 0xC0FFEE42,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            board_min_x: 0.0,
            board_min_y: 0.0,
            board_max_x: 80.0,
            board_max_y: 60.0,
            grid_step: 0.5,
            wirelength_weight: 1.0,
            crossing_weight: 200.0,
            overlap_penalty: 25000.0,
            boundary_penalty: 50000.0,
            passive_affinity_weight: 80.0,
            passive_max_dist: 3.5,
            base_temp: 5.0,
        };

        let mut sa_fp_descs = Vec::with_capacity(sa_nf as usize);
        for f in 0..sa_nf {
            let locked = if f == 0 { 1.0 } else { 0.0 };
            let fine_pitch = if f == 1 { 1.0 } else { 0.0 };
            sa_fp_descs.push([1.5, 1.2, locked, fine_pitch]);
        }

        let mut sa_edges = Vec::with_capacity((sa_ne * 2) as usize);
        for e in 0..sa_ne {
            let fa = (e % sa_nf) as f32;
            let fb = ((e * 3 + 1) % sa_nf) as f32;
            let fa_safe = if (fa - fb).abs() < 0.5 { ((e + 2) % sa_nf) as f32 } else { fa };
            let tag = if e >= 20 { 2.0 } else if e >= 16 { 1.0 } else { 0.0 };
            sa_edges.push([fa_safe, -0.5, 0.2, tag]);
            sa_edges.push([fb, 0.5, -0.2, tag]);
        }

        let mut sa_poses = Vec::with_capacity((sa_nc * sa_nf) as usize);
        for c in 0..sa_nc {
            for f in 0..sa_nf {
                let px = if f == 0 {
                    10.0
                } else {
                    10.0 + ((f * 9 + c * 3) % 60) as f32
                };
                let py = if f == 0 {
                    10.0
                } else {
                    10.0 + ((f * 7 + c * 5) % 40) as f32
                };
                sa_poses.push([px, py, ((f % 4) * 90) as f32, 0.0]);
            }
        }

        for dev in &devs {
            // Also verify 0-step initial energy evaluation matches across all 3 variants
            let mut zero_params = sa_params;
            zero_params.steps_per_pass = 0;
            let (_, scores_full_0) = dev
                .run_gpu_sa_placer_chains_variant(&sa_fp_descs, &sa_edges, &sa_poses, &zero_params, 1, SaPlacerVariant::SimtFullEval)
                .expect("SimtFullEval 0-step");
            let (_, scores_de_0) = dev
                .run_gpu_sa_placer_chains_variant(&sa_fp_descs, &sa_edges, &sa_poses, &zero_params, 1, SaPlacerVariant::SimtIncrementalDeltaE)
                .expect("SimtIncrementalDeltaE 0-step");
            let (_, scores_tc_0) = dev
                .run_gpu_sa_placer_chains_variant(&sa_fp_descs, &sa_edges, &sa_poses, &zero_params, 1, SaPlacerVariant::TensorCore)
                .expect("TensorCore 0-step");

            for c in 0..(sa_nc as usize) {
                assert!(
                    (scores_full_0[c][0] - scores_de_0[c][0]).abs() < 1e-2,
                    "[{}] 0-step energy mismatch Full vs DeltaE on chain {}: {} vs {}",
                    dev.device_name,
                    c,
                    scores_full_0[c][0],
                    scores_de_0[c][0]
                );
                assert!(
                    (scores_full_0[c][0] - scores_tc_0[c][0]).abs() <= 1e-2 * scores_full_0[c][0].abs().max(1.0),
                    "[{}] 0-step energy mismatch Full vs TensorCore on chain {}: {} vs {}",
                    dev.device_name,
                    c,
                    scores_full_0[c][0],
                    scores_tc_0[c][0]
                );
            }

            // Now run multi-pass MCMC annealing with each variant
            for &variant in &[
                SaPlacerVariant::SimtFullEval,
                SaPlacerVariant::SimtIncrementalDeltaE,
                SaPlacerVariant::TensorCore,
            ] {
                let t0 = std::time::Instant::now();
                let (poses_out, scores_out) = dev
                    .run_gpu_sa_placer_chains_variant(&sa_fp_descs, &sa_edges, &sa_poses, &sa_params, 4, variant)
                    .expect("SA placer variant execution failed");
                let elapsed_us = t0.elapsed().as_secs_f64() * 1e6;

                let min_best = scores_out.iter().map(|s| s[1]).fold(f32::INFINITY, f32::min);
                println!(
                    "[{}] {:?}: {:.1} us, min_best_energy = {:.2}",
                    dev.device_name, variant, elapsed_us, min_best
                );

                let nc = sa_nc as usize;
                let nf = sa_nf as usize;
                assert_eq!(poses_out.len(), nc * nf);
                for c in 0..nc {
                    let init_e = scores_full_0[c][0];
                    let cur_e = scores_out[c][0];
                    let best_e = scores_out[c][1];
                    assert!(
                        cur_e.is_finite() && cur_e > 0.0,
                        "Chain {} current energy must be finite positive for {:?}",
                        c,
                        variant
                    );
                    assert!(
                        best_e.is_finite() && best_e <= init_e + 1.0,
                        "Chain {} best_energy ({}) must be <= initial energy ({}) for {:?}",
                        c,
                        best_e,
                        init_e,
                        variant
                    );
                    // Locked footprint 0 in best poses must remain at (10.0, 10.0)
                    let fp0_best = poses_out[c * nf];
                    assert!(
                        (fp0_best[0] - 10.0).abs() < 1e-4 && (fp0_best[1] - 10.0).abs() < 1e-4,
                        "Locked best footprint 0 moved on {:?}",
                        variant
                    );
                }
            }
        }
    }

    #[test]
    fn test_gpu_batch_drc_eval_via_and_segment_clearance() {
        let cuda = match CudaEngine::new() {
            Some(c) => c,
            None => return,
        };

        let pads = vec![
            // Foreign SMD pad on F.Cu (layer 0, mask = 1) at (10.0, 10.0), size 1.0 x 1.0 (hw=0.5, hh=0.5), net 2
            GpuDrcPad {
                x: 10.0,
                y: 10.0,
                hw: 0.5,
                hh: 0.5,
                rot_rad: 0.0,
                clearance: 0.20,
                layer_mask: 1,
                net_id: 2,
            },
            // Same-net SMD pad on F.Cu at (20.0, 10.0), size 1.0 x 1.0, net 1
            GpuDrcPad {
                x: 20.0,
                y: 10.0,
                hw: 0.5,
                hh: 0.5,
                rot_rad: 0.0,
                clearance: 0.20,
                layer_mask: 1,
                net_id: 1,
            },
        ];

        let tracks = vec![
            // Foreign track on F.Cu (layer 0) from (5.0, 20.0) to (15.0, 20.0), half_width=0.125, net 3
            GpuDrcTrack {
                x1: 5.0,
                y1: 20.0,
                x2: 15.0,
                y2: 20.0,
                half_width: 0.125,
                clearance: 0.20,
                layer: 0,
                net_id: 3,
            },
        ];

        let vias = vec![
            // Foreign via at (30.0, 30.0), radius=0.30, net 4
            GpuDrcVia {
                x: 30.0,
                y: 30.0,
                radius: 0.30,
                clearance: 0.20,
                net_id: 4,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            },
        ];

        let candidates = vec![
            // 0: Legal via at (2.0, 2.0), net 1
            GpuDrcCandidate {
                x1: 2.0,
                y1: 2.0,
                x2: 0.0,
                y2: 0.0,
                radius_or_half_width: 0.30,
                clearance: 0.20,
                layer: 0,
                net_id: 1,
                kind: 0,
                ignore_same_net_pads: 0,
                _pad0: 0,
                _pad1: 0,
            },
            // 1: Illegal via too close to foreign pad at (10.8, 10.0) -> dist to pad box is 0.3 < 0.30 + 0.20 = 0.50
            GpuDrcCandidate {
                x1: 10.8,
                y1: 10.0,
                x2: 0.0,
                y2: 0.0,
                radius_or_half_width: 0.30,
                clearance: 0.20,
                layer: 0,
                net_id: 1,
                kind: 0,
                ignore_same_net_pads: 0,
                _pad0: 0,
                _pad1: 0,
            },
            // 2: Illegal via overlapping same-net SMD pad at (20.2, 10.0) when ignore_same_net_pads == 0
            GpuDrcCandidate {
                x1: 20.2,
                y1: 10.0,
                x2: 0.0,
                y2: 0.0,
                radius_or_half_width: 0.30,
                clearance: 0.20,
                layer: 0,
                net_id: 1,
                kind: 0,
                ignore_same_net_pads: 0,
                _pad0: 0,
                _pad1: 0,
            },
            // 3: Illegal segment crossing foreign track on layer 0
            GpuDrcCandidate {
                x1: 10.0,
                y1: 18.0,
                x2: 10.0,
                y2: 22.0,
                radius_or_half_width: 0.125,
                clearance: 0.20,
                layer: 0,
                net_id: 1,
                kind: 1,
                ignore_same_net_pads: 0,
                _pad0: 0,
                _pad1: 0,
            },
            // 4: Legal segment crossing foreign track's XY on layer 1 (B.Cu)
            GpuDrcCandidate {
                x1: 10.0,
                y1: 18.0,
                x2: 10.0,
                y2: 22.0,
                radius_or_half_width: 0.125,
                clearance: 0.20,
                layer: 1,
                net_id: 1,
                kind: 1,
                ignore_same_net_pads: 0,
                _pad0: 0,
                _pad1: 0,
            },
            // 5: Illegal segment too close to foreign via at (30.0, 30.0) even on layer 1 (vias span all layers)
            GpuDrcCandidate {
                x1: 28.0,
                y1: 30.3,
                x2: 32.0,
                y2: 30.3,
                radius_or_half_width: 0.125,
                clearance: 0.20,
                layer: 1,
                net_id: 1,
                kind: 1,
                ignore_same_net_pads: 0,
                _pad0: 0,
                _pad1: 0,
            },
        ];

        let res = cuda
            .evaluate_drc_candidates(&pads, &tracks, &vias, &candidates)
            .expect("GPU DRC eval");
        assert_eq!(res, vec![1, 0, 0, 0, 1, 0]);
    }

    #[test]
    fn test_cuda_vs_vulkan_batch_drc_equivalence() {
        let cuda = match CudaEngine::new() {
            Some(c) => c,
            None => return,
        };
        let vk = match crate::vulkan::VulkanEngine::new_vulkan_only() {
            Some(v) => v,
            None => return,
        };

        let pads = vec![
            GpuDrcPad {
                x: 12.0,
                y: 12.0,
                hw: 0.6,
                hh: 0.4,
                rot_rad: 0.7853982, // 45 deg
                clearance: 0.20,
                layer_mask: 0xFFFF_FFFF,
                net_id: 2,
            },
        ];
        let tracks = vec![
            GpuDrcTrack {
                x1: 0.0,
                y1: 5.0,
                x2: 20.0,
                y2: 5.0,
                half_width: 0.125,
                clearance: 0.20,
                layer: 0,
                net_id: 3,
            },
        ];
        let vias = vec![
            GpuDrcVia {
                x: 8.0,
                y: 8.0,
                radius: 0.30,
                clearance: 0.20,
                net_id: 4,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            },
        ];

        // Generate 200 candidates (>= 128 to exercise multi-GPU splitting on CUDA as well)
        let mut candidates = Vec::with_capacity(200);
        for i in 0..200 {
            let x = (i % 20) as f32 * 1.0 + 1.0;
            let y = (i / 20) as f32 * 2.0 + 1.0;
            candidates.push(GpuDrcCandidate {
                x1: x,
                y1: y,
                x2: x + 1.5,
                y2: y + 0.5,
                radius_or_half_width: if i % 2 == 0 { 0.30 } else { 0.125 },
                clearance: 0.20,
                layer: (i % 2) as u32,
                net_id: 1,
                kind: (i % 2) as u32,
                ignore_same_net_pads: 0,
                _pad0: 0,
                _pad1: 0,
            });
        }

        let cu_res = cuda
            .evaluate_drc_candidates(&pads, &tracks, &vias, &candidates)
            .expect("CUDA DRC batch");
        let vk_res = vk
            .evaluate_drc_candidates_vulkan(&pads, &tracks, &vias, &candidates)
            .expect("Vulkan DRC batch");
        assert_eq!(cu_res.len(), 200);
        assert_eq!(cu_res, vk_res, "CUDA and Vulkan WGSL DRC batch kernels must return identical results");
    }

    #[test]
    fn test_rt_bvh_corridor_cuts_cuda_and_vulkan_equivalence() {
        let cuda = match CudaEngine::new() {
            Some(c) => c,
            None => return,
        };
        let vk = match crate::vulkan::VulkanEngine::new_vulkan_only() {
            Some(v) => v,
            None => return,
        };

        let num_segs = 320usize;
        let num_cuts = 180usize;
        let mut segs = Vec::with_capacity(num_segs);
        for i in 0..num_segs {
            let x1 = ((i * 31) % 100) as f32 + (i as f32) * 0.013;
            let y1 = ((i * 47) % 100) as f32 + (i as f32) * 0.017;
            let x2 = ((i * 59 + 17) % 100) as f32 + (i as f32) * 0.019;
            let y2 = ((i * 79 + 23) % 100) as f32 + (i as f32) * 0.011;
            segs.push([x1, y1, x2, y2]);
        }

        let mut cuts = Vec::with_capacity(num_cuts);
        for j in 0..num_cuts {
            let x1 = ((j * 23 + 7) % 95) as f32 + (j as f32) * 0.021;
            let y1 = ((j * 41 + 11) % 95) as f32 + (j as f32) * 0.015;
            let x2 = ((j * 67 + 29) % 95) as f32 + (j as f32) * 0.023;
            let y2 = ((j * 83 + 37) % 95) as f32 + (j as f32) * 0.027;
            cuts.push([x1, y1, x2, y2]);
        }

        let bvh_nodes = build_gpu_lbvh_from_segments(&segs);
        let cpu_brute = evaluate_corridor_cuts_brute_force_cpu(&segs, &cuts);
        let cpu_lbvh = evaluate_corridor_cuts_lbvh_cpu(&segs, &bvh_nodes, &cuts);
        let cu_cuts = cuda
            .evaluate_corridor_cuts_rt_bvh(&segs, &bvh_nodes, &cuts)
            .expect("CUDA BVH corridor cuts");
        let vk_cuts = vk
            .evaluate_corridor_cuts_rt_bvh_vulkan_wgsl(&segs, &bvh_nodes, &cuts)
            .expect("Vulkan WGSL BVH corridor cuts");

        assert_eq!(cpu_lbvh, cpu_brute);
        assert_eq!(cu_cuts, cpu_brute);
        assert_eq!(vk_cuts, cpu_brute);
    }

    #[test]
    fn test_gpu_3d_wavefront_backtrack_path_valid_and_obstacle_free() {
        let cuda = match CudaEngine::new() {
            Some(c) => c,
            None => return,
        };

        let gw = 20usize;
        let gh = 20usize;
        let layers = 2usize;
        let layer_stride = gw * gh;
        let mut obs = vec![0u32; gw * gh * layers];

        // Build a complete vertical obstacle wall on Layer 0 at x = 10 for foreign net 9,
        // forcing a 3D via transition onto Layer 1 or routing around a partial wall.
        // For request 1 (net 1): partial wall y = 0..15 at x = 8 on both layers -> must detour via y >= 15.
        for l in 0..layers {
            for y in 0..15 {
                obs[l * layer_stride + y * gw + 8] = 9;
            }
        }
        // For request 2 (net 2): full wall on Layer 0 at x = 14 (y = 0..20), clear on Layer 1 -> must via!
        for y in 0..gh {
            obs[0 * layer_stride + y * gw + 14] = 9;
        }

        let src1 = (2usize, 5usize, 0usize);
        let dst1 = (12usize, 5usize, 0usize);
        let path1 = cuda
            .solve_3d_wavefront_path(&obs, gw, gh, layers, src1, dst1, 1, 3.0)
            .expect("CUDA 3D wavefront must find detour path around obstacle wall");

        assert_eq!(path1.first().copied(), Some(src1));
        assert_eq!(path1.last().copied(), Some(dst1));
        for w in path1.windows(2) {
            let (x0, y0, l0) = w[0];
            let (x1, y1, l1) = w[1];
            let dx = (x1 as isize - x0 as isize).abs();
            let dy = (y1 as isize - y0 as isize).abs();
            let dl = (l1 as isize - l0 as isize).abs();
            assert!(
                (dl == 0 && dx <= 1 && dy <= 1 && (dx + dy > 0)) || (dl > 0 && dx == 0 && dy == 0),
                "Invalid step {:?} -> {:?}",
                w[0],
                w[1]
            );
        }
        for &(gx, gy, l) in &path1 {
            let cell_obs = obs[l * layer_stride + gy * gw + gx];
            assert!(
                cell_obs == 0 || cell_obs == 1,
                "Path stepped onto foreign obstacle {} at ({}, {}, {})",
                cell_obs,
                gx,
                gy,
                l
            );
        }

        // Verify MultiGpuRouterPool batch solving across all available GPUs
        let pool = MultiGpuRouterPool::new();
        assert!(pool.device_count() >= 1);
        let requests = vec![
            (src1, dst1, 1u32, 3.0f32),
            ((11usize, 10usize, 0usize), (17usize, 10usize, 0usize), 2u32, 2.5f32),
            ((0usize, 0usize, 0usize), (8usize, 5usize, 0usize), 3u32, 3.0f32), // dst is inside foreign obstacle 9 -> None
        ];
        let batch_paths = pool.solve_batch_3d_wavefront_paths(&obs, gw, gh, layers, &requests);
        assert_eq!(batch_paths.len(), 3);
        assert_eq!(batch_paths[0].as_ref(), Some(&path1));

        let path2 = batch_paths[1].as_ref().expect("Net 2 must bridge wall on L0 via L1");
        assert_eq!(path2.first().copied(), Some((11, 10, 0)));
        assert_eq!(path2.last().copied(), Some((17, 10, 0)));
        assert!(
            path2.iter().any(|&(_, _, l)| l == 1),
            "Net 2 must transition to layer 1 to cross full L0 wall"
        );
        for &(gx, gy, l) in path2 {
            let cell_obs = obs[l * layer_stride + gy * gw + gx];
            assert!(cell_obs == 0 || cell_obs == 2);
        }

        assert!(batch_paths[2].is_none(), "Destination on foreign obstacle must return None");
    }
}


