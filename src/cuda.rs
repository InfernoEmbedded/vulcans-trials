// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials

use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::vulkan::{GpuSaParams, PlacementBatchParams};

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
        let lib_paths = ["libcuda.so.1", "libcuda.so"];
        for path in &lib_paths {
            if let Ok(lib) = unsafe { libloading::Library::new(path) } {
                unsafe {
                    let cuInit = match lib.get(b"cuInit\0") {
                        Ok(sym) => *sym,
                        Err(_) => continue,
                    };
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
    }
}

pub struct CudaEngine {
    pub device_name: String,
    pub backend_name: String,
    pub driver_info: String,
    ctx: CUcontext,
    module: CUmodule,
    ampere_module: CUmodule,
    fn_dist_field: CUfunction,
    fn_dist_field_tiled: CUfunction,
    fn_batch_placement: CUfunction,
    fn_gpu_sa_placer: CUfunction,
    pub fn_df_ampere: Option<CUfunction>,
    pub fn_df_ampere_tiled: Option<CUfunction>,
    pub fn_bp_ampere: Option<CUfunction>,
    cuda: Arc<CudaDriverApi>,
    dispatch_lock: Mutex<()>,
    pub compute_busy_ns: Arc<AtomicU64>,
}

unsafe impl Send for CudaEngine {}
unsafe impl Sync for CudaEngine {}

impl std::fmt::Debug for CudaEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaEngine")
            .field("device_name", &self.device_name)
            .field("backend_name", &self.backend_name)
            .finish()
    }
}

impl Drop for CudaEngine {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.cuda.cuCtxSetCurrent)(self.ctx);
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
    pub fn new() -> Option<Arc<Self>> {
        Self::new_with_device(0)
    }

    pub fn new_with_device(ordinal: c_int) -> Option<Arc<Self>> {
        let api = CudaDriverApi::load()?;
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
            let mut fn_batch_placement: CUfunction = ptr::null_mut();
            let mut fn_gpu_sa_placer: CUfunction = ptr::null_mut();

            let c_df = CString::new("distance_field_kernel").unwrap();
            let c_dft = CString::new("distance_field_tiled_kernel").unwrap();
            let c_bp = CString::new("batch_placement_eval_kernel").unwrap();
            let c_sa = CString::new("gpu_sa_placer_kernel").unwrap();

            if (api.cuModuleGetFunction)(&mut fn_dist_field, module, c_df.as_ptr()) != CUDA_SUCCESS
                || (api.cuModuleGetFunction)(&mut fn_dist_field_tiled, module, c_dft.as_ptr()) != CUDA_SUCCESS
                || (api.cuModuleGetFunction)(&mut fn_batch_placement, module, c_bp.as_ptr()) != CUDA_SUCCESS
                || (api.cuModuleGetFunction)(&mut fn_gpu_sa_placer, module, c_sa.as_ptr()) != CUDA_SUCCESS
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

            Some(Arc::new(Self {
                device_name,
                backend_name,
                driver_info,
                ctx,
                module,
                ampere_module,
                fn_dist_field,
                fn_dist_field_tiled,
                fn_batch_placement,
                fn_gpu_sa_placer,
                fn_df_ampere,
                fn_df_ampere_tiled,
                fn_bp_ampere,
                cuda: api,
                dispatch_lock: Mutex::new(()),
                compute_busy_ns: Arc::new(AtomicU64::new(0)),
            }))
        }
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

            let mut d_obs: CUdeviceptr = 0;
            let mut d_out: CUdeviceptr = 0;

            if (self.cuda.cuMemAlloc_v2)(&mut d_obs, in_bytes) != CUDA_SUCCESS {
                return None;
            }
            if (self.cuda.cuMemAlloc_v2)(&mut d_out, out_bytes) != CUDA_SUCCESS {
                let _ = (self.cuda.cuMemFree_v2)(d_obs);
                return None;
            }

            let start_t = Instant::now();

            if (self.cuda.cuMemcpyHtoD_v2)(d_obs, obstacles.as_ptr() as *const c_void, in_bytes) != CUDA_SUCCESS {
                let _ = (self.cuda.cuMemFree_v2)(d_obs);
                let _ = (self.cuda.cuMemFree_v2)(d_out);
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
                    let _ = (self.cuda.cuMemFree_v2)(d_obs);
                    let _ = (self.cuda.cuMemFree_v2)(d_out);
                    return None;
                }
            }

            let mut result = vec![0.0f32; total];
            if (self.cuda.cuMemcpyDtoH_v2)(result.as_mut_ptr() as *mut c_void, d_out, out_bytes) != CUDA_SUCCESS {
                let _ = (self.cuda.cuMemFree_v2)(d_obs);
                let _ = (self.cuda.cuMemFree_v2)(d_out);
                return None;
            }

            let _ = (self.cuda.cuCtxSynchronize)();
            let elapsed_ns = start_t.elapsed().as_nanos() as u64;
            self.compute_busy_ns.fetch_add(elapsed_ns, Ordering::Relaxed);

            let _ = (self.cuda.cuMemFree_v2)(d_obs);
            let _ = (self.cuda.cuMemFree_v2)(d_out);

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

            let mut d_obs: CUdeviceptr = 0;
            let mut d_out: CUdeviceptr = 0;

            if (self.cuda.cuMemAlloc_v2)(&mut d_obs, in_bytes) != CUDA_SUCCESS {
                return None;
            }
            if (self.cuda.cuMemAlloc_v2)(&mut d_out, out_bytes) != CUDA_SUCCESS {
                let _ = (self.cuda.cuMemFree_v2)(d_obs);
                return None;
            }

            let start_t = Instant::now();

            if (self.cuda.cuMemcpyHtoD_v2)(d_obs, obstacles.as_ptr() as *const c_void, in_bytes) != CUDA_SUCCESS {
                let _ = (self.cuda.cuMemFree_v2)(d_obs);
                let _ = (self.cuda.cuMemFree_v2)(d_out);
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
                    let _ = (self.cuda.cuMemFree_v2)(d_obs);
                    let _ = (self.cuda.cuMemFree_v2)(d_out);
                    return None;
                }
            }

            let mut result = vec![0.0f32; total];
            if (self.cuda.cuMemcpyDtoH_v2)(result.as_mut_ptr() as *mut c_void, d_out, out_bytes) != CUDA_SUCCESS {
                let _ = (self.cuda.cuMemFree_v2)(d_obs);
                let _ = (self.cuda.cuMemFree_v2)(d_out);
                return None;
            }

            let _ = (self.cuda.cuCtxSynchronize)();
            let elapsed_ns = start_t.elapsed().as_nanos() as u64;
            self.compute_busy_ns.fetch_add(elapsed_ns, Ordering::Relaxed);

            let _ = (self.cuda.cuMemFree_v2)(d_obs);
            let _ = (self.cuda.cuMemFree_v2)(d_out);

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
}
