// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials

use std::fs;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

const DISTANCE_FIELD_WGSL: &str = r#"
struct GridParams {
    width: u32,
    height: u32,
    radius: i32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> obstacles: array<u32>;
@group(0) @binding(1) var<storage, read_write> dist_out: array<f32>;
@group(0) @binding(2) var<uniform> params: GridParams;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    let w = i32(params.width);
    let h = i32(params.height);
    if (x >= w || y >= h) {
        return;
    }

    let idx = u32(y * w + x);
    if (obstacles[idx] != 0u) {
        dist_out[idx] = 0.0;
        return;
    }

    let r = params.radius;
    var min_d2: f32 = f32(r * r + 1);
    for (var dy: i32 = -r; dy <= r; dy = dy + 1) {
        let ny = y + dy;
        if (ny < 0 || ny >= h) {
            continue;
        }
        for (var dx: i32 = -r; dx <= r; dx = dx + 1) {
            let nx = x + dx;
            if (nx < 0 || nx >= w) {
                continue;
            }
            let nidx = u32(ny * w + nx);
            if (obstacles[nidx] != 0u) {
                let d2 = f32(dx * dx + dy * dy);
                if (d2 < min_d2) {
                    min_d2 = d2;
                }
            }
        }
    }
    dist_out[idx] = sqrt(min_d2);
}
"#;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GridParams {
    width: u32,
    height: u32,
    radius: i32,
    _pad: u32,
}

pub struct VulkanEngine {
    pub device_name: String,
    pub backend_name: String,
    pub driver_info: String,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    pub compute_busy_ns: Arc<AtomicU64>,
}

impl VulkanEngine {
    pub fn new() -> Option<Arc<Self>> {
        pollster::block_on(Self::init_async()).map(Arc::new)
    }

    async fn init_async() -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });

        let mut adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await;

        if adapter.is_none() {
            let fallback_inst = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });
            adapter = fallback_inst
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await;
        }

        let adapter = adapter?;
        let info = adapter.get_info();
        let device_name = info.name.clone();
        let backend_name = format!("{:?}", info.backend);
        let driver_info = if info.driver_info.is_empty() {
            info.driver.clone()
        } else {
            format!("{} ({})", info.driver, info.driver_info)
        };

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("VulcansTrials Vulkan Compute Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .ok()?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vulkan Distance Field Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(DISTANCE_FIELD_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DistanceField BindGroupLayout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DistanceField PipelineLayout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("DistanceField ComputePipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Some(Self {
            device_name,
            backend_name,
            driver_info,
            device,
            queue,
            pipeline,
            bind_group_layout,
            compute_busy_ns: Arc::new(AtomicU64::new(0)),
        })
    }

    pub fn compute_distance_field(&self, obstacles: &[u32], width: u32, height: u32, radius: i32) -> Option<Vec<f32>> {
        let start_t = Instant::now();
        let total = (width * height) as usize;
        if obstacles.len() != total || total == 0 {
            return None;
        }
        let byte_size = (total * std::mem::size_of::<f32>()) as u64;

        let obs_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Obstacles Buffer"),
            contents: bytemuck::cast_slice(obstacles),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dist Out Buffer"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params = GridParams {
            width,
            height,
            radius,
            _pad: 0,
        };
        let param_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GridParams Uniform"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DistanceField BindGroup"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: obs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: param_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("DistanceField CommandEncoder"),
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("DistanceField ComputePass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let wg_x = (width + 15) / 16;
            let wg_y = (height + 15) / 16;
            cpass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging_buf, 0, byte_size);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);

        let result = if rx.recv().ok()?.is_ok() {
            let data = slice.get_mapped_range();
            let vec: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buf.unmap();
            Some(vec)
        } else {
            None
        };

        let elapsed_ns = start_t.elapsed().as_nanos() as u64;
        self.compute_busy_ns.fetch_add(elapsed_ns, Ordering::Relaxed);
        result
    }
}

#[derive(Debug, Clone)]
pub struct TelemetrySnapshot {
    pub cpu_load_pct: f32,
    pub gpu_load_pct: f32,
    pub cpu_history: Vec<f32>,
    pub gpu_history: Vec<f32>,
}

impl Default for TelemetrySnapshot {
    fn default() -> Self {
        Self {
            cpu_load_pct: 0.0,
            gpu_load_pct: 0.0,
            cpu_history: vec![0.0; 60],
            gpu_history: vec![0.0; 60],
        }
    }
}

pub struct SystemTelemetry {
    pub state: Arc<Mutex<TelemetrySnapshot>>,
}

impl SystemTelemetry {
    pub fn spawn(compute_busy_ns: Option<Arc<AtomicU64>>) -> Self {
        let state = Arc::new(Mutex::new(TelemetrySnapshot::default()));
        let state_clone = Arc::clone(&state);

        thread::spawn(move || {
            let mut prev_total = 0u64;
            let mut prev_idle = 0u64;
            let mut prev_compute_ns = 0u64;
            let mut last_tick = Instant::now();

            loop {
                thread::sleep(Duration::from_millis(400));
                let dt_ns = last_tick.elapsed().as_nanos() as u64;
                last_tick = Instant::now();

                // 1. Sample CPU load from /proc/stat
                let mut cpu_pct = 0.0f32;
                if let Ok(stat) = fs::read_to_string("/proc/stat") {
                    if let Some(first) = stat.lines().next() {
                        let vals: Vec<u64> = first
                            .split_whitespace()
                            .skip(1)
                            .filter_map(|s| s.parse::<u64>().ok())
                            .collect();
                        if vals.len() >= 4 {
                            let idle = vals[3] + vals.get(4).copied().unwrap_or(0);
                            let total: u64 = vals.iter().sum();
                            let d_total = total.saturating_sub(prev_total);
                            let d_idle = idle.saturating_sub(prev_idle);
                            if prev_total > 0 && d_total > 0 {
                                cpu_pct = ((d_total.saturating_sub(d_idle)) as f32 / d_total as f32) * 100.0;
                            }
                            prev_total = total;
                            prev_idle = idle;
                        }
                    }
                }

                // 2. Sample GPU load from Vulkan compute duty + system GPU sensors
                let mut gpu_pct = 0.0f32;
                if let Some(ref busy_counter) = compute_busy_ns {
                    let cur_ns = busy_counter.load(Ordering::Relaxed);
                    let d_busy = cur_ns.saturating_sub(prev_compute_ns);
                    prev_compute_ns = cur_ns;
                    if dt_ns > 0 {
                        gpu_pct = ((d_busy as f32 / dt_ns as f32) * 100.0).min(100.0);
                    }
                }

                if let Some(hw_gpu) = read_hw_gpu_utilization() {
                    gpu_pct = gpu_pct.max(hw_gpu);
                }

                if let Ok(mut guard) = state_clone.lock() {
                    let c_pct = cpu_pct.clamp(0.0, 100.0);
                    let g_pct = gpu_pct.clamp(0.0, 100.0);
                    guard.cpu_load_pct = c_pct;
                    guard.gpu_load_pct = g_pct;
                    guard.cpu_history.push(c_pct);
                    if guard.cpu_history.len() > 60 {
                        guard.cpu_history.remove(0);
                    }
                    guard.gpu_history.push(g_pct);
                    if guard.gpu_history.len() > 60 {
                        guard.gpu_history.remove(0);
                    }
                }
            }
        });

        Self { state }
    }
}

fn read_hw_gpu_utilization() -> Option<f32> {
    for drm_path in [
        "/sys/class/drm/card0/device/gpu_busy_percent",
        "/sys/class/drm/card1/device/gpu_busy_percent",
    ] {
        if let Ok(s) = fs::read_to_string(drm_path) {
            if let Ok(v) = s.trim().parse::<f32>() {
                return Some(v);
            }
        }
    }

    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if out.status.success() {
        let s = String::from_utf8_lossy(&out.stdout);
        if let Some(line) = s.lines().next() {
            if let Ok(v) = line.trim().parse::<f32>() {
                return Some(v);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vulkan_distance_field_compute_shader() {
        let vk = VulkanEngine::new().expect("VulkanEngine must initialize");
        assert!(!vk.device_name.is_empty());
        let mut obs = vec![0u32; 16 * 16];
        obs[4 * 16 + 4] = 1;
        let df = vk.compute_distance_field(&obs, 16, 16, 10).expect("Distance field output");
        assert!((df[4 * 16 + 4] - 0.0).abs() < 1e-4);
        assert!((df[4 * 16 + 8] - 4.0).abs() < 1e-3);
        assert!((df[7 * 16 + 8] - 5.0).abs() < 1e-3); // 3-4-5 right triangle
    }
}

