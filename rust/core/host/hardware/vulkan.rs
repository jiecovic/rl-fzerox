// rust/core/host/hardware/vulkan.rs
//! Experimental synchronous libretro Vulkan frontend support.
//!
//! ParaLLEl-RDP uses the libretro Vulkan interface, which is substantially
//! more involved than OpenGL. This module deliberately implements only the
//! blocking readback path needed for RL observations.

use std::ffi::{c_char, c_void};
use std::ptr;

use ash::vk;
use libretro_sys::{HwContextResetFn, HwRenderCallback};

use crate::core::video::VideoFrame;

const VULKAN_INTERFACE_TYPE: u32 = 0;
const VULKAN_INTERFACE_VERSION: u32 = 5;
const VULKAN_NEGOTIATION_INTERFACE_TYPE: u32 = 0;
const VULKAN_NEGOTIATION_INTERFACE_VERSION: u32 = 2;

type VulkanGetApplicationInfo =
    Option<unsafe extern "C" fn() -> *const vk::ApplicationInfo<'static>>;
type VulkanCreateDevice = Option<
    unsafe extern "C" fn(
        *mut RetroVulkanContext,
        vk::Instance,
        vk::PhysicalDevice,
        vk::SurfaceKHR,
        vk::PFN_vkGetInstanceProcAddr,
        *const *const c_char,
        libc::c_uint,
        *const *const c_char,
        libc::c_uint,
        *const vk::PhysicalDeviceFeatures,
    ) -> bool,
>;
type VulkanDestroyDevice = Option<unsafe extern "C" fn()>;
type VulkanCreateInstance = Option<
    unsafe extern "C" fn(
        vk::PFN_vkGetInstanceProcAddr,
        *const vk::ApplicationInfo,
        Option<unsafe extern "C" fn(*mut c_void, *const vk::InstanceCreateInfo) -> vk::Instance>,
        *mut c_void,
    ) -> vk::Instance,
>;
type VulkanCreateDevice2 = Option<
    unsafe extern "C" fn(
        *mut RetroVulkanContext,
        vk::Instance,
        vk::PhysicalDevice,
        vk::SurfaceKHR,
        vk::PFN_vkGetInstanceProcAddr,
        Option<
            unsafe extern "C" fn(
                vk::PhysicalDevice,
                *mut c_void,
                *const vk::DeviceCreateInfo,
            ) -> vk::Device,
        >,
        *mut c_void,
    ) -> bool,
>;

#[derive(Clone, Copy)]
pub struct VulkanNegotiationInterface {
    get_application_info: VulkanGetApplicationInfo,
    create_device: VulkanCreateDevice,
    destroy_device: VulkanDestroyDevice,
    create_instance: VulkanCreateInstance,
    create_device2: VulkanCreateDevice2,
}

impl VulkanNegotiationInterface {
    pub fn from_raw(data: *mut c_void) -> Option<Self> {
        if data.is_null() {
            return None;
        }
        let raw = unsafe { &*data.cast::<RetroVulkanNegotiationInterface>() };
        if raw.interface_type != VULKAN_NEGOTIATION_INTERFACE_TYPE || raw.interface_version == 0 {
            return None;
        }
        Some(Self {
            get_application_info: raw.get_application_info,
            create_device: raw.create_device,
            destroy_device: raw.destroy_device,
            create_instance: if raw.interface_version >= 2 {
                raw.create_instance
            } else {
                None
            },
            create_device2: if raw.interface_version >= 2 {
                raw.create_device2
            } else {
                None
            },
        })
    }
}

pub fn write_negotiation_support(data: *mut c_void) -> bool {
    if data.is_null() {
        return false;
    }
    let support = unsafe { &mut *data.cast::<RetroVulkanNegotiationSupport>() };
    support.interface_version = if support.interface_type == VULKAN_NEGOTIATION_INTERFACE_TYPE {
        VULKAN_NEGOTIATION_INTERFACE_VERSION
    } else {
        0
    };
    true
}

pub struct VulkanRenderContext {
    _entry: ash::Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    queue_family_index: u32,
    command_pool: vk::CommandPool,
    interface: RetroVulkanInterface,
    context_reset: HwContextResetFn,
    context_destroy: HwContextResetFn,
    destroy_device_callback: VulkanDestroyDevice,
    staging: Option<StagingBuffer>,
    latest_image: Option<LatestImage>,
    pending_command_buffers: Vec<vk::CommandBuffer>,
    signal_semaphore: vk::Semaphore,
}

impl VulkanRenderContext {
    pub fn prepare_callback(callback: &mut HwRenderCallback) {
        callback.get_current_framebuffer = get_current_framebuffer;
        callback.get_proc_address = get_proc_address;
    }

    pub fn from_callback(
        callback: &mut HwRenderCallback,
        negotiation: Option<VulkanNegotiationInterface>,
    ) -> Result<Self, String> {
        let entry = unsafe { ash::Entry::load().map_err(|error| error.to_string())? };
        let instance = create_instance(&entry, negotiation)?;
        let physical_device = pick_physical_device(&instance)?;
        let device_context =
            create_device_context(&entry, &instance, physical_device, negotiation)?;
        let command_pool =
            create_command_pool(&device_context.device, device_context.queue_family_index)?;

        Self::prepare_callback(callback);

        let interface = RetroVulkanInterface {
            interface_type: VULKAN_INTERFACE_TYPE,
            interface_version: VULKAN_INTERFACE_VERSION,
            handle: ptr::null_mut(),
            instance: instance.handle(),
            gpu: device_context.physical_device,
            device: device_context.device.handle(),
            get_device_proc_addr: instance.fp_v1_0().get_device_proc_addr,
            get_instance_proc_addr: entry.static_fn().get_instance_proc_addr,
            queue: device_context.queue,
            queue_index: device_context.queue_family_index,
            set_image,
            get_sync_index,
            get_sync_index_mask,
            set_command_buffers,
            wait_sync_index,
            lock_queue,
            unlock_queue,
            set_signal_semaphore,
        };

        Ok(Self {
            _entry: entry,
            instance,
            physical_device: device_context.physical_device,
            device: device_context.device,
            queue: device_context.queue,
            queue_family_index: device_context.queue_family_index,
            command_pool,
            interface,
            context_reset: callback.context_reset,
            context_destroy: callback.context_destroy,
            destroy_device_callback: negotiation.and_then(|value| value.destroy_device),
            staging: None,
            latest_image: None,
            pending_command_buffers: Vec::new(),
            signal_semaphore: vk::Semaphore::null(),
        })
    }

    pub fn set_interface_handle(&mut self) {
        // Libretro stores this opaque pointer and passes it back to our callbacks.
        // Set it only after boxing so the pointed-to address remains stable.
        self.interface.handle = (self as *mut Self).cast::<c_void>();
    }

    pub fn interface_ptr(&self) -> *const c_void {
        (&self.interface as *const RetroVulkanInterface).cast::<c_void>()
    }

    pub fn reset_core_context(&self) -> Result<(), String> {
        unsafe {
            (self.context_reset)();
        }
        Ok(())
    }

    pub fn destroy_core_context(&self) {
        if self.context_destroy as usize == 0 {
            return;
        }
        unsafe {
            (self.context_destroy)();
        }
    }

    pub fn capture_frame(&mut self, width: usize, height: usize) -> Option<VideoFrame> {
        let latest_image = self.latest_image?;
        if width == 0 || height == 0 {
            return None;
        }
        self.submit_pending_command_buffers().ok()?;
        self.ensure_staging(width, height).ok()?;
        self.copy_latest_image_to_staging(latest_image, width, height)
            .ok()?;
        let rgb = self
            .read_staging_rgb(latest_image.format, width, height)
            .ok()?;
        Some(VideoFrame { width, height, rgb })
    }

    fn submit_pending_command_buffers(&mut self) -> Result<(), String> {
        if self.pending_command_buffers.is_empty() {
            unsafe {
                self.device
                    .queue_wait_idle(self.queue)
                    .map_err(|error| format!("vkQueueWaitIdle failed: {error:?}"))?;
            }
            return Ok(());
        }

        let command_buffers = std::mem::take(&mut self.pending_command_buffers);
        let submit_info = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
        unsafe {
            self.device
                .queue_submit(self.queue, &submit_info, vk::Fence::null())
                .map_err(|error| format!("vkQueueSubmit failed: {error:?}"))?;
            self.device
                .queue_wait_idle(self.queue)
                .map_err(|error| format!("vkQueueWaitIdle failed: {error:?}"))?;
        }
        Ok(())
    }

    fn ensure_staging(&mut self, width: usize, height: usize) -> Result<(), String> {
        let required_size = (width as vk::DeviceSize)
            .checked_mul(height as vk::DeviceSize)
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| "staging buffer size overflow".to_owned())?;
        if self
            .staging
            .as_ref()
            .is_some_and(|staging| staging.size >= required_size)
        {
            return Ok(());
        }
        if let Some(staging) = self.staging.take() {
            staging.destroy(&self.device);
        }
        self.staging = Some(StagingBuffer::new(
            &self.instance,
            &self.device,
            self.physical_device,
            required_size,
        )?);
        Ok(())
    }

    fn copy_latest_image_to_staging(
        &mut self,
        latest_image: LatestImage,
        width: usize,
        height: usize,
    ) -> Result<(), String> {
        let staging = self
            .staging
            .as_ref()
            .ok_or_else(|| "missing staging buffer".to_owned())?;
        let command_buffer = self.begin_one_time_commands()?;

        let to_transfer = vk::ImageMemoryBarrier::default()
            .old_layout(latest_image.layout)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_queue_family_index(latest_image.src_queue_family)
            .dst_queue_family_index(self.queue_family_index)
            .image(latest_image.image)
            .subresource_range(color_subresource_range())
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
        let copy_region = vk::BufferImageCopy::default()
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .image_extent(vk::Extent3D {
                width: width as u32,
                height: height as u32,
                depth: 1,
            });
        let to_original = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .new_layout(latest_image.layout)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(latest_image.src_queue_family)
            .image(latest_image.image)
            .subresource_range(color_subresource_range())
            .src_access_mask(vk::AccessFlags::TRANSFER_READ)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);

        unsafe {
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_transfer],
            );
            self.device.cmd_copy_image_to_buffer(
                command_buffer,
                latest_image.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                staging.buffer,
                &[copy_region],
            );
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_original],
            );
        }
        self.end_one_time_commands(command_buffer)
    }

    fn begin_one_time_commands(&self) -> Result<vk::CommandBuffer, String> {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe {
            self.device
                .allocate_command_buffers(&allocate_info)
                .map_err(|error| format!("vkAllocateCommandBuffers failed: {error:?}"))?
                .into_iter()
                .next()
                .ok_or_else(|| "vkAllocateCommandBuffers returned no buffer".to_owned())?
        };
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &begin_info)
                .map_err(|error| format!("vkBeginCommandBuffer failed: {error:?}"))?;
        }
        Ok(command_buffer)
    }

    fn end_one_time_commands(&self, command_buffer: vk::CommandBuffer) -> Result<(), String> {
        unsafe {
            self.device
                .end_command_buffer(command_buffer)
                .map_err(|error| format!("vkEndCommandBuffer failed: {error:?}"))?;
            let command_buffers = [command_buffer];
            let submit_info = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
            self.device
                .queue_submit(self.queue, &submit_info, vk::Fence::null())
                .map_err(|error| format!("vkQueueSubmit failed: {error:?}"))?;
            self.device
                .queue_wait_idle(self.queue)
                .map_err(|error| format!("vkQueueWaitIdle failed: {error:?}"))?;
            self.device
                .free_command_buffers(self.command_pool, &[command_buffer]);
        }
        Ok(())
    }

    fn read_staging_rgb(
        &self,
        format: vk::Format,
        width: usize,
        height: usize,
    ) -> Result<Vec<u8>, String> {
        let staging = self
            .staging
            .as_ref()
            .ok_or_else(|| "missing staging buffer".to_owned())?;
        let byte_len = width
            .checked_mul(height)
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| "staging read size overflow".to_owned())?;
        let mapped = unsafe {
            self.device
                .map_memory(
                    staging.memory,
                    0,
                    byte_len as u64,
                    vk::MemoryMapFlags::empty(),
                )
                .map_err(|error| format!("vkMapMemory failed: {error:?}"))?
        };
        let bytes = unsafe { std::slice::from_raw_parts(mapped.cast::<u8>(), byte_len) };
        let rgb = match format {
            vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB => bgra_to_rgb(bytes),
            vk::Format::R8G8B8A8_UNORM | vk::Format::R8G8B8A8_SRGB => rgba_to_rgb(bytes),
            _ => {
                unsafe {
                    self.device.unmap_memory(staging.memory);
                }
                return Err(format!("unsupported Vulkan frame format {format:?}"));
            }
        };
        unsafe {
            self.device.unmap_memory(staging.memory);
        }
        Ok(flip_rgb_rows(&rgb, width, height))
    }
}

impl Drop for VulkanRenderContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            if let Some(staging) = self.staging.take() {
                staging.destroy(&self.device);
            }
            if self.command_pool != vk::CommandPool::null() {
                self.device.destroy_command_pool(self.command_pool, None);
            }
            if let Some(destroy_device) = self.destroy_device_callback {
                destroy_device();
            }
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

unsafe impl Send for VulkanRenderContext {}
unsafe impl Sync for VulkanRenderContext {}

struct DeviceContext {
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    queue_family_index: u32,
}

#[derive(Clone, Copy)]
struct LatestImage {
    image: vk::Image,
    layout: vk::ImageLayout,
    format: vk::Format,
    src_queue_family: u32,
}

struct StagingBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
}

impl StagingBuffer {
    fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        size: vk::DeviceSize,
    ) -> Result<Self, String> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe {
            device
                .create_buffer(&buffer_info, None)
                .map_err(|error| format!("vkCreateBuffer failed: {error:?}"))?
        };
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let memory_type_index = find_memory_type(
            instance,
            physical_device,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index);
        let memory = unsafe {
            device
                .allocate_memory(&allocate_info, None)
                .map_err(|error| format!("vkAllocateMemory failed: {error:?}"))?
        };
        unsafe {
            device
                .bind_buffer_memory(buffer, memory, 0)
                .map_err(|error| format!("vkBindBufferMemory failed: {error:?}"))?;
        }
        Ok(Self {
            buffer,
            memory,
            size,
        })
    }

    fn destroy(self, device: &ash::Device) {
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
struct RetroVulkanContext {
    gpu: vk::PhysicalDevice,
    device: vk::Device,
    queue: vk::Queue,
    queue_family_index: u32,
    presentation_queue: vk::Queue,
    presentation_queue_family_index: u32,
}

#[repr(C)]
struct RetroVulkanNegotiationInterface {
    interface_type: u32,
    interface_version: u32,
    get_application_info: VulkanGetApplicationInfo,
    create_device: VulkanCreateDevice,
    destroy_device: VulkanDestroyDevice,
    create_instance: VulkanCreateInstance,
    create_device2: VulkanCreateDevice2,
}

#[repr(C)]
struct RetroVulkanNegotiationSupport {
    interface_type: u32,
    interface_version: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct RetroVulkanImage {
    image_view: vk::ImageView,
    image_layout: vk::ImageLayout,
    create_info: vk::ImageViewCreateInfo<'static>,
}

#[repr(C)]
struct RetroVulkanInterface {
    interface_type: u32,
    interface_version: u32,
    handle: *mut c_void,
    instance: vk::Instance,
    gpu: vk::PhysicalDevice,
    device: vk::Device,
    get_device_proc_addr: vk::PFN_vkGetDeviceProcAddr,
    get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
    queue: vk::Queue,
    queue_index: u32,
    set_image:
        unsafe extern "C" fn(*mut c_void, *const RetroVulkanImage, u32, *const vk::Semaphore, u32),
    get_sync_index: unsafe extern "C" fn(*mut c_void) -> u32,
    get_sync_index_mask: unsafe extern "C" fn(*mut c_void) -> u32,
    set_command_buffers: unsafe extern "C" fn(*mut c_void, u32, *const vk::CommandBuffer),
    wait_sync_index: unsafe extern "C" fn(*mut c_void),
    lock_queue: unsafe extern "C" fn(*mut c_void),
    unlock_queue: unsafe extern "C" fn(*mut c_void),
    set_signal_semaphore: unsafe extern "C" fn(*mut c_void, vk::Semaphore),
}

fn create_instance(
    entry: &ash::Entry,
    negotiation: Option<VulkanNegotiationInterface>,
) -> Result<ash::Instance, String> {
    let fallback_app_info = vk::ApplicationInfo::default()
        .application_name(c"rl-fzerox")
        .engine_name(c"rl-fzerox")
        .api_version(vk::API_VERSION_1_1);
    let app_info = match negotiation.and_then(|value| value.get_application_info) {
        Some(get_application_info) => {
            let pointer = unsafe { get_application_info() };
            if pointer.is_null() {
                &fallback_app_info
            } else {
                unsafe { &*pointer }
            }
        }
        None => &fallback_app_info,
    };
    if let Some(create_instance) = negotiation.and_then(|value| value.create_instance) {
        let handle = unsafe {
            create_instance(
                entry.static_fn().get_instance_proc_addr,
                app_info,
                Some(create_instance_wrapper),
                (entry as *const ash::Entry).cast_mut().cast::<c_void>(),
            )
        };
        if handle != vk::Instance::null() {
            return Ok(unsafe { ash::Instance::load(entry.static_fn(), handle) });
        }
    }

    let create_info = vk::InstanceCreateInfo::default().application_info(app_info);
    unsafe {
        entry
            .create_instance(&create_info, None)
            .map_err(|error| format!("vkCreateInstance failed: {error:?}"))
    }
}

fn pick_physical_device(instance: &ash::Instance) -> Result<vk::PhysicalDevice, String> {
    let devices = unsafe {
        instance
            .enumerate_physical_devices()
            .map_err(|error| format!("vkEnumeratePhysicalDevices failed: {error:?}"))?
    };
    devices
        .into_iter()
        .find(|&device| find_queue_family(instance, device).is_some())
        .ok_or_else(|| "no Vulkan device with graphics queue found".to_owned())
}

fn create_device_context(
    entry: &ash::Entry,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    negotiation: Option<VulkanNegotiationInterface>,
) -> Result<DeviceContext, String> {
    if let Some(create_device2) = negotiation.and_then(|value| value.create_device2) {
        let mut context = RetroVulkanContext {
            gpu: vk::PhysicalDevice::null(),
            device: vk::Device::null(),
            queue: vk::Queue::null(),
            queue_family_index: 0,
            presentation_queue: vk::Queue::null(),
            presentation_queue_family_index: 0,
        };
        let created = unsafe {
            create_device2(
                &mut context,
                instance.handle(),
                physical_device,
                vk::SurfaceKHR::null(),
                entry.static_fn().get_instance_proc_addr,
                Some(create_device_wrapper),
                (instance as *const ash::Instance)
                    .cast_mut()
                    .cast::<c_void>(),
            )
        };
        if created && context.device != vk::Device::null() && context.queue != vk::Queue::null() {
            let device = unsafe { ash::Device::load(instance.fp_v1_0(), context.device) };
            return Ok(DeviceContext {
                physical_device: if context.gpu == vk::PhysicalDevice::null() {
                    physical_device
                } else {
                    context.gpu
                },
                device,
                queue: context.queue,
                queue_family_index: context.queue_family_index,
            });
        }
    }

    if let Some(create_device) = negotiation.and_then(|value| value.create_device) {
        let features = vk::PhysicalDeviceFeatures::default();
        let mut context = RetroVulkanContext {
            gpu: vk::PhysicalDevice::null(),
            device: vk::Device::null(),
            queue: vk::Queue::null(),
            queue_family_index: 0,
            presentation_queue: vk::Queue::null(),
            presentation_queue_family_index: 0,
        };
        let created = unsafe {
            create_device(
                &mut context,
                instance.handle(),
                physical_device,
                vk::SurfaceKHR::null(),
                entry.static_fn().get_instance_proc_addr,
                ptr::null(),
                0,
                ptr::null(),
                0,
                &features,
            )
        };
        if created && context.device != vk::Device::null() && context.queue != vk::Queue::null() {
            let device = unsafe { ash::Device::load(instance.fp_v1_0(), context.device) };
            return Ok(DeviceContext {
                physical_device: context.gpu,
                device,
                queue: context.queue,
                queue_family_index: context.queue_family_index,
            });
        }
    }

    let queue_family_index = find_queue_family(instance, physical_device)
        .ok_or_else(|| "no graphics queue family found".to_owned())?;
    let priorities = [1.0_f32];
    let queue_info = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&priorities)];
    let features = vk::PhysicalDeviceFeatures::default();
    let device_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_info)
        .enabled_features(&features);
    let device = unsafe {
        instance
            .create_device(physical_device, &device_info, None)
            .map_err(|error| format!("vkCreateDevice failed: {error:?}"))?
    };
    let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
    Ok(DeviceContext {
        physical_device,
        device,
        queue,
        queue_family_index,
    })
}

unsafe extern "C" fn create_instance_wrapper(
    opaque: *mut c_void,
    create_info: *const vk::InstanceCreateInfo,
) -> vk::Instance {
    if opaque.is_null() || create_info.is_null() {
        return vk::Instance::null();
    }
    let entry = unsafe { &*opaque.cast::<ash::Entry>() };
    match unsafe { entry.create_instance(&*create_info, None) } {
        Ok(instance) => instance.handle(),
        Err(_) => vk::Instance::null(),
    }
}

unsafe extern "C" fn create_device_wrapper(
    physical_device: vk::PhysicalDevice,
    opaque: *mut c_void,
    create_info: *const vk::DeviceCreateInfo,
) -> vk::Device {
    if opaque.is_null() || create_info.is_null() {
        return vk::Device::null();
    }
    let instance = unsafe { &*opaque.cast::<ash::Instance>() };
    match unsafe { instance.create_device(physical_device, &*create_info, None) } {
        Ok(device) => device.handle(),
        Err(_) => vk::Device::null(),
    }
}

fn create_command_pool(
    device: &ash::Device,
    queue_family_index: u32,
) -> Result<vk::CommandPool, String> {
    let create_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(queue_family_index)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    unsafe {
        device
            .create_command_pool(&create_info, None)
            .map_err(|error| format!("vkCreateCommandPool failed: {error:?}"))
    }
}

fn find_queue_family(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> Option<u32> {
    let families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    families
        .iter()
        .position(|family| {
            family
                .queue_flags
                .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
        })
        .map(|index| index as u32)
}

fn find_memory_type(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    memory_type_bits: u32,
    properties: vk::MemoryPropertyFlags,
) -> Result<u32, String> {
    let memory_properties =
        unsafe { instance.get_physical_device_memory_properties(physical_device) };
    for index in 0..memory_properties.memory_type_count {
        let supported = (memory_type_bits & (1 << index)) != 0;
        let memory_type = memory_properties.memory_types[index as usize];
        if supported && memory_type.property_flags.contains(properties) {
            return Ok(index);
        }
    }
    Err("no compatible Vulkan memory type found".to_owned())
}

fn color_subresource_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1)
}

fn bgra_to_rgb(bytes: &[u8]) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(bytes.len() / 4 * 3);
    for pixel in bytes.chunks_exact(4) {
        rgb.extend_from_slice(&[pixel[2], pixel[1], pixel[0]]);
    }
    rgb
}

fn rgba_to_rgb(bytes: &[u8]) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(bytes.len() / 4 * 3);
    for pixel in bytes.chunks_exact(4) {
        rgb.extend_from_slice(&[pixel[0], pixel[1], pixel[2]]);
    }
    rgb
}

fn flip_rgb_rows(rgb_bottom_left: &[u8], width: usize, height: usize) -> Vec<u8> {
    let row_len = width * 3;
    let mut rgb = vec![0_u8; rgb_bottom_left.len()];
    for y in 0..height {
        let src = (height - 1 - y) * row_len;
        let dst = y * row_len;
        rgb[dst..dst + row_len].copy_from_slice(&rgb_bottom_left[src..src + row_len]);
    }
    rgb
}

unsafe extern "C" fn set_image(
    handle: *mut c_void,
    image: *const RetroVulkanImage,
    _num_semaphores: u32,
    _semaphores: *const vk::Semaphore,
    src_queue_family: u32,
) {
    let Some(context) = context_from_handle(handle) else {
        return;
    };
    if image.is_null() {
        context.latest_image = None;
        return;
    }
    let image = unsafe { *image };
    context.latest_image = Some(LatestImage {
        image: image.create_info.image,
        layout: image.image_layout,
        format: image.create_info.format,
        src_queue_family,
    });
}

unsafe extern "C" fn get_sync_index(_handle: *mut c_void) -> u32 {
    0
}

unsafe extern "C" fn get_sync_index_mask(_handle: *mut c_void) -> u32 {
    1
}

unsafe extern "C" fn set_command_buffers(
    handle: *mut c_void,
    num_cmd: u32,
    command_buffers: *const vk::CommandBuffer,
) {
    let Some(context) = context_from_handle(handle) else {
        return;
    };
    if command_buffers.is_null() || num_cmd == 0 {
        context.pending_command_buffers.clear();
        return;
    }
    let buffers = unsafe { std::slice::from_raw_parts(command_buffers, num_cmd as usize) };
    context.pending_command_buffers.clear();
    context.pending_command_buffers.extend_from_slice(buffers);
}

unsafe extern "C" fn wait_sync_index(handle: *mut c_void) {
    let Some(context) = context_from_handle(handle) else {
        return;
    };
    unsafe {
        let _ = context.device.device_wait_idle();
    }
}

unsafe extern "C" fn lock_queue(_handle: *mut c_void) {}

unsafe extern "C" fn unlock_queue(_handle: *mut c_void) {}

unsafe extern "C" fn set_signal_semaphore(handle: *mut c_void, semaphore: vk::Semaphore) {
    let Some(context) = context_from_handle(handle) else {
        return;
    };
    context.signal_semaphore = semaphore;
}

unsafe extern "C" fn get_current_framebuffer() -> libc::uintptr_t {
    0
}

unsafe extern "C" fn get_proc_address(_sym: *const c_char) -> libretro_sys::ProcAddressFn {
    missing_proc_stub
}

unsafe extern "C" fn missing_proc_stub() {}

fn context_from_handle(handle: *mut c_void) -> Option<&'static mut VulkanRenderContext> {
    if handle.is_null() {
        return None;
    }
    Some(unsafe { &mut *handle.cast::<VulkanRenderContext>() })
}
