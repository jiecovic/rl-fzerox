// rust/core/host/hardware.rs
//! Minimal synchronous hardware-render frontend support.
//!
//! This intentionally starts with a blocking OpenGL readback path. It is slow
//! compared with a real async PBO pipeline, but it proves whether a renderer can
//! run correctly before adding more moving parts.

use std::ffi::{CStr, CString, c_char, c_void};
use std::ptr;
use std::sync::OnceLock;

use libloading::Library;
use libretro_sys::{
    HW_FRAME_BUFFER_VALID, HwContextResetFn, HwContextType, HwRenderCallback, ProcAddressFn,
};

use crate::core::video::VideoFrame;

// EGL/GL enum values mirror the external C headers. Keep those as constants;
// only project-owned settings should be grouped into local structs.
const EGL_FALSE: EglBoolean = 0;
const EGL_NONE: EglInt = 0x3038;
const EGL_RED_SIZE: EglInt = 0x3024;
const EGL_GREEN_SIZE: EglInt = 0x3023;
const EGL_BLUE_SIZE: EglInt = 0x3022;
const EGL_ALPHA_SIZE: EglInt = 0x3021;
const EGL_SURFACE_TYPE: EglInt = 0x3033;
const EGL_PBUFFER_BIT: EglInt = 0x0001;
const EGL_RENDERABLE_TYPE: EglInt = 0x3040;
const EGL_OPENGL_BIT: EglInt = 0x0008;
const EGL_WIDTH: EglInt = 0x3057;
const EGL_HEIGHT: EglInt = 0x3056;
const EGL_OPENGL_API: EglEnum = 0x30A2;
const EGL_CONTEXT_MAJOR_VERSION: EglInt = 0x3098;
const EGL_CONTEXT_MINOR_VERSION: EglInt = 0x30FB;
const EGL_CONTEXT_OPENGL_PROFILE_MASK_KHR: EglInt = 0x30FD;
const EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT_KHR: EglInt = 0x0000_0001;
const EGL_PLATFORM_SURFACELESS_MESA: EglEnum = 0x31DD;

const GL_RGB: GlEnum = 0x1907;
const GL_UNSIGNED_BYTE: GlEnum = 0x1401;
const GL_PACK_ALIGNMENT: GlEnum = 0x0D05;

const PBUFFER_SIZE: PbufferSize = PbufferSize {
    width: 1024,
    height: 1024,
};

#[derive(Clone, Copy)]
struct PbufferSize {
    width: EglInt,
    height: EglInt,
}

type EglBoolean = libc::c_uint;
type EglEnum = libc::c_uint;
type EglInt = libc::c_int;
type EglDisplay = *mut c_void;
type EglConfig = *mut c_void;
type EglSurface = *mut c_void;
type EglContext = *mut c_void;

type EglGetDisplay = unsafe extern "C" fn(*mut c_void) -> EglDisplay;
type EglInitialize = unsafe extern "C" fn(EglDisplay, *mut EglInt, *mut EglInt) -> EglBoolean;
type EglChooseConfig = unsafe extern "C" fn(
    EglDisplay,
    *const EglInt,
    *mut EglConfig,
    EglInt,
    *mut EglInt,
) -> EglBoolean;
type EglBindApi = unsafe extern "C" fn(EglEnum) -> EglBoolean;
type EglCreatePbufferSurface =
    unsafe extern "C" fn(EglDisplay, EglConfig, *const EglInt) -> EglSurface;
type EglCreateContext =
    unsafe extern "C" fn(EglDisplay, EglConfig, EglContext, *const EglInt) -> EglContext;
type EglMakeCurrent =
    unsafe extern "C" fn(EglDisplay, EglSurface, EglSurface, EglContext) -> EglBoolean;
type EglDestroySurface = unsafe extern "C" fn(EglDisplay, EglSurface) -> EglBoolean;
type EglDestroyContext = unsafe extern "C" fn(EglDisplay, EglContext) -> EglBoolean;
type EglTerminate = unsafe extern "C" fn(EglDisplay) -> EglBoolean;
type EglGetError = unsafe extern "C" fn() -> EglInt;
type EglGetProcAddress = unsafe extern "C" fn(*const c_char) -> *const c_void;
type EglGetPlatformDisplayExt =
    unsafe extern "C" fn(EglEnum, *mut c_void, *const EglInt) -> EglDisplay;

type GlEnum = libc::c_uint;
type GlInt = libc::c_int;
type GlSizei = libc::c_int;
type GlReadPixels =
    unsafe extern "C" fn(GlInt, GlInt, GlSizei, GlSizei, GlEnum, GlEnum, *mut c_void);
type GlPixelStorei = unsafe extern "C" fn(GlEnum, GlInt);
type GlFinish = unsafe extern "C" fn();

#[derive(Clone, Copy)]
struct GlFns {
    read_pixels: GlReadPixels,
    pixel_storei: GlPixelStorei,
    finish: GlFinish,
}

pub struct HardwareRenderContext {
    egl: &'static EglFns,
    display: EglDisplay,
    surface: EglSurface,
    context: EglContext,
    context_reset: HwContextResetFn,
    context_destroy: HwContextResetFn,
    gl: GlFns,
    scratch: Vec<u8>,
}

impl HardwareRenderContext {
    pub fn from_callback(callback: &mut HwRenderCallback) -> Result<Self, String> {
        let context_type = context_type(callback.context_type);
        if !matches!(
            context_type,
            Some(HwContextType::OpenGL | HwContextType::OpenGLCore)
        ) {
            return Err(format!(
                "hardware context {:?} is not supported by this host yet",
                context_type
            ));
        }

        let egl = egl_fns()?;
        let display = create_display(egl)?;
        let config = choose_config(egl, display)?;
        let surface = create_surface(egl, display, config)?;
        bind_opengl_api(egl)?;
        let context = create_context(egl, display, config, callback)?;
        make_current(egl, display, surface, context)?;

        callback.get_current_framebuffer = get_current_framebuffer;
        callback.get_proc_address = get_proc_address;

        let gl = GlFns::load()?;
        Ok(Self {
            egl,
            display,
            surface,
            context,
            context_reset: callback.context_reset,
            context_destroy: callback.context_destroy,
            gl,
            scratch: Vec::new(),
        })
    }

    pub fn can_capture(data: *const c_void) -> bool {
        data == HW_FRAME_BUFFER_VALID
    }

    pub fn reset_core_context(&self) -> Result<(), String> {
        make_current(self.egl, self.display, self.surface, self.context)?;
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
        if width == 0 || height == 0 {
            return None;
        }
        make_current(self.egl, self.display, self.surface, self.context).ok()?;
        let byte_len = width.checked_mul(height)?.checked_mul(3)?;
        self.scratch.resize(byte_len, 0);
        unsafe {
            (self.gl.finish)();
            (self.gl.pixel_storei)(GL_PACK_ALIGNMENT, 1);
            (self.gl.read_pixels)(
                0,
                0,
                width as GlSizei,
                height as GlSizei,
                GL_RGB,
                GL_UNSIGNED_BYTE,
                self.scratch.as_mut_ptr().cast::<c_void>(),
            );
        }
        Some(VideoFrame {
            width,
            height,
            rgb: flip_rgb_rows(&self.scratch, width, height),
        })
    }
}

impl Drop for HardwareRenderContext {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.egl.make_current)(
                self.display,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
            );
            let _ = (self.egl.destroy_context)(self.display, self.context);
            let _ = (self.egl.destroy_surface)(self.display, self.surface);
            let _ = (self.egl.terminate)(self.display);
        }
    }
}

#[derive(Clone, Copy)]
struct EglFns {
    _library: &'static Library,
    _gl_library: &'static Library,
    get_display: EglGetDisplay,
    initialize: EglInitialize,
    choose_config: EglChooseConfig,
    bind_api: EglBindApi,
    create_pbuffer_surface: EglCreatePbufferSurface,
    create_context: EglCreateContext,
    make_current: EglMakeCurrent,
    destroy_surface: EglDestroySurface,
    destroy_context: EglDestroyContext,
    terminate: EglTerminate,
    get_error: EglGetError,
    get_proc_address: EglGetProcAddress,
    get_platform_display_ext: Option<EglGetPlatformDisplayExt>,
}

impl EglFns {
    fn load() -> Result<Self, String> {
        let library = Box::leak(Box::new(unsafe {
            Library::new("libEGL.so.1").map_err(|error| error.to_string())?
        }));
        let gl_library = Box::leak(Box::new(unsafe {
            Library::new("libGL.so.1").map_err(|error| error.to_string())?
        }));
        Ok(Self {
            _library: library,
            _gl_library: gl_library,
            get_display: symbol(library, b"eglGetDisplay\0")?,
            initialize: symbol(library, b"eglInitialize\0")?,
            choose_config: symbol(library, b"eglChooseConfig\0")?,
            bind_api: symbol(library, b"eglBindAPI\0")?,
            create_pbuffer_surface: symbol(library, b"eglCreatePbufferSurface\0")?,
            create_context: symbol(library, b"eglCreateContext\0")?,
            make_current: symbol(library, b"eglMakeCurrent\0")?,
            destroy_surface: symbol(library, b"eglDestroySurface\0")?,
            destroy_context: symbol(library, b"eglDestroyContext\0")?,
            terminate: symbol(library, b"eglTerminate\0")?,
            get_error: symbol(library, b"eglGetError\0")?,
            get_proc_address: symbol(library, b"eglGetProcAddress\0")?,
            get_platform_display_ext: optional_symbol(library, b"eglGetPlatformDisplayEXT\0"),
        })
    }

    fn proc_address(&self, symbol_name: &CStr) -> *const c_void {
        let from_egl = unsafe { (self.get_proc_address)(symbol_name.as_ptr()) };
        if !from_egl.is_null() {
            return from_egl;
        }
        unsafe {
            self._gl_library
                .get::<*const c_void>(symbol_name.to_bytes_with_nul())
                .map(|symbol| *symbol)
                .unwrap_or(ptr::null())
        }
    }

    fn error_hex(&self) -> String {
        let code = unsafe { (self.get_error)() };
        format!("0x{code:04x}")
    }
}

impl GlFns {
    fn load() -> Result<Self, String> {
        Ok(Self {
            read_pixels: gl_symbol("glReadPixels")?,
            pixel_storei: gl_symbol("glPixelStorei")?,
            finish: gl_symbol("glFinish")?,
        })
    }
}

unsafe impl Send for HardwareRenderContext {}
unsafe impl Sync for HardwareRenderContext {}

fn egl_fns() -> Result<&'static EglFns, String> {
    static EGL: OnceLock<Result<EglFns, String>> = OnceLock::new();
    EGL.get_or_init(EglFns::load).as_ref().map_err(Clone::clone)
}

fn create_display(egl: &EglFns) -> Result<EglDisplay, String> {
    if let Some(get_platform_display_ext) = egl.get_platform_display_ext {
        let display = unsafe {
            get_platform_display_ext(EGL_PLATFORM_SURFACELESS_MESA, ptr::null_mut(), ptr::null())
        };
        if initialize_display(egl, display).is_ok() {
            return Ok(display);
        }
    }

    let display = unsafe { (egl.get_display)(ptr::null_mut()) };
    initialize_display(egl, display)?;
    Ok(display)
}

fn initialize_display(egl: &EglFns, display: EglDisplay) -> Result<(), String> {
    if display.is_null() {
        return Err("eglGetDisplay returned null".to_owned());
    }
    let mut major = 0;
    let mut minor = 0;
    let ok = unsafe { (egl.initialize)(display, &mut major, &mut minor) };
    if ok == EGL_FALSE {
        return Err(format!("eglInitialize failed with {}", egl.error_hex()));
    }
    Ok(())
}

fn choose_config(egl: &EglFns, display: EglDisplay) -> Result<EglConfig, String> {
    let attributes = [
        EGL_SURFACE_TYPE,
        EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE,
        EGL_OPENGL_BIT,
        EGL_RED_SIZE,
        8,
        EGL_GREEN_SIZE,
        8,
        EGL_BLUE_SIZE,
        8,
        EGL_ALPHA_SIZE,
        8,
        EGL_NONE,
    ];
    let mut config = ptr::null_mut();
    let mut config_count = 0;
    let ok = unsafe {
        (egl.choose_config)(
            display,
            attributes.as_ptr(),
            &mut config,
            1,
            &mut config_count,
        )
    };
    if ok == EGL_FALSE || config_count == 0 || config.is_null() {
        return Err(format!("eglChooseConfig failed with {}", egl.error_hex()));
    }
    Ok(config)
}

fn create_surface(
    egl: &EglFns,
    display: EglDisplay,
    config: EglConfig,
) -> Result<EglSurface, String> {
    let attributes = [
        EGL_WIDTH,
        PBUFFER_SIZE.width,
        EGL_HEIGHT,
        PBUFFER_SIZE.height,
        EGL_NONE,
    ];
    let surface = unsafe { (egl.create_pbuffer_surface)(display, config, attributes.as_ptr()) };
    if surface.is_null() {
        return Err(format!(
            "eglCreatePbufferSurface failed with {}",
            egl.error_hex()
        ));
    }
    Ok(surface)
}

fn bind_opengl_api(egl: &EglFns) -> Result<(), String> {
    let ok = unsafe { (egl.bind_api)(EGL_OPENGL_API) };
    if ok == EGL_FALSE {
        return Err(format!(
            "eglBindAPI(OpenGL) failed with {}",
            egl.error_hex()
        ));
    }
    Ok(())
}

fn create_context(
    egl: &EglFns,
    display: EglDisplay,
    config: EglConfig,
    callback: &HwRenderCallback,
) -> Result<EglContext, String> {
    let attributes = context_attributes(callback);
    let context =
        unsafe { (egl.create_context)(display, config, ptr::null_mut(), attributes.as_ptr()) };
    if context.is_null() {
        return Err(format!("eglCreateContext failed with {}", egl.error_hex()));
    }
    Ok(context)
}

fn context_attributes(callback: &HwRenderCallback) -> Vec<EglInt> {
    if context_type(callback.context_type) == Some(HwContextType::OpenGLCore) {
        return vec![
            EGL_CONTEXT_MAJOR_VERSION,
            callback.version_major as EglInt,
            EGL_CONTEXT_MINOR_VERSION,
            callback.version_minor as EglInt,
            EGL_CONTEXT_OPENGL_PROFILE_MASK_KHR,
            EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT_KHR,
            EGL_NONE,
        ];
    }
    vec![EGL_NONE]
}

fn make_current(
    egl: &EglFns,
    display: EglDisplay,
    surface: EglSurface,
    context: EglContext,
) -> Result<(), String> {
    let ok = unsafe { (egl.make_current)(display, surface, surface, context) };
    if ok == EGL_FALSE {
        return Err(format!("eglMakeCurrent failed with {}", egl.error_hex()));
    }
    Ok(())
}

fn context_type(value: libc::c_uint) -> Option<HwContextType> {
    match value {
        value if value == HwContextType::OpenGL as libc::c_uint => Some(HwContextType::OpenGL),
        value if value == HwContextType::OpenGLCore as libc::c_uint => {
            Some(HwContextType::OpenGLCore)
        }
        value if value == HwContextType::OpenGLES2 as libc::c_uint => {
            Some(HwContextType::OpenGLES2)
        }
        value if value == HwContextType::OpenGLES3 as libc::c_uint => {
            Some(HwContextType::OpenGLES3)
        }
        value if value == HwContextType::OpenGLESVersion as libc::c_uint => {
            Some(HwContextType::OpenGLESVersion)
        }
        value if value == HwContextType::None as libc::c_uint => Some(HwContextType::None),
        _ => None,
    }
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

fn symbol<T>(library: &'static Library, name: &[u8]) -> Result<T, String>
where
    T: Copy,
{
    unsafe {
        library
            .get::<T>(name)
            .map(|symbol| *symbol)
            .map_err(|error| error.to_string())
    }
}

fn optional_symbol<T>(library: &'static Library, name: &[u8]) -> Option<T>
where
    T: Copy,
{
    unsafe { library.get::<T>(name).map(|symbol| *symbol).ok() }
}

fn gl_symbol<T>(name: &str) -> Result<T, String>
where
    T: Copy,
{
    let symbol_name = CString::new(name).map_err(|error| error.to_string())?;
    let pointer = egl_fns()?.proc_address(symbol_name.as_c_str());
    if pointer.is_null() {
        return Err(format!("missing GL symbol {name}"));
    }
    Ok(unsafe { std::mem::transmute_copy::<*const c_void, T>(&pointer) })
}

unsafe extern "C" fn get_current_framebuffer() -> libc::uintptr_t {
    0
}

unsafe extern "C" fn get_proc_address(sym: *const c_char) -> ProcAddressFn {
    if sym.is_null() {
        return missing_proc_stub;
    }
    let symbol_name = unsafe { CStr::from_ptr(sym) };
    let pointer = egl_fns()
        .map(|egl| egl.proc_address(symbol_name))
        .unwrap_or(ptr::null());
    if pointer.is_null() {
        return missing_proc_stub;
    }
    unsafe { std::mem::transmute::<*const c_void, ProcAddressFn>(pointer) }
}

unsafe extern "C" fn missing_proc_stub() {}
