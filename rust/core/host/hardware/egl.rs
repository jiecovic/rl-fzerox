// rust/core/host/hardware/egl.rs
//! EGL symbol loading, configuration, and context helpers.

use std::ffi::{CStr, c_char, c_void};
use std::ptr;
use std::sync::OnceLock;

use libloading::Library;
use libretro_sys::{HwContextType, HwRenderCallback};

use crate::core::error::HardwareRenderError;

const EGL_LIBRARY: &str = "libEGL.so.1";
const GL_LIBRARY: &str = "libGL.so.1";

// EGL enum values mirror the external C headers.
#[derive(Clone, Copy)]
struct EglProtocolValues {
    false_value: EglBoolean,
    none: EglInt,
    red_size: EglInt,
    green_size: EglInt,
    blue_size: EglInt,
    alpha_size: EglInt,
    surface_type: EglInt,
    pbuffer_bit: EglInt,
    renderable_type: EglInt,
    opengl_bit: EglInt,
    width: EglInt,
    height: EglInt,
    opengl_api: EglEnum,
    context_major_version: EglInt,
    context_minor_version: EglInt,
    context_opengl_profile_mask: EglInt,
    context_opengl_core_profile_bit: EglInt,
    platform_surfaceless_mesa: EglEnum,
}

const EGL_VALUES: EglProtocolValues = EglProtocolValues {
    false_value: 0,
    none: 0x3038,
    red_size: 0x3024,
    green_size: 0x3023,
    blue_size: 0x3022,
    alpha_size: 0x3021,
    surface_type: 0x3033,
    pbuffer_bit: 0x0001,
    renderable_type: 0x3040,
    opengl_bit: 0x0008,
    width: 0x3057,
    height: 0x3056,
    opengl_api: 0x30A2,
    context_major_version: 0x3098,
    context_minor_version: 0x30FB,
    context_opengl_profile_mask: 0x30FD,
    context_opengl_core_profile_bit: 0x0000_0001,
    platform_surfaceless_mesa: 0x31DD,
};

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
pub(super) type EglInt = libc::c_int;
pub(super) type EglDisplay = *mut c_void;
pub(super) type EglConfig = *mut c_void;
pub(super) type EglSurface = *mut c_void;
pub(super) type EglContext = *mut c_void;

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

#[derive(Clone, Copy)]
pub(super) struct EglFns {
    _library: &'static Library,
    gl_library: &'static Library,
    get_display: EglGetDisplay,
    initialize: EglInitialize,
    choose_config: EglChooseConfig,
    bind_api: EglBindApi,
    create_pbuffer_surface: EglCreatePbufferSurface,
    create_context: EglCreateContext,
    pub(super) make_current: EglMakeCurrent,
    pub(super) destroy_surface: EglDestroySurface,
    pub(super) destroy_context: EglDestroyContext,
    pub(super) terminate: EglTerminate,
    get_error: EglGetError,
    get_proc_address: EglGetProcAddress,
    get_platform_display_ext: Option<EglGetPlatformDisplayExt>,
}

impl EglFns {
    fn load() -> Result<Self, HardwareRenderError> {
        let library = load_library(EGL_LIBRARY)?;
        let gl_library = load_library(GL_LIBRARY)?;
        Ok(Self {
            _library: library,
            gl_library,
            get_display: symbol(EGL_LIBRARY, library, b"eglGetDisplay\0")?,
            initialize: symbol(EGL_LIBRARY, library, b"eglInitialize\0")?,
            choose_config: symbol(EGL_LIBRARY, library, b"eglChooseConfig\0")?,
            bind_api: symbol(EGL_LIBRARY, library, b"eglBindAPI\0")?,
            create_pbuffer_surface: symbol(EGL_LIBRARY, library, b"eglCreatePbufferSurface\0")?,
            create_context: symbol(EGL_LIBRARY, library, b"eglCreateContext\0")?,
            make_current: symbol(EGL_LIBRARY, library, b"eglMakeCurrent\0")?,
            destroy_surface: symbol(EGL_LIBRARY, library, b"eglDestroySurface\0")?,
            destroy_context: symbol(EGL_LIBRARY, library, b"eglDestroyContext\0")?,
            terminate: symbol(EGL_LIBRARY, library, b"eglTerminate\0")?,
            get_error: symbol(EGL_LIBRARY, library, b"eglGetError\0")?,
            get_proc_address: symbol(EGL_LIBRARY, library, b"eglGetProcAddress\0")?,
            get_platform_display_ext: optional_symbol(library, b"eglGetPlatformDisplayEXT\0"),
        })
    }

    pub(super) fn proc_address(&self, symbol_name: &CStr) -> *const c_void {
        // SAFETY: `symbol_name` is NUL-terminated and comes from our GL loader
        // or libretro's get-proc-address callback.
        let from_egl = unsafe { (self.get_proc_address)(symbol_name.as_ptr()) };
        if !from_egl.is_null() {
            return from_egl;
        }
        // SAFETY: `libloading` validates the symbol lookup; the raw pointer is
        // returned to the caller, which casts it to the expected GL signature.
        unsafe {
            self.gl_library
                .get::<*const c_void>(symbol_name.to_bytes_with_nul())
                .map(|symbol| *symbol)
                .unwrap_or(ptr::null())
        }
    }

    fn error_hex(&self) -> String {
        // SAFETY: `eglGetError` takes no arguments and is valid after EGL load.
        let code = unsafe { (self.get_error)() };
        format!("0x{code:04x}")
    }
}

pub(super) fn egl_fns() -> Result<&'static EglFns, HardwareRenderError> {
    static EGL: OnceLock<Result<EglFns, HardwareRenderError>> = OnceLock::new();
    EGL.get_or_init(EglFns::load).as_ref().map_err(Clone::clone)
}

pub(super) fn create_display(egl: &EglFns) -> Result<EglDisplay, HardwareRenderError> {
    if let Some(get_platform_display_ext) = egl.get_platform_display_ext {
        // SAFETY: The optional platform-display entry point is resolved from
        // EGL and accepts null native-display/attribute pointers for surfaceless.
        let display = unsafe {
            get_platform_display_ext(
                EGL_VALUES.platform_surfaceless_mesa,
                ptr::null_mut(),
                ptr::null(),
            )
        };
        if initialize_display(egl, display).is_ok() {
            return Ok(display);
        }
    }

    // SAFETY: `eglGetDisplay` accepts a null native display to request the
    // default display.
    let display = unsafe { (egl.get_display)(ptr::null_mut()) };
    initialize_display(egl, display)?;
    Ok(display)
}

fn initialize_display(egl: &EglFns, display: EglDisplay) -> Result<(), HardwareRenderError> {
    if display.is_null() {
        return Err(HardwareRenderError::NullHandle {
            call: "eglGetDisplay",
        });
    }
    let mut major = 0;
    let mut minor = 0;
    // SAFETY: `display` was returned by EGL and the version output pointers are
    // valid local variables.
    let ok = unsafe { (egl.initialize)(display, &mut major, &mut minor) };
    if ok == EGL_VALUES.false_value {
        return Err(HardwareRenderError::EglCallFailed {
            call: "eglInitialize",
            egl_error: egl.error_hex(),
        });
    }
    Ok(())
}

pub(super) fn choose_config(
    egl: &EglFns,
    display: EglDisplay,
) -> Result<EglConfig, HardwareRenderError> {
    let attributes = [
        EGL_VALUES.surface_type,
        EGL_VALUES.pbuffer_bit,
        EGL_VALUES.renderable_type,
        EGL_VALUES.opengl_bit,
        EGL_VALUES.red_size,
        8,
        EGL_VALUES.green_size,
        8,
        EGL_VALUES.blue_size,
        8,
        EGL_VALUES.alpha_size,
        8,
        EGL_VALUES.none,
    ];
    let mut config = ptr::null_mut();
    let mut config_count = 0;
    // SAFETY: Attribute and output pointers are valid for this call, and EGL
    // writes at most one config because `config_size` is 1.
    let ok = unsafe {
        (egl.choose_config)(
            display,
            attributes.as_ptr(),
            &mut config,
            1,
            &mut config_count,
        )
    };
    if ok == EGL_VALUES.false_value {
        return Err(HardwareRenderError::EglCallFailed {
            call: "eglChooseConfig",
            egl_error: egl.error_hex(),
        });
    }
    if config_count == 0 || config.is_null() {
        return Err(HardwareRenderError::NoConfig {
            call: "eglChooseConfig",
        });
    }
    Ok(config)
}

pub(super) fn create_surface(
    egl: &EglFns,
    display: EglDisplay,
    config: EglConfig,
) -> Result<EglSurface, HardwareRenderError> {
    let attributes = [
        EGL_VALUES.width,
        PBUFFER_SIZE.width,
        EGL_VALUES.height,
        PBUFFER_SIZE.height,
        EGL_VALUES.none,
    ];
    // SAFETY: `display`/`config` were returned by EGL and attributes is
    // NUL-terminated with EGL_NONE.
    let surface = unsafe { (egl.create_pbuffer_surface)(display, config, attributes.as_ptr()) };
    if surface.is_null() {
        return Err(HardwareRenderError::EglCallFailed {
            call: "eglCreatePbufferSurface",
            egl_error: egl.error_hex(),
        });
    }
    Ok(surface)
}

pub(super) fn bind_opengl_api(egl: &EglFns) -> Result<(), HardwareRenderError> {
    // SAFETY: Binding the OpenGL API is a process-local EGL operation with no
    // borrowed pointers.
    let ok = unsafe { (egl.bind_api)(EGL_VALUES.opengl_api) };
    if ok == EGL_VALUES.false_value {
        return Err(HardwareRenderError::EglCallFailed {
            call: "eglBindAPI(OpenGL)",
            egl_error: egl.error_hex(),
        });
    }
    Ok(())
}

pub(super) fn create_context(
    egl: &EglFns,
    display: EglDisplay,
    config: EglConfig,
    callback: &HwRenderCallback,
) -> Result<EglContext, HardwareRenderError> {
    let attributes = context_attributes(callback);
    // SAFETY: `display`/`config` were returned by EGL and attributes is
    // NUL-terminated with EGL_NONE.
    let context =
        unsafe { (egl.create_context)(display, config, ptr::null_mut(), attributes.as_ptr()) };
    if context.is_null() {
        return Err(HardwareRenderError::EglCallFailed {
            call: "eglCreateContext",
            egl_error: egl.error_hex(),
        });
    }
    Ok(context)
}

fn context_attributes(callback: &HwRenderCallback) -> Vec<EglInt> {
    if context_type(callback.context_type) == Some(HwContextType::OpenGLCore) {
        return vec![
            EGL_VALUES.context_major_version,
            callback.version_major as EglInt,
            EGL_VALUES.context_minor_version,
            callback.version_minor as EglInt,
            EGL_VALUES.context_opengl_profile_mask,
            EGL_VALUES.context_opengl_core_profile_bit,
            EGL_VALUES.none,
        ];
    }
    vec![EGL_VALUES.none]
}

pub(super) fn make_current(
    egl: &EglFns,
    display: EglDisplay,
    surface: EglSurface,
    context: EglContext,
) -> Result<(), HardwareRenderError> {
    // SAFETY: `display`, `surface`, and `context` are EGL handles owned by the
    // hardware-render context.
    let ok = unsafe { (egl.make_current)(display, surface, surface, context) };
    if ok == EGL_VALUES.false_value {
        return Err(HardwareRenderError::EglCallFailed {
            call: "eglMakeCurrent",
            egl_error: egl.error_hex(),
        });
    }
    Ok(())
}

pub(super) fn context_type(value: libc::c_uint) -> Option<HwContextType> {
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

fn load_library(name: &'static str) -> Result<&'static Library, HardwareRenderError> {
    // SAFETY: Loading process-global EGL/GL shared libraries is required to
    // resolve the C entry points used by the hardware-render frontend.
    unsafe { Library::new(name) }
        .map(|library| Box::leak(Box::new(library)) as &'static Library)
        .map_err(|error| HardwareRenderError::LoadLibrary {
            library: name,
            message: error.to_string(),
        })
}

fn symbol<T>(
    library_name: &'static str,
    library: &'static Library,
    name: &[u8],
) -> Result<T, HardwareRenderError>
where
    T: Copy,
{
    // SAFETY: `name` is NUL-terminated and `T` is chosen by the caller to match
    // the requested C symbol signature.
    unsafe {
        library
            .get::<T>(name)
            .map(|symbol| *symbol)
            .map_err(|error| HardwareRenderError::MissingSymbol {
                library: library_name,
                symbol: printable_symbol(name),
                message: error.to_string(),
            })
    }
}

fn optional_symbol<T>(library: &'static Library, name: &[u8]) -> Option<T>
where
    T: Copy,
{
    // SAFETY: Same invariant as `symbol`; failure is represented as `None`.
    unsafe { library.get::<T>(name).map(|symbol| *symbol).ok() }
}

fn printable_symbol(symbol_name: &[u8]) -> String {
    symbol_name
        .strip_suffix(b"\0")
        .unwrap_or(symbol_name)
        .iter()
        .copied()
        .map(char::from)
        .collect()
}
