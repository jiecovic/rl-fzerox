// rust/bindings/input.rs
use libretro_sys::{
    DEVICE_ID_JOYPAD_A, DEVICE_ID_JOYPAD_B, DEVICE_ID_JOYPAD_DOWN, DEVICE_ID_JOYPAD_L,
    DEVICE_ID_JOYPAD_L2, DEVICE_ID_JOYPAD_L3, DEVICE_ID_JOYPAD_LEFT, DEVICE_ID_JOYPAD_R,
    DEVICE_ID_JOYPAD_R2, DEVICE_ID_JOYPAD_R3, DEVICE_ID_JOYPAD_RIGHT, DEVICE_ID_JOYPAD_SELECT,
    DEVICE_ID_JOYPAD_START, DEVICE_ID_JOYPAD_UP, DEVICE_ID_JOYPAD_X, DEVICE_ID_JOYPAD_Y,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyTuple};

const JOYPAD_BIT_WIDTH: u32 = u16::BITS;

#[pyfunction]
#[pyo3(signature = (*buttons))]
pub fn joypad_mask(buttons: &Bound<'_, PyTuple>) -> PyResult<u16> {
    let mut mask = 0_u16;

    for button in buttons.iter() {
        let button_id: u32 = button.extract()?;
        if button_id >= JOYPAD_BIT_WIDTH {
            return Err(PyValueError::new_err(format!(
                "Invalid joypad button id {button_id}; expected 0..{}",
                JOYPAD_BIT_WIDTH - 1
            )));
        }
        mask |= 1_u16 << button_id;
    }

    Ok(mask)
}

pub fn register_input_api(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("JOYPAD_B", DEVICE_ID_JOYPAD_B)?;
    module.add("JOYPAD_Y", DEVICE_ID_JOYPAD_Y)?;
    module.add("JOYPAD_SELECT", DEVICE_ID_JOYPAD_SELECT)?;
    module.add("JOYPAD_START", DEVICE_ID_JOYPAD_START)?;
    module.add("JOYPAD_UP", DEVICE_ID_JOYPAD_UP)?;
    module.add("JOYPAD_DOWN", DEVICE_ID_JOYPAD_DOWN)?;
    module.add("JOYPAD_LEFT", DEVICE_ID_JOYPAD_LEFT)?;
    module.add("JOYPAD_RIGHT", DEVICE_ID_JOYPAD_RIGHT)?;
    module.add("JOYPAD_A", DEVICE_ID_JOYPAD_A)?;
    module.add("JOYPAD_X", DEVICE_ID_JOYPAD_X)?;
    module.add("JOYPAD_L", DEVICE_ID_JOYPAD_L)?;
    module.add("JOYPAD_R", DEVICE_ID_JOYPAD_R)?;
    module.add("JOYPAD_L2", DEVICE_ID_JOYPAD_L2)?;
    module.add("JOYPAD_R2", DEVICE_ID_JOYPAD_R2)?;
    module.add("JOYPAD_L3", DEVICE_ID_JOYPAD_L3)?;
    module.add("JOYPAD_R3", DEVICE_ID_JOYPAD_R3)?;
    module.add_function(wrap_pyfunction!(joypad_mask, module)?)?;
    Ok(())
}
