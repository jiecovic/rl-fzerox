// rust/core/host/runtime/telemetry.rs
//! Runtime telemetry reads from the emulated system RAM.

use crate::core::error::CoreError;
use crate::core::telemetry::{StepTelemetrySample, TelemetrySnapshot};

use super::host::Host;

impl Host {
    pub fn telemetry(&mut self) -> Result<TelemetrySnapshot, CoreError> {
        let system_ram = self.system_ram_slice()?;
        crate::core::telemetry::read_snapshot(system_ram)
    }

    pub(super) fn telemetry_sample(&mut self) -> Result<StepTelemetrySample, CoreError> {
        let system_ram = self.system_ram_slice()?;
        crate::core::telemetry::read_step_sample(system_ram)
    }
}
