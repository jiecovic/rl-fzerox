# Policy Inputs

The policy receives rendered game frames and, in managed runs, a small scalar
state vector.

Managed runs use `image_state`: an image stack plus ordered state features from
RAM/telemetry. The Run Manager exposes those features under Observation.

Main state components:

- `vehicle_state`: speed, energy, boost, airborne, and sliding state.
- `machine_context`: vehicle stats, weight, and engine slider.
- `track_position`: progress and track-edge features.
- `surface_state`: dirt, ice, refill, and other surface flags.
- `course_context`: built-in course identity.
- `control_history`: previous action values.

## Machine Context

`machine_context` describes the selected vehicle/setup. It includes body, boost,
grip, weight, and `machine_context.engine`.

`machine_context.engine` is the current engine slider value normalized to
`0..1`. It lets one policy see which setup is active when a run uses variable
engine settings.

Engine modes and tuning are documented in [Engine tuning](engine_tuning.md).
