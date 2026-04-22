// rust/core/host/callbacks/state/render.rs
//! Observation, display, minimap, and render-plan cache methods.

use crate::core::error::CoreError;
use crate::core::minimap::MinimapLayerRequest;
use crate::core::video::{
    ProcessedFramePlan, ProcessedFramePlanKey, ProcessedFramePlanRequest,
    build_processed_frame_plan, decode_frame, processed_frame, processed_frame_from_raw_into,
};

use super::super::stack::{
    StackedObservationBuffer, StackedObservationKey, StackedObservationRequest,
};
use super::{CallbackState, RenderPlanCacheEntry, RenderRequest, StackedObservationCacheEntry};

impl CallbackState {
    pub fn observation_frame(
        &mut self,
        aspect_ratio: f64,
        target_width: usize,
        target_height: usize,
        rgb: bool,
        crop: crate::core::video::VideoCrop,
        resize_filter: crate::core::video::VideoResizeFilter,
    ) -> Result<&[u8], CoreError> {
        // Prefer the raw-frame fast path so we can render directly into the
        // reusable output buffers without fully decoding to an intermediate
        // RGB frame first.
        if let Some((source_width, source_height)) = self
            .raw_frame
            .as_ref()
            .map(|raw_frame| (raw_frame.width, raw_frame.height))
        {
            let request = RenderRequest {
                source_width,
                source_height,
                aspect_ratio,
                target_width,
                target_height,
                rgb,
                crop,
                resize_filter,
            };
            let use_display_buffer = self.use_display_buffer(target_width, target_height);
            let plan = *self.render_plan(request)?;
            let raw_frame = self.raw_frame.as_ref().ok_or(CoreError::NoFrameAvailable)?;
            let target_buffer = if use_display_buffer {
                &mut self.display_buffer
            } else {
                &mut self.observation_buffer
            };
            processed_frame_from_raw_into(raw_frame, &plan, target_buffer)?;
            return Ok(target_buffer.as_slice());
        }

        let frame = self.frame().ok_or(CoreError::NoFrameAvailable)?;
        let rendered = processed_frame(
            frame,
            aspect_ratio,
            target_width,
            target_height,
            rgb,
            crop,
            resize_filter,
        )?;
        let use_display_buffer = self.use_display_buffer(target_width, target_height);
        let target_buffer = if use_display_buffer {
            &mut self.display_buffer
        } else {
            &mut self.observation_buffer
        };
        target_buffer.clear();
        target_buffer.extend_from_slice(&rendered);
        Ok(target_buffer.as_slice())
    }

    pub fn stacked_observation_frame(
        &mut self,
        request: StackedObservationRequest,
    ) -> Result<&[u8], CoreError> {
        let render_request = RenderRequest {
            source_width: self
                .raw_frame
                .as_ref()
                .map(|raw_frame| raw_frame.width)
                .or_else(|| self.frame.as_ref().map(|frame| frame.width))
                .ok_or(CoreError::NoFrameAvailable)?,
            source_height: self
                .raw_frame
                .as_ref()
                .map(|raw_frame| raw_frame.height)
                .or_else(|| self.frame.as_ref().map(|frame| frame.height))
                .ok_or(CoreError::NoFrameAvailable)?,
            aspect_ratio: request.aspect_ratio,
            target_width: request.target_width,
            target_height: request.target_height,
            rgb: request.rgb,
            crop: request.crop,
            resize_filter: request.resize_filter,
        };
        let frame_serial = self.frame_serial;
        let stack_key = StackedObservationKey {
            render_plan: render_request.plan_key(),
            frame_stack: request.frame_stack,
            stack_mode: request.stack_mode,
            extra_channels_per_pixel: usize::from(request.minimap_layer.is_some()),
        };
        self.render_observation_into_buffer(render_request)?;
        if let Some(minimap_request) = request.minimap_layer {
            self.render_minimap_layer_into_buffer(minimap_request)?;
        }

        let observation_buffer = self.observation_buffer.as_slice();
        let minimap_layer = request
            .minimap_layer
            .map(|_| self.minimap_buffer.as_slice());
        let stack_index = self
            .stacked_observation_buffers
            .iter()
            .position(|entry| entry.key == stack_key)
            .unwrap_or_else(|| {
                self.stacked_observation_buffers
                    .push(StackedObservationCacheEntry {
                        key: stack_key,
                        buffer: StackedObservationBuffer::new(
                            observation_buffer.len(),
                            request.frame_stack,
                            if request.rgb { 3 } else { 1 },
                            request.stack_mode,
                            usize::from(request.minimap_layer.is_some()),
                        ),
                    });
                self.stacked_observation_buffers.len() - 1
            });
        let stack_buffer = &mut self.stacked_observation_buffers[stack_index].buffer;
        stack_buffer.update(observation_buffer, frame_serial, minimap_layer)?;
        Ok(stack_buffer.as_slice())
    }

    pub(super) fn render_plan(
        &mut self,
        request: RenderRequest,
    ) -> Result<&ProcessedFramePlan, CoreError> {
        // Observation and display targets are stable for a run, so the sampling
        // plan is cached once per source/target combination.
        let key = request.plan_key();
        if let Some(index) = self.render_plans.iter().position(|entry| entry.key == key) {
            return Ok(&self.render_plans[index].plan);
        }

        self.render_plans.push(RenderPlanCacheEntry {
            key,
            plan: build_processed_frame_plan(ProcessedFramePlanRequest {
                source_width: request.source_width,
                source_height: request.source_height,
                aspect_ratio: request.aspect_ratio,
                target_width: request.target_width,
                target_height: request.target_height,
                rgb: request.rgb,
                crop: request.crop,
                resize_filter: request.resize_filter,
            })?,
        });
        let index = self.render_plans.len() - 1;
        Ok(&self.render_plans[index].plan)
    }

    pub(super) fn render_observation_into_buffer(
        &mut self,
        request: RenderRequest,
    ) -> Result<(), CoreError> {
        if self
            .raw_frame
            .as_ref()
            .map(|raw_frame| (raw_frame.width, raw_frame.height))
            .is_some()
        {
            let plan = *self.render_plan(request)?;
            let raw_frame = self.raw_frame.as_ref().ok_or(CoreError::NoFrameAvailable)?;
            processed_frame_from_raw_into(raw_frame, &plan, &mut self.observation_buffer)?;
            return Ok(());
        }

        let frame = self.frame().ok_or(CoreError::NoFrameAvailable)?;
        let rendered = processed_frame(
            frame,
            request.aspect_ratio,
            request.target_width,
            request.target_height,
            request.rgb,
            request.crop,
            request.resize_filter,
        )?;
        self.observation_buffer.clear();
        self.observation_buffer.extend_from_slice(&rendered);
        Ok(())
    }

    pub(super) fn render_minimap_layer_into_buffer(
        &mut self,
        request: MinimapLayerRequest,
    ) -> Result<(), CoreError> {
        if let Some(raw_frame) = self.raw_frame.as_ref() {
            self.minimap_renderer.render_from_raw_into(
                raw_frame,
                request,
                &mut self.minimap_buffer,
            )?;
            return Ok(());
        }

        if self.frame.is_none() {
            self.frame = self.raw_frame.as_ref().and_then(decode_frame);
        }
        let frame = self.frame.as_ref().ok_or(CoreError::NoFrameAvailable)?;
        self.minimap_renderer
            .render_from_frame_into(frame, request, &mut self.minimap_buffer)?;
        Ok(())
    }

    pub(in crate::core::host::callbacks) fn clear_stacked_observation_buffers(&mut self) {
        for entry in &mut self.stacked_observation_buffers {
            entry.buffer.clear();
        }
    }

    fn use_display_buffer(&self, target_width: usize, target_height: usize) -> bool {
        if let Some((frame_width, frame_height)) = self.geometry {
            if target_width == frame_width {
                return true;
            }
            if target_width <= frame_width && target_height <= frame_height {
                return false;
            }
        }
        target_width >= target_height
    }
}

impl RenderRequest {
    fn plan_key(self) -> ProcessedFramePlanKey {
        ProcessedFramePlanKey {
            source_width: self.source_width,
            source_height: self.source_height,
            crop: self.crop,
            aspect_ratio_bits: self.aspect_ratio.to_bits(),
            target_width: self.target_width,
            target_height: self.target_height,
            rgb: self.rgb,
            resize_filter: self.resize_filter,
        }
    }
}
