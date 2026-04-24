from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples, TensorDict
from stable_baselines3.common.vec_env import VecNormalize

from fzerox_emulator import ObservationStackMode, stacked_observation_channels
from fzerox_emulator.arrays import Float32Array, Int64Array, NumpyArray, UInt8Array
from rl_fzerox.core.config.schema import EnvConfig, TrainConfig
from rl_fzerox.core.domain.training_algorithms import TRAINING_ALGORITHMS

LazyReplayStackMode = Literal["rgb", "gray", "luma_chroma"]
SUPPORTED_LAZY_REPLAY_STACK_MODES: frozenset[LazyReplayStackMode] = frozenset(
    {"rgb", "gray", "luma_chroma"}
)


class LazyMaskableReplaySamples(NamedTuple):
    """Dict replay samples with lazy image reconstruction and action masks."""

    observations: TensorDict
    actions: th.Tensor
    next_observations: TensorDict
    dones: th.Tensor
    rewards: th.Tensor
    action_masks: th.Tensor
    next_action_masks: th.Tensor
    discounts: th.Tensor | None = None


@dataclass(frozen=True, slots=True)
class LazyImageReplayLayout:
    """Static image-layout metadata needed to compress and rebuild stacks."""

    image_shape: tuple[int, int, int]
    state_shape: tuple[int, ...]
    frame_stack: int
    stack_mode: LazyReplayStackMode
    minimap_layer: bool
    current_slice_channels: int
    channels_first: bool

    @property
    def height(self) -> int:
        return self.image_shape[1] if self.channels_first else self.image_shape[0]

    @property
    def width(self) -> int:
        return self.image_shape[2] if self.channels_first else self.image_shape[1]

    @property
    def image_channels(self) -> int:
        return self.image_shape[0] if self.channels_first else self.image_shape[2]

    @property
    def minimap_channels(self) -> int:
        return 1 if self.minimap_layer else 0

    @property
    def stacked_frame_channels(self) -> int:
        return self.image_channels - self.minimap_channels

    def split_image_batch(
        self,
        image_batch: NumpyArray,
    ) -> tuple[UInt8Array, UInt8Array | None]:
        """Return the per-step image slice and optional minimap slice."""

        frame_end = self.stacked_frame_channels
        current_start = frame_end - self.current_slice_channels
        if self.channels_first:
            current_slice = _as_uint8(image_batch[:, current_start:frame_end, :, :])
        else:
            current_slice = _as_uint8(image_batch[..., current_start:frame_end])
        if not self.minimap_layer:
            return current_slice, None
        if self.channels_first:
            minimap_slice = _as_uint8(image_batch[:, frame_end : frame_end + 1, :, :])
        else:
            minimap_slice = _as_uint8(image_batch[..., frame_end : frame_end + 1])
        return current_slice, minimap_slice


def resolve_sac_replay_buffer(
    *,
    train_config: TrainConfig,
    env_config: EnvConfig,
    effective_algorithm: str,
):
    """Return a project-owned replay buffer for supported dict image-state SAC runs."""

    if not train_config.optimize_memory_usage:
        return None, {}
    if env_config.observation.mode != "image_state":
        return None, {}

    replay_kwargs = {
        "frame_stack": env_config.observation.frame_stack,
        "stack_mode": env_config.observation.stack_mode,
        "minimap_layer": env_config.observation.minimap_layer,
    }
    if effective_algorithm == TRAINING_ALGORITHMS.maskable_hybrid_action_sac:
        return LazyMaskableReplayBuffer, replay_kwargs
    if effective_algorithm in TRAINING_ALGORITHMS.sac_family:
        return LazyImageStateReplayBuffer, replay_kwargs
    return None, {}


class LazyImageStateReplayBuffer(BaseBuffer):
    """Replay buffer that stores one per-step image slice and rebuilds stacks on sample."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        *,
        frame_stack: int,
        stack_mode: LazyReplayStackMode,
        minimap_layer: bool,
    ) -> None:
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
        )
        self.buffer_size = max(buffer_size // n_envs, 1)
        self.optimize_memory_usage = bool(optimize_memory_usage)
        image_space = self._image_space()
        state_space = self._state_space()
        image_shape = _image_shape(image_space)
        expected_channels = stacked_observation_channels(
            3,
            frame_stack=frame_stack,
            stack_mode=stack_mode,
            minimap_layer=minimap_layer,
        )
        self.layout = LazyImageReplayLayout(
            image_shape=image_shape,
            state_shape=state_space.shape,
            frame_stack=frame_stack,
            stack_mode=stack_mode,
            minimap_layer=minimap_layer,
            current_slice_channels=_current_slice_channels(stack_mode),
            channels_first=_is_channels_first(image_shape, expected_channels),
        )
        if self.layout.image_channels != expected_channels:
            raise RuntimeError(
                "lazy SAC replay expected image observation channels "
                f"{expected_channels}, got {self.layout.image_channels}"
            )

        self.image_slices = np.zeros(
            (self.buffer_size, self.n_envs, *self._image_slice_shape()),
            dtype=np.uint8,
        )
        self.minimap_slices = (
            np.zeros(
                (self.buffer_size, self.n_envs, *self._minimap_slice_shape()),
                dtype=np.uint8,
            )
            if self.layout.minimap_layer
            else None
        )
        self.states = np.zeros(
            (self.buffer_size, self.n_envs, *self.layout.state_shape),
            dtype=self._state_space().dtype,
        )
        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim),
            dtype=_action_storage_dtype(action_space.dtype),
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.bool_)
        self._next_episode_starts = np.ones(self.n_envs, dtype=np.bool_)
        self._terminal_next_images: dict[tuple[int, int], UInt8Array] = {}
        self._terminal_next_states: dict[tuple[int, int], NumpyArray] = {}

    def reset(self) -> None:
        super().reset()
        self._terminal_next_images.clear()
        self._terminal_next_states.clear()
        self._next_episode_starts.fill(True)

    def add(
        self,
        obs: dict[str, NumpyArray],
        next_obs: dict[str, NumpyArray],
        action: NumpyArray,
        reward: NumpyArray,
        done: NumpyArray,
        infos: list[dict[str, object]],
    ) -> None:
        image_batch, minimap_batch = self.layout.split_image_batch(
            np.asarray(obs["image"], dtype=np.uint8)
        )
        self._clear_terminal_overrides(self.pos)
        self.image_slices[self.pos] = image_batch
        if self.minimap_slices is not None and minimap_batch is not None:
            self.minimap_slices[self.pos] = minimap_batch
        self.states[self.pos] = np.asarray(obs["state"], dtype=self._state_space().dtype)
        self.episode_starts[self.pos] = self._next_episode_starts

        action_array = np.asarray(action).reshape((self.n_envs, self.action_dim))
        self.actions[self.pos] = np.array(action_array)
        self.rewards[self.pos] = np.asarray(reward, dtype=np.float32)
        done_array = np.asarray(done, dtype=np.float32)
        self.dones[self.pos] = done_array
        self.timeouts[self.pos] = np.asarray(
            [bool(info.get("TimeLimit.truncated", False)) for info in infos],
            dtype=np.float32,
        )

        next_images = np.asarray(next_obs["image"], dtype=np.uint8)
        next_states = np.asarray(next_obs["state"], dtype=self._state_space().dtype)
        for env_index, done_value in enumerate(done_array.astype(bool)):
            if done_value:
                self._terminal_next_images[(self.pos, env_index)] = np.array(
                    next_images[env_index],
                    copy=True,
                )
                self._terminal_next_states[(self.pos, env_index)] = np.array(
                    next_states[env_index],
                    copy=True,
                )
        self._next_episode_starts = done_array.astype(np.bool_)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(
        self,
        batch_size: int,
        env: VecNormalize | None = None,
    ) -> Any:  # noqa: ANN401
        batch_inds, env_indices = self._sample_transition_pairs(batch_size)
        return self._get_samples_for_pairs(batch_inds, env_indices, env)

    def _get_samples(
        self,
        batch_inds: Int64Array,
        env: VecNormalize | None = None,
    ) -> Any:  # noqa: ANN401
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        return self._get_samples_for_pairs(batch_inds, env_indices, env)

    def _get_samples_for_pairs(
        self,
        batch_inds: Int64Array,
        env_indices: Int64Array,
        env: VecNormalize | None = None,
    ) -> DictReplayBufferSamples:
        observations_np = {
            "image": self._build_image_batch(batch_inds, env_indices, next_observation=False),
            "state": self.states[batch_inds, env_indices],
        }
        next_observations_np = {
            "image": self._build_image_batch(batch_inds, env_indices, next_observation=True),
            "state": self._build_state_batch(batch_inds, env_indices, next_observation=True),
        }
        normalized_observations = self._normalize_obs(observations_np, env)
        normalized_next_observations = self._normalize_obs(next_observations_np, env)
        assert isinstance(normalized_observations, dict)
        assert isinstance(normalized_next_observations, dict)
        return DictReplayBufferSamples(
            observations={
                key: self.to_torch(value) for key, value in normalized_observations.items()
            },
            actions=self.to_torch(self.actions[batch_inds, env_indices]),
            next_observations={
                key: self.to_torch(value) for key, value in normalized_next_observations.items()
            },
            dones=self.to_torch(
                self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])
            ).reshape(-1, 1),
            rewards=self.to_torch(
                self._normalize_reward(
                    self.rewards[batch_inds, env_indices].reshape(-1, 1),
                    env,
                )
            ),
        )

    def _sample_transition_pairs(self, batch_size: int) -> tuple[Int64Array, Int64Array]:
        slot_indices, env_indices = self._valid_transition_pairs()
        if len(slot_indices) == 0:
            raise ValueError(
                "Cannot sample from replay buffer before a valid next observation exists"
            )
        choice = np.random.randint(0, len(slot_indices), size=batch_size)
        return slot_indices[choice], env_indices[choice]

    def _valid_transition_pairs(self) -> tuple[Int64Array, Int64Array]:
        size = self.size()
        if size == 0:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

        if self.full:
            slots = np.arange(self.buffer_size, dtype=np.int64)
            latest_slot = (self.pos - 1) % self.buffer_size
            latest_row = latest_slot
        else:
            slots = np.arange(self.pos, dtype=np.int64)
            latest_slot = self.pos - 1
            latest_row = len(slots) - 1

        valid = np.ones((len(slots), self.n_envs), dtype=np.bool_)
        latest_terminal = np.asarray(
            [
                self._has_terminal_next_observation(latest_slot, env_index)
                for env_index in range(self.n_envs)
            ],
            dtype=np.bool_,
        )
        valid[latest_row] = latest_terminal
        slot_rows, env_indices = np.nonzero(valid)
        return slots[slot_rows], env_indices.astype(np.int64, copy=False)

    def _build_image_batch(
        self,
        batch_inds: Int64Array,
        env_indices: Int64Array,
        *,
        next_observation: bool,
    ) -> UInt8Array:
        batch = np.empty((len(batch_inds), *self.layout.image_shape), dtype=np.uint8)
        for sample_index, (slot_index, env_index) in enumerate(
            zip(batch_inds.tolist(), env_indices.tolist(), strict=True)
        ):
            if next_observation and self._has_terminal_next_observation(slot_index, env_index):
                batch[sample_index] = self._terminal_next_images[(slot_index, env_index)]
                continue
            reference_slot = self._next_slot(slot_index) if next_observation else slot_index
            batch[sample_index] = self._rebuild_stacked_image(reference_slot, env_index)
        return batch

    def _build_state_batch(
        self,
        batch_inds: Int64Array,
        env_indices: Int64Array,
        *,
        next_observation: bool,
    ) -> NumpyArray:
        batch = np.empty(
            (len(batch_inds), *self.layout.state_shape), dtype=self._state_space().dtype
        )
        for sample_index, (slot_index, env_index) in enumerate(
            zip(batch_inds.tolist(), env_indices.tolist(), strict=True)
        ):
            if next_observation and self._has_terminal_next_observation(slot_index, env_index):
                batch[sample_index] = self._terminal_next_states[(slot_index, env_index)]
                continue
            reference_slot = self._next_slot(slot_index) if next_observation else slot_index
            batch[sample_index] = self.states[reference_slot, env_index]
        return batch

    def _rebuild_stacked_image(self, slot_index: int, env_index: int) -> UInt8Array:
        history_indices = self._stack_history_indices(slot_index, env_index)
        frame_slices = self.image_slices[np.asarray(history_indices, dtype=np.intp), env_index]
        if self.layout.channels_first:
            stacked = np.transpose(frame_slices, (1, 0, 2, 3)).reshape(
                self.layout.stacked_frame_channels,
                self.layout.height,
                self.layout.width,
            )
            if self.minimap_slices is None:
                return np.ascontiguousarray(stacked)
            return np.ascontiguousarray(
                np.concatenate([stacked, self.minimap_slices[slot_index, env_index]], axis=0)
            )

        stacked = np.transpose(frame_slices, (1, 2, 0, 3)).reshape(
            self.layout.height,
            self.layout.width,
            self.layout.stacked_frame_channels,
        )
        if self.minimap_slices is None:
            return np.ascontiguousarray(stacked)
        return np.ascontiguousarray(
            np.concatenate([stacked, self.minimap_slices[slot_index, env_index]], axis=2)
        )

    def _stack_history_indices(self, slot_index: int, env_index: int) -> list[int]:
        history = [slot_index]
        reference = slot_index
        for _ in range(self.layout.frame_stack - 1):
            if self.episode_starts[reference, env_index]:
                history.append(reference)
                continue
            reference = (reference - 1) % self.buffer_size
            history.append(reference)
        history.reverse()
        return history

    def _clear_terminal_overrides(self, slot_index: int) -> None:
        for env_index in range(self.n_envs):
            self._terminal_next_images.pop((slot_index, env_index), None)
            self._terminal_next_states.pop((slot_index, env_index), None)

    def _has_terminal_next_observation(self, slot_index: int, env_index: int) -> bool:
        key = (slot_index, env_index)
        has_image = key in self._terminal_next_images
        has_state = key in self._terminal_next_states
        if has_image != has_state:
            raise RuntimeError(
                "lazy SAC replay terminal next-observation overrides are inconsistent"
            )
        return has_image

    def _next_slot(self, slot_index: int) -> int:
        return (slot_index + 1) % self.buffer_size

    def _image_slice_shape(self) -> tuple[int, int, int]:
        if self.layout.channels_first:
            return (self.layout.current_slice_channels, self.layout.height, self.layout.width)
        return (self.layout.height, self.layout.width, self.layout.current_slice_channels)

    def _minimap_slice_shape(self) -> tuple[int, int, int]:
        if self.layout.channels_first:
            return (1, self.layout.height, self.layout.width)
        return (self.layout.height, self.layout.width, 1)

    def _image_space(self) -> spaces.Box:
        observation_space = self.observation_space
        if not isinstance(observation_space, spaces.Dict):
            raise RuntimeError("lazy SAC replay requires Dict observation spaces")
        image_space = observation_space.spaces.get("image")
        if not isinstance(image_space, spaces.Box):
            raise RuntimeError("lazy SAC replay requires Dict observation key 'image' as Box")
        return image_space

    def _state_space(self) -> spaces.Box:
        observation_space = self.observation_space
        if not isinstance(observation_space, spaces.Dict):
            raise RuntimeError("lazy SAC replay requires Dict observation spaces")
        state_space = observation_space.spaces.get("state")
        if not isinstance(state_space, spaces.Box):
            raise RuntimeError("lazy SAC replay requires Dict observation key 'state' as Box")
        return state_space


class LazyMaskableReplayBuffer(LazyImageStateReplayBuffer):
    """Lazy image-state replay buffer with flattened action masks for hybrid SAC."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        *,
        frame_stack: int,
        stack_mode: LazyReplayStackMode,
        minimap_layer: bool,
        mask_dims: int,
    ) -> None:
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            frame_stack=frame_stack,
            stack_mode=stack_mode,
            minimap_layer=minimap_layer,
        )
        self.mask_dims = int(mask_dims)
        self.action_masks = np.ones(
            (self.buffer_size, self.n_envs, self.mask_dims),
            dtype=np.float32,
        )
        self.next_action_masks = np.ones(
            (self.buffer_size, self.n_envs, self.mask_dims),
            dtype=np.float32,
        )

    def add(
        self,
        obs: dict[str, NumpyArray],
        next_obs: dict[str, NumpyArray],
        action: NumpyArray,
        reward: NumpyArray,
        done: NumpyArray,
        infos: list[dict[str, object]],
        *,
        action_masks: NumpyArray | None = None,
        next_action_masks: NumpyArray | None = None,
    ) -> None:
        self.action_masks[self.pos] = _normalize_masks(
            action_masks,
            n_envs=self.n_envs,
            mask_dims=self.mask_dims,
        )
        self.next_action_masks[self.pos] = _normalize_masks(
            next_action_masks,
            n_envs=self.n_envs,
            mask_dims=self.mask_dims,
        )
        super().add(obs, next_obs, action, reward, done, infos)

    def sample(
        self,
        batch_size: int,
        env: VecNormalize | None = None,
    ) -> Any:  # noqa: ANN401
        batch_inds, env_indices = self._sample_transition_pairs(batch_size)
        base_samples = self._get_samples_for_pairs(batch_inds, env_indices, env)
        return LazyMaskableReplaySamples(
            observations=base_samples.observations,
            actions=base_samples.actions,
            next_observations=base_samples.next_observations,
            dones=base_samples.dones,
            rewards=base_samples.rewards,
            action_masks=self.to_torch(self.action_masks[batch_inds, env_indices]),
            next_action_masks=self.to_torch(self.next_action_masks[batch_inds, env_indices]),
        )


# Backward-compatible checkpoint alias for runs saved before the replay buffer rename.
LazyMaskableHybridActionDictReplayBuffer = LazyMaskableReplayBuffer


def _current_slice_channels(stack_mode: ObservationStackMode) -> int:
    if stack_mode == "rgb":
        return 3
    if stack_mode == "gray":
        return 1
    if stack_mode == "luma_chroma":
        return 2
    supported = ", ".join(sorted(SUPPORTED_LAZY_REPLAY_STACK_MODES))
    raise RuntimeError(
        f"lazy SAC replay does not support observation.stack_mode={stack_mode!r}; "
        f"use one of: {supported}"
    )


def _normalize_masks(
    action_masks: NumpyArray | None,
    *,
    n_envs: int,
    mask_dims: int,
) -> Float32Array:
    if action_masks is None:
        return np.ones((n_envs, mask_dims), dtype=np.float32)
    return np.asarray(action_masks, dtype=np.float32).reshape((n_envs, mask_dims))


def _action_storage_dtype(dtype: np.dtype | type | None) -> np.dtype | type | None:
    if dtype == np.float64:
        return np.float32
    return dtype


def _is_channels_first(image_shape: tuple[int, int, int], expected_channels: int) -> bool:
    if image_shape[0] == expected_channels and image_shape[2] != expected_channels:
        return True
    if image_shape[2] == expected_channels and image_shape[0] != expected_channels:
        return False
    return image_shape[0] <= image_shape[2]


def _image_shape(image_space: spaces.Box) -> tuple[int, int, int]:
    shape = image_space.shape
    if len(shape) != 3:
        raise RuntimeError("lazy SAC replay requires 3D image observations")
    return int(shape[0]), int(shape[1]), int(shape[2])


def _as_uint8(array: NumpyArray) -> UInt8Array:
    return np.ascontiguousarray(array, dtype=np.uint8)
