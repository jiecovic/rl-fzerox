# tests/core/manager/test_training_bridge_observation_policy.py
from __future__ import annotations

from pathlib import Path

from rl_fzerox.core.envs.observations.state import state_feature_names
from rl_fzerox.core.manager import default_managed_run_config
from rl_fzerox.core.manager.architecture.preview import policy_architecture_preview
from rl_fzerox.core.manager.training import (
    build_managed_train_app_config,
)
from rl_fzerox.core.runtime_spec.schema import (
    CustomResolutionChoice,
    SourceCropResolutionChoice,
)


def test_manager_training_bridge_can_override_renderer(tmp_path: Path) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.environment.renderer = "angrylion"
    config.environment.camera_setting = "regular"

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-renderer",
        run_dir=tmp_path / "runs" / "bridge-renderer_0001",
    )

    assert train_config.emulator.renderer == "angrylion"
    assert train_config.env.camera_setting == "regular"


def test_manager_training_bridge_uses_explicit_state_component_membership(
    tmp_path: Path,
) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.observation.state_components = tuple(
        component
        for component in config.observation.state_components
        if component.name != "machine_context"
    )
    config.policy.state_net_arch = ()

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-component-membership",
        run_dir=tmp_path / "runs" / "bridge-component-membership_0001",
    )

    assert "machine_context" not in {
        component.name for component in train_config.env.observation.state_components or ()
    }
    assert train_config.policy.extractor.resolved_state_net_arch() == ()


def test_manager_training_bridge_can_disable_fusion_mlp(tmp_path: Path) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.policy.fusion_features_dim = None

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-no-fusion",
        run_dir=tmp_path / "runs" / "bridge-no-fusion_0001",
    )
    preview = policy_architecture_preview(config)
    fusion_group = next(group for group in preview.parameter_groups if group.name == "Fusion")
    fusion_node = next(
        node for lane in preview.architecture_lanes for node in lane.nodes if node.id == "fusion"
    )

    assert train_config.policy.extractor.fusion_features_dim is None
    assert preview.extractor_output_dim == preview.fusion_input_dim
    assert fusion_group.params == 0
    assert fusion_node.detail == f"identity {preview.fusion_input_dim}"
    assert fusion_node.tone == "muted"


def test_policy_architecture_preview_labels_extractor_activations() -> None:
    config = default_managed_run_config()
    preview = policy_architecture_preview(config)
    node_by_id = {node.id: node for lane in preview.architecture_lanes for node in lane.nodes}

    assert node_by_id["cnn"].detail == "nature → 3136"
    assert node_by_id["image_projection"].detail == "identity 3136"
    assert node_by_id["state_mlp"].detail.endswith(", relu")
    assert node_by_id["fusion"].detail.endswith(", relu")
    assert node_by_id["policy_head"].detail.endswith(", relu")


def test_policy_architecture_preview_shows_auxiliary_head_when_enabled() -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.policy.auxiliary_state_enabled = True

    preview = policy_architecture_preview(config)
    aux_group = next(group for group in preview.parameter_groups if group.name == "Aux head")
    aux_node = next(
        node for lane in preview.architecture_lanes for node in lane.nodes if node.id == "aux_head"
    )

    assert aux_group.params > 0
    assert aux_node.tone == "normal"
    assert aux_node.params == aux_group.params
    assert aux_node.detail.endswith(", relu, 39 targets")


def test_manager_training_bridge_projects_extractor_activations(tmp_path: Path) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.policy.features_dim = 512
    config.policy.image_projection_activation = "gelu"
    config.policy.state_activation = "tanh"
    config.policy.fusion_activation = "tanh"
    config.policy.layer_norm_activation = "gelu"

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-activations",
        run_dir=tmp_path / "runs" / "bridge-activations",
    )
    preview = policy_architecture_preview(config)
    node_by_id = {node.id: node for lane in preview.architecture_lanes for node in lane.nodes}

    assert train_config.policy.extractor.image_projection_activation == "gelu"
    assert train_config.policy.extractor.state_activation == "tanh"
    assert train_config.policy.extractor.fusion_activation == "tanh"
    assert train_config.policy.extractor.layer_norm_activation == "gelu"
    assert node_by_id["image_projection"].detail.endswith(", gelu")
    assert node_by_id["state_mlp"].detail.endswith(", tanh")
    assert node_by_id["fusion"].detail.endswith(", tanh")
    assert node_by_id["layer_norm"].detail == "on, gelu"


def test_manager_training_bridge_projects_individual_state_features(
    tmp_path: Path,
) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.observation.state_components = tuple(
        component.model_copy(update={"included_features": ("track_position.edge_ratio",)})
        if component.name == "track_position"
        else component
        for component in config.observation.state_components
        if component.name == "track_position"
    )
    config.observation.state_feature_dropouts = ()

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-feature-membership",
        run_dir=tmp_path / "runs" / "bridge-feature-membership_0001",
    )
    components = train_config.env.observation.state_components
    assert components is not None

    assert state_feature_names(
        state_components=tuple(component.data() for component in components)
    ) == ("track_position.edge_ratio",)


def test_manager_training_bridge_supports_episode_state_feature_dropout(
    tmp_path: Path,
) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.observation.state_feature_dropouts = (
        config.observation.state_feature_dropouts[0].model_copy(update={"dropout_prob": 0.25}),
    )

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-state-feature-dropout",
        run_dir=tmp_path / "runs" / "bridge-state-feature-dropout_0001",
    )

    assert tuple(
        group.model_dump(mode="python") for group in train_config.train.state_feature_dropout_groups
    ) == ({"dropout_prob": 0.25, "feature_names": ("track_position.edge_ratio",)},)


def test_manager_training_bridge_projects_custom_observation_resolution(
    tmp_path: Path,
) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.observation.resolution = CustomResolutionChoice(height=72, width=96)

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-custom-resolution",
        run_dir=tmp_path / "runs" / "bridge-custom-resolution_0001",
    )

    assert train_config.env.observation.resolution == CustomResolutionChoice(height=72, width=96)


def test_manager_training_bridge_projects_source_crop_observation_resolution(
    tmp_path: Path,
) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.environment.renderer = "angrylion"
    config.observation.resolution = SourceCropResolutionChoice()

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-source-crop-resolution",
        run_dir=tmp_path / "runs" / "bridge-source-crop-resolution_0001",
    )

    assert train_config.env.observation.resolution == SourceCropResolutionChoice()
    assert train_config.env.observation.image_geometry(renderer=train_config.emulator.renderer) == (
        208,
        592,
    )


def test_manager_training_bridge_supports_nature_conv_profile_for_custom_resolution() -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.observation.resolution = CustomResolutionChoice(height=72, width=96)

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-custom-resolution-nature-conv",
        run_dir=Path("unused"),
    )

    assert train_config.policy.extractor.conv_profile == "nature"


def test_manager_training_bridge_supports_multilayer_state_mlp(
    tmp_path: Path,
) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.policy.state_net_arch = (128, 64)

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-state-mlp",
        run_dir=tmp_path / "runs" / "bridge-state-mlp_0001",
    )

    assert train_config.policy.extractor.resolved_state_net_arch() == (128, 64)


def test_manager_training_bridge_uses_discrete_only_hybrid_ppo_when_no_continuous_axes(
    tmp_path: Path,
) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.policy.recurrent_enabled = False
    config.action.steering_mode = "discrete"
    config.action.drive_mode = "on_off"

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-discrete",
        run_dir=tmp_path / "runs" / "bridge-discrete_0001",
    )

    assert train_config.train.algorithm == "maskable_hybrid_action_ppo"
    assert train_config.env.action.runtime().name == "configured_hybrid"
    assert train_config.env.action.layout_continuous_axes == ()
    assert train_config.env.action.layout_discrete_axes == (
        "steer",
        "gas",
        "air_brake",
        "boost",
        "lean",
        "pitch",
    )
