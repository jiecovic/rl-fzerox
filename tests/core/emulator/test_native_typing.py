# tests/core/emulator/test_native_typing.py
from __future__ import annotations

import ast
from pathlib import Path

import fzerox_emulator
import fzerox_emulator._native as native

STUB_ONLY_TYPES = frozenset(
    {
        "FrameObservationOptionsDict",
        "ObservationSpecDict",
        "VehicleSetupInfoDict",
    }
)


def test_fzerox_emulator_declares_typed_package_marker() -> None:
    package_dir = Path(fzerox_emulator.__file__).parent

    assert (package_dir / "py.typed").is_file()


def test_native_stub_public_api_matches_loaded_extension() -> None:
    stub_path = Path(fzerox_emulator.__file__).parent / "_native.pyi"
    stub_tree = ast.parse(stub_path.read_text(encoding="utf-8"))

    missing_module_names = sorted(
        name
        for name in _stub_module_names(stub_tree)
        if name not in STUB_ONLY_TYPES and not hasattr(native, name)
    )
    assert missing_module_names == []

    missing_class_members: dict[str, list[str]] = {}
    for class_node in _stub_classes(stub_tree):
        if class_node.name in STUB_ONLY_TYPES:
            continue
        runtime_class = getattr(native, class_node.name)
        missing_members = sorted(
            name for name in _stub_class_members(class_node) if not hasattr(runtime_class, name)
        )
        if missing_members:
            missing_class_members[class_node.name] = missing_members

    assert missing_class_members == {}


def _stub_module_names(stub_tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in stub_tree.body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _stub_classes(stub_tree: ast.Module) -> tuple[ast.ClassDef, ...]:
    return tuple(node for node in stub_tree.body if isinstance(node, ast.ClassDef))


def _stub_class_members(class_node: ast.ClassDef) -> set[str]:
    members: set[str] = set()
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            members.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            members.add(node.target.id)
    return members
