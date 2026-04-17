# scripts/check_numpy_typing.py
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import TypeAlias

PathIssue: TypeAlias = tuple[Path, int, int, str]

_CENTRAL_ARRAY_ALIAS_MODULE = Path("src/fzerox_emulator/arrays.py")
_FORBIDDEN_NUMPY_ANNOTATION_NAMES = {
    "NDArray",
    "np.ndarray",
    "npt.NDArray",
    "numpy.ndarray",
    "numpy.typing.NDArray",
}


def main(argv: list[str] | None = None) -> int:
    roots = [Path(arg) for arg in (sys.argv[1:] if argv is None else argv)]
    if not roots:
        roots = [Path("src"), Path("tests"), Path("scripts")]

    issues: list[PathIssue] = []
    for root in roots:
        for path in _python_files(root):
            issues.extend(_raw_numpy_annotations(path))

    if not issues:
        return 0

    print(
        "raw/direct NumPy ndarray annotations are not allowed; use fzerox_emulator.arrays aliases"
    )
    for path, line, column, expression in issues:
        print(f"{path}:{line}:{column}: {expression}")
    return 1


def _python_files(root: Path) -> tuple[Path, ...]:
    if root.is_file():
        return (root,) if root.suffix == ".py" else ()
    if not root.exists():
        return ()
    return tuple(sorted(path for path in root.rglob("*.py") if ".venv" not in path.parts))


def _raw_numpy_annotations(path: Path) -> list[PathIssue]:
    if _is_central_array_alias_module(path):
        return []

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    issues: list[PathIssue] = []
    for node in ast.walk(tree):
        for annotation in _annotation_nodes(node):
            for raw_node in ast.walk(annotation):
                if _qualified_name(raw_node) in _FORBIDDEN_NUMPY_ANNOTATION_NAMES and isinstance(
                    raw_node, ast.expr
                ):
                    issues.append(
                        (
                            path,
                            raw_node.lineno,
                            raw_node.col_offset + 1,
                            ast.unparse(raw_node),
                        )
                    )
    return issues


def _annotation_nodes(node: ast.AST) -> tuple[ast.AST, ...]:
    if isinstance(node, ast.arg):
        return (node.annotation,) if node.annotation is not None else ()
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return (node.returns,) if node.returns is not None else ()
    if isinstance(node, ast.AnnAssign):
        if _qualified_name(node.annotation) == "TypeAlias" and node.value is not None:
            return (node.annotation, node.value)
        return (node.annotation,)
    return ()


def _qualified_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return f"{node.value.id}.{node.attr}"
    if isinstance(node, ast.Attribute):
        value_name = _qualified_name(node.value)
        if value_name is not None:
            return f"{value_name}.{node.attr}"
    return None


def _is_central_array_alias_module(path: Path) -> bool:
    normalized = Path(*path.parts[-3:]) if len(path.parts) >= 3 else path
    return normalized == _CENTRAL_ARRAY_ALIAS_MODULE


if __name__ == "__main__":
    raise SystemExit(main())
