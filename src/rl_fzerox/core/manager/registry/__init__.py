# src/rl_fzerox/core/manager/registry/__init__.py
"""Internal SQLite registry surface for manager-owned state.

The registry package is split by domain object:
- ``drafts`` manages editable drafts and templates
- ``runs`` manages launched runs, runtime snapshots, commands, and workers
- ``lineages`` manages fork/delete ordering across related runs
"""
