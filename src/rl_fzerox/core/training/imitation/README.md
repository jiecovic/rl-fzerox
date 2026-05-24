# Imitation

This package is scaffold-only right now.

It is not wired into training, the run manager, or any executable behavior
cloning / DAgger loop yet.

Current scope:

- teacher/student observation-view planning
- canonical teacher/student action adaptation seams
- small behavior-cloning sample containers

Intended direction:

- behavior cloning warm starts from another run's checkpoint
- optional teacher/student observation mismatch support
- optional teacher/student action-space mismatch support
- later DAgger-style rollout relabeling on top of the same seams

The design goal is to keep observation adaptation, action adaptation, and BC /
DAgger data plumbing separate so the eventual training integration does not
grow into one large special-case path.
