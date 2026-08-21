# Agent Guide for `share-rl`

This repo is small enough that a few architectural rules matter more than cleverness. Follow them and the codebase stays coherent.

## Mental model

- `src/share/envs/manipulation_primitive/config_manipulation_primitive.py`
  Holds primitive configs, processor setup, validation, and entry-time target resolution logic.
- `src/share/envs/manipulation_primitive/env_manipulation_primitive.py`
  Holds primitive env runtime state. Configs do not own runtime state.
- `src/share/envs/manipulation_primitive_net/env_manipulation_primitive_net.py`
  Holds MP-Net lifecycle orchestration and primitive switching.
- `src/share/envs/manipulation_primitive_net/transitions.py`
  Holds transition conditions only. Keep them declarative and observation/info driven.
- `src/share/workspace/mpnet.py`
  Holds config serialization, editing helpers, and prompt-facing summaries.

## Primitive architecture rules

- Runtime state belongs to envs, not configs.
- Primitive configs may compute entry-time targets in `on_entry(...)`, but they should write the result into the env.
- Keep the primitive step API stable. If a primitive needs fundamentally different stepping behavior, make a dedicated env subclass in the config's `make(...)`.
- When writing an env config (`experiments/envs/*.py`), reach for a plain `ManipulationPrimitiveConfig` with `env_class`/`env_kwargs` (or plain field overrides) first. Only introduce a dedicated `ManipulationPrimitiveConfig` subclass -- with its own dataclass fields and a custom `make(...)` -- when the behavior genuinely cannot be expressed as parameters on the existing config surface (see `FoundationPosePrimitiveConfig` for a case that needed it: entry-time frame-origin resolution). A subclass is not the default; it's the fallback once parameterization runs out.
- Keep problem-specific bits (custom env subclasses, one-off event names/constants, bespoke file formats) local to the env module that needs them. Don't promote something into `share/` -- a shared enum, a shared processor step, a shared primitive base -- until a second real call site needs it too. `TeleopEvents` in particular is a canonical, cross-env vocabulary; a keypress meaningful to only one env config is a local module-level string constant, not a new `TeleopEvents` member.
- Primitive entry context is intentionally small: processed observation plus the previous task-frame origin.
- Do not add broad new info keys casually. Prefer the existing narrow surface:
  - `primitive_target_pose`
  - `primitive_complete`
  - `trajectory_progress`
- When a transition needs the current EE pose, fetch it from the processed observation through the shared observation-pose utility instead of republishing extra pose state in `info`.

## Code quality expectations

- Keep docstrings brief, accurate, and local to the file being changed.
- When changing primitive lifecycle behavior, update config/runtime docs in the same patch.
- Prefer small pure helpers over spreading frame math across multiple call sites.
- Preserve backwards compatibility where it is cheap, especially in workspace serialization and lightweight tests.
- Avoid adding new abstractions unless they remove duplication in at least two real call sites.
- Keep examples runnable and readable. They are part of the product surface here.

## Style and editing

- Use ASCII unless the file already uses non-ASCII.
- Prefer explicit names over short ones in config/runtime code.
- Add comments only where the control flow or math would otherwise be hard to infer.
- Do not silently widen the public API. If you add a new user-facing config field, document it in examples or docs in the same change.

## Validation checklist

- If you touched `src/share/envs`, run `py_compile` on each changed module.
- If you touched MP-Net lifecycle, transitions, or examples, run the focused env tests:
  - `tests/share/envs/test_dynamic_primitive_types.py`
  - `tests/share/envs/test_manipulation_primitive_net_step.py`
  - `tests/share/envs/test_manipulation_primitive_net_transitions.py`
- If you changed serialization or editor helpers, also sanity-check `src/share/workspace/mpnet.py`.

## Examples and docs

- New examples should show a real use case, not just a unit-test-shaped config.
- Prefer one clear idea per example, with short notes on what it demonstrates.
- Keep the top-level `wiki/` in sync with major architecture changes. It is meant to grow into structured webpage documentation.

## Current open problems worth keeping in mind

- Open-loop primitives are intentionally simple and scripted-only. Mixed policy + scripted residual control is still open design space.
- The observation/task-frame origin contract is now much cleaner, but relative-frame semantics still deserve more explanation and more end-to-end examples.
- Joint-only robots rely on FK-injected EE observations. This works, but examples and docs should keep making that dependency obvious.
- Import ergonomics are improved, but circular-import pressure is still a reason to keep package `__init__` files light.
