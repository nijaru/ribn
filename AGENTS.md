# Ribn agent guidance

Ribn is a Rust, server-first inference engine targeting competitive serving across
model families and hardware. Qwen GGUF, Gemma 4 and the RTX 4090 are qualification
workloads, not product constraints. Use qualified vendor kernels where they win.

- Work on `main` unless asked otherwise. v0 APIs are unstable; replace flawed
  contracts rather than adding compatibility shims for obsolete paths.
- Use [decision milestones](docs/roadmap.md#decision-milestones) to select open-ended
  work. If device access is unavailable, advance a concrete baseline or model
  integration prerequisite instead of stacking unqualified device changes.
- Follow the relevant owner, not the whole doc set:
  [source map](docs/architecture.md#source-owners) for navigation,
  [target design](docs/inference-engine-design.md) for architecture,
  [resource protocol](docs/resource-protocol.md) for execution semantics, and
  [roadmap](docs/roadmap.md) for priorities and exit gates. Update the owner when a
  change alters its contract; use the roadmap for unresolved choices.
- Use [execution-contracts](.agents/skills/execution-contracts/SKILL.md) for lifecycle
  and resource changes; use [model-integration](.agents/skills/model-integration/SKILL.md)
  for model support or automatic variant selection.
- Before retrying rejected kernels, read the relevant
  [Qwen](benchmarks/qwen-prefill-qualification.md) or
  [Gemma](docs/roadmap.md#conditional-work-and-known-non-promotions) evidence.
  Qwen GDN chunk scan stays opt-in while the full-model gate fails.
- On `ssh desktop`, check GPU utilization before launching and serialize full-model
  tests. Use `ssh desktop bash -s` for multiline scripts (login shell is fish).
  Preserve untracked `cuda-rust-probe/simt/*`. Profiler counters currently require
  unavailable privilege; do not report inferred stall/occupancy causes as measured.
- Use the [verification entry points](docs/README.md#verification) for affected
  surfaces. Host and CUDA-feature checks do not qualify device execution. Report
  failed and unrun gates explicitly.
- Keep the public README focused on current use. Keep maintainer contracts and
  reproducible gates with the code, not personal notes or superseded work logs.
