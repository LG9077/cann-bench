# CannBench project skills

Source: <https://gitcode.com/cann/cannbot-skills>

Pinned commit: `99d43fe479cfb115a3d39c9dc6d2e87f5cf51548` (installed 2026-10-27).
The top-level skill directories below are copied from upstream `ops/`, including their
references, scripts and evaluation definitions. License: [CANN 2.0](LICENSE.cannbot).

| Skill | Use |
| --- | --- |
| `npu-arch` | Target architecture and hardware capabilities |
| `ascendc-api-best-practices` | API usage, alignment, buffers and synchronization |
| `ascendc-tiling-design` | Multicore partitioning, UB tiles and boundary cases |
| `ascendc-precision-debug` | Numerical errors and correctness diagnosis |
| `ascendc-runtime-debug` | ACLNN errors, kernel lookup and plog analysis |
| `ascendc-crash-debug` | Kernel hangs, crashes and suspected memory corruption |
| `ops-profiling` | Profiling collection and performance bottleneck analysis |

`ops-profiling` also includes its upstream `msopprof-visualization` subskill and
report templates; it is used only when a visualization is requested.

## Project scope

- Target CANN **9.1.0**, Ascend **910_93** (A3 / arch22). Upstream includes
  other architectures; use only applicable guidance and verify APIs against
  the installed SDK. A5 / arch35 examples are not implementations for this target.
- CannBench's official task contract, Golden, tolerances, cases and scoring
  remain authoritative. Skill helper comparisons are diagnostic, not substitutes
  for official evaluation. Optional references to uninstalled skills do not
  require installing another workflow or changing the benchmark.
- Device experiments go through Ascopt device jobs, not ordinary Bash. Pass
  `Ascend910_93` explicitly to `lookup_arch_variant.sh` to avoid device probing.
- Crash reproduction, sanitizer runs and profiling also use Ascopt device jobs
  and the job's local device IDs. Read existing logs first; use an explicit
  writable output directory under `/tmp` for new diagnostic artifacts.
- Profiling is auxiliary diagnosis. Its warmup protocol, runner templates and
  speedup reports do not replace CannBench's evaluation or `overall_score`.
  Keep the configured build/install/evaluate Stages unchanged. Check required
  tools (`msprof`, `mssanitizer`, or GDB for the selected diagnostic) in the
  container; report missing tools rather than installing toolchains or changing
  host driver/device settings. Installing these skills does not install those tools.
- Ascopt discovers these skills under `/workspace/.agents/skills` in a new
  Agent session. A Task that already captured its project inputs retains that
  snapshot; installing here does not modify existing Task history.

## Local adaptations

1. `ascendc-precision-debug/scripts/lookup_arch_variant.sh`: select
   `$(uname -m)-linux` instead of hard-coding `x86_64-linux`, so both x86_64 and
   aarch64 SDK layouts can be queried. No SDK, driver or device settings change.
2. `ascendc-tiling-design/references/conversion/transpose-small-channel.md`:
   replace the link to the uninstalled A5 performance skill with its pinned
   upstream URL. The separate `ascendc-performance-best-practices` skill and
   orchestration plugins are not installed.

All other upstream files are unchanged. These resources have not been validated
on a real CANN 9.1.0 / 910_93 system.
