"""Separate Ascopt build/install/evaluate entry, based on official staged_eval.

Upstream: 5777206c3b0c62849289af6577af504532a4cb10.
Only orchestration is adapted: no build/install during evaluate. Case execution,
performance partitioning, result merging, scoring and reports remain upstream.
"""

import json
from pathlib import Path
import shutil
import sys
import tempfile

from kernel_eval import staged_eval as official
from kernel_eval.base.perf_metric import resolve_perf_metric
from kernel_eval.base.score_weights import score_weights_context
from kernel_eval.data.package_manager import PackageManager, _detect_opp_path
from kernel_eval.security.api_guard import APIGuard
from kernel_eval.utils.path_resolver import resolve_task_dir


def build(args, cfg) -> int:
    package = PackageManager(config=cfg).build_packages(args.source_dir)
    # Upstream captures compiler output in _compile.log; retain it in the Stage log too.
    log_path = Path(args.source_dir) / "_compile.log"
    print(f"\nCannBench build log: {log_path}", flush=True)
    with log_path.open(encoding="utf-8", errors="replace") as stream:
        shutil.copyfileobj(stream, sys.stdout)
    print(flush=True)
    # Upstream's compile-failure report initializes an NPU. Build leases no device.
    return int(package.build_failed or not package.whl_path)


def _scan_installed(cfg):
    from kernel_eval.benches import cann as _cann_bench  # noqa: F401

    # Each Stage is a fresh process. Guard the submission import here as upstream
    # does after installation, without reinstalling during correctness/performance.
    with APIGuard() as guard:
        matched = PackageManager(config=cfg).prepare_skip_build()
        guard.verify()
        return matched


def evaluate(args, bench_root, filter_prefix) -> int:
    # This is the evaluation portion of staged_eval._run_with_score_profile.
    print("[ascopt_eval] correctness", flush=True)
    correctness_cfg = official._make_config(args, bench_root, enable_profiler=False)
    matched = _scan_installed(correctness_cfg)
    rel_paths = official._operator_rel_paths(matched, bench_root, selected=args.selected_operators)
    correctness_cases = official._load_cases(args, bench_root, rel_paths, filter_prefix=filter_prefix)
    correctness_ops = official._evaluate_cases(args, correctness_cfg, correctness_cases, enable_profiler=False)
    correctness_payload, _ = official._save_report(
        args, correctness_cfg, correctness_ops, stage="correctness", contains_performance=False,
    )
    if args.no_perf:
        return min(int(correctness_payload.get("failed_cases") or 0), 255)

    allowlist = official._passed_case_keys(correctness_ops)
    batch_allowlist, individual_allowlist = official._partition_performance_case_keys(
        correctness_ops,
        batch_enabled=args.perf_batch_cases,
        memory_limit_mb=args.perf_batch_case_memory_limit_mb,
    )
    print(
        f"[ascopt_eval] performance ({len(allowlist)} correctness-passed cases; "
        f"{len(batch_allowlist)} batch eligible, {len(individual_allowlist)} per-case profiler)",
        flush=True,
    )
    performance_cfg = official._make_config(args, bench_root, enable_profiler=True)
    matched = _scan_installed(performance_cfg)
    rel_paths = official._operator_rel_paths(matched, bench_root, selected=args.selected_operators)
    performance_cases = official._load_cases(
        args, bench_root, rel_paths, filter_prefix=filter_prefix, allowlist=allowlist,
    )
    batch_cases = [case for case in performance_cases if official._case_spec_key(case) in batch_allowlist]
    individual_cases = [case for case in performance_cases if official._case_spec_key(case) in individual_allowlist]

    batch_ops = official._evaluate_cases(args, performance_cfg, batch_cases, enable_profiler=True)
    individual_cfg = official._make_config(args, bench_root, enable_profiler=True)
    individual_cfg.perf_batch_cases = False
    individual_cfg.monitor_case_memory = False
    individual_ops = official._evaluate_cases(args, individual_cfg, individual_cases, enable_profiler=True)
    performance_ops = official._combine_operator_results([batch_ops, individual_ops])
    official._save_report(args, performance_cfg, performance_ops, stage="performance", contains_performance=True)

    merged_ops = official._merge_results(correctness_ops, performance_ops)
    final_payload, _ = official._save_report(
        args, performance_cfg, merged_ops, stage="final", contains_performance=True,
    )
    # Pass the report just produced, not a file selected from an earlier evaluation.
    print("\nASCOPT_CANNBENCH_REPORT=" + json.dumps(final_payload, ensure_ascii=False), flush=True)
    return min(int(final_payload.get("failed_cases") or 0), 255)


def run(args) -> int:
    if args.bench_name != "cann":
        raise ValueError("ascopt_eval supports --bench-name cann only")
    if not args.source_dir:
        raise ValueError("--source-dir is required for the split-stage workflow")
    bench_root, filter_prefix = resolve_task_dir(args.task_dir, official.get_project_root())
    if args.reports_dir is None:
        args.reports_dir = tempfile.mkdtemp(prefix="ascopt-cannbench-")
    Path(args.reports_dir).mkdir(parents=True, exist_ok=True)
    print(f"CannBench reports: {args.reports_dir}", flush=True)

    with score_weights_context(bench_root):
        resolve_perf_metric()
        cfg = official._make_config(args, bench_root, enable_profiler=False)
        if args.stage == "build":
            return build(args, cfg)

        mismatch = official._check_build_cann_consistency(args)
        if mismatch is not None:
            raise RuntimeError(mismatch)
        if args.stage == "install":
            # Includes upstream snapshot -> install -> import/scan -> verify.
            official._install_or_scan(args, cfg)
            return 0

        PackageManager._source_set_env_bash(_detect_opp_path())
        return evaluate(args, bench_root, filter_prefix)


def create_parser():
    parser = official.create_parser()
    parser.description = __doc__
    parser.add_argument("stage", choices=("build", "install", "evaluate"))
    parser.set_defaults(reports_dir=None)
    return parser


if __name__ == "__main__":
    sys.exit(run(create_parser().parse_args()))
