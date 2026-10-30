"""Create 15 task definitions; Ascopt assigns their IDs when it reads the project."""

import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import shutil

from .generate import BY_NAME, TEMPLATES, spec_for, write


def prompt(spec):
    name = spec["id"]
    return f'''You are an Ascend C operator development engineer. Your task is to implement and optimize {spec['name']} for CANN 9.1.0 on Ascend 910_93.

Working directory: /workspace
ACLNN project: /workspace/submission
Operator source: /workspace/submission/csrc/ops/{name}/
Specification and test cases: /workspace/{spec['task_dir']}/ (proto.yaml, desc.md, cases.yaml)
Submission rules: /workspace/docs/guide/submission_rules.md
Interface: {spec['schema']}

Skills are available in /workspace/.agents/skills/.

Read the specification and submission rules, then implement the operator in its editable directories:

- /workspace/submission/csrc/ops/{name}/op_host/: shape/dtype inference and tiling.
- /workspace/submission/csrc/ops/{name}/op_kernel/: Ascend C computation and data movement.
- /workspace/submission/csrc/ops/{name}/op_plugin/: output allocation and ACLNN dispatch.

The build generates ACLNN APIs from the host prototype.

Use python3; Python dependencies, cann_bench_utils and msprof are preinstalled.
Use the `device` tool to access NPUs for experiments and profiling.
Query `get_job_status` for the status, results and log file paths of submitted jobs.
Call `run_full_evaluation` to submit the complete build, install and evaluate pipeline and obtain the official CannBench score.

Optimize the operator to maximize this score.
'''


def configure(repo, work_dir, replace=False):
    config_path = repo / ".ascopt/config.json"
    existing = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else []
    if existing and not replace:
        raise FileExistsError("Existing config.json is preserved; use --replace-config to archive and replace it")
    if existing:
        backup = work_dir / "previous" / datetime.now().strftime("config-%Y%m%d-%H%M%S-%f")
        backup.mkdir(parents=True)
        shutil.copyfile(config_path, backup / "config.json")
        for source in [repo / ".ascopt/prompt.md", repo / ".ascopt/prompts", repo / ".ascopt/hooks.py"]:
            if source.is_dir():
                shutil.copytree(source, backup / source.name)
            elif source.is_file():
                shutil.copyfile(source, backup / source.name)
        print(f"Previous configuration retained at {backup}")
    by_name = {item["task_name"]: item for item in existing}
    configs = []
    for name in BY_NAME:
        spec = spec_for(repo, name)
        source = "submission"
        if not (repo / source / "build.sh").is_file():
            raise FileNotFoundError(f"Generate this submission first: {source}")
        prior = by_name.get(spec["name"], existing[0] if existing else {})
        steps = []
        for stage, timeout, count in (("build", 1800, 0), ("install", 300, 0), ("evaluate", 3600, 1)):
            command = (f"python3 -u -m ascopt_integration.evaluate {stage} "
                       f"--source-dir {source} --task-dir {spec['task_dir']} --operator {spec['name']}")
            if stage == "evaluate":
                command += " --device-id 0"
            steps.append({"step_id": stage, "command": command,
                          "env": {"PYTHONPATH": "/workspace/src", "PYTHONNOUSERSITE": "1", "CANN_BENCH_OPERATORS": name},
                          "timeout_s": timeout, "npu_count": count})
        docker = copy.deepcopy(prior.get("docker", {}))
        docker.update(image="cannbench:9.1.0-a3", shm_size="2g",
                      startup_command="", startup_timeout_s=None,
                      startup_npu_count=0, startup_as_root=True)
        docker.setdefault("environment", {}).update(
            PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", BASH_ENV="/workspace/ascopt_integration/env.sh")
        prompt_path = f".ascopt/prompts/{name}.txt"
        config = {"task_name": spec["name"],
                  "prompt_path": prompt_path,
                  "writable_paths": [f"{source}/csrc/ops/{name}/{folder}"
                                     for folder in ("op_host", "op_kernel", "op_plugin")],
                  "unreadable_paths": [], "max_rounds": 100, "round_timeout_minutes": 600,
                  "max_concurrent_agents": 3, "max_concurrent_agents_per_node": 2,
                  "dsh": copy.deepcopy(prior.get("dsh", {"provider": "", "model": ""})),
                  "strategy": copy.deepcopy(prior.get("strategy", {"kind": "greedy", "params": {}})),
                  "docker": docker, "allowed_npu_ids": copy.deepcopy(prior.get("allowed_npu_ids", [0, 1])),
                  "evaluate": {"steps": steps, "hooks_file": ".ascopt/hooks.py", "stop_on_failure": True,
                               "callback_timeout_s": 30, "resource_wait_timeout_s": 1800}}
        configs.append(config)
        write(repo / prompt_path, prompt(spec))
    write(repo / ".ascopt/hooks.py", (TEMPLATES / "hooks.py").read_text(encoding="utf-8"))
    write(config_path, json.dumps(configs, ensure_ascii=False, indent=2) + "\n")
    # A convenient user-facing copy; no credentials are printed to the terminal.
    write(work_dir / "ascopt.config.json", json.dumps(configs, ensure_ascii=False, indent=2) + "\n")
    print(f"Configured {len(configs)} task definitions in {config_path}; Ascopt assigns IDs on first read")
    return configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True, help="Directory for backups and the configuration copy")
    parser.add_argument("--replace-config", action="store_true")
    args = parser.parse_args()
    configure(args.repo.resolve(), args.work_dir.resolve(), args.replace_config)
