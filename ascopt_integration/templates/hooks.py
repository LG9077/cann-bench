"""Keep the official final score; missing or unusable evaluation reports score zero."""

import json
import math


def score(ctx) -> float:
    try:
        node = ctx.get_node(ctx.node_id)
        log = node["full_evals"][node["last_eval_id"]]["logs"]["evaluate"]
        prefix = "ASCOPT_CANNBENCH_REPORT="
        reports = [json.loads(line[len(prefix):]) for line in log.read_text(encoding="utf-8").splitlines()
                   if line.startswith(prefix)]
        if len(reports) != 1 or reports[0]["result_stage"] != "final":
            return 0.0
        value = reports[0]["overall_score"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return 0.0
        return float(value)
    except (OSError, UnicodeError, KeyError, TypeError, ValueError, AttributeError, OverflowError):
        return 0.0
