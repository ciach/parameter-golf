#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from parse_experiment_log import format_summary, parse_path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE = ROOT / "experiment_queue.json"
DEFAULT_STATE = ROOT / "research_state.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_env(env: dict[str, Any]) -> dict[str, str]:
    return {str(key): "" if value is None else str(value) for key, value in env.items()}


def deny_reasons(exp: dict[str, Any], state: dict[str, Any]) -> list[str]:
    env = normalize_env(exp.get("env", {}))
    reasons: list[str] = []
    if exp.get("allow_denylist", False):
        return reasons
    for rule in state.get("denylist", []):
        match = rule.get("env_match", {})
        contains = rule.get("env_contains", {})
        matched = bool(match or contains)
        for key, value in match.items():
            matched = matched and env.get(key) == str(value)
        for key, value in contains.items():
            parts = [part.strip() for part in env.get(key, "").split(",") if part.strip()]
            matched = matched and str(value) in parts
        if matched:
            reasons.append(f"{rule.get('name', 'deny')}: {rule.get('reason', '')}")
    if exp.get("kind") == "byte" and not exp.get("allow_byte_probe", False):
        reasons.append("byte-only probe requires allow_byte_probe=true")
    return reasons


def experiments(queue: dict[str, Any]) -> list[dict[str, Any]]:
    return list(queue.get("experiments", []))


def _run_statuses(queue: dict[str, Any]) -> dict[str, str]:
    return {str(exp.get("id")): str(exp.get("status", "pending")) for exp in experiments(queue)}


def _gate_dependencies_met(exp: dict[str, Any], statuses: dict[str, str]) -> bool:
    gate = exp.get("gate", {})
    if not gate:
        return False
    for run_id in gate.get("requires_all_completed", []):
        if statuses.get(str(run_id)) not in {"completed", "rejected"}:
            return False
    requires_completed = gate.get("requires_completed_run")
    if requires_completed and statuses.get(str(requires_completed)) not in {"completed", "rejected"}:
        return False
    requires_passed = gate.get("requires_passed_run")
    if requires_passed and statuses.get(str(requires_passed)) != "completed":
        return False
    requires_failed = gate.get("requires_failed_run") or gate.get("requires_rejected_run")
    if requires_failed and statuses.get(str(requires_failed)) != "rejected":
        return False
    return any(
        key in gate
        for key in (
            "requires_all_completed",
            "requires_completed_run",
            "requires_passed_run",
            "requires_failed_run",
            "requires_rejected_run",
        )
    )


def refresh_blocked_experiments(queue: dict[str, Any]) -> list[str]:
    statuses = _run_statuses(queue)
    unblocked: list[str] = []
    for exp in experiments(queue):
        if exp.get("status") == "blocked" and _gate_dependencies_met(exp, statuses):
            exp["status"] = "pending"
            exp["unblocked_by"] = "dependency gate satisfied"
            statuses[str(exp.get("id"))] = "pending"
            unblocked.append(str(exp.get("id")))
    return unblocked


def validate_queue(queue: dict[str, Any], state: dict[str, Any]) -> int:
    refresh_blocked_experiments(queue)
    bad = 0
    for exp in experiments(queue):
        status = exp.get("status", "pending")
        reasons = deny_reasons(exp, state)
        prefix = "OK"
        if status == "blocked":
            prefix = "BLOCKED"
        elif reasons:
            prefix = "DENY"
            if status == "pending":
                bad += 1
        print(f"{prefix} {exp.get('id')} status:{status}")
        for reason in reasons:
            print(f"  - {reason}")
    return bad


def first_runnable(queue: dict[str, Any], state: dict[str, Any]) -> dict[str, Any] | None:
    refresh_blocked_experiments(queue)
    for exp in experiments(queue):
        if exp.get("status", "pending") != "pending":
            continue
        reasons = deny_reasons(exp, state)
        if reasons:
            raise SystemExit(f"Refusing {exp.get('id')}: " + "; ".join(reasons))
        return exp
    return None


def make_remote_script(exp: dict[str, Any], defaults: dict[str, Any]) -> str:
    exp_id = exp["id"]
    env = normalize_env(exp.get("env", {}))
    env.setdefault("RUN_ID", exp_id)
    env.setdefault("SUBMISSION_SIZE_CAP_BYTES", str(defaults.get("cap_bytes", 16000000)))
    log_root = defaults.get("log_root", "experiment_logs_autoresearch")
    train_script = exp.get("train_script", defaults.get("train_script", "train_gpt_frontier_min.py"))
    source_script = exp.get("source_script", defaults.get("source_script", "train_gpt_frontier.py"))
    minifier_script = exp.get("minifier_script", defaults.get("minifier_script", "tools/minify_train_script.py"))
    nproc = int(exp.get("nproc_per_node", defaults.get("nproc_per_node", 8)))
    timeout = int(exp.get("timeout_seconds", defaults.get("timeout_seconds", 1200)))
    log_name = "log_" + exp_id + ".txt"
    grep_re = (
        "flash_attention_backend|VE:|TTT|ttt_|mixed_quant|step:[0-9]+/|late_qat|"
        "warmdown_iters|ema_decay|matrix_lr_layer_mults|cache_mode|factorized_emb_dim|outlier_audit|matformer_|mlp_eq_|spectral_mode|spectral_layers|gptq|legal_ttt|stopping_early|DIAGNOSTIC post_ema|quant_codec_selected|Total submission|"
        "Submission cap|final_int6|final_int8|Traceback|RuntimeError|ValueError|SignalException"
    )
    env_text = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    return f"""set -uo pipefail
cd {shlex.quote(str(defaults.get("remote_dir", "/workspace/parameter-golf")))}
BATCH={shlex.quote(exp_id)}
DIR={shlex.quote(str(log_root))}/${{BATCH}}
ARCHIVE={shlex.quote(str(log_root))}/${{BATCH}}.tar.gz
mkdir -p "${{DIR}}"
python3 - <<'PY'
import zstandard as zstd
print('zstandard', zstd.__version__)
PY
python3 {shlex.quote(str(minifier_script))} --input {shlex.quote(str(source_script))} --output {shlex.quote(str(train_script))}
python3 -m py_compile {shlex.quote(str(train_script))}
echo "batch:${{BATCH}}" | tee "${{DIR}}/results_summary.txt"
echo "start:$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${{DIR}}/results_summary.txt"
set +e
timeout {timeout}s env {env_text} torchrun --standalone --nproc_per_node={nproc} {shlex.quote(str(train_script))} > "${{DIR}}/{log_name}" 2>&1
code=$?
set -e
echo "exit:${{code}}" | tee -a "${{DIR}}/results_summary.txt"
grep -E {shlex.quote(grep_re)} "${{DIR}}/{log_name}" | tail -n 260 | tee -a "${{DIR}}/results_summary.txt" || true
tar -czf "${{ARCHIVE}}" -C {shlex.quote(str(log_root))} "${{BATCH}}"
echo "archive:${{ARCHIVE}}" | tee -a "${{DIR}}/results_summary.txt"
exit "${{code}}"
"""


def make_ssh_command(exp: dict[str, Any], defaults: dict[str, Any], ssh_host: str, port: str | None, key: str) -> str:
    script = make_remote_script(exp, defaults)
    key_path = str(Path(key).expanduser())
    cmd = ["ssh", "-o", "StrictHostKeyChecking=accept-new"]
    if port:
        cmd += ["-p", str(port)]
    cmd += ["-i", key_path, ssh_host, script]
    return " ".join(shlex.quote(part) for part in cmd)


def remote_archive_path(exp: dict[str, Any], defaults: dict[str, Any]) -> str:
    log_root = defaults.get("log_root", "experiment_logs_autoresearch")
    return f"{defaults.get('remote_dir', '/workspace/parameter-golf')}/{log_root}/{exp['id']}.tar.gz"


def make_scp_command(
    exp: dict[str, Any],
    defaults: dict[str, Any],
    ssh_host: str,
    port: str | None,
    key: str,
    download_dir: str,
) -> str:
    key_path = str(Path(key).expanduser())
    cmd = ["scp", "-o", "StrictHostKeyChecking=accept-new"]
    if port:
        cmd += ["-P", str(port)]
    cmd += ["-i", key_path, f"{ssh_host}:{remote_archive_path(exp, defaults)}", download_dir]
    return " ".join(shlex.quote(part) for part in cmd)


def update_with_result(
    queue_path: Path,
    state_path: Path,
    archive: Path,
) -> None:
    queue = load_json(queue_path)
    state = load_json(state_path)
    result = parse_path(archive)
    run_id = result.get("run_id") or archive.stem
    exp = next((item for item in experiments(queue) if item.get("id") == run_id), None)
    decision = "recorded"
    if exp is not None:
        gate = exp.get("gate", {})
        bpb = result.get("sliding_exact_bpb")
        cap_ok = result.get("cap_status") == "PASS"
        max_bpb = gate.get("max_bpb")
        beat_gate = bpb is not None and (max_bpb is None or bpb <= float(max_bpb))
        cap_gate = (not gate.get("require_cap", False)) or cap_ok
        max_step_avg_ms = gate.get("max_step_avg_ms")
        if max_step_avg_ms is None:
            speed_gate = True
        else:
            step_avg_ms = result.get("step_avg_ms")
            speed_gate = step_avg_ms is not None and step_avg_ms <= float(max_step_avg_ms)
        max_bytes = gate.get("max_bytes")
        if max_bytes is None:
            bytes_gate = True
        else:
            total_bytes = result.get("total_bytes")
            bytes_gate = total_bytes is not None and total_bytes <= int(max_bytes)
        decision = "pass" if beat_gate and cap_gate and speed_gate and bytes_gate else "reject"
        exp["status"] = "completed" if decision == "pass" else "rejected"
        exp["result"] = result
    unblocked = refresh_blocked_experiments(queue)
    history = state.setdefault("history", [])
    if not any(item.get("run_id") == run_id for item in history):
        history.append(
            {
                "run_id": run_id,
                "bpb": result.get("sliding_exact_bpb"),
                "bytes": result.get("total_bytes"),
                "decision": decision,
                "archive": str(archive),
            }
        )
    save_json(queue_path, queue)
    save_json(state_path, state)
    print(format_summary(result))
    print(f"decision:{decision}")
    for exp_id in unblocked:
        print(f"unblocked:{exp_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Guarded Parameter Golf autoresearch controller.")
    parser.add_argument("--queue", default=str(DEFAULT_QUEUE))
    parser.add_argument("--state", default=str(DEFAULT_STATE))
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("validate", help="Validate queue against denylist.")

    next_parser = sub.add_parser("next", help="Print or execute next runnable remote command.")
    next_parser.add_argument("--ssh-host", help="SSH host, e.g. root@1.2.3.4")
    next_parser.add_argument("--port", help="SSH port for direct TCP.")
    next_parser.add_argument("--key", default="~/.ssh/id_ed25519")
    next_parser.add_argument("--download-dir", default="remote_logs", help="Local directory for archive SCP.")
    next_parser.add_argument("--execute", action="store_true", help="Execute instead of only printing.")

    record_parser = sub.add_parser("record", help="Parse archive and update state/queue.")
    record_parser.add_argument("archive")

    parse_parser = sub.add_parser("parse", help="Parse archive/log only.")
    parse_parser.add_argument("archive")
    parse_parser.add_argument("--json", action="store_true")

    args = parser.parse_args()
    queue_path = Path(args.queue)
    state_path = Path(args.state)

    if args.cmd == "validate":
        raise SystemExit(validate_queue(load_json(queue_path), load_json(state_path)))
    if args.cmd == "next":
        queue = load_json(queue_path)
        state = load_json(state_path)
        exp = first_runnable(queue, state)
        if exp is None:
            print("No pending runnable experiment.")
            return
        if not args.ssh_host:
            print(make_remote_script(exp, queue.get("defaults", {})))
            return
        defaults = queue.get("defaults", {})
        command = make_ssh_command(exp, defaults, args.ssh_host, args.port, args.key)
        scp_command = make_scp_command(exp, defaults, args.ssh_host, args.port, args.key, args.download_dir)
        print(command)
        print(f"\n# download after run\n{scp_command}")
        if args.execute:
            code = subprocess.call(command, shell=True)
            Path(args.download_dir).mkdir(parents=True, exist_ok=True)
            scp_code = subprocess.call(scp_command, shell=True)
            raise SystemExit(code if code != 0 else scp_code)
        return
    if args.cmd == "record":
        update_with_result(queue_path, state_path, Path(args.archive))
        return
    if args.cmd == "parse":
        result = parse_path(Path(args.archive))
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(format_summary(result))
        return
    raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
