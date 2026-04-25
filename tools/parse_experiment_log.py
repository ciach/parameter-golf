#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import tarfile
from pathlib import Path
from typing import Any


def _last_float(pattern: str, text: str) -> float | None:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return float(matches[-1]) if matches else None


def _last_int(pattern: str, text: str) -> int | None:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return int(matches[-1]) if matches else None


def _last_str(pattern: str, text: str) -> str | None:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return matches[-1] if matches else None


def read_log_text(path: Path) -> str:
    if path.suffixes[-2:] == [".tar", ".gz"] or path.suffix == ".tgz":
        with tarfile.open(path, "r:gz") as tar:
            names = tar.getnames()
            preferred = [
                name
                for name in names
                if name.endswith("results_summary.txt")
            ]
            preferred += [
                name
                for name in names
                if "/log_" in name and name.endswith(".txt")
            ]
            if not preferred:
                raise FileNotFoundError(f"No results_summary/log file in archive: {path}")
            chunks: list[str] = []
            for name in preferred:
                member = tar.extractfile(name)
                if member is not None:
                    chunks.append(member.read().decode("utf-8", errors="replace"))
            return "\n".join(chunks)
    return path.read_text(encoding="utf-8", errors="replace")


def parse_text(text: str, source: str = "") -> dict[str, Any]:
    run_id = _last_str(r"^batch:([^\s]+)$", text)
    if run_id is None:
        run_id = _last_str(r"logs/([^/\s]+)\.txt", text)
    result: dict[str, Any] = {
        "source": source,
        "run_id": run_id,
        "exit_code": _last_int(r"^exit:(-?\d+)$", text),
        "backend": _last_str(r"flash_attention_backend:([^\s]+)", text),
        "config_line": _last_str(r"(submission_size_cap_bytes:[^\n]+)", text),
        "optimizer_line": _last_str(r"(tie_embeddings:[^\n]+matrix_lr_layer_mults:[^\n]+)", text),
        "schedule_line": _last_str(r"(train_batch_tokens:[^\n]+warmdown_iters:[^\n]+)", text),
        "cache_line": _last_str(r"(cache_mode:[^\n]+factorized_emb_dim:[^\n]+)", text),
        "matformer_line": _last_str(r"(matformer_widths:[^\n]+)", text),
        "matformer_export_line": _last_str(r"(matformer_export_slice[^\n]+)", text),
        "mlp_eq_line": _last_str(r"(mlp_eq_mode:[^\n]+)", text),
        "mlp_eq_calibration_line": _last_str(r"(mlp_eq_calibration[^\n]+)", text),
        "mlp_eq_applied_line": _last_str(r"(mlp_eq_applied[^\n]+)", text),
        "mixed_quant_int4_auto_mlp_line": _last_str(r"(mixed_quant_int4_auto_mlp[^\n]+)", text),
        "spectral_line": _last_str(r"(spectral_mode:[^\n]+spectral_layers:[^\n]+)", text),
        "ttt_line": _last_str(r"(ttt_enabled:[^\n]+)", text),
        "ve_line": _last_str(r"(VE:enabled:[^\n]+)", text),
        "late_qat_step": _last_int(r"late_qat:enabled step:(\d+)", text),
        "stop_step": _last_int(r"stopping_early: wallclock_cap train_time:\d+ms step:(\d+)/", text),
        "step_avg_ms": _last_float(r"step:\d+/\d+ .*?step_avg:([0-9.]+)ms", text),
        "post_ema_bpb": _last_float(r"DIAGNOSTIC post_ema val_loss:[0-9.]+ val_bpb:([0-9.]+)", text),
        "roundtrip_exact_bpb": _last_float(r"final_int6_roundtrip_exact val_loss:[0-9.]+ val_bpb:([0-9.]+)", text),
        "sliding_exact_bpb": _last_float(r"final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:([0-9.]+)", text),
        "int8_exact_bpb": _last_float(r"final_int8_zlib_roundtrip_exact val_loss:[0-9.]+ val_bpb:([0-9.]+)", text),
        "total_bytes": _last_int(r"Total submission size int[68]\+[a-z0-9]+: (\d+) bytes", text),
        "cap_status": _last_str(r"Submission cap check: (PASS|FAIL)", text),
        "cap_margin": _last_int(r"Submission cap check: (?:PASS|FAIL) cap:\d+ total:\d+ margin:([+-]?\d+)", text),
        "eval_time_ms": _last_int(r"final_int6_sliding_window .*?eval_time:(\d+)ms", text),
        "quant_codec": _last_str(r"quant_codec_selected:([^\s]+)", text),
    }
    codec_line = _last_str(r"quant_codec_sizes:([^\n]+)", text)
    result["quant_codec_sizes"] = _parse_codec_sizes(codec_line) if codec_line else {}
    result["errors"] = re.findall(
        r"(Traceback|RuntimeError:[^\n]+|ValueError:[^\n]+|SignalException[^\n]*)",
        text,
    )[-10:]
    return result


def _parse_codec_sizes(line: str) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for codec, size in re.findall(r"([a-z0-9]+):(\d+)", line):
        sizes[codec] = int(size)
    return sizes


def parse_path(path: Path) -> dict[str, Any]:
    return parse_text(read_log_text(path), source=str(path))


def format_summary(result: dict[str, Any]) -> str:
    fields = [
        ("run", result.get("run_id")),
        ("bpb", result.get("sliding_exact_bpb")),
        ("bytes", result.get("total_bytes")),
        ("cap", result.get("cap_status")),
        ("margin", result.get("cap_margin")),
        ("backend", result.get("backend")),
        ("codec", result.get("quant_codec")),
        ("step_ms", result.get("step_avg_ms")),
        ("eval_ms", result.get("eval_time_ms")),
    ]
    return " ".join(f"{key}:{value}" for key, value in fields if value is not None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse Parameter Golf experiment logs/archives.")
    parser.add_argument("paths", nargs="+", help="Log files or .tar.gz archives.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of one-line summaries.")
    args = parser.parse_args()

    results = [parse_path(Path(path)) for path in args.paths]
    if args.json:
        print(json.dumps(results if len(results) != 1 else results[0], indent=2, sort_keys=True))
    else:
        for result in results:
            print(format_summary(result))


if __name__ == "__main__":
    main()
