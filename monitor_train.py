#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training wrapper that monitors stdout/stderr for crashes, segfaults, and CUDA
errors, writes timestamped logs, and can relaunch with extra debugging flags
(e.g. CUDA_LAUNCH_BLOCKING) to pinpoint the exact line that crashes.

Usage:
    cd /home/wwj/文档/AVGZSL/新方案
    /home/wwj/anaconda3/envs/MSTR/bin/python monitor_train.py \
        --log_dir logs/mstr_monitor \
        --exp_name mstr_ucf_val_pure \
        -- ./run_stage_a.sh

The arguments after `--` are passed directly to the training subprocess.

Flags:
    --cuda_launch_blocking  Set CUDA_LAUNCH_BLOCKING=1 for the child. This
                            makes CUDA errors report the kernel that actually
                            crashed, at the cost of slower training. Use this
                            once a segfault has reproduced to get an accurate
                            traceback.
    --python_faulthandler   Set PYTHONFAULTHANDLER=1 (default: on). This prints
                            a Python traceback if the process receives a
                            fatal signal such as SIGSEGV.
"""
import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# Regex patterns that indicate a fatal or suspicious failure
FATAL_PATTERNS = [
    (re.compile(r"(?i)segfault|sigsegv|segmentation fault"), "SEGFAULT"),
    (re.compile(r"(?i)CUBLAS_STATUS_\w+"), "CUBLAS_ERROR"),
    (re.compile(r"(?i)CUDA error"), "CUDA_ERROR"),
    (re.compile(r"(?i)RuntimeError.*cuda"), "CUDA_RUNTIME_ERROR"),
    (re.compile(r"(?i)torch\.cuda\.(OutOfMemoryError|out of memory)"), "OOM"),
    (re.compile(r"(?i)RuntimeError"), "RUNTIME_ERROR"),
    (re.compile(r"(?i)Traceback \(most recent call first\)"), "PYTHON_TRACEBACK"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Monitor an MSTR training run")
    parser.add_argument("--log_dir", default="logs/monitor",
                        help="Directory to write monitor logs")
    parser.add_argument("--exp_name", default="",
                        help="Experiment name (used in log filenames)")
    parser.add_argument("--cuda_launch_blocking", action="store_true",
                        help="Set CUDA_LAUNCH_BLOCKING=1 for the child")
    parser.add_argument("--python_faulthandler", action="store_true",
                        default=True, help="Set PYTHONFAULTHANDLER=1")
    parser.add_argument("--tail_lines", type=int, default=80,
                        help="How many lines to print at the end of a failed run")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="Command to run after '--'")
    args = parser.parse_args()

    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("Please provide a command to run after '--'")
    return args


def make_log_paths(log_dir, exp_name):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = exp_name if exp_name else "train"
    stdout_log = log_dir / f"{name}_{timestamp}.log"
    summary_log = log_dir / f"{name}_{timestamp}_summary.log"
    return stdout_log, summary_log


def main():
    args = parse_args()
    stdout_log, summary_log = make_log_paths(args.log_dir, args.exp_name)

    env = os.environ.copy()
    if args.python_faulthandler:
        env["PYTHONFAULTHANDLER"] = "1"
    if args.cuda_launch_blocking:
        env["CUDA_LAUNCH_BLOCKING"] = "1"

    summary_lines = []
    summary_lines.append(f"Command: {' '.join(args.command)}")
    summary_lines.append(f"Start time: {datetime.now().isoformat()}")
    summary_lines.append(f"CUDA_LAUNCH_BLOCKING={env.get('CUDA_LAUNCH_BLOCKING', 'not set')}")
    summary_lines.append(f"PYTHONFAULTHANDLER={env.get('PYTHONFAULTHANDLER', 'not set')}")
    summary_lines.append(f"stdout log: {stdout_log}")
    summary_lines.append("-" * 70)

    print(f"[monitor] Starting: {' '.join(args.command)}")
    print(f"[monitor] Logs: {stdout_log}")
    print(f"[monitor] Summary: {summary_log}")
    if args.cuda_launch_blocking:
        print("[monitor] CUDA_LAUNCH_BLOCKING=1 enabled (slower but more accurate errors)")

    start = time.time()
    proc = subprocess.Popen(
        args.command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    detected_flags = set()
    last_lines = []
    line_count = 0
    with open(stdout_log, "w", encoding="utf-8") as f:
        for line in proc.stdout:
            line_count += 1
            f.write(line)
            f.flush()
            sys.stdout.write(line)
            sys.stdout.flush()

            last_lines.append(line.rstrip("\n"))
            if len(last_lines) > args.tail_lines:
                last_lines.pop(0)

            for pattern, flag in FATAL_PATTERNS:
                if pattern.search(line):
                    detected_flags.add(flag)

    returncode = proc.wait()
    elapsed = time.time() - start

    summary_lines.append(f"End time: {datetime.now().isoformat()}")
    summary_lines.append(f"Duration: {elapsed:.1f}s")
    summary_lines.append(f"Return code: {returncode}")
    summary_lines.append(f"Lines logged: {line_count}")
    if detected_flags:
        summary_lines.append(f"Detected flags: {', '.join(sorted(detected_flags))}")
    else:
        summary_lines.append("No fatal-error keywords detected in stdout")
    summary_lines.append("-" * 70)
    summary_lines.append("Tail of output:")
    summary_lines.extend(last_lines)
    summary_lines.append("-" * 70)

    if returncode != 0 or detected_flags:
        summary_lines.append(
            "\nThe run failed or triggered a fatal-error keyword. "
            "If the crash is a segfault/CUDA error and the exact line is unclear, "
            "re-run with --cuda_launch_blocking to get the true traceback location."
        )
        if not args.cuda_launch_blocking:
            summary_lines.append(
                "Suggested next run:\n"
                "  /home/wwj/anaconda3/envs/MSTR/bin/python monitor_train.py "
                "--cuda_launch_blocking --exp_name <name> -- ./run_stage_a.sh"
            )

    with open(summary_log, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"\n[monitor] Process exited with code {returncode}")
    print(f"[monitor] Summary: {summary_log}")
    if detected_flags:
        print(f"[monitor] Detected flags: {', '.join(sorted(detected_flags))}")
    return returncode


if __name__ == "__main__":
    sys.exit(main())
