"""Independent safety monitor for a controlled DWQ process."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path


GIB = 1024**3


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--disk-stop-gib", type=float, required=True)
    parser.add_argument("--swap-growth-stop-gib", type=float, required=True)
    return parser.parse_args()


def swap_used_bytes() -> int:
    output = subprocess.run(
        ["sysctl", "vm.swapusage"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    match = re.search(r"used = ([0-9.]+)([MGT])", output)
    if not match:
        raise RuntimeError(f"Could not parse swap usage: {output.strip()}")
    scale = {"M": 1024**2, "G": 1024**3, "T": 1024**4}[match.group(2)]
    return int(float(match.group(1)) * scale)


def process_record(pid: int) -> dict[str, str | int]:
    output = subprocess.run(
        ["ps", "-o", "pid=,state=,rss=,etime=,command=", "-p", str(pid)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    pieces = output.split(maxsplit=4)
    if len(pieces) < 5:
        raise RuntimeError(f"Could not parse process state: {output}")
    return {
        "pid": int(pieces[0]),
        "state": pieces[1],
        "rss_kib": int(pieces[2]),
        "elapsed": pieces[3],
        "command": pieces[4],
    }


def memory_pressure_record() -> str:
    result = subprocess.run(
        ["memory_pressure", "-Q"],
        check=True,
        capture_output=True,
        text=True,
    )
    return " ".join(result.stdout.split())


def request_stop(path: Path, reason: str) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(reason + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = arguments()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    baseline_swap = swap_used_bytes()
    disk_floor = int(args.disk_stop_gib * GIB)
    swap_growth_limit = int(args.swap_growth_stop_gib * GIB)

    with args.log.open("a", encoding="utf-8", buffering=1) as log:
        while True:
            try:
                process = process_record(args.pid)
            except subprocess.CalledProcessError:
                return
            free_disk = shutil.disk_usage("/").free
            swap_used = swap_used_bytes()
            record = {
                "timestamp": time.time(),
                "free_disk_bytes": free_disk,
                "swap_used_bytes": swap_used,
                "swap_growth_bytes": swap_used - baseline_swap,
                "memory_pressure": memory_pressure_record(),
                "process": process,
            }
            log.write(json.dumps(record, sort_keys=True) + "\n")
            if "U" in str(process["state"]):
                request_stop(args.stop_file, f"process entered U state: {process['state']}")
                return
            if free_disk < disk_floor:
                request_stop(
                    args.stop_file,
                    f"free disk {free_disk} fell below floor {disk_floor}",
                )
                return
            if swap_used - baseline_swap > swap_growth_limit:
                request_stop(
                    args.stop_file,
                    f"swap growth {swap_used - baseline_swap} exceeded {swap_growth_limit}",
                )
                return
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
