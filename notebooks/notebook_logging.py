from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Mapping, Sequence


def run_logged(
    cmd: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    cwd: str | None = None,
    log_path: str | Path,
) -> None:
    """Run a command, stream combined stdout/stderr, and persist it to a log file."""

    target = Path(log_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    printable_cmd = " ".join(str(part) for part in cmd)
    print(f"Running: {printable_cmd}")
    print(f"Logging to: {target}")

    with target.open("w", encoding="utf-8") as handle:
        handle.write(f"$ {printable_cmd}\n\n")
        process = subprocess.Popen(
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=dict(env) if env is not None else None,
            cwd=cwd,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)

    print(f"Log saved to: {target}")
