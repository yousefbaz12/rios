"""MCP Tool Integration Layer — Local Python Sandbox."""
from __future__ import annotations

import os
import subprocess
import tempfile


class PythonSandbox:
    def run(self, code: str, timeout: int = 5) -> dict:
        """Run python code in a subprocess, return stdout/stderr/status."""
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                         encoding="utf-8") as f:
            f.write(code)
            path = f.name
        try:
            res = subprocess.run(["python", path], capture_output=True, text=True,
                                 timeout=timeout, cwd=os.getcwd())
            return {
                "status": "ok" if res.returncode == 0 else "error",
                "stdout": res.stdout.strip()[-500:],
                "stderr": res.stderr.strip()[-500:],
                "returncode": res.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "stdout": "", "stderr": "Execution timed out"}
        finally:
            os.unlink(path)