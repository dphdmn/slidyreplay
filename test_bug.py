import subprocess
import sys

if __name__ == "__main__":
    result = subprocess.run(
        [sys.executable, "main.py", "--file", "test_bugs/11x11"],
        capture_output=False,
    )
    sys.exit(result.returncode)
