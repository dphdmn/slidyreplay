# release.py
import re
import subprocess
import sys

# Read version from pyproject.toml
with open("pyproject.toml") as f:
    content = f.read()

version = re.search(r'version = "(.+?)"', content).group(1)

tag = f"pypi-v{version}"

print(f"Version: {version}")
print(f"Tag: {tag}")

# Stage changed files
subprocess.run(["git", "add", "pyproject.toml", "build_package.py", ".github/workflows/publish.yml"])

# Commit
subprocess.run(["git", "commit", "-m", f"v{version}"])

# Push
subprocess.run(["git", "push"])

# Tag and push tag
subprocess.run(["git", "tag", tag])
subprocess.run(["git", "push", "origin", tag])

print(f"\nReleased {tag}")