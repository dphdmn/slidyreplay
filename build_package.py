# build_package.py
import shutil
import os
import subprocess
import glob

ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(ROOT, "_pkg_build")
PKG_DIR = os.path.join(BUILD_DIR, "slidyreplay")

# Clean previous build
if os.path.exists(BUILD_DIR):
    shutil.rmtree(BUILD_DIR)

os.makedirs(PKG_DIR)

# Copy all .py files from root
for f in glob.glob(os.path.join(ROOT, "*.py")):
    shutil.copy2(f, PKG_DIR)

# Copy folders
for folder in ["assets", "fonts"]:
    src = os.path.join(ROOT, folder)
    if os.path.exists(src):
        shutil.copytree(src, os.path.join(PKG_DIR, folder))

# Copy pyproject.toml and README to build dir
shutil.copy2(os.path.join(ROOT, "pyproject.toml"), BUILD_DIR)
shutil.copy2(os.path.join(ROOT, "README.md"), BUILD_DIR)

# Build
subprocess.run(["python", "-m", "build"], cwd=BUILD_DIR, check=True)

# Copy dist back
root_dist = os.path.join(ROOT, "pypi_dist")
if os.path.exists(root_dist):
    shutil.rmtree(root_dist)
shutil.copytree(os.path.join(BUILD_DIR, "dist"), root_dist)

# Cleanup
shutil.rmtree(BUILD_DIR)

print(f"\nDone! Package in {root_dist}/")