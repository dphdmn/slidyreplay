# Implementation Plan

Cross-platform compatibility fix to ensure slidyreplay runs correctly on Linux, addressing platform-specific paths and providing clear README documentation for Linux users.

The ffmpeg subprocess calls and PyTorch/CUDA code are already cross-platform and require no changes.

[Remaining Work]
- **README.md** — add Linux installation instructions, Linux-specific section about ffmpeg/Distro package names, PyTorch/CUDA on Linux, differences from Windows (no build.bat, no bundled ffmpeg).
- **TODO** — mark item 2 ("Clear the project for import in google collabs and linux support") as completed.

[Testing]
Manual verification approach:

1. On Linux: install `python3-venv`, create venv, pip install dependencies, run `python3 main.py` and verify GUI opens without errors.
2. On Linux: test CLI with `python3 main.py --solution R2D2L2U2 --size 3x3 --tps 10 -o test.mp4` and confirm ffmpeg encoding works.
3. On Windows: verify no regressions — GUI opens, CLI works, font rendering is unchanged.

[Implementation Order]
1. Update README.md with Linux installation instructions, dependency notes, and platform-specific guidance.
2. Update TODO to mark Linux support as done.
