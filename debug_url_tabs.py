"""Debug: compare URL tab vs File tab params."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_video import parse_replay_url, expand_solution

d = os.path.dirname(os.path.abspath(__file__))

# Simulate URL tab: read URL from ScrolledText get("1.0", "end-1c")
with open(os.path.join(d, "test_replays", "8x8"), "r", encoding="utf-8") as f:
    file_content = f.read()

# Simulate ScrolledText behavior
# On Windows, tk normalizes \r\n to \n
scrolledtext_content = file_content.replace('\r\n', '\n').strip()

print("=== File raw bytes (first 50) ===")
print(repr(file_content[:50]))
print()

print("=== ScrolledText simulation (first 50) ===")
print(repr(scrolledtext_content[:50]))
print()

# Parse both
sol1, tps1, scram1, mtimes1 = parse_replay_url(file_content.strip())
sol2, tps2, scram2, mtimes2 = parse_replay_url(scrolledtext_content)

print("=== File tab (direct read) ===")
print(f"  sol_len={len(expand_solution(sol1))}, tps={tps1}, scramble={'yes' if scram1 else 'no'}, movetimes_type={type(mtimes1).__name__}, movetimes_len={len(mtimes1) if isinstance(mtimes1, list) else 'N/A'}")
if isinstance(mtimes1, list):
    print(f"  movetimes[0]={mtimes1[0]}, movetimes[-1]={mtimes1[-1]}, len={len(mtimes1)}")

print()
print("=== URL tab (scrolledtext simulate) ===")
print(f"  sol_len={len(expand_solution(sol2))}, tps={tps2}, scramble={'yes' if scram2 else 'no'}, movetimes_type={type(mtimes2).__name__}, movetimes_len={len(mtimes2) if isinstance(mtimes2, list) else 'N/A'}")
if isinstance(mtimes2, list):
    print(f"  movetimes[0]={mtimes2[0]}, movetimes[-1]={mtimes2[-1]}, len={len(mtimes2)}")

# Also test splitlines behavior (URL tab reads line by line)
print()
print("=== ScrolledText splitlines behavior ===")
lines = scrolledtext_content.splitlines()
print(f"  Number of lines: {len(lines)}")
for i, line in enumerate(lines):
    line = line.strip()
    if line and not line.startswith("#"):
        trimmed = line[:100]
        print(f"  Line {i}: starts_with_http={line.startswith('http')}, len={len(line)}, preview={trimmed}...")
        if line.startswith(("http://", "https://")):
            sol3, tps3, scram3, mtimes3 = parse_replay_url(line)
            print(f"    sol_len={len(expand_solution(sol3))}, tps={tps3}")
