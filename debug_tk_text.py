"""Test if tkinter ScrolledText modifies pasted URL content."""
import tkinter as tk
from tkinter import scrolledtext
import os

d = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(d, "test_replays", "8x8"), "r", encoding="utf-8") as f:
    url = f.read().strip()

# Create a Text widget, insert the URL, then read it back
root = tk.Tk()
root.withdraw()

st = scrolledtext.ScrolledText(root)
st.insert("1.0", "# paste URLs here, one per line\n")
st.insert("2.0", url)

read_back = st.get("1.0", "end-1c").strip()
st.destroy()
root.destroy()

# Compare
print(f"Original URL len: {len(url)}")
print(f"Read-back len:    {len(read_back)}")
print(f"Original starts:  {url[:80]}")
print(f"Read-back starts: {read_back[:80]}")
print(f"Original ends:    {url[-80:]}")
print(f"Read-back ends:   {read_back[-80:]}")
print(f"Equal: {url == read_back}")

# Check splitlines
lines = read_back.splitlines()
print(f"\nLines count: {len(lines)}")
for i, line in enumerate(lines):
    ls = line.strip()
    print(f"  Line {i}: len={len(ls)}, starts_http={ls.startswith('http')}")
