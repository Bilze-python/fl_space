import os
import subprocess

os.chdir(r"d:/Desktop/fl_space")
git = r"C:\Program Files\Git\bin\git.exe"

# 检查最近一次提交的 message 字节
b = subprocess.check_output([git, "log", "--format=%B", "-1"])
print("=== commit message bytes ===")
print(repr(b))
for enc in ("utf-8", "gbk", "utf-8-sig"):
    try:
        print(f"decode {enc}: {b.decode(enc)}")
    except Exception as e:
        print(f"decode {enc}: FAIL {e}")

# 检查中文文件名的存储情况
files = subprocess.check_output([git, "ls-files"], cwd=r"d:/Desktop/fl_space").decode("utf-8", errors="replace").splitlines()
print("\n=== 中文文件名 (ls-files) ===")
for f in files:
    # 判断是否含非 ASCII
    if any(ord(c) > 127 for c in f):
        fb = f.encode("utf-8")
        print(f"  {fb!r}")
        for enc in ("utf-8", "gbk"):
            try:
                print(f"    decode {enc}: {fb.decode(enc)}")
            except Exception:
                print(f"    decode {enc}: FAIL")
