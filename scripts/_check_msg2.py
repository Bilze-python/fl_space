import os
import subprocess

os.chdir(r"d:/Desktop/fl_space")
git = r"C:\Program Files\Git\bin\git.exe"
b = subprocess.check_output([git, "log", "--format=%B", "-1"])
print("bytes:", repr(b[:120]))
try:
    s = b.decode("utf-8")
    print("utf8-OK:", s[:50])
except Exception as e:
    print("utf8-FAIL:", e)
try:
    s = b.decode("gbk")
    print("gbk-OK:", s[:50])
except Exception as e:
    print("gbk-FAIL:", e)
