import json
import urllib.request
import os

token = os.environ.get('GITHUB_TOKEN', '')  # 从环境变量读取

def api(url):
    req = urllib.request.Request(url, headers={
        "User-Agent": "PS",
        "Authorization": f"Bearer {token}",
    })
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read().decode("utf-8"))

commit = api("https://api.github.com/repos/Bilze-python/fl_space/commits/master")
msg = commit["commit"]["message"]
print("=== commit message (GitHub API) ===")
print(repr(msg))
print("encoded utf-8:", repr(msg.encode("utf-8"))[:200])

print("\n=== commit message 前 60 字符每个字符的码点 ===")
for ch in msg[:60]:
    print(f"  U+{ord(ch):04X} {ch!r}")

print("\n=== 文件列表（含中文名）===")
tree = api(f"https://api.github.com/repos/Bilze-python/fl_space/git/trees/{commit['sha']}?recursive=1")
cn = [t["path"] for t in tree["tree"] if t["type"] == "blob" and any(ord(c) > 127 for c in t["path"])]
for p in cn[:30]:
    pb = p.encode("utf-8")
    print(f"  {pb!r} -> {p}")
print(f"共 {len(cn)} 个含非ASCII文件名")
