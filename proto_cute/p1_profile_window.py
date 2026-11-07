import json, urllib.request, time, sys
P = sys.argv[1] if len(sys.argv) > 1 else "8100"
CODE = ("import sqlite3\nfrom typing import Any\n\n def connect(db_path: str) -> sqlite3.Connection:\n    \"\"\"Open a connection.\"\"\"\n    return sqlite3.connect(db_path)\n\n"
        "def close(conn: sqlite3.Connection) -> None:\n    \"\"\"Close a connection.\"\"\"\n    return conn.close()\n\n") * 4 + "Continue the module:\n\n"
def go(n):
    req = urllib.request.Request(f"http://127.0.0.1:{P}/v1/completions", data=json.dumps(dict(model="swift", prompt=CODE, max_tokens=n, temperature=0.0, ignore_eos=True)).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))["usage"]["completion_tokens"]
go(20); go(20)                                                    # warm, prefix cached afterwards
urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{P}/start_profile", method="POST")).read()
t0 = time.time(); n = go(60); print("tokens", n, f"{time.time()-t0:.2f}s")
urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{P}/stop_profile", method="POST")).read()
