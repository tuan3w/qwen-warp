#!/usr/bin/env python
"""Image input through the qwen-warp OpenAI server.  Needs an RTX 3090 and the checkpoint ($SWIFT_MODEL, default ~/models/swift-1.5-int4); starts and stops its own servers.

  python tests/test_vision_server.py [--port 8031] [--ref-tree DIR] [--vision auto|device|host] [--skip-off] [--bench N] [--keep]

--ref-tree DIR: a checkout of the previous release with its engine built (e.g. `git archive v0.7.0 | tar -x -C DIR && make -C DIR/engine`): its server runs first, its greedy ids of
the text prompts are written to tests/data/server_text_ids_v070.json (the reference of the text-identity checks; without --ref-tree that file is read) and, with --bench N, its
decode tok/s are the reference of the speed comparison.

Tier: 2 --skip-off
Checks (synthetic PIL images sent as base64 data URLs through the chat API, greedy, thinking off):
  text-only greedy ids == the reference release's (before and after the image requests: an image request leaves no state behind) |
  solid colours, shapes, large printed words and digits at 448x448, 1024x768, 1920x1080 and non-square sizes: the answer names the colour / shape / text |
  same text, two different images: different answers, and the second request resumes nowhere past the image (cached_tokens <= the image's first placeholder row) |
  multi-turn chat about one image: turn 2 reports cached_tokens > 0 and still answers about the image |
  two images in one prompt | the same image request twice -> the same ids | video -> 400 | --vision off (a second server): an image request -> 400, text ids unchanged.
--bench N: decode tok/s of the README short prompts (385 tokens, ignore_eos, median of N runs per prompt) on both servers, image-request time to first token at three sizes, decode
tok/s after an image vs the same request without it.
"""
import argparse
import base64
import io
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVE = HERE.parent / "server" / "serve.py"
REF = HERE / "data" / "server_text_ids_v070.json"
PROMPTS = {
    "code": "Write a Python function that returns the n-th Fibonacci number using memoisation, then explain its complexity.",
    "prose": "Write a short story about a lighthouse keeper who discovers that the light has been guiding something other than ships.",
    "math": "Prove that the sum of the first n odd numbers equals n squared, then give two different proofs.",
}
NEW = 64
NOTHINK = dict(chat_template_kwargs={"enable_thinking": False}, return_token_ids=True)


class T:
    def __init__(self):
        self.res = []

    def check(self, name, ok, detail=""):
        self.res.append((name, bool(ok)))
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not ok else (f"  ({detail})" if detail else "")), flush=True)


# ------------------------------------------------------------------------------------------------ images
def font(size):
    from PIL import ImageFont

    for name in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def solid(rgb, w=448, h=448):
    from PIL import Image

    return Image.new("RGB", (w, h), rgb)


def shape(kind, w=640, h=480):
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(im)
    s = min(w, h) * 0.35
    cx, cy = w / 2, h / 2
    if kind == "circle":
        d.ellipse([cx - s, cy - s, cx + s, cy + s], fill=(0, 0, 0))
    elif kind == "square":
        d.rectangle([cx - s, cy - s, cx + s, cy + s], fill=(0, 0, 0))
    else:
        d.polygon([(cx, cy - s), (cx - s, cy + s * 0.8), (cx + s, cy + s * 0.8)], fill=(0, 0, 0))
    return im


def words(text, w, h):
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(im)
    f = font(int(min(h * 0.4, w / (0.75 * len(text)))))
    x0, y0, x1, y1 = d.textbbox((0, 0), text, font=f)
    d.text(((w - (x1 - x0)) / 2 - x0, (h - (y1 - y0)) / 2 - y0), text, fill=(0, 0, 0), font=f)
    return im


def url(im):
    b = io.BytesIO()
    im.save(b, format="PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def user(ims, text):
    return {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url(im)}} for im in ims] + [{"type": "text", "text": text}]}


# ------------------------------------------------------------------------------------------------ server plumbing
def start(py, serve, port, args, log_path, env):
    log = open(log_path, "w")
    srv = subprocess.Popen([py, str(serve), "--port", str(port)] + args, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    import httpx

    t0 = time.time()
    while True:
        if srv.poll() is not None:
            raise SystemExit(f"server died, see {log_path}")
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                return srv, time.time() - t0
        except httpx.HTTPError:
            pass
        time.sleep(0.5)


def stop(srv):
    if srv.poll() is None:
        os.killpg(srv.pid, signal.SIGINT)
        try:
            srv.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(srv.pid, signal.SIGKILL)
            srv.wait()
    time.sleep(3)


def client(port):
    from openai import OpenAI

    return OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="x", timeout=300)


def chat(cl, messages, max_tokens=NEW, **extra):
    return cl.chat.completions.create(model="swift", messages=messages, temperature=0, max_tokens=max_tokens, extra_body=dict(NOTHINK, **extra))


def ids_of(out):
    ch = out.choices[0]
    return (ch.model_extra or {}).get("token_ids")


def cached(out):
    d = out.usage.prompt_tokens_details
    return d.cached_tokens if d is not None else 0


def text_ids(cl):
    return {n: ids_of(chat(cl, [{"role": "user", "content": p}])) for n, p in PROMPTS.items()}


def stream_stats(cl, messages, max_tokens=385):
    """(seconds to the first content chunk, decode tok/s = (tokens - 1) / (last chunk - first chunk), tokens) of one greedy streamed request of exactly max_tokens tokens."""
    t0 = time.time()
    first = last = None
    n = 0
    st = cl.chat.completions.create(model="swift", messages=messages, temperature=0, max_tokens=max_tokens, stream=True, stream_options={"include_usage": True},
                                    extra_body=dict(chat_template_kwargs={"enable_thinking": False}, ignore_eos=True))
    for c in st:
        if c.choices and (c.choices[0].delta.content or getattr(c.choices[0].delta, "reasoning", None)):
            last = time.time()
            first = first or last
        if c.usage:
            n = c.usage.completion_tokens
    return first - t0, (n - 1) / (last - first), n


def bench_text(cl, reps):
    return {name: statistics.median(stream_stats(cl, [{"role": "user", "content": p}])[1] for _ in range(reps)) for name, p in PROMPTS.items()}


# ------------------------------------------------------------------------------------------------ image checks
def image_checks(t, cl, base):
    import httpx

    def ask(ims, q, max_tokens=24):
        out = chat(cl, [user(ims, q)], max_tokens=max_tokens)
        return out, (out.choices[0].message.content or "").strip()

    def norm(s):
        return "".join(ch for ch in s.lower() if ch.isalnum())

    colour_q = "What is the colour of this image? Answer with one word."
    for rgb, name, size in [((220, 20, 20), "red", (448, 448)), ((20, 160, 40), "green", (1024, 768)), ((20, 40, 220), "blue", (1920, 1080)), ((250, 220, 0), "yellow", (300, 700))]:
        out, a = ask([solid(rgb, *size)], colour_q)
        t.check(f"solid {name} {size[0]}x{size[1]} -> names the colour", name in a.lower(), f"{a!r}, prompt {out.usage.prompt_tokens} tokens")
    for kind, size in [("circle", (640, 480)), ("square", (448, 448)), ("triangle", (800, 600))]:
        out, a = ask([shape(kind, *size)], "What geometric shape is drawn in this image? Answer with one word.")
        t.check(f"shape {kind} {size[0]}x{size[1]} -> names the shape", kind in a.lower(), f"{a!r}")
    for txt, size in [("HELLO", (800, 300)), ("2048", (300, 600)), ("OPEN DOOR", (1024, 768)), ("QWEN 3090", (1920, 1080))]:
        out, a = ask([words(txt, *size)], "What text is written in this image? Reply with the text only.")
        t.check(f"printed {txt!r} {size[0]}x{size[1]} -> reads it", norm(txt) in norm(a), f"{a!r}")

    # same text, two different images: different answers; the second must not resume past its image (a raw-id cache would: the plan cuts at the assistant turn after a 768-token image)
    q = colour_q
    o1, a1 = ask([solid((220, 20, 20), 1024, 768)], q)
    o1b, _ = ask([solid((220, 20, 20), 1024, 768)], q)
    t.check("the same image request again -> the same ids, resumed past the image (cold vs prefix-resumed)", ids_of(o1b) == ids_of(o1) and cached(o1b) > 768,
            f"cached {cached(o1)} / {cached(o1b)}")
    o2, a2 = ask([solid((20, 40, 220), 1024, 768)], q)
    t.check("same text, red then blue 1024x768 image -> different answers (no false cache hit)", "red" in a1.lower() and "blue" in a2.lower() and a1 != a2,
            f"{a1!r} / {a2!r}, cached {cached(o1)} / {cached(o2)}")
    t.check("the blue request resumes nowhere past its image (its placeholder rows start after '<|im_start|>user\\n<|vision_start|>')", cached(o2) <= 8, f"cached_tokens {cached(o2)}")

    # multi-turn about one image: turn 2 resumes from the cache (past the image) and still answers about it
    m = [user([words("TIGER", 1024, 512)], "What word is written in this image? Reply with the word only.")]
    r1 = chat(cl, m, max_tokens=16)
    w1 = (r1.choices[0].message.content or "").strip()
    m += [{"role": "assistant", "content": w1}, {"role": "user", "content": "Name the animal that word refers to, and how many letters the word has. Answer briefly."}]
    r2 = chat(cl, m, max_tokens=48)
    w2 = (r2.choices[0].message.content or "").strip()
    t.check("multi-turn image chat: turn 1 reads the word", "tiger" in w1.lower(), f"{w1!r}")
    t.check("multi-turn image chat: turn 2 cached_tokens > 0 and still about the image", cached(r2) > 0 and ("5" in w2 or "five" in w2.lower()) and "tiger" in w2.lower(),
            f"cached {cached(r2)} of {r2.usage.prompt_tokens}, {w2!r}")

    # two images in one prompt
    out, a = ask([solid((220, 20, 20)), solid((20, 40, 220))], "There are two images. What are their colours, in order? Answer as 'first, second'.")
    t.check("two images in one prompt -> both colours in order", "red" in a.lower() and "blue" in a.lower() and a.lower().index("red") < a.lower().index("blue"), f"{a!r}")

    # video is refused with a 400 that says so
    r = httpx.post(base + "/v1/chat/completions", json=dict(model="swift", max_tokens=8, messages=[{"role": "user", "content": [
        {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}}, {"type": "text", "text": "What happens?"}]}]), timeout=60)
    t.check("video input -> 400", r.status_code == 400 and "video" in r.text.lower(), f"{r.status_code} {r.text[:200]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8031)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--ref-tree", help="checkout of the previous release with its engine built (text-identity reference); without it the text checks compare this server with itself")
    ap.add_argument("--vision", default="auto", choices=["auto", "device", "host"])
    ap.add_argument("--skip-off", action="store_true", help="do not start the --vision off server")
    ap.add_argument("--bench", type=int, default=0, help="also measure speed (N runs per cell)")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    env = dict(os.environ, HF_HUB_OFFLINE="1", VLLM_SERVER_DEV_MODE="1")
    py = sys.executable
    common = ["--max-model-len", str(a.max_model_len)]
    t = T()
    ref, ref_bench = None, None
    if a.ref_tree:
        srv, dt = start(py, Path(a.ref_tree) / "server" / "serve.py", a.port, common, "/tmp/swift_mk_vision_ref.log", env)
        try:
            cl = client(a.port)
            ref = text_ids(cl)
            REF.parent.mkdir(exist_ok=True)
            json.dump(dict(tree=str(a.ref_tree), args=common, ids=ref), open(REF, "w"))
            if a.bench:
                ref_bench = bench_text(cl, a.bench)
                print("BENCH reference text decode tok/s", json.dumps({k: round(v, 1) for k, v in ref_bench.items()}), flush=True)
        finally:
            stop(srv)
    elif REF.exists():
        ref = json.load(open(REF))["ids"]
    srv, dt = start(py, SERVE, a.port, common + ["--vision", a.vision], "/tmp/swift_mk_vision_server.log", env)
    base = f"http://127.0.0.1:{a.port}"
    try:
        t.check("server with image input starts", True, f"ready in {dt:.1f} s")
        cl = client(a.port)
        mine = text_ids(cl)
        t.check("a text-identity reference exists", ref is not None, f"{REF} (write it once with --ref-tree)")
        if ref is not None:
            for n in PROMPTS:
                fd = next((i for i, (x, y) in enumerate(zip(mine[n], ref[n])) if x != y), None)
                t.check(f"text-only greedy ids == reference release [{n}]", mine[n] == ref[n], f"{len(mine[n])}/{len(ref[n])} ids, first difference at {fd}")
        image_checks(t, cl, base)
        again = text_ids(cl)
        t.check("text-only greedy ids after the image requests == before (no state left behind)", again == mine)
        if a.bench:
            b = bench_text(cl, a.bench)
            print("BENCH text decode tok/s", json.dumps({k: round(v, 1) for k, v in b.items()}), flush=True)
            if ref_bench:
                print("BENCH ratio to reference", json.dumps({k: round(b[k] / ref_bench[k], 3) for k in b}), flush=True)
            for size in [(448, 448), (1024, 768), (1920, 1080)]:
                res = []
                for i in range(a.bench):  # a fresh image each run (another colour): no prefix-cache hit
                    im = shape("circle", *size)
                    im.putpixel((i, 0), (i, 0, 0))
                    res.append(stream_stats(cl, [user([im], "Describe this image in detail.")], max_tokens=128))
                print(f"BENCH image {size[0]}x{size[1]}: TTFT median {statistics.median(r[0] for r in res) * 1e3:.0f} ms, decode {statistics.median(r[1] for r in res):.1f} tok/s (n={a.bench})", flush=True)
            img = [stream_stats(cl, [user([shape("circle", 1024, 768)], "Write a long story inspired by this image.")]) for _ in range(a.bench)]
            txt = [stream_stats(cl, [{"role": "user", "content": "Write a long story inspired by a black circle on a white background."}]) for _ in range(a.bench)]
            print(f"BENCH decode after a 1024x768 image {statistics.median(r[1] for r in img):.1f} tok/s vs text-only {statistics.median(r[1] for r in txt):.1f} tok/s (385 tokens, n={a.bench})", flush=True)
    finally:
        if not a.keep:
            stop(srv)
    if not a.skip_off and not a.keep:
        srv, dt = start(py, SERVE, a.port, common + ["--vision", "off"], "/tmp/swift_mk_vision_off.log", env)
        try:
            import httpx

            cl = client(a.port)
            off = text_ids(cl)
            t.check("--vision off: text-only greedy ids == the image server's", off == mine)
            r = httpx.post(base + "/v1/chat/completions", json=dict(model="swift", max_tokens=8, messages=[user([solid((220, 20, 20))], "Colour?")]), timeout=60)
            t.check("--vision off: image request -> 400", r.status_code == 400, f"{r.status_code} {r.text[:200]}")
        finally:
            stop(srv)
    bad = [n for n, ok in t.res if not ok]
    print(f"{len(t.res) - len(bad)}/{len(t.res)} passed" + (f"; FAILED: {bad}" if bad else ""), flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
