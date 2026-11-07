"""Builds hf/index.html from the recorded streams in data/*.json (real chunk arrival times).  python build_video.py"""
import json, math, html
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/home/fatman/models/swift-1.5-int4")
W = 62                                   # characters per code line (22px monospace in an 820px panel)
LH = 32                                  # line height
VIS = 560                                # visible code height

def load(label):
    d = json.load(open(f"data/{label}.json"))
    text, cum, lines, rows = "", [], 0, []
    t_first = d["chunks"][0][0]
    for t, delta in d["chunks"]:
        text += delta
        cum.append(len(tok(text, add_special_tokens=False)["input_ids"]))
    scale = d["tokens"] / cum[-1]; cum = [round(c * scale) for c in cum]
    for (t, delta), c in zip(d["chunks"], cum):
        nl = sum(max(1, math.ceil(len(l) / W)) for l in text[:0].split("\n")) if False else None
    # lines after each chunk
    acc = ""; line_counts = []
    for t, delta in d["chunks"]:
        acc += delta
        line_counts.append(sum(max(1, math.ceil(len(l) / W)) for l in acc.split("\n")))
    rates = []
    for (t, _), c in zip(d["chunks"], cum):
        rates.append(None if t <= t_first + 1e-6 else (c - cum[0]) / (t - t_first))
    return dict(label=label, chunks=d["chunks"], cum=cum, lines=line_counts, rates=rates, t_end=d["chunks"][-1][0], tokens=d["tokens"], prompt=d["prompt"],
                final_rate=(d["tokens"] - cum[0]) / (d["chunks"][-1][0] - t_first))

S = load("stock_vllm"); E = load("engine_dflash2")
speedup = S["t_end"] / E["t_end"]
print(f"stock {S['tokens']} tok {S['t_end']:.2f}s {S['final_rate']:.0f} tok/s | engine {E['tokens']} tok {E['t_end']:.2f}s {E['final_rate']:.0f} tok/s | {speedup:.2f}x")

R0 = 4.8                                  # race clock origin (s on the film timeline)
TITLE_END, RACE_END, STATS_END, TOTAL = 3.6, 13.4, 19.6, 24.0
ops = []                                  # [selector, props-json, time]
spans = {}                                # panel -> html of streamed text spans
def esc(s): return html.escape(s, quote=False)

def panel(side, D, color, name, sub):
    p = side
    chunk_html = []
    for i, (t, delta) in enumerate(D["chunks"]):
        chunk_html.append(f'<span id="{p}-c{i}" class="ch">{esc(delta)}</span>')
        ops.append([f"#{p}-c{i}", {"opacity": 1}, R0 + t, "set"])
        top = max(0, D["lines"][i] * LH - (VIS - 2 * LH))
        ops.append([f"#{p}-code-in", {"y": -top, "duration": 0.12, "ease": "power1.out"}, R0 + t, "to"])
    # counters as discrete states
    def series(el, items):
        h = []
        for k, (t, text) in enumerate(items):
            h.append(f'<span id="{el}-{k}" class="st">{esc(text)}</span>')
            if k == 0: ops.append([f"#{el}-{k}", {"opacity": 1}, 0, "set"])
            else:
                ops.append([f"#{el}-{k-1}", {"opacity": 0}, R0 + t, "set"]); ops.append([f"#{el}-{k}", {"opacity": 1}, R0 + t, "set"])
        return "".join(h)
    tokens_items = [(0, "0")] + [(t, f"{c}") for (t, _), c in zip(D["chunks"], D["cum"])]
    rate_items = [(0, "–")] + [(t, "–" if r is None else f"{r:.0f}") for (t, _), r in zip(D["chunks"], D["rates"])]
    step = 0.05; n = int(D["t_end"] / step) + 1
    clock_items = [(k * step, f"{k*step:.2f}") for k in range(n)] + [(D["t_end"], f"{D['t_end']:.2f}")]
    tokens_h, rate_h, clock_h = series(f"{p}-tok", tokens_items), series(f"{p}-rate", rate_items), series(f"{p}-clock", clock_items)
    # final badge
    ops.append([f"#{p}-badge", {"opacity": 1, "scale": 1, "duration": 0.35, "ease": "back.out(1.6)"}, R0 + D["t_end"] + 0.05, "fromTo", {"opacity": 0, "scale": 0.8}])
    return f'''
    <div class="panel {p}" id="{p}">
      <div class="phead"><div class="pname" style="color:{color}">{name}</div><div class="psub">{sub}</div></div>
      <div class="pstats">
        <div class="stat"><div class="sv" style="color:{color}"><span class="stack">{rate_h}</span></div><div class="sl">tok/s</div></div>
        <div class="stat"><div class="sv"><span class="stack">{tokens_h}</span></div><div class="sl">tokens</div></div>
        <div class="stat"><div class="sv"><span class="stack">{clock_h}</span><span class="unit"> s</span></div><div class="sl">elapsed</div></div>
      </div>
      <div class="pcode"><div class="code-in" id="{p}-code-in"><pre>{"".join(chunk_html)}</pre></div></div>
      <div class="badge" id="{p}-badge" style="border-color:{color};color:{color}">done in {D["t_end"]:.1f} s</div>
    </div>'''

left = panel("L", S, "#8fa6c1", "Stock vLLM 0.30", "MTP · Marlin W4A16")
right = panel("R", E, "#4ade80", "qwen-warp", "CUDA engine · DFlash2")

bars = [("code prompt", [("Stock vLLM", 97.5, "#8fa6c1"), ("qwen-warp · MTP", 212.9, "#38bdf8"), ("qwen-warp · DFlash2", 301.3, "#4ade80")]),
        ("prose prompt", [("Stock vLLM", 64.3, "#8fa6c1"), ("qwen-warp · MTP", 121.0, "#38bdf8"), ("qwen-warp · DFlash2", 129.2, "#4ade80")])]
bar_html = ""; bi = 0
for gi, (gname, items) in enumerate(bars):
    bar_html += f'<div class="bg" id="bg{gi}"><div class="bgname">{gname}</div>'
    for (lab, val, col) in items:
        w = val / 301.3 * 1000
        bar_html += f'<div class="brow"><div class="blab">{lab}</div><div class="btrack"><div class="bar" id="bar{bi}" style="width:{w:.0f}px;background:{col}"></div><div class="bval" id="bval{bi}" style="left:{w+18:.0f}px;color:{col}">{val:.0f}</div></div></div>'
        t0 = TITLE_END + 0 + 0
        ops.append([f"#bar{bi}", {"scaleX": 1, "duration": 0.7, "ease": "power3.out"}, RACE_END + 0.5 + gi * 1.3 + (bi % 3) * 0.25, "fromTo", {"scaleX": 0}])
        ops.append([f"#bval{bi}", {"opacity": 1, "duration": 0.3}, RACE_END + 0.9 + gi * 1.3 + (bi % 3) * 0.25, "fromTo", {"opacity": 0}])
        bi += 1
    bar_html += "</div>"
    ops.append([f"#bg{gi}", {"opacity": 1, "y": 0, "duration": 0.4}, RACE_END + 0.3 + gi * 1.3, "fromTo", {"opacity": 0, "y": 20}])

page = f'''<!doctype html>
<html lang="en"><head><meta charset="UTF-8" /><meta name="viewport" content="width=1920, height=1080" />
<script src="gsap.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
html, body {{ width:1920px; height:1080px; overflow:hidden; background:#0a0e13; }}
#root {{ position:relative; width:100%; height:100%; background:#0a0e13; font-family: "DejaVu Sans", "Liberation Sans", sans-serif; color:#e6edf3; }}
.scene {{ position:absolute; inset:0; }}
.center {{ display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; }}
.big {{ font-size:104px; font-weight:700; letter-spacing:-0.03em; }}
.mid {{ font-size:44px; color:#9fb0c3; margin-top:28px; }}
.small {{ font-size:28px; color:#9aabbf; margin-top:36px; }}
.green {{ color:#4ade80; }}
.panel {{ position:absolute; top:150px; width:880px; height:800px; background:#10161d; border:2px solid #1d2733; border-radius:22px; padding:28px 28px 0 28px; overflow:hidden; }}
.panel.L {{ left:60px; }} .panel.R {{ left:980px; border-color:#1f4b33; }}
.phead {{ display:flex; align-items:baseline; gap:18px; }}
.pname {{ font-size:40px; font-weight:700; }} .psub {{ font-size:20px; color:#9aabbf; }}
.pstats {{ display:flex; gap:44px; margin:18px 0 14px 0; height:96px; }}
.stat {{ min-width:150px; }} .sv {{ position:relative; height:62px; font-size:60px; font-weight:700; font-family:"DejaVu Sans Mono", monospace; line-height:62px; }}
.sl {{ font-size:20px; color:#9aabbf; letter-spacing:0.08em; text-transform:uppercase; }}
.stack {{ position:relative; display:inline-block; min-width:130px; height:62px; }} .stack .st {{ position:absolute; left:0; top:0; opacity:0; white-space:nowrap; }}
.unit {{ font-size:28px; color:#9aabbf; }}
.pcode {{ position:relative; height:{VIS}px; overflow:hidden; border-top:1px solid #1d2733; padding-top:6px; }}
.code-in pre {{ font-family:"DejaVu Sans Mono", monospace; font-size:22px; line-height:{LH}px; white-space:pre-wrap; word-break:break-all; color:#d2dae3; width:824px; }}
.ch {{ opacity:0; }}
.badge {{ position:absolute; right:28px; top:26px; font-size:30px; font-weight:700; padding:6px 20px; border:3px solid; border-radius:40px; background:#10161d; opacity:0; }}
.prompt {{ position:absolute; top:40px; left:60px; right:60px; font-size:26px; color:#9fb0c3; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.prompt b {{ color:#e6edf3; }}
.foot {{ position:absolute; bottom:36px; left:0; right:0; text-align:center; font-size:24px; color:#9aabbf; }}
#verdict {{ position:absolute; top:960px; left:0; right:0; text-align:center; font-size:46px; font-weight:700; opacity:0; }}
.bgname {{ font-size:32px; color:#9fb0c3; margin:0 0 14px 0; text-transform:uppercase; letter-spacing:0.1em; }}
#bars {{ position:absolute; left:140px; top:190px; }}
.bg {{ margin-bottom:56px; opacity:0; }}
.brow {{ display:flex; align-items:center; margin-bottom:14px; }}
.blab {{ width:330px; font-size:30px; color:#cfd9e4; }}
.btrack {{ position:relative; width:1130px; height:62px; }}
.bar {{ height:62px; border-radius:10px; transform-origin:left center; }}
.bval {{ position:absolute; top:0; font-size:46px; font-weight:700; line-height:62px; font-family:"DejaVu Sans Mono", monospace; opacity:0; }}
.link {{ font-size:50px; color:#4ade80; margin-top:48px; font-family:"DejaVu Sans Mono", monospace; }}
</style></head>
<body>
<div id="root" data-composition-id="main" data-start="0" data-duration="{TOTAL}" data-width="1920" data-height="1080">

  <div class="scene clip" id="s-title" data-start="0" data-duration="{TITLE_END}" data-track-index="0">
    <div class="center" style="position:absolute;inset:0" id="title-in">
      <div class="big">Qwen3.8-27B on <span class="green">one RTX 3090</span></div>
      <div class="mid">same prompt · same card · greedy · one user</div>
      <div class="small">real recordings, replayed at real speed</div>
    </div>
  </div>

  <div class="scene clip" id="s-race" data-start="{TITLE_END}" data-duration="{RACE_END - TITLE_END}" data-track-index="1">
    <div id="race-in" style="position:absolute;inset:0">
      <div class="prompt"><b>Prompt</b> &nbsp;“Write a Python function that merges overlapping intervals … then a few pytest test cases.”</div>
      {left}
      {right}
      <div id="verdict" class="green">same prompt, {speedup:.1f}× sooner</div>
      <div class="foot">Swift-1.5 Qwen3.8-27B INT4 · RTX 3090 · single stream · greedy decoding</div>
    </div>
  </div>

  <div class="scene clip" id="s-stats" data-start="{RACE_END}" data-duration="{STATS_END - RACE_END}" data-track-index="2">
    <div id="stats-in" style="position:absolute;inset:0">
      <div style="position:absolute;top:56px;left:140px;font-size:56px;font-weight:700">Decode speed, tokens per second</div>
      <div id="bars">{bar_html}</div>
      <div class="foot">Swift-1.5 Qwen3.8-27B INT4 · RTX 3090 (350 W) · single stream · greedy · stock = vLLM 0.30.0 with MTP</div>
    </div>
  </div>

  <div class="scene clip" id="s-end" data-start="{STATS_END}" data-duration="{TOTAL - STATS_END}" data-track-index="3">
    <div class="center" style="position:absolute;inset:0" id="end-in">
      <div class="big">qwen-warp</div>
      <div class="mid">Qwen3.8-27B at up to <span class="green">300 tok/s</span> on a single RTX 3090</div>
      <div class="mid" style="font-size:34px;margin-top:14px">262k context · images · OpenAI-compatible API</div>
      <div class="link">github.com/tuan3w/qwen-warp</div>
    </div>
  </div>
</div>
<script>
const tl = gsap.timeline({{ paused: true }});
const OPS = {json.dumps(ops)};
for (const [sel, props, t, kind, from] of OPS) {{
  if (kind === "set") tl.set(sel, props, t);
  else if (kind === "fromTo") tl.fromTo(sel, from, props, t);
  else tl.to(sel, props, t);
}}
// scene transitions: fades on inner wrappers
tl.fromTo("#title-in", {{ opacity: 0, y: 30 }}, {{ opacity: 1, y: 0, duration: 0.7, ease: "power3.out" }}, 0.15);
tl.to("#title-in", {{ opacity: 0, duration: 0.4 }}, {TITLE_END - 0.4});
tl.fromTo("#race-in", {{ opacity: 0 }}, {{ opacity: 1, duration: 0.5 }}, {TITLE_END});
tl.fromTo("#L", {{ x: -60, opacity: 0 }}, {{ x: 0, opacity: 1, duration: 0.6, ease: "power3.out" }}, {TITLE_END + 0.2});
tl.fromTo("#R", {{ x: 60, opacity: 0 }}, {{ x: 0, opacity: 1, duration: 0.6, ease: "power3.out" }}, {TITLE_END + 0.2});
tl.fromTo("#verdict", {{ opacity: 0, y: 20 }}, {{ opacity: 1, y: 0, duration: 0.5, ease: "power3.out" }}, {R0 + S["t_end"] + 0.4});
tl.to("#race-in", {{ opacity: 0, duration: 0.5 }}, {RACE_END - 0.5});
tl.fromTo("#stats-in", {{ opacity: 0 }}, {{ opacity: 1, duration: 0.4 }}, {RACE_END});
tl.to("#stats-in", {{ opacity: 0, duration: 0.5 }}, {STATS_END - 0.5});
tl.fromTo("#end-in", {{ opacity: 0, y: 30 }}, {{ opacity: 1, y: 0, duration: 0.7, ease: "power3.out" }}, {STATS_END + 0.1});
window.__timelines["main"] = tl;
tl.seek(0);
</script>
</body></html>'''
open("hf/index.html", "w").write(page)
print("wrote hf/index.html", len(page) // 1024, "KB,", len(ops), "timeline ops")
