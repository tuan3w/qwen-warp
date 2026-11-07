"""Builds hf/index.html from the recorded streams in data/*.json (real chunk arrival times).  python build_video.py
Scenes: title, one user before (MTP) / now (DFlash2) on the same prompt, 4 users at once, published numbers, end card."""

import json, math, html
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("/home/fatman/models/swift-1.5-int4")
MONO_ADV = 0.602  # DejaVu Sans Mono advance width, em


def stream(chunks, tokens):
    """Per chunk: cumulative tokens (tokenizer counts scaled to the server's usage) and the decode rate since the first chunk."""
    text, cum = "", []
    for _, delta in chunks:
        text += delta
        cum.append(len(tok(text, add_special_tokens=False)["input_ids"]))
    scale = tokens / cum[-1]
    cum = [round(c * scale) for c in cum]
    t_first = chunks[0][0]
    rates = [
        None if t <= t_first + 1e-6 else (c - cum[0]) / (t - t_first)
        for (t, _), c in zip(chunks, cum)
    ]
    return dict(
        chunks=chunks,
        cum=cum,
        rates=rates,
        t_end=chunks[-1][0],
        tokens=tokens,
        final_rate=(tokens - cum[0]) / (chunks[-1][0] - t_first),
    )


def load(label):
    d = json.load(open(f"data/{label}.json"))
    return stream(d["chunks"], d["tokens"])


B = load("engine_mtp")
N = load("engine_dflash2")
C = json.load(open("data/engine_4users.json"))
U = [stream(s["chunks"], s["tokens"]) for s in C["streams"]]
speedup = B["t_end"] / N["t_end"]
agg = C["tokens"] / C["t_end"]
print(
    f"MTP {B['tokens']} tok {B['t_end']:.2f}s {B['final_rate']:.0f} tok/s | DFlash2 {N['tokens']} tok {N['t_end']:.2f}s {N['final_rate']:.0f} tok/s | {speedup:.2f}x"
    f" | 4 users {C['tokens']} tok {C['t_end']:.2f}s {agg:.0f} tok/s"
)

TITLE_END = 3.4
R0 = 4.4
RACE_END = 10.0  # race clock origin (film seconds), scene end
C0 = 11.0
CONC_END = 18.6
STATS_END = 24.2
TOTAL = 27.6
ops = []  # [selector, props, time, kind, from]


def esc(s):
    return html.escape(s, quote=False)


def series(el, items, t0):
    """Discrete text states shown at film time t0 + t."""
    h = []
    for k, (t, text) in enumerate(items):
        h.append(f'<span id="{el}-{k}" class="st">{esc(text)}</span>')
        if k == 0:
            ops.append([f"#{el}-{k}", {"opacity": 1}, 0, "set"])
        else:
            ops.append([f"#{el}-{k - 1}", {"opacity": 0}, t0 + t, "set"])
            ops.append([f"#{el}-{k}", {"opacity": 1}, t0 + t, "set"])
    return "".join(h)


def panel(p, D, t0, color, name, sub, x, y, w, h, fs, lh, clock=True):
    """A terminal-like panel streaming D's chunks at their recorded times (film time t0 + t), with tok/s, token and clock counters."""
    code_w = w - 56
    vis = h - (232 if clock else 150)
    W = int(code_w / (fs * MONO_ADV))
    acc, chunk_html = "", []
    for i, (t, delta) in enumerate(D["chunks"]):
        acc += delta
        lines = sum(max(1, math.ceil(len(l) / W)) for l in acc.split("\n"))
        chunk_html.append(f'<span id="{p}-c{i}" class="ch">{esc(delta)}</span>')
        ops.append([f"#{p}-c{i}", {"opacity": 1}, t0 + t, "set"])
        top = max(0, lines * lh - (vis - 2 * lh))
        ops.append(
            [
                f"#{p}-code-in",
                {"y": -top, "duration": 0.12, "ease": "power1.out"},
                t0 + t,
                "to",
            ]
        )
    rate_h = series(
        f"{p}-rate",
        [(0, "–")]
        + [
            (t, "–" if r is None else f"{r:.0f}")
            for (t, _), r in zip(D["chunks"], D["rates"])
        ],
        t0,
    )
    tokens_h = series(
        f"{p}-tok",
        [(0, "0")] + [(t, f"{c}") for (t, _), c in zip(D["chunks"], D["cum"])],
        t0,
    )
    stats = (
        f'<div class="stat"><div class="sv" style="color:{color}"><span class="stack">{rate_h}</span></div><div class="sl">tok/s</div></div>'
        f'<div class="stat"><div class="sv"><span class="stack">{tokens_h}</span></div><div class="sl">tokens</div></div>'
    )
    if clock:
        step = 0.05
        n = int(D["t_end"] / step) + 1
        clock_h = series(
            f"{p}-clock",
            [(k * step, f"{k * step:.2f}") for k in range(n)]
            + [(D["t_end"], f"{D['t_end']:.2f}")],
            t0,
        )
        stats += f'<div class="stat"><div class="sv"><span class="stack">{clock_h}</span><span class="unit"> s</span></div><div class="sl">elapsed</div></div>'
    ops.append(
        [
            f"#{p}-badge",
            {"opacity": 1, "scale": 1, "duration": 0.35, "ease": "back.out(1.6)"},
            t0 + D["t_end"] + 0.05,
            "fromTo",
            {"opacity": 0, "scale": 0.8},
        ]
    )
    size = "" if clock else " small"
    return f'''
    <div class="panel{size}" id="{p}" data-layout-allow-occlusion style="left:{x}px;top:{y}px;width:{w}px;height:{h}px;border-color:{color}55">
      <div class="phead"><div class="pname" style="color:{color}">{name}</div><div class="psub">{esc(sub)}</div></div>
      <div class="pstats">{stats}</div>
      <div class="pcode" data-layout-allow-overflow style="height:{vis}px"><div class="code-in" id="{p}-code-in"><pre style="font-size:{fs}px;line-height:{lh}px;width:{code_w}px">{"".join(chunk_html)}</pre></div></div>
      <div class="badge" id="{p}-badge" data-layout-allow-occlusion style="border-color:{color};color:{color}">done in {D["t_end"]:.1f} s</div>
    </div>'''


race = panel(
    "L",
    B,
    R0,
    "#38bdf8",
    "Before",
    "qwen-warp · MTP drafter",
    60,
    150,
    880,
    800,
    22,
    32,
) + panel(
    "R",
    N,
    R0,
    "#4ade80",
    "Now",
    "qwen-warp · DFlash2 drafter",
    980,
    150,
    880,
    800,
    22,
    32,
)

USER_SUBS = ["Python + tests", "explain CPU caches", "SQL query", "train arithmetic"]
conc = "".join(
    panel(
        f"U{i}",
        D,
        C0,
        "#4ade80",
        f"User {i + 1}",
        USER_SUBS[i],
        60 + (i % 2) * 920,
        170 + (i // 2) * 380,
        880,
        360,
        18,
        25,
        clock=False,
    )
    for i, D in enumerate(U)
)
# aggregate counter: all tokens so far / time since the 4 requests were sent
events = sorted(
    (t, i, c) for i, D in enumerate(U) for (t, _), c in zip(D["chunks"], D["cum"])
)
cur = [0] * len(U)
agg_items, tot_items = [(0, "–")], [(0, "0")]
for t, i, c in events:
    cur[i] = c
    tot = sum(cur)
    tot_items.append((t, f"{tot:,}"))
    agg_items.append((t, "–" if t < 0.4 else f"{tot / t:.0f}"))
agg_h, tot_h = series("agg", agg_items, C0), series("tot", tot_items, C0)
ops.append(
    [
        "#conc-verdict",
        {"opacity": 1, "y": 0, "duration": 0.5, "ease": "power3.out"},
        C0 + C["t_end"] + 0.3,
        "fromTo",
        {"opacity": 0, "y": 20},
    ]
)

BAR_MAX = 440.5
groups = [
    (
        "one user · short code prompt",
        [("MTP drafter", 183.9, "#38bdf8"), ("DFlash2 drafter", 276.7, "#4ade80")],
    ),
    (
        "several users at once · total",
        [
            ("1 user", 154.4, "#8fa6c1"),
            ("4 users", 424.9, "#4ade80"),
            ("8 users", 440.5, "#4ade80"),
        ],
    ),
]
bar_html = ""
bi = 0
for gi, (gname, items) in enumerate(groups):
    bar_html += f'<div class="bg" id="bg{gi}"><div class="bgname">{gname}</div>'
    g0 = CONC_END + 0.3 + gi * 1.6
    ops.append(
        [
            f"#bg{gi}",
            {"opacity": 1, "y": 0, "duration": 0.4},
            g0,
            "fromTo",
            {"opacity": 0, "y": 20},
        ]
    )
    for j, (lab, val, col) in enumerate(items):
        w = val / BAR_MAX * 1000
        bar_html += f'<div class="brow"><div class="blab">{lab}</div><div class="btrack"><div class="bar" id="bar{bi}" style="width:{w:.0f}px;background:{col}"></div><div class="bval" id="bval{bi}" style="left:{w + 18:.0f}px;color:{col}">{val:.0f}</div></div></div>'
        ops.append(
            [
                f"#bar{bi}",
                {"scaleX": 1, "duration": 0.7, "ease": "power3.out"},
                g0 + 0.2 + j * 0.25,
                "fromTo",
                {"scaleX": 0},
            ]
        )
        ops.append(
            [
                f"#bval{bi}",
                {"opacity": 1, "duration": 0.3},
                g0 + 0.6 + j * 0.25,
                "fromTo",
                {"opacity": 0},
            ]
        )
        bi += 1
    bar_html += "</div>"

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
.h1 {{ position:absolute; top:44px; left:60px; font-size:52px; font-weight:700; }}
.panel {{ position:absolute; background:#10161d; border:2px solid #1d2733; border-radius:22px; padding:28px 28px 0 28px; overflow:hidden; }}
.phead {{ display:flex; align-items:baseline; gap:18px; }}
.pname {{ font-size:40px; font-weight:700; }} .psub {{ font-size:22px; color:#9aabbf; }}
.pstats {{ display:flex; gap:44px; margin:18px 0 14px 0; height:96px; }}
.stat {{ min-width:150px; }} .sv {{ position:relative; height:62px; font-size:60px; font-weight:700; font-family:"DejaVu Sans Mono", monospace; line-height:62px; }}
.sl {{ font-size:20px; color:#9aabbf; letter-spacing:0.08em; text-transform:uppercase; }}
.stack {{ position:relative; display:inline-block; min-width:130px; height:62px; }} .stack .st {{ position:absolute; left:0; top:0; opacity:0; white-space:nowrap; }}
.unit {{ font-size:28px; color:#9aabbf; }}
.panel.small {{ padding:20px 28px 0 28px; }}
.panel.small .pname {{ font-size:30px; }}
.panel.small .pstats {{ margin:6px 0 8px 0; height:60px; gap:36px; }}
.panel.small .sv, .panel.small .stack {{ height:40px; font-size:38px; line-height:40px; }}
.panel.small .sl {{ font-size:20px; }}
.panel.small .badge {{ top:18px; font-size:24px; }}
.pcode {{ position:relative; overflow:hidden; border-top:1px solid #1d2733; padding-top:6px; }}
.code-in pre {{ font-family:"DejaVu Sans Mono", monospace; white-space:pre-wrap; word-break:break-all; color:#d2dae3; }}
.ch {{ opacity:0; }}
.badge {{ position:absolute; right:28px; top:26px; font-size:30px; font-weight:700; padding:6px 20px; border:3px solid; border-radius:40px; background:#10161d; opacity:0; }}
.prompt {{ position:absolute; top:44px; left:60px; right:60px; font-size:26px; color:#9fb0c3; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.prompt b {{ color:#e6edf3; }}
.foot {{ position:absolute; bottom:36px; left:0; right:0; text-align:center; font-size:24px; color:#9aabbf; }}
.verdict {{ position:absolute; top:968px; left:0; right:0; text-align:center; font-size:46px; font-weight:700; opacity:0; }}
#aggbox {{ position:absolute; top:34px; right:60px; display:flex; gap:44px; }}
#aggbox .sv {{ font-size:64px; height:70px; line-height:70px; }} #aggbox .stack {{ height:70px; min-width:170px; }}
.bgname {{ font-size:32px; color:#9fb0c3; margin:0 0 14px 0; text-transform:uppercase; letter-spacing:0.1em; }}
#bars {{ position:absolute; left:140px; top:180px; }}
.bg {{ margin-bottom:48px; opacity:0; }}
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
      <div class="big">qwen-warp now runs <span class="green">DFlash2</span></div>
      <div class="mid">Qwen3.8-27B INT4 · one RTX 3090 · faster, and several users at once</div>
      <div class="small">real streams recorded from the server, replayed at real speed</div>
    </div>
  </div>

  <div class="scene clip" id="s-race" data-start="{TITLE_END}" data-duration="{RACE_END - TITLE_END}" data-track-index="1">
    <div id="race-in" style="position:absolute;inset:0">
      <div class="prompt"><b>One user</b> &nbsp;“Write a Python function that merges overlapping intervals … then a few pytest test cases.”</div>
      {race}
      <div class="verdict green" id="race-verdict">same prompt, same {N["tokens"]} tokens, {speedup:.2f}× sooner · greedy</div>
    </div>
  </div>

  <div class="scene clip" id="s-conc" data-start="{RACE_END}" data-duration="{CONC_END - RACE_END}" data-track-index="2">
    <div id="conc-in" style="position:absolute;inset:0">
      <div class="h1">4 users at once</div>
      <div id="aggbox">
        <div class="stat"><div class="sv green"><span class="stack">{agg_h}</span></div><div class="sl">tok/s in total</div></div>
        <div class="stat"><div class="sv"><span class="stack">{tot_h}</span></div><div class="sl">tokens</div></div>
      </div>
      {conc}
      <div class="verdict green" id="conc-verdict" style="top:984px;font-size:40px">{C["tokens"]:,} tokens for 4 users in {C["t_end"]:.1f} s · {agg:.0f} tok/s on one card</div>
    </div>
  </div>

  <div class="scene clip" id="s-stats" data-start="{CONC_END}" data-duration="{STATS_END - CONC_END}" data-track-index="3">
    <div id="stats-in" style="position:absolute;inset:0">
      <div style="position:absolute;top:56px;left:140px;font-size:56px;font-weight:700">Tokens per second, one RTX 3090</div>
      <div id="bars">{bar_html}</div>
      <div class="foot">greedy · 350 W · several users: --profile multiuser (MTP drafter) · prose prompt: DFlash2 113 vs MTP 118 · docs/PERFORMANCE.md</div>
    </div>
  </div>

  <div class="scene clip" id="s-end" data-start="{STATS_END}" data-duration="{TOTAL - STATS_END}" data-track-index="4">
    <div class="center" style="position:absolute;inset:0" id="end-in">
      <div class="big">qwen-warp</div>
      <div class="mid">now with DFlash2: up to <span class="green">300 tok/s</span> for one user</div>
      <div class="mid" style="margin-top:14px"><span class="green">400+ tok/s</span> in total for several users</div>
      <div class="mid" style="font-size:34px;margin-top:20px">262k context · images · OpenAI-compatible API</div>
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
tl.fromTo("#title-in", {{ y: 12 }}, {{ y: 0, duration: 0.8, ease: "power3.out" }}, 0);
tl.to("#title-in", {{ opacity: 0, duration: 0.4 }}, {TITLE_END - 0.4});
tl.fromTo("#race-in", {{ opacity: 0 }}, {{ opacity: 1, duration: 0.5 }}, {TITLE_END});
tl.fromTo("#L", {{ x: -60, opacity: 0 }}, {{ x: 0, opacity: 1, duration: 0.6, ease: "power3.out" }}, {TITLE_END + 0.2});
tl.fromTo("#R", {{ x: 60, opacity: 0 }}, {{ x: 0, opacity: 1, duration: 0.6, ease: "power3.out" }}, {TITLE_END + 0.2});
tl.fromTo("#race-verdict", {{ opacity: 0, y: 20 }}, {{ opacity: 1, y: 0, duration: 0.5, ease: "power3.out" }}, {R0 + B["t_end"] + 0.4});
tl.to("#race-in", {{ opacity: 0, duration: 0.5 }}, {RACE_END - 0.5});
tl.fromTo("#conc-in", {{ opacity: 0 }}, {{ opacity: 1, duration: 0.5 }}, {RACE_END});
tl.to("#conc-in", {{ opacity: 0, duration: 0.5 }}, {CONC_END - 0.5});
tl.fromTo("#stats-in", {{ opacity: 0 }}, {{ opacity: 1, duration: 0.4 }}, {CONC_END});
tl.to("#stats-in", {{ opacity: 0, duration: 0.5 }}, {STATS_END - 0.5});
tl.fromTo("#end-in", {{ opacity: 0, y: 30 }}, {{ opacity: 1, y: 0, duration: 0.7, ease: "power3.out" }}, {STATS_END + 0.1});
window.__timelines["main"] = tl;
tl.seek(0);
</script>
</body></html>'''
open("hf/index.html", "w").write(page)
print("wrote hf/index.html", len(page) // 1024, "KB,", len(ops), "timeline ops")
