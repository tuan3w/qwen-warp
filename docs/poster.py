"""Builds infographic.svg (1200 px wide poster). All numbers come from data.json / the snapshots / git (see gen.py)."""
import math
from datetime import datetime

from svgkit import *  # noqa: F401,F403
from svgkit import Svg, tw, wrap, esc, highlight


def fx(v, d=2):
    return f"\u00d7{v:.{d}f}"


def fmt_dt(t, f="%b %d %H:%M"):
    return datetime.fromtimestamp(t).astimezone().strftime(f)


# =============================================================================================== 0. title
def title_block(s, y, D, C, X):
    hd, cp = D["headline"], D["campaign"]
    s.text(M, y + 38, "AN EXPERIMENT IN AUTONOMOUS RESEARCH  \u00b7  ONE RTX 3090  \u00b7  ~2 DAYS", 13, AGENT, weight=700, ls=2.4)
    s.text(M, y + 112, "qwen-warp", 76, INK, weight=900)
    s.text(M + 4, y + 158, D["meta"]["question"], 31, INK, weight=700)
    allx = X["x_srv_greedy"] + X["x_srv_t1"]
    lo, hi = min(allx), max(allx)
    ans = (f"Yes, on decode speed. A human directed AI agents; they built a Qwen server that decodes "
           f"{fx(lo, 1)}\u2013{fx(hi, 1)} faster than stock vLLM (MTP k=3, greedy and sampled, same client and card) and {fx(X['x_plain'], 1)} faster on plain decode.")
    y2 = s.para(M + 4, y + 196, ans, 1110, 19, INK, weight=500, lh=1.4)
    # ---- gauge (code prompt, k=3 greedy, server, same client)
    gy = y + 262
    srv = D["server"]
    v0, v2 = srv["greedy"]["vllm"][0], srv["greedy"]["ours"][0]
    sv = cp.get("server_verified") or {}
    gx, gw = M + 4, 560
    s.text(gx, gy, "SERVER DECODE SPEED, CODE PROMPT (tokens/s)", 12.5, DIM, weight=700, ls=1.4)
    scale = gw / 200.0
    rows = [("stock vLLM 0.30", v0, DIM, False, f"{v0:g}"),
            ("qwen-warp", v2, AGENT, not sv, f"{v2:g}  {fx(v2 / v0)}")]
    for i, (lab, v, col, dashed, val) in enumerate(rows):
        yy = gy + 26 + i * 50
        s.text(gx, yy + 4, lab, 14, MUTED, weight=600)
        bx = gx + 130
        bw = v * scale * (gw - 130) / gw
        s.rect(bx, yy - 15, bw, 28, col if not dashed else "none", stroke=col, rx=4, sw=2 if dashed else 0, dash="5 4" if dashed else None, opacity=0.9)
        s.text(bx + bw + 10, yy + 5, val, 16, INK, weight=800)
    s.text(gx, gy + 26 + 2 * 50 + 4, "Greedy, k=3, same client, server-verified (2 visits x 3 reps).", 12.5, DIM, italic=True)
    # ---- tiles
    tx, ty = 690, gy - 14
    tiles = [(fx(X["x_srv_greedy"][0]), "vs stock vLLM", "server, code, greedy k=3", GPU),
             (fx(X["x_plain"]), "plain decode", f"{hd['plain_k9_ms']['v']} vs {hd['vllm_plain_ms']['v']} ms/step", GPU),
             (f"+{min((a / b - 1) * 100 for a, b in zip(cp['port_ab']['m3'], cp['port_ab']['base'])):.1f}\u2013{max((a / b - 1) * 100 for a, b in zip(cp['port_ab']['m3'], cp['port_ab']['base'])):.1f}%", "campaign gain on the engine A/B", f"harness: {fx(cp['gate_verdict'], 3)}; 6 agents, 3.6 h", AGENT),
             (f"{X['pred_miss']}/{X['pred_n']}", "bets missed", "predictions frozen first", PART)]
    for i, (big, l1, l2, col) in enumerate(tiles):
        x = tx + (i % 2) * 238
        yy = ty + (i // 2) * 128
        s.rect(x, yy, 226, 116, PANEL, stroke=LINE, rx=12)
        s.rect(x, yy, 6, 116, col, rx=3)
        s.text(x + 22, yy + 54, big, 40 if len(big) < 8 else 30, col, weight=900)
        s.text(x + 22, yy + 82, l1, 15 if len(l1) <= 22 else 13, INK, weight=700)
        s.text(x + 22, yy + 102, l2, 12.5, MUTED)
    # ---- caveat
    cy = max(gy + 22 + 3 * 44 + 26, ty + 2 * 128 + 12)
    cav = ("Honest caveats: one card, one model, one machine; stock vLLM was run with one configuration (max-model-len 16384, 4 seqs). The campaign gate's \u00d71.107 is a harness number; "
           "the winner, ported into the engine, measured +8.5\u201310.9% on the engine and \u00d71.9\u20132.0 vs vLLM at the server. HyperQwen's DFlash2 is still ahead on code and math in the head-to-head test. 18 of 33 pre-registered bets were wrong.")
    cav = cav.replace("18 of 33", f"{X['pred_miss']} of {X['pred_n']}")
    ln = wrap(cav, 1090, 14)
    h = 24 + len(ln) * 19
    s.rect(M, cy, W - 2 * M, h, "#2a2314", stroke="#6b5521", rx=10)
    yy = cy + 28
    for l in ln:
        s.text(M + 20, yy, l, 14, "#f3dfa8")
        yy += 19
    return cy + h + 34


# =============================================================================================== 1. loop + setup
def loop_and_setup(s, y, D, C, X):
    y = s.section_title(y, "HOW IT WORKS", "A research loop with a human outside it",
                        "The agents ran the loop below. The human set the goal, vetoed, and asked for status; the harness, not the agent, decided what was kept.") + 18
    top = y
    # ------------------------------------------------ loop ring (left)
    cx, cy, R = 330, top + 250, 168
    nodes = [("ideate", f"{X['n_ideas']} ideas queued"),
             ("pre-register", f"{len(D['experiments'])} experiments, {X['pred_n']} bets"),
             ("run", f"{X['gpu_h']:.0f} GPU-h leased"),
             ("record", f"{X['outcomes']['negative'] + X['outcomes']['abandoned']} negatives kept"),
             ("synthesise", f"{X['pred_miss']} of {X['pred_n']} bets missed"),
             ("campaign gate", f"{X['camp_rounds']} gates, {X['camp_kept']} kept"),
             ("merge", f"{fx(D['campaign']['gate_verdict'], 3)} stacked")]
    n = len(nodes)
    pts = []
    for i in range(n):
        a = -math.pi / 2 + 2 * math.pi * i / n
        pts.append((cx + R * math.cos(a), cy + R * math.sin(a)))
    s.circle(cx, cy, R, "none", stroke=LINE, sw=2, dash="3 7")
    for i in range(n):
        a1 = -math.pi / 2 + 2 * math.pi * i / n + 0.24
        a2 = -math.pi / 2 + 2 * math.pi * (i + 1) / n - 0.24
        x1, y1 = cx + R * math.cos(a1), cy + R * math.sin(a1)
        x2, y2 = cx + R * math.cos(a2), cy + R * math.sin(a2)
        col = AGENT if i < 5 else HARN
        s.add(f'<path d="M{x1:.1f},{y1:.1f} A{R},{R} 0 0 1 {x2:.1f},{y2:.1f}" fill="none" stroke="{col}" stroke-width="3" opacity="0.85"/>')
        ah = 8
        ang = a2 + math.pi / 2
        s.add(f'<path d="M{x2:.1f},{y2:.1f} L{x2 - ah * math.cos(ang - 0.5):.1f},{y2 - ah * math.sin(ang - 0.5):.1f} L{x2 - ah * math.cos(ang + 0.5):.1f},{y2 - ah * math.sin(ang + 0.5):.1f} Z" fill="{col}"/>')
    for i, ((px, py), (lab, cnt)) in enumerate(zip(pts, nodes)):
        col = AGENT if i < 5 else HARN
        s.circle(px, py, 25, PANEL, stroke=col, sw=3)
        s.text(px, py + 6, str(i + 1), 18, col, "middle", 800)
        ox = 1 if px > cx + 20 else -1 if px < cx - 20 else 0
        oy = -1 if py < cy - 20 else 1 if py > cy + 20 else 0
        if ox == 0:
            ax, tyy, anc = px, py + (-38 if oy < 0 else 52), "middle"
            ax -= 0
            if oy < 0:
                ax, tyy = px, py - 50
        else:
            ax, anc, tyy = px + ox * 34, ("start" if ox > 0 else "end"), py - 2
        s.text(ax, tyy, lab, 15, INK, anc, 800)
        s.text(ax, tyy + 17, cnt, 12.5, MUTED, anc)
    s.text(cx, cy - 18, "the loop", 22, INK, "middle", 800)
    s.text(cx, cy + 6, "agents run it,", 13.5, MUTED, "middle")
    s.text(cx, cy + 24, "the harness gates it", 13.5, MUTED, "middle")
    # human outside the loop
    hx, hy = 60, top + 22
    icon_human(s, hx, hy, 18, HUMAN)
    s.text(hx + 28, hy - 4, f"{X['h_n']}+ human messages", 14, HUMAN, weight=800)
    s.text(hx + 28, hy + 14, f"{X['h_chars']:,} characters in all", 12.5, MUTED)
    s.arrow(hx + 120, hy + 28, cx - 80, cy - R + 22, HUMAN, 1.6, dash="4 4")
    # ------------------------------------------------ roles (right)
    rx, rw = 662, 498
    ry = top - 4

    def box(yy, h, col, ttl, sub, icon):
        s.rect(rx, yy, rw, h, PANEL, stroke=col, rx=12, sw=1.6)
        icon(s, rx + 34, yy + h / 2, 17, col)
        s.text(rx + 70, yy + 26, ttl, 16, INK, weight=800)
        s.para(rx + 70, yy + 46, sub, rw - 90, 12.8, MUTED, lh=1.32)
    box(ry, 78, HUMAN, "Human: goals, vetoes, 'stop it'", f"{X['h_class']['status']} status asks, {X['h_class']['steer']} steers, {X['h_class']['stop']} explicit stops, 1 idea. Every commit is agent-tagged.", icon_human)
    s.arrow(rx + 34, ry + 80, rx + 34, ry + 98, MUTED)
    box(ry + 100, 78, AGENT, "Orchestrator agent", "plans, spawns work packages, reads pointers instead of trusting summaries, commits with provenance", icon_agent)
    s.arrow(rx + 34, ry + 180, rx + 34, ry + 198, MUTED)
    # workers
    s.rect(rx, ry + 200, rw, 104, PANEL, stroke=AGENT, rx=12, sw=1.6)
    s.text(rx + 20, ry + 224, f"{X['n_packages']} work packages + {X['camp_sessions']} campaign lineage sessions, in parallel", 14, INK, weight=800)
    chips = [("study", "8 repos"), ("kernels", "11 builds"), ("experiments", "pre-registered"), ("campaign", f"{X['camp_sessions']} lineages")]
    cw = (rw - 40 - 3 * 10) / 4
    for i, (a, b) in enumerate(chips):
        x = rx + 20 + i * (cw + 10)
        s.rect(x, ry + 238, cw, 52, PANEL2, stroke=LINE, rx=8)
        s.circle(x + 12, ry + 258, 3.5, AGENT)
        s.text(x + 22, ry + 262, a, 13, INK, weight=700)
        s.text(x + 12, ry + 279, b, 11.5, MUTED)
    s.arrow(rx + 34, ry + 306, rx + 34, ry + 324, MUTED)
    # lease
    s.rect(rx, ry + 326, rw, 78, PANEL, stroke=GPU, rx=12, sw=1.6)
    icon_gpu(s, rx + 34, ry + 359, 17, GPU)
    icon_lock(s, rx + 74, ry + 356, 11, GPU)
    s.text(rx + 100, ry + 352, "ONE RTX 3090, one lease", 16, INK, weight=800)
    s.para(rx + 100, ry + 372, f"a mutex around every timed run; held {X['gpu_h']:.0f} h of {X['gpu_span_h']:.0f} h. Agents queue; clocks and power cap are shared state.", rw - 120, 12.8, MUTED, lh=1.32)
    s.arrow(rx + 34, ry + 406, rx + 34, ry + 424, MUTED)
    # harness
    s.rect(rx, ry + 426, rw, 92, PANEL, stroke=HARN, rx=12, sw=1.6)
    icon_gate(s, rx + 34, ry + 472, 17, HARN)
    s.text(rx + 70, ry + 452, "Harness: keep / revert", 16, INK, weight=800)
    s.para(rx + 70, ry + 472, "paired pristine controls, a hidden held-out workload, a plain-decode guard and a teacher-forced correctness check. Agents cannot edit it.", rw - 90, 12.8, MUTED, lh=1.32)
    # rules
    ry2 = ry + 540
    s.text(rx, ry2 + 4, "RULES THAT EARNED THEIR PLACE", 12.5, DIM, weight=700, ls=1.6)
    rw2 = (rw - 10) / 2
    for i, r in enumerate(D["rules"]):
        x = rx + (i % 2) * (rw2 + 10)
        yy = ry2 + 16 + (i // 2) * 70
        s.rect(x, yy, rw2, 62, PANEL2, stroke=LINE, rx=8)
        s.text(x + 12, yy + 22, r["t"], 13.5, HARN, weight=800)
        s.para(x + 12, yy + 38, r["d"], rw2 - 20, 11.8, MUTED, lh=1.25)
        s.text(60, cy + R + 52, "purple arrows: agent work.  teal arrows: harness-decided.", 12.5, MUTED)
    s.text(60, cy + R + 70, "every hand-off carries a pointer: a commit, a log or a file:line.", 12.5, MUTED)
    return max(cy + R + 80, ry2 + 16 + 2 * 70) + 20


# =============================================================================================== 2. stats strip
def stats_strip(s, y, D, C, X):
    o = X["outcomes"]
    tiles = [
        (f"{X['wall_h']:.0f} h", "wall-clock", f"{fmt_dt(X['t_first'], '%b %d')} \u2192 {fmt_dt(X['t_last'], '%b %d')}"),
        (f"{X['n_commits']}", "commits", "program paths only"),
        (f"{len(D['experiments'])}", "experiments", f"{o['win']} win \u00b7 {o['partial']} partial \u00b7 {o['negative']} neg \u00b7 {o['abandoned']} aband."),
        (f"{X['pred_n']}", "pre-registered bets", f"{X['pred_hit']} hit \u00b7 {X['pred_mixed']} mixed \u00b7 {X['pred_miss']} miss"),
        (f"{X['cu_lines']:,}", "CUDA lines", f"{X['cu_files']} .cu files at HEAD"),
        (f"{X['test_only']}", "test files", "engine/tests, vs oracle"),
        (f"{X['n_packages']}", "agent work packages", "+ 6 campaign lineages"),
        (f"{X['gpu_h']:.0f} h", "GPU leased", f"{X['gpu_h'] / X['gpu_span_h'] * 100:.0f}% of the window"),
    ]
    w = (W - 2 * M - 7 * 8) / 8
    for i, (big, l1, l2) in enumerate(tiles):
        x = M + i * (w + 8)
        s.rect(x, y, w, 96, PANEL, stroke=LINE, rx=10)
        s.text(x + w / 2, y + 38, big, 25, INK, "middle", 900)
        s.text(x + w / 2, y + 58, l1, 12.5, MUTED, "middle", 700)
        for j, ln in enumerate(wrap(l2, w - 12, 10.8)[:2]):
            s.text(x + w / 2, y + 74 + j * 13, ln, 10.8, DIM, "middle")
    return y + 96 + 40


# =============================================================================================== 3. campaign (hero)
def lerp(c1, c2, t):
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{int(a[i] + (b[i] - a[i]) * max(0, min(1, t))):02x}" for i in range(3))


def halo(s, x, y, txt, size, fill, anchor="middle", weight=600):
    s.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{SANS}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" '
          f'stroke="{BG}" stroke-width="4" paint-order="stroke" stroke-linejoin="round">{esc(txt)}</text>')


def campaign_section(s, y, D, C, X):
    rounds = {r["id"]: r for r in C["rounds"]}
    cp = D["campaign"]
    s.text(M, y, "THE POWER OF AUTORESEARCH", 13, AGENT, weight=700, ls=2.4)
    s.text(M, y + 36, f"{X['camp_rounds']} gated rounds, {X['camp_kept']} kept, {fx(cp['gate_verdict'], 3)} in {X['camp_wall_h']:.1f} hours of agent time", 29, INK, weight=900)
    y += 58
    y = s.para(M, y + 8, ("Six agents each got one focus and a private copy of the engine. They profiled, wrote kernels and tested them; a harness they could not touch decided what was kept. "
                          "Winners were merged by predicting the stacked gain first."), W - 2 * M, 15, MUTED)
    y += 6
    # headline chips
    chips = [(f"{X['camp_rounds']}", "gated rounds", "12.6 GPU-min each", HARN),
             (f"{X['camp_kept']} / {X['camp_rejected']}", "kept / rejected", "6 lineages + 3 merges", WIN),
             (f"{X['camp_tries']}", "agent try runs", f"{C.get('tries_tagged_agent', 0)} tagged agent, 0 human", AGENT),
             (f"{min(X['camp_session_min'].values()):.0f}\u2013{max(X['camp_session_min'].values()):.0f} min", "per lineage session", f"{X['camp_sessions']} sessions", AGENT),
             (fx(cp["gate_verdict"], 3), "vs pristine, dev", f"held-out {fx(cp['held_out'], 3)} \u00b7 plain {fx(cp['plain_guard'], 3)}", WIN)]
    cw = (W - 2 * M - 4 * 10) / 5
    for i, (big, l1, l2, col) in enumerate(chips):
        x = M + i * (cw + 10)
        s.rect(x, y, cw, 82, PANEL, stroke=LINE, rx=10)
        s.rect(x, y, cw, 4, col, rx=2)
        s.text(x + 16, y + 40, big, 26, col, weight=900)
        s.text(x + 16, y + 60, l1, 13, INK, weight=700)
        s.text(x + 16, y + 76, l2, 11.5, MUTED)
    y += 82 + 34
    # ---------------------------------------------------------------- state tree
    top = y
    tree = {n["id"]: n for n in C["tree"]}
    col_x = {"base": 96, "w1": 318, "m1": 540, "w2": 770, "m2": 962, "m3": 1100}
    rowA, rowB, rowC = top + 84, top + 214, top + 344
    pos = {"stock": (col_x["base"], rowB),
           "gemv:1": (col_x["w1"], rowA), "verify:1": (col_x["w1"], rowB), "smallk:1": (col_x["w1"], rowC),
           "m1:1": (col_x["m1"], (rowA + rowB) / 2),
           "gemv2:1": (col_x["w2"], rowA), "dnet2:1": (col_x["w2"], rowB), "cycle2:1": (col_x["w2"], rowC),
           "m2:1": (col_x["m2"], (rowA + rowB) / 2), "m3:1": (col_x["m3"], rowB + 40)}
    for t, x in (("BASELINE", col_x["base"]), ("WAVE 1: three agents, from stock", col_x["w1"]), ("MERGE", col_x["m1"]), ("WAVE 2: three agents, from merge 1", col_x["w2"]), ("MERGE", col_x["m2"]), ("MERGE", col_x["m3"])):
        s.text(x, top + 12, t, 11.5, DIM, "middle", 700, ls=1.2)
    s.text(M, top + 30, "node size and colour = speedup vs the pristine control; edge text = what the agent changed", 12.5, MUTED, italic=True)

    def rad(sp):
        return 15 + (sp - 1) * 250

    def edge(a, b, label=None, lab_dy=-10, lab_dx=0, col=LINE, dash=None):
        (x1, y1), (x2, y2) = pos[a], pos[b]
        r1 = rad(tree[a]["speedup"]) if a in tree else 13
        r2 = rad(tree[b]["speedup"])
        sx, ex = x1 + r1 + 2, x2 - r2 - 4
        mx = (sx + ex) / 2
        s.path(f"M{sx:.1f},{y1:.1f} C{mx:.1f},{y1:.1f} {mx:.1f},{y2:.1f} {ex:.1f},{y2:.1f}", stroke=col, sw=2.2, dash=dash, opacity=0.9)
        s.add(f'<path d="M{ex + 3:.1f},{y2:.1f} L{ex - 7:.1f},{y2 - 5:.1f} L{ex - 7:.1f},{y2 + 5:.1f} Z" fill="{col}" opacity="0.9"/>')
        if label:
            lx, ly = (sx + ex) / 2 + lab_dx, (y1 + y2) / 2 + lab_dy
            for k, ln in enumerate(wrap(label, 150, 12)):
                halo(s, lx, ly + k * 14, ln, 12, MUTED)
    labels = D["tree_labels"]
    edge("stock", "gemv:1", labels["gemv:1"], -8, -2)
    edge("stock", "verify:1", labels["verify:1"], -9, 0)
    edge("stock", "smallk:1", labels["smallk:1"], 14, 0)
    edge("gemv:1", "m1:1", None)
    edge("verify:1", "m1:1", None)
    edge("m1:1", "gemv2:1", labels["gemv2:1"], -9, 6)
    edge("m1:1", "dnet2:1", labels["dnet2:1"], 16, 6)
    edge("m1:1", "cycle2:1", labels["cycle2:1"], 20, -34)
    edge("gemv2:1", "m2:1", None)
    edge("dnet2:1", "m2:1", None)
    edge("m2:1", "m3:1", None)
    edge("cycle2:1", "m3:1", None)
    s.rect(col_x["m1"] - 100, (rowA + rowB) / 2 - 62, 200, 18, "none")
    # nodes
    s.circle(pos["stock"][0], pos["stock"][1], 15, PANEL2, stroke=DIM, sw=2)
    s.text(pos["stock"][0], pos["stock"][1] + 38, "engine before campaign", 13, INK, "middle", 700)
    s.text(pos["stock"][0], pos["stock"][1] + 55, f"{C['baseline']['dev']:.1f} tok/s", 11.5, MUTED, "middle")
    s.text(pos["stock"][0], pos["stock"][1] + 70, "6-prompt mix", 11.5, MUTED, "middle")
    gate_gpu = {g["lineage"]: g["gpu_s"] / 60 for g in C["gates"]}
    for nid, (px, py) in pos.items():
        if nid == "stock":
            continue
        r = rounds[nid]
        sp = r["speedup_vs_baseline"]
        rr = rad(sp)
        col = lerp("#6f86a8", WIN, (sp - 1.0) / 0.107)
        s.circle(px, py, rr, col, stroke=INK, sw=1.5, opacity=0.95)
        # gate badge
        icon_gate(s, px + rr * 0.75, py - rr * 0.75, 8, HARN, BG)
        name = nid.split(":")[0]
        merge = name.startswith("m") and name[1:].isdigit()
        s.text(px, py + rr + 18, f"{D['lineage_names'][name]}  {fx(sp, 3)}", 14.5, INK, "middle", 800)
        if merge:
            pred = 1 + r["pred_gain_pct"] / 100
            s.text(px, py + rr + 34, labels[nid], 12, HARN, "middle", 700)
            s.text(px, py + rr + 49, f"predicted {fx(pred, 3)}", 11.8, MUTED, "middle")
        else:
            par = tree[nid]["parent"]
            stp = f"step {fx(r['step_vs_previous_kept'], 3)} over {D['lineage_names'][par.split(':')[0]]} \u00b7 " if par != "stock" else ""
            s.text(px, py + rr + 34, f"{stp}{r['session_s'] / 60:.0f} min \u00b7 {C['tries_by_lineage'].get(name, 0)} tries", 11.8, MUTED, "middle")
    # legend
    ly = rowC + 86
    icon_gate(s, M + 8, ly - 4, 8, HARN, BG)
    gmin = min(min(r["held_out_ratio"] or 9, r["plain_guard_ratio"]) for r in C["rounds"])
    s.text(M + 24, ly, f"gate = paired pristine controls + hidden held-out + plain guard + teacher-forced check, 12.6 GPU-min; held-out and plain guard never dropped below {gmin:.3f}", 12.2, MUTED)
    y = ly + 22
    s.rect(M, y, W - 2 * M, 46, PANEL, stroke=LINE, rx=8)
    s.para(M + 14, y + 19, "Why zero rejections? The agents' quick tries filtered ideas first (their lab notebooks keep the rejected ones: a register-rotation variant -2%, a side-stream split measured slower, PDL impossible on sm_86). "
           "The gate's rejection power was proven on deliberately broken variants instead, below.", W - 2 * M - 28, 12.6, MUTED, lh=1.32)
    # port verification strip (K12)
    y += 46 + 12
    ab, sv = cp["port_ab"], D["server"]
    s.rect(M, y, W - 2 * M, 112, PANEL, stroke=WIN, rx=10, sw=1.4)
    s.text(M + 16, y + 24, "THEN THE WINNER WAS PORTED INTO THE REAL ENGINE AND MEASURED THROUGH THE SERVER", 12, WIN, weight=800, ls=1.2)
    cw = (W - 2 * M - 32) / 4
    for i, lab in enumerate(ab["labels"]):
        x = M + 16 + i * cw
        s.text(x, y + 52, f"{ab['base'][i]:g} \u2192 {ab['m3'][i]:g}", 17, INK, weight=800)
        s.text(x, y + 69, f"{lab}: engine {fx(ab['m3'][i] / ab['base'][i], 3)}", 11.8, MUTED)
        s.text(x, y + 88, f"server: {sv['greedy']['ours'][i]:g} vs vLLM {sv['greedy']['vllm'][i]:g}", 12.5, WIN, weight=800)
        s.text(x, y + 104, f"{fx(X['x_srv_greedy'][i], 2)} (sampled {fx(X['x_srv_t1'][i], 2)})", 11.8, MUTED)
    return y + 112 + 36


def found_and_harness(s, y, D, C, X):
    s.text(M, y, "AGENTS FOUND THINGS THE HUMAN DID NOT ASK FOR", 13, AGENT, weight=700, ls=2)
    y += 20
    cw = (W - 2 * M - 3 * 12) / 4
    hmax = 0
    cards = []
    for it in D["found"]:
        tl = wrap(it["title"], cw - 28, 14.5, bold=True)
        tx = wrap(it["text"], cw - 28, 12.2)
        cards.append((it, tl, tx))
        hmax = max(hmax, 20 + len(tl) * 18 + len(tx) * 16 + 46)
    for i, (it, tl, tx) in enumerate(cards):
        x = M + i * (cw + 12)
        s.rect(x, y, cw, hmax, PANEL, stroke=AGENT, rx=12, sw=1.2)
        s.text(x + 14, y + 20, it["who"].upper(), 10.8, AGENT, weight=800, ls=1.4)
        yy = y + 40
        for l in tl:
            s.text(x + 14, yy, l, 14.5, INK, weight=800)
            yy += 18
        yy += 2
        for l in tx:
            s.text(x + 14, yy, l, 12.2, MUTED)
            yy += 16
        s.text(x + 14, y + hmax - 12, it["result"], 12.5, WIN, weight=800)
    y += hmax + 30
    # harness
    s.text(M, y, "THE HARNESS IS THE STAR: WHAT AUTONOMY NEEDS TO BE SAFE", 13, HARN, weight=700, ls=2)
    y += 20
    n = len(D["harness"])
    cw = (W - 2 * M - (n - 1) * 10) / n
    cards, hmax = [], 0
    for it in D["harness"]:
        tl = wrap(it["title"], cw - 24, 14, bold=True)
        tx = wrap(it["text"], cw - 24, 11.8)
        cards.append((it, tl, tx))
        hmax = max(hmax, 18 + len(tl) * 17 + len(tx) * 15 + 34)
    for i, (it, tl, tx) in enumerate(cards):
        x = M + i * (cw + 10)
        s.rect(x, y, cw, hmax, PANEL, stroke=HARN, rx=12, sw=1.2)
        icon_gate(s, x + 20, y + 22, 9, HARN)
        yy = y + 46
        for l in tl:
            s.text(x + 12, yy, l, 14, INK, weight=800)
            yy += 17
        yy += 3
        for l in tx:
            s.text(x + 12, yy, l, 11.8, MUTED)
            yy += 15
    return y + hmax + 40


def axis_x(X, left, width):
    t0, t1 = X["win0"], X["win1"]
    return lambda t: left + (t - t0) / (t1 - t0) * width


def human_vs_agent(s, y, D, C, X):
    s.text(M, y, "HUMAN MINUTES VS AGENT HOURS", 13, HUMAN, weight=700, ls=2)
    y += 14
    left, width = M + 130, W - 2 * M - 130
    ax = axis_x(X, left, width)
    rows = [("human messages", HUMAN), ("GPU leased", GPU), ("commits", AGENT)]
    rh = 34
    for i, (lab, col) in enumerate(rows):
        yy = y + 8 + i * rh
        s.text(M, yy + 20, lab, 13, col, weight=700)
        s.rect(left, yy + 2, width, rh - 8, PANEL, stroke=None, rx=4)
    # human ticks
    yy = y + 8
    for t, c in X["h_times"]:
        if X["win0"] <= t <= X["win1"]:
            col = NEG if c == "stop" else HUMAN
            s.rect(ax(t) - 1.2, yy + 4, 2.4, rh - 12, col, opacity=0.95)
    yy = y + 8 + rh
    for a, b in X["gpu_union"]:
        a, b = max(a, X["win0"]), min(b, X["win1"])
        if b > a:
            s.rect(ax(a), yy + 4, max(1.2, ax(b) - ax(a)), rh - 12, GPU, opacity=0.9)
    yy = y + 8 + 2 * rh
    for c in X["commits"]:
        t = datetime.fromisoformat(c["iso"]).timestamp()
        s.rect(ax(t) - 0.8, yy + 4, 1.6, rh - 12, AGENT, opacity=0.9)
    # day ticks
    day = datetime.fromtimestamp(X["win0"]).astimezone().replace(hour=0, minute=0, second=0, microsecond=0)
    for k in range(0, 4):
        t = day.timestamp() + k * 86400
        if X["win0"] <= t <= X["win1"]:
            s.line(ax(t), y + 4, ax(t), y + 8 + 3 * rh, DIM, 1, dash="2 4")
            s.text(ax(t) + 4, y + 2, fmt_dt(t, "%b %d 00:00"), 11, DIM)
    y += 8 + 3 * rh + 16
    msg = (f"{X['h_n']} messages (\u2265, unique prompts only), {X['h_chars']:,} characters over {(max(t for t, c in X['h_times']) - min(t for t, c in X['h_times'])) / 3600:.0f} h: "
           f"{X['h_class']['status']} status asks, {X['h_class']['steer']} steers, {X['h_class']['stop']} stops (red), {X['h_class']['idea']} idea. "
           f"Agents: {X['n_packages']} work packages, {X['n_commits']} commits, {X['camp_tries']} campaign tries, {X['gpu_h']:.0f} GPU-hours leased.")
    y = s.para(M, y + 4, msg, W - 2 * M, 13.5, MUTED)
    return y + 36


# =============================================================================================== 4. timeline
LANES = ["Study", "Baselines", "Kernels", "Speculation", "Campaign", "Comparison"]


def timeline(s, y, D, C, X):
    y = s.section_title(y, "WHAT HAPPENED, IN ORDER", f"{X['wall_h']:.0f} hours, six lanes, {len(D['milestones'])} milestones",
                        "Each dot is a commit (or a campaign gate), coloured by verdict. Bottom strip: the one GPU, leased almost continuously.", color=PART) + 14
    left, width = M + 112, W - 2 * M - 112
    ax = axis_x(X, left, width)
    lane_h = 118
    top = y + 26
    day = datetime.fromtimestamp(X["win0"]).astimezone().replace(hour=0, minute=0, second=0, microsecond=0)
    for k in range(0, 4):
        t0 = day.timestamp() + k * 86400
        for hh in range(0, 24, 6):
            t = t0 + hh * 3600
            if X["win0"] <= t <= X["win1"]:
                strong = hh == 0
                s.line(ax(t), top - 6, ax(t), top + lane_h * 6 + 52, DIM if strong else LINE, 1, dash=None if strong else "2 5", opacity=0.9 if strong else 0.6)
                s.text(ax(t) + 4, top - 10, fmt_dt(t, "%b %d") if strong else f"{hh:02d}h", 11.5, MUTED if strong else DIM, weight=700 if strong else 400)
    lane_col = {"Study": MUTED, "Baselines": PART, "Kernels": GPU, "Speculation": AGENT, "Campaign": HARN, "Comparison": HUMAN}
    items = []
    for m in D["milestones"]:
        if "commit" in m:
            t = X["ctime"](m["commit"])
        elif "campaign" in m:
            t = next(r["t"] for r in C["rounds"] if r["id"] == m["campaign"])
        else:
            t = datetime.fromisoformat(m["at"]).timestamp()
        items.append(dict(m, t=t))
    for li, lane in enumerate(LANES):
        ly = top + li * lane_h
        s.rect(M, ly + 2, W - 2 * M, lane_h - 4, PANEL if li % 2 == 0 else "#121823", rx=8)
        s.rect(M, ly + 2, 5, lane_h - 4, lane_col[lane], rx=2)
        s.text(M + 16, ly + lane_h / 2 + 5, lane.upper(), 12.5, lane_col[lane], weight=800, ls=1.3)
        cy = ly + lane_h / 2
        s.line(left, cy, left + width, cy, LINE, 2)
        # label levels: up 0..2 / down 0..2. Items are processed left to right; the first label of a cluster takes the highest level
        # (long leader), later ones step down, so leaders never cross later labels. Collision-checked.
        placed = []  # (side, level, x0, x1, x)
        lane_items = sorted([i for i in items if i["lane"] == lane], key=lambda i: i["t"])
        xs_l = [ax(i["t"]) for i in lane_items]
        wid = [tw(i["label"], 12) + 4 if i.get("label") else 0 for i in lane_items]
        last_x, flip, nside = -99, 1, 0
        for idx, it in enumerate(lane_items):
            x = xs_l[idx]
            col = VERDICT[it["v"]]
            dy = 0
            if x - last_x < 13:
                dy = 9 * flip
                flip = -flip
            last_x = x
            lab = it.get("label")
            if lab:
                w = wid[idx]
                anchor = "end" if x + w > left + width + 30 else "start"
                x0, x1 = (x - w + 6, x + 6) if anchor == "end" else (x - 6, x - 6 + w)
                k = sum(1 for j in range(idx + 1, len(lane_items)) if wid[j] and xs_l[j] < x1 + 8)
                target = min(2, k)

                def valid(side, lv):
                    for (s2, l2, a2, b2, px) in placed:
                        if s2 != side:
                            continue
                        if l2 == lv and not (x1 + 6 < a2 or x0 - 6 > b2):
                            return False
                        if l2 > lv and x0 - 4 < px < x1 + 4:
                            return False
                        if l2 < lv and a2 - 4 < x < b2 + 4:
                            return False
                    return True
                cands = []
                for side in (("up", "down") if nside % 2 == 0 else ("down", "up")):
                    for lv in range(target, -1, -1):
                        cands.append((side, lv))
                for side in ("up", "down"):
                    for lv in range(2, target, -1):
                        cands.append((side, lv))
                chosen = next((c for c in cands if valid(*c)), None)
                side, lv = chosen or ("up", 0)
                nside += 1
                placed.append((side, lv, x0, x1, x))
                ty = cy - 16 - lv * 14 if side == "up" else cy + 26 + lv * 14
                if lv >= 1:
                    s.line(x, cy + (-7 if side == "up" else 7) + dy, x, ty + (3 if side == "up" else -10), col, 1, opacity=0.7)
                tx = x - 6 if anchor == "start" else x + 6
                s.text(tx, ty, lab, 12, INK, anchor, 500)
            if it["v"] == "abandoned":
                s.circle(x, cy + dy, 7, BG, stroke=col, sw=2.4, dash="3 2")
            else:
                s.circle(x, cy + dy, 7, col, stroke=BG, sw=2)
    gy = top + lane_h * 6 + 8
    s.text(M + 16, gy + 22, "GPU LEASE", 12.5, GPU, weight=800, ls=1.3)
    s.rect(left, gy + 4, width, 26, PANEL, rx=4)
    kc = {"experiment": "#2dd4bf", "build": GPU, "campaign": AGENT}
    for k in ("experiment", "build", "campaign"):
        for a, b, kind, tag in X["gpu_spans"]:
            if kind == k:
                a2, b2 = max(a, X["win0"]), min(b, X["win1"])
                if b2 > a2:
                    s.rect(ax(a2), gy + 7, max(1.0, ax(b2) - ax(a2)), 20, kc[k], opacity=0.92)
    ly = gy + 52
    xx = left
    for lab, col in (("win", WIN), ("partial", PART), ("negative", NEG), ("abandoned", ABAN)):
        s.circle(xx + 6, ly - 4, 6, col if lab != "abandoned" else BG, stroke=col if lab == "abandoned" else BG, sw=2)
        s.text(xx + 18, ly, lab, 12.5, MUTED)
        xx += tw(lab, 12.5) + 44
    xx += 20
    for lab, col in (("lease: experiments", kc["experiment"]), ("lease: kernel builds", kc["build"]), ("lease: campaign", kc["campaign"])):
        s.rect(xx, ly - 11, 14, 11, col, rx=2)
        s.text(xx + 20, ly, lab, 12.5, MUTED)
        xx += tw(lab, 12.5) + 50
    s.text(M, ly + 22, f"times: commit timestamps, campaign events, lease log (UTC+7); GPU busy = lease held {X['gpu_h']:.0f} h of {X['gpu_span_h']:.0f} h. Sources: data.json milestones, lease_snapshot.json", 12, DIM, italic=True)
    return ly + 22 + 44


# =============================================================================================== 5. progression
def bars_panel(s, x, y, w, h, labels, vals, colors, ymax, unit, title, sub=None, fmt="{:.1f}", ymin=0, hl=None, tag=None, small=False):
    s.text(x, y, title, 13.5, INK, weight=800)
    if sub:
        s.text(x, y + 16, sub, 11.5, DIM)
    top = y + (32 if sub else 20)
    ph = h - (top - y) - (50 if not small else 36)
    n = len(vals)
    gap = 10
    bw = (w - gap * (n - 1)) / n
    for i, (l, v, c) in enumerate(zip(labels, vals, colors)):
        bh = max(2, (v - ymin) / (ymax - ymin) * ph)
        bx = x + i * (bw + gap)
        s.rect(bx, top + ph - bh, bw, bh, c, rx=3, opacity=0.95)
        s.text(bx + bw / 2, top + ph - bh - 6, fmt.format(v), 12.5 if not small else 12, INK, "middle", 800)
        for k, ln in enumerate(wrap(l, bw + gap - 2, 11.5 if not small else 11.5)[:2 if small else 3]):
            s.text(bx + bw / 2, top + ph + 16 + k * 13, ln, 11.5, MUTED, "middle")
    s.line(x, top + ph, x + w, top + ph, LINE, 1.5)
    return top + ph


def progression(s, y, D, C, X):
    sr = D["series"]
    y = s.section_title(y, "PERFORMANCE PROGRESSION", "From a tie with vLLM to \u00d71.9 at the server",
                        "Decode tok/s, greedy, MTP k=3 chain, same weights and card. Each bar is a measured build; the tag is the experiment that justified it.", color=GPU) + 18
    steps = sr["steps"]
    col_for = {"vllm": "#6c7a93", "k4": "#4a90e2", "k5": "#4da3ff", "k9": "#5fb6ff", "k11": "#7cc8ff", "k12": AGENT}
    x0, w0 = M, 640
    top = y + 20
    ph = 250
    n = len(steps)
    gap = 12
    bw = (w0 - gap * (n - 1)) / n
    ymax = 200.0
    s.text(x0, y + 12, "CODE PROMPT, tok/s", 12.5, DIM, weight=700, ls=1.4)
    vals = [st["vals"][0] for st in steps]
    for i, st in enumerate(steps):
        v = vals[i]
        bh = v / ymax * ph
        bx = x0 + i * (bw + gap)
        s.rect(bx, top + 20 + ph - bh, bw, bh, col_for[st["id"]], rx=3)
        s.text(bx + bw / 2, top + 20 + ph - bh - 7, f"{v:g}", 15, INK, "middle", 900)
        if i > 0:
            s.text(bx + bw / 2, top + 20 + ph - bh - 24, f"{(v / vals[i - 1] - 1) * 100:+.1f}%" if st["id"] != "k12" else f"{fx(v / vals[i - 1], 2)} vs previous build", 11.5, WIN if v > vals[i - 1] else MUTED, "middle", 700)
        s.text(bx + bw / 2, top + 20 + ph + 17, st["short"], 12, INK, "middle", 700)
        s.text(bx + bw / 2, top + 20 + ph + 31, "ref " + st["exp"], 12, DIM, "middle")
    s.line(x0, top + 20 + ph, x0 + w0, top + 20 + ph, LINE, 1.5)
    yy = top + 20 + ph + 52
    yy = s.para(x0, yy, "First end-to-end engine: 46.9 tok/s plain, a tie with vLLM's 46.8. Speculation, a leaner GEMV, a quantised head and a lookup tail took it from there.", w0, 12.5, MUTED)
    # sampling + copy-heavy cards under the chart
    sc = sr["sampling_code"]
    yy += 8
    s.rect(x0, yy, w0, 96, PANEL, stroke=LINE, rx=10)
    s.text(x0 + 16, yy + 24, "EXACT SAMPLING (T=1, top-p .95, top-k 20), code prompt", 11.5, DIM, weight=700, ls=1.2)
    s.text(x0 + 16, yy + 56, f"vLLM {sc['vllm_t1']:g}  \u2192  {sc['k7']:g} (exact sampling)  \u2192  {sc['k9']:g} (int4 head) tok/s", 15.5, INK, weight=800)
    s.text(x0 + 16, yy + 78, f"Through the server, final build: {D['server']['t1']['ours'][0]:g} vs vLLM {D['server']['t1']['vllm'][0]:g} tok/s = {fx(D['server']['t1']['ours'][0] / D['server']['t1']['vllm'][0], 2)}", 13, WIN, weight=700)
    yy += 108
    s.rect(x0, yy, w0, 76, PANEL, stroke=LINE, rx=10)
    s.text(x0 + 16, yy + 24, "COPY-HEAVY WORK (MTP chain + lookup tail, T\u22648), tok/s before \u2192 after", 11.5, DIM, weight=700, ls=1.2)
    for i, c in enumerate(sr["copy_heavy"]):
        x = x0 + 16 + i * 205
        s.text(x, yy + 50, f"{c['base']:g} \u2192 {c['tail']:g}", 16, INK, weight=800)
        s.text(x, yy + 67, f"{c['label']} {fx(c['tail'] / c['base'], 2)}", 11.5, MUTED)
    yend = yy + 76
    # small multiples right
    rx, rw = 720, W - M - 720
    pl = sr["plain"]
    ys = y + 12
    bars_panel(s, rx, ys, rw, 160, [p["label"].replace("a|b on a parallel branch", "a|b branch").replace("first e2e engine", "first e2e") for p in pl],
               [p["ms"] for p in pl], ["#6c7a93", "#4a90e2", "#4a90e2", "#4da3ff", "#5fb6ff", AGENT], 22.5, "ms", "Plain decode, ms/step (lower is better; axis from 16 ms)",
               fmt="{:.2f}", ymin=16.0, small=True)
    ys += 178
    for ci, cname in ((1, "prose"), (2, "math")):
        bars_panel(s, rx, ys, rw, 160, [st["short"] for st in steps], [st["vals"][ci] for st in steps], [col_for[st["id"]] for st in steps], 125 if ci == 1 else 190,
                   "tok/s", f"{cname.capitalize()} prompt, tok/s", fmt="{:g}", small=True)
        ys += 178
    yb = max(yend, ys - 18)
    s.text(M, yb + 24, "Last bar: campaign winner measured directly on the engine (its own before/after base was 168.3). Bars come from different runs. Refs: BENCH.md sections named under each bar.", 12, DIM, italic=True)
    return yb + 24 + 40


# =============================================================================================== 6. where the speed came from
def waterfall(s, y, D, C, X):
    hd = D["headline"]
    y = s.section_title(y, "WHERE THE SPEED CAME FROM", "A bandwidth-bound problem with a hard ceiling",
                        "One token reads 15.1 GB of weights. At 936 GB/s that is 16.1 ms (61.9 tok/s); the best read-only kernel measured 891 GB/s (16.93 ms). Everything below is the distance to that floor.", color=GPU) + 10
    pl = D["series"]["plain"]
    x0, w0 = M, 520
    top, ph = y + 36, 270
    lo, hi = 15.5, 22.0
    Y = lambda v: top + (hi - v) / (hi - lo) * ph
    s.text(x0, y + 14, "PLAIN DECODE, ms PER STEP", 12.5, DIM, weight=700, ls=1.4)
    ms = [p["ms"] for p in pl]
    seq = [("stock vLLM", ms[0], None, "baseline"), ("first engine + parallel branch", ms[2], ms[2] - ms[0], "K3"), ("GEMV rewrite", ms[3], ms[3] - ms[2], "K5"),
           ("int4 head + rescoring", ms[4], ms[4] - ms[3], "K9"), ("campaign winner", ms[5], ms[5] - ms[4], "K12")]
    n = len(seq)
    gap = 12
    bw = (w0 - gap * (n - 1)) / n
    prev = None
    for i, (lab, v, d, tag) in enumerate(seq):
        bx = x0 + i * (bw + gap)
        if d is None:
            s.rect(bx, Y(v), bw, Y(lo) - Y(v), "#6c7a93", rx=3)
            s.text(bx + bw / 2, Y(v) - 8, f"{v:.1f}", 15, INK, "middle", 900)
        else:
            camp = False
            s.rect(bx, Y(prev), bw, Y(v) - Y(prev), WIN if not camp else AGENT, rx=3, opacity=0.9 if not camp else 0.5)
            if camp:
                s.rect(bx, Y(prev), bw, Y(v) - Y(prev), "none", stroke=AGENT, sw=2, dash="5 3", rx=3)
            s.text(bx + bw / 2, Y(prev) - 8, f"{d:+.2f} ms", 13, INK, "middle", 800)
            s.text(bx + bw / 2, (Y(prev) + Y(v)) / 2 + 4, f"{v:.2f}", 11.5, BG if not camp else INK, "middle", 800)
            s.line(bx - gap, Y(prev), bx, Y(prev), DIM, 1, dash="2 3")
        for k, ln in enumerate(wrap(lab, bw + 8, 11.5)[:2]):
            s.text(bx + bw / 2, Y(lo) + 17 + k * 13, ln, 11.5, INK, "middle", 600)
        s.text(bx + bw / 2, Y(lo) + 45, tag, 11.5, MUTED, "middle")
        prev = v
    s.line(x0, Y(lo), x0 + w0, Y(lo), LINE, 1.5)
    for v, col, l1, l2 in ((16.13, NEG, "copy roofline", "16.13 ms"), (hd["read_only_ms"]["v"], PART, "read-only kernel", f"{hd['read_only_ms']['v']} ms")):
        s.line(x0, Y(v), x0 + w0 + 8, Y(v), col, 1.6, dash="6 4")
    s.text(x0 + w0 + 14, Y(hd["read_only_ms"]["v"]) - 2, "read-only kernel", 11.8, PART, "start", 700)
    s.text(x0 + w0 + 14, Y(hd["read_only_ms"]["v"]) + 12, f"{hd['read_only_ms']['v']} ms, 891 GB/s", 11.5, PART)
    s.text(x0 + w0 + 14, Y(16.13) + 12, "copy roofline", 11.8, NEG, "start", 700)
    s.text(x0 + w0 + 14, Y(16.13) + 26, f"16.13 ms = {hd['roofline_toks']['v']} tok/s", 11.5, NEG)
    s.text(x0, Y(lo) + 68, "axis starts at 15.5 ms. The last two bars come from different runs (in its own run the winner went 18.133 \u2192 17.227 ms).", 11.5, DIM, italic=True)
    rx, rw = 730, W - M - 730
    s.text(rx, y + 14, "SPECULATION MULTIPLIES THE PLAIN RATE (tok/s)", 12.5, DIM, weight=700, ls=1.2)
    k9 = D["series"]["steps"][5]["vals"][0]
    plain = hd["plain_k9_toks"]["v"]
    hy = D["campaign"]["port_ab"]["hybrid"]
    ladder = [("plain decode, final build", plain, "K12", GPU, None),
              ("+ MTP chain k=3 with int4 draft head", k9, "K12", GPU, f"{fx(k9 / plain, 2)}  (3.50 tokens per cycle)"),
              ("code-edit work: + lookup tail", hy["a"][0], "K12", AGENT, f"{fx(hy['a'][0] / hy['a'][1], 2)} over {hy['a'][1]:g} without the tail"),
              ("stock vLLM server, MTP k=3 (code prompt)", hd["vllm_k3_code"]["v"], "K12", "#6c7a93", f"the qwen-warp server ({hd['engine_k3_code']['v']:g}) is {fx(hd['engine_k3_code']['v'] / hd['vllm_k3_code']['v'], 2)} faster")]
    ly = y + 38
    maxv = 300.0
    for i, (lab, v, tag, col, note) in enumerate(ladder):
        yy = ly + i * 70
        s.text(rx, yy + 12, lab, 13, INK, weight=700)
        s.text(rx + rw, yy + 12, tag, 11.5, MUTED, "end")
        s.rect(rx, yy + 20, v / maxv * (rw - 50), 22, col, rx=4, opacity=0.92)
        s.text(rx + v / maxv * (rw - 50) + 8, yy + 37, f"{v:g}", 14, INK, weight=900)
        if note:
            s.text(rx, yy + 58, note, 12, WIN, weight=800)
    s.para(rx, ly + 4 * 70 + 4, "Verifying 4 rows costs only ~7% more than 1 row (the weights are read once), so a chain that lands 3.5 tokens per cycle multiplies the plain rate by ~3. The int4 draft head costs 1.0% acceptance; the 32k-vocab shortcut would have cost 6%.", rw, 12.3, MUTED)
    return y + 36 + ph + 96


# =============================================================================================== 7. predictions vs reality
def predictions(s, y, D, C, X):
    pm = {p["id"]: p for p in D["predictions"]}
    y = s.section_title(y, "PREDICTIONS VS REALITY", f"{X['pred_n']} numbers frozen before the data; {X['pred_miss']} were wrong",
                        f"Hit {X['pred_hit']}, mixed {X['pred_mixed']}, missed {X['pred_miss']}: {X['pred_miss_for']} of the misses in our favour, {X['pred_miss_against']} against us. Predicting wrong is cheap; not writing it down is expensive.", color=PART) + 16
    lx, px0, pw = M, 400, 460
    s.text(px0 + pw / 2, y + 6, "measured / predicted (log scale)", 12, DIM, "middle", 700)
    s.text(px0, y + 6, "\u2190 lower", 11.5, DIM)
    s.text(px0 + pw, y + 6, "higher \u2192", 11.5, DIM, "end")
    top = y + 22
    rh = 44
    xs = lambda r: px0 + pw / 2 + math.log2(r) * 100
    clip = lambda v: max(px0 + 6, min(px0 + pw - 6, v))
    s.line(px0 + pw / 2, top - 4, px0 + pw / 2, top + rh * len(D["dumbbells"]) + 2, MUTED, 1.4)
    for i, d in enumerate(D["dumbbells"]):
        p = pm[d["pid"]]
        yy = top + i * rh
        col = WIN if d["better"] == "for" else NEG
        r = d["meas"] / d["pred"]
        x = clip(xs(r))
        if i % 2 == 0:
            s.rect(M, yy, W - 2 * M, rh - 2, "#121823", rx=6)
        s.text(lx + 10, yy + 20, d["label"], 14, INK, weight=700)
        s.text(lx + 10, yy + 37, d["note"], 11.5, MUTED)
        s.line(px0 + pw / 2, yy + 22, x, yy + 22, col, 5)
        s.circle(px0 + pw / 2, yy + 22, 5, BG, stroke=MUTED, sw=2)
        s.circle(x, yy + 22, 8, col, stroke=BG, sw=2)
        u = d["unit"]
        if u.startswith("%"):
            val = f"{d['pred']:g}% \u2192 {d['meas']:g}% loss"
        elif u in ("x", "x time", "gain"):
            val = f"\u00d7{d['pred']:g} \u2192 \u00d7{d['meas']:g}" + (" time" if u == "x time" else "")
        else:
            val = f"{d['pred']:g} \u2192 {d['meas']:g} {u}"
        s.text(px0 + pw + 22, yy + 20, val, 14, INK, weight=800)
        s.text(px0 + pw + 22, yy + 37, f"\u00d7{r:.2f} of the prediction, {'in our favour' if d['better'] == 'for' else 'against us'}", 11.8, col, weight=600)
    y = top + rh * len(D["dumbbells"]) + 14
    # grid of all bets
    s.text(M, y + 8, f"ALL {X['pred_n']} BETS, IN ORDER OF RESOLUTION", 12.5, DIM, weight=700, ls=1.4)
    cw = 28
    gx = M
    gy = y + 20
    for i, p in enumerate(D["predictions"]):
        if p["verdict"] == "hit":
            c = WIN
        elif p["verdict"] == "mixed":
            c = PART
        else:
            c = GPU if p["direction"] == "for" else NEG
        s.rect(gx + i * (cw + 4), gy, cw, cw, c, rx=5, opacity=0.95)
        s.text(gx + i * (cw + 4) + cw / 2, gy + cw / 2 + 4, str(i + 1), 11, BG, "middle", 800)
    lx2 = gx + len(D["predictions"]) * (cw + 4) + 14
    leg = [("hit", WIN), ("mixed", PART), ("missed, better for us", GPU), ("missed, worse for us", NEG)]
    ly = gy + 11
    xx = M
    for lab, col in leg:
        s.rect(xx, gy + cw + 14, 12, 12, col, rx=3)
        s.text(xx + 18, gy + cw + 25, lab, 12.5, MUTED)
        xx += tw(lab, 12.5) + 48
    s.text(M, gy + cw + 52, "Source: dashboard/gen.py PREDICTIONS at HEAD (pre-registered numbers vs experiment README answers); full text in history.html.", 12, DIM, italic=True)
    return gy + cw + 52 + 42


# =============================================================================================== 8. what failed
def failures(s, y, D, C, X):
    y = s.section_title(y, "WHAT FAILED", "The negative results are the most useful part of the repo",
                        f"{X['outcomes']['negative']} experiments ended negative, {X['outcomes']['abandoned']} abandoned, {X['outcomes']['partial']} partial, and every one stayed on the record.", color=NEG) + 16
    fl = D["failures"]
    cols = 3
    cw = (W - 2 * M - (cols - 1) * 14) / cols
    rh = 128
    for i, f in enumerate(fl):
        x = M + (i % cols) * (cw + 14)
        yy = y + (i // cols) * (rh + 12)
        s.rect(x, yy, cw, rh, PANEL, stroke="#5a2a30", rx=12)
        s.rect(x, yy, 5, rh, NEG, rx=2)
        s.text(x + 18, yy + 22, f["exp"], 11.5, NEG, weight=800, ls=1.2)
        s.text(x + 18, yy + 42, f["title"], 14.5, INK, weight=800)
        s.text(x + cw - 14, yy + 30, f["number"], 17, NEG, "end", 900)
        yy2 = yy + 62
        for ln in wrap(f["text"], cw - 34, 12.2)[:4]:
            s.text(x + 18, yy2, ln, 12.2, MUTED)
            yy2 += 15.5
    rows = (len(fl) + cols - 1) // cols
    return y + rows * (rh + 12) + 30


# =============================================================================================== 9. comparison
def comparison(s, y, D, C, X):
    cmpd = D["comparison"]
    y = s.section_title(y, "THE HONEST COMPARISON", "HyperQwen: same weights, same HTTP client, same card",
                        "Before the head-to-head test we predicted we would win 3 of 4 prompts; the build before the campaign won one. The strongest open competitor is ahead where speculation has the most to copy.", color=HUMAN) + 14
    x0, w0 = M, 700
    top, ph = y + 62, 240
    cats = cmpd["labels"]
    sysc = [GPU, "#8d99ae", HUMAN, AGENT]
    gw = w0 / len(cats)
    bw = (gw - 30) / 4
    ymax = 280.0
    for ci, cat in enumerate(cats):
        for si, row in enumerate(cmpd["rows"]):
            v = row["vals"][ci]
            bx = x0 + ci * gw + 10 + si * (bw + 4)
            bh = v / ymax * ph
            if row.get("dashed"):
                s.rect(bx, top + ph - bh, bw, bh, "none", stroke=sysc[si], sw=2, dash="4 3", rx=3)
            else:
                s.rect(bx, top + ph - bh, bw, bh, sysc[si], rx=3)
            s.text(bx + bw / 2, top + ph - bh - 6, f"{v:.0f}", 12.3, INK, "middle", 800)
        s.text(x0 + ci * gw + gw / 2, top + ph + 20, cat, 13.5, INK, "middle", 700)
    s.line(x0, top + ph, x0 + w0, top + ph, LINE, 1.5)
    for si, row in enumerate(cmpd["rows"]):
        lx = x0 + (si % 2) * 350
        ly2 = top - 40 + (si // 2) * 20
        if row.get("dashed"):
            s.rect(lx, ly2 - 11, 12, 12, "none", stroke=sysc[si], sw=2, dash="3 2", rx=2)
        else:
            s.rect(lx, ly2 - 11, 12, 12, sysc[si], rx=2)
        s.text(lx + 18, ly2, row["label"], 12.5, MUTED)
    s.text(x0, top - 2, "decode tokens/s", 12, DIM)
    s.para(x0, top + ph + 42, "Greedy decoding through the HTTP server, 385 tokens, two interleaved rounds per system. The dashed bar was measured on another day, not interleaved with HyperQwen.", w0, 12, DIM)
    # right side
    us, hq, df, usm3 = cmpd["rows"]
    pct = lambda a, b: (a / b - 1) * 100
    ahead = [f"DFlash2 on code {fx(df['vals'][0] / us['vals'][0])} and math {fx(df['vals'][2] / us['vals'][2])} over us ({df['tokens_per_step'][0]:g} / {df['tokens_per_step'][2]:g} tokens per step vs our {us['tokens_per_step'][0]:g} / {us['tokens_per_step'][2]:g})",
             f"HyperQwen MTP over our pre-campaign build on code, math and 8K by {min(pct(hq['vals'][i], us['vals'][i]) for i in (0, 2, 3)):.0f}-{max(pct(hq['vals'][i], us['vals'][i]) for i in (0, 2, 3)):.0f}%; the final build is above it (measured in a separate run)"]
    tt = cmpd["ttft"]
    behind = [f"prose decode: +{pct(us['vals'][1], hq['vals'][1]):.0f}% over HyperQwen MTP, +{pct(us['vals'][1], df['vals'][1]):.0f}% over DFlash2",
              f"TTFT: {tt['us_20tok_ms']} ms vs {tt['hq_20tok_ms']} ms (20 tokens), {tt['us_8k_s']} s vs {tt['hq_8k_s']} s (8K)",
              "determinism: bit-reproducible run to run; speculative greedy equals plain greedy except documented rounding near-ties"]
    rx, rw = 780, W - M - 780
    ty = y + 14
    s.rect(rx, ty, rw, 116, PANEL, stroke=HUMAN, rx=10)
    s.text(rx + 14, ty + 22, "WHERE HYPERQWEN IS AHEAD", 11.8, HUMAN, weight=800, ls=1.4)
    yy = ty + 42
    for t in ahead:
        for ln in wrap("\u2022 " + t, rw - 28, 12.3):
            s.text(rx + 14, yy, ln, 12.3, INK)
            yy += 15.5
    ty2 = ty + 116 + 12
    s.rect(rx, ty2, rw, 150, PANEL, stroke=GPU, rx=10)
    s.text(rx + 14, ty2 + 22, "WHERE WE ARE AHEAD", 11.8, GPU, weight=800, ls=1.4)
    yy = ty2 + 42
    for t in behind:
        for ln in wrap("\u2022 " + t, rw - 28, 12.3):
            s.text(rx + 14, yy, ln, 12.3, INK)
            yy += 15.5
    td = cmpd["rows"][2]["tokens_per_step"]
    tu = cmpd["rows"][0]["tokens_per_step"]
    s.para(rx, ty2 + 168, f"Same ~22 ms step; DFlash2 lands {td[0]:g} tokens per step on code vs our {tu[0]:g}. We have not ported a block drafter.", rw, 12.3, MUTED)
    return top + ph + 84


# =============================================================================================== 10. code along
def code_along(s, y, D, C, X):
    y = s.section_title(y, "CODE ALONG", "Four real snippets, verbatim, with pointers",
                        "Copied from the committed engine (commit 9602811; the first one from the campaign lineage). Common indent removed; line numbers are the source file's.", color=WIN) + 16
    shown = [x for x in X["snips"] if x["id"] != "dn_snap"]
    for sn in shown:
        lines = sn["text"]
        ind = min((len(l) - len(l.lstrip()) for l in lines if l.strip() and not l.lstrip().startswith("#")), default=0)
        lines = [l[ind:] if l.startswith(" " * ind) else l for l in lines]
        a = sn["lines"][0]
        maxlen = max(len(l.expandtabs(2)) for l in lines)
        fs = min(12.5, (W - 2 * M - 110) / (maxlen * 0.61))
        lh = fs * 1.42
        h = 62 + len(lines) * lh + 14
        s.rect(M, y, W - 2 * M, h, PANEL, stroke=LINE, rx=12)
        s.text(M + 18, y + 26, sn["title"], 15, INK, weight=800)
        s.text(M + 18, y + 46, sn["ptr"], 11.8, GPU, mono=True)
        s.text(W - M - 18, y + 46, sn["why"], 12, MUTED, "end", italic=True)
        s.rect(M + 12, y + 56, W - 2 * M - 24, len(lines) * lh + 12, "#0f141d", rx=8)
        yy = y + 56 + 6 + fs
        for i, ln in enumerate(lines):
            s.text(M + 24, yy, str(a + i), fs, DIM, "start", mono=True)
            s.tspans(M + 24 + 4.6 * fs, yy, highlight(ln.expandtabs(2)), size=fs)
            yy += lh
        y += h + 16
    return y + 14


# =============================================================================================== 11. footer
def footer(s, y, D, C, X):
    s.rect(M, y, W - 2 * M, 176, PANEL, stroke=LINE, rx=12)
    verified = "server-verified (BENCH.md K12: server vs stock vllm serve, same client, median of 2 visits x 3 reps)" if X["camp_verified"] else "campaign harness only (port + server-level verification pending)"
    s.text(M + 20, y + 30, f"STATUS AS OF commit {X['head']}  \u00b7  {fmt_dt(ts_of(X['head_iso']), '%Y-%m-%d %H:%M')} (UTC+7)  \u00b7  poster generated {X['generated'][:16].replace('T', ' ')}", 13, AGENT, weight=800, ls=0.6)
    lines = [
        f"Campaign winner (ported): {verified}; the gate's x1.107 is the harness number. Everything else is from committed READMEs, BENCH.md and git history; every number has a pointer in data.json 'sources'.",
        "One RTX 3090 (24 GB), Swift-1.5-Qwen3.8-27B-INT4 + MTP head, vLLM 0.30 as the stock baseline. Run-to-run noise \u00b10.1% on paired A/B, \u00b13% on single-task greedy streams. Known open test: test_headq e2e parity fails on the base build too (BENCH K12).",
        "HyperQwen DFlash2 is ahead on code and math; we have not ported a block drafter. 8K-context and concurrency comparisons were not completed.",
        "Reproduce: python3 gen.py  \u00b7  Update numbers: edit data.json, re-run gen.py  \u00b7  Code: MIT.  github.com/tuan3w/qwen-warp",
    ]
    yy = y + 56
    for t in lines:
        yy = s.para(M + 20, yy, t, W - 2 * M - 40, 12.8, MUTED, lh=1.35) + 4
    return y + 176 + 20


def ts_of(iso):
    return datetime.fromisoformat(iso).timestamp()


# =============================================================================================== build
def build(D, C, X):
    s = Svg()
    y = 0
    body = Svg()
    y = title_block(body, 6, D, C, X)
    y = loop_and_setup(body, y + 10, D, C, X)
    y = stats_strip(body, y, D, C, X)
    y = campaign_section(body, y + 10, D, C, X)
    y = found_and_harness(body, y, D, C, X)
    y = human_vs_agent(body, y, D, C, X)
    y = timeline(body, y + 10, D, C, X)
    y = progression(body, y + 10, D, C, X)
    y = waterfall(body, y + 10, D, C, X)
    y = predictions(body, y + 10, D, C, X)
    y = failures(body, y + 10, D, C, X)
    y = comparison(body, y + 10, D, C, X)
    y = code_along(body, y + 10, D, C, X)
    y = footer(body, y, D, C, X)
    H = int(y + 10)
    s.add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="t d">')
    s.add(f'<title id="t">qwen-warp: can autonomous AI research beat vLLM on one RTX 3090?</title>')
    s.add('<desc id="d">Infographic: an experiment in autonomous research. A human directed AI agents that built a Qwen serving stack faster than vLLM on one RTX 3090. '
          'Shows the research loop, the autonomous optimisation campaign, a timeline, performance progression, predictions versus reality, failures, a comparison with HyperQwen and real code snippets.</desc>')
    s.add(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
    s.add(f'<rect x="0" y="0" width="{W}" height="6" fill="{AGENT}"/>')
    s.p.extend(body.p)
    s.add("</svg>")
    return "\n".join(s.p)
