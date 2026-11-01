"""Builds history.html: one self-contained, interactive experiment-history page (no external requests)."""
import json
from datetime import datetime

LANES = ["Study", "Baselines", "Kernels", "Speculation", "Campaign", "Comparison"]


def iso(t):
    return datetime.fromtimestamp(t).astimezone().strftime("%Y-%m-%d %H:%M")


def build(D, C, X):
    ctime = X["ctime"]
    pred_by_dir = {}
    for p in D["predictions"]:
        pred_by_dir.setdefault(p["exp"], []).append({k: p[k] for k in ("id", "claim", "predicted", "measured", "verdict_text", "verdict", "direction")})
    cards = []
    for e in X["exps"]:
        lease = X["gpu_tag_h"].get(e["tag"], 0.0)
        cards.append({
            "kind": "experiment", "id": e["eid"], "anchor": "x-" + e["eid"], "title": e["name"], "lane": e["lane"], "outcome": e["outcome"], "label": e["label"],
            "t": e["t_post"], "t_pre": e["t_pre"], "when": iso(e["t_post"]), "path": f"experiments/{e['dir']}/",
            "question": e["question"], "prediction": e["prediction"], "answer": e["answer"], "dead": e["dead"], "conclusion": e["conclusion"],
            "confidence": e["confidence"], "wall_h": (e["t_post"] - e["t_pre"]) / 3600, "lease_h": lease, "preds": pred_by_dir.get(e["dir"], []), "reproduce": e.get("reproduce", ""),
            "pre_commit": e["pre"], "post_commit": e["post"],
        })
    for b in D["builds"]:
        t = ctime(b["commit"])
        cards.append({"kind": "build", "id": b["id"], "anchor": "b-" + b["id"], "title": b["title"], "lane": b["lane"], "outcome": b["outcome"], "label": b["pkg"],
                      "t": t, "when": iso(t), "result": b["result"], "lease_h": X["gpu_tag_h"].get(b["tag"], 0.0) if b["tag"] else None, "commit": b["commit"]})
    notes = C.get("lineage_notes", {})
    tree = {n["id"]: n for n in C["tree"]}
    for r in C["rounds"]:
        name = r["lineage"]
        merge = name[0] == "m" and name[1:].isdigit()
        gate_min = next((g["gpu_s"] for g in C["gates"] if g["lineage"] == name), 0) / 60
        cards.append({"kind": "campaign", "id": r["id"], "anchor": "c-" + r["id"].replace(":", "-"), "title": D["tree_labels"][r["id"]], "lane": "Campaign", "outcome": "win",
                      "label": f"{'merge' if merge else 'lineage'} {name}: x{r['speedup_vs_baseline']:.3f} vs pristine", "t": r["t"], "when": iso(r["t"]),
                      "headline": r["headline"], "hypothesis": r["hypothesis"], "change": r["change"], "files": r["files_changed"],
                      "speedup": r["speedup_vs_baseline"], "step": r["step_vs_previous_kept"], "parent": tree[r["id"]]["parent"], "parents2": tree[r["id"]]["parents2"],
                      "held": r["held_out_ratio"], "plain": r["plain_guard_ratio"], "pred": 1 + r["pred_gain_pct"] / 100, "session_min": r["session_s"] / 60,
                      "tries": C["tries_by_lineage"].get(name, 0), "gate_min": gate_min, "notes": notes.get(name, ""), "merge": merge, "check": r["check_ok"]})
    cards.sort(key=lambda c: c["t"])
    # timeline dots
    dots = []
    for m in D["milestones"]:
        if "commit" in m:
            t = ctime(m["commit"])
        elif "campaign" in m:
            t = next(r["t"] for r in C["rounds"] if r["id"] == m["campaign"])
        else:
            t = datetime.fromisoformat(m["at"]).timestamp()
        label = m.get("label") or ""
        if "campaign" in m and not label:
            label = D["tree_labels"][m["campaign"]]
        target = next((c["anchor"] for c in cards if (m.get("campaign") and c["kind"] == "campaign" and c["id"] == m["campaign"]) or
                       (m.get("commit") and c["kind"] != "campaign" and c.get("post_commit", c.get("commit")) == m["commit"])), "")
        dots.append({"lane": m["lane"], "t": t, "v": m["v"], "label": label, "target": target})
    snips = [{"id": s["id"], "tab": s["tab"], "title": s["title"], "ptr": s["ptr"], "why": s["why"], "lines": s["text"], "start": s["lines"][0], "lang": s["lang"]} for s in X["snips"]]
    hp = [(t, c) for t, c in X["h_times"]]
    data = {
        "meta": {**D["meta"], "head": X["head"], "head_iso": X["head_iso"], "generated": X["generated"]},
        "stats": {"wall_h": X["wall_h"], "commits": X["n_commits"], "exps": len(D["experiments"]), "outcomes": X["outcomes"], "bets": X["pred_n"], "hit": X["pred_hit"], "mixed": X["pred_mixed"],
                  "miss": X["pred_miss"], "cu_lines": X["cu_lines"], "gpu_h": X["gpu_h"], "packages": X["n_packages"], "gates": X["camp_rounds"], "kept": X["camp_kept"],
                  "tries": X["camp_tries"], "human_msgs": X["h_n"]},
        "cards": cards, "dots": dots, "lanes": LANES, "win0": X["win0"], "win1": X["win1"],
        "gpu": [[a, b] for a, b in X["gpu_union"]], "human": [[t, c] for t, c in hp], "commits": [datetime.fromisoformat(c["iso"]).timestamp() for c in X["commits"]],
        "predictions": D["predictions"], "snips": snips, "rules": D["rules"], "campaign": D["campaign"],
        "tree_note": "campaign numbers: campaign harness; engine port + server-level verification pending" if not X["camp_verified"] else "server-verified",
    }
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return TEMPLATE.replace("__DATA__", blob)


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>qwen-warp: experiment history</title>
<style>
:root{--bg:#0d1117;--panel:#151b26;--panel2:#1c2433;--line:#2e394e;--ink:#e8edf5;--muted:#a3b0c6;--dim:#7586a3;--win:#3ecf8e;--partial:#f5b83d;--negative:#ff6b6b;--abandoned:#8d99ae;--human:#ff9f5a;--agent:#b794f6;--gpu:#4da3ff;--harn:#2dd4bf;--code:#0f141d}
@media (prefers-color-scheme: light){:root:not([data-theme=dark]){--bg:#f6f8fb;--panel:#ffffff;--panel2:#eef2f8;--line:#d4dbe7;--ink:#17202e;--muted:#4a5870;--dim:#6b7a92;--win:#0f9d63;--partial:#b7791f;--negative:#d23b3b;--abandoned:#6b7a92;--human:#d9731a;--agent:#7c4dd6;--gpu:#1a73d9;--harn:#0e9f8e;--code:#f1f4f9}}
:root[data-theme=light]{--bg:#f6f8fb;--panel:#ffffff;--panel2:#eef2f8;--line:#d4dbe7;--ink:#17202e;--muted:#4a5870;--dim:#6b7a92;--win:#0f9d63;--partial:#b7791f;--negative:#d23b3b;--abandoned:#6b7a92;--human:#d9731a;--agent:#7c4dd6;--gpu:#1a73d9;--harn:#0e9f8e;--code:#f1f4f9}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 Inter,-apple-system,"Segoe UI",Helvetica,Arial,"DejaVu Sans",sans-serif}
a{color:var(--gpu)}code,pre,.mono{font-family:ui-monospace,"SF Mono",Menlo,Consolas,"DejaVu Sans Mono",monospace}
.wrap{max-width:1180px;margin:0 auto;padding:0 20px 80px}
header{padding:36px 0 10px}.kick{color:var(--agent);font-weight:700;letter-spacing:.16em;font-size:12px;text-transform:uppercase}
h1{font-size:40px;line-height:1.1;margin:6px 0 8px;font-weight:900}h2{font-size:24px;margin:44px 0 6px;font-weight:800}h3{margin:0;font-size:16px}
.lead{color:var(--muted);max-width:860px}.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.chips{display:grid;grid-template-columns:repeat(auto-fit,minmax(138px,1fr));gap:10px;margin:18px 0}
.chip{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px 14px}.chip b{display:block;font-size:24px;font-weight:900}.chip span{color:var(--muted);font-size:12px}
.btn{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:999px;padding:6px 12px;font:inherit;font-size:13px;cursor:pointer}.btn[aria-pressed=true]{border-color:var(--ink);background:var(--panel2);font-weight:700}
.btn .d{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;vertical-align:baseline}
input[type=search],select{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:7px 10px;font:inherit;font-size:13px}
.bar{position:sticky;top:0;z-index:5;background:color-mix(in srgb,var(--bg) 92%,transparent);backdrop-filter:blur(6px);padding:10px 0;border-bottom:1px solid var(--line);margin-top:20px}
.tl{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:12px 12px 6px;overflow-x:auto}
.tl svg{display:block;min-width:900px;width:100%;height:auto}
.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--muted);font-size:13px;margin:8px 2px}.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
.day{margin:26px 0 8px;color:var(--dim);font-weight:700;letter-spacing:.1em;font-size:12px;text-transform:uppercase}
.card{background:var(--panel);border:1px solid var(--line);border-left:5px solid var(--abandoned);border-radius:12px;margin:10px 0;scroll-margin-top:96px}
.card.win{border-left-color:var(--win)}.card.partial{border-left-color:var(--partial)}.card.negative{border-left-color:var(--negative)}.card.abandoned{border-left-color:var(--abandoned)}
.card.hide{display:none}.card.flash{box-shadow:0 0 0 2px var(--gpu)}
.card>summary{list-style:none;cursor:pointer;padding:12px 16px;display:grid;grid-template-columns:auto 1fr auto;gap:2px 14px;align-items:center}.card>summary::-webkit-details-marker{display:none}
.tag{font-size:11px;font-weight:800;letter-spacing:.08em;padding:2px 8px;border-radius:999px;border:1px solid var(--line);color:var(--muted);text-transform:uppercase}
.verd{font-size:12px;font-weight:800;text-transform:uppercase;letter-spacing:.08em}.verd.win{color:var(--win)}.verd.partial{color:var(--partial)}.verd.negative{color:var(--negative)}.verd.abandoned{color:var(--abandoned)}
.sub{color:var(--muted);font-size:13px;grid-column:2/4}.when{color:var(--dim);font-size:12px;white-space:nowrap}
.body{padding:2px 18px 16px;border-top:1px solid var(--line)}.body h4{margin:16px 0 4px;font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:var(--dim)}
.body p,.body .txt{margin:0;white-space:pre-wrap;color:var(--ink);font-size:14px}.txt.scroll{max-height:240px;overflow:auto;background:var(--panel2);border-radius:8px;padding:10px 12px}
.facts{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}.facts span{background:var(--panel2);border-radius:8px;padding:4px 10px;font-size:12px;color:var(--muted)}.facts b{color:var(--ink)}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}th{color:var(--dim);font-size:11px;letter-spacing:.08em;text-transform:uppercase}
td.v{font-weight:800;white-space:nowrap}.v.hit{color:var(--win)}.v.miss{color:var(--negative)}.v.mixed{color:var(--partial)}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin:10px 0}.code{background:var(--code);border:1px solid var(--line);border-radius:10px;padding:12px 14px;overflow:auto;font-size:13px;line-height:1.5}
.code .ln{color:var(--dim);user-select:none;display:inline-block;width:3.2em}.c-com{color:var(--dim)}.c-kw{color:#e0607e}.c-num{color:var(--partial)}.c-fn{color:var(--gpu)}.c-str{color:var(--win)}
.ptr{color:var(--gpu);font-size:12px}.note{color:var(--muted);font-size:13px}
footer{margin-top:60px;color:var(--muted);font-size:13px;border-top:1px solid var(--line);padding-top:16px}
.empty{color:var(--dim);padding:20px;text-align:center}
@media print{.bar{position:static}}
</style></head><body><div class="wrap">
<header>
<div class="kick">qwen-warp &middot; experiment history</div>
<h1>Can autonomous AI research beat vLLM on one RTX 3090?</h1>
<p class="lead">Every experiment, engine build and campaign round of the program, in the order it happened: the question, the prediction frozen before the data, the result, the dead ends. Filter by outcome, expand a card, copy the code.
<a href="infographic.svg">The poster</a> tells the story in one picture; <a href="data.json">data.json</a> holds every number with its pointer.</p>
<div class="row"><button class="btn" id="theme">Toggle light / dark</button><span class="note" id="asof"></span></div>
</header>
<div class="chips" id="chips"></div>
<h2>Timeline</h2>
<div class="tl"><svg id="tl" viewBox="0 0 1160 520" role="img" aria-label="Timeline of milestones by lane"></svg></div>
<div class="legend"><span><i style="background:var(--win)"></i>win</span><span><i style="background:var(--partial)"></i>partial</span><span><i style="background:var(--negative)"></i>negative</span><span><i style="background:var(--abandoned)"></i>abandoned</span><span class="note" id="hover" style="min-width:260px"></span><span class="note">Click a dot to jump to its card. Dots dim when filtered out. Blue = GPU lease held, orange = human messages (red = "stop it"), purple = commits.</span></div>
<div class="bar"><div class="row">
<span id="f-out"></span><span id="f-kind"></span>
<select id="f-lane"><option value="">all lanes</option></select>
<input type="search" id="q" placeholder="search text, e.g. rounding, lop3, PDL" size="30">
<button class="btn" id="expand">Expand all</button><button class="btn" id="collapse">Collapse all</button><span class="note" id="count"></span></div></div>
<div id="cards"></div>
<h2>Predictions vs reality</h2>
<p class="lead" id="pred-lead"></p>
<div class="row" id="f-pred" style="margin:10px 0"></div>
<table id="pred"><thead><tr><th>#</th><th>Claim</th><th>Predicted (frozen)</th><th>Measured</th><th>Verdict</th></tr></thead><tbody></tbody></table>
<h2>Code along</h2>
<p class="lead">Verbatim snippets with pointers; every one is real engine code. Tabs switch between them.</p>
<div class="tabs" id="tabs"></div><div id="snip"></div>
<h2>Rules that earned their place</h2>
<div class="chips" id="rules"></div>
<footer id="foot"></footer>
</div>
<script id="data" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);
const $=s=>document.querySelector(s), el=(t,c,h)=>{const e=document.createElement(t);if(c)e.className=c;if(h!==undefined)e.innerHTML=h;return e};
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const COL={win:'var(--win)',partial:'var(--partial)',negative:'var(--negative)',abandoned:'var(--abandoned)'};
const KIND={experiment:'experiment',build:'engine build',campaign:'campaign'};
const st={out:new Set(),kind:new Set(),lane:'',q:''};
// ---- header
$('#asof').textContent='Status as of commit '+D.meta.head+' ('+D.meta.head_iso.slice(0,16).replace('T',' ')+', UTC+7). '+D.tree_note+'.';
$('#theme').onclick=()=>{const r=document.documentElement;const dark=getComputedStyle(r).getPropertyValue('--bg').trim()==='#0d1117';r.dataset.theme=dark?'light':'dark'};
const s=D.stats;
[[Math.round(s.wall_h)+' h','wall-clock'],[s.commits,'commits'],[s.exps,'experiments'],[s.outcomes.win+' / '+s.outcomes.partial+' / '+s.outcomes.negative+' / '+s.outcomes.abandoned,'win / partial / neg / aband.'],
[s.bets,'pre-registered bets'],[s.miss+' missed','hit '+s.hit+', mixed '+s.mixed],[s.cu_lines.toLocaleString(),'CUDA lines'],[Math.round(s.gpu_h)+' h','GPU leased'],
[s.gates+' / '+s.kept,'campaign gates / kept'],[s.tries,'agent try runs'],[s.human_msgs+'+','human messages']].forEach(([a,b])=>$('#chips').append(el('div','chip','<b>'+esc(a)+'</b><span>'+esc(b)+'</span>')));
// ---- filters
function pill(host,key,items,multi){items.forEach(([v,label,color,n])=>{const b=el('button','btn',(color?'<i class="d" style="background:'+color+'"></i>':'')+esc(label)+(n!==undefined?' <span class="note">'+n+'</span>':''));b.setAttribute('aria-pressed','false');b.onclick=()=>{if(st[key].has(v)){st[key].delete(v)}else{st[key].add(v)}b.setAttribute('aria-pressed',st[key].has(v));render()};host.append(b)})}
const cnt=(f)=>D.cards.filter(f).length;
pill($('#f-out'),'out',['win','partial','negative','abandoned'].map(o=>[o,o,COL[o],cnt(c=>c.outcome===o)]));
pill($('#f-kind'),'kind',Object.keys(KIND).map(k=>[k,KIND[k],null,cnt(c=>c.kind===k)]));
D.lanes.forEach(l=>{const o=el('option');o.value=l;o.textContent=l;$('#f-lane').append(o)});
$('#f-lane').onchange=e=>{st.lane=e.target.value;render()};$('#q').oninput=e=>{st.q=e.target.value.toLowerCase();render()};
function match(c){if(st.out.size&&!st.out.has(c.outcome))return false;if(st.kind.size&&!st.kind.has(c.kind))return false;if(st.lane&&c.lane!==st.lane)return false;
 if(st.q){const hay=(c.title+' '+c.id+' '+c.label+' '+(c.question||'')+(c.prediction||'')+(c.answer||'')+(c.dead||'')+(c.conclusion||'')+(c.hypothesis||'')+(c.change||'')+(c.notes||'')+(c.result||'')+(c.headline||'')).toLowerCase();if(!hay.includes(st.q))return false}return true}
// ---- cards
const fmtP=x=>x.toFixed(3);
function body(c){let h='';const sec=(t,x,scroll)=>x?'<h4>'+t+'</h4><div class="txt'+(scroll?' scroll':'')+'">'+esc(x)+'</div>':'';
 if(c.kind==='experiment'){
  h+=sec('Question',c.question)+sec('Pre-registered prediction (committed before the data, commit '+c.pre_commit+')',c.prediction,true)+sec('Result',c.answer)+sec('Conclusion',c.conclusion,true)+sec('Dead ends',c.dead,true);
  if(c.preds.length){h+='<h4>Numbers it was held to</h4><table><thead><tr><th>Claim</th><th>Predicted</th><th>Measured</th><th></th></tr></thead><tbody>'+c.preds.map(p=>'<tr><td>'+esc(p.claim)+'</td><td>'+esc(p.predicted)+'</td><td>'+esc(p.measured)+'</td><td class="v '+p.verdict+'">'+esc(p.verdict_text)+'</td></tr>').join('')+'</tbody></table>'}
  h+='<div class="facts"><span>confidence <b>'+esc(c.confidence||'n/a')+'</b></span><span>pre-registration to result <b>'+c.wall_h.toFixed(1)+' h</b> wall-clock</span><span>GPU lease held <b>'+c.lease_h.toFixed(2)+' h</b></span><span>folder <b class="mono">'+esc(c.path)+'</b></span>'+(c.reproduce?'<span>reproduce <b class="mono">'+esc(c.reproduce.slice(0,110))+'</b></span>':'')+'</div>';
 }else if(c.kind==='build'){
  h+=sec('What was built',c.title)+sec('Outcome',c.result)+'<div class="facts"><span>work package <b class="mono">'+esc(c.label)+'</b></span><span>commit <b class="mono">'+esc(c.commit)+'</b></span>'+(c.lease_h!=null?'<span>GPU lease held <b>'+c.lease_h.toFixed(2)+' h</b></span>':'')+'</div>';
 }else{
  h+=sec('Agent hypothesis (measured by the agent before the gate)',c.hypothesis)+sec('What it changed',c.change)+
   '<div class="facts"><span>vs pristine <b>x'+fmtP(c.speedup)+'</b></span><span>step over '+esc(c.parent)+(c.parents2.length?' + '+esc(c.parents2.join(', ')):'')+' <b>x'+fmtP(c.step)+'</b></span>'+(c.merge?'<span>predicted before the gate <b>x'+fmtP(c.pred)+'</b></span>':'<span>agent predicted <b>+'+((c.pred-1)*100).toFixed(1)+'%</b></span>')+
   '<span>held-out <b>x'+(c.held?fmtP(c.held):'n/a')+'</b></span><span>plain guard <b>x'+fmtP(c.plain)+'</b></span><span>correctness check <b>'+(c.check?'passed':'failed')+'</b></span>'+(c.merge?'':'<span>agent session <b>'+c.session_min.toFixed(0)+' min</b>, '+c.tries+' quick tries</span>')+'<span>gate <b>'+c.gate_min.toFixed(1)+' GPU-min</b></span><span>diff <b>'+esc(c.files)+'</b></span></div>'+
   (c.notes?'<h4>Agent lab notebook (excerpt)</h4><div class="txt scroll mono" style="font-size:12px">'+esc(c.notes)+'</div>':'')+'<p class="note" style="margin-top:10px">Numbers: campaign harness (paired pristine controls); engine port and server-level verification pending.</p>';
 }return h}
let cardEls=[];
function buildCards(){const host=$('#cards');let lastDay='';D.cards.forEach(c=>{const day=c.when.slice(0,10);if(day!==lastDay){lastDay=day;host.append(el('div','day',day))}
 const d=el('details','card '+c.outcome);d.id=c.anchor;d._c=c;
 d.innerHTML='<summary><span class="tag">'+esc(KIND[c.kind])+' '+esc(c.id)+'</span><div><h3>'+esc(c.title)+'</h3></div><span class="when">'+esc(c.when)+'</span><div class="sub"><span class="verd '+c.outcome+'">'+c.outcome+'</span> &middot; '+esc(c.label)+'</div></summary><div class="body"></div>';
 d.addEventListener('toggle',()=>{const b=d.querySelector('.body');if(d.open&&!b.dataset.f){b.innerHTML=body(c);b.dataset.f=1}});host.append(d);cardEls.push(d)})}
function render(){let n=0;cardEls.forEach(d=>{const m=match(d._c);d.classList.toggle('hide',!m);if(m)n++});$('#count').textContent=n+' of '+D.cards.length+' cards';drawTl()}
$('#expand').onclick=()=>cardEls.forEach(d=>{if(!d.classList.contains('hide'))d.open=true});$('#collapse').onclick=()=>cardEls.forEach(d=>d.open=false);
// ---- timeline
function drawTl(){const W=1160,L=110,R=W-20,H=520,lh=62,top=34;const t0=D.win0,t1=D.win1,x=t=>L+(t-t0)/(t1-t0)*(R-L);const svg=$('#tl');let h='';
 const day0=new Date(t0*1000);day0.setHours(0,0,0,0);for(let k=0;k<4;k++){const t=day0.getTime()/1000+k*86400;if(t>=t0&&t<=t1){h+='<line x1="'+x(t)+'" x2="'+x(t)+'" y1="'+(top-6)+'" y2="'+(top+lh*6+48)+'" stroke="var(--dim)" stroke-dasharray="2 4"/><text x="'+(x(t)+4)+'" y="'+(top-12)+'" fill="var(--muted)" font-size="11">'+new Date(t*1000).toLocaleDateString('en-GB',{day:'2-digit',month:'short'})+'</text>'}}
 D.lanes.forEach((l,i)=>{const y=top+i*lh;h+='<rect x="0" y="'+y+'" width="'+W+'" height="'+(lh-4)+'" rx="8" fill="'+(i%2?'transparent':'var(--panel2)')+'" opacity=".6"/><text x="8" y="'+(y+lh/2+3)+'" fill="var(--muted)" font-size="12" font-weight="700" letter-spacing="1">'+l.toUpperCase()+'</text><line x1="'+L+'" x2="'+R+'" y1="'+(y+lh/2-2)+'" y2="'+(y+lh/2-2)+'" stroke="var(--line)" stroke-width="2"/>'});
 const gy=top+lh*6+6;h+='<text x="8" y="'+(gy+16)+'" fill="var(--gpu)" font-size="11" font-weight="700">GPU LEASE</text><text x="8" y="'+(gy+36)+'" fill="var(--human)" font-size="11" font-weight="700">HUMAN</text><text x="8" y="'+(gy+56)+'" fill="var(--agent)" font-size="11" font-weight="700">COMMITS</text>';
 D.gpu.forEach(([a,b])=>{const aa=Math.max(a,t0),bb=Math.min(b,t1);if(bb>aa)h+='<rect x="'+x(aa)+'" y="'+(gy+4)+'" width="'+Math.max(1,x(bb)-x(aa))+'" height="14" fill="var(--gpu)" opacity=".85"/>'});
 D.human.forEach(([t,c])=>{h+='<rect x="'+(x(t)-1.2)+'" y="'+(gy+24)+'" width="2.4" height="14" fill="'+(c==='stop'?'var(--negative)':'var(--human)')+'"/>'});
 D.commits.forEach(t=>{h+='<rect x="'+(x(t)-.8)+'" y="'+(gy+44)+'" width="1.6" height="14" fill="var(--agent)" opacity=".9"/>'});
 const byLane={};D.dots.forEach(d=>(byLane[d.lane]=byLane[d.lane]||[]).push(d));
 D.dots.forEach(d=>{const i=D.lanes.indexOf(d.lane),cy=top+i*lh+lh/2-2;const card=cardEls.find(e=>e.id===d.target);const on=!card||!card.classList.contains('hide');
  h+='<g class="dot" data-t="'+d.target+'" style="cursor:'+(d.target?'pointer':'default')+'" opacity="'+(on?1:.18)+'"><title>'+esc(d.label||'')+'</title><circle cx="'+x(d.t)+'" cy="'+cy+'" r="7" fill="'+(d.v==='abandoned'?'var(--bg)':COL[d.v])+'" stroke="'+(d.v==='abandoned'?'var(--abandoned)':'var(--bg)')+'" stroke-width="2.4" '+(d.v==='abandoned'?'stroke-dasharray="3 2"':'')+'/></g>'});
 svg.setAttribute('viewBox','0 0 '+W+' '+(gy+70));svg.innerHTML=h;svg.querySelectorAll('.dot').forEach(g=>{g.onmouseenter=()=>{$('#hover').textContent=g.querySelector('title').textContent};g.onmouseleave=()=>{$('#hover').textContent=''}});svg.querySelectorAll('.dot').forEach(g=>g.onclick=()=>{const id=g.dataset.t;const c=id&&document.getElementById(id);if(c){c.classList.remove('hide');c.open=true;c.scrollIntoView({behavior:'smooth',block:'start'});c.classList.add('flash');setTimeout(()=>c.classList.remove('flash'),1600)}})}
// ---- predictions
const pst=new Set();function drawPred(){const tb=$('#pred tbody');tb.innerHTML='';D.predictions.forEach((p,i)=>{if(pst.size&&!pst.has(p.verdict+(p.verdict==='miss'?':'+p.direction:'')))return;tb.append(el('tr','','<td>'+(i+1)+'</td><td>'+esc(p.claim)+'</td><td>'+esc(p.predicted)+'</td><td>'+esc(p.measured)+'</td><td class="v '+p.verdict+'">'+esc(p.verdict_text)+'</td>'))})}
const P=D.predictions;$('#pred-lead').textContent=P.length+' numeric predictions were committed before the data. '+s.hit+' hit, '+s.mixed+' mixed, '+s.miss+' missed ('+P.filter(p=>p.verdict==='miss'&&p.direction==='for').length+' in our favour, '+P.filter(p=>p.verdict==='miss'&&p.direction==='against').length+' against us).';
[['hit','hit','var(--win)'],['mixed','mixed','var(--partial)'],['miss:for','missed, better for us','var(--gpu)'],['miss:against','missed, worse for us','var(--negative)']].forEach(([v,l,c])=>{const b=el('button','btn','<i class="d" style="background:'+c+'"></i>'+l);b.setAttribute('aria-pressed','false');b.onclick=()=>{pst.has(v)?pst.delete(v):pst.add(v);b.setAttribute('aria-pressed',pst.has(v));drawPred()};$('#f-pred').append(b)});drawPred();
// ---- code
function hl(line){const e=esc(line);return e.replace(/(\/\/.*$)|(&quot;[^&]*?&quot;)|\b(const|constexpr|if|else|for|while|return|break|continue|int|float|float4|uint32_t|uint4|size_t|void|auto|template|struct|static|__device__|__global__|__forceinline__|__shared__|asm|true|false)\b|\b(0x[0-9A-Fa-f]+u?|\d+\.?\d*f?)\b|\b([A-Za-z_]\w*)(?=\()/g,(m,a,b,k,n,f)=>a?'<span class="c-com">'+a+'</span>':b?'<span class="c-str">'+b+'</span>':k?'<span class="c-kw">'+k+'</span>':n?'<span class="c-num">'+n+'</span>':'<span class="c-fn">'+f+'</span>')}
D.snips.forEach((sn,i)=>{const b=el('button','btn',esc(sn.tab));b.setAttribute('aria-pressed',i===0);b.onclick=()=>{[...$('#tabs').children].forEach(x=>x.setAttribute('aria-pressed','false'));b.setAttribute('aria-pressed','true');showSnip(sn)};$('#tabs').append(b)});
function showSnip(sn){const h=sn.lines.map((l,i)=>'<span class="ln">'+(sn.start+i)+'</span>'+hl(l)).join('\n');$('#snip').innerHTML='<h3>'+esc(sn.title)+'</h3><div class="ptr mono">'+esc(sn.ptr)+'</div><div class="note">'+esc(sn.why)+'</div><pre class="code">'+h+'</pre><button class="btn" id="cp">Copy code</button>';$('#cp').onclick=()=>navigator.clipboard&&navigator.clipboard.writeText(sn.lines.join('\n'))}
showSnip(D.snips[0]);
D.rules.forEach(r=>$('#rules').append(el('div','chip','<b style="font-size:15px">'+esc(r.t)+'</b><span>'+esc(r.d)+'</span>')));
$('#foot').innerHTML='Generated '+esc(D.meta.generated.slice(0,16).replace('T',' '))+' from commit <code>'+esc(D.meta.head)+'</code> by <code>gen.py</code>. Experiment text is copied from <code>experiments/*/README.md</code>; every number has a pointer in <code>data.json</code>. One RTX 3090, one model: results are not claimed beyond that. MIT licensed.';
buildCards();render();
if(location.hash){const c=document.getElementById(location.hash.slice(1));if(c&&c.tagName==='DETAILS')c.open=true}
</script></body></html>
"""
