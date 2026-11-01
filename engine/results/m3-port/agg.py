import json, re, statistics as S, glob, os
O=os.path.expanduser("~/llm-lab/port-m3/ab")
med=S.median
print("== bench_spec k=3 greedy (tok/s, per round base | new; median)")
for p in ["code","prose","math","long8k"]:
    rows={b:[json.load(open(f"{O}/spec_{b}_r{r}.json"))[f"{p}/k3"]["tok_s"] for r in (1,2,3) if os.path.exists(f"{O}/spec_{b}_r{r}.json")] for b in ("base","new")}
    print(p, {b:[round(x,1) for x in v] for b,v in rows.items()}, "median base %.1f new %.1f x%.3f"%(med(rows["base"]),med(rows["new"]),med(rows["new"])/med(rows["base"])))
print("== plain step ms (bench_e2e)")
for ctx in (64,2048):
    rows={}
    for b in ("base","new"):
        v=[]
        for r in (1,2,3):
            f=f"{O}/e2e_{b}_r{r}.log"
            if os.path.exists(f):
                m=re.search(rf"ctx {ctx} .*?: ([\d.]+) ms/step = ([\d.]+) tok/s",open(f).read())
                if m: v.append((float(m.group(1)),float(m.group(2))))
        rows[b]=v
    print("ctx",ctx,{b:[x[0] for x in v] for b,v in rows.items()},"median ms base %.3f new %.3f (%.1f%%)  tok/s base %.1f new %.1f"%(med([x[0] for x in rows["base"]]),med([x[0] for x in rows["new"]]),100*(med([x[0] for x in rows["new"]])/med([x[0] for x in rows["base"]])-1),med([x[1] for x in rows["base"]]),med([x[1] for x in rows["new"]])))
print("== bench_pld e6 (class: base | pld tok/s, median over rounds)")
for c in "abcde":
    R={}
    for b in ("base","new"):
        for r in (1,2,3):
            f=f"{O}/pld_{b}_r{r}.log"
            if os.path.exists(f):
                m=re.search(rf"class {c}: base ([\d.]+) tok/s \(E [\d.]+\) \| pld ([\d.]+) tok/s \(E [\d.]+\) x([\d.]+)",open(f).read())
                if m: R.setdefault(b,[]).append(tuple(map(float,m.groups())))
    if R: print(c,{b:[x[:2] for x in v] for b,v in R.items()}, {b:"base %.1f pld %.1f x%.3f"%tuple(med([x[i] for x in v]) for i in range(3)) for b,v in R.items()})
