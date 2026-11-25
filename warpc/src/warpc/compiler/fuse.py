"""Capability-driven fusion pass: graph + capability table -> claimed regions.

Four generic mechanisms over the table's data: candidate generation, a three-tier legality check (structural / semantic
/ memory), the rounding-point rule, and cost-model selection (launches removed x the target's launch cost + saved bytes
at its DRAM bandwidth, with a CTA-utilisation term), solved exactly per connected component, maximising (nodes covered,
total gain). Nodes no selected match covers are returned with the reasons their candidates were rejected.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter, deque
from dataclasses import dataclass, field

from ..ir import Graph, Node, states_of, weights_of
from .caps import REDUCTION_OPS, Absorb, Cap, CapsTable
from .plan import Pack

# Operands (inputs, outputs, weights, states) of one region: bounds a fused kernel's argument list. The widest dense
# region, attention with qk-norm, has 8.
OPERAND_CAP = 16
# Anchors on one input are tried as every subset of up to `segments` members while the group is this small (q|k|v: 3,
# so 92 seeds at most); a wider fan-out is cut into contiguous chunks to keep the seed count linear.
SUBSET_FANOUT_MAX = 8
# The exact set packing is exponential in the nodes of a component (a dense layer's largest is its attention block,
# about 10 nodes); larger components fall back to greedy selection in gain order.
EXACT_COMPONENT_MAX = 48
# The cost model prices one decode row; kernel limits are checked at the plan's rows_max.
COST_ROWS = 1
_BYTES = {"bf16": 2, "f32": 4, "i32": 4}


@dataclass
class Region:
    """A set of graph nodes one backend kernel implements, with its operands by role (graph value names)."""

    kernel: str
    covers: set[int]
    ins: dict[str, str]
    outs: dict[str, str]
    weights: dict[str, str] = field(default_factory=dict)  # role -> pack name
    states: dict[str, str] = field(default_factory=dict)
    constants: dict[str, tuple[float, ...]] = field(default_factory=dict)  # role -> fp32 values (e.g. a rope table)
    attrs: dict = field(default_factory=dict)
    packs: list[Pack] = field(default_factory=list)
    numerics: str = ""  # prose
    tolerance: tuple[str, ...] = ()  # reference rounding points ("op.point") the kernel does not reproduce
    inplace: tuple[str, str] | None = None  # (out role, in role): the output lives in the input's buffer


class Analysis:
    """Graph facts the pass and the tables' predicates read: producer / consumer maps, node ancestors (bitmasks), and
    value sizes at COST_ROWS rows. Kernel limits are checked at `rows_max` rows."""

    def __init__(self, graph: Graph, rows_max: int):
        self.graph, self.rows_max = graph, rows_max
        self.prod: dict[str, int] = {}
        self.cons: dict[str, list[int]] = {}
        for i, n in enumerate(graph.nodes):
            for o in n.outs:
                self.prod[o] = i
            for v in n.ins:
                self.cons.setdefault(v, []).append(i)
        self.outputs = set(graph.outputs.values())
        self.anc = [0] * len(graph.nodes)
        for i, n in enumerate(graph.nodes):
            m = 0
            for v in n.ins:
                p = self.prod.get(v)
                if p is not None:
                    m |= self.anc[p] | (1 << p)
            self.anc[i] = m

    def node(self, i: int) -> Node:
        return self.graph.nodes[i]

    def consumers(self, v: str, op: str | None = None) -> list[int]:
        return [i for i in self.cons.get(v, []) if op is None or self.graph.nodes[i].op == op]

    def elements(self, v: str) -> int:
        n = 1
        for d in self.graph.types[v].shape:
            n *= COST_ROWS if d == "T" else d
        return n

    def vbytes(self, v: str) -> int:
        return self.elements(v) * _BYTES[self.graph.types[v].dtype]

    def weight_bytes(self, n: Node) -> int:
        return sum(w.nbytes for w in (self.graph.weights[k] for k in weights_of(n)) if len(w.shape) == 2)


@dataclass
class Match:
    cap: Cap
    rank: int  # the capability's position in its table: the cost-model tie-break
    anchors: list[int]
    pair: int | None = None
    pre: list[int] = field(default_factory=list)
    post: list[int] = field(default_factory=list)
    covers: frozenset = frozenset()
    exposed: frozenset = frozenset()  # values the kernel writes that outside nodes may read
    inplace_node: int | None = None  # the absorbed add that updates `h` in place
    h: str | None = None  # the value updated in place (the residual stream)
    consts: dict = field(default_factory=dict)
    gaps: tuple = ()  # (op, point) pairs of reference rounding the kernel does not reproduce
    gain: float = 0.0
    why_not: str = ""

    def key(self):
        return (self.cap.kernel, tuple(sorted(self.covers)))


@dataclass
class Fused:
    regions: list[Region]
    unclaimed: dict[int, str]  # node -> why no capability claimed it

    def backlog(self, g: Graph) -> list[tuple[str, str, int]]:
        """(op, reason, count) of the unclaimed nodes, largest first."""
        c = Counter((g.nodes[i].op, r) for i, r in self.unclaimed.items())
        return sorted(((op, r, n) for (op, r), n in c.items()), key=lambda t: (-t[2], t[0], t[1]))


def _anchor_reason(cap: Cap, n: Node, g: Graph) -> str | None:
    if cap.anchor_schemes and n.op == "linear":
        s = g.weights[n.attrs["weight"]].scheme
        if s not in cap.anchor_schemes:
            return f"weight scheme {s} (kernel reads {'/'.join(cap.anchor_schemes)})"
    return None


def _seeds(cap: Cap, A: Analysis, rej: dict):
    elig = []
    for i, n in enumerate(A.graph.nodes):
        if n.op != cap.anchor:
            continue
        why = _anchor_reason(cap, n, A.graph)
        if why:
            rej.setdefault(i, []).append((_depth(cap), f"{cap.kernel}: {why}"))
        else:
            elig.append(i)
    eset = set(elig)
    if cap.pair:  # two anchors on one input merged by one op: the gate|up pair
        for i, n in enumerate(A.graph.nodes):
            if n.op in cap.pair and len(n.ins) == 2:
                a, b = A.prod.get(n.ins[0]), A.prod.get(n.ins[1])
                if a in eset and b in eset and a != b and A.node(a).ins == A.node(b).ins:
                    yield [a, b], i
        return
    if cap.segments > 1:  # anchors on one input
        groups: dict[tuple, list[int]] = {}
        for i in elig:
            groups.setdefault(tuple(A.node(i).ins), []).append(i)
        for members in groups.values():
            if len(members) <= SUBSET_FANOUT_MAX:
                for r in range(1, cap.segments + 1):
                    for sub in itertools.combinations(members, r):
                        yield list(sub), None
            else:
                for s in range(0, len(members), cap.segments):
                    yield members[s : s + cap.segments], None
        return
    for i in elig:
        yield [i], None


def _attrs_ok(a: Absorb, n: Node) -> bool:
    return all(n.attrs.get(k) == v for k, v in a.attrs.items())


def _post_chain(cap: Cap, A: Analysis, value: str) -> tuple[list[int], str | None]:
    """The consumer chain the capability's `post` records describe, starting at `value`: (nodes, reason it cannot be
    matched)."""
    chain = []
    for a in cap.post:
        nxt = None
        for c in A.consumers(value, a.op):
            n = A.node(c)
            if _attrs_ok(a, n) and not (n.op == "add" and n.ins[0] == n.ins[1]):
                nxt = c
                break
        if nxt is None:
            if a.min > 0:
                return [], f"no {a.op} consumer to absorb"
            break
        if a.op in REDUCTION_OPS and not a.reduction_ok:
            return [], f"{a.op} is a reduction: not legal in an epilogue"
        chain.append(nxt)
        value = A.node(nxt).outs[0]
    return chain, None


def _pre_closure(cap: Cap, A: Analysis, covers: set[int], include: set[str]) -> list[int]:
    """Producers upstream of the seed that the capability's `pre` records allow, restricted to the ops in `include`."""
    by_op = {a.op: a for a in cap.pre}
    count: Counter = Counter()
    absorbed: list[int] = []
    work = deque(v for i in sorted(covers) for v in A.node(i).ins)
    while work:
        p = A.prod.get(work.popleft())
        if p is None or p in covers:
            continue
        n = A.node(p)
        a = by_op.get(n.op)
        if a is None or n.op not in include or count[n.op] >= a.max or not _attrs_ok(a, n):
            continue
        covers.add(p)
        count[n.op] += 1
        absorbed.append(p)
        work.extend(n.ins)
    return absorbed


def _external(A: Analysis, covers) -> tuple[list[str], list[str]]:
    """(input values read from outside, output values read from outside or graph outputs), in node order."""
    ins, outs = [], []
    for i in sorted(covers):
        n = A.node(i)
        for v in n.ins:
            p = A.prod.get(v)
            if (p is None or p not in covers) and v not in ins:
                ins.append(v)
        for o in n.outs:
            if o in A.outputs or any(c not in covers for c in A.cons.get(o, [])):
                outs.append(o)
    return ins, outs


def _operands(A: Analysis, covers, ext_in, ext_out) -> int:
    ws, st = set(), set()
    for i in covers:
        ws.update(weights_of(A.node(i)))
        st.update(states_of(A.node(i)))
    return len(ext_in) + len(ext_out) + len(ws) + len(st)


def _build(cap: Cap, rank: int, A: Analysis, anchors: list[int], pair: int | None, include: set[str]) -> Match:
    """One match for a seed and a choice of optional prologue ops; `why_not` is set when a legality tier rejects it.
    The epilogue chain is matched first, so the prologue also absorbs producers of the epilogue's other operands (a
    gate's reshape in front of the op that applies it)."""
    g = A.graph
    covers = set(anchors) | ({pair} if pair is not None else set())
    m = Match(cap, rank, anchors, pair)
    res = None
    if pair is not None:
        res = A.node(pair).outs[0]
    elif len(anchors) == 1:
        res = A.node(anchors[0]).outs[0]
    if cap.post:
        if res is None:
            m.why_not = "an epilogue chain needs a single result"
            return m
        m.post, why = _post_chain(cap, A, res)
        if why:
            m.why_not = why
            return m
        covers |= set(m.post)
    m.pre = _pre_closure(cap, A, covers, include)
    cnt = Counter(A.node(i).op for i in m.pre)
    for a in cap.pre:
        if cnt[a.op] < a.min:
            m.why_not = f"needs {a.min} {a.op} upstream, found {cnt[a.op]}"
            return m
    m.covers = frozenset(covers)
    # structural tier: values leaving the region must be ones the kernel writes
    exposed = set()
    last = A.node(m.post[-1] if m.post else (pair if pair is not None else anchors[-1])).outs
    if not m.post and pair is None:
        for i in anchors:
            exposed.update(A.node(i).outs)
    exposed.update(last)
    absorbs = {a.op: a for a in (*cap.pre, *cap.post)}
    for i in (*m.pre, *m.post):
        if absorbs[A.node(i).op].exposes:
            exposed.update(A.node(i).outs)
    m.exposed = frozenset(exposed)
    ext_in, ext_out = _external(A, covers)
    for v in ext_out:
        if v not in exposed:
            m.why_not = f"intermediate of {A.node(A.prod[v]).op} has users outside the region (single-user rule)"
            return m
    if _operands(A, covers, ext_in, ext_out) > OPERAND_CAP:
        m.why_not = f"more than {OPERAND_CAP} operands"
        return m
    for i in itertools.chain([pair] if pair is not None else [], m.post):
        n = A.node(i)
        if n.op in ("add", "silu_mul") and len({g.types[v] for v in n.ins}) != 1:
            m.why_not = f"{n.op} operands differ in shape or dtype"
            return m
    # memory tier: in-place residual update
    for i in (*m.pre, *m.post):
        if absorbs[A.node(i).op].inplace:
            why = _inplace(A, m, i, covers)
            if why:
                m.why_not = why
                return m
    # semantic tier
    m.consts = cap.constants(m, A) if cap.constants else {}
    verdict = cap.evaluate(m.consts)
    if verdict.reasons:
        m.why_not = "; ".join(verdict.reasons)
        return m
    if verdict.ctas is not None:
        m.consts = {**m.consts, "ctas": verdict.ctas}
    if cap.check:
        why = cap.check(m, A)
        if why:
            m.why_not = why
            return m
    # rounding-point rule
    gaps = []
    for i in sorted(covers):
        n = A.node(i)
        for p in cap.gaps(n.op, g.types[n.outs[0]].dtype):
            gaps.append((n.op, p))
    m.gaps = tuple(dict.fromkeys(gaps))
    return m


def _inplace(A: Analysis, m: Match, i: int, covers) -> str | None:
    """Memory tier for an absorbed residual add `i`: pick the operand updated in place and prove that no later reader
    exists and that the region itself does not read it (the kernel's CTAs write their slice of `h` while others still
    read their inputs, so `h` must not be any other input of the task: y = linear(x); add(x, y) would update the GEMV's
    own input). One region updates one value in place."""
    n = A.node(i)
    if m.h is not None:
        return "two in-place updates in one region: the region has one in-place output"
    cover_anc = 0
    for c in covers:
        cover_anc |= A.anc[c]
    why = "no operand of the add can be updated in place"
    for v in n.ins:
        p = A.prod.get(v)
        if p is not None and p in covers:  # the other operand is the kernel's own result
            continue
        if v in A.outputs or v in A.graph.inputs:
            why = "the operand is a graph input or output: no in-place update"
            continue
        late = [c for c in A.cons.get(v, []) if c not in covers and not (cover_anc >> c) & 1]
        if late:
            reader = A.node(late[0]).op
            why = f"the operand is read by a {reader} that does not precede the region: in-place update would race it"
            continue
        own = [c for c in sorted(covers) if c != i and v in A.node(c).ins]
        if own or n.ins.count(v) > 1:
            reader = A.node(own[0]).op if own else "add"
            why = f"the operand is also an input of the region's {reader}: in-place update would race its own read"
            continue
        m.inplace_node, m.h = i, v
        return None
    return why


def _gain_us(A: Analysis, table: CapsTable, m: Match) -> float:
    """Launch cost x launches removed + (bytes moved unfused - bytes moved fused) / DRAM bandwidth, each side at its own
    CTA utilisation. A match of one bare anchor gains 0."""
    covers, tgt = m.covers, table.target
    if len(covers) == 1:
        return 0.0

    def util(ctas: int) -> float:
        return min(1.0, max(1, ctas) / tgt.sms)

    def launches(n: Node) -> int:
        return table.unfused_launches.get(n.op, 1)

    removed = sum(launches(A.node(i)) for i in covers) - m.cap.launches
    t_sep, total, saved = 0.0, 0, 0
    for i in covers:
        n = A.node(i)
        b = sum(A.vbytes(v) for v in n.ins) + sum(A.vbytes(o) for o in n.outs) + A.weight_bytes(n)
        total += b
        if launches(n):
            t_sep += b / (tgt.dram_bandwidth * util(table.unfused_ctas(n, A)))
        for o in n.outs:
            internal = sum(1 for c in A.cons.get(o, []) if c in covers)
            written = o in A.outputs or any(c not in covers for c in A.cons.get(o, [])) or o in m.exposed
            saved += A.vbytes(o) * (internal + (0 if written else 1))
    anchor_ctas = m.consts.get("ctas") or table.unfused_ctas(A.node(m.anchors[0]), A)
    t_fused = (total - saved) / (tgt.dram_bandwidth * util(anchor_ctas))
    return tgt.launch_us * removed + (t_sep - t_fused) * 1e6


def _depth(cap: Cap) -> int:
    """How much a capability asks of the graph around its anchor: the reasons of the plainest kernel are the
    informative ones."""
    return sum(a.min for a in cap.pre) + sum(a.min for a in cap.post) + (1 if cap.pair else 0)


def _candidates(A: Analysis, table: CapsTable, strict: bool) -> tuple[list[Match], dict[int, list[str]]]:
    """Every legal match of every capability with its gain, and per node why its seeds were turned down (plainest
    kernel first)."""
    rej: dict[int, list] = {}
    out: dict[tuple, Match] = {}
    for cap in table.caps:
        rank = table.rank[cap.kernel]
        optional = [a.op for a in cap.pre if a.min == 0]
        required = {a.op for a in cap.pre if a.min > 0}
        subsets = (set(s) for r in range(len(optional), -1, -1) for s in itertools.combinations(optional, r))
        variants = [required | s for s in subsets]
        for anchors, pair in _seeds(cap, A, rej):
            seen = set()
            for inc in variants:
                m = _build(cap, rank, A, anchors, pair, inc)
                if m.why_not:
                    for i in (*anchors, *([pair] if pair is not None else []), *m.post):
                        rej.setdefault(i, []).append((_depth(cap), f"{cap.kernel}: {m.why_not}"))
                    continue
                if m.covers in seen:
                    continue
                if strict and m.gaps:
                    gaps = ", ".join(f"{o}.{p}" for o, p in m.gaps)
                    for i in (*anchors, *([pair] if pair is not None else [])):
                        rej.setdefault(i, []).append((_depth(cap), f"{cap.kernel}: rounding gap {gaps}"))
                    continue
                seen.add(m.covers)
                m.gain = _gain_us(A, table, m)
                out.setdefault(m.key(), m)
    # stable sort: table order within a depth
    reasons = {i: [t for _, t in sorted(lst, key=lambda x: x[0])] for i, lst in rej.items()}
    return list(out.values()), reasons


def _components(cands: list[Match]) -> list[list[Match]]:
    parent: dict[int, int] = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for c in cands:
        it = iter(sorted(c.covers))
        first = next(it)
        for n in it:
            parent[find(n)] = find(first)
        find(first)
    comps: dict[int, list[Match]] = {}
    for c in cands:
        comps.setdefault(find(min(c.covers)), []).append(c)
    return sorted(comps.values(), key=lambda cs: min(min(c.covers) for c in cs))


def _order(c: Match):
    return (-round(c.gain, 9), c.rank, min(c.covers))


def _solve(cs: list[Match]) -> list[Match]:
    """Exact weighted set packing over one component: maximise (nodes covered, total gain); the first optimum in
    `_order` wins ties."""
    cs = sorted(cs, key=_order)
    nodes = sorted(set().union(*(c.covers for c in cs)))
    if len(nodes) > EXACT_COMPONENT_MAX:
        taken, cov = [], set()
        for c in cs:
            if not (c.covers & cov):
                taken.append(c)
                cov |= c.covers
        return taken
    by_node = {n: [c for c in cs if n in c.covers] for n in nodes}
    best: list = [(-1, -math.inf), []]
    total = len(nodes)

    def dfs(k, covered, chosen, ncov, gain):
        while k < total and nodes[k] in covered:  # `covered` also holds nodes deliberately left uncovered
            k += 1
        if k == total:
            score = (ncov, round(gain, 9))
            if score > best[0]:
                best[0], best[1] = score, list(chosen)
            return
        n = nodes[k]
        opts = [c for c in by_node[n] if not (c.covers & covered)]
        for c in opts:
            chosen.append(c)
            dfs(k + 1, covered | c.covers, chosen, ncov + len(c.covers), gain + c.gain)
            chosen.pop()
        # leaving n uncovered must still be able to match the best coverage
        reachable = ncov + sum(1 for x in nodes[k + 1 :] if x not in covered)
        if not opts or reachable >= best[0][0]:
            dfs(k + 1, covered | {n}, chosen, ncov, gain)

    dfs(0, frozenset(), [], 0, 0.0)
    return best[1]


def _region_cycle(A: Analysis, chosen: list[Match]) -> list[Match]:
    """Multi-node matches on a cycle of the contracted region graph (empty if acyclic); uncovered nodes are their own
    regions."""
    rid = list(range(len(A.graph.nodes)))
    members: dict[int, Match] = {}
    for m in chosen:
        r = min(m.covers)
        for i in m.covers:
            rid[i] = r
        members[r] = m
    succ: dict[int, set[int]] = {}
    pred: dict[int, set[int]] = {}
    for i, nd in enumerate(A.graph.nodes):
        for v in nd.ins:
            p = A.prod.get(v)
            if p is not None and rid[p] != rid[i]:
                succ.setdefault(rid[p], set()).add(rid[i])
                pred.setdefault(rid[i], set()).add(rid[p])
    left = set(rid)
    # peel sources, then sinks: what survives lies on cycles or between them
    for fwd, back in ((succ, pred), (pred, succ)):
        deg = {r: len(back.get(r, set()) & left) for r in left}
        stack = [r for r, d in deg.items() if d == 0]
        while stack:
            r = stack.pop()
            left.discard(r)
            for s in fwd.get(r, ()):
                if s in left:
                    deg[s] -= 1
                    if deg[s] == 0:
                        stack.append(s)
    return [members[r] for r in sorted(left) if r in members and len(members[r].covers) > 1]


def _select(A: Analysis, cands: list[Match]) -> list[Match]:
    banned: set = set()
    while True:
        live = [c for c in cands if c.key() not in banned]
        chosen: list[Match] = []
        for comp in _components(live):
            chosen.extend(_solve(comp))
        cyc = _region_cycle(A, chosen)
        if not cyc:
            return sorted(chosen, key=lambda c: min(c.covers))
        # drop the cheapest fusion on a cycle and re-solve
        banned.add(min(cyc, key=lambda c: (c.gain, -min(c.covers))).key())


def _region_of(m: Match, A: Analysis) -> Region:
    r = m.cap.emit(m, A)
    r.tolerance = tuple(f"{o}.{p}" for o, p in m.gaps)
    return r


def fuse(g: Graph, table: CapsTable, rows_max: int, *, strict_rounding: bool = False) -> Fused:
    """Run the pass with kernel limits checked at `rows_max` rows. A fusion that drops a reference rounding point is
    taken and the point recorded in the region's `tolerance`, or, with `strict_rounding`, refused."""
    A = Analysis(g, rows_max)
    cands, rej = _candidates(A, table, strict_rounding)
    chosen = _select(A, cands)
    regions = [_region_of(m, A) for m in chosen]
    covered = set().union(*(m.covers for m in chosen)) if chosen else set()
    ops = table.ops()
    anchors = {c.anchor for c in table.caps}
    absorbed_by: dict[str, set[str]] = {}
    for c in table.caps:
        for a in c.pre:
            absorbed_by.setdefault(a.op, set()).add(f"prologue of {c.kernel}")
        for a in c.post:
            absorbed_by.setdefault(a.op, set()).add(f"epilogue of {c.kernel}")
        for op in c.pair:
            absorbed_by.setdefault(op, set()).add(f"pair epilogue of {c.kernel}")
    unclaimed = {}
    for i, n in enumerate(g.nodes):
        if i in covered:
            continue
        if n.op not in ops:
            unclaimed[i] = "no capability for this op"
        elif rej.get(i):
            unclaimed[i] = rej[i][0]
        elif n.op not in anchors:
            unclaimed[i] = "only absorbable as " + ", ".join(sorted(absorbed_by[n.op])) + ": no such match here"
        else:
            unclaimed[i] = "no legal match"
    return Fused(regions, unclaimed)
