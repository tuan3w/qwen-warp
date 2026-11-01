"""Tiny SVG toolkit for poster.py (stdlib only): palette, text measuring/wrapping, shapes, icons, code highlighting."""
import re
from xml.sax.saxutils import escape

W = 1200
M = 40
BG, PANEL, PANEL2, LINE = "#0d1117", "#151b26", "#1c2433", "#2e394e"
INK, MUTED, DIM = "#e8edf5", "#a3b0c6", "#7586a3"
WIN, PART, NEG, ABAN = "#3ecf8e", "#f5b83d", "#ff6b6b", "#8d99ae"
HUMAN, AGENT, GPU, HARN = "#ff9f5a", "#b794f6", "#4da3ff", "#2dd4bf"
VERDICT = {"win": WIN, "partial": PART, "negative": NEG, "abandoned": ABAN}
SANS = "Inter, 'Segoe UI', -apple-system, 'Helvetica Neue', Helvetica, Arial, 'DejaVu Sans', sans-serif"
MONO = "ui-monospace, 'SF Mono', Menlo, Consolas, 'DejaVu Sans Mono', 'Liberation Mono', monospace"


def esc(s):
    return escape(str(s), {'"': "&quot;"})


def tw(s, size, mono=False, bold=False):
    """Conservative text width (px) so labels never overflow whichever system font is used."""
    if mono:
        return len(s) * size * 0.61
    narrow = sum(1 for c in s if c in "iljtfIr.,:;'|!()[] ")
    wide = sum(1 for c in s if c in "MWmw@%—")
    n = len(s)
    return (n * 0.555 - narrow * 0.2 + wide * 0.25) * size * (1.06 if bold else 1.0)


def wrap(s, width, size, mono=False, bold=False):
    words, lines, cur = s.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if tw(t, size, mono, bold) <= width or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


class Svg:
    def __init__(self):
        self.p = []

    def add(self, s):
        self.p.append(s)

    def text(self, x, y, s, size=14, fill=INK, anchor="start", weight=400, mono=False, opacity=None, ls=None, italic=False, cls=""):
        a = f' text-anchor="{anchor}"' if anchor != "start" else ""
        o = f' opacity="{opacity}"' if opacity is not None else ""
        l = f' letter-spacing="{ls}"' if ls else ""
        it = ' font-style="italic"' if italic else ""
        fam = MONO if mono else SANS
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{fam}" font-size="{size}" font-weight="{weight}" fill="{fill}"{a}{o}{l}{it}>{esc(s)}</text>')

    def tspans(self, x, y, runs, size=14, mono=True, anchor="start"):
        """runs = [(text, fill)] on one line."""
        fam = MONO if mono else SANS
        inner = "".join(f'<tspan fill="{c}">{esc(t)}</tspan>' for t, c in runs)
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{fam}" font-size="{size}" xml:space="preserve">{inner}</text>')

    def para(self, x, y, s, width, size=14, fill=MUTED, lh=1.38, weight=400, bold=False, mono=False, anchor="start"):
        for ln in wrap(s, width, size, mono, bold or weight >= 600):
            self.text(x, y, ln, size, fill, anchor, weight, mono)
            y += size * lh
        return y

    def rect(self, x, y, w, h, fill="none", stroke=None, rx=0, sw=1, dash=None, opacity=None, extra=""):
        st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        d = f' stroke-dasharray="{dash}"' if dash else ""
        o = f' opacity="{opacity}"' if opacity is not None else ""
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"{st}{d}{o} {extra}/>')

    def line(self, x1, y1, x2, y2, stroke=LINE, sw=1, dash=None, opacity=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        o = f' opacity="{opacity}"' if opacity is not None else ""
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}"{d}{o}/>')

    def path(self, d, fill="none", stroke=None, sw=1, dash=None, opacity=None, marker=False):
        st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        da = f' stroke-dasharray="{dash}"' if dash else ""
        o = f' opacity="{opacity}"' if opacity is not None else ""
        m = ' marker-end="url(#arr)"' if marker else ""
        self.add(f'<path d="{d}" fill="{fill}"{st}{da}{o}{m} stroke-linecap="round" stroke-linejoin="round"/>')

    def circle(self, cx, cy, r, fill="none", stroke=None, sw=1, opacity=None, dash=None):
        st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        o = f' opacity="{opacity}"' if opacity is not None else ""
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}"{st}{o}{d}/>')

    def arrow(self, x1, y1, x2, y2, stroke=MUTED, sw=1.6, dash=None, head=7):
        import math
        a = math.atan2(y2 - y1, x2 - x1)
        hx, hy = x2 - head * math.cos(a), y2 - head * math.sin(a)
        self.line(x1, y1, hx, hy, stroke, sw, dash)
        p1 = (x2 - head * math.cos(a - 0.45), y2 - head * math.sin(a - 0.45))
        p2 = (x2 - head * math.cos(a + 0.45), y2 - head * math.sin(a + 0.45))
        self.add(f'<path d="M{x2:.1f},{y2:.1f} L{p1[0]:.1f},{p1[1]:.1f} L{p2[0]:.1f},{p2[1]:.1f} Z" fill="{stroke}"/>')

    def section_title(self, y, kicker, title, sub=None, color=AGENT):
        self.text(M, y, kicker, 13, color, weight=700, ls=2.2)
        self.text(M, y + 34, title, 28, INK, weight=800)
        yy = y + 34
        if sub:
            yy = self.para(M, y + 60, sub, W - 2 * M, 15, MUTED)
            return yy
        return yy + 14


# ----------------------------------------------------------------------------- icons (simple shapes)
def icon_human(s, cx, cy, r, color):
    s.circle(cx, cy - r * 0.45, r * 0.38, color)
    s.path(f"M{cx - r * 0.8:.1f},{cy + r * 0.95:.1f} Q{cx - r * 0.8:.1f},{cy + r * 0.05:.1f} {cx:.1f},{cy + r * 0.05:.1f} Q{cx + r * 0.8:.1f},{cy + r * 0.05:.1f} {cx + r * 0.8:.1f},{cy + r * 0.95:.1f} Z", color)


def icon_agent(s, cx, cy, r, color, bg=PANEL):
    s.rect(cx - r * 0.85, cy - r * 0.55, r * 1.7, r * 1.25, color, rx=r * 0.28)
    s.circle(cx - r * 0.32, cy + r * 0.02, r * 0.17, bg)
    s.circle(cx + r * 0.32, cy + r * 0.02, r * 0.17, bg)
    s.line(cx, cy - r * 0.55, cx, cy - r * 0.95, color, 2)
    s.circle(cx, cy - r * 1.02, r * 0.1, color)


def icon_gpu(s, cx, cy, r, color, bg=PANEL):
    s.rect(cx - r, cy - r * 0.65, r * 2, r * 1.3, color, rx=r * 0.14)
    s.circle(cx - r * 0.38, cy, r * 0.34, bg)
    s.circle(cx - r * 0.38, cy, r * 0.12, color)
    s.circle(cx + r * 0.42, cy, r * 0.28, bg)
    s.circle(cx + r * 0.42, cy, r * 0.1, color)
    for i in range(4):
        s.rect(cx - r * 0.75 + i * r * 0.5, cy + r * 0.65, r * 0.18, r * 0.2, color)


def icon_lock(s, cx, cy, r, color, bg=PANEL):
    s.rect(cx - r * 0.7, cy - r * 0.1, r * 1.4, r * 1.0, color, rx=r * 0.15)
    s.path(f"M{cx - r * 0.42:.1f},{cy - r * 0.1:.1f} V{cy - r * 0.45:.1f} A{r * 0.42:.1f},{r * 0.42:.1f} 0 0 1 {cx + r * 0.42:.1f},{cy - r * 0.45:.1f} V{cy - r * 0.1:.1f}", stroke=color, sw=r * 0.2)
    s.circle(cx, cy + r * 0.38, r * 0.13, bg)


def icon_gate(s, cx, cy, r, color, bg=PANEL):
    s.path(f"M{cx:.1f},{cy - r:.1f} L{cx + r * 0.85:.1f},{cy - r * 0.55:.1f} V{cy + r * 0.1:.1f} Q{cx + r * 0.85:.1f},{cy + r * 0.75:.1f} {cx:.1f},{cy + r:.1f} Q{cx - r * 0.85:.1f},{cy + r * 0.75:.1f} {cx - r * 0.85:.1f},{cy + r * 0.1:.1f} V{cy - r * 0.55:.1f} Z", color)
    s.path(f"M{cx - r * 0.38:.1f},{cy + r * 0.02:.1f} L{cx - r * 0.08:.1f},{cy + r * 0.34:.1f} L{cx + r * 0.42:.1f},{cy - r * 0.3:.1f}", stroke=bg, sw=r * 0.2)


# ----------------------------------------------------------------------------- code highlighting
CUDA_KW = r"\b(const|constexpr|if|else|for|while|return|break|continue|int|float|float4|uint32_t|uint4|size_t|void|auto|template|struct|static|__device__|__global__|__forceinline__|__shared__|asm|true|false|nullptr)\b"
COM = "#7d8ba5"
KWC, NUMC, STRC, FNC = "#ff9aa2", "#f5b83d", "#7ee0b6", "#79c0ff"


def highlight(line):
    """[(text, colour)] for one code line (comments, keywords, numbers, calls)."""
    code, com = line, ""
    i = code.find("//")
    if i >= 0:
        code, com = line[:i], line[i:]
    runs = []
    pos = 0
    tok = re.compile(rf"({CUDA_KW})|(0x[0-9A-Fa-f]+u?|\b\d+(?:\.\d+)?f?\b)|(\b[A-Za-z_]\w*)(?=\()|(\"[^\"]*\")")
    for m in tok.finditer(code):
        if m.start() > pos:
            runs.append((code[pos:m.start()], INK))
        if m.group(1):
            runs.append((m.group(0), KWC))
        elif m.group(2):
            runs.append((m.group(0), NUMC))
        elif m.group(3):
            runs.append((m.group(0), FNC))
        else:
            runs.append((m.group(0), STRC))
        pos = m.end()
    if pos < len(code):
        runs.append((code[pos:], INK))
    if com:
        runs.append((com, COM))
    return runs or [("", INK)]
