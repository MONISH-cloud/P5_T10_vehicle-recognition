"""Presentation layer for the Gradio app: theme, CSS, the eagle intro animation and HTML result cards.
Pure strings and helpers, so the recognition code stays untouched."""
from __future__ import annotations
import html
import json

# An original, geometric eagle mark (head in profile with swept wings), drawn as strokes so it can animate.
EAGLE_SVG = """<svg class="{cls}" viewBox="0 0 160 120" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <defs>
    <linearGradient id="egMetal{uid}" x1="0" y1="1" x2="1" y2="0">
      <stop offset="0" stop-color="#b8945a"/><stop offset=".35" stop-color="#f3e3b8"/>
      <stop offset=".6" stop-color="#d7b878"/><stop offset="1" stop-color="#fff3d6"/>
    </linearGradient>
  </defs>
  <g class="eg-body" fill="url(#egMetal{uid})">
    <path class="eg-blade" d="M44 66 C74 46 112 26 154 6 C146 30 118 54 72 72 C62 74 52 72 44 66 Z"/>
    <path class="eg-blade" d="M66 78 C96 64 126 50 156 34 C150 56 126 74 90 84 C82 86 72 84 66 78 Z"/>
    <path class="eg-blade" d="M78 90 C106 84 130 76 150 62 C144 80 124 94 96 98 C88 99 82 96 78 90 Z"/>
    <path class="eg-head" d="M48 62 C35 62 24 68 17 77 C11 85 8 97 11 110 L16 101 C19 97 23 95 29 94 C37 93 45 96 52 100 C74 112 108 112 136 96 C112 104 82 102 64 92 C54 86 50 76 48 62 Z"/>
  </g>
  <path class="eg-cut" d="M20 84 L50 75 L44 82 C38 85 28 86 20 84 Z"/>
  <path class="eg-cut eg-gape" d="M18 98 C24 95 30 94 36 95" stroke="#0a0e15" stroke-width="2.2" stroke-linecap="round" fill="none"/>
  <circle class="eg-spark" cx="153" cy="8" r="2.2"/><circle class="eg-spark" cx="155" cy="35" r="2"/><circle class="eg-spark" cx="149" cy="63" r="1.8"/>
</svg>"""

INTRO_HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
#eagle-intro{position:fixed;inset:0;z-index:99999;display:flex;flex-direction:column;align-items:center;justify-content:center;
  background:radial-gradient(1200px 600px at 50% 40%,#16202f 0%,#0a0e15 60%,#06080c 100%);transition:opacity .7s ease,visibility .7s ease;cursor:pointer}
#eagle-intro.done{opacity:0;visibility:hidden}
#eagle-intro .intro-mark{width:min(56vw,280px);height:auto;overflow:visible;filter:drop-shadow(0 8px 14px rgba(0,0,0,.55)) drop-shadow(0 0 30px rgba(232,179,57,.28))}
#eagle-intro .intro-mark .eg-head{opacity:0;transform-box:fill-box;transform-origin:30% 60%;animation:eg-head .6s cubic-bezier(.2,.7,.2,1) .1s forwards}
#eagle-intro .intro-mark .eg-blade{opacity:0;transform-box:fill-box;transform-origin:0% 100%;transform:rotate(-14deg) scale(.35)}
#eagle-intro .intro-mark .eg-blade:nth-child(3){animation:eg-sweep .55s cubic-bezier(.2,.8,.2,1) .45s forwards}
#eagle-intro .intro-mark .eg-blade:nth-child(2){animation:eg-sweep .55s cubic-bezier(.2,.8,.2,1) .62s forwards}
#eagle-intro .intro-mark .eg-blade:nth-child(1){animation:eg-sweep .6s cubic-bezier(.2,.8,.2,1) .79s forwards}
#eagle-intro .intro-mark .eg-cut{fill:#0a0e15;opacity:0;animation:eg-fade .25s ease 1.25s forwards}
#eagle-intro .intro-mark .eg-gape{fill:none}
#eagle-intro .intro-mark .eg-spark{fill:#fffaf0;opacity:0;transform-box:fill-box;transform-origin:center;filter:drop-shadow(0 0 6px #fff3d6) drop-shadow(0 0 14px rgba(255,235,190,.9))}
#eagle-intro .intro-mark .eg-spark:nth-of-type(1){animation:eg-spark .9s ease 1.45s forwards}
#eagle-intro .intro-mark .eg-spark:nth-of-type(2){animation:eg-spark .9s ease 1.6s forwards}
#eagle-intro .intro-mark .eg-spark:nth-of-type(3){animation:eg-spark .9s ease 1.75s forwards}
#eagle-intro .intro-word{margin-top:22px;font:700 clamp(20px,4vw,30px)/1.1 Inter,system-ui,sans-serif;letter-spacing:.32em;color:#f3efe6;
  opacity:0;transform:translateY(8px);animation:eg-rise .7s ease 1.2s forwards;text-align:center}
#eagle-intro .intro-sub{margin-top:10px;font:500 12px/1.4 Inter,system-ui,sans-serif;letter-spacing:.28em;color:#9aa4b2;
  opacity:0;animation:eg-rise .7s ease 1.5s forwards;text-transform:uppercase;text-align:center;padding:0 16px}
@keyframes eg-head{from{opacity:0;transform:translate(-10px,8px) scale(.9)}to{opacity:1;transform:none}}
@keyframes eg-sweep{to{opacity:1;transform:none}}
@keyframes eg-fade{to{opacity:1}}
@keyframes eg-spark{0%{opacity:0;transform:scale(.2)}45%{opacity:1;transform:scale(2.4)}100%{opacity:.85;transform:scale(1)}}
@keyframes eg-rise{to{opacity:1;transform:none}}
</style>
<script>
(function(){
  function mount(){
    if(document.getElementById('eagle-intro')) return;
    var el=document.createElement('div'); el.id='eagle-intro'; el.setAttribute('role','presentation');
    el.innerHTML=__EAGLE__+'<div class="intro-word">VEHICLE&nbsp;INTELLIGENCE</div><div class="intro-sub">Fine-grained recognition · 196 classes</div>';
    document.body.appendChild(el);
    function close(){el.classList.add('done'); setTimeout(function(){el.remove()},800)}
    el.addEventListener('click',close); setTimeout(close,2700);
  }
  if(document.body) mount(); else document.addEventListener('DOMContentLoaded',mount);
})();
</script>
"""


def head_html() -> str:
    svg = EAGLE_SVG.replace("{uid}", "i").format(cls="intro-mark").replace("\n", "")
    return INTRO_HEAD.replace("__EAGLE__", json.dumps(svg))


CSS = """
:root{--vi-bg:#0a0e15;--vi-panel:#111824;--vi-panel-2:#162030;--vi-line:#243044;--vi-text:#f3efe6;--vi-muted:#9aa4b2;
  --vi-gold:#e8b339;--vi-gold-2:#f5cf72;--vi-good:#3fb68b;--vi-warn:#e0a53a;--vi-low:#e0703a;--vi-bad:#e05656}
body,.gradio-container{background:var(--vi-bg)!important;color:var(--vi-text)!important;font-family:Inter,system-ui,-apple-system,Segoe UI,sans-serif!important}
.gradio-container{max-width:1240px!important;margin:0 auto!important;padding:0 16px 40px!important}
footer{display:none!important}
.vi-header{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:18px 4px 14px;border-bottom:1px solid var(--vi-line);margin-bottom:22px;flex-wrap:wrap}
.vi-brand{display:flex;align-items:center;gap:12px}
.vi-brand .vi-mark{width:58px;height:44px;filter:drop-shadow(0 2px 4px rgba(0,0,0,.5)) drop-shadow(0 0 10px rgba(232,179,57,.22))}
.vi-brand .vi-mark .eg-cut{fill:#0a0e15} .vi-brand .vi-mark .eg-gape{fill:none} .vi-brand .vi-mark .eg-spark{display:none}
.vi-name{font-weight:800;letter-spacing:.18em;font-size:15px;color:var(--vi-text)}
.vi-tag{font-size:12px;color:var(--vi-muted);letter-spacing:.06em;margin-top:2px}
.vi-pills{display:flex;gap:8px;flex-wrap:wrap}
.vi-pill{font-size:12px;color:var(--vi-muted);border:1px solid var(--vi-line);padding:5px 10px;border-radius:999px;background:var(--vi-panel)}
.vi-hero{display:grid;grid-template-columns:1.2fr 1fr;gap:28px;align-items:center;margin:8px 0 26px}
.vi-hero h1{font-size:clamp(28px,4vw,44px);line-height:1.08;margin:0 0 12px;font-weight:800;letter-spacing:-.02em;color:var(--vi-text)}
.vi-hero h1 span{color:var(--vi-gold)!important}
.vi-hero p{color:var(--vi-muted);font-size:15px;line-height:1.6;margin:0;max-width:56ch}
.vi-stats{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
.vi-stat{background:linear-gradient(180deg,var(--vi-panel-2),var(--vi-panel));border:1px solid var(--vi-line);border-radius:14px;padding:14px 16px}
.vi-stat .v{font-size:26px;font-weight:800;color:var(--vi-text);letter-spacing:-.01em}
.vi-stat .k{font-size:12px;color:var(--vi-muted);margin-top:2px}
@media (max-width:820px){.vi-hero{grid-template-columns:1fr}}
.vi-panel{background:var(--vi-panel)!important;border:1px solid var(--vi-line)!important;border-radius:16px!important;padding:16px!important}
.vi-section-title{font-size:12px;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:var(--vi-muted);margin:0 0 10px}
button.primary,.vi-run{background:linear-gradient(90deg,var(--vi-gold),var(--vi-gold-2))!important;color:#16110a!important;border:none!important;font-weight:700!important;letter-spacing:.02em}
button.primary:hover{filter:brightness(1.06)}
.tab-nav button,[role=tab]{font-weight:600!important}
[role=tab][aria-selected=true]{color:var(--vi-gold)!important;border-color:var(--vi-gold)!important}
.vi-card{background:var(--vi-panel-2);border:1px solid var(--vi-line);border-radius:14px;padding:16px 18px;margin-bottom:12px}
.vi-card .top{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap}
.vi-card .idx{font-size:11px;color:var(--vi-muted);letter-spacing:.14em;text-transform:uppercase}
.vi-card .label{font-size:21px;font-weight:800;color:var(--vi-text);margin-top:2px;letter-spacing:-.01em}
.vi-badge{font-size:12px;font-weight:700;padding:5px 10px;border-radius:999px;border:1px solid;white-space:nowrap}
.vi-badge.full{color:var(--vi-good);border-color:rgba(63,182,139,.45);background:rgba(63,182,139,.10)}
.vi-badge.model{color:var(--vi-warn);border-color:rgba(224,165,58,.45);background:rgba(224,165,58,.10)}
.vi-badge.make{color:var(--vi-low);border-color:rgba(224,112,58,.45);background:rgba(224,112,58,.10)}
.vi-badge.abstain{color:var(--vi-bad);border-color:rgba(224,86,86,.45);background:rgba(224,86,86,.10)}
.vi-note{color:var(--vi-muted);font-size:13px;margin:8px 0 12px}
.vi-chips{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin:10px 0 14px}
.vi-chip{background:var(--vi-panel);border:1px solid var(--vi-line);border-radius:10px;padding:8px 10px}
.vi-chip .k{font-size:11px;color:var(--vi-muted);letter-spacing:.08em;text-transform:uppercase}
.vi-chip .v{font-size:14px;font-weight:600;color:var(--vi-text);margin-top:2px;overflow-wrap:anywhere}
.vi-meter{display:flex;align-items:center;gap:10px;margin:6px 0}
.vi-meter .k{width:92px;font-size:12px;color:var(--vi-muted)}
.vi-meter .bar{display:block;flex:1;height:8px;background:var(--vi-panel);border:1px solid var(--vi-line);border-radius:999px;overflow:hidden}
.vi-meter .fill{display:block;height:100%;border-radius:999px;background:linear-gradient(90deg,var(--vi-gold),var(--vi-gold-2))}
.vi-meter .fill.det{background:linear-gradient(90deg,#5b8def,#8fb1f5)}
.vi-meter .v{width:56px;text-align:right;font-variant-numeric:tabular-nums;font-size:13px;color:var(--vi-text)}
.vi-card details{margin-top:10px;color:var(--vi-muted)} .vi-card summary{cursor:pointer;font-size:13px}
.vi-card table{width:100%;margin-top:8px;font-size:13px;border-collapse:collapse}
.vi-card td{padding:5px 0;border-top:1px solid var(--vi-line);color:var(--vi-text)} .vi-card td.p{text-align:right;color:var(--vi-muted);font-variant-numeric:tabular-nums}
.vi-empty{border:1px dashed var(--vi-line);border-radius:14px;padding:22px;text-align:center;color:var(--vi-muted);font-size:14px}
.vi-empty b{color:var(--vi-text)}
.vi-foot{margin-top:30px;padding-top:16px;border-top:1px solid var(--vi-line);display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;color:var(--vi-muted);font-size:12px}
@media (max-width:560px){.vi-chips{grid-template-columns:1fr}.vi-meter .k{width:76px}}
"""


def header_html() -> str:
    mark = EAGLE_SVG.replace("{uid}", "h").format(cls="vi-mark")
    return f"""<div class="vi-header"><div class="vi-brand">{mark}<div><div class="vi-name">VEHICLE INTELLIGENCE</div>
<div class="vi-tag">P5 · Team T10 · Fine-grained recognition</div></div></div>
<div class="vi-pills"><span class="vi-pill">196 classes</span><span class="vi-pill">CPU · ONNX Runtime</span><span class="vi-pill">Research demo</span></div></div>"""


def hero_html(stats: list[tuple[str, str]], test_images: str | None) -> str:
    tiles = "".join(f'<div class="vi-stat"><div class="v">{html.escape(v)}</div><div class="k">{html.escape(k)}</div></div>' for v, k in stats)
    measured = f"Measured on {html.escape(test_images)} unseen test images." if test_images else ""
    return f"""<div class="vi-hero"><div><h1>Identify any car's <span>make, model and year</span></h1>
<p>Upload a photo. A YOLO11 detector finds the vehicle, ConvNeXt V2 classifies it across 196 Stanford Cars classes, and classical
computer vision checks image quality. When the system is unsure, it says so. {measured}</p></div>
<div class="vi-stats">{tiles}</div></div>"""


LEVEL_BADGE = {"full": "Confident", "model": "Year uncertain", "make": "Make only · low confidence", "abstain": "Not recognised"}


def vehicle_card(vehicle: dict, index: int, level_text: dict) -> str:
    make = vehicle["make"] or "unknown"
    model = vehicle["model"] or "uncertain"
    year = vehicle["year"] if vehicle["year"] is not None else "uncertain"
    conf = vehicle["confidence"]
    det = vehicle["detector_score"]
    det_meter = (f'<div class="vi-meter"><span class="k">Detector</span><span class="bar"><span class="fill det" style="width:{det:.0%}"></span></span>'
                 f'<span class="v">{det:.0%}</span></div>') if det is not None else \
        '<div class="vi-meter"><span class="k">Detector</span><span class="v" style="width:auto">n/a (contour region)</span></div>'
    rows = "".join(f"<tr><td>{html.escape(item['name'])}</td><td class='p'>{item['probability']:.1%}</td></tr>" for item in vehicle["top"])
    level = vehicle["level"]
    return f"""<div class="vi-card"><div class="top"><div><div class="idx">Vehicle {index + 1}</div>
<div class="label">{html.escape(vehicle['label'])}</div></div><span class="vi-badge {level}">{LEVEL_BADGE[level]}</span></div>
<div class="vi-note">{html.escape(level_text[level])}</div>
<div class="vi-chips"><div class="vi-chip"><div class="k">Make</div><div class="v">{html.escape(make)}</div></div>
<div class="vi-chip"><div class="k">Model</div><div class="v">{html.escape(model)}</div></div>
<div class="vi-chip"><div class="k">Year</div><div class="v">{html.escape(str(year))}</div></div></div>
<div class="vi-meter"><span class="k">Confidence</span><span class="bar"><span class="fill" style="width:{conf:.0%}"></span></span><span class="v">{conf:.1%}</span></div>
{det_meter}
<details><summary>Closest matches</summary><table>{rows}</table></details></div>"""


def empty_state(message: str) -> str:
    return f'<div class="vi-empty"><b>{html.escape(message)}</b></div>'


PLACEHOLDER = '<div class="vi-empty">Results appear here. Upload a photo or pick an example, then press <b>Identify vehicle</b>.</div>'

FOOTER = """<div class="vi-foot"><span>Research demo · Stanford Cars (196 classes) · Not for surveillance, enforcement or insurance decisions</span>
<span>P5 · Team T10</span></div>"""
