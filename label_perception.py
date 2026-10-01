"""Perception validation tool.

Runs perception over a frame sequence, then serves a review UI where you
rate each class per frame. Computes per-class PRECISION live so we can
certify the 95% target and see exactly which class needs work.

Rubric per frame (one keypress each, fast):
  self  : is the green blob our real snake?           [g]ood / [b]ad
  enemy : how many shown heads are WRONG?             number key 0-9
  food  : are the cyan dots really food?              [g]/[b]
  loot  : are yellow rings really loot piles?         [g]/[b] / [n]a
  border: is the red border correct?                  [g]/[b] / [n]a
Then [Enter]/click Next. Labels persist to <out>/labels.json.

Precision(self)   = frames self-good / frames judged
Precision(enemy)  = (heads_shown - heads_wrong) / heads_shown
Precision(food/loot/border) = good / (good+bad)

Usage:
  python label_perception.py <frames_dir> [--start N] [--count M]
                             [--stride S] [--port 8778]
"""
import argparse
import base64
import glob
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

import perception

STATE = {"items": [], "labels": {}, "labels_path": None}


def build(src, start, count, stride):
    files = sorted(glob.glob(os.path.join(src, "*.jpg"))) or \
        sorted(glob.glob(os.path.join(src, "*.png")))
    files = files[start:start + count * stride:stride]
    per = perception.Perception()
    items = []
    for fp in files:
        img = cv2.imread(fp)
        if img is None:
            continue
        if perception.classify_state(img) != perception.STATE_PLAYING:
            per.reset()
            continue
        sc = per.process(img)
        ovl = perception.render(sc, img)
        raw = cv2.resize(img, (sc.w, sc.h))
        items.append({
            "name": os.path.basename(fp),
            "raw": _b64(raw), "ovl": _b64(ovl),
            "self_conf": sc.self_conf,
            "n_enemy": len(sc.enemies), "n_food": len(sc.food),
            "n_loot": len(sc.loot),
        })
    return items


def _b64(img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def metrics():
    labs = STATE["labels"]
    items = STATE["items"]
    n = sf = sg = 0
    heads = hw = 0
    fg = fb = lg = lb = bg = bb = 0
    for i, it in enumerate(items):
        lb_ = labs.get(str(i))
        if not lb_:
            continue
        n += 1
        if lb_.get("self") == "g":
            sg += 1
            sf += 1
        elif lb_.get("self") == "b":
            sf += 1
        heads += it["n_enemy"]
        hw += int(lb_.get("enemy_wrong", 0) or 0)
        if lb_.get("food") == "g":
            fg += 1
        elif lb_.get("food") == "b":
            fb += 1
        if lb_.get("loot") == "g":
            lg += 1
        elif lb_.get("loot") == "b":
            lb += 1
        if lb_.get("border") == "g":
            bg += 1
        elif lb_.get("border") == "b":
            bb += 1

    def pct(a, b):
        return round(100 * a / b, 1) if b else None
    return {
        "judged": n,
        "self_precision": pct(sg, sf),
        "enemy_head_precision": pct(heads - hw, heads),
        "enemy_heads_total": heads,
        "food_precision": pct(fg, fg + fb),
        "loot_precision": pct(lg, lg + lb),
        "border_precision": pct(bg, bg + bb),
    }


PAGE = """<!doctype html><html><head><meta charset=utf-8>
<title>perception labeling</title><style>
:root{color-scheme:dark}body{margin:0;font:13px Consolas,monospace;background:#0e1319;color:#d6e8f6}
#imgs{display:flex;gap:6px;justify-content:center;padding:8px;flex-wrap:wrap}
img{max-width:49%;border:1px solid #263847;border-radius:6px}
#bar{position:sticky;top:0;background:#0a0f14;padding:8px;border-bottom:1px solid #22303d;display:flex;gap:14px;flex-wrap:wrap;align-items:center}
.grp{border:1px solid #26384a;border-radius:6px;padding:4px 8px}
b{color:#6fd3ff}button{background:#16212c;color:#d6e8f6;border:1px solid #2b3b49;border-radius:4px;padding:4px 9px;margin:2px;cursor:pointer}
button.on{background:#2a6;border-color:#3d8}button.bad{background:#a33;border-color:#d55}
#m span{margin-right:12px}.hint{color:#7fa0b8}
</style></head><body>
<div id=bar>
 <div class=grp>self <button id=sg onclick="set('self','g')">good(g)</button><button id=sb onclick="set('self','b')">bad(b)</button></div>
 <div class=grp>enemy heads wrong: <span id=ew>0</span> <button onclick="dec()">-</button><button onclick="inc()">+</button></div>
 <div class=grp>food <button id=fg onclick="set('food','g')">g</button><button id=fb onclick="set('food','b')">b</button></div>
 <div class=grp>loot <button id=lg onclick="set('loot','g')">g</button><button id=lb onclick="set('loot','b')">b</button><button onclick="set('loot','n')">na</button></div>
 <div class=grp>border <button id=bg onclick="set('border','g')">g</button><button id=bb onclick="set('border','b')">b</button><button onclick="set('border','n')">na</button></div>
 <button onclick="prev()">&#9664; prev</button><button onclick="next()">next &#9654; (Enter)</button>
 <span id=pos></span>
</div>
<div id=m></div>
<div id=imgs><img id=raw><img id=ovl></div>
<div class=hint style="padding:8px">keys: g/b self · 0-9 enemy-wrong · f/F food g/b · l/L loot g/b · Enter next · &larr;/&rarr; nav · auto-good: unset food/border count as good on Next</div>
<script>
let items=[],idx=0,cur={};
async function j(u,o){return(await fetch(u,o)).json()}
async function boot(){const d=await j('/api/items');items=d.items;idx=d.resume||0;show();refresh()}
function show(){const it=items[idx];document.getElementById('raw').src=it.raw;document.getElementById('ovl').src=it.ovl;
 document.getElementById('pos').textContent=`${idx+1}/${items.length}  ${it.name}  self_conf=${it.self_conf} enemy=${it.n_enemy} food=${it.n_food} loot=${it.n_loot}`;
 cur=Object.assign({enemy_wrong:0},window._lab[idx]||{});paint()}
function paint(){document.getElementById('ew').textContent=cur.enemy_wrong||0;
 for(const k of ['self','food','loot','border']){for(const v of ['g','b']){const el=document.getElementById(k[0]==='s'&&v==='g'?'sg':null)}}
 mark('sg',cur.self==='g');mark('sb',cur.self==='b');mark('fg',cur.food==='g');mark('fb',cur.food==='b');
 mark('lg',cur.loot==='g');mark('lb',cur.loot==='b');mark('bg',cur.border==='g');mark('bb',cur.border==='b');}
function mark(id,on){const e=document.getElementById(id);if(e)e.className=on?(id[1]==='b'?'bad on':'on'):''}
function set(k,v){cur[k]=v;paint();save()}
function inc(){cur.enemy_wrong=(cur.enemy_wrong||0)+1;paint();save()}
function dec(){cur.enemy_wrong=Math.max(0,(cur.enemy_wrong||0)-1);paint();save()}
async function save(){window._lab[idx]=cur;await j('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({idx,label:cur})});refresh()}
function next(){if(cur.food===undefined)cur.food='g';if(cur.border===undefined)cur.border='g';if(cur.self===undefined)cur.self='g';save();idx=Math.min(items.length-1,idx+1);show()}
function prev(){idx=Math.max(0,idx-1);show()}
async function refresh(){const m=await j('/api/metrics');document.getElementById('m').innerHTML='<b>PRECISION</b> '+
 Object.entries(m).map(([k,v])=>`<span>${k}: <b>${v==null?'-':v}</b></span>`).join('')}
window._lab={};
document.addEventListener('keydown',e=>{if(e.key==='Enter')next();else if(e.key==='ArrowLeft')prev();else if(e.key==='ArrowRight'){idx=Math.min(items.length-1,idx+1);show()}
 else if(e.key==='g')set('self','g');else if(e.key==='b')set('self','b');
 else if(e.key==='f')set('food','g');else if(e.key==='F')set('food','b');
 else if(e.key==='l')set('loot','g');else if(e.key==='L')set('loot','b');
 else if(/[0-9]/.test(e.key)){cur.enemy_wrong=+e.key;paint();save()}})
boot()
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _j(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/":
            b = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(b)
        elif self.path == "/api/items":
            resume = max([int(k) for k in STATE["labels"]] + [-1]) + 1 \
                if STATE["labels"] else 0
            self._j({"items": STATE["items"], "resume": min(resume,
                     max(0, len(STATE["items"]) - 1))})
        elif self.path == "/api/metrics":
            self._j(metrics())
        else:
            self._j({"err": "no route"}, 404)

    def do_POST(self):
        if self.path == "/api/label":
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
            STATE["labels"][str(data["idx"])] = data["label"]
            with open(STATE["labels_path"], "w", encoding="utf-8") as f:
                json.dump(STATE["labels"], f, indent=1)
            self._j({"ok": True})
        else:
            self._j({"err": "no route"}, 404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=120)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--port", type=int, default=8778)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "labelset")
    os.makedirs(out, exist_ok=True)
    STATE["labels_path"] = os.path.join(out, "labels.json")
    if os.path.isfile(STATE["labels_path"]):
        STATE["labels"] = json.load(open(STATE["labels_path"],
                                         encoding="utf-8"))
    print(f"processing {args.src} (start={args.start} count={args.count} "
          f"stride={args.stride}) ...")
    STATE["items"] = build(args.src, args.start, args.count, args.stride)
    print(f"{len(STATE['items'])} playing frames ready")
    print(f"open http://127.0.0.1:{args.port}  (labels -> {STATE['labels_path']})")
    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
