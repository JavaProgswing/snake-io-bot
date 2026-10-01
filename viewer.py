"""Snake-bot mission control: replay browser + live status UI.

Serves a single-page app on http://127.0.0.1:8777 showing every death
replay (scrubbable, with HUD overlays), calibration snapshots, current
strategy config/tuner params, life-duration stats and the live bot log.

Run:  python viewer.py   (or the SnakeBotViewer.bat on the Desktop)
"""
import glob
import json
import os
import re
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = 8777
SCRATCH_ROOTS = [
    HERE,
    os.environ.get("SNAKE_SCRATCH", HERE),
]
TASK_ROOTS = [os.path.join(os.path.dirname(r), "tasks")
              for r in SCRATCH_ROOTS]


def runs():
    out = []
    for root in SCRATCH_ROOTS:
        for d in sorted(glob.glob(os.path.join(root, "debug*"))):
            if os.path.isdir(d):
                out.append({"name": os.path.basename(d), "path": d,
                            "mtime": os.path.getmtime(d)})
    out.sort(key=lambda r: r["mtime"])
    return out


def run_dir(name):
    for r in runs():
        if r["name"] == name:
            return r["path"]
    return None


def deaths(run):
    d = run_dir(run)
    if not d:
        return {}
    groups = {}
    for f in sorted(os.listdir(d)):
        if not f.endswith(".png"):
            continue
        m = re.match(r"death(\d+)_raw_", f)
        if m:
            groups.setdefault(int(m.group(1)), []).append(f)
            continue
        # Legacy deaths were baked overlay frames named deathNNN_t-...
        m = re.match(r"death(\d+)_t-", f)
        if m:
            groups.setdefault(int(m.group(1)), []).append(f)
    # order frames oldest -> death: t-minus descending
    for k, fs in groups.items():
        def key(f):
            m = re.search(r"t-(\d+\.\d)s", f)
            return -float(m.group(1)) if m else 0.0
        fs.sort(key=lambda f: (key(f), f))
    return {str(k): v for k, v in sorted(groups.items())}


def _death_id_from_name(name):
    m = re.match(r"death(\d+)_", name)
    return int(m.group(1)) if m else None


def _frame_tminus(name):
    m = re.search(r"t-(\d+\.\d)s", name)
    return float(m.group(1)) if m else 0.0


def snaps(run):
    d = run_dir(run)
    if not d:
        return []
    return [f for f in sorted(os.listdir(d))
            if f.endswith(".png") and not f.startswith("death")]


def death_meta(run, death_id):
    d = run_dir(run)
    if not d:
        return {}
    try:
        n = int(death_id)
    except (TypeError, ValueError):
        return {}
    fp = os.path.join(d, f"death{n:03d}_meta.json")
    if not os.path.isfile(fp):
        fp = os.path.join(d, f"death{n}_meta.json")
    try:
        with open(fp, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def death_summaries(run):
    groups = deaths(run)
    out = []
    for key, fs in groups.items():
        meta = death_meta(run, key)
        frames = meta.get("frames", [])
        scores = [f.get("score", 0) for f in frames
                  if isinstance(f.get("score", 0), (int, float))]
        min_es = [f.get("min_enemy_px", 9999) for f in frames
                  if isinstance(f.get("min_enemy_px", 9999), (int, float))]
        modes = [str(f.get("mode", "")) for f in frames if f.get("mode")]
        own = [f.get("own_area", 0) for f in frames
               if isinstance(f.get("own_area", 0), (int, float))]
        duration = max((_frame_tminus(f) for f in fs), default=0.0)
        last = frames[-1] if frames else {}
        mode_counts = {}
        for m in modes:
            mode_counts[m] = mode_counts.get(m, 0) + 1
        dominant = max(mode_counts, key=mode_counts.get) if mode_counts else ""
        out.append({
            "id": int(key),
            "frames": len(fs),
            "duration": round(duration, 1),
            "max_score": int(max(scores) if scores else 0),
            "min_enemy": round(min(min_es) if min_es else 0, 1),
            "last_mode": str(last.get("mode", "")),
            "dominant_mode": dominant,
            "own_min": int(min(own) if own else 0),
            "own_last": int(last.get("own_area", 0) or 0),
        })
    out.sort(key=lambda r: r["id"])
    return out


def latest_log(lines=80):
    cands = []
    for root in TASK_ROOTS:
        cands.extend(glob.glob(os.path.join(root, "*.output")))
    if not cands:
        return []
    newest = max(cands, key=os.path.getmtime)
    try:
        with open(newest, encoding="utf-8", errors="replace") as f:
            return [os.path.basename(newest)] + f.readlines()[-lines:]
    except OSError:
        return []


def status():
    st = {"config": {}, "params": {}, "lives": []}
    try:
        with open(os.path.join(HERE, "config.json"), encoding="utf-8") as f:
            st["config"] = json.load(f)
    except OSError:
        pass
    try:
        with open(os.path.join(HERE, "learn.json"), encoding="utf-8") as f:
            learn = json.load(f)
        st["params"] = learn.get("params", {})
        st["lives"] = learn.get("history", [])[-60:]
    except (OSError, json.JSONDecodeError):
        pass
    return st


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>snake-bot review console</title><style>
:root{color-scheme:dark;--bg:#0e1318;--panel:#0a0f14;--panel2:#101821;--line:#21303d;--ink:#d3e7f7;--muted:#86a7bd;--accent:#6fd3ff;--bad:#ff5d5d;--warn:#ffc857;--good:#52e182}
*{box-sizing:border-box}body{margin:0;font:13px/1.45 Consolas,monospace;background:var(--bg);color:var(--ink);display:grid;grid-template-columns:300px minmax(540px,1fr) 380px;height:100vh;overflow:hidden}
h3{margin:12px 10px 6px;color:var(--accent);font-size:12px;text-transform:uppercase;letter-spacing:.08em}
#side,#right{overflow-y:auto;background:var(--panel);border-color:var(--line)}
#side{border-right:1px solid var(--line)}#right{border-left:1px solid var(--line)}
#main{display:flex;flex-direction:column;align-items:center;gap:8px;overflow:auto;padding:10px;background:#10151b}
button,select,input[type=text]{background:#15202a;color:var(--ink);border:1px solid #2b3b49;border-radius:4px;padding:4px 8px;margin:2px;cursor:pointer;font:inherit}
button:hover,select:hover{background:#203140}.sel{background:#244b62!important;border-color:var(--accent)!important}
.row{display:flex;gap:6px;align-items:center;flex-wrap:wrap}.sidePad{padding:0 8px}
.death{display:block;width:calc(100% - 12px);text-align:left;margin:4px 6px;padding:7px 8px;background:#111a22;border:1px solid #1f2d38;border-radius:6px}
.death small{color:var(--muted)}.death .top{display:flex;justify-content:space-between;gap:8px}.death .bad{color:var(--bad)}.death .warn{color:var(--warn)}.death .good{color:var(--good)}
.snap{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.chip{display:inline-flex;align-items:center;gap:4px;background:#0c151d;border:1px solid #263847;border-radius:4px;padding:3px 7px;margin:2px;color:#bcd8eb}
.chip b{color:#fff}.danger{border-color:#5c2930;color:#ffb1b1}.warning{border-color:#5a4a20;color:#ffe19a}.ok{border-color:#245039;color:#a9f4bf}
#frameWrap{width:100%;display:flex;justify-content:center;align-items:center;min-height:320px}
#frame{max-width:100%;border:1px solid #2a3a4a;border-radius:6px;image-rendering:pixelated;background:#06090c}
#bar{width:100%;max-width:1040px;margin:2px 0}.label{color:var(--muted)}#fname{color:#8fb8d8;margin:0 4px;text-align:center}
pre{white-space:pre-wrap;margin:4px 8px;font-size:11px;color:#9fb9cf}.stat{margin:2px 10px}.k{color:var(--accent)}
#legend span{display:inline-block;margin:2px 4px;padding:2px 7px;border-radius:4px;font-size:11px;color:#071017}
#overlayCtl,#toolbar,#chips{display:flex;flex-wrap:wrap;justify-content:center;gap:6px;max-width:1040px}
#overlayCtl label{display:inline-flex;align-items:center;gap:4px;background:#0d151d;border:1px solid #203040;border-radius:4px;padding:3px 6px}
#overlayCtl input[type=color]{width:28px;height:20px;border:0;background:transparent;padding:0}#overlayCtl input[type=range]{width:90px}
#timeline{width:100%;max-width:1040px;height:42px;border:1px solid #243542;border-radius:5px;background:#06090d;cursor:pointer}
#frameMeta{width:100%;max-width:1040px;min-height:72px;background:#070b10;border:1px solid #203040;border-radius:5px;padding:7px;color:#b8d7ef}
#runInfo{color:var(--muted);padding:0 10px 8px}.muted{color:var(--muted)}
@media(max-width:1050px){body{grid-template-columns:250px 1fr}#right{display:none}}
</style></head><body>
<div id="side">
 <h3>Runs</h3>
 <div class="sidePad row"><select id="runSel" onchange="loadRun(false)"></select><button onclick="reloadRuns()">refresh</button></div>
 <div id="runInfo"></div>
 <h3>Deaths</h3>
 <div class="sidePad row"><button onclick="showLatestDeath()">latest</button><label class="chip"><input id="autoLatest" type="checkbox"> auto latest</label></div>
 <div id="deaths"></div>
 <h3>Snapshots</h3><div id="snaps"></div>
</div>
<div id="main">
 <div id="liveBar" class="row" style="justify-content:center;width:100%;max-width:1040px">
  <button id="liveBtn" onclick="toggleLive()">&#9679; watch LIVE</button>
  <label class="chip">view <select id="liveMode"><option value="split" selected>raw | overlay</option><option value="raw">raw</option><option value="overlay">overlay</option></select></label>
  <span class="chip" id="liveInfo">live: off</span>
  <span class="chip warning" id="nudgeInfo" style="display:none">NUDGING — release mouse to stop, hold Shift = boost</span>
 </div>
 <div id="liveWrap" style="display:none;width:100%;max-width:1040px">
  <canvas id="live" style="width:100%;border:1px solid #2a3a4a;border-radius:6px;background:#06090c;cursor:crosshair"></canvas>
  <div class="muted" style="text-align:center">drag on the frame to nudge the snake toward your cursor (hold Shift while dragging to boost); nudged lives are excluded from learning</div>
 </div>
 <div id="legend">
  <span style="background:#ff4c4c">obstacle</span>
  <span style="background:#ff9a22">danger memory</span>
  <span style="background:#3dff5f">self</span>
  <span style="background:#2eefff">food</span>
  <span style="background:#f3e64b">loot</span>
  <span style="background:#ff5bff;color:#fff">target</span>
 </div>
 <div id="chips"></div>
 <div id="frameWrap"><canvas id="frame"></canvas></div>
 <canvas id="timeline" onclick="timelineSeek(event)"></canvas>
 <input id="bar" type="range" min="0" max="0" value="0" oninput="seek(this.value)">
 <div id="toolbar">
  <button onclick="play()">play</button><button onclick="pause()">pause</button>
  <button onclick="step(-1)">prev</button><button onclick="step(1)">next</button>
  <label class="chip">view <select id="viewMode" onchange="show()"><option value="raw">raw + custom overlay</option><option value="baked">baked debug</option><option value="split">split raw/debug</option><option value="layer">layer only</option></select></label>
  <label class="chip">speed <select id="fps"><option>2</option><option selected>5</option><option>10</option><option>15</option></select> fps</label>
 </div>
 <div id="overlayCtl">
  <label><input id="showOverlay" type="checkbox" checked onchange="show()"> overlay</label>
  <label>opacity <input id="alpha" type="range" min="0" max="100" value="62" oninput="show()"></label>
  <label>obstacle <input data-code="1" type="color" value="#ff2f2f" oninput="show()"></label>
  <label>danger <input data-code="2" type="color" value="#ff9a22" oninput="show()"></label>
  <label>self <input data-code="3" type="color" value="#3dff5f" oninput="show()"></label>
  <label>food <input data-code="4" type="color" value="#2eefff" oninput="show()"></label>
  <label>loot <input data-code="5" type="color" value="#f3e64b" oninput="show()"></label>
  <label>target <input data-code="6" type="color" value="#ff5bff" oninput="show()"></label>
 </div>
 <div id="fname"></div>
 <pre id="frameMeta"></pre>
</div>
<div id="right">
 <h3>Life Stats</h3><div id="stats"></div>
 <h3>Selected Death</h3><pre id="deathStats"></pre>
 <h3>Tuned Params</h3><pre id="params"></pre>
 <h3>Strategy Config</h3><pre id="config"></pre>
 <h3>Live Log</h3><pre id="log"></pre>
</div>
<script>
let frames=[],idx=0,timer=null,curRun="",metaFrames=[],deathMap={},summaryMap={},selectedDeath=null,lastDeathCount=0;
const imgCache=new Map();
async function j(u){return (await fetch(u,{cache:'no-store'})).json()}
function enc(s){return encodeURIComponent(s)}
function img(url){if(imgCache.has(url))return imgCache.get(url);const p=new Promise((res,rej)=>{const im=new Image();im.onload=()=>res(im);im.onerror=rej;im.src=url});imgCache.set(url,p);return p}
function hex(v){v=v.replace('#','');return [parseInt(v.slice(0,2),16),parseInt(v.slice(2,4),16),parseInt(v.slice(4,6),16)]}
function layerName(f){return f&&f.includes('_raw_')?f.replace('_raw_','_layer_'):null}
function bakedName(f){return f&&f.includes('_raw_')?f.replace('_raw_','_overlay_'):f}
function clsFor(s){if((s.min_enemy||99)<=10||s.own_min===0)return'bad';if((s.min_enemy||99)<=22)return'warn';return'good'}
async function init(){await reloadRuns();refreshStatus();setInterval(refreshStatus,4000);setInterval(autoRefresh,3000);document.addEventListener('keydown',keys)}
async function reloadRuns(){const rs=await j('/api/runs');const sel=document.getElementById('runSel');const prev=sel.value;sel.innerHTML=rs.map(r=>`<option>${r.name}</option>`).join('');sel.value=prev&&rs.some(r=>r.name===prev)?prev:(rs.length?rs[rs.length-1].name:'');await loadRun(false)}
async function loadRun(keep){curRun=document.getElementById('runSel').value;imgCache.clear();deathMap=await j('/api/deaths?run='+enc(curRun));const sums=await j('/api/death_summaries?run='+enc(curRun));summaryMap={};sums.forEach(s=>summaryMap[s.id]=s);lastDeathCount=Object.keys(deathMap).length;renderDeathList();renderRunInfo(sums);renderSnaps();if(!keep)showLatestDeath()}
function renderRunInfo(sums){const max=Math.max(0,...sums.map(s=>s.max_score||0));const med=sums.length?[...sums.map(s=>s.duration)].sort((a,b)=>a-b)[Math.floor(sums.length/2)]:0;document.getElementById('runInfo').innerHTML=`${curRun}<br><span class="muted">${sums.length} deaths, max score ${max}, median replay ${med.toFixed(1)}s</span>`}
function renderDeathList(){document.getElementById('deaths').innerHTML=Object.entries(deathMap).map(([k,v])=>{const s=summaryMap[k]||{};const c=clsFor(s);return `<button class="death ${selectedDeath==k?'sel':''}" id="d${k}" onclick='showDeath(${k})'><div class="top"><b>death ${k}</b><span class="${c}">minE ${s.min_enemy??'?'} px</span></div><small>${v.length}f ${s.duration??0}s score ${s.max_score??0} mode ${s.last_mode||s.dominant_mode||'?'}</small></button>`}).join('')||'<div class="stat muted">No deaths in this run yet.</div>'}
async function renderSnaps(){const sn=await j('/api/snaps?run='+enc(curRun));document.getElementById('snaps').innerHTML=sn.slice(-80).reverse().map(f=>`<button class="death snap" onclick='showOne("${f}")'>${f}</button>`).join('')}
async function showDeath(k){pause();selectedDeath=String(k);renderDeathList();frames=deathMap[selectedDeath]||[];idx=0;metaFrames=[];try{const m=await j('/api/meta?run='+enc(curRun)+'&death='+k);metaFrames=m.frames||[]}catch(e){}document.getElementById('bar').max=Math.max(0,frames.length-1);document.getElementById('deathStats').textContent=JSON.stringify(summaryMap[selectedDeath]||{},null,1);drawTimeline();show()}
function showLatestDeath(){const ks=Object.keys(deathMap).map(Number).sort((a,b)=>a-b);if(ks.length)showDeath(ks[ks.length-1])}
function showOne(f){pause();selectedDeath=null;frames=[f];metaFrames=[];idx=0;document.getElementById('bar').max=0;document.getElementById('deathStats').textContent='snapshot';drawTimeline();show()}
function colors(){const out={};document.querySelectorAll('input[type=color][data-code]').forEach(el=>out[+el.dataset.code]=hex(el.value));return out}
async function colorLayer(layerFile, alpha=255, black=false){const lay=await img('/img?run='+enc(curRun)+'&f='+enc(layerFile));const off=document.createElement('canvas');off.width=lay.naturalWidth;off.height=lay.naturalHeight;const ox=off.getContext('2d');if(black){ox.fillStyle='#020508';ox.fillRect(0,0,off.width,off.height)}ox.drawImage(lay,0,0);const data=ox.getImageData(0,0,off.width,off.height);const cs=colors();for(let i=0;i<data.data.length;i+=4){const code=data.data[i],c=cs[code];if(c){data.data[i]=c[0];data.data[i+1]=c[1];data.data[i+2]=c[2];data.data[i+3]=alpha}else if(!black){data.data[i+3]=0}}ox.putImageData(data,0,0);return off}
async function drawBase(ctx,f,x,y,w,h,mode){if(mode==='baked'){ctx.drawImage(await img('/img?run='+enc(curRun)+'&f='+enc(bakedName(f))),x,y,w,h);return}ctx.drawImage(await img('/img?run='+enc(curRun)+'&f='+enc(f)),x,y,w,h);const lf=layerName(f);if(lf&&document.getElementById('showOverlay').checked){try{const alpha=Math.round(+document.getElementById('alpha').value*2.55);ctx.drawImage(await colorLayer(lf,alpha,false),x,y,w,h)}catch(e){}}}
async function show(){if(!frames.length)return;const f=frames[idx],mode=document.getElementById('viewMode').value,canvas=document.getElementById('frame'),ctx=canvas.getContext('2d');const base=await img('/img?run='+enc(curRun)+'&f='+enc(mode==='baked'?bakedName(f):f));const bw=base.naturalWidth,bh=base.naturalHeight;canvas.width=mode==='split'?bw*2:bw;canvas.height=bh;ctx.clearRect(0,0,canvas.width,canvas.height);ctx.imageSmoothingEnabled=false;if(mode==='split'){ctx.drawImage(await img('/img?run='+enc(curRun)+'&f='+enc(f)),0,0,bw,bh);await drawBase(ctx,f,bw,0,bw,bh,'raw')}else if(mode==='layer'&&layerName(f)){ctx.drawImage(await colorLayer(layerName(f),255,true),0,0,bw,bh)}else{await drawBase(ctx,f,0,0,bw,bh,mode)}document.getElementById('bar').value=idx;const mf=metaFrames[idx]||{};renderChips(mf);document.getElementById('fname').textContent=`${idx+1}/${frames.length} ${f}`;document.getElementById('frameMeta').textContent=Object.keys(mf).length?JSON.stringify(mf,null,1):'';drawTimeline()}
function renderChips(m){const c=document.getElementById('chips');if(!Object.keys(m).length){c.innerHTML='';return}const danger=(m.min_enemy_px||99)<=10?'danger':(m.min_enemy_px||99)<=22?'warning':'ok';c.innerHTML=`<span class="chip"><b>${m.mode||'?'}</b></span><span class="chip ${danger}">minE <b>${m.min_enemy_px??'?'}</b></span><span class="chip">score <b>${m.score??0}</b></span><span class="chip">own <b>${m.own_area??0}</b></span><span class="chip">food <b>${m.food??0}</b></span><span class="chip">loot <b>${m.loot??0}</b></span><span class="chip">heads <b>${m.enemy_heads??0}</b></span>`}
function drawTimeline(){const c=document.getElementById('timeline'),ctx=c.getContext('2d');const w=c.clientWidth||900,h=c.clientHeight||42;c.width=w;c.height=h;ctx.fillStyle='#05080c';ctx.fillRect(0,0,w,h);if(!frames.length)return;for(let i=0;i<frames.length;i++){const m=metaFrames[i]||{},min=m.min_enemy_px??50;ctx.fillStyle=min<=10?'#ff5d5d':min<=22?'#ffc857':'#52e182';const x=Math.floor(i*w/frames.length),x2=Math.ceil((i+1)*w/frames.length);ctx.fillRect(x,8,Math.max(1,x2-x),24);if(m.boost){ctx.fillStyle='#6fd3ff';ctx.fillRect(x,4,Math.max(1,x2-x),4)}}ctx.strokeStyle='#fff';const px=(idx+.5)*w/frames.length;ctx.beginPath();ctx.moveTo(px,0);ctx.lineTo(px,h);ctx.stroke()}
function timelineSeek(ev){if(!frames.length)return;const r=ev.currentTarget.getBoundingClientRect();idx=Math.max(0,Math.min(frames.length-1,Math.floor((ev.clientX-r.left)/r.width*frames.length)));pause();show()}
function seek(v){pause();idx=+v;show()}function step(d){pause();idx=Math.min(frames.length-1,Math.max(0,idx+d));show()}
function play(){pause();const fps=+document.getElementById('fps').value;timer=setInterval(()=>{idx=(idx+1)%frames.length;show()},1000/fps)}
function pause(){if(timer){clearInterval(timer);timer=null}}
function keys(e){if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT')return;if(e.key===' '){e.preventDefault();timer?pause():play()}else if(e.key==='ArrowLeft')step(-1);else if(e.key==='ArrowRight')step(1);else if(e.key==='Home'){idx=0;show()}else if(e.key==='End'){idx=frames.length-1;show()}}
async function autoRefresh(){if(!curRun)return;const old=lastDeathCount;await loadRun(true);if(document.getElementById('autoLatest').checked&&lastDeathCount!==old)showLatestDeath()}
/* ---------- LIVE view + nudge ---------- */
let liveTimer=null,nudging=false,lastNudgeSent=0;
function toggleLive(){liveTimer?stopLive():startLive()}
function startLive(){document.getElementById('liveWrap').style.display='block';document.getElementById('liveBtn').classList.add('sel');liveTick();liveTimer=setInterval(liveTick,160)}
function stopLive(){clearInterval(liveTimer);liveTimer=null;document.getElementById('liveWrap').style.display='none';document.getElementById('liveBtn').classList.remove('sel');document.getElementById('liveInfo').textContent='live: off'}
function loadImg(u){return new Promise((res)=>{const im=new Image();im.onload=()=>res(im);im.onerror=()=>res(null);im.src=u})}
async function liveTick(){const ts=Date.now();const mode=document.getElementById('liveMode').value;const wantRaw=mode!=='overlay',wantOv=mode!=='raw';const[raw,ov,meta]=await Promise.all([wantRaw?loadImg('/live/raw.jpg?'+ts):null,wantOv?loadImg('/live/overlay.jpg?'+ts):null,fetch('/live/meta.json?'+ts,{cache:'no-store'}).then(r=>r.ok?r.json():null).catch(()=>null)]);const c=document.getElementById('live'),ctx=c.getContext('2d');const base=raw||ov;if(!base)return;const bw=base.naturalWidth,bh=base.naturalHeight;
 if(mode==='split'&&raw&&ov){c.width=bw*2;c.height=bh;ctx.imageSmoothingEnabled=false;ctx.drawImage(raw,0,0,bw,bh);ctx.drawImage(ov,bw,0,bw,bh)}else{c.width=bw;c.height=bh;ctx.imageSmoothingEnabled=false;ctx.drawImage(mode==='overlay'&&ov?ov:base,0,0,bw,bh)}
 if(meta){const age=(Date.now()/1000-meta.t).toFixed(1);const cls=`self ${meta.self_area??'?'} &middot; <span style="color:#ff6b6b">enemy ${meta.enemies??'?'}</span> &middot; <span style="color:#2eefff">food ${meta.food??'?'}</span> &middot; <span style="color:#f3e64b">loot ${meta.loot??'?'}</span>`;const mE=meta.min_enemy_px,mc=mE<20?'#ff5d5d':mE<40?'#ffc857':'#52e182';document.getElementById('liveInfo').innerHTML=`game ${meta.game} &middot; ${meta.life_s}s &middot; <b>${meta.mode}</b> &middot; <span style="color:${mc}">minE ${mE}</span> &middot; ${cls} &middot; ${meta.boost?'<b>BOOST</b> &middot; ':''}${meta.nudged?'<b style="color:#ffc857">NUDGING</b> &middot; ':''}lag ${age}s`}}
const liveC=document.getElementById('live');
function sendNudge(ev){const now=Date.now();if(now-lastNudgeSent<120)return;lastNudgeSent=now;const r=liveC.getBoundingClientRect();const mode=document.getElementById('liveMode').value;let fx=(ev.clientX-r.left)/r.width,fy=(ev.clientY-r.top)/r.height;if(mode==='split'){fx=(fx%0.5)*2}const ang=Math.atan2(fy-0.5,fx-0.5)*180/Math.PI;fetch('/api/nudge',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({angle:ang,boost:ev.shiftKey?1:0,ttl:1.2})})}
liveC.addEventListener('mousedown',e=>{nudging=true;document.getElementById('nudgeInfo').style.display='inline-flex';sendNudge(e)});
window.addEventListener('mousemove',e=>{if(nudging)sendNudge(e)});
window.addEventListener('mouseup',()=>{nudging=false;document.getElementById('nudgeInfo').style.display='none'});
async function refreshStatus(){const st=await j('/api/status');document.getElementById('params').textContent=JSON.stringify(st.params,null,1);document.getElementById('config').textContent=JSON.stringify(st.config.strategy||{},null,1);const ds=st.lives.map(l=>l.dur);if(ds.length){const s=[...ds].sort((a,b)=>a-b),med=s[Math.floor(s.length/2)],mx=Math.max(...ds);const modes={};st.lives.forEach(l=>modes[l.mode]=(modes[l.mode]||0)+1);document.getElementById('stats').innerHTML=`<div class="stat"><span class="k">lives:</span> ${ds.length}</div><div class="stat"><span class="k">median:</span> ${med.toFixed(1)}s</div><div class="stat"><span class="k">max:</span> ${mx.toFixed(1)}s</div><div class="stat"><span class="k">death modes:</span> ${Object.entries(modes).map(([m,c])=>m+':'+c).join(' ')}</div>`}const lg=await j('/api/log');document.getElementById('log').textContent=lg.join('')}
init();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body if isinstance(body, bytes)
                         else json.dumps(body).encode())

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif u.path == "/api/runs":
            self._send(200, runs())
        elif u.path == "/api/deaths":
            self._send(200, deaths(q.get("run", [""])[0]))
        elif u.path == "/api/death_summaries":
            self._send(200, death_summaries(q.get("run", [""])[0]))
        elif u.path == "/api/snaps":
            self._send(200, snaps(q.get("run", [""])[0]))
        elif u.path == "/api/meta":
            self._send(200, death_meta(q.get("run", [""])[0],
                                       q.get("death", [""])[0]))
        elif u.path == "/api/status":
            self._send(200, status())
        elif u.path == "/api/log":
            self._send(200, latest_log())
        elif u.path.startswith("/live/"):
            f = os.path.basename(u.path)
            fp = os.path.join(HERE, "live", f)
            if os.path.isfile(fp):
                ctype = ("image/jpeg" if f.endswith(".jpg")
                         else "application/json")
                with open(fp, "rb") as fh:
                    body = fh.read()
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            else:
                self._send(404, {"err": "no live data"})
        elif u.path == "/img":
            d = run_dir(q.get("run", [""])[0])
            f = os.path.basename(q.get("f", [""])[0])   # no path escape
            fp = os.path.join(d, f) if d else None
            if fp and os.path.isfile(fp):
                with open(fp, "rb") as fh:
                    self._send(200, fh.read(), "image/png")
            else:
                self._send(404, {"err": "not found"})
        else:
            self._send(404, {"err": "no route"})

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/api/nudge":
            try:
                n = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(n) or b"{}")
                out = {"angle": float(data.get("angle", 0.0)),
                       "boost": int(data.get("boost", 0)),
                       "ttl": min(3.0, float(data.get("ttl", 1.2))),
                       "t": __import__("time").time()}
                live = os.path.join(HERE, "live")
                os.makedirs(live, exist_ok=True)
                tmp = os.path.join(live, "nudge.tmp")
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(out, f)
                os.replace(tmp, os.path.join(live, "nudge.json"))
                self._send(200, {"ok": True})
            except (ValueError, OSError) as e:
                self._send(400, {"err": str(e)})
        else:
            self._send(404, {"err": "no route"})


if __name__ == "__main__":
    print(f"snake-bot mission control -> http://127.0.0.1:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
