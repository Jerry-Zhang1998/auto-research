#!/usr/bin/env python3
"""
Generate HTML visualisations from training logs and test results.

Usage:
    python3 scripts/generate_viz.py \\
        --log-dir  logs/attention-is-all-you-need/run_20260615_143022 \\
        --output-dir outputs/attention-is-all-you-need

Writes:
    outputs/{name}/train.html      loss + metric curves over training
    outputs/{name}/evaluate.html   ROC, PR curve, confusion matrix, metrics table
"""
import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List


# ── Chart CSS / JS shared by both HTML files ─────────────────────────────────

_SHARED_STYLE = """
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0f1117;--surface:#1a1d27;--surface2:#222535;--border:#2e3148;
  --text:#e2e4f0;--muted:#8b8fa8;--accent:#7c6af5;--cyan:#4fc3f7;
  --green:#69f0ae;--amber:#ffb74d;--red:#ef5350;
  --font-sans:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  --font-mono:'JetBrains Mono','Fira Code',monospace;
  --r:8px;
}
body{background:var(--bg);color:var(--text);font-family:var(--font-sans);
  font-size:15px;line-height:1.6;max-width:1100px;margin:0 auto;padding:32px 24px 80px}
h1{font-size:1.5rem;font-weight:700;color:#fff;margin-bottom:6px}
.meta{color:var(--muted);font-size:0.875rem;margin-bottom:28px}
.meta span{margin-right:16px}
.section{margin-bottom:44px}
.section-title{font-size:1rem;font-weight:700;color:var(--cyan);
  text-transform:uppercase;letter-spacing:.08em;margin-bottom:16px;
  padding-bottom:8px;border-bottom:1px solid var(--border)}
.card{background:var(--surface);border:1px solid var(--border);
  border-radius:var(--r);padding:20px 22px;margin-bottom:14px}
.chart-wrap{background:var(--surface);border:1px solid var(--border);
  border-radius:var(--r);padding:16px;margin-bottom:16px}
canvas{display:block;width:100%;height:280px}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:700px){.grid-2{grid-template-columns:1fr}}
table{width:100%;border-collapse:collapse;font-size:0.9rem;margin-top:8px}
th{background:var(--surface2);border:1px solid var(--border);padding:10px 14px;
  text-align:left;color:var(--muted);font-weight:600;font-size:.8rem;
  text-transform:uppercase;letter-spacing:.05em}
td{border:1px solid var(--border);padding:10px 14px;vertical-align:middle}
tr:nth-child(even) td{background:var(--surface)}
.badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:.8rem;font-weight:600}
.metric-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:12px;margin-top:4px}
.metric-card{background:var(--surface2);border:1px solid var(--border);border-radius:var(--r);
  padding:16px 18px;text-align:center}
.metric-card .label{font-size:.75rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:6px}
.metric-card .value{font-size:1.6rem;font-weight:700;font-family:var(--font-mono)}
.metric-card .bar{height:4px;border-radius:2px;background:var(--border);margin-top:10px}
.metric-card .bar-fill{height:100%;border-radius:2px;background:var(--accent);transition:width .4s}
footer{margin-top:60px;padding-top:16px;border-top:1px solid var(--border);
  color:var(--muted);font-size:.8rem;text-align:center}
"""

_CHART_JS = """
const COLORS=['#7c6af5','#4fc3f7','#69f0ae','#ffb74d','#ef5350','#26a69a','#ab47bc'];
function lineChart(id,{title,series,xLabel='Step',yLabel='',xTicks}){
  const el=document.getElementById(id);
  if(!el)return;
  const dpr=window.devicePixelRatio||1;
  const W=el.parentElement.clientWidth-32,H=280;
  el.style.width=W+'px';el.style.height=H+'px';
  el.width=W*dpr;el.height=H*dpr;
  const g=el.getContext('2d');g.scale(dpr,dpr);
  const p={t:36,r:20,b:52,l:70},pw=W-p.l-p.r,ph=H-p.t-p.b;

  g.fillStyle='#1a1d27';g.fillRect(0,0,W,H);

  const vals=series.flatMap(s=>s.y.filter(v=>v!=null&&isFinite(v)));
  if(!vals.length){
    g.fillStyle='#8b8fa8';g.font='14px sans-serif';g.textAlign='center';
    g.fillText('No data',W/2,H/2);return;
  }
  let mn=Math.min(...vals),mx=Math.max(...vals),rng=mx-mn||1;
  mn-=rng*.05;mx+=rng*.05;rng=mx-mn;
  const N=Math.max(...series.map(s=>s.y.length));
  const tx=i=>p.l+(N>1?i/(N-1):0)*pw;
  const ty=v=>p.t+ph*(1-(v-mn)/rng);

  // grid
  for(let i=0;i<=5;i++){
    const v=mn+rng*i/5,y=ty(v);
    g.strokeStyle='#2e3148';g.lineWidth=1;
    g.beginPath();g.moveTo(p.l,y);g.lineTo(p.l+pw,y);g.stroke();
    g.fillStyle='#8b8fa8';g.font='11px monospace';g.textAlign='right';
    g.fillText(v<10?v.toFixed(4):v.toFixed(1),p.l-5,y+4);
  }
  const step=Math.max(1,Math.ceil(N/8));
  for(let i=0;i<N;i+=step){
    const x=tx(i);
    g.strokeStyle='#2e3148';g.lineWidth=1;
    g.beginPath();g.moveTo(x,p.t);g.lineTo(x,p.t+ph);g.stroke();
    g.fillStyle='#8b8fa8';g.font='11px monospace';g.textAlign='center';
    g.fillText(xTicks?xTicks[i]:i,x,p.t+ph+16);
  }

  // axes labels
  g.fillStyle='#e2e4f0';g.font='bold 13px sans-serif';g.textAlign='center';
  g.fillText(title,W/2,20);
  g.fillStyle='#8b8fa8';g.font='11px sans-serif';
  g.fillText(xLabel,p.l+pw/2,H-6);
  if(yLabel){
    g.save();g.translate(12,p.t+ph/2);g.rotate(-Math.PI/2);
    g.textAlign='center';g.fillText(yLabel,0,0);g.restore();
  }

  // lines
  series.forEach((s,si)=>{
    g.strokeStyle=COLORS[si%COLORS.length];g.lineWidth=2;g.lineJoin='round';
    g.beginPath();let first=true;
    s.y.forEach((v,i)=>{
      if(v==null||!isFinite(v)){first=true;return;}
      first?g.moveTo(tx(i),ty(v)):g.lineTo(tx(i),ty(v));first=false;
    });
    g.stroke();
  });

  // border
  g.strokeStyle='#2e3148';g.lineWidth=1;g.strokeRect(p.l,p.t,pw,ph);

  // legend
  g.font='12px sans-serif';
  const lw=series.reduce((a,s)=>a+g.measureText(s.label).width+28,0);
  let lx=p.l+(pw-lw)/2;
  series.forEach((s,si)=>{
    g.fillStyle=COLORS[si%COLORS.length];g.fillRect(lx,p.t+ph+30,12,12);
    g.fillStyle='#e2e4f0';g.textAlign='left';
    g.fillText(s.label,lx+16,p.t+ph+41);
    lx+=g.measureText(s.label).width+30;
  });
}

function scatterChart(id,{title,series,xLabel='',yLabel='',diagonal}){
  const el=document.getElementById(id);if(!el)return;
  const dpr=window.devicePixelRatio||1;
  const W=el.parentElement.clientWidth-32,H=280;
  el.style.width=W+'px';el.style.height=H+'px';
  el.width=W*dpr;el.height=H*dpr;
  const g=el.getContext('2d');g.scale(dpr,dpr);
  const p={t:36,r:20,b:52,l:70},pw=W-p.l-p.r,ph=H-p.t-p.b;

  g.fillStyle='#1a1d27';g.fillRect(0,0,W,H);

  // grid 0..1
  for(let i=0;i<=5;i++){
    const v=i/5,y=p.t+ph*(1-v),x=p.l+pw*v;
    g.strokeStyle='#2e3148';g.lineWidth=1;
    g.beginPath();g.moveTo(p.l,y);g.lineTo(p.l+pw,y);g.stroke();
    g.beginPath();g.moveTo(x,p.t);g.lineTo(x,p.t+ph);g.stroke();
    g.fillStyle='#8b8fa8';g.font='11px monospace';
    g.textAlign='right';g.fillText(v.toFixed(1),p.l-4,y+4);
    g.textAlign='center';g.fillText(v.toFixed(1),x,p.t+ph+16);
  }

  // diagonal
  if(diagonal){
    g.strokeStyle='#2e3148';g.lineWidth=1;g.setLineDash([4,4]);
    g.beginPath();g.moveTo(p.l,p.t+ph);g.lineTo(p.l+pw,p.t);g.stroke();
    g.setLineDash([]);
  }

  // axes labels
  g.fillStyle='#e2e4f0';g.font='bold 13px sans-serif';g.textAlign='center';
  g.fillText(title,W/2,20);
  g.fillStyle='#8b8fa8';g.font='11px sans-serif';
  g.fillText(xLabel,p.l+pw/2,H-6);
  if(yLabel){
    g.save();g.translate(12,p.t+ph/2);g.rotate(-Math.PI/2);
    g.textAlign='center';g.fillText(yLabel,0,0);g.restore();
  }

  // series
  series.forEach((s,si)=>{
    if(!s.x||!s.y||!s.x.length)return;
    g.strokeStyle=COLORS[si%COLORS.length];g.lineWidth=2;g.lineJoin='round';
    g.beginPath();
    s.x.forEach((xv,i)=>{
      const px_=p.l+xv*pw, py_=p.t+ph*(1-s.y[i]);
      i===0?g.moveTo(px_,py_):g.lineTo(px_,py_);
    });
    g.stroke();
  });

  g.strokeStyle='#2e3148';g.lineWidth=1;g.strokeRect(p.l,p.t,pw,ph);

  g.font='12px sans-serif';
  const lw=series.reduce((a,s)=>a+g.measureText(s.label).width+28,0);
  let lx=p.l+(pw-lw)/2;
  series.forEach((s,si)=>{
    g.fillStyle=COLORS[si%COLORS.length];g.fillRect(lx,p.t+ph+30,12,12);
    g.fillStyle='#e2e4f0';g.textAlign='left';
    g.fillText(s.label,lx+16,p.t+ph+41);
    lx+=g.measureText(s.label).width+30;
  });
}
"""

# ── HTML templates ────────────────────────────────────────────────────────────

_TRAIN_HTML = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{paper} — Training</title>
<style>{style}</style></head><body>
<h1>{paper} — Training Curves</h1>
<div class="meta">
  <span>📁 Run: <strong>{run}</strong></span>
  <span>📊 {n_steps} steps · {n_epochs} epochs</span>
  <span>⏱ {date}</span>
</div>

<div class="section">
  <div class="section-title">Loss</div>
  <div class="chart-wrap"><canvas id="loss_chart"></canvas></div>
</div>

<div class="section">
  <div class="section-title">Validation Metrics</div>
  <div class="chart-wrap"><canvas id="metric_chart"></canvas></div>
</div>

<div class="section">
  <div class="section-title">Learning Rate</div>
  <div class="chart-wrap"><canvas id="lr_chart"></canvas></div>
</div>

<div class="section">
  <div class="section-title">Final Summary</div>
  <div class="card">
    <table>
      <tr><th>Metric</th><th>Final Value</th></tr>
      {summary_rows}
    </table>
  </div>
</div>

<footer>auto-research · {date} · <a href="../../analyses/{paper}/innovations.md" style="color:var(--muted)">analysis</a></footer>

<script>
const DATA = {data_json};
{chart_js}
window.addEventListener('load',()=>{{
  lineChart('loss_chart',{{
    title:'Training & Validation Loss',
    series:DATA.loss_series,
    xLabel:'Step',yLabel:'Loss',
    xTicks:DATA.steps
  }});
  lineChart('metric_chart',{{
    title:'Validation Metrics',
    series:DATA.metric_series,
    xLabel:'Step',
    xTicks:DATA.steps
  }});
  lineChart('lr_chart',{{
    title:'Learning Rate Schedule',
    series:[{{label:'lr',y:DATA.lr}}],
    xLabel:'Step',yLabel:'LR',
    xTicks:DATA.steps
  }});
}});
</script></body></html>"""

_EVAL_HTML = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{paper} — Evaluation</title>
<style>{style}</style></head><body>
<h1>{paper} — Evaluation Results</h1>
<div class="meta">
  <span>📁 Run: <strong>{run}</strong></span>
  <span>🔖 Split: <strong>{split}</strong></span>
  <span>📅 {date}</span>
</div>

<div class="section">
  <div class="section-title">Metrics</div>
  <div class="metric-grid" id="metric_cards"></div>
</div>

<div class="section">
  <div class="section-title">ROC &amp; PR Curves</div>
  <div class="grid-2">
    <div class="chart-wrap"><canvas id="roc_chart"></canvas></div>
    <div class="chart-wrap"><canvas id="pr_chart"></canvas></div>
  </div>
</div>

<div class="section" id="cm_section">
  <div class="section-title">Confusion Matrix</div>
  <div class="card"><div id="cm_table"></div></div>
</div>

<footer>auto-research · {date} · <a href="../../analyses/{paper}/innovations.md" style="color:var(--muted)">analysis</a></footer>

<script>
const DATA = {data_json};
{chart_js}

const METRIC_COLORS={{
  auc:'#7c6af5',accuracy:'#4fc3f7',f1:'#69f0ae',
  precision:'#ffb74d',recall:'#ef5350',auc_pr:'#26a69a',
  mse:'#ef5350',rmse:'#ffb74d',mae:'#4fc3f7',r2:'#69f0ae',mape:'#ab47bc'
}};

window.addEventListener('load',()=>{{
  // Metric cards
  const grid=document.getElementById('metric_cards');
  Object.entries(DATA.metrics).forEach(([k,v])=>{{
    const pct=Math.min(Math.max(v,0),1);
    const col=METRIC_COLORS[k]||'#7c6af5';
    grid.innerHTML+=`<div class="metric-card">
      <div class="label">${{k}}</div>
      <div class="value" style="color:${{col}}">${{v.toFixed(4)}}</div>
      <div class="bar"><div class="bar-fill" style="width:${{pct*100}}%;background:${{col}}"></div></div>
    </div>`;
  }});

  // ROC curve
  if(DATA.curves?.roc?.fpr?.length){{
    scatterChart('roc_chart',{{
      title:`ROC Curve  (AUC=${{(DATA.metrics.auc||0).toFixed(4)}})`,
      series:[{{label:'ROC',x:DATA.curves.roc.fpr,y:DATA.curves.roc.tpr}}],
      xLabel:'False Positive Rate',yLabel:'True Positive Rate',diagonal:true
    }});
  }} else {{
    document.getElementById('roc_chart').parentElement.innerHTML=
      '<p style="color:var(--muted);padding:20px;text-align:center">ROC curve unavailable (multi-class or no sklearn)</p>';
  }}

  // PR curve
  if(DATA.curves?.pr?.recall?.length){{
    scatterChart('pr_chart',{{
      title:`PR Curve  (AUC-PR=${{(DATA.metrics.auc_pr||0).toFixed(4)}})`,
      series:[{{label:'PR',x:DATA.curves.pr.recall,y:DATA.curves.pr.precision}}],
      xLabel:'Recall',yLabel:'Precision'
    }});
  }} else {{
    document.getElementById('pr_chart').parentElement.innerHTML=
      '<p style="color:var(--muted);padding:20px;text-align:center">PR curve unavailable</p>';
  }}

  // Confusion matrix
  if(DATA.confusion?.length){{
    const n=DATA.confusion.length;
    const flat=DATA.confusion.flat();
    const mx=Math.max(...flat)||1;
    let html='<table><tr><th>Actual \\ Pred</th>';
    for(let j=0;j<n;j++) html+=`<th>Class ${{j}}</th>`;
    html+='</tr>';
    DATA.confusion.forEach((row,i)=>{{
      html+=`<tr><th style="text-align:left">Class ${{i}}</th>`;
      row.forEach((v,j)=>{{
        const alpha=(v/mx*.6+.1).toFixed(2);
        const col=i===j?`rgba(105,240,174,${{alpha}})`:`rgba(239,83,80,${{alpha}})`;
        html+=`<td style="background:${{col}};text-align:center;font-family:monospace">${{v}}</td>`;
      }});
      html+='</tr>';
    }});
    html+='</table>';
    document.getElementById('cm_table').innerHTML=html;
  }} else {{
    document.getElementById('cm_section').style.display='none';
  }}
}});
</script></body></html>"""


# ── Data extraction ───────────────────────────────────────────────────────────

def parse_metrics_jsonl(path: str) -> Dict[str, Any]:
    rows: List[Dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not rows:
        return {"steps": [], "loss_series": [], "metric_series": [], "lr": []}

    # Collect unique step-indexed fields
    steps, train_loss, val_loss, val_auc, val_f1, val_acc, lr = [], [], [], [], [], [], []
    for r in rows:
        if "step" not in r:
            continue
        steps.append(r["step"])
        train_loss.append(r.get("train_loss") or r.get("train_total"))
        val_loss.append(r.get("val_loss") or r.get("val_total"))
        val_auc.append(r.get("val_auc"))
        val_f1.append(r.get("val_f1"))
        val_acc.append(r.get("val_accuracy"))
        lr.append(r.get("lr"))

    loss_series = [{"label": "train_loss", "y": train_loss},
                   {"label": "val_loss",   "y": val_loss}]
    metric_series = []
    if any(v is not None for v in val_auc):
        metric_series.append({"label": "val_auc",      "y": val_auc})
    if any(v is not None for v in val_f1):
        metric_series.append({"label": "val_f1",       "y": val_f1})
    if any(v is not None for v in val_acc):
        metric_series.append({"label": "val_accuracy", "y": val_acc})

    n_epochs = len({r.get("epoch") for r in rows if "epoch" in r})

    return {
        "steps":          steps,
        "loss_series":    loss_series,
        "metric_series":  metric_series,
        "lr":             lr,
        "n_steps":        len(steps),
        "n_epochs":       n_epochs,
        "summary":        {k: v for r in rows[-1:] for k, v in r.items()
                           if isinstance(v, (int, float)) and k != "step"},
    }


def build_summary_rows(summary: Dict[str, float]) -> str:
    rows = ""
    for k, v in sorted(summary.items()):
        if k in ("epoch", "epoch_time_s") or not isinstance(v, float):
            continue
        rows += f"<tr><td>{k}</td><td><code>{v:.6f}</code></td></tr>\n"
    return rows or "<tr><td colspan='2' style='color:var(--muted)'>No summary data</td></tr>"


# ── Writers ───────────────────────────────────────────────────────────────────

def generate_train_html(log_dir: str, output_path: str, paper: str) -> None:
    jsonl = os.path.join(log_dir, "metrics.jsonl")
    if not os.path.exists(jsonl):
        print(f"  [skip train.html] metrics.jsonl not found in {log_dir}")
        return

    run  = os.path.basename(log_dir)
    data = parse_metrics_jsonl(jsonl)

    html = _TRAIN_HTML.format(
        paper=paper,
        run=run,
        n_steps=data["n_steps"],
        n_epochs=data["n_epochs"],
        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        style=_SHARED_STYLE,
        chart_js=_CHART_JS,
        data_json=json.dumps(data),
        summary_rows=build_summary_rows(data.get("summary", {})),
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"  ✓ train.html    → {output_path}")


def generate_evaluate_html(log_dir: str, output_path: str, paper: str) -> None:
    results_path = os.path.join(log_dir, "test_results.json")
    if not os.path.exists(results_path):
        print(f"  [skip evaluate.html] test_results.json not found in {log_dir}")
        print(f"                       Run:  python test.py --run-name {os.path.basename(log_dir)}")
        return

    with open(results_path) as f:
        results = json.load(f)

    run   = results.get("run", os.path.basename(log_dir))
    split = results.get("split", "test")
    data  = {
        "metrics":   results.get("metrics", {}),
        "curves":    results.get("curves",  {}),
        "confusion": results.get("confusion", []),
    }

    html = _EVAL_HTML.format(
        paper=paper,
        run=run,
        split=split,
        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        style=_SHARED_STYLE,
        chart_js=_CHART_JS,
        data_json=json.dumps(data),
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"  ✓ evaluate.html → {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Generate train.html and evaluate.html from log files")
    p.add_argument("--log-dir",     required=True, help="Path to logs/{paper}/{run_name}/")
    p.add_argument("--output-dir",  required=True, help="Path to outputs/{paper}/")
    p.add_argument("--paper",       type=str, default=None,
                   help="Paper name (derived from output-dir if omitted)")
    args = p.parse_args()

    paper = args.paper or os.path.basename(os.path.normpath(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    generate_train_html(
        args.log_dir,
        os.path.join(args.output_dir, "train.html"),
        paper,
    )
    generate_evaluate_html(
        args.log_dir,
        os.path.join(args.output_dir, "evaluate.html"),
        paper,
    )


if __name__ == "__main__":
    main()
