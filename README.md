# AI-Unlocked---exploring-ML-DL-LLM-PART1-CME-
demonstration 1:
code html:

!!!COPY THIS CODE IN CHATGPT

||||||

<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Visual ML Playground — Live Decision Boundary</title>
<style>
*{box-sizing:border-box} body{margin:0;font-family:Inter,system-ui,Arial,sans-serif;background:#0b1020;color:#eef2ff}
.wrap{max-width:1180px;margin:auto;padding:24px}
header{display:flex;justify-content:space-between;align-items:end;gap:20px;margin-bottom:18px}
h1{margin:0;font-size:28px}.sub{color:#aab4d0;margin-top:5px}
.grid{display:grid;grid-template-columns:1fr 300px;gap:18px}
.card{background:#121a30;border:1px solid #263250;border-radius:16px;padding:16px;box-shadow:0 12px 35px #0004}
#board{width:100%;height:auto;aspect-ratio:1/1;background:#0e1629;border-radius:12px;display:block;cursor:crosshair}
.controls{display:grid;gap:12px}.row{display:flex;gap:8px;align-items:center}.row>*{flex:1}
button,select,input{font:inherit}button{border:0;border-radius:10px;padding:10px 12px;background:#273656;color:#fff;cursor:pointer}
button:hover{background:#34476d}.primary{background:#4f7cff}.primary:hover{background:#6b91ff}
label{font-size:13px;color:#b9c3dc}.value{float:right;color:#fff}
.stat{padding:10px 12px;background:#0d1527;border-radius:10px;margin-top:8px}.stat b{display:block;font-size:20px;margin-top:2px}
.hint{font-size:12px;color:#8995b3;line-height:1.45}.legend{display:flex;gap:16px;font-size:12px;color:#aab4d0;margin:10px 0}.dot{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:5px}
.blue{background:#54a8ff}.orange{background:#ff9f43}.boundary{background:#fff}
.status{font-size:12px;padding:7px 9px;border-radius:8px;background:#18233d;color:#9fc0ff}
@media(max-width:850px){.grid{grid-template-columns:1fr}header{display:block}}
</style>
</head>
<body>
<div class="wrap">
<header>
  <div><h1>Visual ML Playground</h1><div class="sub">Train a tiny classifier and watch its decision boundary update in real time.</div></div>
  <div class="status" id="status">Ready — add data or start training</div>
</header>

<div class="grid">
  <section class="card">
    <canvas id="board" width="700" height="700"></canvas>
    <div class="legend">
      <span><i class="dot blue"></i>Class A</span>
      <span><i class="dot orange"></i>Class B</span>
      <span><i class="dot boundary"></i>Decision boundary</span>
    </div>
    <div class="hint">Click the canvas to add points. Use the selected class, then switch classes and add another cluster. The model learns a linear boundary using gradient descent.</div>
  </section>

  <aside class="card">
    <div class="controls">
      <div>
        <label>Model</label>
        <select id="model" style="width:100%;margin-top:6px;padding:10px;border-radius:10px;background:#0d1527;color:#fff;border:1px solid #34476d">
          <option value="logistic">Logistic Regression</option>
          <option value="linear">Linear Regression</option>
          <option value="knn">K-Nearest Neighbors (KNN)</option>
        </select>
      </div>
      <div class="row">
        <button id="classA" class="primary">Class A</button>
        <button id="classB">Class B</button>
      </div>
      <div>
        <label>Learning rate <span class="value" id="lrVal">0.08</span></label>
        <input id="lr" type="range" min="0.005" max="0.3" step="0.005" value="0.08">
      </div>
      <div>
        <label>Steps / frame <span class="value" id="spfVal">8</span></label>
        <input id="spf" type="range" min="1" max="30" value="8">
      </div>
      <div class="row">
        <button id="train" class="primary">▶ Start training</button>
        <button id="reset">Reset model</button>
      </div>
      <div class="row">
        <button id="clear">Clear points</button>
        <button id="demo">Load demo</button>
      </div>

      <div class="stat">Training step <b id="step">0</b></div>
      <div class="stat">Loss <b id="loss">—</b></div>
      <div class="stat">Accuracy <b id="acc">—</b></div>
      <div class="stat">Model <b id="modelName" style="font-size:14px">Logistic Regression</b></div>

      <div class="hint">
        <b>What to observe:</b><br>
        1. Add labeled examples.<br>
        2. Start training.<br>
        3. The background shows the model's current prediction probability.<br>
        4. The white contour is the 50% decision boundary.<br>
        5. Change the data and train again.
      </div>
    </div>
  </aside>
</div>
</div>

<script>
const c=document.getElementById('board'),ctx=c.getContext('2d');
const W=c.width,H=c.height;
let points=[], selected=0, running=false, step=0, model='logistic';
let lr=.08, spf=8, w=[0,0,0], knnK=5;

const $=id=>document.getElementById(id);
function sigmoid(z){return 1/(1+Math.exp(-Math.max(-40,Math.min(40,z))))}
function feat(p){return [1,p.x/W*2-1,p.y/H*2-1]}
function predict(p){
  if(model==='knn'){
    if(!points.length)return .5;
    const ranked=points.map(q=>({d:(q.x-p.x)**2+(q.y-p.y)**2,y:q.yc})).sort((a,b)=>a.d-b.d);
    const k=Math.min(knnK,ranked.length);
    return ranked.slice(0,k).reduce((s,q)=>s+q.y,0)/k;
  }
  let f=feat(p);
  return sigmoid(w[0]*f[0]+w[1]*f[1]+w[2]*f[2]);
}

function lossAndGrad(){
  if(!points.length)return {loss:0,g:[0,0,0]};
  let g=[0,0,0],L=0;
  for(const p of points){
    const f=feat(p), y=p.yc, q=sigmoid(w[0]*f[0]+w[1]*f[1]+w[2]*f[2]);
    L += -(y*Math.log(q+1e-9)+(1-y)*Math.log(1-q+1e-9));
    for(let j=0;j<3;j++)g[j]+=(q-y)*f[j];
  }
  return {loss:L/points.length,g:g.map(v=>v/points.length)}
}
function trainOne(){
  if(!points.length || model==='knn') return;
  for(let k=0;k<spf;k++){
    if(model==='logistic'){
      const r=lossAndGrad();
      for(let j=0;j<3;j++)w[j]-=lr*r.g[j];
    } else if(model==='linear'){
      // Least-squares regression on the binary labels 0/1.
      let g=[0,0,0], n=points.length;
      for(const p of points){
        const f=feat(p), pred=w[0]*f[0]+w[1]*f[1]+w[2]*f[2], err=pred-p.yc;
        for(let j=0;j<3;j++)g[j]+=2*err*f[j]/n;
      }
      for(let j=0;j<3;j++)w[j]-=lr*g[j];
    }
    step++;
  }
}
function accuracy(){
  if(!points.length)return null;
  let n=0; for(const p of points)if((predict(p)>=.5)===(p.yc===1))n++;
  return n/points.length*100;
}
function draw(){
  ctx.clearRect(0,0,W,H);
  // probability heatmap
  const s=10;
  for(let y=0;y<H;y+=s)for(let x=0;x<W;x+=s){
    const q=predict({x,y});
    const a=Math.abs(q-.5)*1.35;
    ctx.fillStyle=q>.5?`rgba(255,159,67,${.055+a*.12})`:`rgba(84,168,255,${.055+a*.12})`;
    ctx.fillRect(x,y,s,s);
  }
  // grid
  ctx.strokeStyle='rgba(255,255,255,.06)';ctx.lineWidth=1;
  for(let i=0;i<=10;i++){let x=i*W/10,y=i*H/10;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke()}
  // boundary w0+w1*x+w2*y=0 in normalized coords
  if(Math.abs(w[2])>.000001 || Math.abs(w[1])>.000001){
    let pts=[];
    function yy(x){let xn=x/W*2-1;return H*((-(w[0]+w[1]*xn)/w[2]+1)/2)}
    function xx(y){let yn=y/H*2-1;return W*((-(w[0]+w[2]*yn)/w[1]+1)/2)}
    if(Math.abs(w[2])>.000001){let y0=yy(0),y1=yy(W);if(y0>=0&&y0<=H)pts.push([0,y0]);if(y1>=0&&y1<=H)pts.push([W,y1])}
    if(pts.length<2&&Math.abs(w[1])>.000001){let x0=xx(0),x1=xx(H);if(x0>=0&&x0<=W)pts.push([x0,0]);if(x1>=0&&x1<=W)pts.push([x1,H])}
    if(pts.length>=2){ctx.strokeStyle='#fff';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(...pts[0]);ctx.lineTo(...pts[1]);ctx.stroke()}
  }
  for(const p of points){
    ctx.beginPath();ctx.arc(p.x,p.y,7,0,Math.PI*2);
    ctx.fillStyle=p.yc?'#ff9f43':'#54a8ff';ctx.fill();
    ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.stroke();
  }
  $('step').textContent=step;
  if(points.length){
    let L=0;
    if(model==='linear'){
      for(const p of points){const f=feat(p), pred=w[0]*f[0]+w[1]*f[1]+w[2]*f[2];L+=(pred-p.yc)**2}
      L/=points.length;
    } else if(model==='knn'){
      for(const p of points){const q=predict(p);L+=-(p.yc*Math.log(q+1e-9)+(1-p.yc)*Math.log(1-q+1e-9))}
      L/=points.length;
    } else L=lossAndGrad().loss;
    $('loss').textContent=L.toFixed(4);
    $('acc').textContent=accuracy().toFixed(1)+'%';
  } else {$('loss').textContent='—';$('acc').textContent='—'}
}
function loop(){
  if(running){trainOne();$('status').textContent='Training… boundary is updating live';draw();requestAnimationFrame(loop)}
}
c.addEventListener('click',e=>{
  const r=c.getBoundingClientRect(); points.push({x:(e.clientX-r.left)*W/r.width,y:(e.clientY-r.top)*H/r.height,yc:selected});
  draw();
});
$('classA').onclick=()=>{selected=0;$('classA').className='primary';$('classB').className='';}
$('classB').onclick=()=>{selected=1;$('classB').className='primary';$('classA').className='';}
$('model').onchange=e=>{
  model=e.target.value;
  $('modelName').textContent=model==='knn'?'K-Nearest Neighbors (KNN)':model==='linear'?'Linear Regression':'Logistic Regression';
  w=[0,0,0];step=0;
  running=false;
  $('train').textContent='▶ Start training';
  $('status').textContent=model==='knn'?'KNN predicts directly from nearest labeled points':'Ready — train the selected model';
  $('step').textContent='0';
  draw();
}
$('lr').oninput=e=>{lr=+e.target.value;$('lrVal').textContent=lr.toFixed(3)}
$('spf').oninput=e=>{spf=+e.target.value;$('spfVal').textContent=spf}
$('train').onclick=()=>{
  if(model==='knn'){draw();$('status').textContent='KNN has no iterative training — predictions update immediately from the data';return}
  running=!running;
  $('train').textContent=running?'⏸ Pause training':'▶ Start training';
  if(running){requestAnimationFrame(loop)}else{$('status').textContent='Paused'}
}
$('reset').onclick=()=>{w=[0,0,0];step=0;running=false;$('train').textContent='▶ Start training';$('status').textContent='Model reset';draw()}
$('clear').onclick=()=>{points=[];w=[0,0,0];step=0;running=false;$('train').textContent='▶ Start training';$('status').textContent='Ready — add data';draw()}
$('demo').onclick=()=>{
  points=[];
  for(let i=0;i<28;i++){
    points.push({x:100+Math.random()*230,y:110+Math.random()*430,yc:0});
    points.push({x:370+Math.random()*230,y:110+Math.random()*430,yc:1});
  }
  w=[0,0,0];step=0;running=false;$('train').textContent='▶ Start training';$('status').textContent='Demo dataset loaded';draw();
}
draw();


  ||||||
  copy this code in   CHATGPT
</script>
</body>
</html>
