const fs=require("fs");const {JSDOM,VirtualConsole}=require("jsdom");
const html=fs.readFileSync("/workspace/web/templates/index.html","utf8");
const appjs=fs.readFileSync("/workspace/web/static/app.js","utf8");
const vc=new VirtualConsole(); vc.on("error",(...a)=>console.log("CONSOLE.ERROR",a.join(" ")));
const dom=new JSDOM(html,{url:"http://127.0.0.1:8000/",runScripts:"outside-only",pretendToBeVisual:true,virtualConsole:vc});
const {window}=dom;
window.fetch=(u,o)=>fetch(new URL(u,"http://127.0.0.1:8000"),o);
function ctx(){return new Proxy({},{get(t,p){if(p==="measureText")return()=>({width:10});if(p in t)return t[p];return ()=>{};},set(t,p,v){t[p]=v;return true;}});}
window.HTMLCanvasElement.prototype.getContext=()=>ctx();
Object.defineProperty(window.HTMLCanvasElement.prototype,"clientWidth",{get:()=>520,configurable:true});
Object.defineProperty(window.HTMLCanvasElement.prototype,"clientHeight",{get:()=>180,configurable:true});
window.HTMLMediaElement.prototype.play=()=>Promise.resolve();
window.HTMLMediaElement.prototype.pause=()=>{};
window.eval(appjs);
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
(async()=>{
 await sleep(9000);
 console.log("chips",window.document.querySelectorAll(".sample-chip").length);
 const sr=window.document.getElementById("smoothRange");
 sr.value="95"; sr.dispatchEvent(new window.Event("input",{bubbles:true}));
 await sleep(1500);
 console.log("mHfSub:",window.document.getElementById("mHfSub").textContent);
 console.log("stage:",window.document.getElementById("stageText").textContent);
 try{ window.eval("window.__dbg={stage:state.job&&state.job.stage, hasPlan:!!state.plan, sjid:state.jobId}"); }catch(e){console.log("eval dbg err",e.message);}
 console.log("dbg:",window.__dbg);
 process.exit(0);
})();
