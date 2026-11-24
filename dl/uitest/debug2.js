const fs=require("fs");const {JSDOM,VirtualConsole}=require("jsdom");
const html=fs.readFileSync("/workspace/web/templates/index.html","utf8");
let appjs=fs.readFileSync("/workspace/web/static/app.js","utf8");
appjs += "\nwindow.__getState=()=>state;";
const vc=new VirtualConsole();
const dom=new JSDOM(html,{url:"http://127.0.0.1:8000/",runScripts:"outside-only",pretendToBeVisual:true,virtualConsole:vc});
const {window}=dom;
window.fetch=(u,o)=>fetch(new URL(u,"http://127.0.0.1:8000"),o);
function ctx(){return new Proxy({},{get(t,p){if(p==="measureText")return()=>({width:10});if(p in t)return t[p];return ()=>{};},set(t,p,v){t[p]=v;return true;}});}
window.HTMLCanvasElement.prototype.getContext=()=>ctx();
Object.defineProperty(window.HTMLCanvasElement.prototype,"clientWidth",{get:()=>520,configurable:true});
Object.defineProperty(window.HTMLCanvasElement.prototype,"clientHeight",{get:()=>180,configurable:true});
window.HTMLMediaElement.prototype.play=()=>Promise.resolve();
window.HTMLMediaElement.prototype.pause=()=>{};
window.HTMLMediaElement.prototype.load=()=>{};
window.eval(appjs);
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
(async()=>{
 await sleep(9000);
 let s=window.__getState();
 console.log("before slider: stage",s.job.stage,"loadedVer",s.loadedVersion,"jobVer",s.job.video_version);
 const sr=window.document.getElementById("smoothRange");
 sr.value="95"; sr.dispatchEvent(new window.Event("input",{bubbles:true}));
 await sleep(1500);
 s=window.__getState();
 console.log("after input: stage",s.job.stage,"queued",s.renderQueued);
 // wait up to 20s for version change
 for(let k=0;k<40;k++){
   await sleep(500);
   s=window.__getState();
   if(s.job.stage==="done" && s.loadedVersion===s.job.video_version && s.result?.verification){
      console.log("RELOADED at",k*0.5,"s ver",s.loadedVersion); break;
   }
 }
 s=window.__getState();
 console.log("final stage",s.job.stage,"loadedVer",s.loadedVersion,"jobVer",s.job.video_version,
   "hasVerif",!!s.result?.verification);
 console.log("HF sub:",window.document.getElementById("mHfSub").textContent.trim());
 process.exit(0);
})();
