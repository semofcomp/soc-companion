/* Companion v2 shell: nav, switchers, theme, state, data loader. */
(function(){
const PAGES=[
  ["explore-3d.html","3-D Graph"],
  ["explore-2d.html","2-D Graph"],
  {label:"Properties",children:[
    ["generality.html","Generality"],
    ["reciprocity.html","Reciprocity"],
    ["ecosystems.html","Community Clusters"]
  ]},
  ["analytics-braided.html","Analytics"],
  ["validation.html","Validation"],["methods.html","Methods"],["data.html","Data"],["code.html","Code"]
];
const DEF={country:"CA",generator:"gpt-4o",domain:"both",rep:"gemini"};
const LS="cv2-state";
function load(){try{return Object.assign({},DEF,JSON.parse(localStorage.getItem(LS)||"{}"));}catch(e){return Object.assign({},DEF);}}
let S=load();
if(!S.theme){try{S.theme=localStorage.getItem("site-theme")||"dark";}catch(e){S.theme="dark";}}
window.CV2={
  state:S,
  get(k){return S[k];},
  set(k,v){S[k]=v;localStorage.setItem(LS,JSON.stringify(S));window.dispatchEvent(new CustomEvent("cv2:change",{detail:{key:k,value:v,state:S}}));},
  econName(){return S.country==="US"?"United States":"Canada";},
  // grey out (but keep BROWSABLE — no disabled attr) a header control from a page
  setControlActive(key,on){const m={economy:"sw-country",generator:"sw-generator",domain:"sw-domain",embedding:"sw-rep"};
    const s=document.getElementById(m[key]);if(s){const c=s.closest(".ctl");if(c)c.classList.toggle("off",!on);}},
  // fetch+cache a data bundle, e.g. CV2.data("meta") -> data/meta_CA.json
  _cache:{},
  async data(stem,country){country=country||S.country;const key=stem+"_"+country;
    if(this._cache[key])return this._cache[key];
    const r=await fetch("data/"+key+".json");if(!r.ok)throw new Error("missing "+key);
    const j=await r.json();this._cache[key]=j;return j;}
};
// theme (pre-applied by inline head script; this toggles). Apply BOTH the
// data-theme attribute (app.css) and the .light class (pages with html.light CSS),
// and persist under both keys so the choice carries across every page.
function applyTheme(t){
  document.documentElement.setAttribute("data-theme",t);
  document.documentElement.classList.toggle("light",t==="light");
  try{localStorage.setItem("site-theme",t);}catch(e){}
}
applyTheme(S.theme);

function sel(id,label,opts,val,cls){
  // Small uppercase caption rendered ABOVE the bordered select (in normal flow,
  // so it can never collide with the box border). Replaces the old <optgroup>
  // header, whose native label row clipped into the popup frame in some browsers.
  return '<span class="ctl '+(cls||"")+'"><span class="clbl">'+label+'</span><select id="'+id+'" title="'+label+'" aria-label="'+label+'">'+
    opts.map(o=>'<option value="'+o[0]+'"'+(o[0]===val?" selected":"")+'>'+o[1]+'</option>').join("")+
    '</select></span>';
}
// Per-page control relevance: a page sets <body data-controls="economy,generator">
// to show only those selectors; others are hidden. Omit the attribute to show all;
// data-controls="none" hides all four. Theme button always shows.
function applyControlRelevance(){
  const raw=document.body.getAttribute("data-controls");
  if(raw===null) return;                       // not declared -> show all
  const allow=new Set(raw.split(",").map(s=>s.trim()).filter(Boolean));
  [["ctl-economy","economy"],["ctl-generator","generator"],
   ["ctl-domain","domain"],["ctl-embedding","embedding"]].forEach(function(p){
    const el=document.querySelector("."+p[0]);
    if(el && !allow.has(p[1])) el.classList.add("hidden");
  });
}
function header(){
  const here=(location.pathname.split("/").pop()||"index.html");
  const navItem=function(p){
    if(Array.isArray(p)) return '<a href="'+p[0]+'"'+(p[0]===here?' class="active"':"")+'>'+p[1]+'</a>';
    const ca=p.children.some(function(c){return c[0]===here;});   // group is active when on a child page
    return '<div class="navdrop"><span class="navdrop-t'+(ca?' active':'')+'">'+p.label+' ▾</span>'+
      '<div class="navdrop-menu">'+p.children.map(function(c){return '<a href="'+c[0]+'"'+(c[0]===here?' class="active"':"")+'>'+c[1]+'</a>';}).join("")+'</div></div>';
  };
  const nav='<nav class="main">'+PAGES.map(navItem).join("")+'</nav>';
  const ctrls='<div class="controls">'+
    '<div class="ctools" id="ctools"></div>'+
    '<div class="cselect">'+
    sel("sw-country","Economy",[["CA","Canada"],["US","United States"]],S.country,"ctl-economy")+
    sel("sw-generator","Generator",[["gpt-4o","GPT-4o"],["gemini-2.5-pro","Gemini"],["consensus","Consensus"]],S.generator,"ctl-generator")+
    sel("sw-domain","Domain",[["both","Both"],["consumer","Consumption"],["production","Production"]],S.domain,"ctl-domain")+
    sel("sw-rep","Embedding",[["tfidf","TF-IDF"],["glove","GloVe"],["gemini","Gemini"],["openai","OpenAI"]],S.rep,"ctl-embedding")+
    '</div></div>';
  return '<div class="anon">Anonymized companion — double-blind review version. Do not distribute author-identifying information.</div>'+
    '<header class="site"><div class="wrap hbar"><span class="brand">The Semantics of <span class="dot">Complementarity</span></span>'+nav+
    '<button class="themebtn" id="sw-theme" title="Light/Dark">◐</button></div>'+
    '<div class="wrap" style="padding-bottom:8px">'+ctrls+'</div></header>';
}
function footer(){
  return '<footer class="site"><div class="wrap">Companion to <em>“The Semantics of Complementarity.”</em> '+
    'General-construct, co-primary (Canada = consumption exemplar · United States = production exemplar). '+
    'Anonymized for double-blind review.</div></footer>';
}
function wire(){
  const b=(id,k)=>{const e=document.getElementById(id);if(e)e.addEventListener("change",ev=>CV2.set(k,ev.target.value));};
  b("sw-country","country");b("sw-generator","generator");b("sw-domain","domain");b("sw-rep","rep");
  const t=document.getElementById("sw-theme");
  if(t)t.addEventListener("click",()=>{const nt=(CV2.get("theme")==="dark")?"light":"dark";applyTheme(nt);CV2.set("theme",nt);});
}
document.addEventListener("DOMContentLoaded",function(){
  const h=document.getElementById("hdr"); if(h)h.innerHTML=header();
  const f=document.getElementById("ftr"); if(f)f.innerHTML=footer();
  wire(); applyControlRelevance();
  if(h){var sp=document.getElementById("hdrspace");
    if(!sp){sp=document.createElement("div");sp.id="hdrspace";sp.setAttribute("aria-hidden","true");h.insertAdjacentElement("afterend",sp);}
    var sz=function(){sp.style.height=h.getBoundingClientRect().height+"px";};
    sz();window.addEventListener("resize",sz);setTimeout(sz,80);}
  window.dispatchEvent(new CustomEvent("cv2:ready",{detail:{state:S}}));
});
})();
