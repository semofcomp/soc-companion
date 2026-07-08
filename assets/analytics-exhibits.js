/* Analytics exhibits — shared renderers for every section. Plotly charts + HTML
   blocks, themed to the site tokens (app.css), re-rendered on theme toggle.
   window.AX.<fn>(elId-or-el, ...). Data: data/*.json (served). */
(function(){
  function cssv(n,f){var v=getComputedStyle(document.documentElement).getPropertyValue(n).trim();return v||f;}
  function theme(){return {fg:cssv('--fg','#1a1d21'),fg2:cssv('--fg2','#5a6068'),fg3:cssv('--fg3','#8a9099'),
    grid:cssv('--border','#e2e5e9'),bg2:cssv('--bg2','#f6f7f9'),teal:cssv('--teal','#1D9E75'),coral:cssv('--coral','#D85A30')};}
  function E(x){return typeof x==='string'?document.getElementById(x):x;}
  function esc(s){return (s||'').replace(/[&<>]/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;'}[c];});}
  function rgba(hex,a){var h=hex.replace('#','');return 'rgba('+parseInt(h.substr(0,2),16)+','+parseInt(h.substr(2,2),16)+','+parseInt(h.substr(4,2),16)+','+a+')';}
  var FONT='-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif';
  var CFG={displayModeBar:false,responsive:true};
  var REPS=['tfidf','glove','gemini','openai'],RNAME={tfidf:'TF-IDF',glove:'GloVe',gemini:'Gemini',openai:'OpenAI'};
  function baseLay(t,m){return {paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{family:FONT,size:12.5,color:t.fg2},
    margin:m||{l:56,r:14,t:10,b:44},hoverlabel:{bgcolor:t.bg2,bordercolor:t.grid,font:{family:FONT,color:t.fg}}};}

  var AX={ _cache:{}, _charts:[],
    get:function(p){var s=this;if(this._cache[p])return Promise.resolve(this._cache[p]);
      return fetch(p).then(function(r){return r.json();}).then(function(j){s._cache[p]=j;return j;});},
    load:function(econ){return this.get('data/probe_scatter_'+econ+'.json');},
    _reg:function(el,fn){el=E(el);if(!el)return;this._charts=this._charts.filter(function(c){return c.el!==el;});this._charts.push({el:el,fn:fn});fn();},

    /* §3 — off-axis scatter + ROC */
    offaxis:function(el,d){el=E(el);var s=this;this._reg(el,function(){var t=theme();
      var C=d.points.filter(function(p){return p.t==='c';}),S=d.points.filter(function(p){return p.t==='s';});
      function tr(pts,name,color){return {x:pts.map(function(p){return p.x;}),y:pts.map(function(p){return p.y;}),
        customdata:pts.map(function(p){return [p.f,p.p];}),mode:'markers',type:'scatter',name:name,
        marker:{size:6,color:color,opacity:.72,line:{width:0}},
        hovertemplate:'<b>%{customdata[0]}</b><br>+ %{customdata[1]}<br>cosine %{x:.2f}  ·  probe %{y:.1f}<extra></extra>'};}
      var lay=baseLay(t,{l:66,r:16,t:8,b:50});
      lay.showlegend=true;lay.legend={orientation:'h',y:1.1,x:1,xanchor:'right',font:{size:12}};
      lay.xaxis={title:{text:'pair cosine similarity   (right = more alike)',font:{size:12.5,color:t.fg2}},gridcolor:t.grid,zerolinecolor:t.grid,color:t.fg2};
      lay.yaxis={title:{text:'learned probe score   (up = more complement-like)',font:{size:12.5,color:t.fg2}},gridcolor:t.grid,zerolinecolor:t.grid,color:t.fg2};
      lay.shapes=[{type:'line',xref:'paper',x0:0,x1:1,y0:d.cutoff,y1:d.cutoff,line:{color:t.fg2,width:1.3,dash:'dash'}}];
      lay.annotations=[{xref:'paper',x:0.99,y:d.cutoff,yshift:10,xanchor:'right',showarrow:false,
        text:'probe cut-off · '+Math.round(d.balanced_accuracy*100)+'% balanced accuracy',font:{size:11,color:t.fg2}}];
      Plotly.react(el,[tr(S,'substitute pairs',t.coral),tr(C,'complement–focal pairs',t.teal)],lay,CFG);});},

    roc:function(el,d){el=E(el);this._reg(el,function(){var t=theme();
      var probe={x:d.roc_probe.map(function(p){return p[0];}),y:d.roc_probe.map(function(p){return p[1];}),mode:'lines',
        name:'learned probe  (AUC '+d.auc_probe.toFixed(2)+')',line:{color:t.teal,width:2.6},fill:'tozeroy',fillcolor:rgba(t.teal,.13),hovertemplate:'FPR %{x:.2f} · TPR %{y:.2f}<extra>probe</extra>'};
      var cos={x:d.roc_cosine.map(function(p){return p[0];}),y:d.roc_cosine.map(function(p){return p[1];}),mode:'lines',
        name:'raw cosine  (AUC '+d.auc_cosine.toFixed(2)+')',line:{color:t.coral,width:2.6},fill:'tozeroy',fillcolor:rgba(t.coral,.12),hovertemplate:'FPR %{x:.2f} · TPR %{y:.2f}<extra>cosine</extra>'};
      var chance={x:[0,1],y:[0,1],mode:'lines',line:{color:t.fg2,width:1.2,dash:'dash'},hoverinfo:'skip',showlegend:false};
      var lay=baseLay(t,{l:56,r:16,t:8,b:48});lay.legend={x:.98,y:.04,xanchor:'right',yanchor:'bottom',font:{size:12}};
      lay.xaxis={title:{text:'false-positive rate',font:{size:12.5,color:t.fg2}},range:[0,1],gridcolor:t.grid,zerolinecolor:t.grid,color:t.fg2,constrain:'domain'};
      lay.yaxis={title:{text:'true-positive rate',font:{size:12.5,color:t.fg2}},range:[0,1],gridcolor:t.grid,zerolinecolor:t.grid,color:t.fg2,scaleanchor:'x'};
      Plotly.react(el,[cos,probe,chance],lay,CFG);});},

    /* generic themed bar chart (used by the validation data pages) */
    bar:function(el,cfg){el=E(el);var s=this;this._reg(el,function(){var t=theme();
      var data=[{x:cfg.x,y:cfg.y,type:'bar',marker:{color:cfg.colors||cfg.x.map(function(){return t.teal;})},
        text:(cfg.text||cfg.y),textposition:'outside',cliponaxis:false,hovertemplate:'%{x}: %{y}<extra></extra>'}];
      var lay=baseLay(t,{l:56,r:16,t:20,b:42});
      lay.yaxis={title:{text:cfg.ytitle||'',font:{size:12,color:t.fg2}},gridcolor:t.grid,zerolinecolor:t.grid,color:t.fg2,rangemode:'tozero'};
      lay.xaxis={color:t.fg2,tickfont:{size:12}};
      if(cfg.ymax)lay.yaxis.range=[0,cfg.ymax];
      if(cfg.hline!=null)lay.shapes=[{type:'line',xref:'paper',x0:0,x1:1,y0:cfg.hline,y1:cfg.hline,line:{color:t.fg2,width:1.2,dash:'dash'}}];
      Plotly.react(el,data,lay,CFG);});},

    /* §1 — ego-net: a focal + its complements (scatter) and substitutes (hug), in map space */
    egonet:function(el,d){el=E(el);var s=this;this._reg(el,function(){var t=theme();
      function pts(arr,color,name){return {x:arr.map(function(p){return p.x;}),y:arr.map(function(p){return p.y;}),
        customdata:arr.map(function(p){return [p.name,(p.cos!=null?('<br>cosine '+(+p.cos).toFixed(2)):'')];}),
        mode:'markers',type:'scatter',name:name,marker:{size:8,color:color,opacity:.82,line:{width:0}},
        hovertemplate:'<b>%{customdata[0]}</b>%{customdata[1]}<extra></extra>'};}
      var lx=[],ly=[]; d.comp.concat(d.sub).forEach(function(p){lx.push(d.x,p.x,null);ly.push(d.y,p.y,null);});
      var lines={x:lx,y:ly,mode:'lines',line:{color:t.grid,width:0.7},hoverinfo:'skip',showlegend:false};
      var focal={x:[d.x],y:[d.y],customdata:[[d.name,'']],mode:'markers',type:'scatter',name:'focal',showlegend:false,
        marker:{size:16,color:t.fg,symbol:'diamond',line:{width:1.5,color:cssv('--bg','#fff')}},
        hovertemplate:'<b>%{customdata[0]}</b><br>focal<extra></extra>'};
      var lay=baseLay(t,{l:6,r:6,t:24,b:6}); lay.showlegend=true;
      lay.legend={orientation:'h',y:1.14,x:0.5,xanchor:'center',font:{size:11.5}};
      lay.xaxis={visible:false}; lay.yaxis={visible:false,scaleanchor:'x'}; lay.hoverlabel={namelength:-1,align:'left'};
      Plotly.react(el,[lines,focal,pts(d.sub,t.coral,'substitutes'),pts(d.comp,t.teal,'complements')],lay,CFG);});},

    /* §2 — cosine gradient: mean cosine by relationship type, per representation */
    gradient:function(el){el=E(el);var s=this;this.get('data/CA_general_stats.json').then(function(gs){s._reg(el,function(){var t=theme();var sp=gs.spectrum;
      function ser(key,name,color){return {x:REPS.map(function(r){return RNAME[r];}),y:REPS.map(function(r){return sp[r][key];}),type:'bar',name:name,marker:{color:color},hovertemplate:'%{y:.2f}<extra>'+name+'</extra>'};}
      var data=[ser('random','random pair',t.fg3),ser('complements','complement',t.teal),ser('substitute','substitute',t.coral)];
      var lay=baseLay(t,{l:54,r:14,t:8,b:38});lay.barmode='group';lay.legend={orientation:'h',y:1.13,x:1,xanchor:'right',font:{size:12}};
      lay.yaxis={title:{text:'mean cosine similarity',font:{size:12.5,color:t.fg2}},gridcolor:t.grid,zerolinecolor:t.grid,color:t.fg2};lay.xaxis={color:t.fg2};
      Plotly.react(el,data,lay,CFG);});});},

    /* §5 — dose-response: mean percentile by generation rank */
    dose:function(el){el=E(el);var s=this;this.get('data/CA_dose.json').then(function(dd){s._reg(el,function(){var t=theme();var pm=dd.per_model;
      var COL={tfidf:t.fg3,glove:'#7FB5E6',gemini:t.teal,openai:t.coral};
      var data=REPS.map(function(r){var ks=Object.keys(pm[r]).sort();return {x:ks.map(Number),y:ks.map(function(k){return pm[r][k].mean_pct*100;}),mode:'lines+markers',name:RNAME[r],line:{color:COL[r],width:2.2},marker:{size:6},hovertemplate:'rank %{x} · %{y:.0f}th pct<extra>'+RNAME[r]+'</extra>'};});
      var lay=baseLay(t,{l:54,r:14,t:8,b:42});lay.legend={orientation:'h',y:1.14,x:1,xanchor:'right',font:{size:12}};
      lay.xaxis={title:{text:"generator's rank of the complement  (1 = listed first)",font:{size:12.5,color:t.fg2}},dtick:1,gridcolor:t.grid,color:t.fg2};
      lay.yaxis={title:{text:'mean cosine percentile',font:{size:12.5,color:t.fg2}},gridcolor:t.grid,zerolinecolor:t.grid,color:t.fg2};
      Plotly.react(el,data,lay,CFG);});});},

    /* §2 stat tiles — comp-vs-sub cosine AUC per representation (all below 0.5) */
    cosineAUCtiles:function(el){el=E(el);this.get('data/RESULTS_CANONICAL.json').then(function(rc){el=E(el);if(!el)return;var h=rc.CA.headline_6_4;
      el.innerHTML=REPS.map(function(r){var v=h[r].comp_vs_sub_cosine_auc;return '<div class="ax-stat"><div class="v coral">'+v.toFixed(2)+'</div><div class="l">'+RNAME[r]+' · comp-vs-sub cosine AUC</div></div>';}).join('');});},

    /* §4 — 2x2 economy x domain grid (cosine AUC vs probe AUC, Gemini) */
    twobytwo:function(el){el=E(el);this.get('data/RESULTS_CANONICAL.json').then(function(rc){el=E(el);if(!el)return;
      function cell(c,econ,dom){var x=rc[c].two_by_two_6_6[dom];var co=x.cosine_auc_gemini,pr=x.probe_auc_gemini;
        return '<div class="ax-cell"><div class="ax-cellh">'+econ+' · '+dom+'</div>'+
          '<div class="ax-bar"><span class="lab">cosine</span><span class="track"><span class="fill coral" style="width:'+Math.round(co*100)+'%"></span></span><span class="val coral">'+co.toFixed(2)+'</span></div>'+
          '<div class="ax-bar"><span class="lab">probe</span><span class="track"><span class="fill teal" style="width:'+Math.round(pr*100)+'%"></span></span><span class="val teal">'+pr.toFixed(2)+'</span></div>'+
          '<div class="ax-celln">'+x.n_cats.toLocaleString()+' categories</div></div>';}
      el.innerHTML='<div class="ax-2x2">'+cell('CA','Canada','consumption')+cell('CA','Canada','production')+cell('US','United States','consumption')+cell('US','United States','production')+'</div>'+
        '<div class="ax-figcap" style="margin-top:10px"><b>All four cells tell the same story:</b> cosine AUC sits below 0.50 (the pathology, coral); the probe climbs above it (recovery, teal). Consumption and production, Canada and the United States.</div>';});},

    /* §5 — representation report-card: cosine vs probe AUC per representation */
    reportcard:function(el){el=E(el);this.get('data/RESULTS_CANONICAL.json').then(function(rc){el=E(el);if(!el)return;var h=rc.CA.headline_6_4,p=rc.CA.probe_6_5;
      var rows=REPS.map(function(r){var co=h[r].comp_vs_sub_cosine_auc,pr=p[r].probe_auc;var trans=(r==='gemini'||r==='openai');
        return '<tr><td>'+RNAME[r]+(trans?' <span class="ax-badge" style="font-size:10px;padding:1px 6px">transformer</span>':'')+'</td>'+
          '<td class="num"><span class="ax-chip coral">'+co.toFixed(2)+'</span></td><td class="num"><span class="ax-chip teal">'+pr.toFixed(2)+'</span></td><td class="num ax-muted">'+(pr-co).toFixed(2)+'</td></tr>';}).join('');
      el.innerHTML='<table class="ax-table"><thead><tr><th>representation</th><th class="num">cosine AUC</th><th class="num">probe AUC</th><th class="num">gain</th></tr></thead><tbody>'+rows+'</tbody></table>'+
        '<div class="ax-figcap" style="margin-top:8px">The two <b>transformer</b> embeddings are the <b>worst</b> on raw cosine (0.06) yet the <b>best</b> under the probe (0.91) — the pathology is sharpest exactly where the recovery is strongest.</div>';});},

    /* §5 — evidence board (claim ledger) */
    evidenceBoard:function(el){el=E(el);el=E(el);if(!el)return;
      var rows=[
       ["Pathology — comp-vs-sub cosine AUC (transformers)","below 0.50","0.06 gem / 0.06 ope","ok","0.12 gem / 0.13 ope","ok"],
       ["Probe recovery — AUC","above 0.50","0.91","ok","0.85","ok"],
       ["Consumer · complement vs random (Amazon / Yelp)","above 0.50","0.66 / 0.58","ok","0.63 Amazon","ok"],
       ["Consumer · substitute-over-complement cosine","above 0.50","0.59 / 0.93","ok","0.60 Amazon","ok"],
       ["Consumer · concordance κ (census vs Amazon)","above 0","0.31 / 0.22","ok","0.01 / 0.02","grey"],
       ["Production · complement vs random (I-O)","above 0.50","0.63 StatCan '23","ok","0.57–0.58 BEA '17","warn"],
       ["Production · cosine separates I-O complements","above 0.50","0.58–0.72","ok","≈0.50 BEA","warn"],
       ["Robustness · generator (finding holds)","holds","4o + Gemini","ok","4o ∩ Gemini","ok"]];
      var map={ok:'cell-info',warn:'cell-warn',grey:'cell-grey'};
      var h='<table class="ax-table evid"><thead><tr><th>test</th><th>want</th><th>Canada</th><th>United States</th></tr></thead><tbody>';
      rows.forEach(function(r){h+='<tr><td>'+r[0]+'</td><td class="ax-muted">'+r[1]+'</td><td class="'+map[r[3]]+'">'+r[2]+'</td><td class="'+map[r[5]]+'">'+r[4]+'</td></tr>';});
      el.innerHTML=h+'</tbody></table><div class="ax-figcap" style="margin-top:8px"><b>Blue</b> clears the benchmark · <b>amber</b> caveated · <b>grey</b> outstanding. Every in-model and consumer cell clears; production is supported on StatCan and caveated on BEA under crosswalk limits.</div>';},

    /* §6 — external validation cards */
    validationCards:function(el){el=E(el);if(!el)return;
      var cards=[
        {src:"amazon",t:"Amazon co-purchase",k:"Behavioral · consumer",a:"0.66",b:"complement-vs-random AUC (also-buy / also-view lift)"},
        {src:"yelp",t:"Yelp co-visitation",k:"Behavioral · consumer",a:"0.58–0.93",b:"complement signal across a user's reviewed categories"},
        {src:"statcan",t:"StatCan Input–Output",k:"Economic · production",a:"0.63",b:"complement-vs-random, 2023 symmetric I-O (Canada)"},
        {src:"bea",t:"BEA Input–Output",k:"Economic · production",a:"0.57–0.58",b:"complement-vs-random, 2017 benchmark detail (US)"}];
      el.innerHTML='<div class="ax-grid" style="grid-template-columns:repeat(auto-fit,minmax(230px,1fr))">'+cards.map(function(c){
        return '<a class="ax-card pad ax-vcard" href="validation-source.html?src='+c.src+'"><div class="ax-badge teal" style="margin-bottom:8px">'+c.k+'</div>'+
          '<div style="font-size:15px;font-weight:640;margin-bottom:2px;color:var(--fg)">'+c.t+'</div>'+
          '<div style="font-size:28px;font-weight:720;color:var(--teal);font-variant-numeric:tabular-nums;letter-spacing:-.02em">'+c.a+'</div>'+
          '<div class="ax-muted" style="font-size:12.5px;line-height:1.4;margin-top:2px">'+c.b+'</div>'+
          '<div class="ax-vmore">Break down the data →</div></a>';}).join('')+'</div>';},

    /* §7 — graph stat tiles + reciprocal pairs */
    graphStats:function(el){el=E(el);var s=this;Promise.all([this.get('data/graph_CA.json'),this.get('data/RESULTS_CANONICAL.json')]).then(function(a){el=E(el);if(!el)return;var g=a[0],rc=a[1];
      var hub=rc.CA.dependence_5_1.top_hub, rec=g.reciprocity;
      var tiles='<div class="ax-stats">'+
        '<div class="ax-stat"><div class="v">'+g.n_communities+'</div><div class="l">community clusters (Louvain)</div></div>'+
        '<div class="ax-stat"><div class="v">'+g.modularity.toFixed(2)+'</div><div class="l">modularity — strong structure</div></div>'+
        '<div class="ax-stat"><div class="v">'+(rec.pct_of_edges*100).toFixed(1).replace(/\.0$/,'')+'%</div><div class="l">'+rec.n_unique_pairs+' mutual (reciprocal) pairs</div></div>'+
        '<div class="ax-stat"><div class="v">'+hub.degree+'</div><div class="l">top hub — '+esc(hub.name)+'</div></div></div>';
      var pairs=(rec.pairs||[]).slice(0,5).map(function(p){return '<div class="ax-pair"><span class="dot" style="background:var(--teal)"></span><span><b>'+esc(p.title_a)+'</b> ⇄ '+esc(p.title_b)+'</span></div>';}).join('');
      el.innerHTML=tiles+'<div class="ax-card pad" style="margin-top:14px"><div style="font-size:13px;font-weight:640;margin-bottom:6px">A few mutual pairs <span class="ax-muted" style="font-weight:400">— each names the other as a complement</span></div>'+pairs+'</div>';});},

    /* §7 / §3 — surprising pairs leaderboard from the probe scatter points */
    leaderboard:function(el,d){el=E(el);if(!el)return;
      var c=d.points.filter(function(p){return p.t==='c'&&p.y>d.cutoff;}).sort(function(a,b){return a.x-b.x;}).slice(0,7);
      el.innerHTML=c.map(function(p){return '<div class="ax-pair"><span class="dot" style="background:var(--teal)"></span><span><b>'+esc(p.f)+'</b> + '+esc(p.p)+'</span><span class="num">cos '+p.x.toFixed(2)+'</span></div>';}).join('')||'<span class="ax-muted">—</span>';}
  };
  window.AX=AX;
  window.addEventListener('cv2:change',function(e){if(e.detail.key==='theme')setTimeout(function(){AX._charts.forEach(function(c){c.fn();});},30);});
})();
