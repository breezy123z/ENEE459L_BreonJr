import sys, json, csv, shutil, zipfile, html
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from tensors import load_model
import prune

out=ROOT / 'charts'; out.mkdir(exist_ok=True)
source=ROOT / 'prune_outputs.json'
report=json.loads(source.read_text()); f=report['functions']
model=load_model(ROOT / 'model.json'); ts=model['tensors']
rows=f['sweep_model']; fine=[r for r in rows if r['granularity']=='fine']; channel=[r for r in rows if r['granularity']=='channel']
ratios=[r['nominal_ratio']*100 for r in fine]
blue='#2563eb'; orange='#ea580c'; green='#15803d'; gray='#94a3b8'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':16,'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.labelcolor':'#334155','text.color':'#0f172a','savefig.facecolor':'white'})
items=[]
def save(fig, name, title, explanation):
 fig.text(.02,.012,'ENEE459L Lab 5 | tinyconv | generated weights | October 7, 2026',fontsize=9,color='#64748b')
 fig.tight_layout(rect=[0,.04,1,.98]); fig.savefig(out/(name+'.png'),dpi=180); fig.savefig(out/(name+'.svg')); plt.close(fig)
 items.append((name,title,explanation))

fig,axs=plt.subplots(1,2,figsize=(13,5))
for ax,key,title in [(axs[0],'parameters_after','Parameter slots after pruning'),(axs[1],'values_zeroed','Zero values still stored')]:
 ax.plot(ratios,[r[key] for r in fine],'o-',color=blue,label='Fine: mask weights'); ax.plot(ratios,[r[key] for r in channel],'s-',color=orange,label='Channel: remove slices'); ax.set(title=title,xlabel='Requested pruning (%)',ylabel='Count',xticks=ratios); ax.grid(alpha=.18); ax.legend()
save(fig,'01-parameters-and-zeros','Parameters and zeros answer different questions','Masking leaves all 11,740 parameter slots in place. Channel pruning removes slots. Values removed with a channel are absent and therefore do not count as stored zeros.')

fig,ax=plt.subplots(figsize=(10,6))
ax.plot(ratios,ratios,'--',color=gray,label='Requested = achieved')
for r,color,label in [(fine,blue,'Fine masking'),(channel,orange,'Channel removal')]: ax.plot(ratios,[x['achieved_reduction']*100 for x in r],'o-',color=color,label=label)
for r in channel: ax.annotate(f"{r['achieved_reduction']*100:.3f}%",(r['nominal_ratio']*100,r['achieved_reduction']*100),xytext=(7,-16),textcoords='offset points',fontsize=10)
ax.set(title='Requested pruning vs. actual parameter reduction',xlabel='Requested pruning (%)',ylabel='Achieved parameter reduction (%)',xticks=ratios); ax.grid(alpha=.18); ax.legend()
save(fig,'02-requested-vs-achieved','Requested percentage is not achieved reduction','Fine masking achieves 0% shape reduction at every ratio. Channel counts must be whole numbers; the 10-channel fully connected tensor rounds 25% to three channels and 75% to eight. Total reductions are 25.1704% and 75.1704%.')

fig,ax=plt.subplots(figsize=(11,6))
for storage,encoding,label,color in [('dense','framework','Fine: dense weights',blue),('masked','framework','Fine: weights + framework mask','#7c3aed'),('masked','bitmap','Fine: weights + bitmap','#0891b2'),('sparse','framework','Fine: values + sparse indices',green)]:
 vals=[]
 for r in ratios:
  masked=[prune.apply_mask(t,prune.magnitude_mask(t,r/100)) for t in ts]
  vals.append(prune.bytes_stored(masked,storage,encoding))
 ax.plot(ratios,vals,'o-',label=label,color=color)
ax.plot(ratios,[r['bytes_stored'] for r in channel],'s-',color=orange,label='Channel: dense remaining weights')
ax.set(title='Storage across pruning ratios',xlabel='Requested pruning (%)',ylabel='Accounted storage (bytes)',xticks=ratios); ax.grid(alpha=.18); ax.legend(fontsize=9)
save(fig,'03-storage-sweep','Storage depends on representation','Dense masked weights stay at 46,960 bytes. A framework mask doubles storage to 93,920. A bitmap adds 1,468 bytes. Sparse storage costs four bytes per FP32 value plus four bytes per index, so 50% sparsity only breaks even with the original dense model. These are accounting estimates, excluding container overhead and sparse row pointers.')

labels=['Original dense','Fine dense','Fine framework','Fine bitmap','Fine sparse','Channel dense','Channel sparse']
values=[46960,f['bytes_stored']['fine_pruned']['dense'],f['bytes_stored']['fine_pruned']['masked_framework'],f['bytes_stored']['fine_pruned']['masked_bitmap'],f['bytes_stored']['fine_pruned']['sparse'],f['bytes_stored']['channel_pruned']['dense'],f['bytes_stored']['channel_pruned']['sparse']]
fig,ax=plt.subplots(figsize=(11,6)); bars=ax.barh(labels,values,color=[gray,blue,'#7c3aed','#0891b2',green,orange,'#fbbf24']); ax.invert_yaxis(); ax.bar_label(bars,labels=[f'{v:,} bytes ({v/46960:.2f}x)' for v in values],padding=6,fontsize=10); ax.set_xlim(0,max(values)*1.4); ax.axvline(46960,color=gray,ls='--'); ax.set(title='50% pruning: storage comparison',xlabel='Accounted storage (bytes)')
save(fig,'04-storage-at-50-percent','50% pruning does not guarantee 50% storage savings','Channel removal halves dense storage. Framework-masked storage is twice the original. Sparse channel tensors still carry an index for every remaining nonzero, doubling their own dense size.')

fig,ax=plt.subplots(figsize=(10,6)); names=[t.name.replace('.weight','') for t in ts]; base=[t.parameters for t in ts]; remaining=[f['channel_keep_and_drop_channels'][t.name]['tensor_after']['parameters'] for t in ts]; x=np.arange(len(ts)); w=.26
for offset,vals,label,color in [(-w,base,'Original',gray),(0,base,'Fine mask: 50%',blue),(w,remaining,'Channel: 50%',orange)]:
 bars=ax.bar(x+offset,vals,w,label=label,color=color); ax.bar_label(bars,padding=3,fontsize=9)
ax.set(title='Parameter count by tensor at 50% pruning',ylabel='Parameter slots',xticks=x,xticklabels=names); ax.legend(); ax.set_ylim(0,max(base)*1.22)
save(fig,'05-layer-parameters','Where the parameters are','conv3 holds 7,200 of the original 11,740 parameters, followed by conv2 with 3,600. At 50%, every channel-pruned tensor halves its parameter count; fine masking preserves each shape.')

fig,axs=plt.subplots(2,2,figsize=(13,8))
for ax,t in zip(axs.flat,ts):
 scores=f['helpers']['channel_group_scores'][t.name]['scores']; kept=set(f['channel_keep_and_drop_channels'][t.name]['channels_kept']); ax.bar(range(t.channels),scores,color=[orange if i in kept else gray for i in range(t.channels)]); ax.set(title=t.name,xlabel='Original output channel index',ylabel='L2 norm')
fig.suptitle('Channel selection at 50%: orange kept, gray removed',fontsize=17,fontweight='bold')
save(fig,'06-channel-importance','Which channels survive','Channel pruning ranks the L2 norm of all weights in each channel. Kept indices are 10–19 in conv1 and conv2, 20–39 in conv3, and 5–9 in fc. This importance score is a heuristic and does not establish accuracy.')

fig,axs=plt.subplots(2,2,figsize=(13,8)); cmap=ListedColormap(['#e2e8f0',blue])
for ax,t in zip(axs.flat,ts):
 mask=np.array(f['magnitude_mask_and_apply_mask'][t.name]['mask']).reshape(t.channels,t.channel_stride)
 ax.imshow(mask,aspect='auto',interpolation='nearest',cmap=cmap,vmin=0,vmax=1); ax.set(title=f'{t.name} | {t.channels} x {t.channel_stride}',xlabel='Weight position within channel',ylabel='Output channel')
fig.suptitle('Fine pruning at 50%: blue kept, light gray zeroed',fontsize=17,fontweight='bold')
save(fig,'07-fine-mask-map','Unstructured pruning leaves holes','Each row is an output channel; each column is one weight within it. Exactly half the weights are masked per tensor. Scattered holes do not make the tensor shape smaller or remove the dense kernel workload.')

fig,ax=plt.subplots(figsize=(11,6)); labels=['Original dense','Fine framework mask','Fine bitmap mask','Fine sparse','Channel dense']; weights=[46960,46960,46960,23480,23480]; extra=[0,46960,1468,23480,0]; x=np.arange(5)
b=ax.bar(x,weights,color=blue,label='Stored weight values'); e=ax.bar(x,extra,bottom=weights,color='#a78bfa',label='Mask or sparse indices')
for i,(v,w) in enumerate(zip(weights,extra)): ax.text(i,v+w+1500,f'{v+w:,}',ha='center',fontsize=11)
ax.set(title='50% pruning: separating values from overhead',ylabel='Accounted bytes',xticks=x,xticklabels=labels); ax.legend(); ax.set_ylim(0,110000)
save(fig,'08-storage-breakdown','Representation overhead can erase savings','The sparse FP32 representation stores 23,480 bytes of nonzero values plus 23,480 bytes of indices. The framework mask alone costs 46,960 bytes. Bitmap masks round up to whole bytes separately for each tensor.')

shutil.copy2(source,out/'prune_outputs.json')
with (out/'sweep-results.csv').open('w',newline='') as file:
 writer=csv.DictWriter(file,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
sections=''.join(f'<section><h2>{i+1}. {html.escape(title)}</h2><img src="{name}.png" alt="{html.escape(title)}"><p>{html.escape(explanation)}</p><a href="{name}.svg">Vector SVG</a> · <a href="{name}.png">PNG</a></section>' for i,(name,title,explanation) in enumerate(items))
(out/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lab 5 chart bundle</title><style>body{font:17px/1.6 system-ui;color:#183047;background:#f1f5f9;margin:0}main{max-width:1150px;margin:auto;padding:40px 24px}h1{font-size:38px;line-height:1.2}h2{font-size:25px}section,.intro{background:white;padding:26px;margin:24px 0;border-radius:14px;border:1px solid #dbe2eb}img{width:100%;height:auto}a{color:#2563eb}small{color:#526579}@media print{section{break-inside:avoid}body{background:white}main{padding:0}}</style><main><h1>Lab 5 · What actually shrinks?</h1><p>Detailed pruning charts for ENEE459L · October 7, 2026</p><div class="intro"><b>Verified:</b> all 12 tests passed on the Jetson; the generated report matched the instructor sample 100%.<p>Original model: 11,740 parameters and 46,960 FP32 bytes. Fine pruning masks individual weights. Channel pruning removes whole output-channel slices.</p><p>The four-ratio sweep comes from the generated report. Additional storage representations are calculated using the tested Lab 5 functions on the same deterministic tensors. No inference latency or accuracy was measured; tensors are pruned independently, so this does not demonstrate a runnable end-to-end pruned network.</p><a href="sweep-results.csv">Sweep data (CSV)</a> · <a href="prune_outputs.json">Full results (JSON)</a> · <a href="https://github.com/breezy123z/ENEE459L_BreonJr/tree/solution5/lab05">Source code</a></div>'''+sections+'</main></html>',encoding='utf-8')
(out/'README.txt').write_text('Open index.html to read the eight-chart report. Each chart has PNG and SVG exports. Data: sweep-results.csv and prune_outputs.json. Calculated storage only; no accuracy or inference performance measurement. Verified source branch: solution5, commit 7268cfe.\n',encoding='utf-8')
with zipfile.ZipFile(ROOT / 'lab05-chart-bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in out.iterdir(): z.write(p,'lab05-chart-bundle/'+p.name)
print('Created',len(items),'charts, HTML report, CSV, JSON, and ZIP.')
