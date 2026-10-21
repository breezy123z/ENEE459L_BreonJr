"""Build figures and a six-page report from computed Assignment 2 outputs."""
from pathlib import Path
import csv,json
from dataclasses import replace
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from graph import load_graph,Graph
from arithmetic import arithmetic_intensity
from main import traffic_breakdown
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT

ROOT=Path(__file__).resolve().parent
FIG=ROOT/'charts'; FIG.mkdir(exist_ok=True)
g=load_graph(ROOT/'model.json')
r=arithmetic_intensity(g)
rows=traffic_breakdown(g)
for row in rows: row.update(r['per_layer'][row['layer']])
BLUE='#2563eb'; TEAL='#0f766e'; ORANGE='#ea580c'; GRAY='#94a3b8'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':14,'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})

def save(fig,name):
 fig.tight_layout(); fig.savefig(FIG/(name+'.png'),dpi=200); fig.savefig(FIG/(name+'.svg')); plt.close(fig)

names=[row['layer'] for row in rows]
fig,ax=plt.subplots(figsize=(9,4))
bars=ax.barh(names,[row['intensity'] for row in rows],color=[BLUE if row['flops'] else GRAY for row in rows]); ax.invert_yaxis(); ax.bar_label(bars,padding=4,fmt='%.4f'); ax.axvline(r['value'],color=TEAL,ls='--',label=f"Whole graph: {r['value']:.3f}"); ax.set(xlabel='Arithmetic intensity (FLOPs/byte)',title='Convolutions do more work per byte'); ax.set_xlim(0,40); ax.legend(loc='lower right'); save(fig,'01-layer-intensity')

fig,ax=plt.subplots(figsize=(9,4))
left=np.zeros(len(rows))
for key,label,color in [('input_bytes','Input reads',BLUE),('output_bytes','Output writes',TEAL),('weight_bytes','Weights',ORANGE),('buffer_bytes','BN state buffers','#a855f7')]:
 vals=np.array([row[key]/1024 for row in rows]); ax.barh(names,vals,left=left,label=label,color=color); left+=vals
ax.invert_yaxis(); ax.set(xlabel='Read + write traffic (KiB, 1 KiB = 1024 bytes)',title='Activation traffic dominates the unfused graph'); ax.legend(ncol=2,loc='lower right',fontsize=9); save(fig,'02-traffic-components')

fig,ax=plt.subplots(figsize=(9,4)); x=np.arange(len(rows)); w=.36
ax.bar(x-w/2,[100*row['flops']/r['total_flops'] for row in rows],w,color=BLUE,label='Share of FLOPs')
ax.bar(x+w/2,[100*row['bytes']/r['total_bytes'] for row in rows],w,color=ORANGE,label='Share of bytes')
ax.set(xticks=x,xticklabels=names,ylabel='Share of whole graph (%)',title='A layer can move many bytes while contributing zero MAC-based FLOPs'); ax.tick_params(axis='x',rotation=25); ax.legend(); save(fig,'03-work-versus-traffic')

half=Graph('toy_resnet_fp16',g.input_shape,[replace(layer,weight_dtype='fp16',act_dtype='fp16') for layer in g],precision='fp16')
h=arithmetic_intensity(half)
(ROOT/'results/precision_scenarios.json').write_text(json.dumps({'description':'Hypothetical all-FP16 weights/activations/input; BN state remains FP32; same MACs. Not a runtime benchmark.','fp32':r,'fp16':h},indent=2)+'\n',encoding='utf-8')
fig,axs=plt.subplots(1,2,figsize=(9,3.6))
for ax,vals,title,ylabel in [(axs[0],[r['total_bytes']/1024,h['total_bytes']/1024],'Same graph, fewer bytes','Traffic (KiB)'),(axs[1],[r['value'],h['value']],'Same FLOPs, higher intensity','FLOPs/byte')]:
 bars=ax.bar(['FP32','FP16'],vals,color=[BLUE,TEAL]); ax.bar_label(bars,fmt='%.3f',padding=4); ax.set(title=title,ylabel=ylabel); ax.set_ylim(0,max(vals)*1.23)
save(fig,'04-precision-sensitivity')

styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleCustom',fontName='Helvetica-Bold',fontSize=25,leading=30,textColor=colors.HexColor('#0f172a'),spaceAfter=14))
styles.add(ParagraphStyle(name='Deck',fontName='Helvetica',fontSize=12,leading=17,textColor=colors.HexColor('#475569'),spaceAfter=14))
styles['BodyText'].fontSize=10; styles['BodyText'].leading=14; styles['BodyText'].spaceAfter=8
styles['Heading2'].textColor=colors.HexColor(TEAL); styles['Heading2'].spaceBefore=10
story=[]
def P(text,style='BodyText'): story.append(Paragraph(text,styles[style]))
def title(k,t): P(f'ENEE459L / ASSIGNMENT 2 / {k}','Deck'); P(t,'TitleCustom')
def chart(name): story.append(Image(str(FIG/(name+'.png')),width=480,height=480*(plt.imread(FIG/(name+'.png')).shape[0]/plt.imread(FIG/(name+'.png')).shape[1]))); story.append(Spacer(1,8))
def table(data,widths):
 tab=Table([[Paragraph(str(x),styles['BodyText']) for x in row] for row in data],colWidths=widths,repeatRows=1)
 tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e2e8f0')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#94a3b8')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f8fafc')]),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),3)])); story.append(tab); story.append(Spacer(1,10))
def nextpage(): story.append(PageBreak())

title('01 / PURPOSE & RESULT','Arithmetic intensity: work per byte')
P('Computed analysis and implementation | October 21, 2026','Deck')
P('<b>Task.</b> Implement arithmetic_intensity(graph) to report FLOPs per byte for each layer and the full graph. Every layer reads its inputs and stored parameters/buffers, then writes its output. Assume no cache reuse and no operator fusion.')
P(f"<b>Demonstration result:</b> {r['total_flops']:,} FLOPs / {r['total_bytes']:,.0f} bytes = <b>{r['value']:.4f} FLOPs/byte</b>.")
table([['Requirement','Implementation'],['Input precision','First named producer; otherwise predecessor; external input uses graph.precision.'],['Two-input add','Double input bytes using the first producer dtype, exactly as specified.'],['BatchNorm state','Add two FP32 running-statistic values per output channel.'],['Zero-FLOP layers','Include their byte costs and report intensity 0.0.'],['Output contract','computed(...) returns type, value, message, per_layer and all required metadata.'],['Whole graph','Sum unrounded FLOPs and bytes before dividing and rounding to 4 decimals.']],[130,350])
P('<b>Evidence.</b> 15 independent unit tests pass. The example model is the eight-layer synthetic toy_resnet from Lab 4, with shape-derived MACs. It is not a measured inference run or an official Assignment 2 model fixture.')
P('<b>Source.</b> Assign2.pdf, four-page handout dated October 19, 2026; especially output requirements on page 3 and zero-FLOP handling on page 4. Code, data, tests, and figures are included in this submission.')
nextpage()

title('02 / PER-LAYER INTENSITY','Convolution has more work to amortize its traffic')
chart('01-layer-intensity')
P('The residual convolution performs 4,718,592 FLOPs while moving 140,288 bytes: <b>33.6350 FLOPs/byte</b>. The first convolution reaches 11.1215. Its smaller input-channel count means less arithmetic per output value.')
P('The fully connected head reaches only 0.4082 FLOPs/byte because its small input/output vectors provide little reuse of stored weights. ReLU and other non-MAC operators remain in the report at zero intensity.')
P('<b>Important convention.</b> The PDF refers to _layer_macs but does not supply its body. The bundled helper counts convolution and linear MACs; BN, add, pooling, flatten, and ReLU return zero under this MAC-only example convention. This does not mean every real implementation performs literally zero arithmetic. Replace the helper if the course supplies a different operator-count convention.')
P('<b>Hardware interpretation.</b> Higher intensity makes it easier to amortize memory traffic. Declaring an operator compute-bound requires a device compute/bandwidth ratio for comparison. No such hardware ceiling or latency measurement is claimed here.')
nextpage()

title('03 / DRAM ACCOUNTING','Every unfused intermediate has a read and a write')
chart('02-traffic-components')
activation=sum(row['input_bytes']+row['output_bytes'] for row in rows)
P(f"Inputs and outputs account for <b>{activation:,.0f} bytes ({activation/r['total_bytes']*100:.2f}%)</b> of total traffic. Weights and BN buffers contribute the remaining {r['total_bytes']-activation:,.0f} bytes.")
P('The add layer reads two FP32 streams (131,072 bytes) and writes one result (65,536 bytes), totaling <b>196,608 bytes</b>. It has the largest byte cost despite zero MAC-based FLOPs in this example.')
P('BatchNorm adds 128 bytes of FP32 running mean/variance alongside its 128-byte trainable scale/shift payload. These buffers must not be omitted or priced at the activation dtype.')
P('<b>Meaning of “once.”</b> Each layer access is charged once. A producer output is written, then charged again when a consumer reads it. Shared tensors read by two different layers are charged for both accesses. The sum is traffic over a graph execution, not peak resident memory or stored-model size.')
nextpage()

title('04 / AGGREGATION','FLOP share and traffic share tell different stories')
chart('03-work-versus-traffic')
nonmac=sum(row['bytes'] for row in rows if row['flops']==0)
P(f"Zero-MAC layers contribute <b>{nonmac:,.0f} bytes ({nonmac/r['total_bytes']*100:.2f}%)</b> of traffic. Omitting them would inflate the graph intensity and hide the cost of processing intermediate tensors.")
P('<b>Correct aggregate:</b> total FLOPs / total bytes. Equivalently, this is the byte-weighted mean of unrounded per-layer intensities. A simple mean would give a tiny flatten layer the same influence as a large convolution and is not the required result.')
table([['Layer','FLOPs','Bytes','FLOPs/byte']]+[[row['layer'],f"{row['flops']:,}",f"{row['bytes']:,.0f}",f"{row['intensity']:.4f}"] for row in rows],[96,128,128,128])
nextpage()

title('05 / PRECISION SCENARIO','Reducing byte width raises the ratio mathematically')
chart('04-precision-sensitivity')
P(f"Hypothetically changing input, activations, and weights to FP16 reduces traffic to <b>{h['total_bytes']:,.0f} bytes</b> and raises intensity to <b>{h['value']:.4f} FLOPs/byte</b>. FLOPs stay at {h['total_flops']:,}; only the denominator changes.")
P('Traffic is slightly more than half of the FP32 baseline because BN state buffers stay FP32. This is a calculated sensitivity analysis, not a claim of measured speedup, preserved numerical accuracy, or hardware execution support.')
P('<b>Residual precision rule.</b> For two-input addition the assignment explicitly doubles the first producer\'s byte width. That assumes equal branch widths for physically exact traffic. For unequal branch dtypes, real traffic would require separately pricing each input. The implementation deliberately retains the stated formula, with a test exposing this convention.')
P('<b>Other limits.</b> No cache reuse, fusion, workspace, layout conversions, allocator overhead, kernel-launch overhead, or training gradients are modeled. Operators may be implemented differently by a real runtime. This report estimates the handout\'s defined traffic, not actual DRAM counter measurements.')
nextpage()

title('06 / REPRODUCIBILITY','Implementation, verification, and submission map')
P('<b>Run the computation:</b><br/><font name="Courier">python main.py</font><br/>Creates results/intensity_results.json and results/layer_results.csv. Use --model PATH to analyze another compatible graph JSON.')
P('<b>Run tests:</b><br/><font name="Courier">python -m unittest -v test_arithmetic.py</font><br/>The core code needs only Python 3.10+. A Jetson, camera, and CUDA are not required.')
P('<b>Regenerate this report and charts:</b><br/><font name="Courier">python -m pip install -r requirements-report.txt<br/>python build_report.py</font>')
table([['Files','Purpose'],['arithmetic.py','Required function, assignment-specific computed wrapper, and documented standalone helpers.'],['graph.py / model.json','Instructor Lab 4 graph utilities plus an explicit synthetic example description.'],['test_arithmetic.py / test-results.txt','15 independent tests and saved run.'],['results/','Machine-readable layer totals, traffic breakdown, and precision scenarios.'],['charts/','Four figures in PNG and SVG formats.'],['Assignment2_Report.pdf','This explanation and findings.'],['README.md','Requirement checklist, provenance, assumptions, and usage.']],[170,310])
P('<b>Test coverage.</b> Hand-calculated linear and grouped-convolution cases, external/implicit/explicit dtype resolution, dual add, mixed-dtype rule, FP32 BN buffers, zero-FLOP layers, weighted aggregation, fractional INT4 accounting, injected MAC helper, zero bytes, empty graph, exact record schema, invalid dependencies, and duplicate names.')
P('<b>Integration.</b> The PDF requires type/message fields; the older Lab 4 computed wrapper uses status/source. arithmetic.py supplies the requested wrapper explicitly. In a course starter with existing helpers, retain those imports and copy the required function plus its dtype-resolution helper. Keep the required note and metadata flags.')
P('<b>Authorship and provenance.</b> arithmetic.py, tests, runner, analysis, and report were developed for this assignment. graph.py comes from the instructor\'s Lab 4 starter. No timing, hardware bottleneck classification, or neural-network accuracy result was invented.')

def footer(canvas,doc):
 canvas.setStrokeColor(colors.HexColor('#cbd5e1')); canvas.line(48,43,564,43)
 canvas.setFont('Helvetica',8); canvas.setFillColor(colors.HexColor('#64748b'))
 canvas.drawString(48,29,'Assignment 2 | computed model analysis | October 21, 2026')
 canvas.drawRightString(564,29,str(doc.page))

pdf=ROOT/'Assignment2_Report.pdf'
doc=SimpleDocTemplate(str(pdf),pagesize=(612,792),rightMargin=60,leftMargin=60,topMargin=43,bottomMargin=57)
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print('Created report and four charts:',pdf)
