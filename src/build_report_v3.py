"""Offline Korean report, executed results + reproducibility ZIP + appendix PNG."""
from pathlib import Path
import argparse,base64,hashlib,html,json,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_report import setup,table,db
from protocol_v3 import ROOT,OUT

REPORT=ROOT/'report_v3';FIG=REPORT/'figures'
METHODS=['uncorrected','local_gpga','per_ping_entropy']
LABELS=['무보정','국소 GPGA 계열','핑별 엔트로피']
COLORS=['#8b99a5','#bc924f','#16788a']


def read(name):return json.loads((OUT/name).read_text())
def figsave(fig,name):fig.savefig(FIG/name,dpi=155,bbox_inches='tight');plt.close(fig)
def picture(name,alt):return '<figure><img class="plot" alt="'+html.escape(alt)+'" src="data:image/png;base64,'+base64.b64encode((FIG/name).read_bytes()).decode()+'"><figcaption>'+html.escape(alt)+'</figcaption></figure>'
def pct(v):return f'{100*v:.1f}%'
def interval(v):return f'{100*v[0]:.1f}–{100*v[1]:.1f}%'


def figures(af,det,bridge,common):
 setup();FIG.mkdir(parents=True,exist_ok=True)
 fig,axs=plt.subplots(1,2,figsize=(11,3.9),layout='constrained')
 for i,(method,label,color) in enumerate(zip(METHODS,LABELS,COLORS)):
  for b in range(3):
   v=[r['complex_correlation'] for r in af['rows'] if r['amount'] and r['method']==method and r['range_bin']==b]
   pos=b*4+i;axs[0].boxplot(v,positions=[pos],widths=.65,patch_artist=True,boxprops={'facecolor':color,'alpha':.65},medianprops={'color':'#203344'},flierprops={'markersize':2})
  v=[1-r['complex_correlation'] for r in af['rows'] if not r['amount'] and r['method']==method]
  axs[1].scatter(np.arange(1,25),v,color=color,s=19,label=label)
 axs[0].set(xticks=[1,5,9],xticklabels=['15–35 m','35–55 m','55–80 m'],ylim=(-.03,1.05),ylabel='무오차 자체 BP와 complex NCC',title='좌우 2 mm RMS · 24점 × 새 경로 5개')
 axs[0].axhline(.9,color='#b44845',ls='--',lw=1,label='성공 기준 0.90');axs[0].legend(frameon=False,fontsize=8)
 axs[1].set(xlabel='고정된 시험 점 순서',ylabel='무오차 입력의 NCC 감소',title='무오차 입력 보호 · 점 24개')
 axs[1].axhline(.01,color='#b44845',ls='--',lw=1,label='허용 최대 0.01');axs[1].legend(frameon=False,fontsize=8)
 figsave(fig,'autofocus_range_clean.png')
 ex=np.load(OUT/'autofocus_examples.npz');fig,axs=plt.subplots(1,4,figsize=(11,3.1),layout='constrained',sharex=True,sharey=True)
 for ax,key,label in zip(axs,['reference',*[f'sway2mm_{m}' for m in METHODS]],['무오차 기준',*LABELS]):
  im=ax.imshow(db(ex[key],abs(ex['reference']).max()),origin='lower',cmap='gray',vmin=-40,vmax=0,
   extent=[ex['x'][0],ex['x'][-1],ex['y'][0],ex['y'][-1]],aspect='equal')
  ax.set_title(label);ax.set_xlabel('Along track (m)');ax.tick_params(labelsize=7)
 axs[0].set_ylabel('Cross track (m)');fig.colorbar(im,ax=axs,shrink=.75,label='무오차 peak 기준 amplitude dB');figsave(fig,'autofocus_example.png')
 fig,ax=plt.subplots(figsize=(8.5,3.4),layout='constrained')
 for j,(key,label,col) in enumerate([('common_all_ranges','전체 거리 공통 위상','#b0b8c0'),('common_within_range_band','거리 구간 공통 위상','#c49a55')]):
  vals=[np.median([r['complex_correlation'] for r in common['rows'] if r['method']==key and r['range_bin']==b]) for b in range(3)]
  ax.bar(np.arange(3)+j*.24-.24,vals,width=.23,label=label,color=col)
 vals=[np.median([r['complex_correlation'] for r in af['rows'] if r['amount'] and r['method']=='local_gpga' and r['range_bin']==b]) for b in range(3)]
 ax.bar(np.arange(3)+.24,vals,width=.23,label='점별 국소 GPGA',color=COLORS[1])
 ax.set(xticks=range(3),xticklabels=['15–35 m','35–55 m','55–80 m'],ylim=(0,1.05),ylabel='complex NCC 중앙값',title='사후 진단: 같은 위상을 어디까지 공유할 수 있는가')
 ax.legend(frameon=False,fontsize=9);figsave(fig,'common_phase.png')
 fig,axs=plt.subplots(1,2,figsize=(11,3.8),layout='constrained')
 for key,label,color in [('froc_proposal','대비 후보만','#8b99a5'),('froc_rf','후보 + RF 위치 탐지','#16788a'),('froc_class_aware','후보 + RF 위치·분류','#bc924f')]:
  curve=det[key];xx=[r['fp_per_scene'] for r in curve];yy=[r['recall'] for r in curve]
  for ax in axs:ax.step(xx,yy,where='post',label=label,color=color,lw=1.6)
 op=det['operating_point_rf'];lo,hi=op['wilson95']
 axs[0].errorbar(op['fp_per_scene'],op['recall'],yerr=[[op['recall']-lo],[hi-op['recall']]],fmt='o',color='#203344',capsize=4,label='검증 고정 임계값 + Wilson95%')
 axs[0].set(xlim=(0,30),ylim=(-.01,1.03),xlabel='오탐 / 장면',ylabel='탐지한 객체 / 36',title='시험 4장면 · 1 m point matching')
 axs[1].set(xscale='log',xlim=(.1,650),ylim=(-.01,1.03),xlabel='오탐 / 장면 (로그)',ylabel='탐지한 객체 / 36',title='전체 곡선 · 낮은 임계값은 오탐 증가')
 axs[0].legend(frameon=False,fontsize=8,loc='lower right');figsave(fig,'froc.png')
 ex=np.load(OUT/'detection_example.npz');power=ex['power'];normal=np.median(power,axis=1,keepdims=True)+1e-25
 fig,axs=plt.subplots(1,2,figsize=(10,5.2),layout='constrained')
 ax=axs[0];ax.imshow(10*np.log10(np.maximum(power/normal,1e-5)),origin='lower',cmap='gray',vmin=-12,vmax=20,extent=[ex['x'][0],ex['x'][-1],ex['y'][0],ex['y'][-1]],aspect='equal')
 g=ex['truth'];ax.scatter(g[:,0],g[:,1],s=75,facecolors='none',edgecolors='#43b9c7',label='GT 삽입점')
 threshold=read('detection_selection.json')['rf_threshold'];keep=1-ex['probability'][:,0]>=threshold;p=ex['positions'][keep]
 ax.scatter(p[:,0],p[:,1],s=28,marker='+',c='#edb866',label='고정 임계값 탐지');ax.set(title='scene04 · 전체 영상에서 후보 생성',xlabel='Along track (m)',ylabel='Cross track (m)');ax.legend(frameon=True,facecolor='white',framealpha=.95,fontsize=8)
 cm=np.array(det['operating_point_rf']['confusion_rows_truth_columns_background_trap_rock_cylinder']);ax=axs[1];ax.imshow(cm,cmap='Blues',vmin=0,vmax=max(12,cm.max()))
 for iy in range(3):
  for ix in range(4):ax.text(ix,iy,str(cm[iy,ix]),ha='center',va='center',color='white' if cm[iy,ix]>7 else '#203344')
 ax.set(xticks=range(4),xticklabels=['미탐','통발','암석','원통'],yticks=range(3),yticklabels=['통발 GT','암석 GT','원통 GT'],title='1 m 위치 탐지 이후 클래스 결과\n잘못 분류된 위치 탐지도 포함')
 figsave(fig,'detection_map_confusion.png')
 ex=np.load(OUT/'raw_bridge_examples.npz');fig,axs=plt.subplots(1,3,figsize=(11,5.4),layout='constrained',sharex=True,sharey=True)
 norm=np.median(ex['clean_bp_power'],axis=1,keepdims=True)+1e-25
 for ax,row in zip(axs,bridge['rows']):
  name=row['image'];p=ex[name+'_power'];im=ax.imshow(10*np.log10(np.maximum(p/norm,1e-5)),origin='lower',cmap='gray',vmin=-12,vmax=20,extent=[ex['x'][0],ex['x'][-1],ex['y'][0],ex['y'][-1]],aspect='equal')
  gt=ex['truth'];ax.scatter(gt[:,0],gt[:,1],s=38,facecolors='none',edgecolors='#43b9c7')
  keep=1-ex[name+'_probability'][:,0]>=threshold;pos=ex[name+'_positions'][keep];ax.scatter(pos[:,0],pos[:,1],s=20,marker='+',c='#edb866')
  label={'clean_bp':'무오차 자체 BP','corrupted_bp':'항법 오차 BP','automatic_entropy_bp':'자동 후보 기반 entropy'}[name]
  op=row['detection'];ax.set_title(f'{label}\n{op["tp"]}/9 탐지 · {op["fp"]} 오탐');ax.set_xlabel('Along track (m)')
 axs[0].set_ylabel('Cross track (m)');fig.colorbar(im,ax=axs,shrink=.7,label='무오차 BP의 거리별 median power 기준 dB · 표시용');figsave(fig,'raw_bridge.png')
 appendix(af,det)


def appendix(af,det):
 """One-page scientific figure; no unavailable portfolio edited."""
 v=json.loads((ROOT/'results/v2/validation_results.json').read_text());s=json.loads((ROOT/'results/v2/stress_results.json').read_text())
 fig=plt.figure(figsize=(11.7,8.3),facecolor='white');gs=fig.add_gridspec(2,3,left=.065,right=.96,top=.75,bottom=.22,hspace=.65,wspace=.35,height_ratios=[1,1])
 fig.text(.065,.93,'공개 합성 SAS 원시 신호 처리·검증',size=23,weight='bold',color='#203344')
 fig.text(.065,.875,'GDF 해석 → 매치드필터 → bistatic BP → 항법 오차 검증 → 객체 위치 탐지',size=13,color='#356778')
 fig.text(.065,.82,'PoSSM · 무잡음 시뮬레이션 · 11장면 / 90배치 · 실측 수중 SAS 및 실제 기뢰 자료 아님',size=10,color='#566d7d')
 ax=fig.add_subplot(gs[:,0]);e=np.load(ROOT/'results/v2/point_validation_examples.npz');x=e['dense_x'];center=x[len(x)//2]
 for key,label,col in [('moving_dense','기존 처리','#8b99a5'),('v2_dense','Hann + Gaussian','#16788a')]:
  z=e[key];iy=abs(z).max(1).argmax();ax.plot((x-center)*1000,db(z[iy],abs(z).max()),label=label,color=col)
 ax.set(xlim=(-75,75),ylim=(-45,1),xlabel='Along-track offset (mm)',ylabel='Normalized amplitude (dB)',title='PSF: 부엽 감소·주엽 확대');ax.legend(frameon=False,fontsize=8)
 ax=fig.add_subplot(gs[0,1]);r=v['point_rows'];a=[z['native_legacy']['complex_correlation'] for z in r];b=[z['native_v2']['complex_correlation'] for z in r]
 ax.scatter(a,b,s=25,color='#16788a');ax.plot([0,1],[0,1],ls=':',color='#9aa8b3');ax.set(xlim=(0,1),ylim=(0,1),xlabel='기존 complex NCC',ylabel='v2 complex NCC',title='미사용 24점 · 원래 SLC 격자')
 ax=fig.add_subplot(gs[0,2]);
 for kind,label,col in [('surge','진행 오차','#8b99a5'),('sway','측면 오차','#16788a')]:
  rows=[r for r in s['summary'] if r['kind']==kind];ax.plot([r['amount']*1000 for r in rows],[r['complex_correlation']['mean'] for r in rows],'o-',ms=4,label=label,color=col)
 ax.set(ylim=(0,1.05),xlabel='항법 오차 RMS (mm)',ylabel='무오차 BP와 평균 NCC',title='24점 × 경로 5개 · 방향별 민감도');ax.legend(frameon=False,fontsize=8)
 ax=fig.add_subplot(gs[1,1:]);ax.axis('off');en=af['success']['per_ping_entropy'];op=det['operating_point_rf']
 text=f'v2 검증  |  24점 complex NCC 중앙값 0.672 → 0.875 / 진폭 상관 ≥0.9: 22/24\n\n'
 baseline=next(r['complex_correlation']['median'] for r in af['summary'] if r['method']=='uncorrected' and r['amount']==.002 and r['range_bin'] is None)
 text+=f'v3 보정  |  측면2mm: NCC 중앙값 {baseline:.3f} → {en["corrupt_median_ncc"]:.3f} (entropy)\n'
 text+=f'             무오차 NCC 최대 감소 {en["clean_max_ncc_loss"]:.4f} / 사전 기준 '+('통과' if en['overall_pass'] else '미달')+'\n\n'
 text+=f'v3 탐지  |  {op["tp"]}/36 위치 탐지 / {op["fp"]} 오탐 / Wilson95% {interval(op["wilson95"])}\n'
 text+='             해저 family 분리 시험 · 탐지 채택 기준 미달'
 ax.text(0,1,text,va='top',size=10.5,linespacing=1.55,color='#203344')
 fig.text(.065,.13,'해석 범위: 기준 영상과의 유사도는 영상 품질과 다름. 실제로 입증한 것은 약16mm 위치 편향 제거와 합성 오차 검증.',size=10,color='#566d7d')
 fig.text(.065,.085,'GT 삽입점 기반 point-FROC · bbox mAP 아님 · 동일 형상 재사용 / 작은 표본 / 실해역 일반화 미확인',size=10,color='#566d7d')
 fig.text(.065,.035,'자료: Philtron · Johnson · Brown / MGDS PoSSM, CC BY-NC-SA 3.0 US. 원본 Python 코드: MIT. 2026-10-20',size=8.5,color='#728491')
 figsave(fig,'SAS_Appendix_One_Page.png')


def body(af,det,bridge,common):
 sel=read('autofocus_selection.json');ds=read('detection_selection.json');pr=read('protocol.json');verify=read('verification.json')
 en=af['success']['per_ping_entropy'];op=det['operating_point_rf'];ca=det['class_aware_rf'];ceil=det['candidate_ceiling'];aux=det['auxiliary_GT_crop_classifier']
 mainrows=[]
 for method,label in zip(METHODS,LABELS):
  g=next(r for r in af['summary'] if r['amount']==.002 and r['method']==method and r['range_bin'] is None)
  controls=[r for r in af['rows'] if not r['amount'] and r['method']==method]
  success=af['success'].get(method)
  mainrows.append([label,f'{g["complex_correlation"]["median"]:.3f}',f'{g["complex_correlation"]["mean"]:.3f}',f'{max(1-r["complex_correlation"] for r in controls):.5f}',
   f'{g["width_x_m"]["median"]*1000:.2f}',f'{g["position_error_m"]["median"]*1000:.2f}',f'{g["seconds_estimator"]["median"]:.3f}',('통과' if success['overall_pass'] else '미달') if success else '기준선'])
 rangerows=[]
 range_entropy=[]
 for band,rg in enumerate(['15–35m','35–55m','55–80m']):
  for method,label in zip(METHODS,LABELS):
   g=next(r for r in af['summary'] if r['amount']==.002 and r['method']==method and r['range_bin']==band)
   rangerows.append([rg,label,g['cases'],f'{g["complex_correlation"]["median"]:.3f}',f'{g["complex_correlation"]["q25"]:.3f}–{g["complex_correlation"]["q75"]:.3f}'])
  range_entropy.append(next(r['complex_correlation']['median'] for r in af['summary'] if r['amount']==.002 and r['method']=='per_ping_entropy' and r['range_bin']==band))
 detectrows=[]
 for name,key in [('위치 탐지 · RF','operating_point_rf'),('위치+클래스 일치 · RF','class_aware_rf'),('대비 후보만 · 검증 임계값','operating_point_proposal')]:
  r=det[key];detectrows.append([name,f'{r["tp"]}/{r["total"]}',pct(r['recall']),r['fp'],f'{r["fp_per_scene"]:.2f}',interval(r['wilson95']),interval(r['scene_bootstrap95'])])
 classrows=[[r['class'],f'{r["tp"]}/{r["total"]}',interval(r['wilson95'])] for r in ca['class_recall']]
 bridgerows=[]
 for r in bridge['rows']:
  d=r['detection'];bridgerows.append([r['image'],f'{r["clean_bp_complex"]["complex_correlation"]:.3f}',f'{d["tp"]}/9',d['fp'],f'{r["bp_seconds"]:.1f}s'])
 rangecommon=[]
 for band,rg in enumerate(['15–35m','35–55m','55–80m']):
  rangecommon.append([rg,*[f'{np.median([r["complex_correlation"] for r in common["rows"] if r["range_bin"]==band and r["method"]==m]):.3f}' for m in ['common_all_ranges','common_within_range_band']]])
 style='''*{box-sizing:border-box}body{margin:0;background:#f2f5f7;color:#203344;font:16px/1.8 system-ui,-apple-system,"Noto Sans CJK KR",sans-serif}header{background:#203344;color:white;padding:42px max(24px,calc((100vw - 1120px)/2))}header p{color:#d1e2e9;max-width:1000px}h1{font-size:clamp(27px,4vw,42px);line-height:1.35;margin:12px 0}.eyebrow{font-size:12px;letter-spacing:.13em;color:#8ec9d4}main{max-width:1170px;padding:24px;margin:auto}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.card,section{background:white;border:1px solid #dce5eb;border-radius:12px;padding:24px}section{margin:22px 0}.value{font-size:28px;font-weight:750;line-height:1.3}.label,.small,figcaption{font-size:13px;color:#607587}h2{font-size:24px;margin:0 0 16px}h2 small{display:block;font-size:12px;letter-spacing:.1em;color:#16788a;margin-bottom:6px}h3{font-size:19px;margin:24px 0 10px}a{color:#16788a}.note{padding:15px 18px;background:#fbf5e9;border-left:4px solid #bd995b;margin:18px 0}.ok{background:#edf6f7;border-color:#16788a}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;font-size:14px;text-align:left}th{background:#eff4f7;white-space:nowrap}th,td{padding:10px;border-bottom:1px solid #dce5eb;vertical-align:top}.plot{width:100%;height:auto;display:block}figure{margin:22px 0}figcaption{margin-top:7px}nav{display:flex;flex-wrap:wrap;gap:16px;font-size:14px;margin-bottom:22px}pre{background:#172b39;color:#e4eef4;padding:19px;border-radius:8px;overflow:auto;font-size:13px}code{font-size:12px;overflow-wrap:anywhere}button{background:#16788a;color:white;border:0;border-radius:6px;padding:12px 16px;font:inherit;cursor:pointer}footer{font-size:12px;color:#687e8d;padding:15px 0 40px}.tag{font-size:12px;color:#8e4f36}details{margin:14px 0}summary{cursor:pointer}@media(max-width:720px){.cards{grid-template-columns:repeat(2,1fr)}section{padding:17px}main{padding:15px}}@media print{nav,.download{display:none}section{break-inside:avoid;border:0}.cards{grid-template-columns:repeat(4,1fr)}header{background:white;color:#203344;padding:15px}header p{color:#203344}}'''
 return f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SAS Focus Benchmark v3 · 보정과 탐지 검증</title><style>{style}</style></head><body>
<header><div class="eyebrow">SAS FOCUS BENCHMARK / EXECUTED V3</div><h1>보정 성능과 탐지 성능을<br>분리해 검증했다</h1><p>공개 합성 SAS 원시 신호에 항법 오차를 주입하고, 국소 GPGA 계열·핑별 엔트로피 보정을 비교했습니다. 전체 영상 후보 생성과 RF 분류도 실행했으며, 시험 결과가 채택 기준에 못 미친 부분을 그대로 남겼습니다.</p></header><main>
<nav><a href="#decision">판정</a><a href="#protocol">사전 기준</a><a href="#focus">보정</a><a href="#detection">point-FROC</a><a href="#bridge">원시→탐지</a><a href="#appendix">부록 1장</a><a href="#repro">재현 묶음</a></nav>
<div class="cards"><div class="card"><div class="value">{en['corrupt_median_ncc']:.3f}</div><div class="label">entropy · 2mm 오차 NCC 중앙값<br>미사용24점 × 새 경로5개</div></div><div class="card"><div class="value">{en['clean_max_ncc_loss']:.4f}</div><div class="label">entropy · 무오차 NCC 최대 감소<br>허용값 ≤0.01</div></div><div class="card"><div class="value">{op['tp']}/36</div><div class="label">시험4장면 · RF 위치 탐지<br>고정 임계값 / 오탐 {op['fp']}개</div></div><div class="card"><div class="value">{ca['tp']}/36</div><div class="label">위치와 클래스 모두 일치<br>통발·암석·원통 / 실제 기뢰 아님</div></div></div>
<section id="decision"><h2><small>01 / DECISION</small>쓸 수 있는 성과와 채택하지 않을 모델</h2><p><strong>v2의 “공개 합성 SAS 원시 신호 처리·검증”은 부록에 쓸 근거가 있습니다.</strong> v3는 이 위에 오차 보정과 탐지의 실제 시험 결과를 추가합니다. 점별 보정 성공이 해저 배경에서의 자동 탐지 성공을 의미하지는 않습니다.</p>
{table(['항목','실행·판정'],[['원시 처리·기준 비교','GDF–매치드필터–bistatic BP. v2 24점 native NCC 중앙값0.672→0.875. 전체 scene00 NCC0.50은 잔여 차이로 유지.'],['국소 GPGA 계열', '사전 성공 기준 '+('통과' if af['success']['local_gpga']['overall_pass'] else '미달')+'; 본 구현의 국소·인접 핑 추정 결과이며 모든 PGA의 한계라는 결론은 아님.'],['핑별 엔트로피','사전 성공 기준 '+('통과' if en['overall_pass'] else '미달')+'; 알려진 고립점 ROI 내부의 blind 추정. 전체 장면 자동 보정과 구분.'],['RF 탐지·분류','학습과 4장면 시험 완료. 재현율80% 기준 미달, 배포·성능 개선 모델로 채택하지 않음.'],['CNN','후속 의견을 반영해 이번 단계 재학습하지 않음.']])}
<div class="note">전체 자료는 무잡음 수중 음향 <strong>시뮬레이션</strong>입니다. 90개 배치는90개의 독립 객체 설계가 아닙니다. 실제 기뢰 식별·실해역 일반화·작전 성능을 입증한 결과로 표현하지 않습니다.</div></section>
<section id="protocol"><h2><small>02 / FROZEN BEFORE TEST</small>결과를 보기 전에 정한 조건</h2>
{table(['구분','사전 고정 조건'],[['보정 성공','측면2mm RMS에서 정합하지 않은 complex NCC 중앙값≥0.90, 무오차24점 모두 NCC 감소≤0.01.'],['개발/시험','개발점1·2·4, seed7001/7002. 미사용24점, 새 seed7101–7105. 기존 민감도 경로6101–6105와도 분리.'],['거리 구간','15–35 / 35–55 / 55–80m. 모든 구간 결과와 전체 공통 위상 진단을 함께 제시.'],['학습/검증/시험','학습02/03/07/08:rough_uniform 36배치. 검증01/06:uniform 18배치. 시험04/05/09/10:ripple·mixed 36배치.'],['탐지 성공','검증 장면에서 ≤5FP/장면 조건의 최대 재현율 임계값을 고정. 시험 재현율≥80%, FP≤5/장면.'],['정답·허용 거리','공식 객체 삽입점CSV, primary 반경1m. 0.5/1.5m는 사전에 정한 보조 민감도. bbox 정답·mAP 없음.']])}
<p class="small">설정 기록 UTC: {html.escape(pr['created_utc'])}<br>protocol SHA-256: <code>{af['protocol_sha256']}</code><br>보정 설정 및 탐지 설정의 별도 hash를 결과 JSON에 저장했습니다. 이 파일은 로컬 사전 기록이며 외부 등록형 preregistration은 아닙니다.</p></section>
<section id="focus"><h2><small>03 / RAW NAVIGATION ERROR → AUTOFOCUS</small>단순 두 계수 위상오차를 벗어나 시험했다</h2><p>공개 IQ는 그대로 두고 영상 형성에 주어지는 Tx·Rx의 측면 좌표를 함께 바꿨습니다. 경로는5ping Gaussian 저주파 성분과0.25배 백색 성분을 합쳐100ping 전체에서2mm RMS로 정규화했습니다. 새 음향 전파를 시뮬레이션한 것이 아니라 <strong>잘못된 항법을 넣은 재구성</strong>입니다.</p>
<p>BP의 복소 기여분을 핑별로 분리했습니다. 합은 기존 frozen BP와 최대 상대오차 {af['stack_sum_relative_error_max']:.2e}로 일치했습니다. 인접 핑 위상차를 적분하는 국소 GPGA 계열과, 핑마다 독립 위상을 두는 최소 엔트로피를 비교했습니다. 이번 GPGA는 짧은 국소 개구에서의 strongest-pixel 추정이며 원 논문의 Max-SDR 전체 재현은 아닙니다.</p>
<p>개발 단계에서 GPGA 반복 {sel['selected']['local_gpga']['iterations']}회, entropy 정규화 {sel['selected']['per_ping_entropy']['regularization']:g}·최대100반복을 선택했습니다. entropy는 zero/GPGA 초기값 두 개 중 목적함수가 낮은 해를 사용했습니다. GT는 ROI 선택과 평가에만 쓰고 위상 추정·목적함수·초기값 선택에는 넣지 않았습니다.</p>
{table(['방법','NCC 중앙값','NCC 평균','무오차 최대 감소','x 3dB폭 중앙값 mm','위치오차 중앙값 mm','추정 시간 중앙값 s','사전 기준'],mainrows)}
{picture('autofocus_range_clean.png','고정 거리 구간별 2mm 항법 오차 시험과 무오차 입력 보호 검사')}
{table(['거리 구간','방법','조건 수','NCC 중앙값','사분위 범위'],rangerows)}
<p>entropy NCC 중앙값은15–35m에서{range_entropy[0]:.3f},35–55m에서{range_entropy[1]:.3f},55–80m에서{range_entropy[2]:.3f}였습니다. 가까운 구간의 성공을 전체 거리로 일반화할 수 없습니다. 구간들은 같은24점·5경로를 나눈 것이며 서로 독립된 해역 시험이 아닙니다.</p>
{picture('autofocus_example.png','미리 정한 첫 시험 점·첫 시험 경로의 동일 ROI: 성공 사례를 골라 표시하지 않음')}
<div class="note"><strong>정합하지 않은 NCC를 대표값으로 유지했습니다.</strong> 상수 위상과 선형 핑 위상은 gauge로 제거했습니다. 선형 위상에 해당하는 위치 이동은 autofocus만으로 완전히 식별되지 않을 수 있습니다. 정답 위치로 영상을 이동시켜 점수를 높이지 않았습니다. 위치 오차는 무잡음 고립점 수치 실험이며 실해역 위치 정확도가 아닙니다.</div>
<h3>전체 거리 공통 위상 가정의 진단</h3><p>같은24점의 blind peak 기여분에서 추정한 위상을 전체 거리에서 공유한 경우와3개 구간 안에서 공유한 경우를 비교했습니다. 각 점이 자기 위상 추정에 포함되는 <strong>보조 진단</strong>이며 독립 장면 시험이나 primary 선택에는 쓰지 않았습니다. 빈 핑 구간의 위상 적분과 실제 지연 이동을 풀지 못하는 한계도 남습니다.</p>
{table(['거리 구간','전체 거리 공통 NCC 중앙값','구간 공통 NCC 중앙값'],rangecommon)}{picture('common_phase.png','범위를 나누는 것만으로 충분한지 확인한 공통 위상 진단')}
<p class="small">120조건은24개 점과5개 경로를 공유하며120개의 독립 해역이 아닙니다. 처리시간은 CPU 추정기 시간으로, raw 읽기·range compression·BP를 포함한 실시간 처리 성능이 아닙니다. 무오차 검사도24조건 전부 공개했습니다.</p></section>
<section id="detection"><h2><small>04 / WHOLE-IMAGE POINT-FROC</small>GT crop 분류와 전체 영상 탐지를 구분했다</h2><p>공식 SLC power를2×2 평균해5cm 격자로 만들었습니다. 원시 단계를 검증한 BP와 학습 영상의 종류를 혼동하지 않도록, 이 절의 학습·시험은 <strong>공식 SLC</strong>임을 명시합니다. 전체 유효 영상에서 다중 크기 국소 대비 후보를 만들고0.65m NMS로 최대600개를 남겼습니다. GT 주변으로 후보 탐색 범위를 제한하지 않았습니다.</p>
<p>후보 주변6×6m patch의 국소 정규화 log-power·변동·highlight/shadow 비율을12×12격자에서 계산했습니다. 절대 좌표·거리는 분류기에 입력하지 않았습니다. RF300trees, 학습 장면의36객체와 배경 후보·GT jitter crop으로 학습했고, leaf {ds['selected_min_samples_leaf']}와 임계값 {ds['rf_threshold']:.6f}는 검증18개로만 선택했습니다. 외부 사전학습 가중치나 새 CNN을 사용하지 않았으며, 본 모델은 자체 저표본 기준선입니다.</p>
{table(['시험 고정 임계값','탐지 수','재현율','전체 FP','FP/장면','Wilson95%','장면 bootstrap95%'],detectrows)}
<p>예측 점을 점수순으로 정답 삽입점과1대1 대응시킵니다. 이미 대응된 표적의 중복 검출은FP입니다. “위치+클래스”는 위치뿐 아니라 통발·암석·원통 클래스가 같아야TP로 인정합니다. 시험36개는 모두 사전 검색 유효 영역 안에 있습니다.</p>
{picture('froc.png','시험 곡선은 전체 점수로 그렸지만 실제 사용 임계값은 검증 장면에서 고정')}
<div class="note">Wilson 구간은 객체 배치 독립성을 가정합니다. 반복 형상·공유 장면 때문에 낙관적일 수 있어 장면 단위 bootstrap도 함께 냈습니다. 시험 장면이4개뿐이라 bootstrap 구간도 안정적 일반화 보장은 아닙니다.</div>
{table(['클래스 · 위치+분류 일치','탐지 수','Wilson95%'],classrows)}
{table(['시험 장면','위치 TP/9','FP'],[[f'scene{r["scene"]:02d}',f'{r["tp"]}/9',r['fp']] for r in op['scenes']])}
{picture('detection_map_confusion.png','scene04 검색 결과와 시험36개의 위치 탐지·클래스 혼동. GT는 평가·표시에만 사용')}
<h3>왜 실패했는가</h3><p>임계값을 최대한 내려도 후보600개 안에서1m 이내 대응되는 객체는 <strong>{ceil['tp']}/36</strong>뿐입니다. 이때FP는 {ceil['fp']}개라 실용적인 성능이 아닙니다. 최소 {36-ceil['tp']}개는 제안 단계에서 놓쳤고, 고정 임계값에서는 그중 더 많은 후보가 거절됐습니다. 따라서 분류기만 크게 바꾸어 해결된다고 보기 어렵습니다.</p>
<p>정답 위치와 전경이라는 사실을 알려준 보조3분류는 {aux['correct']}/{aux['total']}였습니다. 이는 배경 거절과 위치 탐색을 생략한 결과여서 탐지율로 쓸 수 없습니다. 주 결과는 {op['tp']}/36이고, 클래스까지 맞은 결과는 {ca['tp']}/36입니다.</p>
{table(['사전 보조 매칭 반경','TP/36','FP'],[[f'{r["radius_m"]}m',f'{r["result"]["tp"]}/36',r['result']['fp']] for r in det['matching_radius_diagnostic']])}
<p>반경을 바꾸면 점수가 크게 바뀝니다. 삽입점은 영상 highlight 중심이나 객체의 box가 아닙니다. primary1m 기준은 유지하고, 후속 데이터에는 검수된 highlight·shadow·bbox 주석을 보완해야 합니다.</p></section>
<section id="bridge"><h2><small>05 / RAW → OWN BP → FROZEN DETECTOR</small>점 보정이 탐지에도 이득인지 따로 시험</h2><p>시험 scene04 원시 IQ로2.5cm 격자 전면 BP를 계산했습니다. 무오차, 측면2mm RMS(seed7101), 자동 후보 기반 entropy 보정의3조건에 위와 같은 RF·고정 임계값을 그대로 적용했습니다. 공식 SLC에서 자체 BP로 바뀌는 입력 분포 차이까지 포함한 <strong>단일 장면 연결 시험</strong>입니다.</p>
<p>자동 보정은 손상 영상의 대비 후보 중 서로2m 떨어진30개를 뽑고, 각0.4m ROI에서 이미 고정한 entropy를 실행했습니다. 그 위상차를3개 거리 구간에서 합쳐 적용했습니다. GT 객체 위치나 깨끗한 영상을 위상 추정에 쓰지 않았습니다. 이 확장 방법도 실행 전에 별도 설정 파일에 기록했습니다.</p>
{table(['영상','무오차 자체 BP와 NCC','위치 TP','FP','전면 BP 시간'],bridgerows)}
{picture('raw_bridge.png','scene04 원시 신호에서 직접 만든 영상의 연결 시험. 원은 GT, 십자는 고정 임계값의 탐지')}
<p>자동 후보는 고립점과 달리 분산 산란·표면·그림자를 포함합니다. 국소 entropy의 위상해를 거리 구간에 공유하는 과정은 잘못된 위상 보정이나 표적 이동을 만들 수 있습니다. 위 표의 결과를 보정 채택 근거로 과장하지 않습니다. 1장면·1오차 경로로 보정 후 탐지 일반화를 입증하지 못합니다.</p>
<p class="small">range compression·8× 준비 {bridge['preparation_seconds']:.1f}s, 자동30patch 위상 추정 {bridge['autofocus_seconds']:.1f}s. CPU 환경에서 측정했으며 현장 실시간성 검증이 아닙니다.</p></section>
<section id="appendix"><h2><small>06 / PORTFOLIO APPENDIX</small>부록 1장 초안과 다음 보완 방향</h2>
{picture('SAS_Appendix_One_Page.png','PSF·24점 비교·오차 민감도와 v3 판정을 한 장에 담은 부록용 그림')}
<p><a id="appendix-download" download="SAS_Appendix_One_Page.png" href="data:image/png;base64,{base64.b64encode((FIG/'SAS_Appendix_One_Page.png').read_bytes()).decode()}">부록 그림 PNG 내려받기</a>. 원본 포트폴리오 파일은 수정하지 않았습니다. 이 초안을 부록⑨ 다음에 넣을 수 있습니다.</p>
<ol><li><strong>주 성과는 원시 처리·검증으로 유지:</strong> v2 PSF의 부엽 감소와 주엽 확대를 같이 표시합니다. ASASIN과의 일치도와 영상 품질 향상을 같은 주장으로 쓰지 않습니다.</li><li><strong>탐지 입력부터 보완:</strong> 현재 제안 단계 ceiling이 낮습니다. 동일 시험4장면을 계속 튜닝에 쓰지 말고 새로운 개발·시험 장면을 확보한 뒤, highlight·shadow·bbox를 검수해 후보 생성부터 바꿔야 합니다.</li><li><strong>자료의 독립성 확대:</strong> 동일 mesh의90배치에 증강을 더하는 것으로 새 형상·해저·항차의 일반화를 증명할 수 없습니다. 다른 형상·환경·센서 조건이 포함된 자료를 독립 시험에 둡니다.</li><li><strong>보정은 단계적으로:</strong> 고립점 ROI 성공, 자동 산란점 선택 성공, 거리별 위상 공유 성공, 탐지 개선을 차례로 분리합니다. 원시 항법 오차 영상과 충분한 학습 장면이 마련되기 전에는 작은 CNN을 다시 얹지 않습니다.</li></ol></section>
<section id="repro"><h2><small>07 / REPRODUCIBILITY</small>실행 코드·RF 가중치·예측·수치 포함</h2><div class="download" id="zip-slot"></div>
<pre>python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements_v3.txt
python src/run_v3.py --download

# 포함된 결과로 HTML·그림만 다시 만들기
python src/run_v3.py --report-only</pre>
<p>원시 파일은 용량상 묶음에서 제외했습니다. 공개 다운로드 코드와55개 파일의 hash manifest, 사전 설정, 개발 선택 기록,432개 점·오차·방법 결과,4장면의 후보·점수 전부,RF 가중치,환경·검증 기록을 포함했습니다. v2 수치와 부록 재생성에 필요한 배열도 포함합니다. 원시 자료 취득 후 CPU에서 재실행할 수 있으며 전체 행렬 처리에는 메모리 여유가 필요합니다.</p>
<details><summary>통과한 수치·평가 무결성 검사 {len(verify['checks'])}개</summary><ul>{''.join('<li>'+html.escape(c)+'</li>' for c in verify['checks'])}</ul><p>무결성 검사 통과는 모델 채택 기준 통과를 뜻하지 않습니다. 각 단계를 실제 실행했으며, 통합 실행 명령 전체를 다시 처음부터 돌린 결과라고 주장하지 않습니다.</p></details>
<h3>1차 출처·방법론</h3><ul><li>Philtron·Johnson·Brown / MGDS <a href="https://www.marine-geo.org/tools/files/31898">PoSSM raw GDF</a> · <a href="https://www.marine-geo.org/tools/files/31899">SLC</a>. CC BY-NC-SA3.0US.</li><li>Wahl·Eichel·Ghiglia·Jakowatz, <a href="https://doi.org/10.1109/7.303752">Phase gradient autofocus</a>, 1994. Evers·Jackson, <a href="https://doi.org/10.1109/TCI.2019.2899453">Generalized PGA</a>, 2019 / <a href="https://scholar.afit.edu/etd/2240/">저자 학위논문</a>.</li><li>Henrik Forsten, <a href="https://hforsten.com/synthetic-aperture-radar-autofocus-and-calibration.html">BP 기여분의 GPGA 설명·직접 구현 결과</a>. 원 저자의 레이더 결과가 본 SAS 자료의 성능을 보장하지는 않습니다.</li><li><a href="https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.RandomForestClassifier.html">scikit-learn1.8 RandomForest 문서</a>. CPU 저표본 자체 기준선 구현에 사용.</li><li>Gerg et al., <a href="https://arxiv.org/abs/2101.05888">ASASIN 영상 형성 연구</a>. 센서 이동·bistatic 기하 참고; 정확한 제품 설정 재현이라는 주장은 유지하지 않음.</li></ul>
<p class="small">원본 Python 코드 MIT. PoSSM 파생 수치·가중치·그림·HTML은 CC BY-NC-SA3.0US 조건을 구분합니다. 별도 대형 데이터나 외부 사전학습 가중치는 이번 실행에 사용하지 않았습니다.</p></section>
<footer>Executed v3 · 2026-10-20 · 합성 원시 신호 처리 검증 / 고립점 보정 / point-FROC · 시험 실패와 해석 범위 유지</footer></main></body></html>'''


def package(text):
 deliver=ROOT/'deliverables';deliver.mkdir(exist_ok=True);path=deliver/'SAS_Focus_Reproducibility_v3.zip'
 files=[ROOT/x for x in ['README_v3.md','README_v2.md','requirements.txt','requirements_v3.txt','LICENSE','THIRD_PARTY.md']]
 files+=list((ROOT/'src').glob('*.py'))
 files += [ROOT/'references'/x for x in ['PoSSM_README.txt','Scatterer_Locations_Scene00.csv','Object_Locations_All_Scenes.csv','Description_of_Simulated_SAS_Data_Package.pdf','PSU_GDF_Format_20240612.pdf','download_manifest.json','download_manifest_v2.json','scene_split_v2.csv']]
 files += [p for p in OUT.glob('*.json') if p.name!='delivery_validation.json']
 files+=list(OUT.glob('*.npz'))+list(OUT.glob('*.joblib'))
 files += [p for p in OUT.glob('*.log') if p.name not in ['report_build.log','packaged_report_regeneration.log']]
 files += [p for p in (ROOT/'results/v2').glob('*') if p.suffix=='.json' or p.name in ['point_validation_examples.npz','stress_examples.npz']]
 files += list(FIG.glob('*.png'))
 temporary=path.with_suffix('.building.tmp')
 with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in files:z.write(p,'sas_focus_benchmark/'+str(p.relative_to(ROOT)))
  z.writestr('sas_focus_benchmark/README.md',(ROOT/'README_v3.md').read_text())
  z.writestr('sas_focus_benchmark/report_v3/report.html',text)
 temporary.replace(path)
 payload=path.read_bytes();digest=hashlib.sha256(payload).hexdigest()
 button=f'<button id="get-zip">실행 코드·RF 가중치·전체 수치 ZIP ({len(payload)/1e6:.1f}MB)</button><p class="small">ZIP SHA-256: <code>{digest}</code></p>'
 text=text.replace('<div class="download" id="zip-slot"></div>','<div class="download" id="zip-slot">'+button+'</div>')
 js='''document.getElementById('get-zip').addEventListener('click',()=>{const s=atob(document.getElementById('zip-data').textContent.trim());const a=new Uint8Array(s.length);for(let i=0;i<s.length;i++)a[i]=s.charCodeAt(i);const u=URL.createObjectURL(new Blob([a],{type:'application/zip'}));const e=document.createElement('a');e.href=u;e.download='SAS_Focus_Reproducibility_v3.zip';document.body.appendChild(e);e.click();e.remove();setTimeout(()=>URL.revokeObjectURL(u),60000)})'''
 text=text.replace('</body>','<script id="zip-data" type="application/octet-stream">'+base64.b64encode(payload).decode()+'</script><script>'+js+'</script></body>')
 p=deliver/'SAS_Focus_Report_v3.html';p.write_text(text);print('Saved',p,'bytes',p.stat().st_size,'ZIP',len(payload),'SHA256',digest)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--bundle',action='store_true');a=p.parse_args()
 af=read('autofocus_results.json');det=read('detection_results.json');bridge=read('raw_bridge_results.json');common=read('common_phase_diagnostic.json')
 figures(af,det,bridge,common);text=body(af,det,bridge,common);(REPORT/'report.html').write_text(text)
 if a.bundle:package(text)
