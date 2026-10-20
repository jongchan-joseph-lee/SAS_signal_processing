"""Offline stage-1 v2 report with reproducibility ZIP embedded on request."""
from pathlib import Path
import argparse,base64,hashlib,html,json,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import build_report as theme

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'report_v2';FIG=OUT/'figures'


def read(p):return json.loads((ROOT/p).read_text())


def setup():
    theme.setup();OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)


def save(fig,name):fig.savefig(FIG/name,dpi=140,bbox_inches='tight');plt.close(fig)


def image(name,alt):return '<img class="plot" alt="'+html.escape(alt)+'" src="data:image/png;base64,'+base64.b64encode((FIG/name).read_bytes()).decode()+'">'


def figures(v,s,g):
    a=read('results/v2/taper_ablation.json')['rows'];b=read('results/v2/native_ablation.json')['rows']
    labels=['4×','8×','16×','Gaussian','Hann','Hann+Gaussian']
    scores=[]
    from imaging import complex_metrics
    native=np.load(ROOT/'results/v2/native_ablation.npz')
    # Match the taper test's 25x25 native ROI, rather than combining scores
    # from differently sized 41x41 and 25x25 windows.
    for method in ['legacy4','uniform8','uniform16','gauss_same8']:
        scores.append(np.mean([complex_metrics(native[f'p{i}_reference'][8:-8,8:-8],native[f'p{i}_{method}'][8:-8,8:-8])['complex_correlation'] for i in [1,2,4]]))
    scores += [np.mean([r['complex_correlation'] for r in a if r['taper']=='hann60' and r['beam']==beam]) for beam in [0,1]]
    fig,ax=plt.subplots(figsize=(9,3.5),layout='constrained');bars=ax.bar(labels,scores,color=['#9aa8b3']*3+['#5d9aae','#c9a361','#206477'])
    for bar,value in zip(bars,scores):ax.text(bar.get_x()+bar.get_width()/2,value+.015,f'{value:.3f}',ha='center',fontsize=9)
    ax.set(ylim=(0,1),ylabel='개발용 3점 평균 complex NCC',title='보간만 늘리는 효과는 작고, 거리 주파수 창의 영향은 큼');save(fig,'ablation.png')
    fig,axs=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    for key,color,label in [('native_legacy','#9aa8b3','1차 처리'),('native_v2','#206477','Hann+Gaussian')]:
        for ax,metric in zip(axs,['complex_correlation','amplitude_correlation']):ax.plot(range(1,25),[r[key][metric] for r in v['point_rows']],'o-',ms=3,color=color,label=label)
    for ax,title in zip(axs,['복소수 상관','진폭 Pearson 상관']):ax.set(xlabel='고정 순서의 검증 점',ylabel=title,ylim=(0,1.03));ax.legend(frameon=False,fontsize=9)
    axs[1].axhline(.9,c='#c9a361',ls=':',lw=1);save(fig,'heldout.png')
    e=np.load(ROOT/'results/v2/point_validation_examples.npz');pts=np.loadtxt(ROOT/'references/Scatterer_Locations_Scene00.csv',delimiter=',',skiprows=1);truth=pts[v['frozen_configuration']['test_point_indices'][0]]
    fig,axs=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    for name,label,color in [('moving','1차 처리','#9aa8b3'),('v2','Hann+Gaussian','#206477')]:
        z=e[name+'_dense'];iy,ix=np.unravel_index(abs(z).argmax(),z.shape)
        axs[0].plot((e['dense_x']-truth[0])*1000,theme.db(z[iy]),label=label,color=color)
        axs[1].plot((e['dense_y']-truth[1])*1000,theme.db(z[:,ix]),label=label,color=color)
    for ax,title in zip(axs,['진행 방향','거리 방향']):ax.set(xlim=(-80,80),ylim=(-55,2),xlabel='정답 위치 대비 거리 (mm)',ylabel='정규화 amplitude dB',title=title+' · 첫 검증 점 · 1 mm 격자');ax.legend(frameon=False)
    save(fig,'psf.png')
    fig,axs=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    for kind,label,color in [('surge','Surge / 진행','#9aa8b3'),('sway','Sway / 좌우','#206477'),('joint','두 방향 동시','#c9a361')]:
        rows=[r for r in s['summary'] if r['kind']==kind];x=[r['amount']*1000 for r in rows];med=[r['complex_correlation']['mean'] for r in rows]
        axs[0].plot(x,med,'o-',label=label,color=color);axs[0].fill_between(x,[r['complex_correlation']['q25'] for r in rows],[r['complex_correlation']['q75'] for r in rows],color=color,alpha=.1)
    rows=[r for r in s['summary'] if r['kind']=='sound_speed_bias'];axs[1].plot([r['amount'] for r in rows],[r['complex_correlation']['mean'] for r in rows],'o-',c='#206477');axs[1].fill_between([r['amount'] for r in rows],[r['complex_correlation']['q25'] for r in rows],[r['complex_correlation']['q75'] for r in rows],color='#206477',alpha=.13)
    axs[0].set(xlabel='각 방향의 전체 항적 RMS (mm)',ylabel='무오차 BP와 complex NCC',ylim=(0,1.04),title='24점 × 5경로: 선은 평균, 음영은 사분위');axs[0].legend(frameon=False,fontsize=8)
    axs[1].set(xlabel='일정 음속 편향 (m/s)',ylabel='무오차 BP와 complex NCC',ylim=(0,1.04),title='24점: 고정 편향에는 무의미한 seed 반복을 하지 않음');save(fig,'stress.png')
    for scene in ['scene00','scene01']:
        path=ROOT/f'results/v2/{scene}_native_map_preview.npz'
        if not path.exists():
            if not (FIG/(scene+'_maps.png')).exists():raise FileNotFoundError('Run validate_slc_v2.py to regenerate map previews')
            continue
        e=np.load(path);x,y=e['x'],e['y'];fig,axs=plt.subplots(1,3,figsize=(11.5,6.4),layout='constrained',sharex=True,sharey=True)
        for ax,key,title in zip(axs,['reference','legacy4','v2_hann_gauss8'],['제공 ASASIN','1차 처리 · native 재계산','Hann+Gaussian · native']):
            im=ax.imshow(theme.db(e[key]),origin='lower',extent=[x[0],x[-1],y[0],y[-1]],vmin=-50,vmax=0,cmap='gray',aspect='equal');ax.set_title(title,fontsize=11);ax.set_xlabel('Along-track (m)')
        axs[0].set_ylabel('Cross-track (m)');fig.colorbar(im,ax=axs,shrink=.65,label='각 영상 최대값 기준 amplitude dB');save(fig,scene+'_maps.png')
    profile=np.load(ROOT/'results/v2/gain_profiles.npz');y=profile['scene01_y'];fig,ax=plt.subplots(figsize=(9,3.2),layout='constrained')
    for key,label,color in [('ref_rms','제공 SLC','#203344'),('bp_rms','Hann+Gaussian','#9aa8b3'),('r2_rms','추가 R² 보정 · 사후 진단','#c9a361')]:
        r=gaussian_filter1d(profile['scene01_'+key],10);r=r/np.linalg.norm(r);ax.plot(y,r,label=label,c=color)
    ax.set(xlim=(5,90),xlabel='Cross-track (m)',ylabel='전체 L2 정규화한 RMS profile',title='scene01: 거리별 진폭 규약이 남은 차이의 한 요인');ax.legend(frameon=False,fontsize=9);save(fig,'range_profile.png')


def report(v,s,g):
    table=theme.table;vs=v['point_summary'];loc=v['localization_summary'];pointrows=v['point_rows']
    before=vs['native_legacy']['complex_correlation']['median'];after=vs['native_v2']['complex_correlation']['median']
    success=sum(r['native_v2']['amplitude_correlation']>=.9 for r in pointrows)
    maprows=[]
    for scene in ['scene00','scene01']:
        for key,label in [('legacy4','1차 처리 · native 재계산'),('v2_hann_gauss8','Hann+Gaussian')]:
            r=v['native_maps'][scene+'_'+key];maprows.append([scene,label,f"{r['complex_correlation']:.3f}",f"{r['amplitude_correlation']:.3f}",f"{r['aligned_nrmse']:.3f}",f"{r['backprojection_seconds']:.1f}s"])
    pointtable=[]
    for label,key in [('1차 처리 · native 재계산','native_legacy'),('Hann+Gaussian','native_v2')]:
        r=vs[key];pointtable.append([label,f"{r['complex_correlation']['mean']:.3f}",f"{r['complex_correlation']['median']:.3f}",f"{r['amplitude_correlation']['mean']:.3f}",f"{r['amplitude_correlation']['median']:.3f}",str(sum(x[key]['amplitude_correlation']>=.9 for x in pointrows))+'/24'])
    psfrows=[]
    for label,key in [('1차 처리','moving_0.001'),('Hann+Gaussian','v2_0.001')]:
        psfrows.append([label,*[f"{np.median([r[key][a]['width_3db_interpolated_m'] for r in pointrows])*1000:.2f}" for a in ['x','y']],*[f"{np.median([r[key][a]['pslr_db'] for r in pointrows]):.2f}" for a in ['x','y']]])
    locrows=[]
    for label,key in [('이동 미반영 · 5mm','stop_0.005'),('이동 반영 · 5mm','moving_0.005'),('이동 반영 · 1mm','moving_0.001'),('Hann+Gaussian · 1mm','v2_0.001')]:locrows.append([label,f"{loc[key]['grid_error_m']['median']*1000:.4f}",f"{loc[key]['subpixel_error_m']['median']*1000:.4f}"])
    stressrows=[]
    for kind,label in [('surge','진행 RMS'),('sway','좌우 RMS'),('joint','양방향 RMS'),('sound_speed_bias','음속 편향')]:
        for r in [x for x in s['summary'] if x['kind']==kind]:
            a=r['amount']*(1000 if kind!='sound_speed_bias' else 1);n=r['complex_correlation'];stressrows.append([label,f"{a:g} "+('mm' if kind!='sound_speed_bias' else 'm/s'),r['cases'],f"{n['mean']:.3f}",f"{n['median']:.3f}",f"{n['q25']:.3f}–{n['q75']:.3f}"])
    split=read('results/v2/scene_split.json');gainrows=[[r['scene'],f"{r['uncompensated']['complex_correlation']:.3f}",f"{r['fixed_r2']['complex_correlation']:.3f}",f"{r['fixed_r2']['amplitude_correlation']:.3f}"] for r in g['rows']]
    style='''*{box-sizing:border-box}body{margin:0;background:#f2f5f7;color:#203344;font:16px/1.75 system-ui,-apple-system,"Noto Sans CJK KR",sans-serif}header{background:#203344;color:white;padding:45px max(24px,calc((100vw - 1100px)/2))}header p{max-width:900px;color:#d7e3ec}h1{font-size:clamp(28px,4vw,42px);line-height:1.35;margin:12px 0}.eyebrow{letter-spacing:.13em;font-size:12px;color:#a9d1dc}main{max-width:1150px;margin:auto;padding:26px 24px 65px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.card,section{background:white;border:1px solid #dce4eb;border-radius:12px;padding:25px}section{margin-top:22px}.value{font-size:27px;font-weight:750;line-height:1.4}.label,.small,figcaption{font-size:13px;color:#5b6f80}h2{font-size:23px;margin:0 0 17px}h2 small{display:block;font-size:12px;color:#206477;letter-spacing:.12em;margin-bottom:7px}h3{font-size:18px;margin:24px 0 10px}p{margin:12px 0}a{color:#206477}.note{border-left:4px solid #c9a361;background:#fbf6eb;padding:15px 18px;margin:18px 0}.ok{background:#f0f7f8;border-color:#206477}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;font-size:14px;text-align:left}th{background:#f0f4f7;white-space:nowrap}th,td{padding:10px 12px;border-bottom:1px solid #dce4eb;vertical-align:top}.plot{display:block;width:100%;height:auto;margin:16px 0}.formula{background:#f0f4f7;padding:15px;font:14px/1.8 ui-monospace,monospace;border-radius:8px;overflow:auto}pre{background:#162836;color:#e3eff4;padding:17px;border-radius:8px;overflow:auto;font-size:13px}button,select{font:inherit;font-size:14px;border-radius:6px;padding:10px 14px;border:1px solid #dce4eb}button{background:#206477;color:white;border:0;cursor:pointer}nav{display:flex;gap:17px;flex-wrap:wrap;font-size:14px;margin-bottom:23px}nav a{text-decoration:none}.map[hidden]{display:none}details{border-top:1px solid #dce4eb;padding-top:14px;margin-top:18px}summary{cursor:pointer;font-weight:650}li{margin:7px 0}footer{font-size:12px;color:#5b6f80;margin-top:26px}code{font-family:ui-monospace,monospace}@media(max-width:750px){main{padding:18px 14px}.cards{grid-template-columns:1fr 1fr}.card{padding:17px}section{padding:20px}}@media print{button,select,nav{display:none}header{background:white;color:#203344}header p{color:#5b6f80}body{background:white}}'''
    return f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SAS Focus Benchmark v2 · 원인 추적·검증</title><style>{style}</style></head><body>
<header><div class="eyebrow">SAS FOCUS BENCHMARK / V2 FIRST CHECKPOINT / 2026.10.20</div><h1>기준 영상과의 차이를 추적하고,<br>같은 격자에서 다시 검증하다</h1><p>기뢰대응 프로젝트의 영상 형성 기반 실험. 원시 PoSSM 신호로 빔 가중·보간·거리 주파수 창을 비교하고, 원래 SLC 격자의 전면 영상과 별도 산란체 24개에서 검증했습니다.</p><p class="small" style="color:#c4d8e1">전 자료는 수중 음향 시뮬레이션입니다. 이 보고서는 요청한 1번의 중간 제출이며 새 기뢰 탐지기·새 autofocus 보정기 성능 보고가 아닙니다.</p></header><main>
<nav><a href="#judgment">평가 교정</a><a href="#native">동일 격자 비교</a><a href="#psf">PSF·위치</a><a href="#stress">오차 민감도</a><a href="#data">자료·분할</a><a href="#next">다음 실험</a><a href="#repro">재현</a></nav>
<div class="cards"><div class="card"><div class="value">{before:.3f} → {after:.3f}</div><div class="label">24점 complex NCC 중앙값</div></div><div class="card"><div class="value">{success}/24</div><div class="label">진폭 상관 0.9 이상인 검증 점</div></div><div class="card"><div class="value">11 scenes</div><div class="label">GDF44개 + SLC11개 확보·검사</div></div><div class="card"><div class="value">24 × 5</div><div class="label">산란체 × 확률적 항법 경로</div></div></div>
<section><p><strong>쓸 만한 신호처리 기준선으로 개선됐습니다. 제공 ASASIN의 완전 재현이나 실해역 기뢰 ATR 성과로 표현할 수준은 아닙니다.</strong> 보간 배수보다 거리 압축 후 주파수 창의 영향이 컸고, 여러 점의 좌우 항법 오차 민감도도 확인했습니다. 기준 처리의 정확한 창·진폭 규약은 아직 확정하지 못했습니다.</p><div id="bundle-placeholder"><p class="small">코드 묶음: README_v2.md · src/ · results/v2/</p></div></section>
<section id="judgment"><h2><small>01 / REVIEW OF THE REVIEW</small>붙여주신 평가에서 맞는 점과 수정할 점</h2>
{table(['평가 항목','대조 결과'],[['원시 처리와 검증이 기초 증거라는 평가','타당합니다. 합성·무잡음·이상 항로 조건이며 기뢰 탐지 성능이 아닙니다.'],['±0.05 rad 대신 16°를 썼을 가능성','실행에는 ±0.05 rad가 사용됐습니다. 16°는 호출되지 않은 기본값입니다. HDF5 AzLimit도0.05(HWHP)이지만 실제 culling/taper 정의가 완전히 같다는 뜻은 아닙니다.'],['PDF Gaussian을 적용하면 ASASIN 가중을 맞출 수 있다는 설명','PDF는 전방 모델의 송수신 지향성을 설명합니다. ASASIN 역투영 apodization 명세는 아닙니다. 이번 Gaussian은 비교용 설계이며 정확한 ASASIN 가중이라고 주장하지 않습니다.'],['“sample based, not an image”이므로 좌표가 다른 가능성','그 문구만으로 미복원 신호라는 뜻은 아닙니다. Data/Real·Imag와 Offset/Step를 가진 이미 복원된 complex 제품입니다. native격자 전체를 직접 계산했습니다.'],['전통 방법만 정답인2계수 모델을 알아 불공정','CNN도 같은2계수를 출력합니다. 가설 공간이 전통 방법에만 주어진 것은 아닙니다. 다만 두 방법 모두 단순 인공오차 문제라 원시 항법 오차로 일반화할 수 없습니다.'],['CNN은 학습이 거의 안 되어 실패 결과 가치가 낮음','절차 생성 시험에서는 NCC≈0.970이었습니다. 학습 자체의 부재보다 해저 배경으로의 전이 실패입니다. 파인튜닝 원패치34개는 확실히 작습니다.'],['11장면을 쓰면 데이터 부족이 꽤 해결됨','개발 범위는 늘지만 객체는90배치입니다. 독립적인90개 표적 설계나 다양한 실해역 자료가 생기는 것은 아닙니다.']])}
<div class="note">이론 분해능과 비슷한 PSF 폭은 물리적 타당성의 보조 증거입니다. c/(2B)는 명목 분해능이며 창을 적용한3dB폭과 항상 같은 값이 아닙니다. 이론과 숫자가 비슷하다고 전체 신호처리가 검증됐다고 볼 수는 없습니다.</div></section>
<section id="native"><h2><small>02 / NATIVE-GRID VALIDATION</small>원래2.5cm 격자에서 8,064,000픽셀씩 계산</h2><p>HDF5의 Offset+i·Step를 그대로 사용했습니다. 두 장면 각각 <code>3600×2240</code> 격자에서1차 처리와 새 처리를 모두 계산했습니다. 1차 보고서의10cm 값(scene00≈0.264, scene01≈0.577)과 아래2.5cm 값을 섞어 개선률을 계산하지 않습니다.</p>
{table(['장면','처리','complex NCC ↑','진폭 Pearson ↑','aligned NRMSE ↓','BP시간¹'],maprows)}
<p class="small">¹ CPU8 Numba threads, 준비·resampling 별도. 같은 환경에서의 실행 시간이며 실시간 장비 성능을 주장하지 않습니다. complex NCC=|〈ref,image〉|/(‖ref‖‖image‖), 평균을 빼지 않습니다. NRMSE만 최적 전역 complex scalar를 정합했습니다. 공간 정합·픽셀별 위상 fitting은 하지 않았습니다. 단순한 전역 위상 차이나 상수 진폭 배율은 이NCC를 낮추지 않습니다. 남은 차이는 위치·거리·채널에 따라 달라지는 처리 효과여야 합니다.</p>
<h3>개발용3점에서 무엇이 달랐나</h3>{image('ablation.png','개발용 점의 보간·Gaussian·Hann 비교')}<p>4×→8×→16× 보간만으로 차이가 줄지 않았습니다. 추가 Hann60 주파수 창의 효과가 컸습니다. 개발용 점1/2/4에서 평균 complex NCC가 가장 높은 <strong>Hann60+horizontal Gaussian+8×</strong>를 고정한 뒤 나머지24개 점을 평가했습니다. 공개 HDF에 창 이름이 기록돼 있지 않아 실제 ASASIN 창을 찾아냈다고 주장하지 않습니다.</p>
<div class="formula">H(f)=0.5+0.5·cos(πf/30000), |f|≤30000Hz; 그 밖은0<br>Gaussian half-power angle: asin(0.442946·λ/D)<br>Tx D=0.05m, Rx D=0.0333m; amplitude Gaussian을 곱함</div>
<h3>개발에 쓰지 않은24점의 native ROI</h3>{table(['처리','평균 complex NCC','중앙값 complex NCC','평균 진폭 상관','중앙값 진폭 상관','진폭≥0.9'],pointtable)}
{image('heldout.png','24개 점의 동일 native ROI 비교')}<p>모든 점에서 개선된 것은 아닙니다. 새 처리의 최저 complex NCC는 {vs['native_v2']['complex_correlation']['min']:.3f}이고, 진폭0.9 미만도 {24-success}개 남았습니다. 단일 점의0.9 목표는 여러 점에서 달성했지만 전면 일치나 모든 PSF 재현을 달성한 것은 아닙니다.</p>
<label>지도 미리보기 </label><select id="map-select"><option value="scene01">scene01 · 해저 배경·객체</option><option value="scene00">scene00 · 점산란체</option></select><div class="map" id="map-scene01">{image('scene01_maps.png','scene01 native 계산의 미리보기')}</div><div class="map" id="map-scene00" hidden>{image('scene00_maps.png','scene00 native 계산의 미리보기')}</div><p class="small">그림 파일은 표시 용량을 줄이기 위해4칸마다 추출한 미리보기입니다. 위 숫자는 미리보기가 아니라2.5cm 원래 격자 전체에서 계산했습니다. 그림 dB는 각 영상의 최대값으로 정규화했습니다.</p>
<h3>추가 진단: 거리별 진폭 보정</h3><p>단순한 전파 모델의1/(Rtx·Rrx)를 고려해 픽셀별 <code>y²+6²</code>를 곱하는 고정 R² 보정을 시험했습니다. <strong>주 검증 뒤에 추가한 사후 진단</strong>이며 앞의 고정 설정 성능표에 합치지 않습니다. 지수나 곡선을 시험 자료에 fitting하지 않았습니다.</p>{table(['장면','고정 v2 NCC','추가R² 진단 NCC','추가R² 진폭 상관'],gainrows)}
{image('range_profile.png','거리별 진폭 분포와 사후 R2 진단')}<p>진폭 정규화가 남은 차이의 일부라는 근거는 늘었지만 정확한 처리식을 증명한 것은 아닙니다. scene00의 잔여 차이는 여전히 큽니다.</p></section>
<section id="psf"><h2><small>03 / PSF & SUBPIXEL LOCATION</small>양자화와 위치 편향을 구분했다</h2><p>peak 주변3×3 log-power에2차 곡면을 맞춰 sub-pixel 정점을 추정했습니다. 추정기에 GT 좌표를 주지 않았으며, 정답은 ROI 선택과 오차 계산에만 사용했습니다. 5mm와1mm 고정 월드 격자에서 다시 계산했습니다.</p>
{table(['조건','격자 peak 오차 중앙값 mm','sub-pixel 오차 중앙값 mm'],locrows)}
<div class="note"><strong>μm 단위 결과는 실해역 정확도가 아닙니다.</strong> 무잡음 신호, 정확한 기하, 고립된 점, 알려진 영상 평면을 가정한 수치 실험입니다. 1차의1.8mm는 대부분 격자 양자화였다는 평가가 맞습니다. 실제로 보여준 것은 약16mm 위치 편향의 제거이며 분해능 향상이 아닙니다.</div>
{image('psf.png','1mm 격자에서 재구성한 PSF 단면')}
{table(['자체 BP ·24점 중앙값','x 3dB폭 mm','y 3dB폭 mm','x PSLR dB','y PSLR dB'],psfrows)}
<p>창을 적용하자 주엽은 넓어지고 부엽은 낮아졌습니다. 이것은 예상 가능한 trade-off입니다. PSLR는 단면의 첫 local minima 밖 최대 power를 peak power와 비교했습니다.</p><p class="small">제공 SLC의2.5cm 간격은 약1cm급 range PSF를 충분히 표본화하지 못합니다. native reference의 보간3dB폭·PSLR는 sampling-limited로 JSON에 표시했습니다. 이를 위1mm 격자의 정확한 참값처럼 비교하지 않습니다. native2.5cm 영상의 진폭 상관0.9는 정밀 PSF 분해능 검증을 대체하지 않습니다.</p></section>
<section id="stress"><h2><small>04 / 24 POINTS × 5 NAVIGATION PATHS</small>점 하나의 비단조 곡선을 넘어 분포를 확인</h2><p>원시 IQ는 고정하고 영상 형성에 입력되는 송수신 위치를 바꿨습니다. 항법 경로는 저주파 확률 성분(5ping Gaussian smoothing)과0.25배 백색 성분의 합입니다. 100ping 전체에서 각 방향 RMS를 정규화했습니다. 진행·좌우·양방향 각각4수준×5seed×24점, 총1440조건에 일정 음속 편향96조건을 더했습니다.</p>
{image('stress.png','24점 다중 seed 항법 및 음속 민감도')}
{table(['오차','크기','조건 수','평균 NCC','중앙값 NCC','사분위 범위'],stressrows)}
<p>이번 broadside 기하에서는 좌우 오차가 진행 방향 오차보다 훨씬 민감했습니다. 같은2mm RMS에서 평균 NCC는 진행0.994, 좌우0.499였습니다. 하나의 점에서만 보였던 비단조성에 의존하지 않고 방향별 영향과 분포를 확인했습니다.</p><p class="small">기준은 같은 frozen BP의 무오차 영상입니다. 새로운 실제 해양 환경을 측정한 자료가 아닙니다. 120조건은24점과5경로를 공유하며120개의 독립 해역이 아닙니다. 일정 음속 편향은24점만 평가하고 무의미한5회 반복을 하지 않았습니다. 이 표는 <strong>오차 민감도</strong>이며 PGA나 entropy 추정으로 이를 복원한 결과가 아닙니다.</p></section>
<section id="data"><h2><small>05 / DATA & SPLIT</small>11개 장면을 확보했지만 표적90배치라는 한계는 남는다</h2><p>GDF44개와 SLC11개를 실제 취득했습니다. 전체 hash 검사와 모든 장면의100ping·36채널 파싱 및 navigation checksum 검사를 통과했습니다. 한 파일의 첫 취득이 짧게 저장된 문제는 GDF 경계 검사에서 발견해 다시 받고 검증했습니다.</p>
{table(['장면','해저 family','후속 분할','객체 배치 수'],[[r['scene'],r['family'],r['split'],r['object_placements']] for r in split['rows']])}
<p>원래 예시02–07/08/09–10은 rough·ripple·mixed 유형이 중복됩니다. 위 계획은 family 이름을 train/validation/test 사이에서 분리합니다. 다만 mixed에는 uniform·ripple 성분이 있어 모든 물리 요인이 독립이라는 뜻은 아닙니다.</p>
<div class="note">원래 위치 CSV에는 삽입점과 회전각이 있습니다. bbox·마스크·그림자 범위가 아닙니다. 검증된 영상 주석 없이 이를 그대로 YOLO 정답 박스로 바꾸면 안 됩니다. 먼저 좌표 오차 허용 반경을 validation에서 고정한 point-FROC를 쓰거나, 상자 주석을 추가해 검증해야 합니다.</div>
<h3>실측 자료 조사 · 직접 확인한 내용</h3>{table(['자료','확인 결과','역할'],[['In-air SAS','실제 파일은HDF5: ch1–ch4/ts, img_re/img_im, na. ch1/ts=[1001,3000] 실수 전압. GDF 파서는 예상대로 거부.','실측 신호처리 검증 후보. 별도 adapter·공기 음속·DAQ/필터 전처리 필요. 수중 실측 아님.'],['SCTD3.0','현재 공개 저장소는README/LICENSE, release0개. 데이터 다운로드 경로와 데이터 라이선스 미확인. 논문은240kHz13분류,450kHz10분류를 기술.','향후 실제 배포를 확인해야 하는 후보. 저장소MIT만으로 자료MIT라고 할 수 없음.'],['ApertureLab v1.1','Hugging Face 공개 manifest1000장면,800/100/100 분할 확인. 첫TIF의Range 응답206과TIFF 헤더 확인.','탐지 학습용 더 큰 합성 영상 후보. 원시 IQ·실해역 실측 대체 자료는 아님.']])}
<p><a href="https://huggingface.co/datasets/idg101/Aperture_Lab_Synthetic_Aperture_Sonar_v1_1">ApertureLab v1.1</a>은4개 표적 형상과 clutter를 포함한27개 객체 분류를 설명합니다. 현재 학습에는 사용하지 않았습니다. <a href="https://huggingface.co/datasets/idg101/Aperture_Lab_Synthetic_Aperture_Sonar_v1_1/blob/main/LICENSE.md">ALSD-1.0</a>은Kitware와 특정 미 해군 기뢰대응 관련 조직·프로그램·계약에 대한 권리를 제외하며 모델 배포도 다룹니다. 단순한 무제한 오픈소스로 취급하면 안 됩니다.</p>
<p>부족한 것은 특히 <strong>공개 실해역 SAS + 원시 신호 + 검증된 기뢰 정답</strong>의 조합입니다. 합성 영상의 규모 확대, 원시 처리 검증, 실측 일반화 검증은 서로 다른 자료로 해결할 수 있습니다.</p></section>
<section id="next"><h2><small>06 / WHAT THIS SUPPORTS</small>부록에 넣을 수 있는 주장과 다음 우선순위</h2><p>지금은 “공개 합성 SAS 원시 신호 처리와 검증” 사례로는 쓸 수 있습니다. 실측 SAS 재현 완료나 기뢰 탐지 정확도 향상이라는 주장은 뒷받침하지 않습니다.</p><div class="note ok"><strong>부록1장 예시</strong><br>공개 합성 SAS 원시 신호 기반 영상 형성·오차 검증<br>• GDF 해석–매치드필터–bistatic BP 실행 및 native 기준 영상 비교<br>• 미사용24점 complex NCC 중앙값0.672→0.875; 잔여 전면 차이 공개<br>• 24점×5항법 경로에서 방향별 오차 민감도 확인<br>• 학습 보정은 원시 오차 검증 전까지 채택 보류</div>
<ol><li><strong>기준선:</strong> 정확한ASASIN range compression·진폭 규약을 확인하거나, 독립 점산란체 forward model과 실측In-air 원자료에서 같은 구현을 검증합니다. reference에 맞추려고 물리적으로 타당한 PSF를 무조건 흐리게 만들지 않습니다.</li><li><strong>보정 비교:</strong> 현재 원시 항법 오차 영상에 PGA와 핑별 entropy 추정을 적용합니다. 전역 위치/위상 gauge, isoplanatic 범위와 range migration을 따로 평가하고 GT는 추정에 쓰지 않습니다.</li><li><strong>탐지 연결:</strong> 위치CSV로 가능한 point-FROC와 추가 주석이 필요한 bbox mAP를 분리합니다. 알려진 위치의 crop 분류를 end-to-end 탐지로 표현하지 않습니다.</li><li><strong>데이터:</strong> 더 큰 합성 영상으로 해저 배경 다양성을 늘리고 실측 자료는 별도 시험에 둡니다. 실제 기뢰 라벨·항차 분리를 확보하기 전에는 실전 성능으로 확장하지 않습니다.</li></ol>
{table(['요청 단계','이번 checkpoint 상태'],[['1 SLC 차이·native·PSF·위치','실행 완료; 일부 원인 요인 확인, 완전 일치·정확한ASASIN 설정 규명 미달'],['2 전체 자료·분할','11장면 확보·무결성 검사·분할 계획 완료; 후속 모델 훈련은 별도'],['3 PGA/핑별entropy/CNN 비교','새 보정기 비교 미실행; 원시 항법 오차 영상 생성·민감도 기반을 준비'],['4 다중 점·seed 민감도','24점×5경로 실행 완료'],['5 탐지·분류/FROC','미실행; GT 좌표와bbox 주석의 차이 확인'],['6 실측 자료 조사','형식·접근성 조사 완료']])}</section>
<section id="repro"><h2><small>07 / REPRODUCIBILITY & PRIMARY SOURCES</small>수치는 실행 산출물에서 직접 생성</h2><pre><code>python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python src/run_v2.py --download</code></pre><p>코드,55개 입력 hash manifest, 공식GDF 문서·객체 위치, 단계별JSON, 점 검증 complex 배열, 그림을 묶었습니다. 원시 파일·중간 캐시·native 전면 미리보기NPZ는 용량상 제외했습니다. 보고서만 재생성할 때는 포함된 미리보기PNG를 재사용하고, 전체 실행은 미리보기NPZ도 다시 만듭니다. 상세 절차는README_v2.md입니다.</p><details><summary>검증 항목</summary><ul>{''.join('<li>'+html.escape(x)+'</li>' for x in read('results/v2/verification.json')['checks'])}</ul><p class="small">각 수치 단계를 실제 실행했습니다. 새run_v2.py 통합명령 자체를 다시 처음부터 실행한 결과라고 주장하지 않습니다. 브라우저 전체 렌더링 시험은 미실시이며 PNG는 직접 확인하고 embedded ZIP의 무결성과JavaScript 구문을 검사했습니다.</p></details>
<h3>1차 출처</h3><ul><li>Philtron·Johnson·Brown /MGDS: <a href="https://www.marine-geo.org/tools/files/31898">PoSSM raw GDF</a>, <a href="https://www.marine-geo.org/tools/files/31899">SLC HDF5</a>. 공식 data-package와GDF PDF,CSV를 사용. CC BY-NC-SA3.0US.</li><li>Gerg et al., <a href="https://arxiv.org/abs/2101.05888">GPU Acceleration for Synthetic Aperture Sonar Image Reconstruction</a>: ASASIN forward model과좌표·Rx 이동 설명. 이 논문만으로 현재제품의 필터 설정이 확정되는 것은 아닙니다.</li><li>Blanford et al., <a href="https://www.nature.com/articles/s41597-024-04050-0">In-air SAS / Scientific Data2024</a>, <a href="https://github.com/tblanford/airsas">공식 처리 코드</a>.</li><li><a href="https://arxiv.org/html/2608.08106v1">SCTD3.0 논문</a>, <a href="https://github.com/automlresearch/SCTD-3.0">공개 저장소</a>. 확인 시점2026-10-20.</li><li>Isaac Gerg, <a href="https://gergltd.com/aperturelab/dataset-v1-1.html">ApertureLab v1.1 설명</a>, <a href="https://huggingface.co/datasets/idg101/Aperture_Lab_Synthetic_Aperture_Sonar_v1_1/blob/main/LICENSE.md">배포 라이선스 원문</a>. 해당 자료·모델을 이번 학습에 사용하거나 이 묶음에 재배포하지 않았습니다.</li></ul><p class="small">원본 Python은MIT. PoSSM 파생 결과·그림·보고서는CC BY-NC-SA3.0US로 구분합니다. 비상업 자료의 회사 납품·상용 이용 가능 여부를 코드의MIT로 판단하지 마십시오.</p></section><footer>SAS Focus Benchmark v2 · 2026-10-20 · 실제 기뢰 정답·실해역 탐지율 없음 · 기준 처리의 완전 재현 미달 · 실패/미실행 범위를 그대로 기록</footer></main>
<script>document.getElementById('map-select').addEventListener('change',e=>{{['scene00','scene01'].forEach(s=>document.getElementById('map-'+s).hidden=s!==e.target.value)}})</script></body></html>'''


def package(text):
    deliver=ROOT/'deliverables';deliver.mkdir(exist_ok=True);path=deliver/'SAS_Focus_Reproducibility_v2.zip'
    files=[ROOT/x for x in ['README_v2.md','requirements.txt','LICENSE','THIRD_PARTY.md']]
    files += list((ROOT/'src').glob('*.py'))
    files += [ROOT/'references'/x for x in ['PoSSM_README.txt','Scatterer_Locations_Scene00.csv','Object_Locations_All_Scenes.csv','Description_of_Simulated_SAS_Data_Package.pdf','PSU_GDF_Format_20240612.pdf','download_manifest.json','download_manifest_v2.json','scene_split_v2.csv']]
    files += [p for p in (ROOT/'results/v2').glob('*') if p.is_file() and not p.name.endswith('_native_map_preview.npz')]
    files += [ROOT/'results/possm/possm_results.json',ROOT/'results/environment.json']
    files += list((ROOT/'results/learning').glob('*.json'))+list((ROOT/'results/transfer').glob('*.json'))
    files += list((ROOT/'results/learning').glob('*.pt'))+list((ROOT/'results/transfer').glob('*.pt'))
    files += list(FIG.glob('*.png'))
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,'sas_focus_benchmark/'+str(p.relative_to(ROOT)))
        z.writestr('sas_focus_benchmark/README.md',(ROOT/'README_v2.md').read_text())
        z.writestr('sas_focus_benchmark/report_v2/report.html',text)
    payload=path.read_bytes();digest=hashlib.sha256(payload).hexdigest()
    button=f'<button id="get-zip">2차 재현 코드·수치 ZIP 받기 ({len(payload)/1e6:.1f} MB)</button><p class="small">HTML 안에 포함된 오프라인 묶음입니다. ZIP SHA-256: <code>{digest}</code></p>'
    start=text.index('<div id="bundle-placeholder">');end=text.index('</div>',start)+6;text=text[:start]+'<div id="bundle-placeholder">'+button+'</div>'+text[end:]
    js='''<script id="zip-data" type="application/octet-stream">'''+base64.b64encode(payload).decode()+'''</script><script>document.getElementById('get-zip').addEventListener('click',()=>{const s=atob(document.getElementById('zip-data').textContent.trim()),b=new Uint8Array(s.length);for(let i=0;i<s.length;i++)b[i]=s.charCodeAt(i);const u=URL.createObjectURL(new Blob([b],{type:'application/zip'}));const a=document.createElement('a');a.href=u;a.download='SAS_Focus_Reproducibility_v2.zip';document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(u),60000)});</script>'''
    dest=deliver/'SAS_Focus_Report_v2.html';dest.write_text(text.replace('</body>',js+'</body>'));print(json.dumps({'html':str(dest),'bytes':dest.stat().st_size,'zip_bytes':len(payload),'zip_sha256':digest}))


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',action='store_true');a=p.parse_args();setup()
    v=read('results/v2/validation_results.json');s=read('results/v2/stress_results.json');g=read('results/v2/gain_diagnostic.json');figures(v,s,g);text=report(v,s,g);(OUT/'report.html').write_text(text)
    if a.bundle:package(text)


if __name__=='__main__':main()
