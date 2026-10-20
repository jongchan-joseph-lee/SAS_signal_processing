"""Generate an offline Korean HTML report from executed numeric results.

--bundle adds a downloadable ZIP containing source, results and checkpoints
to a single self-contained delivery HTML. No raw IQ is embedded.
"""
from pathlib import Path
import argparse,base64,hashlib,html,io,json,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'report';FIG=REPORT/'figures'
COLORS=['#8a98a6','#206477','#b9a0cf','#d29b3c']


def load(rel):return json.loads((ROOT/rel).read_text())


def setup():
    REPORT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
    font=Path('/root/.local/share/fonts/NotoSansCJKkr-Regular.otf')
    if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']='Noto Sans CJK KR'
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#31475b',
        'text.color':'#203344','xtick.color':'#526777','ytick.color':'#526777','axes.edgecolor':'#ccd5dd',
        'axes.titlesize':12,'font.size':10,'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})


def save(fig,name):
    fig.savefig(FIG/name,dpi=145,bbox_inches='tight');plt.close(fig)


def db(z,peak=None):
    a=np.abs(z);peak=a.max() if peak is None else peak
    return 20*np.log10(np.maximum(a/(peak+1e-30),1e-4))


def figures(p,t,l):
    for scene in ['scene00','scene01']:
        d=np.load(ROOT/f'results/possm/{scene}_map.npz');x,y=d['x'],d['y']
        fig,axs=plt.subplots(1,2,figsize=(9,6),layout='constrained',sharex=True,sharey=True)
        for ax,key,title in zip(axs,['reference','image'],['제공 ASASIN SLC','직접 구현한 역투영']):
            im=ax.imshow(db(d[key]),origin='lower',extent=[x[0],x[-1],y[0],y[-1]],vmin=-50,vmax=0,cmap='gray',aspect='equal')
            ax.set_title(title);ax.set_xlabel('Along-track (m)')
        axs[0].set_ylabel('Cross-track (m)');fig.colorbar(im,ax=axs,shrink=.7,label='각 영상 최대값 기준 amplitude dB')
        save(fig,scene+'_maps.png')
    rows=p['heldout_points']['rows'];stop=np.array([r['stop_and_hop']['localization_m']*1000 for r in rows]);moving=np.array([r['continuous_receiver']['localization_m']*1000 for r in rows])
    fig,axs=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    axs[0].plot(np.arange(1,25),stop,'o-',color='#8a98a6',label='Stop-and-hop')
    axs[0].plot(np.arange(1,25),moving,'o-',color='#206477',label='수신기 이동 반영')
    axs[0].set(xlabel='고정 순서의 검증 산란체',ylabel='위치 오차 (mm)',title='개발용 3개를 제외한 산란체 24개');axs[0].legend(frameon=False)
    d=np.load(ROOT/'results/possm/point_examples.npz');z=d['moving'][0];s=d['stop'][0];iy=np.abs(z).max(1).argmax();x=d['x'][0];truth=rows[0]['truth'][0]
    axs[1].plot((x-truth)*1000,db(s[iy],np.abs(s).max()),label='Stop-and-hop',color='#8a98a6')
    axs[1].plot((x-truth)*1000,db(z[iy],np.abs(z).max()),label='수신기 이동 반영',color='#206477')
    axs[1].axvline(0,c='#d29b3c',ls=':',label='정답 위치');axs[1].set(xlim=(-80,80),ylim=(-45,2),xlabel='정답 대비 along-track (mm)',ylabel='정규화 amplitude (dB)',title='첫 검증 점의 단면');axs[1].legend(frameon=False,fontsize=8)
    save(fig,'localization.png')
    fig,axs=plt.subplots(1,2,figsize=(10,3.4),layout='constrained')
    for ax,kind,scale,xlabel in zip(axs,['cross_track_rms_m','sound_speed_bias_m_s'],[1000,1],['가정 cross-track 좌표 오차 RMS (mm)','가정 음속 편향 (m/s)']):
        cases=[r for r in p['physical_error']['rows'] if r['kind']==kind]
        ax.plot([r['amount']*scale for r in cases],[r['complex_correlation'] for r in cases],'o-',color='#206477')
        ax.set(xlabel=xlabel,ylabel='무오차 BP와 complex NCC',ylim=(0,1.05));ax.grid(axis='y',alpha=.2)
    save(fig,'physical_sensitivity.png')
    names=['손상 영상','엔트로피 보정','절차 생성 학습 CNN','파인튜닝 CNN'];keys=['corrupted','entropy','procedural_model','finetuned']
    fig,axs=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    raw=np.load(ROOT/'results/transfer/transfer_metrics.npz')
    for i,(name,key) in enumerate(zip(names,keys)):
        values=raw[key+'_complex_correlation']
        axs[0].scatter(np.full(len(values),i)+np.linspace(-.14,.14,len(values)),values,s=8,alpha=.32,c=COLORS[i])
        axs[0].plot(i,values.mean(),'D',ms=7,c='#203344')
    axs[0].set_xticks(range(4),names,rotation=12,ha='right');axs[0].set(ylim=(0,1.03),ylabel='주입 전 SLC와 complex NCC',title='scene01: 24패치 × 4오차; 검은 점은 평균')
    for rel,color,label in [('results/learning/training_history.json','#b9a0cf','절차 생성 사전학습'),('results/transfer/training_history.json','#d29b3c','scene00 파인튜닝')]:
        hist=load(rel);axs[1].plot([r['epoch'] for r in hist],[r['validation_loss'] for r in hist],color=color,label=label)
    axs[1].set(xlabel='Epoch',ylabel='정규화 위상계수 validation MSE',title='시험 장면으로 epoch를 선택하지 않음');axs[1].legend(frameon=False)
    save(fig,'focus_scores.png')
    e=np.load(ROOT/'results/transfer/transfer_examples.npz')
    for i,case in enumerate(e['case_ids']):
        fig,axs=plt.subplots(1,5,figsize=(12,2.8),layout='constrained',sharex=True,sharey=True)
        for ax,key,title in zip(axs,['clean',*keys],['주입 전','손상','엔트로피','절차 생성 CNN','파인튜닝 CNN']):
            im=ax.imshow(db(e[key][i],np.abs(e['clean'][i]).max()),origin='lower',cmap='gray',vmin=-40,vmax=0)
            ax.set_title(title);ax.set_xticks([]);ax.set_yticks([])
        fig.colorbar(im,ax=axs,shrink=.7,label='공통 amplitude dB')
        save(fig,f'focus_case_{int(case)}.png')


def img(name,alt):
    b64=base64.b64encode((FIG/name).read_bytes()).decode()
    return f'<img class="plot" alt="{html.escape(alt)}" src="data:image/png;base64,{b64}">'


def table(head,rows):
    return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+str(x)+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def report(p,t,l):
    verification=load('results/verification.json')
    sloc=p['heldout_points']['stop_and_hop']['localization_m']*1000
    mloc=p['heldout_points']['continuous_receiver']['localization_m']*1000
    focus_rows=[]
    for label,key in [('손상 영상','corrupted'),('엔트로피 최적화','entropy'),('절차 생성 CNN','procedural_model'),('scene00 파인튜닝 CNN','finetuned')]:
        time_key={'entropy':'entropy','procedural_model':'procedural_model_including_features','finetuned':'finetuned_including_features'}.get(key)
        seconds=t['runtime_seconds'][time_key] if time_key else None
        focus_rows.append([label,f"{t[key]['complex_correlation']['mean']:.3f}",f"{t[key]['complex_correlation']['median']:.3f}",f"{t[key]['aligned_nrmse']['mean']:.3f}",f'{seconds*1000:.1f} ms / 96건' if seconds else '—'])
    controls=[[label,f"{t['already_focused_control'][key]['complex_correlation']:.3f}",f"{t['already_focused_control'][key]['aligned_nrmse']:.3f}"] for label,key in [('엔트로피 최적화','entropy'),('절차 생성 CNN','procedural_model'),('파인튜닝 CNN','finetuned')]]
    sourcebytes=sum(i['bytes'] for i in load('references/download_manifest.json')['files'])
    physics_rows=[[r['kind'].replace('cross_track_rms_m','좌표 RMS (mm)').replace('sound_speed_bias_m_s','음속 편향 (m/s)'),f"{r['amount']*(1000 if r['kind']=='cross_track_rms_m' else 1):g}",f"{r['complex_correlation']:.3f}",f"{r['localization_m']*1000:.1f}",f"{r['peak_change_db']:.2f}"] for r in p['physical_error']['rows']]
    refrows=[[s,f"{p[s]['map']['complex_correlation']:.3f}",f"{p[s]['map']['amplitude_correlation']:.3f}",f"{p[s]['map']['aligned_nrmse']:.3f}",f"{p[s]['bp_seconds_including_possible_jit']:.2f} s"] for s in ['scene00','scene01']]
    cases=''.join(f'<div class="case" data-case="{int(c)}"'+(' hidden' if i else '')+'>'+img(f'focus_case_{int(c)}.png',f'고정 시험 사례 {c}의 보정 비교')+'</div>' for i,c in enumerate([0,1,36,37]))
    return f'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SAS 신호처리·보정 재현 실험 | 2026.10.20</title>
<style>
:root{{--ink:#203344;--muted:#5b6f80;--line:#dce4eb;--bg:#f2f5f7;--teal:#206477;--amber:#9b6818}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.75 system-ui,-apple-system,"Noto Sans CJK KR",sans-serif}}a{{color:var(--teal)}}header{{background:#203344;color:#fff;padding:48px max(24px,calc((100vw - 1100px)/2)) 38px}}header .eyebrow{{color:#a9d1dc;letter-spacing:.16em;font-size:12px;font-weight:700}}h1{{font-size:clamp(28px,4vw,44px);line-height:1.3;margin:14px 0}}header p{{max-width:850px;color:#d4e0e9}}.tags{{display:flex;gap:8px;flex-wrap:wrap}}.tag{{border:1px solid #607889;padding:3px 10px;border-radius:30px;font-size:12px}}main{{max-width:1150px;margin:auto;padding:28px 24px 70px}}nav{{display:flex;flex-wrap:wrap;gap:16px;margin:0 0 28px;font-size:14px}}nav a{{text-decoration:none}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}.card,section{{background:white;border:1px solid var(--line);border-radius:12px;padding:25px}}.card .value{{font-size:28px;font-weight:750;line-height:1.3}}.card .label{{color:var(--muted);font-size:13px;margin-top:6px}}.card .sub{{font-size:12px;color:var(--muted)}}section{{margin-top:22px}}h2{{font-size:24px;line-height:1.4;margin:0 0 18px}}h2 small{{display:block;color:var(--teal);font-size:12px;letter-spacing:.12em;margin-bottom:6px}}h3{{font-size:18px;margin:26px 0 10px}}p{{margin:12px 0}}.lead{{font-size:18px}}.note{{border-left:4px solid #d29b3c;background:#fbf6eb;padding:14px 18px;margin:18px 0}}.ok{{border-color:#206477;background:#f0f7f8}}.small,figcaption{{font-size:13px;color:var(--muted)}}.two{{display:grid;grid-template-columns:1fr 1fr;gap:22px}}.table-wrap{{overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:14px;text-align:left}}th{{background:#f0f4f7;white-space:nowrap}}th,td{{padding:10px 12px;border-bottom:1px solid var(--line);vertical-align:top}}tbody tr:last-child td{{border-bottom:0}}.plot{{width:100%;height:auto;border-radius:5px;display:block;margin:14px 0}}figure{{margin:16px 0}}pre{{background:#162836;color:#e3eff4;padding:18px;border-radius:8px;overflow:auto;font-size:13px;line-height:1.65}}code{{font-family:ui-monospace,Consolas,monospace}}.formula{{background:#f0f4f7;padding:16px;border-radius:8px;font-family:ui-monospace,monospace;overflow:auto}}button,.button{{border:0;border-radius:7px;background:#206477;color:#fff;padding:12px 18px;cursor:pointer;font:inherit;font-size:14px;text-decoration:none;display:inline-block}}select{{font:inherit;padding:7px 12px;border:1px solid var(--line);border-radius:6px;background:white}}details{{border-top:1px solid var(--line);margin-top:20px;padding-top:14px}}summary{{cursor:pointer;font-weight:650}}footer{{margin:30px 0;color:var(--muted);font-size:12px}}li{{margin:8px 0}}.case[hidden]{{display:none}}@media(max-width:750px){{.cards{{grid-template-columns:1fr 1fr}}.two{{grid-template-columns:1fr}}section{{padding:20px}}header{{padding:32px 24px}}main{{padding:20px 14px}}.card{{padding:17px}}.card .value{{font-size:24px}}}}@media print{{nav,button,#bundle-placeholder{{display:none}}body{{background:white}}section{{break-inside:avoid}}header{{background:white;color:var(--ink)}}header p{{color:var(--muted)}}}}
</style></head><body>
<header><div class="eyebrow">SAS FOCUS BENCHMARK · EXECUTED STUDY · 2026.10.20</div><h1>원시 SAS에서 영상 형성까지,<br>보정 효과를 수치로 확인하다</h1><p>기뢰대응 프로젝트의 신호처리 기반 실험. 공개 PoSSM 원시 신호의 재구성, 항법·음속 가정 오류의 영향, 전통 autofocus와 작은 학습 모델의 비교를 실행했습니다.</p><div class="tags"><span class="tag">수중 음향 시뮬레이션</span><span class="tag">CPU 실행</span><span class="tag">원시 IQ + 복소수 SLC</span><span class="tag">실해역 기뢰 탐지 평가 제외</span></div></header>
<main><nav><a href="#data">01 자료</a><a href="#imaging">02 영상 형성</a><a href="#physics">03 물리 오차</a><a href="#focus">04 보정 비교</a><a href="#decision">05 판단</a><a href="#reproduce">06 재현·출처</a></nav>
<div class="cards"><div class="card"><div class="value">2 scenes</div><div class="label">원시 신호·기준 영상 확보</div><div class="sub">총 200핑 · 각 핑 36채널</div></div><div class="card"><div class="value">{sloc:.1f} → {mloc:.1f}</div><div class="label">위치 오차 중앙값 (mm)</div><div class="sub">산란체 24개 · 평가 격자 5 mm</div></div><div class="card"><div class="value">0.949</div><div class="label">전통 보정 평균 complex NCC</div><div class="sub">별도 scene01 · 인공오차 96조건</div></div><div class="card"><div class="value">CNN 미채택</div><div class="label">해저 장면으로의 전이 실패</div><div class="sub">빠른 추론이 정확도를 보장하지 않음</div></div></div>
<section><p class="lead"><strong>신호처리 기준선은 실행했고, 작은 CNN은 현재 데이터에서 채택할 근거를 얻지 못했습니다.</strong></p><p>가장 분명한 개선은 음파 왕복 중 수신기 이동을 반영해 위치 편향을 줄인 것입니다. 반면 제공 ASASIN 복소수 영상과의 일치는 낮게 남았습니다. 이를 재현 성공으로 포장하지 않고, 독립적인 산란체 위치 검증과 제공 영상 비교를 함께 제시합니다.</p><div id="bundle-placeholder"><p class="small">코드 묶음의 README.md와 src/에서 전체 재현 절차를 확인할 수 있습니다.</p></div></section>
<section id="data"><h2><small>01 / DATA PROVENANCE</small>무엇을 실제로 받았는가</h2>
{table(['자료','실제 확보·사용','용도'],[['PoSSM scene00','GDF 4개 + SLC HDF5 1개','338개 점산란체; 시간·위상 규약 및 위치 검증'],['PoSSM scene01','GDF 4개 + SLC HDF5 1개','해저 배경 + 통발·암석·원통; 별도 보정 시험'],['메타데이터','소자 좌표, IQ 배율, navigation, 파형, GT CSV, 공식 PDF','단위·좌표·체크섬·다운로드 해시 확인']])}
<p>두 장면의 원본 파일은 압축 해제 기준 총 {sourcebytes/1e6:.1f} MB입니다. 각 장면의 IQ는 <code>[100핑, 36채널, 10942복소수 샘플]</code>, 복제 파형은 300샘플입니다. 반송파 300 kHz, 대역폭 60 kHz, LFM 4 ms, 복소수 표본율 75 kHz, 이동 속도 0.6 m/s입니다.</p>
<div class="note"><strong>자료의 성격:</strong> 실제 물속에서 측정한 자료가 아니라 수중 음향 모델로 생성한 공개 데이터입니다. 원자료는 무잡음·이상적인 직선 항로 조건입니다. scene01의 원통을 실제 기뢰 라벨로 간주하지 않았으며 탐지율·FROC·mAP는 계산하지 않았습니다.</div><p class="small">원자료 장면은 약 60×90 m입니다. 제공 SLC 좌표에 맞춘 이번 지도는 x≈4.50–60.48 m, y=0–89.98 m이며 10 cm 간격으로 비교했습니다. 원본 SLC 표본 간격은 2.5 cm입니다.</p></section>
<section id="imaging"><h2><small>02 / RAW SIGNAL → IMAGE</small>매치드필터만으로 SAS 영상이 완성되지는 않는다</h2><p>매치드필터는 송신 파형의 시간 구조를 압축합니다. 그다음 여러 핑·수신 소자의 <strong>왕복 전파 지연과 위상</strong>을 맞춰 더해야 합성개구가 형성됩니다. 이번 코드는 송신·수신 경로를 따로 계산하는 시간영역 역투영을 사용했습니다.</p>
<div class="formula">yₘ(t) = rₘ(t) ∗ s*(−t)<br>τₘ(p) = (‖p−Txₘ‖ + ‖p−Rxₘ(τₘ)‖) / cₘ<br>I(p) = Σₘ yₘ(τₘ(p)) · exp(+j 2π f꜀ τₘ(p))</div>
<p class="small">∗는 convolution, s*(−t)는 conjugate-reversed replica입니다. 표기 부호는 이번 공개 baseband 자료의 규약입니다. RF/analytic 신호에 같은 반송파 보정을 무조건 중복 적용하면 안 됩니다.</p>
{table(['고정 조건','구현'],[['GDF 해석','sync/경계/navigation checksum 검사; complex int16 ÷ 2¹⁵'],['압축 및 시간','복제 파형 상관, full convolution의 299 input-sample 오프셋'],['보간·개구','4배 polyphase resampling + 복소수 선형 보간; half aperture 0.05 rad'],['수신기 이동','왕복 시간 동안 Rx가 이동; 지연을 fixed-point 3회로 계산'],['조건 확인과 평가','scene00 개발 인덱스 1/2/4에서 규약 확인 후 고정; 추가 24개 점에서 평가']])}
{img('localization.png','24개 산란체의 위치 오차와 고정 첫 점의 단면')}
<p>수신기 이동 반영 후 위치 오차 중앙값은 <strong>{mloc:.2f} mm</strong>였습니다. 평가는 정답에 맞춰 이동한 격자가 아닌 5 mm 고정 월드 격자에서 수행했습니다. 이는 격자 양자화의 영향을 받는 수치이며 “실해역 1.8 mm 정확도”를 의미하지 않습니다. along-track 3 dB 폭 중앙값은 약 {p['heldout_points']['continuous_receiver']['width_3db_x_m']*1000:.0f} mm였습니다.</p>
<h3>제공 영상과의 비교: 잔여 차이 공개</h3>{table(['장면','complex NCC ↑','amplitude Pearson ↑','scale-aligned NRMSE ↓','BP 시간¹'],refrows)}
<p class="small">¹ 900×560 비교 격자, CPU 8 Numba threads. 첫 장면은 JIT 비용이 섞일 수 있습니다. resampling은 각각 {p['scene00']['resampling_seconds']:.1f}/{p['scene01']['resampling_seconds']:.1f}초이며 별도입니다. 원본 2.5 cm 전면 영상의 처리 시간으로 해석하면 안 됩니다. complex NCC = |〈ref, image〉| / (‖ref‖ ‖image‖)이며 평균을 빼지 않습니다. NRMSE만 최적의 전역 복소수 스칼라를 정합했습니다. 위치 정합이나 픽셀별 위상 보정은 하지 않았습니다.</p>
<div class="note"><strong>미해결:</strong> 점의 위치는 맞췄지만 ASASIN SLC의 복소수 출력을 재현했다고 말할 수 없습니다. 원인은 확정하지 않았습니다. 위상 기준, 개구 가중치, 보간·beamforming 세부 구현은 추가 확인이 필요합니다.</div>
<label for="map-select">지도 선택 </label><select id="map-select"><option value="scene01">scene01 · 해저 배경·객체</option><option value="scene00">scene00 · 점산란체</option></select><div id="map-scene01">{img('scene01_maps.png','scene01 제공 영상과 직접 재구성한 영상')}</div><div id="map-scene00" hidden>{img('scene00_maps.png','scene00 제공 영상과 직접 재구성한 영상')}</div><p class="small">표시용 dB는 각 영상의 최대값으로 정규화했습니다. 시각적 유사성을 수치 일치로 간주하지 않습니다.</p></section>
<section id="physics"><h2><small>03 / CONTROLLED PHYSICAL ERRORS</small>위치와 음속의 작은 오차가 왜 중요한가</h2><p>300 kHz, c=1500 m/s에서 파장은 약 5 mm입니다. broadside 왕복 경로를 단순화하면 0.5 mm 거리 오차도 약 1.26 rad 위상 차이를 만들 수 있습니다. 단순한 convolution보다 기하·시간 동기와 코히어런스 관리가 중요한 이유입니다.</p><p>동일한 원시 IQ를 고정한 뒤, 역투영이 가정하는 cross-track 좌표 또는 일정 음속을 바꿨습니다. 좌표 오차는 두 정현파를 합친 부드러운 ping별 변화로 만들었습니다. 아래는 고정 산란체 인덱스 {p['physical_error']['point_index_0based']} 하나의 민감도입니다.</p>
{img('physical_sensitivity.png','좌표 오차와 음속 편향에 따른 무오차 역투영과의 복소수 상관 변화')}
{table(['오차 유형','가정 오차','complex NCC','위치 오차 mm','peak 변화 dB'],physics_rows)}
<p class="small">실제 환경을 바꿔 취득한 자료가 아닙니다. 오차 곡선은 간섭과 sidelobe 때문에 단조롭지 않을 수 있습니다. peak 값의 감소만으로 초점 손실을 판단하면 위치 이동이나 오정합을 놓칠 수 있어 complex NCC와 위치 오차를 함께 기록했습니다. 여러 해역에서의 통계 결과도 아닙니다.</p></section>
<section id="focus"><h2><small>04 / CLASSICAL VS LEARNED AUTOFOCUS</small>물리 오차 실험과 구분한 학습·파인튜닝</h2><p>이 비교는 <strong>복소수 SLC에 인공적으로 주입한 2차·3차 along-track 주파수 위상오차</strong>를 제거하는 문제입니다. 원시 항법·음속 오차를 직접 추정하는 실험과는 다릅니다. 깨끗한 SLC는 상대적 복원 평가에만 사용하며 실제로 완벽히 초점이 맞은 정답이라고 주장하지 않습니다.</p>
{table(['단계','자료와 분리','학습 조건'],[['절차 생성 사전학습','복소수 Gaussian 산란체 train1024 / val128 / test256; 서로 다른 seed','PhaseNet 69,266파라미터; AdamW 1e-3; 최대60 epoch, patience12; 선택 epoch '+str(l['best_epoch'])],['scene00 파인튜닝','train '+str(t['train_source_patches'])+' / val '+str(t['validation_source_patches'])+' 원패치; x 범위 분리; 증강1024/256','사전학습 가중치에서 시작; AdamW 1e-4, 최대30 epoch, patience8; 선택 epoch '+str(t['fine_tuning']['best_epoch'])],['최종 시험','scene01 객체9 + 고정 grid15 = 24패치; 패치마다4오차','총96조건; 시험 장면으로 epoch·학습률을 고르지 않음'],['전통 기준','같은 손상 입력; 정답 없이 image entropy 최소화','Adam70회 × 시작점3개; 최종 entropy로 시작점 선택']])}
<p class="small">PhaseNet은 이 프로젝트의 작은 자체 기준 모델이며 Gerg & Monga의 Deep Autofocus 구조·가중치를 재현한 모델이 아닙니다. fine-tune 증강 횟수는 독립적인 원장면 수가 아닙니다. 실제 해저 장면은 한 개만 시험했습니다.</p>
<h3>시험 결과 · scene01</h3>{table(['방법','평균 NCC ↑','중앙값 NCC ↑','평균 aligned NRMSE ↓','CPU 총 지연시간²'],focus_rows)}
<p class="small">² 두 CNN은 feature 생성+forward+phase correction을 포함합니다. 전통 보정은 반복 최적화를 포함합니다. 동일 CPU, torch4 threads, 96조건 batch, 한 번의 실행값입니다. 신뢰구간이나 production latency benchmark로 해석하지 않습니다. 절차 생성 사전학습 약 {l['training_seconds']:.1f}초, 파인튜닝 약 {t['fine_tuning']['seconds']:.1f}초이며 데이터 생성·feature 준비는 학습 시간에서 제외했습니다.</p>
{img('focus_scores.png','96조건 상관 분포와 학습 validation 곡선')}
<p>이상적인 생성 시험에서는 CNN 평균 NCC가 {l['learned']['complex_correlation']['mean']:.3f}까지 회복됐지만, 공개 해저 장면에서는 {t['procedural_model']['complex_correlation']['mean']:.3f}에 그쳤습니다. scene00 파인튜닝 후에도 {t['finetuned']['complex_correlation']['mean']:.3f}로, 손상 영상의 {t['corrupted']['complex_correlation']['mean']:.3f}와 거의 같습니다. 현재 설정에서는 분포 차이를 극복하지 못했습니다.</p>
<h3>이미 초점이 맞은 대조 입력</h3>{table(['방법','무오차 입력 대비 평균 NCC','평균 aligned NRMSE'],controls)}<p class="small">아무 처리를 하지 않은 대조 입력의 NCC는 1입니다. 엔트로피 최적화도 원본을 일부 바꾸지만 CNN의 훼손이 더 큽니다. 손상 정도를 판단하지 않고 모든 영상에 보정을 적용하면 안 된다는 결과입니다.</p>
<h3>고정 사례 보기</h3><label for="case-select">시험 사례 </label><select id="case-select"><option value="0">첫 객체 · 첫 위상오차</option><option value="1">첫 객체 · 두 번째 위상오차</option><option value="36">첫 고정 grid · 첫 위상오차</option><option value="37">첫 고정 grid · 두 번째 위상오차</option></select>{cases}<p class="small">좋은 사례를 검색해 고르지 않고 첫 객체와 첫 grid의 두 오차를 고정해 표시했습니다. 모든 조건별 수치는 코드 묶음의 transfer_metrics.npz에 포함됩니다.</p></section>
<section id="decision"><h2><small>05 / DECISION & NEXT CRITERIA</small>이번 결과를 어떻게 사용할 것인가</h2><div class="note ok"><strong>채택:</strong> 공개 원시 신호를 읽고 지연·기하·위상을 추적하는 기준선과, 물리 오차를 수치로 확인하는 실험 구조.<br><strong>보류:</strong> 작은 PhaseNet을 해저 장면 보정기로 채택하는 결정. 파인튜닝으로도 정확도와 무오차 입력 보존이 충분하지 않았습니다.</div><p>기뢰대응 프로젝트에서 이번 작업의 실익은 SSS 탐지기 확대와 다른 <strong>원시 신호 → SAS 영상 형성 → 오차 분석</strong> 단계를 증명하는 데 있습니다. 실제 기뢰 ATR을 보완하려면 원시 실측 데이터, 표적 라벨, 해역·항차·관측각 분리 평가가 추가로 필요합니다.</p>
<h3>후속 파인튜닝에 고정할 제안 기준</h3><p class="small">이번 결과를 확인한 뒤 제안한 기준이며 사전등록한 합격 기준은 아닙니다.</p><ol><li>해역/항차/표적 인스턴스 기준으로 분리하고 인접 crop 누수를 검사합니다.</li><li>validation으로 학습률과 epoch를 선택하고 시험 자료는 마지막에 한 번 평가합니다.</li><li>손상 대비 평균 NCC +0.05 이상, 무오차 입력 NCC 감소 0.01 이하를 동시에 요구합니다.</li><li>전통 보정의 정확도를 기준으로 삼고 전처리를 포함한 지연시간을 같은 장치에서 비교합니다.</li><li>원시 위치·음속 오류에서 직접 성능을 확인하기 전에는 SLC 인공오차 결과를 물리 보정 성능으로 일반화하지 않습니다.</li></ol><p>추가적인 모델 크기 확대보다 먼저 실해역 또는 더 다양한 해저 배경을 확보하고, 제공 SLC와의 위상 차이를 설명하는 것이 다음 우선순위입니다. 재구성·보정만으로 기뢰 탐지율이나 안전 운용 성능을 주장하지 않습니다.</p></section>
<section id="reproduce"><h2><small>06 / REPRODUCIBILITY & SOURCES</small>코드·가중치·결과를 함께 보관</h2><p>묶음에는 실행 코드, 데이터 다운로드 manifest와 SHA-256, 공식 형식 문서, GT 위치 CSV, 두 학습 가중치, 개별 수치 결과, 이 HTML 보고서가 포함됩니다. 대용량 원시 IQ와 중간 캐시는 제외했으며 공개 저장소에서 재취득하는 코드를 제공합니다.</p>
<pre><code>python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python src/run_all.py --download</code></pre>
<p class="small">검증 환경: Python 3.12.14, Linux CPU, torch 2.5.1+cpu. 별도 저자 요청 없이 공개 MGDS 다운로드 폼으로 취득했습니다. 내려받기 코드의 실제 소파일 재다운로드·체크섬 검사도 통과했습니다. 다른 운영체제와 GPU는 검증하지 않았습니다.</p>
<details><summary>검증 {verification['passed']}개와 결과 파일</summary><ul>{''.join('<li>'+html.escape(c)+'</li>' for c in verification['checks'])}</ul><p><code>results/possm/possm_results.json</code> — 원시 재구성·점 검증·오차 민감도<br><code>results/learning/</code> — 절차 생성 학습 및 checkpoint<br><code>results/transfer/</code> — scene00 fine-tune, scene01 시험 및 checkpoint<br><code>references/download_manifest.json</code> — 입력 파일의 URL 식별자·SHA-256<br><code>README.md</code> — 처리 규약·실행·제한<br><code>THIRD_PARTY.md</code> — 출처·라이선스 구분</p></details>
<h3>1차 자료와 연구 연결</h3><ul><li>Jason Philtron, Shawn Johnson, Daniel Brown / MGDS: <a href="https://www.marine-geo.org/tools/files/31898">PoSSM raw GDF · DOI 10.60521/331898</a>, <a href="https://www.marine-geo.org/tools/files/31899">SLC HDF5 · DOI 10.60521/331899</a>. 공식 data-package/GDF 문서와 위치 CSV를 함께 사용했습니다.</li><li>Gerg & Monga (2021), <a href="https://arxiv.org/abs/2103.10312">Real-Time, Deep SAS Autofocus</a>: 영상 특징에서 위상 보정을 추정하는 연구 방향. 이번 소형 모델은 그 논문의 재현이 아닙니다.</li><li>Reed et al. (2023), <a href="https://arxiv.org/abs/2306.09909">Neural Volumetric Reconstruction for Coherent SAS</a>: 물리 기반 neural reconstruction의 별도 확장 방향. 이번 실험에서는 구현하지 않았습니다.</li></ul>
<p class="small">코드: MIT (원본 Python에 한함). PoSSM 및 파생 결과·보고서·가중치: <a href="https://creativecommons.org/licenses/by-nc-sa/3.0/us/">CC BY-NC-SA 3.0 US</a>. 데이터의 비상업·동일조건 제한을 코드의 MIT와 혼동하지 마십시오. 본 보고서에는 실해역 실측 SAS, In-air 재구성 실행, 기뢰 탐지기 학습 결과가 포함되지 않습니다.</p></section>
<footer>SAS Focus Benchmark · 생성일 2026-10-20 · 수치는 실행 산출물에서 직접 생성. 남은 한계: 제공 SLC와의 불일치, 작은 학습 자료, 단일 해저 시험 장면, 이상적 무잡음 시뮬레이션, 제한된 위상오차 모델.</footer></main>
<script>document.getElementById('case-select').addEventListener('change',e=>{{document.querySelectorAll('.case').forEach(x=>x.hidden=x.dataset.case!==e.target.value)}});document.getElementById('map-select').addEventListener('change',e=>{{['scene00','scene01'].forEach(s=>document.getElementById('map-'+s).hidden=s!==e.target.value)}});</script></body></html>'''


def bundle(text):
    deliver=ROOT/'deliverables';deliver.mkdir(exist_ok=True)
    archive=deliver/'SAS_Focus_Reproducibility.zip'
    paths=[ROOT/name for name in ['README.md','requirements.txt','LICENSE','THIRD_PARTY.md']]
    for folder in ['src','references','results']:
        paths += [p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pyc','.log']]
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(paths):z.write(p,'sas_focus_benchmark/'+str(p.relative_to(ROOT)))
        z.writestr('sas_focus_benchmark/report/report.html',text)
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    b64=base64.b64encode(archive.read_bytes()).decode()
    button=f'''<button id="download-bundle">코드·결과·가중치 ZIP 받기 ({archive.stat().st_size/1e6:.1f} MB)</button><p class="small">보고서 안에 포함된 오프라인 묶음입니다. ZIP을 풀면 report/report.html에서 보고서를 다시 볼 수 있습니다.<br>ZIP SHA-256: <code>{digest}</code></p>'''
    start=text.index('<div id="bundle-placeholder">');end=text.index('</div>',start)+6
    text=text[:start]+f'<div id="bundle-placeholder">{button}</div>'+text[end:]
    script='''<script id="bundle-data" type="application/octet-stream">'''+b64+'''</script><script>document.getElementById('download-bundle').addEventListener('click',()=>{const b=atob(document.getElementById('bundle-data').textContent.trim());const bytes=new Uint8Array(b.length);for(let i=0;i<b.length;i++)bytes[i]=b.charCodeAt(i);const url=URL.createObjectURL(new Blob([bytes],{type:'application/zip'}));const a=document.createElement('a');a.href=url;a.download='SAS_Focus_Reproducibility.zip';a.click();setTimeout(()=>URL.revokeObjectURL(url),60000)});</script>'''
    delivery=deliver/'SAS_Focus_Report.html';delivery.write_text(text.replace('</body>',script+'</body>'))
    print(json.dumps({'html':str(delivery),'bytes':delivery.stat().st_size,'zip_sha256':digest,'zip_bytes':archive.stat().st_size}))


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',action='store_true');a=p.parse_args();setup()
    physics=load('results/possm/possm_results.json');transfer=load('results/transfer/transfer_results.json');learning=load('results/learning/learning_results.json')
    figures(physics,transfer,learning);text=report(physics,transfer,learning)
    (REPORT/'report.html').write_text(text)
    if a.bundle:bundle(text)
    print('Report:',REPORT/'report.html')


if __name__=='__main__':main()
