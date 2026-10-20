# SAS Focus Benchmark v2 — first checkpoint

요청하신 **1번(SLC 차이·native 격자·PSF·sub-pixel 위치)의 중간 결과**입니다. 2번의 전 자료 확보·분할, 4번의 24점×5seed 민감도, 6번의 실측 자료 형식·접근성 조사도 실행했습니다. PGA/핑별 entropy 보정기 비교(3번)와 탐지기 학습·FROC(5번)는 이 checkpoint에서 실행하지 않았습니다. 1차 CNN을 재학습하지 않았습니다.

## 보기

`report_v2/report.html`이 오프라인 보고서입니다. 숫자는 `results/v2/`에서 직접 읽어 생성합니다. 원래 v1과 같은 격자·같은 조건으로 비교한 숫자를 별도로 보여 줍니다. v1의 10 cm 격자 결과와 v2의 2.5 cm 격자 결과를 직접 개선률로 나누지 않습니다.

## 재실행

Linux CPU / Python3.12.14 / torch2.5.1+cpu 환경에서 각 단계를 실제 실행했습니다. RAM8GB 이상, 데이터·캐시 여유 디스크5GB 이상을 권합니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python src/run_v2.py --download
```

위 명령은 모두 공개 PoSSM 자료입니다. 원시 GDF44개와 SLC11개, 약2.44GB(압축 해제 기준)를 받고 무결성을 검사합니다. 저자 요청·메일 전송이 아닙니다. 전체 행렬의 BP를 2개 장면에서 실행하며 native격자는 각8,064,000픽셀입니다. 중간 resampling 배열 때문에 메모리 사용량이 높아집니다. 환경이 작으면 장면별로 실행하십시오.

개별 실행 순서:

```bash
python src/acquire_all_scenes.py
python src/prepare_possm.py data/possm_scene00 --output data/scene00_cache
python src/prepare_possm.py data/possm_scene01 --output data/scene01_cache
python src/diagnose_slc.py
python src/diagnose_taper.py
python src/validate_slc_v2.py
python src/stress_points_v2.py
python src/audit_sources_v2.py
python src/diagnose_gain.py
python src/verify_v2.py
python src/build_report_v2.py --bundle
```

현재 보고서만 다시 만들려면 마지막 명령만 사용합니다. 코드 묶음은 원시자료·캐시 및 native 전면 미리보기 NPZ를 제외하고, 보고서용 PNG와 모든 수치 JSON·점 검증 complex NPZ를 포함합니다. 미리보기 NPZ가 없으면 이미 포함된 보고서용 그림을 재사용합니다. 전체 재실행은 NPZ도 다시 만듭니다. 원시 입력은 download_manifest_v2.json으로 SHA-256을 검증합니다.

1차 CNN 가중치2개와 학습 메타데이터는 참고용으로 포함합니다. 이번 checkpoint에서 새로 훈련한 가중치가 아닙니다.

## 핵심 해석

- ±0.05 rad는 v1의 실제 설정이며 16° 기본값은 사용되지 않았습니다. HDF5 AzLimit도0.05이고 설명은HWHP입니다. 값이 같다는 사실이 전체 aperture/taper 규약이 같다는 뜻은 아닙니다.
- 공식 PDF의 Gaussian은 전방 모델의 센서 지향성입니다. ASASIN 역투영의 가중치 명세가 아닙니다. 이번 Gaussian은 비교용 복원 설계입니다. full3D 모드의 수직 boresight=수평이라는 가정도 공식 재현이라고 주장하지 않습니다.
- 4×/8×/16× 보간만 바꿔서는 차이가 거의 줄지 않았습니다. 개발용3점에서 range-spectrum Hann60 + horizontal Gaussian +8×를 선택했습니다. 24점의 native 평가에서는 이 설정을 다시 고르지 않았습니다.
- 주입 전/후의 PSF 폭이 달라집니다. 창 적용은 sidelobe를 줄이지만 mainlobe를 넓힙니다. 그 변화가 기준 모양과 더 비슷한 것이지 물리 분해능이 좋아졌다는 결과가 아닙니다.
- native SLC의2.5cm 표본 간격은 약1cm급 range PSF를 충분히 표본화하지 못합니다. 기준 SLC에서 세밀한3dB폭·PSLR를 확정할 수 없어 sampling-limited로 표시합니다. 자체 BP의 조밀한1mm격자 PSF와 분리합니다.
- sub-pixel 보간은 위치 편향과 양자화를 구분하기 위한 것입니다. 무잡음·정확한 기하·알려진 점산란체 조건에서 나오는 μm 수준 오차를 실해역 성능으로 해석하지 않습니다.
- y²+6² 진폭 보정은 1/r² 전파를 고려한 **사후 진단**입니다. primary24점/전체격자 frozen 결과와 분리하며, 실제 ASASIN 설정이라고 주장하지 않습니다.

## 데이터 분할과 라벨

후속 탐지 모델의 계획: train02/03/07/08(rough uniform,36배치), validation01/06(uniform,18배치), test04/05/09/10(ripple/mixed,36배치). 이름으로 정의된 해저 family는 세 집합 사이에서 겹치지 않습니다. mixed 안에는 uniform/ripple 성분이 있어 모든 물리가 독립이라는 뜻은 아닙니다. 90배치는90개 독립 표적 설계가 아닙니다.

위치 CSV는 객체 삽입점·회전각이며 bbox/mask가 아닙니다. 좌표 기반 FROC를 정의할 수 있지만 bbox mAP를 계산하려면 검증된 영상 주석이 더 필요합니다. 탐지기에 GT crop 좌표를 입력해 놓고 end-to-end 탐지라고 하지 않습니다.

## 라이선스·확인 범위

PoSSM: CC BY-NC-SA3.0US. 원본 Python: MIT. 공식PDF/원자료 및 파생 수치·그림·보고서 조건은THIRD_PARTY.md와 같습니다. 이번에 조사한 ApertureLab은 별도ALSD-1.0이고 특정 조직·계약/프로그램 제한이 있습니다. 해당 데이터·가중치는 이 묶음에 포함하지 않았습니다. 그 자료를 훈련/평가에 사용하지 않았습니다.

In-air 파일은 HDF5 실수 전압 파형이고 PoSSM GDF 파서로 직접 읽히지 않습니다. 별도 입력 adapter와 공기 중 전파·계측 전처리가 필요합니다. SCTD3.0은 현재README/LICENSE만 확인됐고 데이터 라이선스와 다운로드 경로를 확인하지 못했습니다. 자세한 근거 링크는 보고서에 있습니다.
