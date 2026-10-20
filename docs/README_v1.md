# SAS Focus Benchmark — executed CPU study

공개 PoSSM 원시 SAS를 이용한 신호처리·보정 실험입니다. 기뢰대응 분야의 영상 형성 역량을 입증하는 기반 작업이며 **기뢰 탐지/식별 성능 실험이 아닙니다**. PoSSM은 수중 음향 **시뮬레이션**이며 실해역 실측 자료가 아닙니다.

## 결과 보기

`report/report.html`을 브라우저에서 열면 모든 그림과 표를 오프라인으로 볼 수 있습니다. `results/`에 JSON/NPZ 원자료와 학습 가중치가 있습니다. 외부 코드나 가중치를 재현했다고 주장하지 않습니다. 작은 PhaseNet은 이 프로젝트의 자체 기준 모델입니다.

실행 결과 요약:

- scene00/01 각각 100핑 × 36채널, 원시 IQ → 복제 신호 상관(매치드필터) → 복소수 보간 → bistatic 역투영.
- 기준선의 수신기 이동 보정 후 산란체 24개의 위치오차 중앙값: 약 15.1 → 1.8 mm (5 mm 월드 좌표 격자). 이는 초점 개선률이나 실해역 위치 정확도 보장이 아닙니다.
- 별도 scene01의 24패치 × 4위상오차 = 96조건에서 평균 complex NCC: 손상 0.760, 엔트로피 0.949, 절차 생성 영상 학습 CNN 0.755, scene00 파인튜닝 CNN 0.762. 현재 CNN은 채택하지 않습니다.
- 제공 ASASIN SLC와 전면 일치하지 않습니다. 10 cm 격자 비교 complex NCC: scene00 0.264, scene01 0.577. 남은 위상·진폭 처리 차이는 미해결입니다.

## 환경과 실행

검증 환경: Linux x86_64, Python 3.12.14, CPU, torch 2.5.1+cpu. GPU는 사용하지 않았습니다. 현재 환경의 정확한 패키지 버전은 `results/environment.json`을 참고하십시오. 데이터·캐시·실행 중 메모리를 위해 RAM 8 GB 이상과 디스크 여유 3 GB 이상을 권합니다. 전체 데이터셋이 아니라 scene00/01만 다운로드합니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python src/run_all.py --download
```

이미 받은 데이터가 있으면 `--download`를 생략합니다. 보고서만 재생성하려면:

```bash
python src/verify_results.py
python src/build_report.py
```

한국어 그림 글꼴은 Noto Sans CJK KR을 사용했습니다. 없으면 matplotlib 기본 글꼴로 대체되어 일부 한국어가 표시되지 않을 수 있습니다. 수치와 HTML 본문에는 영향이 없습니다.

개별 실행:

```bash
python src/download_possm.py
python src/gdf.py data/possm_scene00 --output results/possm_gdf_audit.json
python src/prepare_possm.py data/possm_scene00 --output data/scene00_cache
python src/prepare_possm.py data/possm_scene01 --output data/scene01_cache
python src/evaluate_possm.py
python src/focus_learning.py --output results/learning
python src/transfer_focus.py
```

원시 GDF/제공 HDF5 및 캐시는 크기 때문에 코드 묶음에서 제외했습니다. 이 작업에서는 두 장면을 실제 다운로드하고 처리했습니다. `download_possm.py`가 공개 MGDS 저장소에서 다시 받고 `references/download_manifest.json`의 바이트 수·SHA-256을 확인합니다. 저자에게 요청 메일을 보내는 방식이 아닙니다. 원시 파서가 navigation checksum을 검사하며 복소수 int16 배율과 좌표·단위도 명시적으로 해석합니다.

## 처리와 평가의 구분

1. **원시 신호 기준선:** KBID=1 샘플의 conjugate reversal, 299 input sample 지연, carrier phase +부호, 4배 resampling, half aperture 0.05 rad. 초기 조건은 scene00 산란체 인덱스 1/2/4에서 확인했습니다. 24개 추가 점은 이 조건 확인에 사용하지 않았습니다. 반사체 위치는 GT 기준으로 평행이동 정합하지 않았고 5 mm 고정 월드 격자를 사용했습니다.
2. **물리 오차:** 원시 IQ를 고정하고 BP의 가정 좌표/음속만 바꾸었습니다. 센서 또는 실제 해양 환경을 변경해 수집한 데이터가 아닙니다. 한 산란체의 예시 민감도이며 오차 곡선은 간섭 때문에 단조적일 필요가 없습니다.
3. **Autofocus:** 원시 위치/음속 오류와 별도로, 복소수 SLC의 along-track FFT에 알려진 2/3차 위상오차를 주입했습니다. 두 계수 모델은 실제 오차의 모든 형태를 포괄하지 않습니다. 주입 전 SLC를 상대적 복원 기준으로만 사용합니다.
4. **학습/파인튜닝:** 절차 생성 train1024/val128/test256, seed1101/2202/3303. fine-tune은 scene00의 공간 분리 train34/val6 원패치에 각각 1024/256회 인공오차를 생성했습니다. 증강 횟수는 독립 장면 수가 아닙니다. scene01은 fine-tune 및 epoch 선택에서 제외했습니다. 표준화/peak centering은 손상 영상만 사용합니다. 시험 장면은 하나이므로 실해역 일반화나 통계적 독립 96장면으로 해석하지 않습니다.
5. **전통 비교:** 2개 위상계수를 image entropy로 최적화(Adam70회, 초기값 -3/0/+3). 정답 영상은 최적화나 시작점 선택에 사용하지 않습니다. oracle phase 제거는 상한/산술 검증에만 사용합니다.

## 후속 파인튜닝 채택 기준

아래는 이번 결과를 바탕으로 **후속 실험에 고정할 제안 기준**이며 사전등록한 기준이 아닙니다.

- 실해역 자료를 구할 경우 해역/항차/표적 인스턴스 기준으로 train/val/test를 분리하고, 붙어 있는 crop을 랜덤 분리하지 않습니다.
- 학습률 1e-4, 최대30 epoch, validation 기반 선택을 시작값으로 고정하고 test로 하이퍼파라미터를 고르지 않습니다.
- 손상 대비 평균 NCC +0.05 이상, 이미 초점이 맞은 대조군 NCC 감소 0.01 이하를 요구합니다.
- 전통 보정 대비 정확도 차이가 허용 범위인지 확인한 다음 전처리를 포함한 지연시간을 비교합니다. CPU 측정과 GPU 측정을 섞지 않습니다.
- 원시 위치/음속 오차에서 직접 효과를 다시 검증하기 전에는 SLC 인공오차 결과를 물리 보정 성능으로 주장하지 않습니다.

현재 모델은 위 제안 기준을 충족하지 못합니다. 작은 이상적 점산란체에서 배운 위상 보정이 해저 배경으로 이전되지 않은 결과이며, 더 큰 모델만으로 해결된다는 근거는 없습니다.

## 소스와 라이선스

`THIRD_PARTY.md` 참고. 이 프로젝트의 원본 Python 코드는 MIT, PoSSM 데이터 및 그 파생 수치·그림·체크포인트·보고서는 보수적으로 **CC BY-NC-SA 3.0**으로 구분합니다. 상용 이용 가능 여부를 코드 라이선스로 판단하면 안 됩니다. 공식 PDF는 원 출처의 문서입니다.

## 제한

ASASIN과 complex 재현 차이, 단일 held-out 해저 장면, 무잡음/직선 항로 시뮬레이션, 단순 위상오차, 작은 CNN, 실해역 자료와 실제 기뢰 라벨 부재가 남아 있습니다. 본 결과를 실전 시스템 성능이나 최신 SOTA 성능으로 표현하지 않습니다. In-air 데이터는 이번 실행 결과에 포함하지 않았습니다. 예전 준비 문서의 필터 설명을 인용한다면 공식 코드의 high-pass stop8/pass10 kHz 조건으로 정정해야 합니다.
