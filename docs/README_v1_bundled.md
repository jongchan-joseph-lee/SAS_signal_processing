# SAS Focus Benchmark v3 — autofocus and whole-image point detection

공개 합성 PoSSM의 남은 실험을 실행한 결과입니다. **수중 실측이나 실제 기뢰 탐지 성능이 아닙니다.** 학습 모델과 보정기가 기준에 실패하더라도 수치와 예측을 유지합니다. 1차 CNN을 다시 학습하지 않았습니다.

`report_v3/report.html`은 오프라인 보고서입니다. `report_v3/figures/SAS_Appendix_One_Page.png`는 부록 한 장 초안입니다. 원래 포트폴리오 PPT를 수정한 파일은 아닙니다.

## 실행

Python3.12 CPU, RAM 약10GB·여유 디스크8GB 이상 환경에서 실행했습니다. 한 번에 한 단계로 실행하면 메모리 경합이 줄어듭니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements_v3.txt
python src/run_v3.py --download
```

공개 취득은 저자 요청이 아닙니다. 다운로드한 55개 파일의 hash는 `references/download_manifest_v2.json`과 대조합니다. 이미 모든 자료를 받았다면 `--download`를 생략합니다. run_v3는 scene00/04 range-compression 캐시가 없을 때만 만듭니다. 원시자료·캐시·각 점 기여분 중간 파일은 ZIP에서 제외했습니다.

보고서만 재생성:

```bash
python src/run_v3.py --report-only
```

각 단계 실행 순서:

```bash
python src/protocol_v3.py
python src/evaluate_autofocus_v3.py --stage develop
python src/point_detection_v3.py --stage train
python src/evaluate_autofocus_v3.py --stage test
python src/point_detection_v3.py --stage test
python src/raw_detection_bridge_v3.py
python src/verify_v3.py
python src/build_report_v3.py --bundle
```

각 단계를 실제 실행했습니다. 통합 run_v3 명령을 데이터 취득부터 재차 전부 실행한 결과라는 뜻은 아닙니다. 브라우저 자동 렌더링 환경은 없으며 과학 그림 직접 확인, HTML 구조·JavaScript 구문·embedded ZIP 무결성·압축 해제 환경 보고서 재생성으로 검증합니다.

## 실험 무결성

- `results/v3/protocol.json`은 개발/시험 출력 전에 쓴 고정 조건입니다. 기존 조건과 다르면 덮어쓰지 않습니다. 외부 등록형 preregistration은 아닙니다.
- 보정: 개발점1/2/4, seed7001/7002; 시험 고립점24개, 새 seed7101–7105. 측면2mm RMS 오차와24점 무오차 보호 검사. success = 정합하지 않은 complex NCC 중앙값≥0.90 AND 무오차 최악 NCC 감소≤0.01. 24×6×3=432개 방법 결과.
- GPGA 계열은 원 논문 전체 Max-SDR 재현이 아닙니다. BP 기여분의 가장 강한 픽셀, 인접 핑 복소곱, 위상차 적분을 쓰는 국소 구현입니다. 핑별 entropy는 두 계수 다항식이 아닌 활동 핑별 위상 벡터를 최소화합니다. 두 추정기 모두 상수·선형 위상 gauge를 제거합니다.
- GT는 고립점 ROI 선택과 평가에만 씁니다. ROI 내부 위상 추정에는 넣지 않습니다. 이 구분 없이 전체 장면 자동 보정으로 부르면 안 됩니다.
- raw IQ는 고정하고 가정 항법을 바꿉니다. 물리적인 새 측정을 만들어낸 것이 아닙니다.
- common_phase_diagnostic은 같은24점이 자신 위상 추정에 포함되는 공통 위상 가정의 보조 진단입니다. 독립 시험/primary 선택 지표가 아닙니다.
- 탐지: train02/03/07/08, validation01/06, test04/05/09/10. 36/18/36배치. 같은 형상이 재사용되므로 독립 객체 설계90개가 아닙니다.
- 후보는 전체 유효 영상에서 만들어집니다. 최대600개, NMS0.65m, patch6m. RF에 절대 좌표·거리 정보는 넣지 않습니다. 영상 가장자리3m는 feature 문맥 때문에 검색에서 제외합니다. 시험36개는 모두 검색 영역 안입니다. 학습 두 장면의 각1개 GT가 가장자리 근처이며, GT jitter crop의 경계는 nearest padding으로 처리됩니다.
- primary point matching 반경1m, 검증 FP≤5/장면 조건으로 임계값 고정. 테스트를 본 뒤 바꾸지 않았습니다. 0.5/1.5m는 사전 보조 진단입니다. 중복 검출은FP입니다.
- CSV는 객체 삽입점·회전각이지 box/mask가 아닙니다. box mAP는 계산하지 않습니다. GT crop의 전경 조건부3분류를 탐지 성능으로 주장하지 않습니다.
- Wilson은 배치 독립성 가정, 별도 장면 bootstrap은 시험장면4개라는 한계가 있습니다. 두 구간을 같이 보고합니다.
- scene04 raw bridge는 공식 SLC 학습 모델의 자체 BP 입력 전환 시험입니다. 오류 영상에서 자동 후보30개를 골라 보정하므로 GT crop 실험과 구분합니다. 한 장면·한 오차 경로라 독립 일반화 근거가 아닙니다.

## 주요 산출물

- `results/v3/autofocus_selection.json`, `autofocus_results.json`: 고정 설정과 전체 보정 지표.
- `results/v3/detection_selection.json`: 학습36배치/검증18배치 선택.
- `results/v3/point_detector.joblib`: 실제 학습한 RF. 외부 사전학습 없음. 버전은 requirements_v3에 고정했습니다. 재현 코드가 hash를 검증합니다.
- `results/v3/detection_test_predictions.json`: 시험4장면의 전체 후보 좌표·점수·클래스 확률. 독립 FROC 재계산 가능.
- `results/v3/detection_results.json`: 곡선, 고정 운영점, 탐지 개수, 클래스·장면 결과, Wilson/cluster-bootstrap.
- `results/v3/raw_bridge_results.json`: raw→자체 BP→고정 탐지기 연결 시험.
- `results/v3/verification.json`, `environment.json`: 무결성·실행 환경.
- 결과 배열과 모든 그림. 원시자료 제외. v2 부록 재생성용 결과도 포함.

## 해석과 라이선스

기준 SLC와의 일치도는 영상 품질과 같지 않습니다. Hann 창은 주엽을 넓히고 부엽을 줄이며, scene00 전면 NCC0.50의 잔여 차이를 지우지 않습니다. v2 거리 진폭 보정의0.864는 사후 진단이므로 대표 개선 수치로 쓰지 않습니다. 이상적 고립점의 subpixel 수치를 실해역 μm 정확도로 표현하지 않습니다.

원본 프로그램 코드는 MIT. PoSSM 및 파생 결과·RF 가중치·그림·보고서는 CC BY-NC-SA3.0US로 구분합니다. 자료: Philtron, Johnson, Brown / MGDS. 상세 출처와 방법론은 THIRD_PARTY.md 및 보고서를 확인하십시오. 별도 대형 데이터와 외부 가중치는 이번 실험에 사용하지 않았습니다.
