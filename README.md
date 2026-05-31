# 야간광 × 고령화 × 아파트 실거래 — 저광도 지역 데이터 격차와 가격 예측 ML

위성 야간광(VIIRS) · 시군구 고령화율 · 전국 아파트 실거래가(239만 건)를 결합해,
**불 꺼진 고령화 지역(저광도 사각지대)**의 데이터 희소성과 가격 예측 모델의 오차 구조를 검증한 프로젝트.

> 2026-1학기 데이터마이닝 기말 프로젝트 · 고경탁, 이성수, 허윤

---

## 데이터 출처

| 데이터 | 출처 | 기간 / 비고 |
|--------|------|-------------|
| 위성 야간광 (VIIRS DNB) | Google Earth Engine (`code.earthengine.google.com`) | 2019–2023, `.tif` 5개 |
| 행정구역 경계 (읍면동) | 국토교통부 V-WORLD shapefile (EPSG:5186) | 5,370개 행정구역 |
| 시군구 고령화율 | KOSIS 국가통계포털 — 주민등록 연앙인구(시군구·성·연령) | 2019–2023, 65세↑ 비율 |
| 아파트 실거래가 | 공공데이터포털 — 국토부 아파트 매매 실거래 Open API | 2019–2023, 취소거래 제외 후 239만 건 |
| 시군구 행정표준코드 | 행정안전부 행정표준코드관리시스템 (`KiKcd_B`) | 법정동 ↔ KOSIS 코드 매핑 |

> 원천·중간 데이터(VIIRS `.tif`, shapefile, 실거래 raw, `apt_ml_master.csv` 739MB, `apt_ml_ready.csv` 359MB)는
> 용량/배포 문제로 저장소에 포함하지 않음. 위 출처에서 재수집 가능하며, 해당 CSV는 6·8번 노트북 실행 시 생성됨.

---

## 폴더 구조

```
1_data_processing/        데이터 수집·통합
├── 1_collection/         0 야간광 · 1 고령화율 · 5 실거래가 수집
├── 2_mapping/            2 법정동 ↔ KOSIS 코드 매핑
├── 3_clustering/         3 K-Means 광도 3분류(고/중/저)
└── 4_correlation/        4 야간광 × 고령화율 상관 (Act 1)
2_preprocessing/          전처리
├── 1_pre_eda/            6 클린징·병합 → apt_ml_master
└── 2_post_eda/           8 EDA 반영 정리 → apt_ml_ready
3_eda/                    7 결측·시각화·저광도 파이프라인 (Act 2)
4_modeling/               9 Baseline · 10 Full · 11 특화 (Act 3)
```

## 실행 순서 (파일 번호 = 실행 순서)

폴더는 단계별 분류이고, **실제 실행은 파일 번호 0 → 11 순서**.

| # | 노트북 | 단계 | 입력 → 출력 |
|---|--------|------|-------------|
| 0 | `0_viirs_emd_nightlight` | 수집 | VIIRS `.tif` + shp → `viirs_emd.csv` |
| 1 | `1_aging_rate` | 수집 | KOSIS 원본 → `aging_rate.csv` |
| 2 | `2_sgg_code_mapping` | 매핑 | 행안부 코드 → `legal_to_kosis_sgg.csv` |
| 3 | `3_kmeans_clustering` | 군집 | 야간광 + 코드 → `kmeans_sgg_cluster.csv` |
| 4 | `4_merge_analysis` | 상관 | 고령화율 + 군집 → `cluster_aging_merged.csv` |
| 5 | `5_apt_data_download` | 수집 | 코드·병합 결과 → 실거래가 raw |
| 6 | `6_preprocessing_apt` | 전처리(전) | raw + 야간광·고령화·군집 → `apt_ml_master.csv` |
| 7 | `7_EDA_1_1 · _2 · _3` | EDA | 결측·시각화·피처·저광도 파이프라인 |
| 8 | `8_preprocessing` | 전처리(후) | EDA 반영 → `apt_ml_ready.csv` |
| 9 | `9_baseline` | 모델 | 건물 피처 8개 (기준선) |
| 10 | `10_fullbaseline` | 모델 | + 야간광·고령화 14개 |
| 11 | `11_특화모델` | 모델 | 저광도 특화 13개 |

> `5_apt_data_download`는 성격상 "수집"이지만, 다운로드 대상 시군구 목록을 2·4번 출력에서 받아오므로 매핑·상관 뒤에 실행됨.

---

## 결과 요약

| 모델 | 대표 알고리즘 | MAE (만원) | 비고 |
|------|--------------|-----------|------|
| Baseline (피처 8개) | Bagging | **5,045** | R²(log) 0.937 |
| Full (피처 14개) | Bagging | **4,461** | R²(log) 0.951 — 야간광·고령화 추가로 개선 |
| 특화 (저광도, 피처 13개) | Bagging | **2,706** | 저광도 test 기준 |

**핵심 발견**
- **Act 1** — 저광도 지역 고령화율 26.95% (고·중광도 ~18%), 야간광 ↔ 고령화율 r = −0.58 → "어두울수록 고령화"
- **Act 2** — 시군구당 거래 중앙값: 고광도 9,211건 vs 저광도 3,597건 (39% 수준) → 저광도 데이터 희소
- **Act 3** — Full 모델 MAPE: 저광도 7.6% / 중광도 7.7% / 고광도 8.3% → 세 그룹 거의 균등 → 모델이 저광도를 불공정하게 다루지 않음 (가설 수정)

---

## 환경
Python 3 · pandas · scikit-learn · geopandas · matplotlib · xgboost · lightgbm
