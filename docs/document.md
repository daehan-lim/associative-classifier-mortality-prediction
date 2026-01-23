# eICU 데이터 전처리 및 사용한 CSV 설명

본 문서는 연관 분류기를 이용한 사망률 예측 프로젝트에서 eICU Collaborative Research Database(eICU-CRD)로부터 어떤 방식으로 데이터를 선택/가공했는지, 그리고 어떤 CSV 파일을 사용했는지를 정리한 문서입니다.

## 1) 원천 데이터(eICU-CRD)

- 원천 데이터: eICU Collaborative Research Database (PhysioNet)
- 본 프로젝트의 “원천 테이블”은 eICU-CRD에서 제공되는 CSV들(예: `patient`, `medication` 등)입니다.

## 2) 예측 문제 정의(라벨)

- 목표: ICU 입원 동안 투여된 약물 정보를 이용하여 **퇴원 시 사망/생존**(binary)을 예측
- 라벨(클래스):
  - `0`: 생존(Alive)
  - `1`: 사망(Expired)
- 라벨은 eICU의 `patient` 테이블의 퇴원 상태(예: `unitdischargestatus`)를 기반으로 생성했습니다.

## 3) 샘플(환자) 선택 기준

전처리 과정에서 공통적으로 적용한 핵심 조건은 다음과 같습니다.

1. **약물 기록이 있는 ICU stay만 포함**
   - `medication.drugname IS NOT NULL` 인 기록이 존재하는 `patientunitstayid` 중심으로 구성
2. **특정 병원(hospital) 단위로 데이터 구성 가능**
   - eICU는 병원별로 기록 양과 약물 코딩이 달라서, 실험을 위해 병원 단위 데이터셋을 만들었습니다.
   - 예: `dataset_264.csv`는 `hospitalid = 264`에서 구성한 데이터셋입니다.

추가로, 표본 수가 충분한 병원을 찾기 위해 병원별 ICU stay 수를 세고(예: 800건 초과) 후보 병원을 선정하는 쿼리를 사용했습니다.

- 관련 쿼리 예시: `data/eICU/queries/hospitals_w_more_t_800_p.sql`

## 4) 특징(Feature) 생성: 약물 기반 벡터화

한 환자(정확히는 `patientunitstayid`)를 **하나의 샘플(row)** 로 만들고, 약물 투여 여부를 특징으로 만들었습니다.

- 각 특징(feature)은 특정 약물명(또는 정규화된 약물명)을 의미합니다.
- 환자가 ICU stay 동안 해당 약물을 한 번이라도 투여 받으면 `1`, 아니면 `0`으로 표시합니다.

### 4.1) “147개 약물 특징” 데이터셋(논문/주요 실험에 사용)

주요 실험(연관 분류기)은 약물 특징을 너무 많이 쓰면 차원이 커지고 희소성이 커져서, 비교적 관리 가능한 개수의 약물을 선택해 **147개 약물 특징**으로 구성했습니다.

- 약물 목록(147개)은 `data/eICU/csv tables/drugnames_big_h.csv`에 저장되어 있습니다(줄 수 147).
- 이 147개 약물만을 기준으로 0/1 행렬을 만들고, 마지막 열에 class(0/1)를 붙였습니다.

## 5) 실제로 사용한 CSV 파일 형식

같은 데이터를 두 가지 형식으로 저장해 두었습니다.

### (A) `dataset_binary.csv` (추천: 일반 ML 모델 입력용)

- 경로: `data/results/dataset_binary.csv`
- 형식: “헤더가 있는 0/1 행렬”
  - 열: 147개 약물 feature + 마지막 열 `class`
  - 행: 환자(총 4615명)
- 장점: scikit-learn MPL classifier로 바로 사용 가능

### (B) `dataset.csv` (연관분류기/트랜잭션 형태)

- 경로: `data/results/dataset.csv`
- 형식: “각 행이 아이템(약물명)들의 리스트 + 마지막에 class(0/1)”
  - 0인 특징들은 빈 칸으로 두고, 1인 약물명만 행에 등장하는 형태(트랜잭션 형태)
- 이 형식은 연관규칙/연관분류기에서 아이템셋을 다루기 쉬워서 사용했습니다.

## 6) 병원별 데이터셋

병원별로 따로 만든 데이터도 있으며, 파일명에 병원 ID가 들어갑니다.

- 예시(병원별 트랜잭션/바이너리):
  - `data/results/dataset_264.csv`
  - `data/results/dataset_binary_264.csv`
  - `data/results/dataset_167.csv`, `data/results/dataset_binary_167.csv`
  - `data/results/dataset_458.csv`, `data/results/dataset_binary_458.csv`

병원별 데이터는 병원마다 약물 코드/분포가 달라서, “한 병원만 사용할 때” 혹은 “병원 간 일반화 실험”을 할 때 유용합니다.

## 7) MLP 베이스라인용 데이터(참고)

연관 분류기 외에, scikit-learn 기반 MLP 베이스라인 실험을 위해 만든 파일들이 `data/mlp/`에 있습니다.

- 예: `training_set_binary.csv`, `test_set_binary.csv` 및 정규화/표준화 버전들
- 이 폴더의 데이터는 특징 수가 더 크고(약물 목록이 큼), 전처리 변형(norm/stand 등)이 여러 개라서 “현재 프로젝트 목적”에 따라 선택해서 사용하면 됩니다.

## 8) 전처리 SQL/스크립트 위치(재현용)

원천 eICU CSV를 DB에 적재하고(patient/medication 등), 환자 선택 및 특징 생성에 사용한 SQL들은 아래에 있습니다.

- SQL 모음: `data/eICU/queries/`
  - 병원 후보 선정: `hospitals_w_more_t_800_p.sql`
  - 환자 ID 필터링(병원별): `filtered_patient_ids*.sql`
  - 약물 특징 뷰/테이블 생성: `final_dataset_view.sql`, `final_dataset_for_csv.sql`, `final_all_view_mlp.sql` 등

## 9) 파일 권장 목록

바로 실험 가능한 형태 기준으로는 아래 두 파일을 우선 추천

1. `data/results/dataset_binary.csv` (가장 범용적)
2. `data/results/dataset.csv` (연관규칙/연관분류기용)

추가로 병원 단위 분석이 필요하면 `dataset_*.csv`, `dataset_binary_*.csv` 중 원하는 병원 ID의 파일을 함께 사용하시면 됩니다.