# Title (Please modify the title)
## Team

| ![박패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이봉균](https://avatars.githubusercontent.com/u/1223020?v=4) | ![최패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![오패캠](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: |:-----------------------------------------------------------:| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [박패캠](https://github.com/UpstageAILab)             |           [이패캠](https://github.com/UpstageAILab)            |            [최패캠](https://github.com/UpstageAILab)             |            [김패캠](https://github.com/UpstageAILab)             |            [오패캠](https://github.com/UpstageAILab)             |
|                            팀장, 담당 역할                             |                         git                          |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |

## 0. Overview
### Environment
- _Write Development environment_

### Requirements
- _Write Requirements_

## 1. Competiton Info

### Overview

- _Write competition information_

### Timeline

- ex) January 10, 2024 - Start Date
- ex) February 10, 2024 - Final submission deadline

## 2. Components

### Directory

- _Insert your directory structure_
```sh 
.
├── Pipfile
├── Pipfile.lock
├── README.md
├── code                            # ipynb
│   ├── XGBoost.ipynb
│   ├── baseline_code.ipynb
│   ├── baseline_code_LGBM.ipynb
│   ├── deptno                            
│   │   ├── load_log_model.py       # 저장된 pkl 로 모델을 불러와 test.csv 로 부터 예측
│   │   ├── run.py                  # optuna 를 통해 LGBRegressor 하이퍼 파라메터를 찾고 가장 좋은 모델, 파라메터를 기록
│   │   ├── run_with_log.py         # 모델 학습시 target 에  log scale 을 적용한다
│   │   ├── train-vs-test-xy.ipynb
│   │   └── train-vs-test.ipynb
│   ├── deptno.ipynb
│   └── requirements.txt
└── data                            # 원본 데이터, 및 전처리를 거진 데이터
    ├── X_test_6.csv
    ├── X_train.5.2.csv
    ├── bus_feature.csv
    ├── sample_submission.csv
    ├── selected_xy.csv
    ├── subway_feature.csv
    ├── test.csv
    ├── test_xy.csv
    ├── train.csv
    ├── train_xy.csv
    └── y_train.5.2.csv
```

## 3. Data descrption

### Dataset overview

- _Explain using data_

### EDA

- _Describe your EDA process and step-by-step conclusion_

### Data Processing

- _Describe data processing process (e.g. Data Labeling, Data Cleaning..)_

## 4. Modeling

### Model descrition

- _Write model information and why your select this model_

### Modeling Process

- _Write model train and test process with capture_

## 5. Result

### Leader Board

- _Insert Leader Board Capture_
- _Write rank and score_

### Presentation

- _Insert your presentaion file(pdf) link_

## etc

### Meeting Log

- _Insert your meeting log link like Notion or Google Docs_

### Reference

- _Insert related reference_
