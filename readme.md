# AI-Based Live Fall Detection GCN

A simple skeleton-based Graph Convolutional Network (GCN) system for real-time fall detection.  
Designed as a baseline/demo for researchers and university students.

## Features

-   Preprocesses the Le2i “Fall Dataset” into CSV skeleton files
-   Two GCN training pipelines:
    -   **Train_GCN.ipynb** – lighter, faster baseline
    -   **Train_GCN2.ipynb** – more complex, higher-accuracy model
-   Live inference demos via Jupyter notebooks or Python scripts
-   Simple HTTP API endpoint for external fall-alert integration

## HOW TO USE

1. **Download the Dataset**
   Visit
   https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia
   and download the “Le2i Fall Dataset” (you’ll receive an `archive.zip`).

2. **Place the ZIP**
   Put that zip file inside a folder named : **Put dataset zip here** at the project root

3. **Generate Skeleton CSVs**
   Open and run `action_class/Make_dataset.ipynb`.
   This notebook will:
    - Unzip `archive.zip`
    - Split videos into clips
    - Extract skeleton keypoints into CSVs (train/val folders)

---

## Training the Models

Choose one of the two notebooks in `action_class/`:

-   **Train_GCN.ipynb**

    -   A lightweight ST-GCN baseline
    -   Faster to train, lower GPU/memory footprint

-   **Train_GCN2.ipynb**
    -   An enhanced, deeper ST-GCN model
    -   Higher accuracy, heavier resource usage

Steps:

1. Open your chosen notebook.
2. Modify any data paths if necessary.
3. Run all cells.
4. Trained model weights (`.pth`) will be saved under `action_class/AI_Train/Models/`.

---

## Live Inference

### Via Jupyter

-   `action_class/AI_Use/Live_GCN1.ipynb` – lightweight demo
-   `action_class/AI_Use/Live_GCN2.ipynb` – heavier, more accurate demo

This script:

-   Opens your webcam
-   Performs skeleton extraction + GCN inference
-   Prints “Fall detected” or “No fall” in the console

---

## Notes & Tips

-   For a quick start:
    1. Prepare data → 2. Run **Train_GCN.ipynb** → 3. Launch **Live_GCN1.ipynb**.
-   If you run into GPU memory errors, stick with the lighter notebooks (`GCN1`).
-   Ensure your OpenCV build matches your Python version (e.g. `opencv-python`).

# AI 기반 라이브 낙상 감지 GCN

실시간 낙상 감지를 위한 간단한 스켈레톤 기반 그래프 합성곱 네트워크(GCN) 시스템입니다.  
연구자 및 대학생을 위한 기본/데모용으로 설계되었습니다.

## 특징

-   Le2i “Fall Dataset”을 CSV 스켈레톤 파일로 전처리
-   두 가지 GCN 학습 파이프라인
    -   **Train_GCN.ipynb** – 경량, 빠른 베이스라인
    -   **Train_GCN2.ipynb** – 더 복잡하고 높은 정확도의 모델
-   Jupyter 노트북 또는 Python 스크립트를 통한 실시간 추론 데모
-   외부 낙상 알림 연동을 위한 간단한 HTTP API 엔드포인트

## 사용 방법

1. **데이터셋 다운로드**  
   https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia  
   위 링크에서 “Le2i Fall Dataset”을 다운로드하면 `archive.zip` 파일을 얻게 됩니다.

2. **ZIP 파일 배치**  
   프로젝트 루트에 `Put dataset zip here` 폴더를 만들고, 그 안에 `archive.zip`을 넣으세요.

3. **스켈레톤 CSV 생성**  
   `action_class/Make_dataset.ipynb` 노트북을 열어 셀을 실행하면:
    - `archive.zip` 압축 해제
    - 동영상을 클립별로 분할
    - 스켈레톤 키포인트를 train/val 폴더에 CSV 파일로 저장

---

## 모델 학습

`action_class/` 폴더 내 두 개의 노트북 중 하나를 선택하세요:

-   **Train_GCN.ipynb**

    -   경량 ST-GCN 베이스라인
    -   빠른 학습, 낮은 GPU/메모리 사용량

-   **Train_GCN2.ipynb**
    -   심화된 딥 ST-GCN 모델
    -   더 높은 정확도, 더 많은 리소스 사용

학습 절차:

1. 원하는 노트북을 엽니다.
2. 필요 시 데이터 경로를 수정합니다.
3. 모든 셀을 실행합니다.
4. 학습된 모델 가중치(`.pth`)가 `action_class/AI_Train/Models/`에 저장됩니다.

---

## 실시간 추론

### Jupyter에서

-   `action_class/AI_Use/Live_GCN1.ipynb` – 경량 데모
-   `action_class/AI_Use/Live_GCN2.ipynb` – 고정밀 데모

### Python 스크립트 실행

```

python action_class/AI_Use/Live_GCN1.py

```

이 스크립트는:

-   웹캠을 켜고
-   스켈레톤 추출 및 GCN 추론을 수행한 뒤
-   콘솔에 “Fall detected” 또는 “No fall” 메시지를 출력합니다

---

## 참고 및 팁

-   빠른 시작:
    1. 데이터 준비 →
    2. **Train_GCN.ipynb** 실행 →
    3. **Live_GCN1.ipynb** 실행
-   GPU 메모리 부족 오류 발생 시 `GCN1`(경량) 노트북 사용을 권장합니다.
-   OpenCV 설치 버전이 Python 버전에 맞는지 확인하세요 (예: `opencv-python`).
