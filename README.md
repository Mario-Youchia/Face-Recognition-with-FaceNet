# Face Recognition with FaceNet

A face-recognition and face-verification project using pretrained FaceNet embeddings.

![Face Recognition Confusion Matrix](assets/confusion-matrix.png)

## Overview

The dataset contains **111 face images across 11 identities**. 

## Results

| Metric | Result |
|---|---:|
| Training images | 88 |
| Held-out test images | 23 |
| Mean repeated cross-validation accuracy | 98.876% |
| Test accuracy | 100.000% |
| Correct test predictions | 23/23 |
| Verification pair accuracy | 96.443% |
| Verification balanced accuracy | 90.849% |

### Test Set Predictions

The classifier correctly identified all 23 images in the test set.

![Held-Out Prediction Gallery](assets/test-prediction-gallery.png)

### Face Verification

The verification threshold was selected using training-image pairs and not chosen manually.

![Verification Similarity Distribution](assets/verification-similarity-distribution.png)

## Workflow

1. Detect and prepare faces using OpenCV.
2. Generate embeddings using pretrained FaceNet.
3. Train a normalized linear SVM classifier.
4. Evaluate using repeated stratified cross-validation and a test set.
5. Verify whether two images contain the same person using cosine similarity.

## Notebook

[View the executed Kaggle notebook](https://www.kaggle.com/code/marioyouchia/face-recognition-with-facenet)

## Limitations

The dataset is very small, so the results do not represent universal real-world performance.
