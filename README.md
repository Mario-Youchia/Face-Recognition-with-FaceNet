# Face Recognition with FaceNet

A face-recognition and face-verification project using a pretrained FaceNet model to generate face embeddings, a linear SVM for identity classification, and cosine similarity for verification.

## Project Overview

The dataset contains **111 images across 11 identities**. Face images are detected and prepared with OpenCV, converted into 128-dimensional FaceNet embeddings, and then used to train and evaluate the classifier.

## Results

| Metric | Result |
|---|---:|
| Training images | 88 |
| Held-out test images | 23 |
| Mean repeated cross-validation accuracy | 98.876% |
| Test accuracy | 100.000% |
| Correct test predictions | 23/23 |
| Verification balanced accuracy | 90.849% |
| Verification pair accuracy | 96.443% |

The verification threshold was selected from training-image pairs and evaluated separately on held-out test pairs.

## Workflow

1. Load and audit the image dataset.
2. Detect and resize faces with OpenCV.
3. Generate embeddings using pretrained FaceNet.
4. Evaluate a normalized linear SVM with repeated stratified cross-validation.
5. Train the final classifier and evaluate it on the held-out test set.
6. Perform face verification using cosine similarity.

## Technologies

Python, TensorFlow, Keras, FaceNet, OpenCV, scikit-learn, NumPy, and Matplotlib.

## Run the Notebook

The notebook is designed to run on Kaggle with the image dataset and `facenet_keras.h5` model attached as inputs.

- [View the Kaggle notebook](https://www.kaggle.com/code/marioyouchia/face-recognition-with-facenet)

## Limitations

The dataset is small and curated, so the results should not be interpreted as universal real-world performance. The classifier is closed-set and always predicts one of the known identities. More varied images and unknown-person evaluation would be required for production use.
