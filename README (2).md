# Traffic Sign Recognition Using Deep Learning

A Convolutional Neural Network (CNN) built with Keras (TensorFlow backend) that classifies German traffic signs from the [GTSRB](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news) dataset. Traffic sign detection is a small but important building block of autonomous driving systems.

The model reaches **97.3%** test accuracy, improved to **98.4%** with data augmentation. Reported human accuracy on this dataset is 98.84%.

## Dataset

[German Traffic Sign Recognition Benchmark (GTSRB)](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news)

- ~39,000 training images and ~12,000 test images
- 43 traffic sign classes
- Images in `.ppm` format

Expected directory layout:

```
data/traffic_sign_dataset/
├── Final_Training/Images/<class_id>/*.ppm
├── Final_Test/Images/*.ppm
└── GT-final_test.csv
```

## Project Workflow

1. **Explore the data**: view sample images from the training set.
2. **Preprocess images**:
   - Histogram equalization on the V channel (in HSV space) to normalize exposure
   - Center crop to a square and resize to 48×48
   - One-hot encode the labels
3. **Build the CNN**:
   - 6 convolutional layers (32 → 32 → 64 → 64 → 128 → 128 filters, 3×3, ReLU)
   - Max pooling and dropout (0.2) after every two conv layers
   - Flatten, Dense(512, ReLU), Dropout(0.5)
   - Dense(43, Softmax) output
4. **Train**:
   - Loss: categorical cross-entropy
   - Optimizer: SGD (lr=0.01, momentum=0.9, Nesterov, decay=1e-6)
   - Batch size 32, up to 30 epochs, 80/20 train/validation split
   - Callbacks: `LearningRateScheduler` (10× decay every 10 epochs), `ModelCheckpoint` (keeps the best model), `EarlyStopping` (patience 5)
5. **Evaluate** on the official GTSRB test set.
6. **Improve with data augmentation** (shifts, zoom, shear, rotation) using `ImageDataGenerator`.

## Results

| Model                      | Test Accuracy |
|----------------------------|---------------|
| CNN                        | 97.30%        |
| CNN + Data Augmentation    | 98.45%        |
| Human performance (GTSRB)  | 98.84%        |

## Requirements

- Python 3
- numpy
- scikit-image
- matplotlib
- pandas
- scikit-learn
- h5py
- Keras with TensorFlow backend (the notebook uses the older standalone Keras API, e.g. `fit_generator` and `predict_classes`)

Install (example):

```bash
pip install numpy scikit-image matplotlib pandas scikit-learn h5py keras tensorflow
```

> **Note:** The code was written for older Keras/TensorFlow versions. With recent TensorFlow 2.x you may need small changes, e.g. `model.fit` instead of `fit_generator`, `np.argmax(model.predict(X), axis=1)` instead of `predict_classes`, and `val_accuracy` instead of `val_acc`.

## Usage

1. Download GTSRB and place it under `data/traffic_sign_dataset/` as shown above.
2. Launch Jupyter:
   ```bash
   jupyter notebook Traffic-Sign-Recognition.ipynb
   ```
3. Run all cells. Trained models are saved as `model.h5` (baseline) and `model_aug.h5` (with augmentation).

A GPU is strongly recommended; the notebook was trained on an NVIDIA Tesla K80.

## Configuration

Key constants at the top of the notebook:

| Parameter     | Value |
|---------------|-------|
| `NUM_CLASSES` | 43    |
| `IMG_SIZE`    | 48    |
| `BATCH_SIZE`  | 32    |
| `EPOCHS`      | 30    |

## References

1. [Keras Tutorial: Traffic Sign Recognition](https://chsasank.github.io/keras-tutorial.html)
2. [Man vs. Computer: Benchmarking Machine Learning Algorithms for Traffic Sign Recognition](http://image.diku.dk/igel/paper/MvCBMLAfTSR.pdf)
