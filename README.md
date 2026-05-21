# Pneumonia Detection from Chest X-Rays (CNN)

A convolutional neural network that classifies chest X-ray images as normal or
pneumonia, built in PyTorch and evaluated against a logistic-regression
baseline on the Kaggle chest X-ray dataset.

## Summary

A simplified CNN learns visual features directly from the X-ray pixels and is
compared with a logistic-regression baseline that treats each pixel
independently. The CNN substantially outperforms the baseline, driven by very
high recall, it catches almost all pneumonia cases. The honest tradeoff is a
high false-positive rate: the model is biased towards predicting pneumonia,
which the analysis attributes to class imbalance in the dataset and overfitting
visible in the training-versus-validation accuracy curve.

In a clinical-support framing this profile is deliberate: missing a pneumonia
case is costlier than a false alarm, so high recall matters, but the model is
not reliable enough for unaided clinical use without further work.

All metrics (accuracy, precision, recall, F1, confusion matrix) are computed on
the test set at runtime and printed at the end of the script. The full report
is available on request.

## Model

- Three convolutional blocks (Conv2d -> ReLU -> MaxPool), then two fully
  connected layers, output over two classes.
- Input resized to 224x224; training uses random horizontal flip and small
  rotation for light augmentation.
- Cross-entropy loss, Adam optimiser (lr 0.001), batch size 32, 5 epochs.
- Evaluation: accuracy, precision, recall, F1 and a confusion matrix on the
  test split.


Download the dataset (see below) and place it in the project root as
`chest_xray/`, then run:

```bash
python CNN.py
```

This prints per-epoch training and validation accuracy, then the test-set
metrics and confusion matrix, and saves the accuracy curve as a PNG.

## Data

The chest X-ray images are not included in this repository. Download from:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Place the extracted dataset in the project root so the structure is:
```bash
chest_xray/
├── train/   (NORMAL/, PNEUMONIA/)
├── val/     (NORMAL/, PNEUMONIA/)
└── test/    (NORMAL/, PNEUMONIA/)
```

The dataset's `val` split has only a handful of images, so validation accuracy
is noisy across epochs. The test split is the meaningful evaluation set.

## Limitations and future work

- False positives: the model over-predicts pneumonia; reducing this is the main
  improvement target.
- Overfitting: small dataset and light augmentation lead to a widening
  train/validation gap. Stronger augmentation or more data would help.
- Architecture: ResNet or DenseNet are the natural next step beyond this
  simplified CNN.
