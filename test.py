import glob
import os

import cv2 as cv
import torch

import drawing_detection
import preprocessing
import CNN_model

correct = 0
total = 0
errors = []

for label in range(10):
    for path in sorted(glob.glob(f"dataset_test/{label}/*.png")):
        image = cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB)

        # Same steps as main.py : binarize/invert + crop, then resize to 28x28 + normalize
        cropped_image, box = drawing_detection.locate_drawing(image)
        if cropped_image is None:
            errors.append((os.path.basename(path), label, "nothing detected"))
            total += 1
            continue
        final_image = preprocessing.preprocess(cropped_image)

        with torch.no_grad():
            prediction = CNN_model.model(final_image).argmax(1).item()

        total += 1
        if prediction == label:
            correct += 1
        else:
            errors.append((os.path.basename(path), label, prediction))

print(f"Accuracy : {correct}/{total} = {correct / total * 100:.2f}%")
for name, label, prediction in errors:
    print(f"  {name} : true = {label}, predicted = {prediction}")