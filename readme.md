# 🚗 License Plate Detection using Faster R-CNN

A deep learning–based license plate detection system built using **PyTorch** and **Faster R-CNN**. Trained on a custom dataset with Pascal VOC-style `.xml` annotations to localize vehicle license plates in real-world images.

---

## 🔍 Project Overview

This project performs:
- Object detection for license plates in images
- Fine-tuning of a pretrained Faster R-CNN model
- OCR-ready bounding boxes as outputs
- Text extraction through Pytesseract

---

## 📁 Dataset

- Custom dataset in Pascal VOC XML format
- Contains labeled images of vehicles with license plates
- Training pipeline loads images and targets using a custom `Dataset` class

📸 Example Training Sample Set:

<p align="center">
  <img src="assets/viz.png" width="500"/>
</p>

## 📉 Loss and 📈 Accuracy Curves

<p align="center">
  <img src="assets/loss.png" width="500"/>
</p>

## 🔮 Sample Prediction on Test Set


<p align="center">
  <img src="assets/pred3.png" width="500"/>
</p>

---
