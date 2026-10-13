---
title: PotholeAI
emoji: 🛣️
colorFrom: blue
colorTo: cyan
sdk: docker
app_port: 8501
---

# PotholeAI

Research prototype for pothole image segmentation and visualization.

## Current features
- Upload a road image
- Generate a segmentation mask and overlay
- Report segmented image-pixel fraction

## Model
U-Net with a ResNet-34 encoder, trained on Pothole440.

## Limitations
This prototype does not yet measure calibrated physical depth or volume,
validate reflection robustness, or track potholes across video frames.
Image-area fraction is not a physical severity measurement.
