# birdclef-2026-my-learnings

# BirdCLEF 2026  Wildlife Audio Detection

This project explores machine learning methods for detecting wildlife species from environmental audio recordings.


So What's my Goal is :
Develop models that identify species from audio soundscapes recorded in the **Pantanal wetlands**.(i also heard it first time , it is the largest tropical wetland on Earth,it is a massive natural floodplain in SA, mostly in Brazil but also spread in Paraguay and Bolivia)
<img width="2048" height="1370" alt="image" src="https://github.com/user-attachments/assets/6b5c45da-e5d8-483d-970d-e68ee251c225" />

---

## Problem

Biodiversity monitoring using manual listening is extremely slow and expensive.

Using machine learning, we can automatically detect species by analyzing their vocalizations.

---

## Approach

Pipeline:

Audio Recording
↓
Mel Spectrogram Generation
↓
CNN Model (ResNet18 baseline)
↓
Multi-label Classification (234 species)

---
( this is the initial approach i will try to learn other different models and try them in the pipeline )
## Dataset

Source: BirdCLEF 2026

Includes:

- 234 wildlife species
- audio recordings from Xeno-Canto and iNaturalist
- environmental soundscapes

---

## Model

Baseline architecture:

ResNet18 CNN

Loss Function:
Binary Cross Entropy (multi-label classification)

---

## Results

Baseline model trained on mel spectrograms.

Expected leaderboard score:
~0.60 ROC-AUC

Future improvements:

- EfficientNet audio models
- spectrogram augmentation
- pseudo-labeling
- ensemble models

---

## Learning Log

Daily progress is documented in the `learning-log` folder.

---

## Author

AKSHIT AGARWAL
