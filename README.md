# Parkinson's Voice Biomarker ML

A machine learning-based biomedical data science project for exploring Parkinson's disease classification using voice measurement features.

## Project Overview

Parkinson's disease is a neurodegenerative disorder that can affect motor control, speech, and voice patterns. Biomedical voice measurements have been studied as potential non-invasive indicators for Parkinson's disease classification.

This project uses a publicly available dataset from the UCI Machine Learning Repository to develop a structured machine learning workflow for Parkinson's disease classification.

## Objectives

- Use real biomedical voice measurement data.
- Explore voice-based features associated with Parkinson's disease.
- Train machine learning models for classification.
- Evaluate model performance using standard metrics.
- Present results using tables and visualizations.

## Repository Structure

```text
parkinsons-voice-biomarker-ml/

README.md
requirements.txt
main.py

data/
    README.md
    raw/
    processed/

notebooks/
    parkinsons_voice_analysis.ipynb

src/
    data_loader.py
    model_training.py
    visualization.py

results/
    model_metrics.csv

figures/
    feature_importance.png
    confusion_matrix.png
