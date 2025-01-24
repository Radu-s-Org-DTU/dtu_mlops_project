## Documentation

Documentation for mushroomclassification

Aim:
- This project aims to classify mushrooms into 4 safety categories: Edible, Conditionally Edible, Poisonous and Deadly. We aim to build a pipeline for data preprocessing, model training, deployment and evaluation.

Problem Statement:
- There exists a multitude of mushrooms in the nature many of which are either toxic or harmful to humans. The main challenge in identifying the poisonous and non-poisonous mushrooms are the visual similarities between them. This project provides an automated solution to identify mushroom safety categories to mitiagte health risks and avoid any accidental posioning.

Key features:
- Data preprocessing using Albumentations
  * Images are transformed using Albumentations library where the transformations applied are re-sizing the images and rotating it.
- Model training with pyTorch Lightning
  *
- Deployment-ready API
  * API for real time interface built using FastAPI
  * Dockerized environment for seamless deployment on cloud platform GCP.
- Version control
   * Data versioning done using DVC
   * CI/CD workflows implemented using Github

Goals:
- Performning dvc and preprocessing for the data we use
- Training a CNN model which predicst the safety level of mushrooms
- build a pipeline which streamlines the task needed.

Dataset Structure:
- The dataset is sourced from Kaggle - https://www.kaggle.com/datasets/zedsden/mushroom-classification-dataset/data
- Raw data is stored in data/raw directory containing unprocessed images.
- The data is augmented and resized at runtime and is not stored. (So, Processed data folder is empty)
