# Mushroom Safety Classification
10.01.2025

## Team Members

- **Gokul Desu** (s242580)  
- **Satrio Muhammad** (s242591)  
- **William Buch Grønholm** (s113117)  
- **Radu Grecu** (s246415)

---

## Overview

This project focuses on creating a pipeline to classify mushrooms into four safety categories: **Edible**, **Conditionally Edible**, **Poisonous**, and **Deadly**. Using computer vision and deep learning, the goal is to develop an accurate and automated system that can handle diverse mushroom images.

The project combines data preparation, model training, and pipeline development to ensure reliable classification. Applications include food safety, foraging, and environmental research, making this system valuable in both practical and scientific contexts.

---

## Goals

1. **Data Quality**:  
   Clean and organize the dataset by addressing inconsistencies and standardizing formats. Create a curated data repository for efficient use.

2. **Model Training**:  
   Develop a Convolutional Neural Network (CNN) to classify mushrooms, with a focus on handling class imbalances effectively.

3. **Pipeline Development**:  
   Build an automated system that integrates preprocessing, model training, and inference. Ensure adaptability for future data updates.

---

## Data

The dataset, available on [Kaggle](https://www.kaggle.com/datasets/zedsden/mushroom-classification-dataset/data), is approximately **13.87 GB**. Since this exceeds the recommended 10 GB, strategic selection and resizing will be required. 

### Key Challenges:
- **Class Imbalance**:  
  Some categories, like "Deadly," have fewer than 50 images, while others exceed 1,500. Balancing these classes is crucial.

- **Image Standardization**:  
  Images vary in size and resolution, so resizing to a consistent format is necessary.

- **Selective Usage**:  
  Overrepresented categories may be trimmed to optimize computational resources.

---

## Framework

We will use [Albumentations](https://albumentations.ai/), a data augmentation library, to improve dataset diversity and model performance. 

### Augmentation Benefits:
- **Balancing Data**:  
  Enhance underrepresented categories by applying transformations like rotation, zoom, and color adjustments.

- **Improving Robustness**:  
  Simulate real-world conditions with lighting, noise, and perspective changes, helping the model generalize better.

This approach ensures a reliable, adaptable system for mushroom safety classification, with potential applications in public health and ecological research.

