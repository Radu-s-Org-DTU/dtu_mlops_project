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

---

### Checklist

### Week 1

* [x] Create a git repository (M5)
* [x] Make sure that all team members have write access to the GitHub repository (M5)
* [x] Create a dedicated environment for you project to keep track of your packages (M2)
* [x] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [?] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6) !!!(**Our script doesn't download automatically**)
* [x] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [x] Remember to fill out the `requirements.txt` and `requirements_dev.txt` file with whatever dependencies that you
    are using (M2+M6)
* [ ] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [?] Do a bit of code typing and remember to document essential parts of your code (M7) !!!(**Add python docs for important parts**)
* [x] Setup version control for your data or part of your data (M8)
* [x] Add command line interfaces and project commands to your code where it makes sense (M9)
* [ ] Construct one or multiple docker files for your code (M10)
* [ ] Build the docker files locally and make sure they work as intended (M10)
* [x] Write one or multiple configurations files for your experiments (M11)
* [x] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [ ] Use profiling to optimize your code (M12)
* [ ] Use logging to log important events in your code (M14)
* [ ] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [ ] Consider running a hyperparameter optimization sweep (M14)
* [x] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [ ] Write unit tests related to the data part of your code (M16)
* [ ] Write unit tests related to model construction and or model training (M16)
* [ ] Calculate the code coverage (M16)
* [ ] Get some continuous integration running on the GitHub repository (M17)
* [ ] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [ ] Add a linting step to your continuous integration (M17)
* [ ] Add pre-commit hooks to your version control setup (M18)
* [ ] Add a continues workflow that triggers when data changes (M19)
* [ ] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [ ] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [ ] Create a trigger workflow for automatically building your docker images (M21)
* [ ] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [ ] Create a FastAPI application that can do inference using your model (M22)
* [ ] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [ ] Write API tests for your application and setup continues integration for these (M24)
* [ ] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [ ] Create a frontend for your API (M26)

### Week 3

* [ ] Check how robust your model is towards data drifting (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [ ] Instrument your API with a couple of system metrics (M28)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Create an architectural diagram over your MLOps pipeline
* [ ] Make sure all group members have an understanding about all parts of the project
* [ ] Uploaded all your code to GitHub
