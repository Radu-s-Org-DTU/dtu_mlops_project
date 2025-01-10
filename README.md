# dtu_mlops_project
Mushroom Safety Classification
10.01.2025

**Team members:**

Gokul Desu (s242580)
Satrio Muhammad (s242591)
William Buch Grønholm (s113117)
Radu Grecu (s246415)


**Overview**

The project consists of creating a pipeline that can classify pictures of mushrooms according to the risk of consuming them - Edible, Conditionally Edible, Deadly and Poisonous.

**Goals**

The goal of the project is to apply the knowledge acquired throughout the course to:
perform a data sanctity check on the mentioned dataset and create a new data repo
train a CNN model which predicts the safety level of the mushrooms
build a pipeline which streamlines the task aforementioned and is automated to keep running

**Data**

Mushroom picture dataset @ [link](https://www.kaggle.com/datasets/zedsden/mushroom-classification-dataset/data)
The size is a bit larger at 13.87 GB than the recommended 10 GB so we might omit some pictures of the species that have a very large count.

**Framework**

Albumentations @ [link](https://albumentations.ai/) is a library used for data augmentation that can do basic to advanced tasks, covering nearly all common augmentation needs.

It will be necessary to augment the species of mushrooms that have a very low picture count. The Deadly classified mushrooms don’t even go past 50 pictures while some other species go as high as 1500.


