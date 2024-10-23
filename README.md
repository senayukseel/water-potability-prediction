# Water Potability Prediction

This project involves the application of various machine learning models to predict water potability using a dataset that includes various chemical and physical properties of water. The dataset is sourced from **Kaggle** and contains measurements that help determine whether water is potable (drinkable) or not. The models used in this project include **Naive Bayes**, **Decision Trees**, **KNN (K-Nearest Neighbors)**, **SVM (Support Vector Machine)**, and **ANN (Artificial Neural Network)**.

**This project includes a MATLAB application where various machine learning implementations are performed using the Water Potability dataset. The dataset, which measures different chemical and physical properties to determine water quality, is used to create machine learning models for predicting water potability.**

---

## Dataset

The dataset used in this project is the **Water Potability** dataset from Kaggle. It contains several features that measure the quality of water:

- **pH**: The pH level of water.
- **Hardness**: The amount of calcium and magnesium salts in water.
- **Solids**: Total dissolved solids in water (in ppm).
- **Chloramines**: The amount of chloramines in water (in ppm).
- **Sulfate**: The amount of sulfate in water (in ppm).
- **Conductivity**: Electrical conductivity of water (in μS/cm).
- **Organic Carbon**: The amount of organic carbon in water (in ppm).
- **Trihalomethanes**: The amount of trihalomethanes in water (in ppm).
- **Turbidity**: Measure of water clarity (in NTU).
- **Potability**: Binary target label (0: Not Potable, 1: Potable).

You can download the dataset from Kaggle [here](https://www.kaggle.com/datasets/gauravduttakiit/water-potability-prediction).

Additionally, the data was first balanced by equalizing the number of samples for each class (0 and 1). Afterward, the dataset underwent preprocessing and was split into training and testing sets. These datasets are also included in the project.

---

## Models

Several machine learning models have been implemented and tested to predict the potability of water:

- **Naive Bayes**: A probabilistic classifier that applies Bayes' theorem with strong independence assumptions.
  
- **Decision Trees**: A tree-structured classifier that splits the dataset into branches based on feature importance and thresholds.

- **K-Nearest Neighbors (KNN)**: A classification algorithm that assigns labels based on the majority vote of the k-nearest points in the feature space.

- **Support Vector Machine (SVM)**: A supervised learning model that finds the optimal hyperplane for classification in higher dimensions.

- **Artificial Neural Network (ANN)**: A fully connected deep learning model with multiple layers for learning non-linear relationships in the dataset.

---

## Key Insights

This project provided valuable insights into how various machine learning models, such as **Naive Bayes**, **Decision Trees**, **KNN**, **SVM**, and **ANN**, can be applied to water potability prediction. By evaluating chemical and physical attributes, these models help determine the safety of water for consumption.

---
