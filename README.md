# KuaiRec
This report is a summary of my work on the KuaiRec project, which is a recommendation system for a video streaming platform. In this report, I will NOT discuss the data analysis of the different dataset that I used as they have their own Notebooks (Check the `./notebook/` folder for more informations on this).

Instead, I will focus on the recommendation system itself and the different models that I implemented and explain my choices. I will also discuss the evaluation metrics that I used to measure the performance of the models and the results that I obtained.


## Table of Contents
- [KuaiRec](#kuairec)
  - [Introduction](#introduction)
  - [Evaluation Metrics](#evaluation-metrics)
  - [Recommendation System](#recommendation-system)
    - [Collaborative Filtering](#collaborative-filtering)
    - [Content-Based Filtering](#content-based-filtering)
  - [Conclusion](#conclusion)

## Introduction
In 2022, the **University of Science and Technology of China** released a scientific article along with a unique dataset collected from the Chinese video-streaming platform **Kuaishou**, covering a period of two months. This dataset captures interactions between over **8,000 users** and more than **10,000 videos**.

A key feature of this dataset is its division into two interaction matrices (see figure below):

- **Large matrix**: A sparse matrix with a density of approximately **16.3%**, resembling real-world data where users interact with only a small fraction of items.
- **Small matrix**: A dense matrix with a density of around **99.6%**, providing a near-complete view of user preferences.

![kuaiRec](./data_final_project/KuaiRec%202.0/figs/KuaiRec.png)

---

In most recommendation systems, the most faithful method for evaluating a model is through **online testing**—deploying the model to real users and observing their behavior. However, such an approach is often **costly, time-consuming**, or simply **infeasible**.

The **KuaiRec** dataset provides an effective alternative by enabling offline model evaluation:

1. **Training phase**: Use the **sparse large matrix** to train the model. This simulates a real-world recommendation scenario with limited interaction data.
2. **Evaluation phase**: Test the model on the **dense small matrix**. This allows for a reliable estimation of model performance in an environment that closely approximates actual user behavior.

This setup offers a **practical and scalable evaluation framework** that mimics real-world deployment—without needing live user experiments.

## Evaluation Metrics

In this project, I employed the following evaluation metrics to assess the performance of the models:

**Note**: After conducting tests, I determined that an item should be recommended if the predicted `watch_ratio` exceeds **0.8**.

- **Error Metrics**:
  - **MAE**: Measures the average absolute error between the predicted `watch_ratio` and the actual.
  - **RMSE**: Measures the square root of the average squared error between the predicted `watch_ratio` and the actual.

- **Accuracy-Based Metrics**:
  - **Precision**: Measures the proportion of relevant items among the recommended items, focusing on items the user is likely to click on.
  - **Hit Rate**: Measures the proportion of relevant items that were recommended, considering the challenge of sparse data in the training set.
  - **MRR**: Measures the average rank of relevant items among the recommended items.
  - **NDCG**: Measures the relevance of recommended items, accounting for their position.

**Note**: For the Collaborative Filtering model, I utilized only MAE and RMSE metrics, as it predicts `watch_ratio`. Other models were evaluated using all metrics due to their focus on predicting item relevance.

## Recommendation System
### Collaborative Filtering

Collaborative filtering is a recommendation technique that relies on user-item interactions to make predictions. As our main dataset consist of user-item interactions, collaborative filtering is a natural choice for our first implementation that will serve as a baseline recommendation system.

An alternative least squares (ALS) algorithm has been implemented to learn the latent factors of users and items. As an ALS algorithm performs well with sparse data, so it is particularly suitable for our large matrix, with its low density.

The result after some hyperparameter tuning has the following results:
| Metric       | Value |
|--------------|-------|
|(RMSE)        | 0.5518|                              
|(MAE)         | 0.3170|                             
|Precision@20  | 0.2322|
|Hit Rate at 20| 1.0000|
|MRR@20        | 0.9634|
