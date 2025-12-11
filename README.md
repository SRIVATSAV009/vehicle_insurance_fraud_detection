# **Vehicle Insurance Fraud Detection Using Machine Learning**

A comprehensive end-to-end machine learning project designed to identify and prevent fraudulent vehicle insurance claims using advanced data analytics, feature engineering, and predictive modeling. This solution demonstrates how insurance companies can reduce financial losses, improve claim verification efficiency, and detect suspicious claim patterns early in the process.

---

## **Project Overview**

Insurance fraud is one of the major challenges in the insurance industry, resulting in millions in financial losses every year. This project develops a **fraud detection pipeline** that analyzes claim records, customer profiles, vehicle information, and incident details to classify claims as **fraudulent or legitimate**.

The system applies **supervised machine learning algorithms** such as Logistic Regression, Random Forest, XGBoost, and Gradient Boosting to detect anomalies and predict the likelihood of fraud with high accuracy.

## **Key Features**

* **End-to-end ML workflow** including data ingestion, preprocessing, balancing, modeling, and evaluation
* **Exploratory Data Analysis (EDA)** to uncover hidden patterns, anomalies, and high-risk claim behaviors
* **Feature Engineering** using domain-specific attributes such as repair cost ratios, prior claims, incident severity, and vehicle age
* **Model comparison** across multiple algorithms to identify the best-performing fraud classifier
* **Hyperparameter tuning** for improved model precision and recall
* **Fraud probability scoring** to help insurers prioritize suspicious claims
* **Visual dashboards** showing fraud distribution, correlations, and model insights
* **Deployment-ready structure**, can be extended into real-time scoring with FastAPI/Flask

## **Tech Stack**

* **Python** (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)
* **Machine Learning** (Logistic Regression, Random Forest, XGBoost, Gradient Boosting, SVM)
* **Data Processing** (EDA, outlier detection, SMOTE balancing, scaling)
* **Model Evaluation** (Confusion Matrix, AUC-ROC, Precision-Recall, F1 Score)
* **Version Control**: Git & GitHub

## **Model Performance**

The optimized model achieved:

* **Accuracy:** ~92–95%
* **Precision (Fraud Class):** High precision to reduce false alarms
* **AUC Score:** Strong classifier performance for fraud vs. non-fraud
