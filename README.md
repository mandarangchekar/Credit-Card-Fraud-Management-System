# Credit Card Fraud Management System

## Overview

Welcome to the Credit Card Fraud Management System repository! This project utilizes a variety of machine learning techniques to detect and manage credit card fraud efficiently. It is designed to handle a large dataset of credit card transactions, applying data preprocessing, exploratory data analysis, and predictive modeling to identify fraudulent activities.

## Features

- **Data Handling:** Robust loading and displaying of large-scale datasets.
- **Data Analysis:** Comprehensive exploratory data analysis with statistical summaries and correlation matrices.
- **Preprocessing:** Implements preprocessing steps like scaling and encoding to prepare data for modeling.
- **Model Training:** Utilizes multiple machine learning models for accurate fraud detection.
- **Model Evaluation:** Detailed evaluation metrics such as accuracy, precision, recall, and F1-score to assess model performance.
- **Data Serialization:** Features for saving and loading model configurations and results for future use.

## Repository Structure

This project's repository is organized into directories and files that are structured for easy navigation and use. Here’s a detailed breakdown:

  - `code.ipynb`: Python script for the main processing and modeling tasks performed in February.
  - `fraudTest.csv`:  CSV file with test data for evaluating the fraud detection model.
  - `fraudTrain.csv`: CSV file containing training data for the fraud detection model.

Please reach out to me for the datasets. Dataset file is very large.

## Getting Started

### Prerequisites

Before running this project, ensure you have the following installed:
- Python 3.x
- Libraries: `pandas`, `matplotlib`, `seaborn`, `sklearn`, `imblearn`, `pickle`, `json`

You can install the required libraries using the following command:

```bash
pip install pandas matplotlib seaborn sklearn imblearn pickle json
```

Running the Scripts

To run this project, clone the repo and execute the main script:

```bash
git clone <repo-url>
cd Credit-Card-Fraud-Management-System
python main.py
```

## Models Used

- **Random Forest Classifier**
- **Logistic Regression**
- **Gaussian Naive Bayes**
- **K-Nearest Neighbors**
- **Voting Classifier** (Ensemble)

## Key Findings

- **Fraud Detection Accuracy:** Achieved an accuracy of approximately 99.48%.
- **Key Features:** The transaction amount showed a significant correlation with the fraud label, indicating its importance in the detection process.
- **Data Imbalance:** Addressed using SMOTE to oversample the minority class in the training data.

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your enhancements.
