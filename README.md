##Credit Card Fraud Management System
This project aims to detect and manage credit card fraud using a variety of machine learning techniques and data preprocessing methods. The repository contains Python scripts that perform data analysis, model training, and evaluation on a large dataset of credit card transactions.

Project Structure
Data Handling: Load and display training and testing data.
Data Analysis: Perform exploratory data analysis including statistical summaries and correlation analysis.
Preprocessing: Standardize numerical features and encode categorical variables.
Model Training: Train multiple machine learning models to detect fraudulent transactions.
Model Evaluation: Evaluate model performance using accuracy, precision, recall, and F1-score.
Data Serialization: Save trained models and configurations for deployment.
Dataset
Training Data: 1,296,675 entries, 23 features.
Testing Data: 555,719 entries, 23 features.
Features: Transaction details, merchant info, customer demographics, transaction coordinates, and more.
Features Details
Numerical: Transaction amount, zip code, geographic coordinates, city population, Unix time, etc.
Categorical: Transaction time, credit card number, merchant details, transaction category, customer details, etc.
Prerequisites
Before running this project, ensure you have the following installed:

Python 3.x
Libraries: pandas, matplotlib, seaborn, sklearn, imblearn, pickle, json
You can install the required libraries using the following command:

bash
Copy code
pip install pandas matplotlib seaborn sklearn imblearn pickle json
Running the Scripts
To run this project, clone the repo and execute the main script:

bash
Copy code
git clone <repo-url>
cd Credit-Card-Fraud-Management-System
python main.py
Models Used
Random Forest Classifier
Logistic Regression
Gaussian Naive Bayes
K-Nearest Neighbors
Voting Classifier (Ensemble)
Key Findings
Fraud Detection Accuracy: Approximately 99.48%.
Key Features: Transaction amount showed a significant correlation with the fraud label.
Data Imbalance: Handled using SMOTE to oversample the minority class in the training data.
Output
Model evaluation results are saved in the output directory, including:

Confusion matrices
Classification reports
Serialized model files
Contributing
Contributions to this project are welcome! Please fork the repository and submit a pull request with your enhancements.
