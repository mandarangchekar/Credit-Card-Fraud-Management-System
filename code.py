#!/usr/bin/env python
# coding: utf-8

# In[37]:





# In[38]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle
import json
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder



# In[39]:


# Load the training data
train_data_path = 'fraudTrain.csv'  
train_df = pd.read_csv(train_data_path)

# Load the testing data
test_data_path = 'fraudTest.csv'  
test_df = pd.read_csv(test_data_path)

# Display the first few rows of the training data
print("Training Data:")
print(train_df.head())

# Display the first few rows of the testing data
print("\nTesting Data:")
print(test_df.head())


# In[ ]:





# In[40]:


# Basic structure
print("Training Data Shape:", train_df.shape)
print("Testing Data Shape:", test_df.shape)
print("\nFeature Data Types:\n", train_df.dtypes)

# Summary statistics for numerical features
print("\nSummary Statistics for Numerical Features:\n", train_df.describe())

# Summary of categorical features
print("\nSummary of Categorical Features:\n", train_df.describe(include='object'))

# Missing values
print("\nMissing Values:\n", train_df.isnull().sum())

# Number of unique values in each feature
print("\nUnique Values:\n", train_df.nunique())

# Distribution of the target variable
print("\nDistribution of Target Variable 'is_fraud':\n", train_df['is_fraud'].value_counts(normalize=True))

# Correlation matrix
numeric_columns = train_df.select_dtypes(include=['int64', 'float64']).columns
print("\nCorrelation Matrix:\n", train_df[numeric_columns].corr())

# Set the aesthetic style of the plots
sns.set_style('whitegrid')

# Histograms for numerical features
train_df.hist(bins=15, figsize=(15, 10), layout=(5, 4))
plt.show()

#Bar chart for categorical features - you can choose specific categorical features to visualize
plt.figure(figsize=(10, 6))
sns.countplot(x='category', data=train_df)
plt.xticks(rotation=90)
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(train_df[numeric_columns].corr(), annot=True, fmt=".2f")
plt.show()

#Boxplot to check for outliers - choose a specific feature to check for outliers
sns.boxplot(x=train_df['amt'])
plt.show()


# In[41]:


# Feature Engineering

# Convert to datetime
train_df['trans_date_trans_time'] = pd.to_datetime(train_df['trans_date_trans_time'])
train_df['dob'] = pd.to_datetime(train_df['dob'])

# Extract features
train_df['transaction_hour'] = train_df['trans_date_trans_time'].dt.hour
train_df['day_of_week'] = train_df['trans_date_trans_time'].dt.dayofweek
train_df['age'] = (train_df['trans_date_trans_time'] - train_df['dob']).dt.days // 365

# Drop original datetime columns to avoid redundancy
train_df.drop(['trans_date_trans_time', 'dob'], axis=1, inplace=True)


# In[42]:


# # Preprocessing

# Encode categorical variables as necessary (Example)
encoder = LabelEncoder()
train_df['gender'] = encoder.fit_transform(train_df['gender'])

# Standardize numerical features
scaler = StandardScaler()
numerical_cols = ['amt', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long', 'age']
train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])


# In[43]:


from sklearn.preprocessing import LabelEncoder

# Updated list with all categorical columns, excluding unique identifiers and directly numerical columns
categorical_cols = ['merchant', 'category', 'gender', 'state', 'first', 'last', 'street', 'city', 'job']

# Apply Label Encoding to each categorical column
for col in categorical_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])

# After encoding, drop any columns that are not useful for the model
columns_to_drop = ['trans_num', 'Unnamed: 0']  # Add or remove columns based on your dataset
train_df = train_df.drop(columns=columns_to_drop)

from imblearn.over_sampling import SMOTE

# Prepare features and target variable, ensuring to exclude any columns not needed for modeling
X = train_df.drop('is_fraud', axis=1)
y = train_df['is_fraud']

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)


# In[44]:
# Initialize LabelEncoders for each categorical column
# Initialize LabelEncoders for each categorical column
# label_encoders = {col: LabelEncoder() for col in ['merchant', 'category', 'gender', 'state', 'job']}

# # Apply Label Encoding to each categorical column and store mappings
# mappings = {}
# for col, encoder in label_encoders.items():
#     train_df[col] = encoder.fit_transform(train_df[col])
#     mappings[col] = {label: index for index, label in enumerate(encoder.classes_)}

# # Save the label encoders to a file for future use
# with open('label_encoders.json', 'w') as file:
#     json.dump(mappings, file, indent=4)

# # Standardize numerical features
# scaler = StandardScaler()
# numerical_cols = ['amt', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long']
# train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])

# # Save the fitted scaler to a file for later use in preprocessing new input data
# with open('scaler.pkl', 'wb') as scaler_file:
#     pickle.dump(scaler, scaler_file)

# # Drop any columns that are not useful for the model
# columns_to_drop = ['trans_num', 'Unnamed: 0']  # Add or remove columns based on your dataset specifics
# train_df = train_df.drop(columns=columns_to_drop)

# # Prepare features and target variable for modeling
# X = train_df.drop('is_fraud', axis=1)
# y = train_df['is_fraud']

# # Apply SMOTE for handling imbalanced data
# sm = SMOTE(random_state=42)
# X_res, y_res = sm.fit_resample(X, y)
# In[45]:


# Preparation of data
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the fitted scaler to a file for later use in preprocessing new input data
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)




# In[46]:


log_clf = LogisticRegression(max_iter=10, random_state=42)
rf_clf = RandomForestClassifier(n_estimators=10, random_state=42)
nb_clf = GaussianNB()
knn_clf = KNeighborsClassifier(n_neighbors=5)


# In[18]:


voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rf_clf), ('nb', nb_clf), ('knn', knn_clf)],
    voting='soft'  # Try 'hard' as well
)
voting_clf.fit(X_train_scaled, y_train)


# In[19]:


y_pred = voting_clf.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# In[20]:


# Assuming 'voting_clf' is your trained model
with open('fraud_model.pkl', 'wb') as file:
    pickle.dump(voting_clf, file)


# In[24]:


# List unique merchants
unique_merchants = train_df['merchant'].unique()
print("Unique Merchants:")
print(unique_merchants)

# List unique categories
unique_categories = train_df['category'].unique()
print("\nUnique Categories:")
print(unique_categories)

# List unique states
unique_states = train_df['state'].unique()
print("\nUnique States:")
print(unique_states)


# In[ ]:




