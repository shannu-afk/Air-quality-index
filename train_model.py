# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Read the dataset
df = pd.read_csv('Dataset/Airquality_index.csv')
logger.info("Dataset loaded successfully.")

# Drop rows with missing values
df = df.dropna()
logger.info(f"Dataset cleaned. Shape after dropping missing values: {df.shape}")

# Prepare features and target
X = df.iloc[:,:-1]  # All columns except the last one as features
y = df.iloc[:,-1]   # Last column as target

# Print dataset info
print("Dataset shape after cleaning:", df.shape)
print("\nFeatures:", X.columns.tolist())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
logger.info(f"Data split into train and test sets. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Train Decision Tree model
dtree = DecisionTreeRegressor(criterion="squared_error", random_state=100)
dtree.fit(X_train, y_train)
logger.info("Decision Tree model trained successfully.")

# Save the model
with open('tree_gridcv.pkl', 'wb') as file:
    pickle.dump(dtree, file)
logger.info("Model saved to tree_gridcv.pkl")

# Print model score
print(f"Model Score on test data: {dtree.score(X_test, y_test):.4f}")