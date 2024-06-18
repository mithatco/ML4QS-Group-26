# from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# from sklearn.utils import resample
# import pandas as pd

# df = pd.read_csv('Processed Data/Instances Imputed.csv', delimiter='\t')

# # Get the count of the most common class
# max_size = df['Stress Level'].value_counts().max()

# # Create a list to collect the resampled dataframes
# dfs = []

# for class_index, group in df.groupby('Stress Level'):
#     df_group_resampled = resample(group, 
#                                   replace=True, 
#                                   n_samples=max_size, 
#                                   random_state=42)
#     dfs.append(df_group_resampled)

# # Concatenate the resampled dataframes
# df_resampled = pd.concat(dfs)

# # Display new class counts
# print(df_resampled['Stress Level'].value_counts())

# X = df.drop(columns=['Stress Level', 'Date/Time'])

# y = df['Stress Level']

# # Calculate the index to split at
# split_index = int(0.8 * len(X))

# # Split the data and target arrays
# X_train, X_test = X[:split_index], X[split_index:]
# y_train, y_test = y[:split_index], y[split_index:]

# print("Training data:", X_train)
# print("Testing data:", X_test)
# print("Training target:", y_train)
# print("Testing target:", y_test)

# # Initialize the Decision Tree Classifier
# clf = DecisionTreeClassifier(random_state=42, class_weight='balanced')

# # Define the parameter grid
# # param_grid = {
# #     'criterion': ['gini', 'entropy'],
# #     'splitter': ['best', 'random'],
# #     'max_depth': [None, 10, 20, 30, 40, 50],
# #     'min_samples_split': [2, 5, 10],
# #     'min_samples_leaf': [1, 2, 4],
# #     'max_features': [None, 'auto', 'sqrt', 'log2']
# # }

# param_grid = {
#     'criterion': ['gini'],
#     'splitter': ['best'],
#     'max_depth': [30],
#     'min_samples_split': [2],
#     'min_samples_leaf': [1],
#     'max_features': ['sqrt']
# }

# # Use TimeSeriesSplit to avoid shuffling the data during cross-validation
# tscv = TimeSeriesSplit(n_splits=5)

# # Initialize GridSearchCV with TimeSeriesSplit
# grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=tscv, n_jobs=-1, verbose=2)

# # Fit the grid search to the data
# grid_search.fit(X_train, y_train)

# # Get the best estimator
# best_clf = grid_search.best_estimator_

# # Make predictions with the best estimator
# y_pred = best_clf.predict(X_test)

# # Print the best parameters found by the grid search
# print("Best parameters found by grid search:", grid_search.best_params_)

# # Print accuracy, precision, and recall for each of the three Stress Level labels (Low, Medium, High)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Precision:", precision_score(y_test, y_pred, average=None))
# print("Recall:", recall_score(y_test, y_pred, average=None))

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('Processed Data/Instances Imputed.csv', delimiter='\t')

# Define features and target variable
X = df.drop(columns=['Stress Level', 'Date/Time'])
y = df['Stress Level']

# Encode target labels with value between 0 and n_classes-1
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Define the XGBoost classifier
clf = xgb.XGBClassifier(random_state=42)

# Define the parameter grid to search
param_grid = {
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='accuracy')

# Fit the model
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best parameters found: ", best_params)

# Train the classifier with the best parameters
best_clf = xgb.XGBClassifier(**best_params, random_state=42)
best_clf.fit(X_train, y_train)

# Make predictions
y_pred = best_clf.predict(X_test)

# Print accuracy, precision, and recall for each of the three Stress Level labels (Low, Medium, High)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average=None))
print("Recall:", recall_score(y_test, y_pred, average=None))

# Decode the predicted labels back to the original string labels (optional)
y_test_decoded = label_encoder.inverse_transform(y_test)
y_pred_decoded = label_encoder.inverse_transform(y_pred)

print("Decoded Precision:", precision_score(y_test_decoded, y_pred_decoded, average=None))
print("Decoded Recall:", recall_score(y_test_decoded, y_pred_decoded, average=None))

# from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# import pandas as pd

# # Load the dataset
# df = pd.read_csv('Processed Data/Instances Imputed.csv', delimiter='\t')

# # Prepare the features (X) and the target (y)
# X = df.drop(columns=['Stress Level', 'Date/Time'])
# y = df['Stress Level']

# # Check for class imbalance
# print(y.value_counts())

# # Split the dataset into training and testing sets without shuffling
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# # Initialize the Random Forest Classifier with class weight adjustment
# clf = RandomForestClassifier(random_state=42, class_weight='balanced')

# # Define the parameter grid
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'criterion': ['gini', 'entropy'],
#     'max_depth': [None, 10, 20, 30, 40, 50],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['auto', 'sqrt', 'log2']
# }

# # Use TimeSeriesSplit to avoid shuffling the data during cross-validation
# tscv = TimeSeriesSplit(n_splits=5)

# # Initialize GridSearchCV with TimeSeriesSplit
# grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=tscv, n_jobs=-1, verbose=2)

# # Fit the grid search to the data
# grid_search.fit(X_train, y_train)

# # Get the best estimator
# best_clf = grid_search.best_estimator_

# # Make predictions with the best estimator
# y_pred = best_clf.predict(X_test)

# # Print the best parameters found by the grid search
# print("Best parameters found by grid search:", grid_search.best_params_)

# # Print accuracy, precision, and recall for each of the three Stress Level labels (Low, Medium, High)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Precision:", precision_score(y_test, y_pred, average=None))
# print("Recall:", recall_score(y_test, y_pred, average=None))