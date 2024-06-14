from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd



df = pd.read_csv('Processed Data/Instances Imputed.csv', delimiter='\t')


#count the number of duplicate rows where all columns are the same (except for date)
# duplicates = df[df.duplicated(subset=['Environmental Audio Exposure (dBASPL)', 'Blood Oxygen Saturation (%)', 'Walking Speed (km/hr)', 'Stress Level'], keep=False)]
# df = df.drop_duplicates(subset=['Environmental Audio Exposure (dBASPL)', 'Blood Oxygen Saturation (%)', 'Walking Speed (km/hr)', 'Stress Level'])
# print(len(duplicates))

X = df.drop(columns=['Stress Level', 'Date/Time'])

y = df['Stress Level']

print(X)
print(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier(max_depth=10,random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# print accuracy, precision, and recall, for each of the three Stress Level labels (Low, Medium, High)
print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred, average=None))
print(recall_score(y_test, y_pred, average=None))

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# import xgboost as xgb
# from sklearn.preprocessing import LabelEncoder

# # Load the dataset
# df = pd.read_csv('instances_final.csv', delimiter='\t')

# # Drop duplicate rows based on specified columns
# df = df.drop_duplicates(subset=['Environmental Audio Exposure (dBASPL)', 'Blood Oxygen Saturation (%)', 'Walking Speed (km/hr)', 'Stress Level'])
# print('remaining len of df', len(df))

# # Define features and target variable
# X = df[['Environmental Audio Exposure (dBASPL)', 'Blood Oxygen Saturation (%)', 'Walking Speed (km/hr)']]
# y = df['Stress Level']

# # Encode target labels with value between 0 and n_classes-1
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# # Initialize the XGBoost Classifier
# clf = xgb.XGBClassifier(random_state=42)#,max_depth=10)

# # Train the classifier
# clf.fit(X_train, y_train)

# # Make predictions
# y_pred = clf.predict(X_test)

# # Print accuracy, precision, and recall for each of the three Stress Level labels (Low, Medium, High)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Precision:", precision_score(y_test, y_pred, average=None))
# print("Recall:", recall_score(y_test, y_pred, average=None))

# # Decode the predicted labels back to the original string labels (optional)
# y_test_decoded = label_encoder.inverse_transform(y_test)
# y_pred_decoded = label_encoder.inverse_transform(y_pred)

# print("Decoded Precision:", precision_score(y_test_decoded, y_pred_decoded, average=None))
# print("Decoded Recall:", recall_score(y_test_decoded, y_pred_decoded, average=None))
