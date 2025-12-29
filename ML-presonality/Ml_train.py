import pandas as pd
import numpy as np
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# loaded dataset
df = pd.read_csv("personality_dataset.csv")

print("Dataset Preview:")
print(df.head())
print("\nDataset Info:")
print(df.info())



categorical_cols = ['Stage_fear', 'Drained_after_socializing']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])


#  Encode TARGET column

target_encoder = LabelEncoder()
df['Personality'] = target_encoder.fit_transform(df['Personality'])
#  Extrovert = 0, Introvert = 1

# print("\nTarget Classes:", target_encoder.classes_)

X = df.drop('Personality', axis=1)
y = df['Personality']



#  Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#  Feature Scaling

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# KNN Model

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

#  how well did knn perform is it accurate
print("\n===== KNN RESULTS =====")
print("Accuracy:", accuracy_score(y_test, knn_pred))
print(classification_report(y_test, knn_pred))
print(confusion_matrix(y_test, knn_pred))



# . SVM Model algorithm

svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

print("\n===== SVM RESULTS =====")
print("Accuracy:", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred))
print(confusion_matrix(y_test, svm_pred))



#  Model Comparison

print("\n===== MODEL COMPARISON =====")
print("KNN Accuracy:", accuracy_score(y_test, knn_pred))
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))

best_model = svm if accuracy_score(y_test, svm_pred) >= accuracy_score(y_test, knn_pred) else knn
best_model_name = "SVM" if best_model == svm else "KNN"
print("Best Model:", best_model_name)



#  Save Model with Joblib
joblib.dump(best_model, "personality_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")

# Save accuracy scores to JSON
accuracy_data = {
    "knn_accuracy": accuracy_score(y_test, knn_pred),
    "svm_accuracy": accuracy_score(y_test, svm_pred),
    "best_model": best_model_name
}
with open("model_accuracy.json", "w") as f:
    json.dump(accuracy_data, f)

print("\n===== MODEL SAVED =====")
print("Saved: personality_model.pkl, scaler.pkl, target_encoder.pkl, model_accuracy.json")

