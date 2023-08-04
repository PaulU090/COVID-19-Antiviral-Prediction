import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    data.drop(columns=["Unnamed: 0"], inplace=True)

    feature_columns = [f"feat_{i}" for i in range(1, 209)]
    data[feature_columns] = data[feature_columns].fillna(0)

    infinite_columns = data.columns[(data == float('inf')).any() | (data == float('-inf')).any()]
    data[infinite_columns] = data[infinite_columns].replace([float('inf'), float('-inf')], 0)

    train_data = data[data['subset'] == 'train']
    valid_data = data[data['subset'] == 'valid']
    test_data = data[data['subset'] == 'test']

    X_train = train_data[feature_columns]
    y_train = train_data['label']
    X_valid = valid_data[feature_columns]
    y_valid = valid_data['label']
    X_test = test_data[feature_columns]
    y_test = test_data['label']

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def feature_scaling(X_train, X_valid, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_valid_scaled, X_test_scaled

def train_models(X_train_scaled, y_train):
    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_train_scaled, y_train)

    random_forest_model = RandomForestClassifier(random_state=42)
    random_forest_model.fit(X_train_scaled, y_train)

    return logistic_model, random_forest_model

def evaluate_model(model, X_test, y_test, model_name):
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    auc = roc_auc_score(y_test, y_test_prob)

    print(f"Evaluation Metrics for {model_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")

    fpr, tpr, _ = roc_curve(y_test, y_test_prob)

    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend()
    plt.show()

def main():
    file_path = 'data/mpro_exp_data2_rdkit_feat.csv'
    data = load_data(file_path)
    X_train, y_train, X_valid, y_valid, X_test, y_test = preprocess_data(data)
    X_train_scaled, X_valid_scaled, X_test_scaled = feature_scaling(X_train, X_valid, X_test)
    logistic_model, random_forest_model = train_models(X_train_scaled, y_train)
    evaluate_model(logistic_model, X_test_scaled, y_test, "Logistic Regression")
    evaluate_model(random_forest_model, X_test_scaled, y_test, "Random Forest")

if __name__ == "__main__":
    main()