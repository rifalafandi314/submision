import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# load data
df = pd.read_csv("dataset_preprocessing/data_clean.csv")

# cleaning ulang biar aman
df = df.dropna(subset=['clean_text'])
df['clean_text'] = df['clean_text'].astype(str)

X = df['clean_text']
y = df['label']

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TF-IDF
tfidf = TfidfVectorizer()
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# model
model = RandomForestClassifier(random_state=42)

# hyperparameter tuning
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [None, 10]
}

grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train_vec, y_train)

best_model = grid_search.best_estimator_

# evaluasi
y_pred = best_model.predict(X_test_vec)

acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

# MLflow manual logging
with mlflow.start_run():
    # log parameter terbaik
    mlflow.log_params(grid_search.best_params_)

    # log metric
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # log model
    mlflow.sklearn.log_model(best_model, "model")

    print("Best Params:", grid_search.best_params_)
    print("Accuracy:", acc)