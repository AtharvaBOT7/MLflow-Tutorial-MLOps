import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Setting the tracking URI for MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Adjust the URI as

wine = load_wine()
X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

max_depth = 15
n_estimators = 20

# Mention your experiment name 
mlflow.set_experiment("Wine_Quality_Classification")

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", accuracy)
    
    # mlflow.sklearn.log_model(rf, "model")

    # Creating and saving confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    plt.savefig("confusion_matrix.png")

    # Log artifacts using MLflow
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)

    # Log the tags
    mlflow.set_tags({
        "Author": "Atharva",
        "Project": "Wine Quality Classification",
        "Model": "RandomForestClassifier",})
    
    # Log the model
    mlflow.sklearn.log_model(rf, "Random Forest Classifier Model")

    print(f"Accuracy: {accuracy}")

    
    
    
   