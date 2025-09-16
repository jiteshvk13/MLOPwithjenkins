from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
import sys

from torch.utils.tensorboard import SummaryWriter
from config.paths_config import *
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

class ModelSelection:
    def __init__(self, data_path):
        self.data_path = data_path
        run_id = time.strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(log_dir=f"tensorboard_logs/run_{run_id}")
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=500),
            'Random Forest': RandomForestClassifier(n_estimators=50, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=50),
            'AdaBoost': AdaBoostClassifier(n_estimators=50),
            'Support Vector Classifier': SVC(),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Naive Bayes': GaussianNB(),
            'Decision Tree': DecisionTreeClassifier(),
            'LightGBM': lgb.LGBMClassifier(),
            'XGBoost': xgb.XGBClassifier(eval_metric='mlogloss')
        }
        self.results = {}

    def load_data(self):
        try:
            logger.info("Loading CSV File")
            df = pd.read_csv(self.data_path)
            df_sample = df.sample(frac=0.05, random_state=42)
            X = df.drop(columns='satisfaction')
            y = df['satisfaction']
            logger.info("Data Loaded Successfully")
            return X, y
        except Exception as e:
            raise CustomException("Error while loading data", e)

    def split_data(self, X, y):
        try:
            logger.info("Splitting Data")
            return train_test_split(X, y, test_size=0.2, random_state=42)
        except Exception as e:
            raise CustomException("Error while splitting data", e)

    def log_confusion_matrix(self, y_true, y_pred, step, model_name):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(x=j, y=i, s=cm[i, j], va="center", ha="center")

        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title(f"Confusion Matrix for {model_name}")
        self.writer.add_figure(f"Confusion_Matrix/{model_name}", fig, global_step=step)
        plt.close(fig)

    def train_evaluate(self, X_train, X_test, y_train, y_test):
        try:
            logger.info("Training and Evaluation Started")

            # Scale features for models that need it
            scalable_models = ['Logistic Regression', 'Support Vector Classifier', 'K-Nearest Neighbors']
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            for idx, (name, model) in enumerate(self.models.items()):
                if name in scalable_models:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                self.results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }

                logger.info(
                    f"{name} trained successfully. Metrics: Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}"
                )

                # Log metrics to TensorBoard
                self.writer.add_scalar(f'Accuracy/{name}', accuracy, idx)
                self.writer.add_scalar(f'Precision/{name}', precision, idx)
                self.writer.add_scalar(f'Recall/{name}', recall, idx)
                self.writer.add_scalar(f'F1_score/{name}', f1, idx)
                self.writer.add_text(
                    'Model Details',
                    f"Name: {name} | Metrics: Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}"
                )

                self.log_confusion_matrix(y_test, y_pred, idx, name)

            self.writer.close()
            logger.info("Training and Evaluation Completed Successfully")

        except Exception as e:
            raise CustomException("Error during model training and evaluation", e)

    def run(self):
        try:
            logger.info("Model Selection Pipeline Started")
            X, y = self.load_data()
            X_train, X_test, y_train, y_test = self.split_data(X, y)
            self.train_evaluate(X_train, X_test, y_train, y_test)
            logger.info("Pipeline Completed Successfully")
        except Exception as e:
            logger.error(f"Error in the pipeline: {e}")
            raise CustomException("Error in the Pipeline", e)


if __name__ == '__main__':
    model_selection = ModelSelection(ENGINEERED_DATA_PATH)
    model_selection.run()
