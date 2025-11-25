"""
Module định nghĩa và huấn luyện các mô hình machine learning

Module này chứa:
- Các mô hình classification và regression
- Hàm đánh giá hiệu suất
- Cross-validation
- Visualization kết quả
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)
import joblib
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Class đánh giá và so sánh các mô hình machine learning"""
    
    def __init__(self):
        """
        Khởi tạo evaluator
        """
        self.classification_results = []
        self.regression_results = []
        self.trained_models = {}
    
    def evaluate_classification_model(self, model, X_train, X_test, y_train, y_test, model_name):
        """
        Đánh giá hiệu suất mô hình classification
        
        Args:
            model: Mô hình cần đánh giá
            X_train, X_test: Dữ liệu train và test
            y_train, y_test: Target train và test
            model_name (str): Tên mô hình
            
        Returns:
            dict: Dictionary chứa kết quả đánh giá
        """
        # Huấn luyện mô hình
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Tính toán metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # In kết quả
        print(f"\n{model_name}:")
        print(f"Accuracy: {acc:.3f}")
        print(f"Precision: {prec:.3f}")
        print(f"Recall: {rec:.3f}")
        print(f"F1: {f1:.3f}")
        
        # Lưu mô hình đã trained
        self.trained_models[model_name] = model
        
        result = {
            'Model': model_name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1
        }
        
        self.classification_results.append(result)
        return result
    
    def evaluate_regression_model(self, model, X_train, X_test, y_train, y_test, model_name):
        """
        Đánh giá hiệu suất mô hình regression
        
        Args:
            model: Mô hình cần đánh giá
            X_train, X_test: Dữ liệu train và test
            y_train, y_test: Target train và test
            model_name (str): Tên mô hình
            
        Returns:
            dict: Dictionary chứa kết quả đánh giá
        """
        # Huấn luyện mô hình
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Tính toán metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # In kết quả
        print(f"\n{model_name}:")
        print(f"MAE: ${mae:,.0f}")
        print(f"RMSE: ${rmse:,.0f}")
        print(f"R²: {r2:.3f}")
        
        # Lưu mô hình đã trained
        self.trained_models[model_name] = model
        
        result = {
            'Model': model_name,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2
        }
        
        self.regression_results.append(result)
        return result
    
    def cross_validate_model(self, model, X, y, cv=5, scoring='accuracy'):
        """
        Thực hiện cross-validation
        
        Args:
            model: Mô hình cần đánh giá
            X: Features
            y: Target
            cv (int): Số folds
            scoring (str): Metric để đánh giá
            
        Returns:
            dict: Kết quả cross-validation
        """
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        print(f"Cross-Validation Results (k={cv}):")
        print(f"Scores: {cv_scores}")
        print(f"Mean: {cv_scores.mean():.3f}")
        print(f"Std: {cv_scores.std():.3f}")
        print(f"Range: {cv_scores.min():.3f} - {cv_scores.max():.3f}")
        
        return {
            'scores': cv_scores,
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'min': cv_scores.min(),
            'max': cv_scores.max()
        }
    
    def compare_classification_models(self):
        """
        So sánh kết quả của các mô hình classification
        
        Returns:
            pd.DataFrame: Bảng so sánh kết quả
        """
        if not self.classification_results:
            print("Chưa có kết quả classification để so sánh!")
            return None
            
        results_df = pd.DataFrame(self.classification_results)
        print("\nSo sánh Classification Models:")
        print(results_df)
        
        return results_df
    
    def compare_regression_models(self):
        """
        So sánh kết quả của các mô hình regression
        
        Returns:
            pd.DataFrame: Bảng so sánh kết quả
        """
        if not self.regression_results:
            print("Chưa có kết quả regression để so sánh!")
            return None
            
        results_df = pd.DataFrame(self.regression_results)
        print("\nSo sánh Regression Models:")
        print(results_df)
        
        # Tính phần trăm cải thiện nếu có ít nhất 2 mô hình
        if len(self.regression_results) >= 2:
            model1, model2 = self.regression_results[0], self.regression_results[1]
            mae_improve = (model1['MAE'] - model2['MAE']) / model1['MAE'] * 100
            rmse_improve = (model1['RMSE'] - model2['RMSE']) / model1['RMSE'] * 100
            r2_improve = (model2['R²'] - model1['R²']) / model1['R²'] * 100
            
            print(f"\nCải thiện của {model2['Model']} so với {model1['Model']}:")
            print(f"MAE: {mae_improve:.1f}%")
            print(f"RMSE: {rmse_improve:.1f}%")
            print(f"R²: {r2_improve:.1f}%")
        
        return results_df
    
    def visualize_classification_results(self):
        """
        Vẽ biểu đồ so sánh kết quả classification
        """
        if len(self.classification_results) < 2:
            print("Cần ít nhất 2 mô hình để so sánh!")
            return
            
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
        x = range(len(metrics))
        
        plt.figure(figsize=(10, 6))
        
        for i, result in enumerate(self.classification_results):
            values = [result[m] for m in metrics]
            plt.bar([j + i*0.35 - 0.175 for j in x], values, 0.35, 
                   label=result['Model'])
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('So sánh Classification Models')
        plt.xticks(x, metrics)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def visualize_regression_results(self):
        """
        Vẽ biểu đồ so sánh kết quả regression
        """
        if len(self.regression_results) < 2:
            print("Cần ít nhất 2 mô hình để so sánh!")
            return
            
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        models = [r['Model'] for r in self.regression_results]
        mae_values = [r['MAE'] for r in self.regression_results]
        rmse_values = [r['RMSE'] for r in self.regression_results]
        r2_values = [r['R²'] for r in self.regression_results]
        
        axes[0].bar(models, mae_values)
        axes[0].set_title('MAE Comparison')
        axes[0].set_ylabel('MAE ($)')
        
        axes[1].bar(models, rmse_values)
        axes[1].set_title('RMSE Comparison')
        axes[1].set_ylabel('RMSE ($)')
        
        axes[2].bar(models, r2_values)
        axes[2].set_title('R² Comparison')
        axes[2].set_ylabel('R² Score')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_cv_results(self, cv_results, model_name="Model"):
        """
        Vẽ biểu đồ kết quả cross-validation
        
        Args:
            cv_results (dict): Kết quả từ cross_validate_model
            model_name (str): Tên mô hình
        """
        scores = cv_results['scores']
        mean_score = cv_results['mean']
        
        plt.figure(figsize=(8, 5))
        plt.bar(range(1, len(scores) + 1), scores)
        plt.axhline(y=mean_score, color='red', linestyle='--', 
                   label=f'Mean: {mean_score:.3f}')
        plt.xlabel('Fold')
        plt.ylabel('Score')
        plt.title(f'{model_name} Cross-Validation Scores')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def save_model(self, model_name, filepath):
        """
        Lưu mô hình đã huấn luyện
        
        Args:
            model_name (str): Tên mô hình
            filepath (str): Đường dẫn lưu file
        """
        if model_name not in self.trained_models:
            print(f"Mô hình {model_name} chưa được huấn luyện!")
            return
            
        joblib.dump(self.trained_models[model_name], filepath)
        print(f"Đã lưu mô hình {model_name} tại {filepath}")
    
    def load_model(self, filepath):
        """
        Tải mô hình đã lưu
        
        Args:
            filepath (str): Đường dẫn file mô hình
            
        Returns:
            object: Mô hình đã tải
        """
        try:
            model = joblib.load(filepath)
            print(f"Đã tải mô hình từ {filepath}")
            return model
        except Exception as e:
            print(f"Lỗi khi tải mô hình: {e}")
            return None


class ModelFactory:
    """Factory class để tạo các mô hình"""
    
    @staticmethod
    def create_classification_models():
        """
        Tạo các mô hình classification
        
        Returns:
            dict: Dictionary chứa các mô hình
        """
        models = {
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        return models
    
    @staticmethod
    def create_regression_models():
        """
        Tạo các mô hình regression
        
        Returns:
            dict: Dictionary chứa các mô hình
        """
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        return models


if __name__ == "__main__":
    # Test modeling
    print("Testing ModelEvaluator and ModelFactory...")
    
    # Tạo evaluator
    evaluator = ModelEvaluator()
    
    # Tạo các mô hình
    class_models = ModelFactory.create_classification_models()
    reg_models = ModelFactory.create_regression_models()
    
    print(f"Đã tạo {len(class_models)} mô hình classification")
    print(f"Đã tạo {len(reg_models)} mô hình regression")
    
    print("Modeling module sẵn sàng sử dụng!")
