"""
Module dự đoán và tổng kết kết quả

Module này chứa:
- Pipeline hoàn chỉnh từ preprocessing đến prediction
- Tổng kết và summary kết quả
- Main script để chạy toàn bộ quy trình
"""

import sys
import os

# Thêm đường dẫn src vào sys.path để import các module khác
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import HousePricePreprocessor
from modeling import ModelEvaluator, ModelFactory
import warnings
warnings.filterwarnings('ignore')


class HousePricePipeline:
    """Pipeline hoàn chỉnh cho dự án House Price Prediction"""
    
    def __init__(self, data_path='../data/House_Prices.csv'):
        """
        Khởi tạo pipeline
        
        Args:
            data_path (str): Đường dẫn đến file dữ liệu
        """
        self.data_path = data_path
        self.preprocessor = HousePricePreprocessor(data_path)
        self.evaluator = ModelEvaluator()
        self.results_summary = {}
    
    def run_full_pipeline(self, save_models=True):
        """
        Chạy toàn bộ pipeline từ preprocessing đến evaluation
        
        Args:
            save_models (bool): Có lưu mô hình hay không
        """
        print("="*50)
        print("       HOUSE PRICE PREDICTION PIPELINE")
        print("="*50)
        
        # 1. Preprocessing
        print("\n1. PREPROCESSING DATA...")
        self.preprocessor.explore_data()
        X, y_class, y_reg = self.preprocessor.prepare_features()
        
        # Chia dữ liệu
        data_splits = self.preprocessor.split_data(X, y_class, y_reg)
        X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = data_splits
        
        # Chuẩn hóa dữ liệu
        X_train_scaled, X_test_scaled = self.preprocessor.scale_features(X_train, X_test)
        
        # 2. Classification
        print("\n2. CLASSIFICATION MODELS...")
        self._run_classification(X_train, X_test, X_train_scaled, X_test_scaled,
                               y_train_class, y_test_class)
        
        # 3. Cross-validation
        print("\n3. CROSS-VALIDATION...")
        self._run_cross_validation(X, y_class)
        
        # 4. Regression
        print("\n4. REGRESSION MODELS...")
        self._run_regression(X_train, X_test, X_train_scaled, X_test_scaled,
                           y_train_reg, y_test_reg)
        
        # 5. Visualization
        print("\n5. VISUALIZATION...")
        self._create_visualizations()
        
        # 6. Summary
        print("\n6. FINAL SUMMARY...")
        self._create_summary()
        
        # 7. Save models
        if save_models:
            print("\n7. SAVING MODELS...")
            self._save_best_models()
    
    def _run_classification(self, X_train, X_test, X_train_scaled, X_test_scaled,
                          y_train, y_test):
        """
        Chạy các mô hình classification
        """
        # Decision Tree (dữ liệu gốc)
        dt_model = ModelFactory.create_classification_models()['Decision Tree']
        dt_result = self.evaluator.evaluate_classification_model(
            dt_model, X_train, X_test, y_train, y_test, "Decision Tree"
        )
        
        # Logistic Regression (dữ liệu chuẩn hóa)
        lr_model = ModelFactory.create_classification_models()['Logistic Regression']
        lr_result = self.evaluator.evaluate_classification_model(
            lr_model, X_train_scaled, X_test_scaled, y_train, y_test, "Logistic Regression"
        )
        
        # So sánh kết quả
        self.evaluator.compare_classification_models()
        
        # Lưu kết quả
        self.results_summary['classification'] = {
            'decision_tree': dt_result,
            'logistic_regression': lr_result
        }
    
    def _run_cross_validation(self, X, y):
        """
        Chạy cross-validation cho Random Forest
        """
        rf_model = ModelFactory.create_classification_models()['Random Forest']
        cv_results = self.evaluator.cross_validate_model(
            rf_model, X, y, cv=5, scoring='accuracy'
        )
        
        # Visualization CV results
        self.evaluator.visualize_cv_results(cv_results, "Random Forest")
        
        # Lưu kết quả
        self.results_summary['cross_validation'] = cv_results
    
    def _run_regression(self, X_train, X_test, X_train_scaled, X_test_scaled,
                       y_train, y_test):
        """
        Chạy các mô hình regression
        """
        # Linear Regression (dữ liệu chuẩn hóa)
        lr_model = ModelFactory.create_regression_models()['Linear Regression']
        lr_result = self.evaluator.evaluate_regression_model(
            lr_model, X_train_scaled, X_test_scaled, y_train, y_test, "Linear Regression"
        )
        
        # Random Forest Regression (dữ liệu gốc)
        rf_model = ModelFactory.create_regression_models()['Random Forest']
        rf_result = self.evaluator.evaluate_regression_model(
            rf_model, X_train, X_test, y_train, y_test, "Random Forest"
        )
        
        # So sánh kết quả
        self.evaluator.compare_regression_models()
        
        # Lưu kết quả
        self.results_summary['regression'] = {
            'linear_regression': lr_result,
            'random_forest': rf_result
        }
    
    def _create_visualizations(self):
        """
        Tạo các biểu đồ so sánh
        """
        # Visualization cho classification
        self.evaluator.visualize_classification_results()
        
        # Visualization cho regression
        self.evaluator.visualize_regression_results()
    
    def _create_summary(self):
        """
        Tạo tổng kết kết quả
        """
        print("="*50)
        print("           TỔNG KẾT KẾT QUẢ")
        print("="*50)
        
        # Classification summary
        if 'classification' in self.results_summary:
            class_results = self.results_summary['classification']
            dt_acc = class_results['decision_tree']['Accuracy']
            lr_acc = class_results['logistic_regression']['Accuracy']
            best_class = 'Decision Tree' if dt_acc > lr_acc else 'Logistic Regression'
            
            print(f"\n1. CLASSIFICATION:")
            print(f"   Tốt nhất: {best_class} (Accuracy: {max(dt_acc, lr_acc):.3f})")
        
        # Cross-validation summary
        if 'cross_validation' in self.results_summary:
            cv_results = self.results_summary['cross_validation']
            print(f"\n2. CROSS-VALIDATION:")
            print(f"   Random Forest CV: {cv_results['mean']:.3f} ± {cv_results['std']:.3f}")
        
        # Regression summary
        if 'regression' in self.results_summary:
            reg_results = self.results_summary['regression']
            lr_r2 = reg_results['linear_regression']['R²']
            rf_r2 = reg_results['random_forest']['R²']
            best_reg = 'Random Forest' if rf_r2 > lr_r2 else 'Linear Regression'
            r2_improve = (rf_r2 - lr_r2) / lr_r2 * 100 if lr_r2 > 0 else 0
            
            print(f"\n3. REGRESSION:")
            print(f"   Tốt nhất: {best_reg} (R²: {max(lr_r2, rf_r2):.3f})")
            print(f"   R² improvement: {r2_improve:.1f}%")
        
        print(f"\n4. KHUYẾN NGHỊ:")
        print(f"   → Random Forest: Versatile và hiệu quả nhất")
        print(f"   → Phù hợp cho production deployment")
        print("="*50)
    
    def _save_best_models(self):
        """
        Lưu các mô hình tốt nhất
        """
        models_dir = '../models'
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        # Lưu Random Forest Classification
        if 'Random Forest' in self.evaluator.trained_models:
            self.evaluator.save_model(
                'Random Forest', 
                os.path.join(models_dir, 'rf_classifier.pkl')
            )
        
        # Lưu Random Forest Regression
        if 'Random Forest' in self.evaluator.trained_models:
            # Tạo lại RF regression để lưu
            rf_reg = ModelFactory.create_regression_models()['Random Forest']
            self.evaluator.trained_models['RF_Regression'] = rf_reg
            self.evaluator.save_model(
                'RF_Regression',
                os.path.join(models_dir, 'rf_regressor.pkl')
            )
    
    def predict_single_house(self, house_features, model_type='regression'):
        """
        Dự đoán cho một ngôi nhà
        
        Args:
            house_features (dict): Dictionary chứa các đặc trưng của ngôi nhà
            model_type (str): 'classification' hoặc 'regression'
            
        Returns:
            Kết quả dự đoán
        """
        # Chuẩn bị features
        feature_names = self.preprocessor.features
        X_single = pd.DataFrame([house_features])[feature_names]
        
        # Xử lý missing values
        X_single = pd.DataFrame(
            self.preprocessor.imputer.transform(X_single),
            columns=feature_names
        )
        
        if model_type == 'regression':
            # Dự đoán giá
            if 'Random Forest' in self.evaluator.trained_models:
                model = self.evaluator.trained_models['Random Forest']
                price_pred = model.predict(X_single)[0]
                return f"Giá dự đoán: ${price_pred:,.0f}"
        else:
            # Dự đoán nhãn phân loại
            if 'Random Forest' in self.evaluator.trained_models:
                model = self.evaluator.trained_models['Random Forest']
                category_pred = model.predict(X_single)[0]
                categories = {0: 'Thấp', 1: 'Trung bình', 2: 'Cao'}
                return f"Mức giá dự đoán: {categories[category_pred]}"
        
        return "Chưa có mô hình được huấn luyện!"


def main():
    """Hàm main để chạy toàn bộ pipeline"""
    try:
        # Khởi tạo pipeline
        pipeline = HousePricePipeline()
        
        # Chạy toàn bộ quy trình
        pipeline.run_full_pipeline(save_models=True)
        
        # Test prediction cho một ngôi nhà mẫu
        print("\nTEST PREDICTION:")
        sample_house = {
            'OverallQual': 7,
            'GrLivArea': 1500,
            'GarageCars': 2,
            'TotalBsmtSF': 1000,
            'FullBath': 2,
            'YearBuilt': 2000,
            '1stFlrSF': 800,
            'TotRmsAbvGrd': 7
        }
        
        regression_result = pipeline.predict_single_house(sample_house, 'regression')
        classification_result = pipeline.predict_single_house(sample_house, 'classification')
        
        print(f"Ngôi nhà mẫu: {sample_house}")
        print(f"Kết quả regression: {regression_result}")
        print(f"Kết quả classification: {classification_result}")
        
        print("\n Pipeline hoàn thành thành công!")
        
    except Exception as e:
        print(f"Lỗi khi chạy pipeline: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
