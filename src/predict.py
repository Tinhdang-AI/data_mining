import sys
import os
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import HousePricePreprocessor
from modeling import ModelEvaluator, ModelFactory

class HousePricePipeline:
    """Pipeline hoan chinh cho bai toan du doan gia nha"""
    
    def __init__(self, data_path='data/House_Prices.csv'):
        self.preprocessor = HousePricePreprocessor(data_path)
        self.evaluator = ModelEvaluator()
    
    def run(self):
        print("="*40)
        print("   BAT DAU PIPELINE DU DOAN GIA NHA")
        print("="*40)
        
        # 1. Preprocessing
        print("\n1. Dang xu ly du lieu...")
        self.preprocessor.explore_data()
        X, y = self.preprocessor.prepare_features()
        
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(X, y)
        X_train_scaled, X_test_scaled = self.preprocessor.scale_features(X_train, X_test)
        
        # 2. Modeling
        print("\n2. Dang huan luyen mo hinh...")
        models = ModelFactory.create_models()
        
        # Model 1: Linear Regression
        self.evaluator.evaluate_model(
            models['Linear Regression'], 
            X_train_scaled, X_test_scaled, y_train, y_test, 
            "Linear Regression"
        )
        
        # Model 2: Random Forest
        self.evaluator.evaluate_model(
            models['Random Forest'], 
            X_train, X_test, y_train, y_test, 
            "Random Forest"
        )
        
        # 3. Tong ket
        print("\n3. Tong ket ket qua:")
        results = self.evaluator.compare_models()
        print(results)
        
        # 4. Luu Model
        print("\n4. Luu mo hinh tot nhat (Random Forest)...")
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        models_dir = os.path.join(project_root, 'models')
        
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            
        model_path = os.path.join(models_dir, 'model.pkl')
        self.evaluator.save_model('Random Forest', model_path)
        
        print("\ PIPELINE HOAN TAT! SAN SANG CHAY APP.")

if __name__ == "__main__":
    try:
        pipeline = HousePricePipeline()
        pipeline.run()
    except Exception as e:
        print(f" Co loi xay ra: {e}")