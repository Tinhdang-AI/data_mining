import pandas as pd
import numpy as np
import sys
import os
import joblib

# Import cac thuat toan Regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class ModelEvaluator:
    """Class chuyen danh gia mo hinh Regression"""
    
    def __init__(self):
        self.results = []        # Danh sach luu ket qua
        self.trained_models = {} # Dictionary luu mo hinh
    
    def evaluate_model(self, model, X_train, X_test, y_train, y_test, model_name):
        """Huan luyen va danh gia mot mo hinh"""
        # Huan luyen
        model.fit(X_train, y_train)
        
        # Du doan
        y_pred = model.predict(X_test)
        
        # Tinh toan metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n--- Danh gia mo hinh: {model_name} ---")
        print(f"MAE  (Sai so tuyet doi): ${mae:,.0f}")
        print(f"RMSE (Sai so binh phuong): ${rmse:,.0f}")
        print(f"R2   (Do chinh xac): {r2:.3f}")
        
        # Luu mo hinh vao bo nho (RAM)
        self.trained_models[model_name] = model
        
        # Luu ket qua vao danh sach de so sanh
        result = {'Model': model_name, 'MAE': mae, 'RMSE': rmse, 'R²': r2}
        self.results.append(result)
        
        return result

    # --- ĐÂY LÀ HÀM QUAN TRỌNG MÀ EM ĐANG THIẾU ---
    def compare_models(self):
        """Xuat bang so sanh cac mo hinh da chay"""
        if not self.results:
            print("Chua co ket qua nao de so sanh!")
            return None
        return pd.DataFrame(self.results)

    def save_model(self, model_name, filepath):
        """Luu model ra file .pkl"""
        if model_name in self.trained_models:
            joblib.dump(self.trained_models[model_name], filepath)
            print(f"-> Da luu file model tai: {filepath}")
        else:
            print(f"Loi: Khong tim thay mo hinh {model_name} de luu!")

class ModelFactory:
    """Kho chua cac mo hinh"""
    @staticmethod
    def create_models():
        return {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }

if __name__ == "__main__":
    print("--- Day la file modeling.py (Phien ban Day Du) ---")
    # Code test nhanh neu can
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from preprocessing import HousePricePreprocessor
        prep = HousePricePreprocessor()
        X, y = prep.prepare_features()
        print("✅ Kiem tra du lieu thanh cong.")
    except:
        pass