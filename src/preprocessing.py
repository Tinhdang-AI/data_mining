import pandas as pd
import numpy as np
import os  # <--- Quan trọng: Phải có thư viện này để xử lý đường dẫn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Module này chứa các hàm để:
# - Đọc và khám phá dữ liệu
# - Tạo nhãn phân loại
# - Xử lý missing values
# - Chuẩn bị features cho machine learning

class HousePricePreprocessor:
    """Class xử lý dữ liệu House Prices cho bài toán Regression"""
    
    def __init__(self, data_path='data/House_Prices.csv'):
        """
        Khởi tạo preprocessor
        
        Args:
            data_path (str): Đường dẫn đến file dữ liệu CSV
        """
        self.data_path = data_path
        self.df = None
        self.features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF',
                        'FullBath', 'YearBuilt', '1stFlrSF', 'TotRmsAbvGrd']
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Đọc dữ liệu từ file CSV"""
        # Kiểm tra xem file có tồn tại không trước khi đọc
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Khong tim thay file du lieu: {self.data_path}")
            
        self.df = pd.read_csv(self.data_path)
        return self.df
    
    def explore_data(self):
        """Khám phá cơ bản về dữ liệu"""
        if self.df is None:
            self.load_data()
            
        info = {
            'shape': self.df.shape,
            'missing_values': self.df.isnull().sum().sum(),
            'duplicates': self.df.duplicated().sum(),
            'avg_price': self.df['SalePrice'].mean()
        }
        
        print(f"Kich thuoc: {info['shape']}")
        print(f"Missing values: {info['missing_values']}")
        print(f"Du lieu trung lap: {info['duplicates']}")
        print(f"Gia trung binh: ${info['avg_price']:,.0f}")
        return info
    
    def prepare_features(self):
        """
        Chuẩn bị features và target
        Returns:
            X: Features DataFrame
            y: Target Series (SalePrice)
        """
        if self.df is None:
            self.load_data()
        
        # Xóa dữ liệu trùng lặp trước khi xử lý
        self.df = self.df.drop_duplicates(keep='first')
        
        # Xử lý missing values
        X = pd.DataFrame(
            self.imputer.fit_transform(self.df[self.features]),
            columns=self.features
        )
        
        y = self.df['SalePrice']   # Target duy nhất là Giá nhà
        
        print(f"Features shape: {X.shape}")
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Chia dữ liệu thành tập train và test"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Train set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        """Chuẩn hóa dữ liệu (quan trọng cho Linear Regression)"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled

if __name__ == "__main__":
    # Test nhanh module
    try:
        prep = HousePricePreprocessor()
        prep.explore_data()
        X, y = prep.prepare_features()
        print("Test Preprocessing thanh cong!")
    except Exception as e:
        print(f"Van con loi: {e}")