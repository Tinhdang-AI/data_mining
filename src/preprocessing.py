
import pandas as pd # Import thư viện xử lý dữ liệu dạng bảng
import numpy as np # Import thư viện xử lý tính toán số học
from sklearn.impute import SimpleImputer # Import thư viện xử lý dữ liệu thiếu
from sklearn.preprocessing import StandardScaler # Import thư viện chuẩn hóa dữ liệu

# 1. ĐỊNH NGHĨA CÁC CỘT QUAN TRỌNG (CONSTANTS)
FEATURE_COLS = [
    'OverallQual',    # Chất lượng tổng thể
    'GrLivArea',      # Diện tích sống
    'GarageCars',     # Số chỗ để xe
    'TotalBsmtSF',    # Diện tích hầm
    'FullBath',       # Số phòng tắm
    'YearBuilt',      # Năm xây dựng
    '1stFlrSF',       # Diện tích tầng 1
    'TotRmsAbvGrd'    # Tổng số phòng
]

TARGET_COL = 'SalePrice' # Cột mục tiêu dự đoán

file_path = 'data/House_Prices.csv'  # Đường dẫn file dữ liệu
def load_data(file_path):
    """
    Hàm đọc dữ liệu từ file CSV.
    Input: file_path (str) - Đường dẫn đến file csv
    """
    try:
        df = pd.read_csv(file_path)
        print(f"--> Da tai du lieu thanh cong: {df.shape}")
        return df
    except Exception as e:
        print(f"Loi khi doc file: {e}")
        return None
print(load_data(file_path).head())
def clean_data(df):
    """
    Hàm xử lý sơ bộ: Chọn cột quan trọng và điền dữ liệu thiếu.
    """
    # Chỉ giữ lại các cột cần thiết có trong dữ liệu
    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_cols].copy()
    
    # Xử lý Missing Values bằng Median
    # Lưu ý: Giữ nguyên tên cột để tạo DataFrame (dễ nhìn hơn numpy array)
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=available_cols)
    
    # Lấy cột Target nếu có (dùng cho tập Train)
    y = None
    if TARGET_COL in df.columns:
        y = df[TARGET_COL]
        
    return X_imputed, y
print(clean_data(load_data(file_path))[0].head())
def scale_data(X):
    """
    Hàm chuẩn hóa dữ liệu dùng StandardScaler.
    Trả về: X_scaled (DataFrame) và scaler (để lưu lại).
    """
    scaler = StandardScaler()
    # Fit và transform dữ liệu
    X_np = scaler.fit_transform(X)
    
    # Chuyển ngược lại thành DataFrame để giữ tên cột (tốt cho việc vẽ biểu đồ Feature Importance)
    X_scaled = pd.DataFrame(X_np, columns=X.columns, index=X.index)
    
    return X_scaled, scaler
print(scale_data(clean_data(load_data(file_path))[0])[0].head())
def preprocess_pipeline(file_path):
    """
    Hàm tổng hợp chạy toàn bộ quy trình từ A-Z.
    Dành cho người làm Modeling (src/modeling.py) gọi 1 lệnh là xong.
    """
    # Bước 1: Đọc dữ liệu
    df = load_data(file_path)
    
    if df is not None:
        # Bước 2: Làm sạch & Chọn lọc
        X, y = clean_data(df)
        
        # Bước 3: Chuẩn hóa
        X_scaled, scaler = scale_data(X)
        
        print("--> Xử lý dữ liệu hoàn tất (Pipeline)!")
        return X_scaled, y, scaler
    else:
        return None, None, None

############# HƯỚNG DẪN SỬ DỤNG ##############
# Dành cho Modeling (src/modeling.py):
# from src.preprocessing import preprocess_pipeline
# X, y, scaler = preprocess_pipeline('data/House_Prices.csv')

# Dành cho App (src/predict.py):
# from src.preprocessing import FEATURE_COLS
# ... (Giữ nguyên các import và hàm bên trên) ...

"""
Module xử lý dữ liệu và feature engineering cho dự án House Prices

Module này chứa các hàm để:
- Đọc và khám phá dữ liệu
- Tạo nhãn phân loại
- Xử lý missing values
- Chuẩn bị features cho machine learning
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class HousePricePreprocessor:
    """Class xử lý dữ liệu House Prices"""
    
    def __init__(self, data_path='../data/House_Prices.csv'):
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
        """
        Đọc dữ liệu từ file CSV
        
        Returns:
            pd.DataFrame: DataFrame chứa dữ liệu
        """
        self.df = pd.read_csv(self.data_path)
        return self.df
    
    def explore_data(self):
        """
        Khám phá cơ bản về dữ liệu
        
        Returns:
            dict: Dictionary chứa thông tin cơ bản về dataset
        """
        if self.df is None:
            self.load_data()
            
        info = {
            'shape': self.df.shape,
            'missing_values': self.df.isnull().sum().sum(),
            'avg_price': self.df['SalePrice'].mean()
        }
        
        print(f"Kích thước: {info['shape']}")
        print(f"Missing values: {info['missing_values']}")
        print(f"Giá trung bình: ${info['avg_price']:,.0f}")
        
        return info
    
    @staticmethod
    def price_category(price):
        """
        Phân loại giá nhà thành 3 mức độ
        
        Args:
            price (float): Giá nhà
            
        Returns:
            int: Nhãn phân loại (0: thấp, 1: trung bình, 2: cao)
        """
        if price < 150000:
            return 0    # Giá thấp
        elif price < 250000:
            return 1    # Giá trung bình
        else:
            return 2    # Giá cao
    
    def create_price_categories(self):
        """
        Tạo nhãn phân loại cho giá nhà
        
        Returns:
            pd.Series: Series chứa nhãn phân loại
        """
        if self.df is None:
            self.load_data()
            
        self.df['Category'] = self.df['SalePrice'].apply(self.price_category)
        print("Phân bố nhãn phân loại:")
        print(self.df['Category'].value_counts())
        
        return self.df['Category']
    
    def prepare_features(self):
        """
        Chuẩn bị features và xử lý missing values
        
        Returns:
            tuple: (X_processed, y_class, y_reg) - Features đã xử lý và target variables
        """
        if self.df is None:
            self.load_data()
            
        if 'Category' not in self.df.columns:
            self.create_price_categories()
        
        # Xử lý missing values
        X = pd.DataFrame(
            self.imputer.fit_transform(self.df[self.features]),
            columns=self.features
        )
        
        y_class = self.df['Category']  # Target cho classification
        y_reg = self.df['SalePrice']   # Target cho regression
        
        print(f"Features shape: {X.shape}")
        
        return X, y_class, y_reg
    
    def split_data(self, X, y_class, y_reg, test_size=0.2, random_state=42):
        """
        Chia dữ liệu thành tập train và test
        
        Args:
            X (pd.DataFrame): Features
            y_class (pd.Series): Target cho classification
            y_reg (pd.Series): Target cho regression
            test_size (float): Tỷ lệ dữ liệu test
            random_state (int): Random seed
            
        Returns:
            tuple: Dữ liệu đã chia cho cả classification và regression
        """
        # Chia dữ liệu cho classification
        X_train, X_test, y_train_class, y_test_class = train_test_split(
            X, y_class, test_size=test_size, random_state=random_state
        )
        
        # Chia dữ liệu cho regression (cùng random_state để đảm bảo consistency)
        _, _, y_train_reg, y_test_reg = train_test_split(
            X, y_reg, test_size=test_size, random_state=random_state
        )
        
        print(f"Train: {X_train.shape}, Test: {X_test.shape}")
        
        return (X_train, X_test, y_train_class, y_test_class, 
                y_train_reg, y_test_reg)
    
    def scale_features(self, X_train, X_test):
        """
        Chuẩn hóa features cho các mô hình cần thiết
        
        Args:
            X_train (pd.DataFrame): Dữ liệu training
            X_test (pd.DataFrame): Dữ liệu test
            
        Returns:
            tuple: (X_train_scaled, X_test_scaled) - Dữ liệu đã chuẩn hóa
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = HousePricePreprocessor()
    preprocessor.explore_data()
    X, y_class, y_reg = preprocessor.prepare_features()
    
    # Chia dữ liệu
    data_splits = preprocessor.split_data(X, y_class, y_reg)
    X_train, X_test = data_splits[0], data_splits[1]
    
    # Chuẩn hóa dữ liệu
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    print("Preprocessing hoàn thành!")
