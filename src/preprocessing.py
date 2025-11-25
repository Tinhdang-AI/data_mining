
import pandas as pd # Import thư viện xử lý dữ liệu dạng bảng
import numpy as np # Import thư viện xử lý tính toán số học
from sklearn.impute import SimpleImputer # Import thư viện xử lý dữ liệu thiếu
from sklearn.preprocessing import StandardScaler # Import thư viện chuẩn hóa dữ liệu


# Module này chứa các hàm để:
# - Đọc và khám phá dữ liệu
# - Tạo nhãn phân loại
# - Xử lý missing values
# - Chuẩn bị features cho machine learning



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
