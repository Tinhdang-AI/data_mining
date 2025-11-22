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

