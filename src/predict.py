"""
predict.py

Script đơn giản để huấn luyện và dự đoán giá nhà từ bộ dữ liệu `House_Prices.csv`.

Mục đích của file:
- Cung cấp hai thao tác chính qua CLI: `--train` để huấn luyện và lưu model, và
    `--predict` để nạp model đã lưu và sinh dự đoán từ một file đầu vào hoặc từ
    một mẫu dữ liệu có sẵn.

Thiết kế ngắn gọn:
- Sử dụng các cột số (numeric) từ dataset để làm features (đơn giản hóa), bỏ cột
    `Id` và dùng `SalePrice` làm target.
- Xử lý giá trị thiếu bằng cách điền median tại mỗi cột numeric.
- Mô hình mặc định: `RandomForestRegressor` (scikit-learn).

Lưu ý/Phạm vi:
- Đây là script minh họa, phù hợp cho thử nghiệm nhanh. Để dùng trong sản xuất
    cần: chọn đặc trưng, mã hóa categorical, scale/normalize, xử lý giá trị thiếu
    tinh vi hơn, và kiểm tra phân phối dữ liệu.
"""

import argparse
import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Project root (one level above `src`) so data lives at the repository root `data/`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Đường dẫn file dữ liệu mặc định (ở thư mục project root)
DATA_PATH = PROJECT_ROOT / 'data' / 'House_Prices.csv'
# Thư mục để lưu model đã huấn luyện (project root)
MODEL_DIR = PROJECT_ROOT / 'models'
# Tên file model (joblib/pickle)
MODEL_FILE = MODEL_DIR / 'model.pkl'


def load_data(path=DATA_PATH):
    """
    Đọc file CSV và trả về DataFrame.

    Tham số:
    - path: đường dẫn tới file CSV (mặc định dùng `DATA_PATH`).

    Trả về:
    - `pandas.DataFrame` chứa toàn bộ dữ liệu thô.
    """
    # Không xử lý gì thêm tại đây — việc tiền xử lý thực hiện trong `preprocess`.
    df = pd.read_csv(path)
    return df


def preprocess(df, target='SalePrice'):
    """
    Tiền xử lý đơn giản cho dữ liệu thô.

    Bước thực hiện:
    1. Chỉ giữ cột có kiểu số (numeric). Lý do: tránh phải mã hóa categorical trong
       ví dụ nhanh này.
    2. Bỏ cột `Id` nếu có (không mang thông tin dự đoán có ích).
    3. Nếu tồn tại cột target (mặc định `SalePrice`), tách `X` và `y`.

    Trả về:
    - Nếu có target: `(X, y)` với `X` là DataFrame features, `y` là Series target.
    - Nếu không có target: `(numeric, None)` — chỉ trả về features.
    """
    # Chọn các cột numeric để tránh xử lý categorical phức tạp trong ví dụ
    numeric = df.select_dtypes(include=[np.number]).copy()

    # Cột Id thường là identifier, không nên dùng làm feature
    if 'Id' in numeric.columns:
        numeric = numeric.drop(columns=['Id'])

    # Nếu tồn tại cột target, tách dataset thành X, y
    if target in numeric.columns:
        X = numeric.drop(columns=[target])
        y = numeric[target]
        return X, y
    else:
        # Trường hợp dùng file đầu vào không có target (chỉ predict)
        return numeric, None


def train_and_save(df):
    """
    Huấn luyện một mô hình RandomForest đơn giản và lưu model + dự đoán.

    Hành động thực hiện:
    - Tiền xử lý (gọi `preprocess`).
    - Điền giá trị thiếu bằng median của từng cột numeric.
    - Chia dữ liệu thành train/test (tỷ lệ test 20%).
    - Huấn luyện `RandomForestRegressor` với 100 cây.
    - Lưu model bằng `joblib` vào `MODEL_FILE`.
    - Sinh dự đoán trên test set, lưu kết quả (features + giá trị thực + dự đoán)
      vào `predictions/predictions.csv`.
    - In RMSE trên test set để tham khảo.

    Lưu ý: Các lựa chọn ở đây (RandomForest, fillna bằng median, giữ numeric) là
    để đơn giản hóa demo. Trong ứng dụng thực tế nên làm cross-validation,
    chọn hyperparameters, và xử lý feature engineering.
    """
    X, y = preprocess(df)
    if y is None:
        # Nếu không có cột target thì không thể huấn luyện
        raise ValueError('Target column SalePrice not found in dataset')

    # Xử lý giá trị thiếu: điền median cho mỗi cột numeric
    X = X.fillna(X.median())

    # Tách tập huấn luyện và kiểm thử
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Khởi tạo và huấn luyện mô hình
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Dự đoán trên tập kiểm thử và tính RMSE
    preds = model.predict(X_test)
    # Compute RMSE in a way that is compatible with older/newer scikit-learn:
    # mean_squared_error(..., squared=False) isn't available in some versions,
    # so compute MSE then take square root.
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)

    # Tạo thư mục lưu model nếu chưa tồn tại, rồi lưu model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)

    # Thông tin nhanh cho người dùng
    print(f'Model saved to {MODEL_FILE}')
    print(f'RMSE on test split: {rmse:.2f}')


def predict_from_model(input_csv=None):
        """
        Tải model đã lưu và sinh dự đoán.

        Tham số:
        - input_csv: (tùy chọn) đường dẫn tới file CSV chứa dữ liệu đầu vào để dự đoán.
            Nếu không cung cấp, hàm sẽ lấy một mẫu nhỏ từ dataset gốc để minh họa.

        Hành vi:
        - Kiểm tra xem file model tồn tại; nếu không thì báo lỗi và yêu cầu chạy `--train`.
        - Nếu có `input_csv`, đọc file đó, tiền xử lý tương tự (chỉ numeric, drop Id),
            điền median, rồi predict và lưu kết quả.
        - Nếu không có `input_csv`, sẽ nạp dataset gốc, lấy một sample nhỏ để dự đoán
            (tránh in ra quá nhiều hàng) và lưu kết quả.
        """
        if not MODEL_FILE.exists():
                raise FileNotFoundError(f'Model file not found at {MODEL_FILE}. Run --train first.')

        # Nạp model từ đĩa
        model = joblib.load(MODEL_FILE)

        if input_csv is None:
                # Dự đoán trên một mẫu nhỏ của dataset gốc để demo
                df = load_data()
                X, y = preprocess(df)
                X = X.fillna(X.median())
                # Lấy tối đa 50 dòng để chạy nhanh
                sample = X.sample(n=min(50, len(X)), random_state=42)
                preds = model.predict(sample)
                out = sample.copy()
                out['PredictedSalePrice'] = preds
        else:
                # Dự đoán từ file input do người dùng cung cấp
                df = pd.read_csv(input_csv)
                X, _ = preprocess(df)
                X = X.fillna(X.median())
                preds = model.predict(X)
                out = X.copy()
                out['PredictedSalePrice'] = preds


def main():
    parser = argparse.ArgumentParser(description='Train or predict house prices')
    parser.add_argument('--train', action='store_true', help='Train a model on the dataset and save it')
    parser.add_argument('--predict', action='store_true', help='Load saved model and make predictions')
    parser.add_argument('--input', type=str, help='Input CSV file for prediction (optional)')
    args = parser.parse_args()

    if args.train:
        if not DATA_PATH.exists():
            raise FileNotFoundError(f'Data file not found at {DATA_PATH}')
        df = load_data()
        train_and_save(df)

    if args.predict:
        predict_from_model(args.input)

    if not args.train and not args.predict:
        parser.print_help()


if __name__ == '__main__':
    main()
