# House Price Prediction Project

Dự án Machine Learning dự đoán giá nhà (Regression) sử dụng dataset `data/House_Prices.csv`.
Mục tiêu: xây dựng pipeline tiền xử lý, huấn luyện và lưu mô hình để phục vụ dự đoán giá nhà.

## Cấu trúc dự án

```
data_mining/
├── data/
│   └── House_Prices.csv        # Dataset gốc (bắt buộc để chạy pipeline/demo)
├── demo/
│   └── app.py                  # Streamlit demo app
├── models/                     # Thư mục lưu model được huấn luyện (sẽ tạo khi chạy pipeline)
├── notebooks/
│   └── notebook.ipynb          # Notebook nguồn (khám phá/experiments)
├── src/
│   ├── preprocessing.py        # Tiền xử lý và chuẩn bị feature
│   ├── modeling.py             # Định nghĩa, huấn luyện và lưu model
│   └── predict.py              # Pipeline chạy toàn bộ quá trình
├── requirements.txt            # Thư viện cần thiết
└── README.md                   # Hướng dẫn này
```

## Cách sử dụng

### 1. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 2. Chạy Demo App (Streamlit)

Streamlit app nằm ở `demo/app.py`. Từ thư mục gốc project chạy:

```powershell
streamlit run demo/app.py
```

### 3. Chạy toàn bộ pipeline

Có hai cách chạy pipeline (từ thư mục gốc của project):

```powershell
python src\predict.py
# hoặc
cd src; python predict.py
```

Kết quả: huấn luyện các mô hình (Linear Regression, Random Forest), in báo cáo đánh giá và lưu model tốt nhất vào `models/model.pkl`.

### 4. Chạy nhanh kiểm tra module (manual)

Các module có các đoạn `if __name__ == "__main__"` để test nhanh. Ví dụ:

```powershell
python src\preprocessing.py
python src\modeling.py
python src\predict.py
```

Nếu muốn viết test tự động, có thể thêm `tests/` và dùng `pytest`.

## Demo App Features

- Giao diện Streamlit cho:
  - Dự đoán nhanh (form nhập features)
  - Phân tích dữ liệu (histogram, scatter, boxplot)
  - Chạy phân tích ML hoàn chỉnh (huấn luyện và so sánh mô hình)
- Nếu chưa có `models/model.pkl`, app sẽ gợi ý chạy "Phân tích ML hoàn chỉnh" để huấn luyện và lưu model.

## Các mô hình được sử dụng

Hiện tại project chỉ tập trung vào Regression (dự đoán giá):

- `LinearRegression` (scikit-learn)
- `RandomForestRegressor` (n_estimators=100)

So sánh giữa Linear Regression (sau scaling) và Random Forest (dùng raw features) được thực hiện trong `src/predict.py` và `demo/app.py`.

## Metrics đánh giá (Regression)

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Square Error)
- **R²** (Coefficient of Determination)

## Features chính

Module `src/preprocessing.py` sử dụng các feature sau (mặc định):

- `OverallQual`, `GrLivArea`, `GarageCars`, `TotalBsmtSF`, `FullBath`, `YearBuilt`, `1stFlrSF`, `TotRmsAbvGrd`.

Bạn có thể mở `src/preprocessing.py` để thêm/sửa danh sách features nếu cần.

## Kết quả mong đợi

- Mục tiêu: đạt R² cao và MAE/RMSE thấp trên tập test. Giá trị cụ thể tùy dataset và feature-engineering.

## Ví dụ sử dụng nhanh (Python)

Preprocessing + chạy pipeline từ code:

```python
from src.preprocessing import HousePricePreprocessor
from src.predict import HousePricePipeline

prep = HousePricePreprocessor('data/House_Prices.csv')
prep.explore_data()
X, y = prep.prepare_features()

pipeline = HousePricePipeline('data/House_Prices.csv')
pipeline.run()  # Chạy toàn bộ pipeline (train, đánh giá, lưu model)
```

Để dùng model đã lưu để dự đoán nhanh, chạy Streamlit demo hoặc load `models/model.pkl` bằng `joblib`.

## Dependencies

Các phụ thuộc chính nằm trong `requirements.txt`. Ví dụ cài nhanh:

```powershell
pip install -r requirements.txt
```

Một số package quan trọng: `pandas`, `numpy`, `scikit-learn`, `joblib`, `streamlit`, `matplotlib`, `seaborn`.

## Troubleshooting

- Lỗi thiếu thư viện: chạy `pip install -r requirements.txt`.
- Lỗi không tìm thấy data: đảm bảo `data/House_Prices.csv` tồn tại.
- Nếu Streamlit báo lỗi import, chạy lệnh từ thư mục gốc và đảm bảo Python environment đúng.
- Thay port của Streamlit nếu cần:

```powershell
streamlit run demo/app.py --server.port 8502
```

## Tác giả & License

Project author: repository owner.

License: MIT (xem file `LICENSE`).

---
