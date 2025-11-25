# House Price Prediction Project 

Dự án Machine Learning dự đoán giá nhà sử dụng dataset House Prices với các thuật toán Classification và Regression.

## Cấu trúc dự án

```
data_mining/
├── data/
│   └── House_Prices.csv          # Dataset gốc
├── demo/
│   ├── app.py                   # Streamlit demo app
│   └── README_DEMO.md           # Hướng dẫn demo
├── models/
│   ├── model.pkl                # Model đã train (hiện tại)
│
├── notebooks/
│   └── notebook.ipynb           # Jupyter notebook gốc
├── src/
│   ├── preprocessing.py         # Xử lý dữ liệu và feature engineering
│   ├── modeling.py              # Định nghĩa và huấn luyện các mô hình
│   └── predict.py               # Pipeline hoàn chỉnh và prediction
├── requirements.txt             # Danh sách thư viện cần thiết
├── test_modules.py              # Script test các modules
└── README.md                    # File hướng dẫn này
```

##  Cách sử dụng

### 1. Cài đặt thư viện

```bash
pip install -r requirements.txt
```
### 2. Chạy Demo App (Streamlit) 

```bash

# chạy trực tiếp
streamlit run app.py
```

### 3. Chạy toàn bộ pipeline

```bash
cd src
python predict.py
```

### 4. Test các modules

```bash
python test_modules.py
```

## Demo App Features

### **Streamlit Web Interface**
- ** Dự đoán nhanh**: Form tương tác để nhập thông tin nhà và dự đoán giá
- ** Phân tích dữ liệu**: Visualizations với Plotly (histograms, scatter plots, box plots)
- ** ML Pipeline**: Chạy toàn bộ quy trình machine learning với real-time results

### **Interactive Controls**
- Sliders và number inputs cho các features
- Real-time prediction khi thay đổi parameters
- Multiple visualization modes
- Responsive design với sidebar navigation

## Các mô hình được sử dụng

### Classification Models (Phân loại mức giá)
- **Decision Tree**: Cây quyết định với max_depth=10
- **Logistic Regression**: Hồi quy logistic với max_iter=1000
- **Random Forest**: 100 cây quyết định

### Regression Models (Dự đoán giá cụ thể)
- **Linear Regression**: Hồi quy tuyến tính
- **Random Forest**: 100 cây hồi quy

##  Metrics đánh giá

### Classification Metrics
- **Accuracy**: Tỷ lệ dự đoán đúng
- **Precision**: Độ chính xác của dự đoán positive
- **Recall**: Khả năng tìm ra các mẫu positive
- **F1-Score**: Trung bình điều hòa của Precision và Recall

### Regression Metrics
- **MAE** (Mean Absolute Error): Sai số tuyệt đối trung bình
- **RMSE** (Root Mean Square Error): Căn bậc 2 của MSE
- **R²** (Coefficient of Determination): Hệ số xác định (0-1)

## Features sử dụng

1. **OverallQual**: Chất lượng tổng thể (1-10)
2. **GrLivArea**: Diện tích sống trên mặt đất (sq ft)
3. **GarageCars**: Số xe có thể đỗ trong garage
4. **TotalBsmtSF**: Tổng diện tích tầng hầm (sq ft)
5. **FullBath**: Số phòng tắm đầy đủ tiện nghi
6. **YearBuilt**: Năm xây dựng
7. **1stFlrSF**: Diện tích tầng 1 (sq ft)
8. **TotRmsAbvGrd**: Tổng số phòng trên mặt đất

## Kết quả mong đợi

- **Classification Accuracy**: > 80%
- **Regression R²**: > 0.7
- **Cross-validation stability**: Độ lệch chuẩn thấp


## Usage Examples

### Sử dụng từng module riêng lẻ

#### Preprocessing (Xử lý dữ liệu)
```python
from src.preprocessing import HousePricePreprocessor

preprocessor = HousePricePreprocessor('./data/House_Prices.csv')
preprocessor.explore_data()
X, y_class, y_reg = preprocessor.prepare_features()
```

#### Modeling (Huấn luyện mô hình)
```python
from src.modeling import ModelEvaluator, ModelFactory

evaluator = ModelEvaluator()
class_models = ModelFactory.create_classification_models()

dt_result = evaluator.evaluate_classification_model(
    class_models['Decision Tree'], 
    X_train, X_test, y_train, y_test, 
    "Decision Tree"
)
```

#### Prediction (Dự đoán)
```python
from src.predict import HousePricePipeline

pipeline = HousePricePipeline('./data/House_Prices.csv')
pipeline.run_full_pipeline()

# Dự đoán cho một ngôi nhà
sample_house = {
    'OverallQual': 7, 'GrLivArea': 1500,
    'GarageCars': 2, 'TotalBsmtSF': 1000,
    'FullBath': 2, 'YearBuilt': 2000,
    '1stFlrSF': 800, 'TotRmsAbvGrd': 7
}

result = pipeline.predict_single_house(sample_house, 'regression')
```

## Dependencies

```txt
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
seaborn>=0.11.0
joblib>=1.0.0
streamlit>=1.28.0    # For demo app
plotly>=5.15.0       # For interactive visualizations
```

## Troubleshooting

### Lỗi thiếu thư viện
```bash
pip install pandas numpy matplotlib scikit-learn seaborn joblib streamlit plotly
```

### Lỗi không tìm thấy file data
- Đảm bảo file `House_Prices.csv` có trong thư mục `data/`
- Demo app vẫn chạy được mà không có data (chế độ prediction only)

### Lỗi import module
- Đảm bảo chạy từ thư mục gốc của project
- Sử dụng các script launcher (`run_demo.py`, `run_pipeline.py`)

### Streamlit app không mở
```bash
# Thử port khác nếu 8501 bị chiếm
streamlit run demo/app.py --server.port 8502
```

## Highlights

✅ **Modular Architecture**: Tách biệt preprocessing, modeling, và prediction  
✅ **Interactive Demo**: Streamlit app với real-time predictions  
✅ **Comprehensive Testing**: Test scripts cho từng module  
✅ **Easy Deployment**: One-command launchers  
✅ **Rich Visualizations**: Plotly charts trong demo app  
✅ **Professional Structure**: Theo chuẩn ML projects  

## Tác giả

Dự án được tạo từ Jupyter Notebook và chia thành cấu trúc modular để dễ bảo trì, mở rộng và deployment.

## License

MIT License - Sử dụng tự do cho mục đích học tập và nghiên cứu.

---
