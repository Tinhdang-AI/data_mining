# House Price Prediction Project

Dự án Machine Learning dự đoán giá nhà (Regression) sử dụng dataset `data/House_Prices.csv`.
Mục tiêu: xây dựng pipeline tiền xử lý, huấn luyện và lưu mô hình để phục vụ dự đoán giá nhà.

Dự án xây dựng hệ thống dự đoán giá nhà sử dụng Machine Learning, bao gồm phân tích dữ liệu, huấn luyện mô hình và triển khai ứng dụng web.

## 🎯 Mục Tiêu

- **Bài toán**: Regression - Dự đoán giá nhà (SalePrice) 
- **Dữ liệu**: House Prices dataset với 81 đặc trưng
- **Mô hình**: So sánh Linear Regression và Random Forest
- **Đánh giá**: MAE, RMSE, R-squared
- **Triển khai**: Ứng dụng web với Streamlit

## 📁 Cấu Trúc Dự Án

```
data_mining/
├── 📊 data/
│   └── House_Prices.csv          # Dataset gốc (1460 mẫu, 81 cột)
├── 📓 notebooks/
│   └── notebook.ipynb            # Jupyter notebook phân tích và báo cáo
├── 🧠 src/
│   ├── preprocessing.py          # Xử lý và chuẩn bị dữ liệu
│   ├── modeling.py              # Huấn luyện và đánh giá mô hình
│   └── predict.py               # Pipeline hoàn chỉnh
├── 🤖 models/
│   └── model.pkl                # Mô hình đã huấn luyện (Random Forest)
├── 🌐 demo/
│   ├── app.py                   # Ứng dụng Streamlit
│   └── templates/               # Templates giao diện
├── 📄 requirements.txt          # Dependencies
├── 📖 report.pdf               # Báo cáo chi tiết
└── 📝 README.md                # File này
```

## 🔄 Luồng Hoạt Động Chính

### 1. 📈 Data Understanding & Preparation (`preprocessing.py`)

**Class: `HousePricePreprocessor`**

```python
# Khởi tạo
prep = HousePricePreprocessor(data_path="data/House_Prices.csv")

# Bước 1: Khám phá dữ liệu
prep.explore_data()
# - Shape: (1460, 81)
# - Missing values: Xử lý bằng median imputation
# - Duplicates: Loại bỏ
# - Avg price: ~$180,000

# Bước 2: Chuẩn bị features
X, y = prep.prepare_features()
# Features được chọn: 8 đặc trưng quan trọng nhất
# - OverallQual: Chất lượng tổng thể (1-10)
# - GrLivArea: Diện tích sinh hoạt (sq ft)
# - GarageCars: Số xe garage
# - TotalBsmtSF: Diện tích tầng hầm
# - FullBath: Số phòng tắm đầy đủ
# - YearBuilt: Năm xây dựng
# - 1stFlrSF: Diện tích tầng 1
# - TotRmsAbvGrd: Tổng số phòng trên mặt đất

# Bước 3: Chia dữ liệu
X_train, X_test, y_train, y_test = prep.split_data(X, y, test_size=0.2)
# Train: 80% (1168 mẫu)
# Test: 20% (292 mẫu)

# Bước 4: Chuẩn hóa (cho Linear Regression)
X_train_scaled, X_test_scaled = prep.scale_features(X_train, X_test)
```

### 2. 🤖 Model Training & Evaluation (`modeling.py`)

**Classes: `ModelEvaluator`, `ModelFactory`**

```python
# Khởi tạo
evaluator = ModelEvaluator()
models = ModelFactory.create_models()

# Model 1: Linear Regression (trên dữ liệu scaled)
lr_result = evaluator.evaluate_model(
    models['Linear Regression'], 
    X_train_scaled, X_test_scaled, y_train, y_test, 
    "Linear Regression"
)

# Model 2: Random Forest (trên dữ liệu gốc)
rf_result = evaluator.evaluate_model(
    models['Random Forest'], 
    X_train, X_test, y_train, y_test, 
    "Random Forest"
)

# So sánh kết quả
results_df = evaluator.compare_models()
```

**Kết quả điển hình:**
| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| Linear Regression | $24,000 | $35,000 | 0.750 |
| Random Forest | $18,000 | $28,000 | 0.850+ |

### 3. 🔄 Complete Pipeline (`predict.py`)

**Class: `HousePricePipeline`**

Pipeline tự động hóa toàn bộ quy trình:

```python
pipeline = HousePricePipeline()
pipeline.run()
# 1. Xử lý dữ liệu
# 2. Huấn luyện mô hình  
# 3. Đánh giá và so sánh
# 4. Lưu model tốt nhất (.pkl)
```

### 4. 📊 Interactive Analysis (`notebook.ipynb`)

Jupyter Notebook thực hiện phân tích chi tiết theo quy trình CRISP-DM:

1. **Business Understanding**: Định nghĩa bài toán
2. **Data Understanding**: Khám phá và thống kê dữ liệu  
3. **Data Preparation**: Xử lý và chuẩn bị features
4. **Modeling**: Huấn luyện và tinh chỉnh mô hình
5. **Evaluation**: So sánh hiệu suất, trực quan hóa
6. **Deployment**: Chuẩn bị cho triển khai

### 5. 🌐 Web Application (`demo/app.py`)

Ứng dụng Streamlit với 3 chế độ:

**Chế độ 1: Dự đoán nhanh**
- Form nhập thông tin nhà
- Dự đoán giá real-time
- Hiển thị kết quả trực quan

**Chế độ 2: Phân tích dữ liệu**  
- Thống kê mô tả
- Biểu đồ phân phối giá
- Phân tích correlation

**Chế độ 3: ML Analysis hoàn chỉnh**
- Chạy pipeline đầy đủ
- So sánh mô hình
- Export kết quả

## 🚀 Hướng Dẫn Chạy Dự Án

### 1. Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

### 2. Chạy Pipeline Hoàn Chỉnh

Có hai cách chạy pipeline (từ thư mục gốc của project):

```powershell
python src\predict.py
# hoặc
cd src; python predict.py
```

### 3. Chạy Jupyter Notebook

```bash
jupyter notebook notebooks/notebook.ipynb
```

### 4. Chạy Web App

```bash
cd demo
streamlit run app.py
```

## 📊 Kết Quả Chính

### Hiệu Suất Mô Hình
- **Random Forest**: R² ≈ 0.891, MAE ≈ $25,000
- **Linear Regression**: R² ≈ 0.795, MAE ≈ $19,000
- **Winner**: Random Forest (tốt hơn ~10-15%)

### Features Quan Trọng Nhất
1. **OverallQual** (40%): Chất lượng tổng thể
2. **GrLivArea** (25%): Diện tích sinh hoạt  
3. **GarageCars** (15%): Số xe garage
4. **YearBuilt** (10%): Năm xây dựng
5. Các features khác (10%)

## 🛠️ Tech Stack

- **Data Processing**: pandas, numpy, scikit-learn
- **Machine Learning**: LinearRegression, RandomForestRegressor
- **Visualization**: matplotlib, seaborn
- **Web Framework**: Streamlit
- **Model Persistence**: joblib
- **Development**: Jupyter Notebook

## 📝 Ghi Chú Kỹ Thuật

### Data Preprocessing
- **Missing Values**: SimpleImputer với strategy='median'
- **Feature Selection**: 8/81 features quan trọng nhất
- **Scaling**: StandardScaler cho Linear Regression
- **Train/Test Split**: 80/20 với random_state=42

### Model Configuration
- **Linear Regression**: Default parameters, requires scaled data
- **Random Forest**: n_estimators=100, random_state=42, works on original data

### Evaluation Metrics
- **MAE**: Mean Absolute Error (dễ hiểu, đơn vị $)
- **RMSE**: Root Mean Square Error (penalize large errors)  
- **R²**: Coefficient of Determination (0-1, higher is better)

🥳 Trải nghiệm mô hình trực tiếp
👉 Demo App:
https://tinhdang-ai-data-mining-demoapp-develop-ac4hja.streamlit.app/

## 🔮 Tương Lai

### Planned Improvements
- [ ] Feature Engineering nâng cao
- [ ] Hyperparameter tuning với GridSearch
- [ ] Thêm mô hình: XGBoost, LightGBM
- [ ] Cross-validation robust hơn
- [ ] Deploy lên cloud (Heroku/AWS)
- [ ] API endpoints cho mobile app

### Advanced Features
- [ ] Real-time price tracking
- [ ] Market trend analysis  
- [ ] Location-based pricing
- [ ] Image recognition cho house features


**Dự Án Data Mining - House Price Prediction**
- Framework: CRISP-DM methodology
- Evaluation: Statistical significance testing
- Deployment: Production-ready Streamlit app

---

*Dự án được phát triển cho môn Data Mining, minh họa quy trình hoàn chỉnh từ Raw Data đến Production Application.*
