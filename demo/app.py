"""
Streamlit Demo App cho House Price Prediction

Ứng dụng web interactive cho phép:
- Upload dữ liệu CSV
- Chọn features và mô hình
- Dự đoán giá nhà real-time
- Visualize kết quả (sử dụng matplotlib thay vì plotly để tránh lỗi pyarrow)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib style
plt.style.use('default')
sns.set_palette("husl")

# Thêm src path để import modules
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.append(src_path)

try:
    # Import với absolute path
    import sys
    import os
    
    # Thêm đường dẫn src vào sys.path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    src_dir = os.path.join(parent_dir, 'src')
    
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    # Import modules từ src
    from preprocessing import HousePricePreprocessor
    from modeling import ModelEvaluator, ModelFactory
    from predict import HousePricePipeline
    
    MODULES_IMPORTED = True
    
except ImportError as e:
    st.error(f"Không thể import modules từ src/: {e}")
    st.info("Vẫn có thể chạy ở chế độ cơ bản mà không cần modules từ src/")
    MODULES_IMPORTED = False

# Cấu hình trang
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.prediction-box {
    background-color: #e8f4fd;
    padding: 1.5rem;
    border-radius: 0.5rem;
    border-left: 5px solid #1f77b4;
    margin: 1rem 0;
}
.success-box {
    background-color: #d4edda;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 5px solid #28a745;
}
</style>
""", unsafe_allow_html=True)

def load_default_data():
    """Tải dữ liệu mặc định"""
    try:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'House_Prices.csv')
        if os.path.exists(data_path):
            return pd.read_csv(data_path)
        else:
            st.warning("Không tìm thấy file data/House_Prices.csv")
            return None
    except Exception as e:
        st.error(f"Lỗi đọc dữ liệu: {e}")
        return None

def load_model(model_path):
    """Tải mô hình đã lưu"""
    try:
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        return None
    except Exception as e:
        st.error(f"Lỗi tải mô hình: {e}")
        return None

def create_sample_prediction_form():
    """Tạo form nhập liệu để dự đoán"""
    st.subheader("Dự đoán giá nhà")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            overall_qual = st.slider("Chất lượng tổng thể (1-10)", 1, 10, 7)
            gr_liv_area = st.number_input("Diện tích sống (sq ft)", 500, 5000, 1500)
            garage_cars = st.slider("Số xe garage", 0, 4, 2)
            total_bsmt_sf = st.number_input("Diện tích tầng hầm (sq ft)", 0, 3000, 1000)
        
        with col2:
            full_bath = st.slider("Số phòng tắm đầy đủ", 0, 4, 2)
            year_built = st.slider("Năm xây dựng", 1850, 2025, 2000)
            first_flr_sf = st.number_input("Diện tích tầng 1 (sq ft)", 300, 3000, 800)
            tot_rms_abv_grd = st.slider("Tổng số phòng", 3, 15, 7)
        
        predict_button = st.form_submit_button("Dự đoán giá", use_container_width=True)
    
    if predict_button:
        house_features = {
            'OverallQual': overall_qual,
            'GrLivArea': gr_liv_area,
            'GarageCars': garage_cars,
            'TotalBsmtSF': total_bsmt_sf,
            'FullBath': full_bath,
            'YearBuilt': year_built,
            '1stFlrSF': first_flr_sf,
            'TotRmsAbvGrd': tot_rms_abv_grd
        }
        
        return house_features
    return None

def display_prediction_result(prediction, house_features):
    """Hiển thị kết quả dự đoán"""
    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
    st.markdown(f"### {prediction}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Hiển thị thông tin ngôi nhà
    st.subheader("Thông tin ngôi nhà")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Chất lượng", f"{house_features['OverallQual']}/10")
        st.metric("Số phòng tắm", house_features['FullBath'])
    
    with col2:
        st.metric("Diện tích sống", f"{house_features['GrLivArea']:,} sq ft")
        st.metric("Năm xây dựng", house_features['YearBuilt'])
    
    with col3:
        st.metric("Garage", f"{house_features['GarageCars']} xe")
        st.metric("Diện tích T1", f"{house_features['1stFlrSF']:,} sq ft")
    
    with col4:
        st.metric("Tầng hầm", f"{house_features['TotalBsmtSF']:,} sq ft")
        st.metric("Tổng số phòng", house_features['TotRmsAbvGrd'])

def create_data_visualization(df):
    """Tạo các biểu đồ trực quan hóa dữ liệu sử dụng matplotlib"""
    if df is None:
        return
    
    st.subheader("Phân tích dữ liệu")
    
    # Tạo price categories
    df_viz = df.copy()
    def price_category(price):
        if price < 150000: return 'Thấp'
        elif price < 250000: return 'Trung bình'
        else: return 'Cao'
    
    df_viz['PriceCategory'] = df_viz['SalePrice'].apply(price_category)
    
    # Layout 2 cột
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram giá nhà
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.hist(df_viz['SalePrice'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('Phân bố giá nhà', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Giá nhà ($)', fontsize=12)
        ax1.set_ylabel('Số lượng', fontsize=12)
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig1)
        
        # Pie chart phân loại giá
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        category_counts = df_viz['PriceCategory'].value_counts()
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        ax2.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
        ax2.set_title('Phân loại mức giá', fontsize=14, fontweight='bold')
        st.pyplot(fig2)
    
    with col2:
        # Scatter plot Quality vs Price
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        colors_map = {'Thấp': 'red', 'Trung bình': 'orange', 'Cao': 'green'}
        for category in df_viz['PriceCategory'].unique():
            mask = df_viz['PriceCategory'] == category
            ax3.scatter(df_viz[mask]['OverallQual'], df_viz[mask]['SalePrice'],
                       c=colors_map[category], label=category, alpha=0.6)
        ax3.set_title('Chất lượng vs Giá nhà', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Chất lượng tổng thể', fontsize=12)
        ax3.set_ylabel('Giá nhà ($)', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig3)
        
        # Box plot Area vs Price Category
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        category_order = ['Thấp', 'Trung bình', 'Cao']
        df_viz['PriceCategory'] = pd.Categorical(df_viz['PriceCategory'], categories=category_order, ordered=True)
        
        box_data = [df_viz[df_viz['PriceCategory'] == cat]['GrLivArea'].values for cat in category_order]
        bp = ax4.boxplot(box_data, labels=category_order, patch_artist=True)
        
        # Tô màu cho boxplot
        colors = ['#ff9999', '#ffcc99', '#99ff99']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            
        ax4.set_title('Diện tích sống theo mức giá', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Mức giá', fontsize=12)
        ax4.set_ylabel('Diện tích sống (sq ft)', fontsize=12)
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig4)

def run_full_analysis():
    """Chạy phân tích ML hoàn chỉnh"""
    if st.button("Chạy phân tích ML hoàn chỉnh", use_container_width=True):
        
        # Kiểm tra modules đã được import thành công chưa
        if not MODULES_IMPORTED:
            st.error("Không thể chạy phân tích ML hoàn chỉnh vì thiếu modules từ src/")
            st.info("Hãy đảm bảo:")
            st.write("- Các file preprocessing.py, modeling.py, predict.py có trong thư mục src/")
            st.write("- Cấu trúc thư mục đúng như thiết kế")
            st.write("- Hoặc sử dụng chế độ 'Dự đoán nhanh' để test cơ bản")
            return None
        
        with st.spinner("Đang chạy phân tích Machine Learning..."):
            try:
                # Tạo pipeline
                data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'House_Prices.csv')
                
                # Kiểm tra file data có tồn tại không
                if not os.path.exists(data_path):
                    st.error("Không tìm thấy file dữ liệu House_Prices.csv")
                    st.info("Đặt file dữ liệu vào thư mục data/")
                    return None
                
                pipeline = HousePricePipeline(data_path)
                
                # Chạy preprocessing
                pipeline.preprocessor.explore_data()
                X, y_class, y_reg = pipeline.preprocessor.prepare_features()
                
                # Hiển thị thông tin cơ bản
                st.success("Preprocessing hoàn thành!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Số mẫu", X.shape[0])
                with col2:
                    st.metric("Số features", X.shape[1])
                with col3:
                    st.metric("Giá trung bình", f"${y_reg.mean():,.0f}")
                
                # Chạy models (simplified)
                st.write("Huấn luyện các mô hình...")
                
                # Chia dữ liệu
                data_splits = pipeline.preprocessor.split_data(X, y_class, y_reg)
                X_train, X_test = data_splits[0], data_splits[1]
                y_train_reg, y_test_reg = data_splits[4], data_splits[5]
                
                # Random Forest Regressor
                from sklearn.ensemble import RandomForestRegressor
                rf_model = RandomForestRegressor(n_estimators=50, random_state=42)  # Giảm n_estimators để nhanh hơn
                rf_model.fit(X_train, y_train_reg)
                
                # Đánh giá
                from sklearn.metrics import r2_score, mean_absolute_error
                y_pred = rf_model.predict(X_test)
                r2 = r2_score(y_test_reg, y_pred)
                mae = mean_absolute_error(y_test_reg, y_pred)
                
                st.success("Huấn luyện mô hình hoàn thành!")
                
                # Hiển thị kết quả
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("R² Score", f"{r2:.3f}")
                with col2:
                    st.metric("MAE", f"${mae:,.0f}")
                
                return rf_model
                
            except Exception as e:
                st.error(f"Lỗi trong quá trình phân tích: {e}")
                st.info("Hãy kiểm tra lại cấu trúc file và dữ liệu")
                return None
    
    return None

def main():
    """Hàm main của ứng dụng Streamlit"""
    
    # Header
    st.title("House Price Prediction")
    # st.markdown("### Ứng dụng dự đoán giá nhà sử dụng Machine Learning")
    
    # Sidebar
    st.sidebar.title("House Price Prediction")
    st.sidebar.markdown("---")

    # Thông báo về tính năng hạn chế
    if not MODULES_IMPORTED:
        st.sidebar.warning("**Import Module Issues**")
        st.sidebar.info("""
        **Tình trạng hiện tại:**
        - Không import được src/ modules  
        - Chế độ Demo cơ bản hoạt động
        
        **Giải pháp:**
        1. Đảm bảo file structure đúng
        2. Kiểm tra thư mục src/
        3. Hoặc dùng chế độ 'Dự đoán nhanh'
        """)
        st.sidebar.markdown("---")
    
    # Lựa chọn chế độ
    mode = st.sidebar.selectbox(
        "Chọn chế độ sử dụng:",
        ["Dự đoán nhanh", "Phân tích dữ liệu", "Phân tích ML hoàn chỉnh"],
        help="Chế độ ML hoàn chỉnh cần src/ modules hoạt động đầy đủ"
    )
    
    # Load dữ liệu
    df = load_default_data()
    
    if df is not None:
        st.sidebar.success(f"Đã tải {len(df)} mẫu dữ liệu")
        
        # Hiển thị thống kê cơ bản
        st.sidebar.subheader("Thống kê cơ bản")
        st.sidebar.write(f"Giá trung bình: ${df['SalePrice'].mean():,.0f}")
        st.sidebar.write(f"Giá cao nhất: ${df['SalePrice'].max():,.0f}")
        st.sidebar.write(f"Giá thấp nhất: ${df['SalePrice'].min():,.0f}")
    else:
        st.sidebar.error("Không thể tải dữ liệu")
    
    # Main content dựa vào mode
    if mode == "Dự đoán nhanh":
        # Form dự đoán
        house_features = create_sample_prediction_form()
        
        if house_features:
            # Dự đoán đơn giản bằng heuristic
            # (Trong thực tế sẽ dùng mô hình đã train)
            base_price = 100000
            price_estimate = (
                base_price +
                house_features['OverallQual'] * 15000 +
                house_features['GrLivArea'] * 80 +
                house_features['GarageCars'] * 10000 +
                house_features['TotalBsmtSF'] * 30 +
                house_features['FullBath'] * 8000 +
                (house_features['YearBuilt'] - 1900) * 200 +
                house_features['1stFlrSF'] * 60 +
                house_features['TotRmsAbvGrd'] * 3000
            )
            
            prediction_text = f"Giá dự đoán: ${price_estimate:,.0f}"
            display_prediction_result(prediction_text, house_features)
    
    elif mode == "Phân tích dữ liệu":
        if df is not None:
            create_data_visualization(df)
            
            # Hiển thị sample data
            st.subheader("Dữ liệu mẫu")
            st.dataframe(df.head(10))
            
            # Thống kê mô tả
            st.subheader("Thống kê mô tả")
            st.dataframe(df.describe())
        else:
            st.warning("Không có dữ liệu để phân tích")
    
    elif mode == "Phân tích ML hoàn chỉnh":
        st.subheader("Phân tích Machine Learning Hoàn chỉnh")
        st.write("Chế độ này sẽ chạy toàn bộ pipeline ML từ preprocessing đến evaluation.")
        
        # Chạy phân tích
        trained_model = run_full_analysis()
        
        if trained_model is not None:
            st.success("Phân tích hoàn thành! Mô hình đã sẵn sàng cho dự đoán.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"  
        "<p> Made with Streamlit | House Price Prediction </p>"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
