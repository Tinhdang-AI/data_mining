"""
Streamlit Demo App cho House Price Prediction
Giao diện: Cơ bản (Standard)
Chức năng: So sánh 2 mô hình (Linear & Random Forest)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

# Cấu hình matplotlib
plt.style.use('default')
sns.set_palette("husl")

# --- 1. THIẾT LẬP HỆ THỐNG ---
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    src_dir = os.path.join(parent_dir, 'src')
    models_dir = os.path.join(parent_dir, 'models')
    
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    from preprocessing import HousePricePreprocessor
    from modeling import ModelEvaluator, ModelFactory
    from predict import HousePricePipeline
    
    MODULES_IMPORTED = True
except ImportError as e:
    st.error(f"Lỗi import modules: {e}")
    MODULES_IMPORTED = False

# --- 2. CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
st.markdown("""
<style>
.metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; }
.prediction-box { background-color: #e8f4fd; padding: 1.5rem; border-radius: 0.5rem; border-left: 5px solid #1f77b4; margin: 1rem 0; }
.success-box { background-color: #d4edda; padding: 1rem; border-radius: 0.5rem; border-left: 5px solid #28a745; }
</style>
""", unsafe_allow_html=True)

# --- 3. CÁC HÀM XỬ LÝ ---

def load_default_data():
    try:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'House_Prices.csv')
        if os.path.exists(data_path):
            return pd.read_csv(data_path)
        return None
    except Exception as e:
        st.error(f"Lỗi đọc dữ liệu: {e}")
        return None

def load_model():
    try:
        model_path = os.path.join(models_dir, 'model.pkl')
        if os.path.exists(model_path):
            return joblib.load(model_path)
        return None
    except Exception as e:
        return None

def create_sample_prediction_form():
    st.subheader("Dự đoán giá nhà")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            overall_qual = st.slider("Chất lượng tổng thể (1-10)", 1, 10, 7)
            gr_liv_area = st.number_input("Diện tích sống (sq ft)", 500, 6000, 1500)
            first_flr_sf = st.number_input("Diện tích tầng 1 (sq ft)", 300, 5000, 800)
            total_bsmt_sf = st.number_input("Diện tích tầng hầm (sq ft)", 0, 6500, 1000)
        with col2:
            full_bath = st.slider("Số phòng tắm đầy đủ", 0, 4, 2)
            year_built = st.slider("Năm xây dựng", 1850, 2025, 2000)
            garage_cars = st.slider("Số xe garage", 0, 4, 2)
            tot_rms_abv_grd = st.slider("Tổng số phòng", 3, 15, 7)
        
        predict_button = st.form_submit_button("Dự đoán giá", use_container_width=True)
    
    if predict_button:
        return {
            'OverallQual': overall_qual, 'GrLivArea': gr_liv_area,
            'GarageCars': garage_cars, 'TotalBsmtSF': total_bsmt_sf,
            'FullBath': full_bath, 'YearBuilt': year_built,
            '1stFlrSF': first_flr_sf, 'TotRmsAbvGrd': tot_rms_abv_grd
        }
    return None

def display_prediction_result(prediction, house_features):
    st.markdown(f"""<div class="prediction-box"><h3>{prediction}</h3></div>""", unsafe_allow_html=True)
    st.subheader("Thông tin ngôi nhà")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Chất lượng", f"{house_features['OverallQual']}/10"); st.metric("Số phòng tắm", house_features['FullBath'])
    with col2: st.metric("Diện tích sống", f"{house_features['GrLivArea']:,} sq ft"); st.metric("Năm xây dựng", house_features['YearBuilt'])
    with col3: st.metric("Garage", f"{house_features['GarageCars']} xe"); st.metric("Diện tích T1", f"{house_features['1stFlrSF']:,} sq ft")
    with col4: st.metric("Tầng hầm", f"{house_features['TotalBsmtSF']:,} sq ft"); st.metric("Tổng số phòng", house_features['TotRmsAbvGrd'])

def create_data_visualization(df):
    st.subheader("Phân tích dữ liệu")
    df_viz = df.copy()
    df_viz['PriceCategory'] = df_viz['SalePrice'].apply(lambda x: 'Thấp' if x < 150000 else ('Trung bình' if x < 250000 else 'Cao'))
    
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.hist(df_viz['SalePrice'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('Phân bố giá nhà')
        st.pyplot(fig1)
        
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        counts = df_viz['PriceCategory'].value_counts()
        ax2.pie(counts.values, labels=counts.index, autopct='%1.1f%%', colors=['#ff9999', '#66b3ff', '#99ff99'])
        ax2.set_title('Phân loại mức giá')
        st.pyplot(fig2)
        
    with col2:
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        for cat, color in zip(['Thấp', 'Trung bình', 'Cao'], ['red', 'orange', 'green']):
            mask = df_viz['PriceCategory'] == cat
            ax3.scatter(df_viz[mask]['OverallQual'], df_viz[mask]['SalePrice'], c=color, label=cat, alpha=0.6)
        ax3.set_title('Chất lượng vs Giá nhà'); ax3.legend()
        st.pyplot(fig3)
        
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        data = [df_viz[df_viz['PriceCategory']==c]['GrLivArea'] for c in ['Thấp', 'Trung bình', 'Cao']]
        ax4.boxplot(data, labels=['Thấp', 'Trung bình', 'Cao'], patch_artist=True)
        ax4.set_title('Diện tích sống theo mức giá')
        st.pyplot(fig4)

def run_full_analysis():
    """Chạy phân tích ML: So sánh 2 mô hình"""
    if st.button("Chạy phân tích ML hoàn chỉnh", use_container_width=True):
        if not MODULES_IMPORTED:
            st.error("Lỗi module!")
            return
        
        with st.spinner("Đang chạy phân tích Machine Learning..."):
            try:
                # 1. Init
                data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'House_Prices.csv')
                # Kiểm tra file data có tồn tại không
                if not os.path.exists(data_path):
                    st.error("Không tìm thấy file dữ liệu House_Prices.csv")
                    st.info("Đặt file dữ liệu vào thư mục data/")
                    return None
                pipeline = HousePricePipeline(data_path)
                
                # 2. Preprocessing
                pipeline.preprocessor.explore_data()
                X, y = pipeline.preprocessor.prepare_features()
                
                st.success("Preprocessing hoàn thành!")
                c1, c2, c3 = st.columns(3)
                c1.metric("Số mẫu", X.shape[0]); c2.metric("Số features", X.shape[1]); c3.metric("Giá TB", f"${y.mean():,.0f}")
                
                # 3. Training & Comparison
                st.write("Huấn luyện và So sánh mô hình...")
                X_train, X_test, y_train, y_test = pipeline.preprocessor.split_data(X, y)
                X_train_scaled, X_test_scaled = pipeline.preprocessor.scale_features(X_train, X_test)
                
                evaluator = ModelEvaluator()
                models = ModelFactory.create_models()
                
                # Train Linear Regression
                lr_res = evaluator.evaluate_model(models['Linear Regression'], X_train_scaled, X_test_scaled, y_train, y_test, "Linear Regression")
                
                # Train Random Forest
                rf_res = evaluator.evaluate_model(models['Random Forest'], X_train, X_test, y_train, y_test, "Random Forest")
                
                st.success("Huấn luyện hoàn tất!")
                
                # --- HIỂN THỊ SO SÁNH (Giong hinh em gui) ---
                st.subheader("📈 Kết quả đánh giá")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Random Forest")
                    st.metric("R² Score", f"{rf_res['R²']:.3f}")
                    st.metric("MAE", f"${rf_res['MAE']:,.0f}")
                    st.metric("RMSE", f"${rf_res['RMSE']:,.0f}")
                
                with col2:
                    st.markdown("### Linear Regression")
                    st.metric("R² Score", f"{lr_res['R²']:.3f}")
                    st.metric("MAE", f"${lr_res['MAE']:,.0f}")
                    st.metric("RMSE", f"${lr_res['RMSE']:,.0f}")
                
                # Vẽ biểu đồ cột so sánh
                fig, ax = plt.subplots(figsize=(10, 5))
                models_names = ['Linear Regression', 'Random Forest']
                r2_scores = [lr_res['R²'], rf_res['R²']]
                bars = ax.bar(models_names, r2_scores, color=['#95a5a6', '#2ecc71'])
                ax.set_title("So sánh độ chính xác (R² Score)")
                ax.set_ylim(0, 1.1)
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', ha='center', va='bottom')
                st.pyplot(fig)
                
                # 4. Lưu Model tốt nhất (Random Forest)
                if not os.path.exists(models_dir): os.makedirs(models_dir)
                model_path = os.path.join(models_dir, 'model.pkl')
                evaluator.save_model('Random Forest', model_path)
                st.success("✅ Đã lưu mô hình tốt nhất (Random Forest) cho chức năng Dự đoán!")
                
            except Exception as e:
                st.error(f"Lỗi: {e}")

# --- 4. MAIN PROGRAM ---
def main():
    st.title("House Price Prediction")
    st.sidebar.title("House Price Prediction")
    st.sidebar.markdown("---")
    
    # Menu Selectbox (Giao dien cu)
    mode = st.sidebar.selectbox("Chọn chế độ sử dụng:", 
                               ["Dự đoán nhanh", "Phân tích dữ liệu", "Phân tích ML hoàn chỉnh"])
    
    df = load_default_data()
    if df is not None:
        st.sidebar.success(f"Đã tải {len(df)} mẫu dữ liệu")
        st.sidebar.subheader("Thống kê cơ bản")
        st.sidebar.write(f"Giá trung bình: ${df['SalePrice'].mean():,.0f}")
        st.sidebar.write(f"Giá cao nhất: ${df['SalePrice'].max():,.0f}")
        st.sidebar.write(f"Giá thấp nhất: ${df['SalePrice'].min():,.0f}")
    
    if mode == "Dự đoán nhanh":
        features = create_sample_prediction_form()
        if features:
            model = load_model()
            if model:
                try:
                    price = model.predict(pd.DataFrame([features]))[0]
                    display_prediction_result(f"Giá dự đoán: ${price:,.0f}", features)
                except Exception as e: st.error(f"Lỗi: {e}")
            else:
                st.warning("Chưa có model AI. Vui lòng chạy 'Phân tích ML hoàn chỉnh' trước.")
                # Fallback heuristic
                base = 100000 + features['OverallQual']*15000 + features['GrLivArea']*80
                display_prediction_result(f"Giá ước tính (Sơ bộ): ${base:,.0f}", features)
                
    elif mode == "Phân tích dữ liệu":
        if df is not None:
            create_data_visualization(df)
            st.subheader("Dữ liệu mẫu"); st.dataframe(df.head(10))
            st.subheader("Thống kê mô tả"); st.dataframe(df.describe())
            
    elif mode == "Phân tích ML hoàn chỉnh":
        st.subheader("Phân tích Machine Learning Hoàn chỉnh")
        st.write("Chế độ này sẽ chạy toàn bộ pipeline ML từ preprocessing đến evaluation.")
        run_full_analysis()
    
    st.markdown("---"); st.markdown("<div style='text-align: center; color: #666;'>Made with Streamlit</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()