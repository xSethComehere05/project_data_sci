import json
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    handlers=[logging.FileHandler('student_gpa_service.log')]
)
logger = logging.getLogger(__name__)

# ============================================================
# Load model bundle
# ============================================================
BUNDLE_PATH = Path('model_artifacts/student_gpa_best_model.pkl')
META_PATH   = Path('model_artifacts/model_metadata.json')

@st.cache_resource
def load_bundle():
    if not BUNDLE_PATH.exists():
        st.error(f"ไม่พบ {BUNDLE_PATH} — กรุณารัน Section 5 ใน notebook ก่อน")
        st.stop()
    with open(BUNDLE_PATH, 'rb') as f:
        bundle = pickle.load(f)
    with open(META_PATH) as f:
        meta = json.load(f)
    return bundle, meta

bundle, METADATA = load_bundle()
model    = bundle['model']
scaler   = bundle.get('scaler')
FEATURES = bundle['features']
PERF     = METADATA.get('performance', {})

CLASS_LABELS = {0: 'Low', 1: 'Medium', 2: 'High'}
CLASS_COLORS = {0: '#EF4444', 1: '#F59E0B', 2: '#22C55E'}
CLASS_BG     = {0: '#FEE2E2', 1: '#FEF3C7', 2: '#DCFCE7'}

# ============================================================
# Helper: Predict & Categorize (Regression Style)
# ============================================================
def predict_class(input_dict: dict):
    row = pd.DataFrame([input_dict])[FEATURES].astype('float32')
    if scaler:
        row = scaler.transform(row)

    # 1. ทำนายค่า GPA (Regression)
    raw_score = float(model.predict(row)[0])
    
    # 2. ตัดเกรดตามช่วงคะแนน 0-4, 4-7, 7-10
    if raw_score < 4:
        pred_class, label = 0, 'Low'
    elif raw_score < 7:
        pred_class, label = 1, 'Medium'
    else:
        pred_class, label = 2, 'High'
        
    color = CLASS_COLORS[pred_class]
    bg    = CLASS_BG[pred_class]

    # ตรวจสอบ Probability (ถ้ามี)
    proba = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(row)[0]

    return pred_class, label, color, bg, proba, raw_score

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Student Performance Classifier",
    page_icon="🎓",
    layout="wide"
)

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.title("🎓 Performance Classifier")
    st.markdown("---")
    st.subheader("📋 Model Info")
    st.write(f"**Model:** {METADATA.get('model_type', 'Best Model')}")
    st.write(f"**Task:** Regression -> Classification")
    st.write(f"**Version:** {METADATA.get('version', '1.0.0')}")
    if PERF.get('accuracy') is not None:
        st.metric("Accuracy",  f"{PERF['accuracy']:.4f}")
    st.markdown("---")
    st.subheader("📊 Training Data")
    st.metric("Dataset", "1,000,000 rows")
    st.markdown("---")
    st.caption(f"Features: {len(FEATURES)} ตัว")
    for feat in FEATURES:
        st.caption(f"  • {feat}")

# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Compare Students", "ℹ️ Model Details"])

# ----------------------------------------------------------
# Tab 1: Predict
# ----------------------------------------------------------
with tab1:
    st.header("🔮 ทำนายระดับผลการเรียน")

    with st.expander("📖 Performance Level คืออะไร?"):
        c1, c2, c3 = st.columns(3)
        c1.markdown("<div style='background:#FEE2E2;padding:12px;border-radius:8px;text-align:center'><b style='color:#991B1B'>🔴 Low</b><br>GPA 0 – 4</div>", unsafe_allow_html=True)
        c2.markdown("<div style='background:#FEF3C7;padding:12px;border-radius:8px;text-align:center'><b style='color:#92400E'>🟡 Medium</b><br>GPA 4 – 7</div>", unsafe_allow_html=True)
        c3.markdown("<div style='background:#DCFCE7;padding:12px;border-radius:8px;text-align:center'><b style='color:#14532D'>🟢 High</b><br>GPA 7 – 10</div>", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📚 การเรียน")
        study_hours = st.slider("ชั่วโมงเรียนต่อวัน", 0.0, 12.0, 5.0, 0.5)
        attendance = st.slider("% การเข้าเรียน", 0.0, 100.0, 80.0, 1.0)
        online_courses = st.slider("คอร์สออนไลน์ที่จบแล้ว", 0.0, 20.0, 2.0, 1.0)
        extracurricular = st.slider("กิจกรรม (ชม./สัปดาห์)", 0.0, 20.0, 2.0, 0.5)

    with col2:
        st.subheader("🧠 จิตใจ & ไลฟ์สไตล์")
        stress = st.slider("ระดับความเครียด (0-10)", 0.0, 10.0, 5.0, 0.5)
        anxiety = st.slider("ระดับความวิตกกังวล (0-10)", 0.0, 10.0, 5.0, 0.5)
        social_media = st.slider("ชั่วโมง Social Media/วัน", 0.0, 12.0, 3.0, 0.5)
        part_time = st.slider("ทำงาน Part-time (ชม./สัปดาห์)", 0.0, 40.0, 0.0, 1.0)

    st.markdown("---")

    if st.button("🔮 Predict Performance Level", type="primary", use_container_width=True):
        input_data = {
            'study_hours': study_hours,
            'attendance': attendance,
            'stress': stress,
            'anxiety': anxiety,
            'social_media_hours': social_media,
            'online_courses_completed': online_courses,
            'part_time_hours': part_time,
            'extracurricular_hours': extracurricular,
        }

        pred_class, label, color, bg, proba, raw_score = predict_class(input_data)
        
        r1, r2 = st.columns([1, 2])

        with r1:
            st.markdown(f"""
            <div style='background:{bg};border:2px solid {color};border-radius:12px;padding:20px;text-align:center'>
              <div style='font-size:40px;margin-bottom:8px'>{"🔴" if pred_class==0 else "🟡" if pred_class==1 else "🟢"}</div>
              <div style='font-size:24px;font-weight:700;color:{color}'>{label}</div>
              <div style='font-size:13px;color:#555;'>Performance Level</div>
            </div>""", unsafe_allow_html=True)

        with r2:
            if proba is not None:
                st.subheader("Probability ของแต่ละ class")
                fig, ax = plt.subplots(figsize=(6, 2.5))
                ax.barh([CLASS_LABELS[i] for i in range(3)], proba, color=[CLASS_COLORS[i] for i in range(3)], height=0.5)
                ax.set_xlim(0, 1)
                for i, p in enumerate(proba):
                    ax.text(p + 0.01, i, f'{p:.1%}', va='center', fontweight='bold')
                st.pyplot(fig)
                plt.close()
            else:
                # เอากล่องข้อความสีฟ้าออกแล้ว: แสดงเป็นข้อความสรุปสั้นๆ แทนหรือปล่อยว่าง
                st.write("---")
                # st.write(f"ผลการทำนายดิบจากโมเดล: ****")

        # ── Suggestions ──────────────────
        st.markdown("---")
        st.subheader("💡 คำแนะนำในการปรับปรุง")
        suggestions = []
        if study_hours < 6: suggestions.append(("📚 เพิ่มชั่วโมงเรียน", f"ลองเพิ่มเป็น **6–8 ชม.**"))
        if attendance < 80: suggestions.append(("🏫 เข้าเรียนให้สม่ำเสมอ", f"ควรให้มากกว่า **80%+**"))
        if stress > 6: suggestions.append(("🧘 ลดความเครียด", f"ลองพักผ่อนหรือออกกำลังกาย"))
        
        if not suggestions:
            st.success("🌟 พฤติกรรมของคุณอยู่ในเกณฑ์ดีทุกด้าน!")
        else:
            for title, desc in suggestions:
                with st.expander(title): st.write(desc)

# ----------------------------------------------------------
# Tab 2: Compare Students
# ----------------------------------------------------------
with tab2:
    st.header("📊 เปรียบเทียบนักศึกษา")
    default_data = pd.DataFrame({
        'study_hours': [8.0, 2.0, 10.0, 3.0],
        'attendance': [90.0, 55.0, 95.0, 60.0],
        'stress': [3.0, 8.0, 2.0, 7.0],
        'anxiety': [2.0, 8.0, 2.0, 7.0],
        'social_media_hours': [2.0, 6.0, 1.0, 5.0],
        'online_courses_completed': [4.0, 0.0, 6.0, 1.0],
        'part_time_hours': [0.0, 15.0, 0.0, 10.0],
        'extracurricular_hours': [3.0, 1.0, 4.0, 2.0],
    }, index=[f'Student {i+1}' for i in range(4)])

    edited_df = st.data_editor(default_data, num_rows="dynamic", use_container_width=True)

    if st.button("📊 Compare", type="primary"):
        results = []
        for idx, row in edited_df.iterrows():
            p_class, label, color, bg, proba, r_score = predict_class(row.to_dict())
            emoji = "🔴" if p_class == 0 else "🟡" if p_class == 1 else "🟢"
            results.append({'Student': str(idx), 'Level': f'{emoji} {label}', 'Class': p_class})

        res_df = pd.DataFrame(results)
        st.dataframe(res_df[['Student', 'GPA', 'Level']], use_container_width=True)

# ----------------------------------------------------------
# Tab 3: Model Details
# ----------------------------------------------------------
with tab3:
    st.header("ℹ️ รายละเอียด Model")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📋 Metadata")
        st.json(METADATA)
    with col2:
        st.subheader("🔑 Features ที่ใช้")
        st.write(FEATURES)

    if hasattr(model, 'feature_importances_'):
        st.subheader("📈 Feature Importance")
        imp_df = pd.DataFrame({'Feature': FEATURES, 'Importance': model.feature_importances_}).sort_values('Importance')
        fig, ax = plt.subplots()
        ax.barh(imp_df['Feature'], imp_df['Importance'], color='skyblue')
        st.pyplot(fig)