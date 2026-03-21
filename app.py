"""
app.py
------
Streamlit Dashboard สำหรับ predict GPA นักศึกษา
ใช้ Best Model จาก Section 5

Usage:
    streamlit run app.py
"""

import json
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path

import streamlit as st
import matplotlib.pyplot as plt

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
# โหลด model bundle
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

# ============================================================
# Helper: predict
# ============================================================
def predict_gpa(input_dict: dict):
    row = pd.DataFrame([input_dict])[FEATURES].astype('float32')
    if scaler:
        row = scaler.transform(row)
    gpa = float(np.clip(model.predict(row)[0], 0, 10))
    if gpa < 4:
        return round(gpa, 2), "Low (0-4)",    "#e74c3c", "🔴"
    elif gpa < 7:
        return round(gpa, 2), "Medium (4-7)", "#f39c12", "🟡"
    else:
        return round(gpa, 2), "High (7-10)",  "#27ae60", "🟢"

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Student GPA Predictor",
    page_icon="🎓",
    layout="wide"
)

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.title("🎓 GPA Predictor")
    st.markdown("---")
    st.subheader("📋 Model Info")
    st.write(f"**Model:** {METADATA.get('model_type', 'Best Model')}")
    st.write(f"**Version:** {METADATA.get('version', '1.0.2')}")
    if PERF.get('r2_score') is not None:
        st.metric("R² Score", f"{PERF['r2_score']:.4f}")
        st.metric("RMSE",     f"{PERF['rmse']:.4f}")
        st.metric("MAE",      f"{PERF['mae']:.4f}")
    st.markdown("---")
    st.subheader("📊 ข้อมูลที่ใช้ Train")
    st.metric("Dataset ทั้งหมด", "1,000,000 แถว")
    col_tr, col_te = st.columns(2)
    col_tr.metric("Train", "800,000", "80%")
    col_te.metric("Test",  "200,000", "20%")
    st.markdown("---")
    st.caption(f"Features: {len(FEATURES)} ตัว")
    st.caption("Features ที่ใช้:")
    for f in FEATURES:
        st.caption(f"  • {f}")

# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Compare Students", "ℹ️ Model Details"])

# ----------------------------------------------------------
# Tab 1: Predict
# ----------------------------------------------------------
with tab1:
    st.header("🔮 ทำนาย GPA นักศึกษา")
    st.markdown("กรอกข้อมูลพฤติกรรม **ก่อนสอบ** แล้วกด Predict")

    with st.expander("❓ ทำไม GPA ถึงเป็นสเกล 0-10?"):
        st.markdown("""
        GPA ในชุดข้อมูลเดิมอยู่ในสเกล **0-2** ซึ่งอ่านค่าได้ยาก  
        จึงแปลงเป็นสเกล **0-10** โดยคูณ 5 เพื่อให้เข้าใจง่ายขึ้น

        | สเกลเดิม (0-2) | สเกลใหม่ (0-10) | ระดับ |
        |---|---|---|
        | 0.0 – 0.8 | 0 – 4 | 🔴 Low |
        | 0.8 – 1.4 | 4 – 7 | 🟡 Medium |
        | 1.4 – 2.0 | 7 – 10 | 🟢 High |
        """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📚 การเรียน")
        study_hours              = st.slider("ชั่วโมงเรียนต่อวัน",              0.0, 12.0,  5.0, 0.5)
        attendance               = st.slider("% การเข้าเรียน",                  0.0, 100.0, 80.0, 1.0)
        online_courses_completed = st.slider("คอร์สออนไลน์ที่จบแล้ว (ชิ้น)",   0.0, 20.0,  2.0, 1.0)
        extracurricular_hours    = st.slider("กิจกรรม extracurricular (ชม./สัปดาห์)", 0.0, 20.0, 2.0, 0.5)

    with col2:
        st.subheader("🧠 จิตใจ & ไลฟ์สไตล์")
        stress                   = st.slider("ระดับความเครียด (0-10)",          0.0, 10.0,  5.0, 0.5)
        anxiety                  = st.slider("ระดับความวิตกกังวล (0-10)",       0.0, 10.0,  5.0, 0.5)
        social_media_hours       = st.slider("ชั่วโมง social media/วัน",        0.0, 12.0,  3.0, 0.5)
        part_time_hours          = st.slider("ชั่วโมงทำงาน part-time/สัปดาห์", 0.0, 40.0,  0.0, 1.0)

    st.markdown("---")

    if st.button("🔮 Predict GPA", type="primary", use_container_width=True):
        input_data = {
            'study_hours':              study_hours,
            'attendance':               attendance,
            'stress':                   stress,
            'anxiety':                  anxiety,
            'social_media_hours':       social_media_hours,
            'online_courses_completed': online_courses_completed,
            'part_time_hours':          part_time_hours,
            'extracurricular_hours':    extracurricular_hours,
        }

        gpa, level, bar_color, emoji = predict_gpa(input_data)
        logger.info(f"predict -> GPA={gpa}  level={level}")

        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.metric("GPA ที่ทำนาย (0-10)", gpa)
            st.markdown(f"### {emoji} {level}")

        with res_col2:
            fig, ax = plt.subplots(figsize=(6, 1.6))
            ax.barh(['GPA'], [gpa],      color=bar_color, height=0.5)
            ax.barh(['GPA'], [10 - gpa], left=gpa, color='#ecf0f1', height=0.5)
            ax.axvline(4, color='#e74c3c', linestyle='--', linewidth=1, alpha=0.6)
            ax.axvline(7, color='#f39c12', linestyle='--', linewidth=1, alpha=0.6)
            ax.text(max(gpa / 2, 0.3), 0, f'{gpa}',
                    ha='center', va='center', color='white', fontweight='bold', fontsize=14)
            ax.set_xlim(0, 10)
            ax.set_yticks([])
            ax.set_xlabel('GPA (0-10)')
            ax.set_title('GPA Gauge')
            st.pyplot(fig)
            plt.close()

        if PERF.get('rmse'):
            st.info(f"R²={PERF.get('r2_score','?')}  |  RMSE={PERF['rmse']:.3f}  — ผลทำนายอาจคลาดเคลื่อน ±{PERF['rmse']:.2f} GPA")

        # ============================================================
        # คำแนะนำในการปรับตัว
        # ============================================================
        st.markdown("---")
        st.subheader("💡 คำแนะนำในการปรับปรุง")

        suggestions = []

        if study_hours < 6:
            suggestions.append(("📚 เพิ่มชั่วโมงเรียน",
                f"ตอนนี้เรียน **{study_hours} ชม./วัน** — ลองเพิ่มเป็น **6-8 ชม.** จะช่วย GPA ได้มาก"))
        if attendance < 80:
            suggestions.append(("🏫 เข้าเรียนให้สม่ำเสมอ",
                f"เข้าเรียน **{attendance:.0f}%** — ควรรักษาให้อยู่ที่ **80%+**"))
        if stress > 6:
            suggestions.append(("🧘 ลดความเครียด",
                f"ความเครียดอยู่ที่ **{stress}/10** — ลองออกกำลังกาย นอนหลับพักผ่อนให้เพียงพอ"))
        if anxiety > 6:
            suggestions.append(("😌 จัดการความวิตกกังวล",
                f"ความวิตกกังวลอยู่ที่ **{anxiety}/10** — ลองเทคนิค breathing หรือพูดคุยกับอาจารย์ที่ปรึกษา"))
        if social_media_hours > 4:
            suggestions.append(("📱 ลด social media",
                f"ใช้ social media **{social_media_hours} ชม./วัน** — ลองลดเหลือ **2 ชม.** แล้วเอาเวลาไปเรียนแทน"))
        if part_time_hours > 15:
            suggestions.append(("⏰ ระวังเวลาทำงาน part-time",
                f"ทำงาน **{part_time_hours:.0f} ชม./สัปดาห์** — ถ้ามากเกินไปจะกระทบเวลาเรียน"))
        if online_courses_completed < 2:
            suggestions.append(("💻 เรียนคอร์สออนไลน์เพิ่ม",
                f"จบคอร์สออนไลน์ **{online_courses_completed:.0f} ชิ้น** — ลองเรียนเพิ่มเพื่อเสริมทักษะ"))
        if extracurricular_hours < 2:
            suggestions.append(("🎯 เข้าร่วมกิจกรรม",
                f"กิจกรรม extracurricular **{extracurricular_hours} ชม./สัปดาห์** — การเข้าร่วมกิจกรรมช่วยพัฒนาทักษะและ GPA"))

        if not suggestions:
            st.success("🌟 พฤติกรรมของคุณอยู่ในเกณฑ์ดีทุกด้าน รักษาระดับนี้ไว้!")
        else:
            priority = {
                "🔴 Low (0-4)":    "ควรปรับปรุงด่วน แนะนำให้เริ่มจากข้อบนก่อน",
                "🟡 Medium (4-7)": "มีจุดที่ปรับได้เพิ่มเติม",
                "🟢 High (7-10)":  "ทำได้ดีแล้ว ปรับเล็กน้อยก็จะยิ่งดีขึ้น",
            }
            st.markdown(f"**{level}** — {priority.get(f'{emoji} {level}', '')}")
            for title, desc in suggestions:
                with st.expander(title):
                    st.markdown(desc)

# ----------------------------------------------------------
# Tab 2: Compare Students
# ----------------------------------------------------------
with tab2:
    st.header("📊 เปรียบเทียบนักศึกษาหลายคน")
    st.markdown("แก้ไขตารางได้เลย แล้วกด **Compare**")

    default_data = pd.DataFrame({
        'study_hours':              [8.0,  2.0,  10.0, 3.0],
        'attendance':               [90.0, 55.0, 95.0, 60.0],
        'stress':                   [3.0,  8.0,  2.0,  7.0],
        'anxiety':                  [2.0,  8.0,  2.0,  7.0],
        'social_media_hours':       [2.0,  6.0,  1.0,  5.0],
        'online_courses_completed': [4.0,  0.0,  6.0,  1.0],
        'part_time_hours':          [0.0,  15.0, 0.0,  10.0],
        'extracurricular_hours':    [3.0,  1.0,  4.0,  2.0],
    }, index=[f'Student {i+1}' for i in range(4)])

    edited_df = st.data_editor(default_data, num_rows="dynamic", use_container_width=True)

    if st.button("📊 Compare", type="primary"):
        results = []
        for idx, row in edited_df.iterrows():
            gpa, level, _, emoji = predict_gpa(row.to_dict())
            results.append({'Student': str(idx), 'GPA': gpa, 'Level': f'{emoji} {level}'})
        result_df = pd.DataFrame(results)

        st.dataframe(result_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        colors = ['#e74c3c' if g < 4 else ('#f39c12' if g < 7 else '#27ae60')
                  for g in result_df['GPA']]
        bars = ax.bar(result_df['Student'], result_df['GPA'], color=colors, width=0.5)
        ax.axhline(4, color='#e74c3c', linestyle='--', linewidth=1, label='Low / Medium')
        ax.axhline(7, color='#f39c12', linestyle='--', linewidth=1, label='Medium / High')
        ax.set_ylim(0, 11)
        ax.set_ylabel('Predicted GPA (0-10)')
        ax.set_title('Predicted GPA Comparison')
        ax.legend(fontsize=9)
        for bar, val in zip(bars, result_df['GPA']):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.15, f'{val}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        logger.info(f"compare {len(result_df)} students")

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
        st.dataframe(
            pd.DataFrame({'Feature': FEATURES, 'Index': range(len(FEATURES))}),
            use_container_width=True
        )

        if hasattr(model, 'coef_'):
            st.subheader("📈 Ridge Coefficients")
            coef_df = pd.DataFrame({
                'Feature': FEATURES, 'Coefficient': model.coef_
            }).sort_values('Coefficient', key=abs, ascending=True)
            fig, ax = plt.subplots(figsize=(7, max(3, len(FEATURES) * 0.5)))
            colors = ['#27ae60' if v > 0 else '#e74c3c' for v in coef_df['Coefficient']]
            ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
            ax.axvline(0, color='black', linewidth=0.8)
            ax.set_title('Coefficients (บวก=เพิ่ม GPA, ลบ=ลด GPA)')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        elif hasattr(model, 'feature_importances_'):
            st.subheader("📈 Feature Importance")
            imp_df = pd.DataFrame({
                'Feature': FEATURES, 'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)
            fig, ax = plt.subplots(figsize=(7, max(3, len(FEATURES) * 0.5)))
            ax.barh(imp_df['Feature'], imp_df['Importance'], color='steelblue')
            ax.set_title('Feature Importance')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()