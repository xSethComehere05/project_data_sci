import streamlit as st
import numpy as np
import joblib
import json
import pandas as pd

# ===== 1. การตั้งค่าหน้าเว็บ =====
st.set_page_config(
    page_title="Student GPA Predictor & Analyzer",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ===== 2. โหลดโมเดลและ Metadata =====
@st.cache_resource
def load_model():
    # โหลดตัวโมเดล Ridge Regression
    model = joblib.load("model_artifacts/student_gpa_ridge_model.pkl")
    # โหลด Metadata และ Feature Names
    with open("model_artifacts/model_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, metadata

with st.spinner("กำลังโหลดสมองกลวิเคราะห์พฤติกรรม..."):
    model, metadata = load_model()
    features = metadata['feature_names']
    coefs = metadata['coefficients']

# ===== 3. Sidebar: ข้อมูลโมเดลและ Insight =====
with st.sidebar:
    st.header("ℹ️ ข้อมูลโมเดล")
    st.write(f"**ประเภท:** {metadata['model_type']}")
    st.write(f"**ความแม่นยำ (R²):** {metadata['r2_score']*100:.2f}%")
    st.write(f"**จำนวนข้อมูลที่ใช้เทรน:** {metadata['training_samples']:,} ราย")
    
    st.divider()
    st.subheader("💡 ปัจจัยที่มีผลต่อเกรด")
    # แสดงค่า Coefficient เพื่อบอกว่าอะไรสำคัญที่สุด
    coef_df = pd.DataFrame.from_dict(coefs, orient='index', columns=['Weight']).sort_values(by='Weight', ascending=False)
    st.dataframe(coef_df)
    st.info("ค่า Weight เป็นบวกหมายถึงส่งผลให้เกรดเพิ่มขึ้น ค่าลบหมายถึงทำให้เกรดลดลง")

# ===== 4. ส่วนหลัก: Header =====
st.title("🎓 ระบบวิเคราะห์พฤติกรรมและทำนาย GPA")
st.markdown("""
ลองกรอกพฤติกรรมประจำวันของคุณด้านล่าง เพื่อให้ AI ช่วยวิเคราะห์ว่า
**วิถีชีวิตแบบนี้จะส่งผลต่อเกรดเฉลี่ย (GPA) ของคุณอย่างไร**
""")

st.divider()

# ===== 5. ส่วนรับ Input (พฤติกรรม 6 ด้าน) =====
st.subheader("📋 บันทึกพฤติกรรมรายวัน")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.slider("ชั่วโมงการอ่านหนังสือ/วัน", 0, 15, 5, help="รวมเวลาทำการบ้านและทบทวนบทเรียน")
    attendance = st.slider("เปอร์เซ็นต์การเข้าเรียน (%)", 0, 100, 85)
    sleep_hours = st.slider("ชั่วโมงการนอนหลับ/คืน", 0, 12, 7, help="การพักผ่อนส่งผลต่อการจดจำข้อมูล")

with col2:
    procrastination = st.select_slider("ระดับการผลัดวันประกันพรุ่ง", options=list(range(1, 11)), value=5)
    screen_time = st.slider("ชั่วโมงการเล่นโซเชียล/เกมต่อวัน", 0, 15, 4)
    stress = st.select_slider("ระดับความเครียดสะสม", options=list(range(1, 11)), value=5)

st.divider()

# ===== 6. การทำนายผลและ What-if Analysis =====
# เตรียมข้อมูลสำหรับทำนาย
input_data = np.array([[study_hours, attendance, sleep_hours, procrastination, screen_time, stress]])

# ทำนาย GPA
predicted_gpa = model.predict(input_data)[0]
# Clip ให้อยู่ในช่วง 0-4
final_gpa = np.clip(predicted_gpa, 0, 4.0)

# แสดงผลลัพธ์หลัก
st.subheader("📊 ผลการวิเคราะห์")
col_res1, col_res2 = st.columns([1, 1])

with col_res1:
    st.metric(label="GPA ที่คาดการณ์", value=f"{final_gpa:.3f}")
    
with col_res2:
    # คำแนะนำด่วนจาก AI ตามค่า Weight ที่สูงที่สุด
    if sleep_hours < 7:
        st.warning("⚠️ AI แนะนำ: การนอนของคุณน้อยเกินไป ซึ่งส่งผลเสียต่อเกรดมากที่สุด")
    elif stress > 7:
        st.warning("⚠️ AI แนะนำ: ระดับความเครียดสูงเกินไป ลองหากิจกรรมผ่อนคลายเพื่อดึงเกรดกลับมา")
    else:
        st.success("✅ พฤติกรรมโดยรวมของคุณอยู่ในเกณฑ์ที่จัดการได้")

# ===== 7. ระบบจำลองการปรับปรุงตัว (Action Plan) =====
with st.expander("🚀 อยากเพิ่มเกรดต้องทำยังไง? (What-if Analysis)"):
    st.write("ลองจำลองสถานการณ์: ถ้าคุณ **เพิ่มเวลานอนเป็น 8 ชม.** และ **ลดความเครียดลง 2 ระดับ**")
    
    # จำลองข้อมูลใหม่
    sim_input = input_data.copy()
    sim_input[0][2] = 8 # เพิ่มนอน
    sim_input[0][5] = max(1, stress - 2) # ลดเครียด
    
    sim_gpa = np.clip(model.predict(sim_input)[0], 0, 4.0)
    diff = sim_gpa - final_gpa
    
    st.write(f"ผลลัพธ์: GPA ของคุณจะกลายเป็น **{sim_gpa:.3f}** (เพิ่มขึ้น {diff:+.3f})")
    st.caption("อ้างอิงจากโมเดล Ridge Regression ที่วิเคราะห์ความสัมพันธ์ของพฤติกรรม")

# บันทึกข้อมูลที่กรอกลงตารางเผื่อเช็ก
with st.expander("📋 สรุปข้อมูลที่ใช้คำนวณ"):
    summary_df = pd.DataFrame(input_data, columns=features)
    st.table(summary_df)

st.divider()
st.caption("หมายเหตุ: ข้อมูลนี้เป็นการพยากรณ์เชิงสถิติจากชุดข้อมูล 1,000,000 ราย เพื่อใช้ในการวางแผนพฤติกรรมเท่านั้น")