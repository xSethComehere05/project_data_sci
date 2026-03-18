# app.py — Streamlit application สำหรับทำนายความเสี่ยงโรคเบาหวาน

import streamlit as st
import numpy as np
import joblib
import json

# ===== การตั้งค่าหน้าเว็บ =====
# st.set_page_config ต้องเป็น Streamlit command แรกเสมอ
# ถ้าเรียกทีหลังจะ error
st.set_page_config(
    page_title="ระบบทำนายความเสี่ยงโรคเบาหวาน",
    page_icon="🩺",          # icon ที่แสดงบน browser tab
    layout="centered",        # centered หรือ wide
    initial_sidebar_state="expanded"
)

# ===== โหลดโมเดลและข้อมูล =====
# ใช้ @st.cache_resource เพื่อโหลดโมเดลครั้งเดียว
# โดยไม่โหลดซ้ำทุกครั้งที่ผู้ใช้ interact กับ app
# นี่คือ performance optimization ที่สำคัญมาก
@st.cache_resource
def load_model():
    """โหลด pipeline และ metadata — ทำครั้งเดียวตอนเริ่ม app"""
    pipeline = joblib.load("model_artifacts/diabetes_pipeline.pkl")
    with open("model_artifacts/model_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return pipeline, metadata

# โหลดโมเดล — Streamlit จะแสดง spinner ระหว่างรอ
with st.spinner("กำลังโหลดโมเดล..."):
    pipeline, metadata = load_model()

# ===== Sidebar: ข้อมูลเกี่ยวกับโมเดล =====
# Sidebar เหมาะสำหรับข้อมูลเสริมที่ไม่ใช่ส่วนหลักของ app
with st.sidebar:
    st.header("ℹ️ เกี่ยวกับโมเดลนี้")
    st.write(f"**ประเภทโมเดล:** {metadata['model_type']}")
    st.write(f"**ความแม่นยำ:** {metadata['accuracy']*100:.1f}%")
    st.write(f"**ข้อมูล train:** {metadata['training_samples']:,} ราย")

    st.divider()  # เส้นคั่น

    st.subheader("⚠️ ข้อควรระวัง")
    st.warning(
        "ผลลัพธ์นี้เป็นการประเมินเบื้องต้นจาก AI เท่านั้น "
        "ไม่สามารถใช้แทนการวินิจฉัยของแพทย์ได้ "
        "กรุณาปรึกษาแพทย์หากมีข้อสงสัย"
    )

# ===== ส่วนหลัก: Header =====
st.title("🩺 ระบบประเมินความเสี่ยงโรคเบาหวาน")
st.markdown("""
กรอกข้อมูลสุขภาพของคุณด้านล่าง ระบบจะประเมินความเสี่ยงการเป็นโรคเบาหวาน
โดยใช้โมเดล Machine Learning ที่ train จากข้อมูลผู้ป่วย 768 ราย
""")

st.divider()

# ===== ส่วนรับ Input =====
st.subheader("📋 กรอกข้อมูลสุขภาพ")

# แบ่งหน้าจอเป็นสองคอลัมน์เพื่อให้ดูกระชับขึ้น
col1, col2 = st.columns(2)

with col1:
    # st.number_input สำหรับตัวเลขที่ต้องการความแม่นยำ
    pregnancies = st.number_input(
        "จำนวนครั้งที่ตั้งครรภ์",
        min_value=0, max_value=20,
        value=1, step=1,
        help="กรอก 0 หากไม่เคยตั้งครรภ์"
    )

    glucose = st.number_input(
        "ระดับน้ำตาลในเลือด (mg/dL)",
        min_value=0, max_value=300,
        value=120, step=1,
        help="ค่าปกติ: 70–100 mg/dL (อดอาหาร)"
    )

    blood_pressure = st.number_input(
        "ความดันโลหิต Diastolic (mmHg)",
        min_value=0, max_value=150,
        value=72, step=1,
        help="ค่าตัวล่างของความดันโลหิต"
    )

    skin_thickness = st.number_input(
        "ความหนาผิวหนัง Tricep (mm)",
        min_value=0, max_value=100,
        value=20, step=1,
        help="วัดที่ต้นแขนด้านหลัง"
    )

with col2:
    insulin = st.number_input(
        "ระดับ Insulin (mu U/ml)",
        min_value=0, max_value=900,
        value=80, step=1,
        help="ระดับ insulin ในเลือด 2 ชั่วโมงหลังทดสอบ"
    )

    bmi = st.number_input(
        "ดัชนีมวลกาย BMI (kg/m²)",
        min_value=0.0, max_value=70.0,
        value=25.0, step=0.1,
        format="%.1f",
        help="น้ำหนัก(kg) ÷ ส่วนสูง(m)²"
    )

    dpf = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0, max_value=3.0,
        value=0.35, step=0.001,
        format="%.3f",
        help="ค่าที่สะท้อนความเสี่ยงทางพันธุกรรม (0–2.5)"
    )

    age = st.number_input(
        "อายุ (ปี)",
        min_value=1, max_value=120,
        value=30, step=1
    )

st.divider()

# ===== ปุ่มทำนายและแสดงผล =====
# การใช้ column เพื่อจัดปุ่มให้อยู่กลาง
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_button = st.button(
        "🔍 ประเมินความเสี่ยง",
        use_container_width=True,
        type="primary"  # ทำให้ปุ่มสีเด่น
    )

# เมื่อผู้ใช้กดปุ่ม จึงค่อยทำการทำนาย
# การตรวจสอบ if predict_button ทำให้ app ไม่ทำนายโดยอัตโนมัติตอนโหลด
if predict_button:

    # รวบรวม input เป็น array ในลำดับเดียวกับที่ train
    # ลำดับนี้ต้องตรงกับ feature_names ที่บันทึกไว้จาก Colab
    input_data = np.array([[
        pregnancies, glucose, blood_pressure,
        skin_thickness, insulin, bmi, dpf, age
    ]])

    # ทำนายด้วย pipeline — การ scale เกิดขึ้นอัตโนมัติภายใน pipeline
    with st.spinner("กำลังประเมิน..."):
        prediction = pipeline.predict(input_data)[0]
        probabilities = pipeline.predict_proba(input_data)[0]

    # ดึงค่า probability ของแต่ละ class
    prob_negative = probabilities[0]  # ความน่าจะเป็นที่จะไม่เป็นเบาหวาน
    prob_positive = probabilities[1]  # ความน่าจะเป็นที่จะเป็นเบาหวาน

    st.subheader("📊 ผลการประเมิน")

    # แสดงผลต่างกันตาม prediction
    if prediction == 1:
        st.error(f"""
        ### ⚠️ ความเสี่ยงสูง
        ผลการประเมินพบว่ามีความเสี่ยงในการเป็นโรคเบาหวาน
        **ความน่าจะเป็น: {prob_positive*100:.1f}%**
        """)
    else:
        st.success(f"""
        ### ✅ ความเสี่ยงต่ำ
        ผลการประเมินไม่พบความเสี่ยงสูงในการเป็นโรคเบาหวาน
        **ความน่าจะเป็น: {prob_negative*100:.1f}%**
        """)

    # แสดง probability gauge ด้วย progress bar
    st.write("**ระดับความเสี่ยง:**")
    st.progress(
        float(prob_positive),
        text=f"โอกาสเป็นเบาหวาน: {prob_positive*100:.1f}%"
    )

    # แสดงข้อมูลที่กรอกสรุปกลับ เพื่อให้ผู้ใช้ตรวจสอบว่าถูกต้อง
    with st.expander("📋 ดูข้อมูลที่กรอก"):
        summary = {
            "จำนวนครั้งตั้งครรภ์": pregnancies,
            "ระดับน้ำตาล (mg/dL)": glucose,
            "ความดันโลหิต (mmHg)": blood_pressure,
            "ความหนาผิวหนัง (mm)": skin_thickness,
            "Insulin (mu U/ml)": insulin,
            "BMI": bmi,
            "Diabetes Pedigree Function": dpf,
            "อายุ (ปี)": age
        }
        import pandas as pd
        st.dataframe(
            pd.DataFrame.from_dict(summary, orient="index", columns=["ค่า"]),
            use_container_width=True
        )