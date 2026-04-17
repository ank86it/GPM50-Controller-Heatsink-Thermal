import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import joblib
import numpy as np

# ✅ PDF LIB
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Thermal Design Tool")

# ---------------- HIDE STREAMLIT UI ----------------
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
div[data-testid="stToolbar"] {display: none;}

.block-container {
    max-width: 900px;
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
ml_model = joblib.load("xgb_thermal_model.pkl")

# ---------------- SESSION STATE ----------------
if "results" not in st.session_state:
    st.session_state.results = {}

# ---------------- PDF FUNCTION ----------------
def generate_pdf(results, inputs, filename="thermal_report.pdf"):
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Thermal Design Report", styles['Title']))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Inputs:", styles['Heading2']))
    for key, val in inputs.items():
        story.append(Paragraph(f"{key}: {val}", styles['Normal']))

    story.append(Spacer(1, 10))

    if "calc" in results:
        tj, margin = results["calc"]
        story.append(Paragraph("Results:", styles['Heading2']))
        story.append(Paragraph(f"Tj: {tj:.2f} °C", styles['Normal']))
        story.append(Paragraph(f"Margin: {margin:.1f} %", styles['Normal']))

    story.append(Spacer(1, 10))
    story.append(Paragraph("Optimizers:", styles['Heading2']))

    if "max_load" in results:
        story.append(Paragraph(f"Max Load: {results['max_load']:.0f} W", styles['Normal']))

    if "min_fin" in results:
        story.append(Paragraph(f"Min Fin: {results['min_fin']:.1f} %", styles['Normal']))

    if "max_Ta" in results:
        story.append(Paragraph(f"Max Ambient: {results['max_Ta']:.1f} °C", styles['Normal']))

    doc.build(story)

# ---------------- PHYSICS ----------------
def interpolate_h(v, v_points, h_points):
    for i in range(len(v_points)-1):
        if v_points[i] <= v <= v_points[i+1]:
            return h_points[i] + (v - v_points[i]) * (h_points[i+1] - h_points[i]) / (v_points[i+1] - v_points[i])
    return h_points[-1]

def thermal_model(load, eff_m, eff_c, Ta, fin, v):
    motor_input = load / eff_m
    controller_input = motor_input / eff_c
    loss = controller_input - motor_input

    A_bc, A_bf = 0.0386, 0.0573
    h_bf = interpolate_h(v, [2,5,10], [11,18,23])
    h_bc = interpolate_h(v, [2,5,10], [13,18,21])

    A_fin = A_bf*(1+fin/100)
    A_total = A_bc + A_fin

    h = (h_bc*A_bc + h_bf*A_fin)/A_total
    R = 1/(h*A_total) + 0.064 - 0.043 - 0.17

    return Ta + loss*R + (loss/24)*0.38

def hybrid_predict(load, eff_m, eff_c, Ta, fin, v):
    tj_p = thermal_model(load, eff_m, eff_c, Ta, fin, v)

    inp = [[load, eff_m, eff_c, Ta, fin, v,
            load*(1-eff_m),
            v*(1+fin/100),
            1/(v*(1+fin/100)+0.1)]]

    return tj_p + ml_model.predict(inp)[0]

def calc_margin(tj):
    return ((125 - tj) / 125) * 100

# =========================
# 🔵 INPUT SECTION
# =========================
st.title("🔥 Thermal Design Tool")
st.header("🔵 Design Evaluation")

load = st.number_input("Load (W)", value=6000.0)
eff_m = st.number_input("Motor Efficiency", value=0.9000, format="%.4f")
eff_c = st.number_input("Controller Efficiency", value=0.9767, format="%.4f")
Ta = st.number_input("Ambient Temp (°C)", value=40.0)
fin = st.number_input("Fin Change (%)", value=0.0)
v = st.number_input("Air Velocity (m/s)", value=5.0)

target_margin = st.number_input("Safety Margin Target (%)", value=10.0)

# =========================
# 🔄 UPDATE ALL BUTTON
# =========================
if st.button("🔄 Update All Results"):

    results = {}

    tj = hybrid_predict(load, eff_m, eff_c, Ta, fin, v)
    margin = calc_margin(tj)
    results["calc"] = (tj, margin)

    for L in np.linspace(1000,15000,120):
        if calc_margin(hybrid_predict(L, eff_m, eff_c, Ta, fin, v)) < target_margin:
            results["max_load"] = L-100
            break

    for f in np.linspace(-20,50,120):
        if calc_margin(hybrid_predict(load, eff_m, eff_c, Ta, f, v)) >= target_margin:
            results["min_fin"] = f
            break

    for T in np.linspace(20,80,120):
        if calc_margin(hybrid_predict(load, eff_m, eff_c, T, fin, v)) < target_margin:
            results["max_Ta"] = T-0.5
            break

    st.session_state.results = results

# =========================
# 🔵 DISPLAY RESULTS
# =========================
if "calc" in st.session_state.results:
    tj, margin = st.session_state.results["calc"]

    st.subheader("Results")
    st.write(f"Tj: {tj:.2f} °C")
    st.write(f"Margin: {margin:.1f} %")
    st.write(f"Target Margin: {target_margin:.1f} %")

# =========================
# 🟢 OPTIMIZERS
# =========================
st.header(f"🟢 Design Optimizers (Target Margin: {target_margin:.1f}%)")

if "max_load" in st.session_state.results:
    st.write(f"Max Load: {st.session_state.results['max_load']:.0f} W")

if "min_fin" in st.session_state.results:
    st.write(f"Minimum Fin: {st.session_state.results['min_fin']:.1f} %")

if "max_Ta" in st.session_state.results:
    st.write(f"Max Ambient: {st.session_state.results['max_Ta']:.1f} °C")

# =========================
# 📄 PDF EXPORT
# =========================
if st.button("📄 Save Results to PDF"):

    inputs = {
        "Load": load,
        "Motor Eff": eff_m,
        "Controller Eff": eff_c,
        "Ambient": Ta,
        "Fin %": fin,
        "Velocity": v,
        "Target Margin": target_margin
    }

    generate_pdf(st.session_state.results, inputs)

    with open("thermal_report.pdf", "rb") as f:
        st.download_button(
            label="Download PDF",
            data=f,
            file_name="thermal_report.pdf",
            mime="application/pdf"
        )

# =========================
# 📷 IMAGE
# =========================
st.markdown("---")
st.image("Controller Heatsink2.png", use_container_width=True)
