import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import joblib
import numpy as np

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Thermal Design Tool")

# ---------------- HIDE UI ----------------
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
div[data-testid="stToolbar"] {display: none;}
.block-container {max-width: 900px;}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
ml_model = joblib.load("xgb_thermal_model.pkl")

# ---------------- SESSION ----------------
if "results" not in st.session_state:
    st.session_state.results = {}

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

# ---------------- HEATMAP IMAGE (PDF) ----------------
def create_heatmap_image(load, eff_m, eff_c, v, target_margin):

    amb = [50,40,35,30,25]
    fins = [-20,-10,0,10,20]

    data = []
    for T in amb:
        row = []
        for f in fins:
            tj = hybrid_predict(load, eff_m, eff_c, T, f, v)
            row.append(calc_margin(tj))
        data.append(row)

    df = pd.DataFrame(data, index=amb, columns=fins)

    fig, ax = plt.subplots(figsize=(5,4))

    color = df.copy()
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            val = df.iloc[i,j]
            color.iloc[i,j] = 2 if val > target_margin+10 else 1 if val > target_margin else 0

    cmap = ListedColormap(["red","yellow","green"])
    ax.imshow(color, cmap=cmap)

    ax.set_xticks(range(len(fins)))
    ax.set_yticks(range(len(amb)))
    ax.set_xticklabels([f"{f}%" for f in fins])
    ax.set_yticklabels([f"{t}°C" for t in amb])

    for i in range(len(amb)):
        for j in range(len(fins)):
            ax.text(j,i,f"{df.iloc[i,j]:.1f}%",ha='center', fontsize=8)

    plt.savefig("heatmap.png")
    plt.close()

# ---------------- PDF ----------------
def generate_pdf(results, inputs, load, eff_m, eff_c, v, target_margin):

    create_heatmap_image(load, eff_m, eff_c, v, target_margin)

    doc = SimpleDocTemplate("thermal_report.pdf")
    styles = getSampleStyleSheet()
    story = []

    # TITLE
    story.append(Paragraph("Thermal Design Report", styles['Title']))
    story.append(Spacer(1, 15))

    # INPUTS
    story.append(Paragraph("Inputs:", styles['Heading2']))
    story.append(Spacer(1, 5))
    for k, v_ in inputs.items():
        story.append(Paragraph(f"{k}: {v_}", styles['Normal']))

    story.append(Spacer(1, 10))

    # RESULTS
    if "calc" in results:
        tj, margin = results["calc"]
        story.append(Paragraph("Results:", styles['Heading2']))
        story.append(Spacer(1, 5))
        story.append(Paragraph(f"Tj: {tj:.2f} °C", styles['Normal']))
        story.append(Paragraph(f"Margin: {margin:.1f} %", styles['Normal']))

    story.append(Spacer(1, 10))

    # OPTIMIZERS
    story.append(Paragraph("Optimizers:", styles['Heading2']))
    story.append(Spacer(1, 5))

    if "max_load" in results:
        story.append(Paragraph(f"Max Load: {results['max_load']:.0f} W", styles['Normal']))
    if "min_fin" in results:
        story.append(Paragraph(f"Min Fin: {results['min_fin']:.1f} %", styles['Normal']))
    if "max_Ta" in results:
        story.append(Paragraph(f"Max Ambient: {results['max_Ta']:.1f} °C", styles['Normal']))

    story.append(Spacer(1, 15))

    # HEATMAP
    story.append(Paragraph("Margin Heatmap:", styles['Heading2']))
    story.append(Spacer(1, 10))
    story.append(Image("heatmap.png", width=400, height=300))

    story.append(Spacer(1, 10))

    # LEGEND
    story.append(Paragraph("Legend:", styles['Heading3']))
    story.append(Spacer(1, 5))
    story.append(Paragraph("🟩 Over Design (> target + 10%)", styles['Normal']))
    story.append(Paragraph("🟨 Safe Design (> target margin)", styles['Normal']))
    story.append(Paragraph("🟥 Poor Design (< target margin)", styles['Normal']))

    doc.build(story)

# =========================
# INPUT
# =========================
st.title("🔥 Thermal Design Tool")

load = st.number_input("Load (W)", value=6000.0)
eff_m = st.number_input("Motor Efficiency", value=0.9000, format="%.4f")
eff_c = st.number_input("Controller Efficiency", value=0.9767, format="%.4f")
Ta = st.number_input("Ambient Temp (°C)", value=40.0)
fin = st.number_input("Fin Change (%)", value=0.0)
v = st.number_input("Air Velocity (m/s)", value=5.0)

target_margin = st.number_input("Safety Margin Target (%)", value=10.0)

# =========================
# UPDATE
# =========================
if st.button("🔄 Update All Results"):

    res = {}

    tj = hybrid_predict(load, eff_m, eff_c, Ta, fin, v)
    res["calc"] = (tj, calc_margin(tj))

    for L in np.linspace(1000,15000,120):
        if calc_margin(hybrid_predict(L, eff_m, eff_c, Ta, fin, v)) < target_margin:
            res["max_load"] = L-100
            break

    for f in np.linspace(-20,50,120):
        if calc_margin(hybrid_predict(load, eff_m, eff_c, Ta, f, v)) >= target_margin:
            res["min_fin"] = f
            break

    for T in np.linspace(20,80,120):
        if calc_margin(hybrid_predict(load, eff_m, eff_c, T, fin, v)) < target_margin:
            res["max_Ta"] = T-0.5
            break

    st.session_state.results = res

# =========================
# RESULTS
# =========================
if "calc" in st.session_state.results:
    tj, margin = st.session_state.results["calc"]
    st.write(f"Tj: {tj:.2f} °C")
    st.write(f"Margin: {margin:.1f}%")

# =========================
# HEATMAP (FIXED)
# =========================
if "calc" in st.session_state.results:

    amb = [50,40,35,30,25]
    fins = [-20,-10,0,10,20]

    data = []
    for T in amb:
        row = []
        for f in fins:
            tj = hybrid_predict(load, eff_m, eff_c, T, f, v)
            row.append(calc_margin(tj))
        data.append(row)

    df = pd.DataFrame(data, index=amb, columns=fins)

    fig, ax = plt.subplots(figsize=(5,4))

    color = df.copy()
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            val = df.iloc[i,j]
            color.iloc[i,j] = 2 if val > target_margin+10 else 1 if val > target_margin else 0

    cmap = ListedColormap(["red","yellow","green"])
    ax.imshow(color, cmap=cmap)

    for i in range(len(amb)):
        for j in range(len(fins)):
            ax.text(j,i,f"{df.iloc[i,j]:.1f}%",ha='center', fontsize=8)

    legend = [
        mpatches.Patch(color='green', label='Over Design'),
        mpatches.Patch(color='yellow', label='Safe Design'),
        mpatches.Patch(color='red', label='Poor Design')
    ]

    ax.legend(handles=legend, bbox_to_anchor=(1.25,1))

    st.pyplot(fig)

# =========================
# OPTIMIZERS
# =========================
if "max_load" in st.session_state.results:
    st.write(f"Max Load: {st.session_state.results['max_load']:.0f} W")

if "min_fin" in st.session_state.results:
    st.write(f"Min Fin: {st.session_state.results['min_fin']:.1f}%")

if "max_Ta" in st.session_state.results:
    st.write(f"Max Ambient: {st.session_state.results['max_Ta']:.1f} °C")

# =========================
# PDF
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

    generate_pdf(st.session_state.results, inputs, load, eff_m, eff_c, v, target_margin)

    with open("thermal_report.pdf", "rb") as f:
        st.download_button("Download PDF", f, "thermal_report.pdf")

# =========================
# IMAGE
# =========================
st.markdown("---")
st.image("Controller Heatsink2.png", use_container_width=True)
