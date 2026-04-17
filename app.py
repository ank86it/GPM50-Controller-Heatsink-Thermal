import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import joblib
import numpy as np

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

# =========================
# 🔄 UPDATE ALL BUTTON
# =========================
if st.button("🔄 Update All Results"):

    results = {}

    # ---- MAIN RESULT ----
    tj = hybrid_predict(load, eff_m, eff_c, Ta, fin, v)
    margin = calc_margin(tj)
    results["calc"] = (tj, margin)

    # ---- HEATMAP ----
    amb = [50,40,35,30,25]
    fins = [-20,-10,0,10,20]
    data = []

    for T in amb:
        row = []
        for f in fins:
            tj_temp = hybrid_predict(load, eff_m, eff_c, T, f, v)
            row.append(calc_margin(tj_temp))
        data.append(row)

    results["heatmap"] = (amb, fins, data)

    # ---- MAX LOAD ----
    for L in np.linspace(1000,15000,120):
        if calc_margin(hybrid_predict(L, eff_m, eff_c, Ta, fin, v)) < 10:
            results["max_load"] = L-100
            break

    # ---- MIN FIN ----
    for f in np.linspace(-20,50,120):
        if calc_margin(hybrid_predict(load, eff_m, eff_c, Ta, f, v)) >= 10:
            results["min_fin"] = f
            break

    # ---- MAX AMBIENT ----
    for T in np.linspace(20,80,120):
        if calc_margin(hybrid_predict(load, eff_m, eff_c, T, fin, v)) < 10:
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

    if margin < 10:
        st.error("❌ Poor Design")
    elif margin < 20:
        st.warning("⚠️ Safe Design")
    else:
        st.success("🟩 Over Design")

# =========================
# 📊 HEATMAP
# =========================
if "heatmap" in st.session_state.results:

    amb, fins, data = st.session_state.results["heatmap"]
    df = pd.DataFrame(data, index=amb, columns=fins)

    fig, ax = plt.subplots()

    color = df.copy()
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            val = df.iloc[i,j]
            color.iloc[i,j] = 2 if val>20 else 1 if val>10 else 0

    ax.imshow(color, cmap=ListedColormap(["red","yellow","green"]))
    ax.set_xticks(range(len(fins)))
    ax.set_yticks(range(len(amb)))
    ax.set_xticklabels([f"{f}%" for f in fins])
    ax.set_yticklabels([f"{t}°C" for t in amb])

    for i in range(len(amb)):
        for j in range(len(fins)):
            ax.text(j,i,f"{df.iloc[i,j]:.1f}%",ha='center')

    st.pyplot(fig)

# =========================
# 🟢 OPTIMIZER RESULTS
# =========================
st.header("🟢 Design Optimizers")

if "max_load" in st.session_state.results:
    st.write(f"Max Load: {st.session_state.results['max_load']:.0f} W")

if "min_fin" in st.session_state.results:
    st.write(f"Minimum Fin: {st.session_state.results['min_fin']:.1f} %")

if "max_Ta" in st.session_state.results:
    st.write(f"Max Ambient: {st.session_state.results['max_Ta']:.1f} °C")

# =========================
# 📷 IMAGE
# =========================
st.markdown("---")
st.image("Controller Heatsink2.png", use_container_width=True)
