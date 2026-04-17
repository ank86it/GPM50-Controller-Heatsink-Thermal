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

/* Centered layout */
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
            v1, v2 = v_points[i], v_points[i+1]
            h1, h2 = h_points[i], h_points[i+1]
            return h1 + (v - v1)*(h2 - h1)/(v2 - v1)
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

    T_case = Ta + loss*R
    return T_case + (loss/24)*0.38

def hybrid_predict(load, eff_m, eff_c, Ta, fin, v):
    tj_p = thermal_model(load, eff_m, eff_c, Ta, fin, v)

    inp = [[load, eff_m, eff_c, Ta, fin, v,
            load*(1-eff_m),
            v*(1+fin/100),
            1/(v*(1+fin/100)+0.1)]]

    delta = ml_model.predict(inp)[0]
    return tj_p + delta

def calc_margin(tj):
    return ((125 - tj) / 125) * 100

# =========================
# 🔵 SECTION 1: ANALYSIS
# =========================
st.title("🔥 Thermal Design Tool")
st.header("🔵 Design Evaluation")

load = st.number_input("Load (W)", value=6000.0)
eff_m = st.number_input("Motor Efficiency", value=0.9000, format="%.4f")
eff_c = st.number_input("Controller Efficiency", value=0.9767, format="%.4f")
Ta = st.number_input("Ambient Temp (°C)", value=40.0)
fin = st.number_input("Fin Change (%)", value=0.0)
v = st.number_input("Air Velocity (m/s)", value=5.0)

# ---------------- CALCULATE ----------------
if st.button("Calculate"):
    tj = hybrid_predict(load, eff_m, eff_c, Ta, fin, v)
    margin = calc_margin(tj)
    st.session_state.results["calc"] = (tj, margin)

# ---------------- SHOW RESULT ----------------
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

# ---------------- HEATMAP ----------------
if st.button("Show Margin Map"):

    amb = [50,40,35,30,25]
    fins = [-20,-10,0,10,20]

    data = []
    for T in amb:
        row = []
        for f in fins:
            tj = hybrid_predict(load, eff_m, eff_c, T, f, v)
            row.append(calc_margin(tj))
        data.append(row)

    st.session_state.results["heatmap"] = (amb, fins, data)

# ---------------- SHOW HEATMAP ----------------
if "heatmap" in st.session_state.results:

    amb, fins, data = st.session_state.results["heatmap"]
    df = pd.DataFrame(data, index=amb, columns=fins)

    fig, ax = plt.subplots()

    color = df.copy()
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            val = df.iloc[i,j]
            color.iloc[i,j] = 2 if val>20 else 1 if val>10 else 0

    cmap = ListedColormap(["red","yellow","green"])
    ax.imshow(color, cmap=cmap)

    ax.set_xticks(range(len(fins)))
    ax.set_yticks(range(len(amb)))

    ax.set_xticklabels([f"{f}%" for f in fins])
    ax.set_yticklabels([f"{t}°C" for t in amb])

    ax.set_xlabel("Fin Area Change (%)")
    ax.set_ylabel("Ambient Temperature (°C)")

    for i in range(len(amb)):
        for j in range(len(fins)):
            ax.text(j,i,f"{df.iloc[i,j]:.1f}%",ha='center')

    legend = [
        mpatches.Patch(color='green', label='Over Design'),
        mpatches.Patch(color='yellow', label='Safe Design'),
        mpatches.Patch(color='red', label='Poor Design')
    ]

    ax.legend(handles=legend, bbox_to_anchor=(1.4,1))
    st.pyplot(fig)

# =========================
# 🟢 SECTION 2: OPTIMIZERS
# =========================
st.header("🟢 Design Optimizers (10% Margin Target)")

# MAX LOAD
if st.button("Find Max Load"):
    for L in np.linspace(1000,15000,120):
        tj = hybrid_predict(L, eff_m, eff_c, Ta, fin, v)
        if calc_margin(tj) < 10:
            st.session_state.results["max_load"] = L-100
            break

if "max_load" in st.session_state.results:
    st.write(f"Max Load: {st.session_state.results['max_load']:.0f} W")

# MIN FIN
if st.button("Find Minimum Fin"):
    for f in np.linspace(-20,50,120):
        tj = hybrid_predict(load, eff_m, eff_c, Ta, f, v)
        if calc_margin(tj) >= 10:
            st.session_state.results["min_fin"] = f
            break

if "min_fin" in st.session_state.results:
    st.write(f"Minimum Fin: {st.session_state.results['min_fin']:.1f} %")

# MAX AMBIENT
if st.button("Find Max Ambient Temperature"):
    for T in np.linspace(20,80,120):
        tj = hybrid_predict(load, eff_m, eff_c, T, fin, v)
        if calc_margin(tj) < 10:
            st.session_state.results["max_Ta"] = T-0.5
            break

if "max_Ta" in st.session_state.results:
    st.write(f"Max Ambient: {st.session_state.results['max_Ta']:.1f} °C")

# =========================
# 📷 IMAGE
# =========================
st.markdown("---")
st.image("Controller Heatsink2.png", use_container_width=True)
