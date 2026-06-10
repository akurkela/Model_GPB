import streamlit as st
import matplotlib.pyplot as plt

from model import evaluate

st.set_page_config(
    page_title="Scientific Plotter",
    layout="centered"
)

st.title("Scientific Plotter")

with st.form("parameters"):

    muL = st.number_input(
        "muL",
        value=0.2,
        format="%.6f"
    )

    nL = st.number_input(
        "nL",
        value=0.5,
        format="%.6f"
    )

    pL = st.number_input(
        "pL",
        value=0.3,
        format="%.6f"
    )

    number_of_interpolations = st.number_input(
        "Number of interpolations",
        min_value=2,
        max_value=100000,
        value=300,
        step=1
    )

    compute = st.form_submit_button("Compute")

if compute:

    x, y = evaluate(
        muL,
        nL,
        pL,
        number_of_interpolations
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(x, y, linewidth=2)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_title(r"$y=\mu_L+n_Lx+p_Lx^3$")

    ax.grid(True)

    st.pyplot(fig)
