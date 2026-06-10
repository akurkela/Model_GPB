import streamlit as st
import matplotlib.pyplot as plt

from model import evaluate

# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="GPB EoS Prior",
    layout="wide"
)

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------

st.title("Gaussian-Process Bridge Extension for Neutron-Star Equations of State")

st.markdown(r"""
This application generates a **prior ensemble of allowed equation-of-state
(EoS) extensions** connecting a low-density neutron-star matter model to the
high-density perturbative QCD (pQCD) regime. Based on arXiv:xxxx.xxxx

The prior is conditioned on the **termination point**

$$
(\mu_L,\; n_L,\; p_L),
$$

where

- $\mu_L$ [GeV] is the baryon chemical potential,
- $n_L$ [1/fm^3] is the baryon number density ,
- $p_L$ [GeV/fm^3] is the pressure.

The generated interpolations satisfy the constraints of

- thermodynamic consistency,
- mechanical stability,
- causality,

while connecting to the perturbative-QCD equation of state at high density.

Enter the termination point and press **Generate prior**.
""")

st.divider()

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

with st.sidebar:

    st.header("Termination point")

    muL = st.number_input(
        r"$\mu_L$",
        value=1000.0,
        format="%.6f"
    )

    nL = st.number_input(
        r"$n_L$",
        value=0.600000,
        format="%.6f"
    )

    pL = st.number_input(
        r"$p_L$",
        value=100.000000,
        format="%.6f"
    )

    st.divider()

    number_of_interpolations = st.number_input(
        "Number of interpolations",
        min_value=2,
        max_value=100000,
        value=300,
        step=1
    )

    compute = st.button(
        "Generate prior",
        use_container_width=True
    )

# -----------------------------------------------------------------------------
# Main panel
# -----------------------------------------------------------------------------

if compute:

    x, y = evaluate(
        muL,
        nL,
        pL,
        number_of_interpolations
    )

    st.subheader("Prior ensemble of allowed extensions")

    st.write(
        "The figure below shows the prior generated from the supplied "
        "termination point. In the full model this corresponds to an "
        "ensemble of thermodynamically consistent EoS extensions that "
        "connect the low-density model to the perturbative-QCD regime."
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        x,
        y,
        linewidth=2
    )

    ax.set_xlabel(r"$x$")

    ax.set_ylabel(r"$y$")

    ax.set_title(
        r"Prior conditioned on $(\mu_L,n_L,p_L)$"
    )

    ax.grid(True)

    st.pyplot(fig)

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------

st.divider()

st.caption(
    "This application implements the Gaussian-Process Bridge (GPB) prior "
    "for neutron-star equation-of-state extensions constrained by "
    "thermodynamic consistency, causality, and perturbative QCD."
)
