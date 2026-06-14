from extensions import extension_generation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmcrameri.cm as cmc
import matplotlib.colors as mcolors

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

st.markdown(r"Authors: T. Gorda, O. Komoltsev, A. Kurkela")

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

    df = extension_generation(
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

    fontsize = 15

    eL = -pL + nL * muL


    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    fig.subplots_adjust(wspace=0.25)

    colors = [cmc.devon(0.75), cmc.acton(0.58), cmc.acton(0.15)]
    cmap = mcolors.ListedColormap(colors)

    for i in range(len(df)):
        if ((df.cs2[i]) <= 1.0).all():
            xnorm = (np.log(df.pQCD_X[i]) - np.log(1)) / (np.log(4) - np.log(1))
            keywords_1 = {"lw": 0.8, "color": cmap(xnorm), "alpha": 0.45 + 0.35 * float(xnorm), "zorder": int(10 * xnorm) + 1}

            axs[0].plot(df.n[i] / 0.16, df.cs2[i], **keywords_1)
            axs[1].plot(df.n[i] / 0.16, 1 / 3 - np.array(df.p[i]) / np.array(df.e[i]), **keywords_1)
            axs[2].plot(df.e[i], np.array(df.p[i]), **keywords_1)


    axs[0].set_ylim(0, 1)
    axs[1].set_ylim(-0.25, 0.35)
    axs[0].set_xlim(nL / 0.16, 40)
    axs[1].set_xlim(nL / 0.16 - 0.1*nL, 40)
    axs[2].set_xlim(eL - eL*0.1, 10)
    axs[2].set_ylim(pL - pL*0.1, 4)

    axs[1].scatter(nL/0.16,1/3-pL/eL, color='black', zorder=100, s=50)
    axs[2].scatter(eL,pL, color='black', zorder=100, s=50)

    for j in range(3):
        axs[j].set_xscale("log")
        axs[j].tick_params(labelsize=fontsize)

    axs[2].set_yscale("log")

    label_kwargs = dict(color="black", fontfamily="Helvetica Neue")

    axs[0].set_xlabel(r"n [n$_{\mathrm{sat}}$]", **label_kwargs, fontsize=fontsize, labelpad=2)
    axs[0].set_ylabel(r"$c_s^2$", **label_kwargs, fontsize=fontsize)

    axs[1].set_xlabel(r"n [n$_{\mathrm{sat}}$]", **label_kwargs, labelpad=2, fontsize=fontsize)
    axs[1].set_ylabel(r"$\Delta$", **label_kwargs, fontsize=fontsize)

    axs[2].set_xlabel(r"$\varepsilon$ [GeV fm$^{-3}$]", **label_kwargs, labelpad=2, fontsize=fontsize)
    axs[2].set_ylabel(r"$p$ [GeV fm$^{-3}$]", **label_kwargs, fontsize=fontsize)

    bounds = np.linspace(0, 1.0, 4)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])

    cax = fig.add_axes([0.93, 0.24, 0.01, 0.5])
    cbar = fig.colorbar(mappable, cax=cax, ticks=[0.0, 0.5, 1], location="right", pad=0.0)

    cbar.set_ticklabels(["1/2", "1", "2"], color="black", fontsize=fontsize)
    cbar.ax.set_title("pQCD $X$", x=1.85, fontsize=fontsize, fontfamily="Helvetica Neue", color="black", pad=15)
    cbar.ax.tick_params(labelsize=fontsize)

    axs[1].axhline(0.0, color="gray", lw=0.7, ls="--")


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
