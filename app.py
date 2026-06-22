from extensions import extension_generation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmcrameri.cm as cmc
import matplotlib.colors as mcolors
import streamlit as st
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

st.markdown(r"Authors: T. Gorda, O. Komoltsev, A. Kurkela, J. Schaffner-Bielich")

st.markdown("Based on arXiv:2606:XXXXX")

st.markdown(r"""
This application generates a **prior ensemble of allowed equation-of-state
(EoS) extensions** connecting a low-density neutron-star matter model to the
high-density perturbative QCD (pQCD) regime.

The prior is given by the Gaussian-Process Bridge [Astrophys.J. 1002 (2026) 1, 40] and is conditioned on the termination point **$(\mu_\mathrm{term},\; n_\mathrm{term},\; p_\mathrm{term})$**,

where

- $\mu_\mathrm{term}$ [GeV] is the baryon chemical potential,
- $n_\mathrm{term}$ [1/fm^3] is the baryon number density ,
- $p_\mathrm{term}$ [GeV/fm^3] is the pressure.

The generated interpolations satisfy the constraints of

- thermodynamic consistency,
- mechanical stability,
- causality,

while smoothly connecting to the perturbative-QCD equation of state at high density [Phys. Rev. Lett. 127, 162003].

Enter the termination point and press **Generate prior**.
""")

st.divider()

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

with st.sidebar:

    st.header("Termination point")

    muL = st.number_input(
        r"$\mu_\mathrm{term}\, [\text{GeV}]$",
        value=1.6,
        format="%.3f"
    )

    nL = st.number_input(
        r"$n_\mathrm{term}\, [\text{fm}^{-3}]$",
        value=0.85,
        format="%.3f"
    )

    pL = st.number_input(
        r"$p_\mathrm{term}\,[\mathrm{GeV}\,\mathrm{fm}^{-3}]$",
        value=0.4,
        format="%.3f"
    )

    st.divider()

    number_of_interpolations = st.number_input(
        "Number of interpolations",
        min_value=2,
        max_value=100000,
        value=100,
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

    fontsize = 20

    eL = -pL + nL * muL


    fig, axs = plt.subplots(1, 3, figsize=(20, 6), dpi=300)
    fig.subplots_adjust(wspace=0.35)

    colors = [cmc.devon(0.75), cmc.acton(0.58), cmc.acton(0.15)]
    cmap = mcolors.ListedColormap(colors)

    for i in range(len(df)):
        if ((df.cs2[i]) <= 1.0).all():
            xnorm = (np.log(df.pQCD_X[i]) - np.log(1)) / (np.log(4) - np.log(1))
            keywords_1 = {"lw": 0.8, "color": cmap(xnorm), "alpha": 0.45 + 0.35 * float(xnorm), "zorder": int(10 * xnorm) + 1}

            axs[0].plot(df.n[i] , df.cs2[i], **keywords_1)
            axs[1].plot(df.n[i] , 1 / 3 - np.array(df.p[i]) / np.array(df.e[i]), **keywords_1)
            axs[2].plot(df.e[i], np.array(df.p[i]), **keywords_1)


    axs[1].scatter(nL/,1/3-pL/eL, color='black', zorder=100, s=50)
    axs[2].scatter(eL,pL, color='black', zorder=100, s=50)


    for j in range(3):
        axs[j].set_xscale("log")
        axs[j].tick_params(axis='both', which='major', labelsize=fontsize)
        axs[j].tick_params(axis='both', which='minor', labelbottom=False, labelleft=False)
    for j in range(2):
        ticks = axs[j].get_xticks()
        axs[j].set_xticks(np.append(ticks, 40))
        ticks = axs[j].get_xticks()
        labels = [f"{t:g}" for t in ticks]
        for i, t in enumerate(ticks):
            if np.isclose(t, 40):
                labels[i] = "40"
    
        axs[j].set_xticklabels(labels)
    
    axs[2].set_yscale("log")
    
    axs[2].plot(
    [eL, eL],
    [pL - 0.1*pL, pL-0.08*pL],
    #transform=axs[2].get_xaxis_transform(),
    color="k",
    lw=1.5,
    clip_on=False,
    )
    
    axs[2].plot([eL - 0.1*eL, eL-0.08*eL],
        [pL, pL],
        #transform=axs[2].get_xaxis_transform(),
        color="k",
        lw=1.5,
        clip_on=False,
    )
    
    axs[0].set_ylim(0, 1)
    #axs[1].set_ylim(-0.25, 0.35)
    axs[0].set_xlim(nL , 40)
    axs[1].set_xlim(nL  - 0.1*nL, 40)
    axs[2].set_xlim(eL - eL*0.1, 10)
    axs[2].set_ylim(pL - pL*0.1, 3.5)

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


    st.pyplot(fig, use_container_width=False)

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------

st.divider()

st.caption(
    "This application implements the Gaussian-Process Bridge (GPB) prior "
    "for neutron-star equation-of-state extensions constrained by "
    "thermodynamic consistency, causality, and perturbative QCD."
)
