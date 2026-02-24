import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import qualitative
from scipy.stats import beta

st.set_page_config(page_title="Bayesian A/B Updating", layout="wide")


# =====================================================
# Helpers
# =====================================================
def init_arm(name, a0, b0):
    if name not in st.session_state:
        st.session_state[name] = {
            "alpha": float(a0),
            "beta": float(b0),
            "history": [(float(a0), float(b0))],
        }


def update_arm(name, s, f):
    arm = st.session_state[name]
    arm["alpha"] += s
    arm["beta"] += f
    arm["history"].append((arm["alpha"], arm["beta"]))


def reset_arm(name, a0, b0):
    st.session_state[name] = {
        "alpha": float(a0),
        "beta": float(b0),
        "history": [(float(a0), float(b0))],
    }


def posterior_mean_ci(alpha, beta_, level=0.95):
    mean = alpha / (alpha + beta_)
    lo, hi = beta.ppf(
        [(1 - level) / 2, 1 - (1 - level) / 2],
        alpha,
        beta_,
    )
    return mean, lo, hi


# =====================================================
# Sidebar: priors and data entry
# =====================================================
st.sidebar.header("Priors")

a0_A = st.sidebar.number_input("A prior α", value=20.0, min_value=0.1)
b0_A = st.sidebar.number_input("A prior β", value=20.0, min_value=0.1)

a0_B = st.sidebar.number_input("B prior α", value=1.0, min_value=0.1)
b0_B = st.sidebar.number_input("B prior β", value=1.0, min_value=0.1)

init_arm("A", a0_A, b0_A)
init_arm("B", a0_B, b0_B)

st.sidebar.divider()
st.sidebar.header("Add data")

sA = st.sidebar.number_input("A successes", min_value=0, value=50)
fA = st.sidebar.number_input("A failures", min_value=0, value=50)

sB = st.sidebar.number_input("B successes", min_value=0, value=60)
fB = st.sidebar.number_input("B failures", min_value=0, value=40)

if st.sidebar.button("Update posteriors"):
    if sA + fA > 0:
        update_arm("A", sA, fA)
    if sB + fB > 0:
        update_arm("B", sB, fB)

if st.sidebar.button("Reset to priors"):
    reset_arm("A", a0_A, b0_A)
    reset_arm("B", a0_B, b0_B)

# =====================================================
# Extract current state
# =====================================================
A = st.session_state["A"]
B = st.session_state["B"]

alpha_A, beta_A = A["alpha"], A["beta"]
alpha_B, beta_B = B["alpha"], B["beta"]

mean_A, lo_A, hi_A = posterior_mean_ci(alpha_A, beta_A)
mean_B, lo_B, hi_B = posterior_mean_ci(alpha_B, beta_B)

# =====================================================
# Layout
# =====================================================
st.title("Bayesian A/B Updating (Beta–Binomial)")

col1, col2 = st.columns([2, 1])

# =====================================================
# Panel 1: Posterior PDFs + CIs
# =====================================================
with col1:
    st.subheader("Posterior distributions")

    x = np.linspace(0, 1, 600)

    pdf_A = beta.pdf(x, alpha_A, beta_A)
    pdf_B = beta.pdf(x, alpha_B, beta_B)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=pdf_A,
            mode="lines",
            name="A posterior",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=pdf_B,
            mode="lines",
            name="B posterior",
        )
    )

    # get default colors
    colors = qualitative.Plotly

    # CI markers
    fig.add_vline(x=lo_A, line_dash="dot", opacity=0.4, line_color=colors[0])
    fig.add_vline(x=hi_A, line_dash="dot", opacity=0.4, line_color=colors[0])
    fig.add_vline(x=lo_B, line_dash="dot", opacity=0.4, line_color=colors[1])
    fig.add_vline(x=hi_B, line_dash="dot", opacity=0.4, line_color=colors[1])

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="θ",
        yaxis_title="Density",
        margin=dict(l=40, r=40, t=40, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"""
**A:** mean = {mean_A:.3f}, 95% CI = [{lo_A:.3f}, {hi_A:.3f}]  
**B:** mean = {mean_B:.3f}, 95% CI = [{lo_B:.3f}, {hi_B:.3f}]
"""
    )

# =====================================================
# Panel 2: Decision metrics
# =====================================================
with col2:
    st.subheader("Decision-relevant quantities")

    # Monte Carlo
    rng = np.random.default_rng(42)
    n_mc = 200_000

    samples_A = rng.beta(alpha_A, beta_A, size=n_mc)
    samples_B = rng.beta(alpha_B, beta_B, size=n_mc)

    p_B_gt_A = np.mean(samples_B > samples_A)

    # Expected loss of choosing B (vs optimal)
    loss_B = np.mean(np.maximum(samples_A - samples_B, 0.0))

    st.metric("P(B > A)", f"{p_B_gt_A:.3f}")
    st.metric("Expected loss if choosing B", f"{loss_B:.4f}")

    st.caption(
        """
- **P(B > A)** estimated via Monte Carlo  
- **Expected loss** = E[max(A − B, 0)]
"""
    )

# =====================================================
# Panel 3: Posterior mean timelines
# =====================================================
st.subheader("Posterior mean over updates")

steps_A = range(len(A["history"]))
# means_A = [a / (a + b) for a, b in A["history"]]
means_A, lo_A, hi_A = zip(
    *[posterior_mean_ci(a, b, level=0.95) for a, b in A["history"]]
)

steps_B = range(len(B["history"]))
# means_B = [a / (a + b) for a, b in B["history"]]
means_B, lo_B, hi_B = zip(
    *[posterior_mean_ci(a, b, level=0.95) for a, b in B["history"]]
)

dodge = 0.05

fig_t = go.Figure()

fig_t.add_trace(
    go.Scatter(
        x=np.array(steps_A) - dodge,
        y=means_A,
        mode="lines+markers",
        name="A mean ± 95% CI",
        error_y=dict(
            type="data",
            symmetric=False,
            array=np.array(hi_A) - np.array(means_A),
            arrayminus=np.array(means_A) - np.array(lo_A),
        ),
    )
)

fig_t.add_trace(
    go.Scatter(
        x=np.array(steps_B) + dodge,
        y=means_B,
        mode="lines+markers",
        name="B mean ± 95% CI",
        error_y=dict(
            type="data",
            symmetric=False,
            array=np.array(hi_B) - np.array(means_B),
            arrayminus=np.array(means_B) - np.array(lo_B),
        ),
    )
)

fig_t.update_layout(
    template="plotly_dark",
    xaxis=dict(
        title="Update step",
        dtick=1,  # tick step = 1 (integers)
        tickmode="linear",
        tick0=0,  # optional: start at 0
        tickformat="d",  # display as integer
    ),
    yaxis=dict(title="Posterior mean", range=[0, 1]),
    margin=dict(l=40, r=40, t=40, b=40),
)


st.plotly_chart(fig_t, use_container_width=True)
