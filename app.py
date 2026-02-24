import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import qualitative
from scipy.stats import beta

st.set_page_config(page_title="Bayesian A/B Updating", layout="wide")


# =====================================================
# Defaults
# =====================================================
DEFAULTS = {"a0_A": 20.0, "b0_A": 20.0, "a0_B": 1.0, "b0_B": 1.0}

# Initialize prior input session state keys on first run
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_sliders():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    # Also wipe the arms so they reinitialize from the default priors
    for arm in ("A", "B"):
        if arm in st.session_state:
            del st.session_state[arm]
    st.session_state.pop("_last_prior", None)


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
st.sidebar.caption(
    "Set your belief about each arm's conversion rate *before* seeing any data. A higher α relative to β means a stronger prior towards success."
)

_alpha_help = "Pseudo-count of prior successes. Prior mean = α / (α + β)."
_beta_help  = "Pseudo-count of prior failures. Prior mean = α / (α + β)."

st.sidebar.markdown("**A**")
_col1, _col2 = st.sidebar.columns(2)
with _col1:
    a0_A = st.number_input("α", min_value=0.1, step=1.0, key="a0_A", help=_alpha_help)
with _col2:
    b0_A = st.number_input("β", min_value=0.1, step=1.0, key="b0_A", help=_beta_help)

st.sidebar.markdown("**B**")
_col1, _col2 = st.sidebar.columns(2)
with _col1:
    a0_B = st.number_input("α", min_value=0.1, step=1.0, key="a0_B", help=_alpha_help)
with _col2:
    b0_B = st.number_input("β", min_value=0.1, step=1.0, key="b0_B", help=_beta_help)

st.sidebar.button("Default priors", on_click=reset_sliders)

# ── Detect prior changes and reset arms accordingly ──────────────────────────
# If the user changed a prior, treat it as a prior change and restart the arm.
current_prior = (a0_A, b0_A, a0_B, b0_B)
if st.session_state.get("_last_prior") != current_prior:
    reset_arm("A", a0_A, b0_A)
    reset_arm("B", a0_B, b0_B)
    st.session_state["_last_prior"] = current_prior
else:
    init_arm("A", a0_A, b0_A)
    init_arm("B", a0_B, b0_B)

st.sidebar.divider()

# Data entry
st.sidebar.header("Add data")
st.sidebar.caption(
    "Enter observed successes and failures for each arm, then add a batch to update the posteriors."
)
# A arm
st.sidebar.markdown("**A**")
col1, col2 = st.sidebar.columns(2)
with col1:
    sA = st.number_input("Successes", min_value=0, value=5, help="Number of successes (e.g. conversions) observed for arm A in this batch.")
with col2:
    fA = st.number_input("Failures", min_value=0, value=5, help="Number of failures (e.g. non-conversions) observed for arm A in this batch.")

# B arm
st.sidebar.markdown("**B**")
col1, col2 = st.sidebar.columns(2)
with col1:
    sB = st.number_input("Successes", min_value=0, value=6, help="Number of successes (e.g. conversions) observed for arm B in this batch.")
with col2:
    fB = st.number_input("B failures", min_value=0, value=4, help="Number of failures (e.g. non-conversions) observed for arm B in this batch.")

if st.sidebar.button("Add batch and update posteriors", width="stretch"):
    update_arm("A", sA, fA)
    update_arm("B", sB, fB)

st.sidebar.divider()
st.sidebar.header("Reset")
st.sidebar.caption(
    "Wipes all accumulated data and returns both arms to the current prior."
)
if st.sidebar.button("Clear data and reset posteriors", width="stretch"):
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
st.title("Bayesian A/B Updating")
st.caption(
    "Each arm's conversion rate is modelled as a [Beta distribution](https://en.wikipedia.org/wiki/Beta_distribution), "
    "a probability distribution over the interval [0, 1] that encodes beliefs about an unknown rate. "
    "Starting from a prior (α, β), each batch of observed successes and failures updates the posterior via Bayes' rule: "
    "α grows with successes, β with failures. "
    "This sequential updating is the core of the [Beta-Binomial model](https://en.wikipedia.org/wiki/Beta-binomial_distribution)."
)

prev_mean_A = (
    posterior_mean_ci(A["history"][-2][0], A["history"][-2][1])[0]
    if len(A["history"]) >= 2
    else None
)
prev_mean_B = (
    posterior_mean_ci(B["history"][-2][0], B["history"][-2][1])[0]
    if len(B["history"]) >= 2
    else None
)

delta_mean_A = mean_A - prev_mean_A if prev_mean_A is not None else None
delta_mean_B = mean_B - prev_mean_B if prev_mean_B is not None else None

_mean_help = "Posterior mean and 95% credible interval for this arm."


def ci_aside(lo, hi):
    st.markdown(
        f"<div style='opacity:0.45'>"
        f"<div style='font-size:0.875rem;font-weight:400;margin-bottom:0.4rem'>95% CI</div>"
        f"<div style='display:flex;flex-direction:column;justify-content:space-between;height:1.6rem'>"
        f"<div style='font-size:0.8rem'>{lo:.2f}</div>"
        f"<div style='font-size:0.8rem'>{hi:.2f}</div>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


col1, col2 = st.columns([3, 1])

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

    colors = qualitative.Plotly

    fig.add_vline(x=lo_A, line_dash="dot", opacity=0.4, line_color=colors[0])
    fig.add_vline(x=hi_A, line_dash="dot", opacity=0.4, line_color=colors[0])
    fig.add_vline(x=lo_B, line_dash="dot", opacity=0.4, line_color=colors[1])
    fig.add_vline(x=hi_B, line_dash="dot", opacity=0.4, line_color=colors[1])

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Success rate",
        yaxis_title="Density",
        margin=dict(l=40, r=40, t=40, b=40),
        height=400,
    )

    st.plotly_chart(fig, width="stretch")

    _, mc_A, ci_A, gap, mc_B, ci_B, _ = st.columns([0.5, 1, 0.5, 0.6, 1, 0.5, 1.0])
    with mc_A:
        st.metric(
            "Mean A",
            f"{mean_A:.2f}",
            delta=f"{delta_mean_A:+.2f}" if delta_mean_A is not None else None,
            delta_color="normal",
            help=_mean_help,
        )
    with ci_A:
        ci_aside(lo_A, hi_A)
    with mc_B:
        st.metric(
            "Mean B",
            f"{mean_B:.2f}",
            delta=f"{delta_mean_B:+.2f}" if delta_mean_B is not None else None,
            delta_color="normal",
            help=_mean_help,
        )
    with ci_B:
        ci_aside(lo_B, hi_B)

# =====================================================
# Panel 2: Decision metrics
# =====================================================
with col2:
    st.subheader("A vs. B")

    rng = np.random.default_rng(42)
    n_mc = 200_000

    samples_A = rng.beta(alpha_A, beta_A, size=n_mc)
    samples_B = rng.beta(alpha_B, beta_B, size=n_mc)

    p_B_gt_A = np.mean(samples_B > samples_A)
    loss_B = np.mean(np.maximum(samples_A - samples_B, 0.0))

    if len(A["history"]) >= 2:
        rng_prev = np.random.default_rng(42)
        prev_samples_A = rng_prev.beta(
            A["history"][-2][0], A["history"][-2][1], size=n_mc
        )
        prev_samples_B = rng_prev.beta(
            B["history"][-2][0], B["history"][-2][1], size=n_mc
        )
        delta_p = p_B_gt_A - np.mean(prev_samples_B > prev_samples_A)
        delta_loss = loss_B - np.mean(np.maximum(prev_samples_A - prev_samples_B, 0.0))
    else:
        delta_p = delta_loss = None

    diff_now = mean_B - mean_A
    diff_prev = (
        (prev_mean_B - prev_mean_A)
        if (prev_mean_A is not None and prev_mean_B is not None)
        else None
    )
    delta_diff = diff_now - diff_prev if diff_prev is not None else None

    st.metric(
        "Mean difference (B − A)",
        f"{diff_now:.2f}",
        delta=f"{delta_diff:+.2f}" if delta_diff is not None else None,
        delta_color="normal",
        help="Difference of posterior means B − A.",
    )
    st.metric(
        "P(B > A)",
        f"{p_B_gt_A:.3f}",
        delta=f"{delta_p:+.3f}" if delta_p is not None else None,
        delta_color="normal",
        help="Probability that B's true rate exceeds A's, estimated via Monte Carlo sampling.",
    )
    st.metric(
        "Expected loss if choosing B",
        f"{loss_B:.4f}",
        delta=f"{delta_loss:+.4f}" if delta_loss is not None else None,
        delta_color="inverse",
        help="E[max(A − B, 0)]: average rate you'd lose by picking B if A were actually better.",
    )


# =====================================================
# Panel 3: Posterior mean timelines
# =====================================================

steps_A = range(len(A["history"]))
means_A, lo_A_hist, hi_A_hist = zip(
    *[posterior_mean_ci(a, b, level=0.95) for a, b in A["history"]]
)

steps_B = range(len(B["history"]))
means_B, lo_B_hist, hi_B_hist = zip(
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
            array=np.array(hi_A_hist) - np.array(means_A),
            arrayminus=np.array(means_A) - np.array(lo_A_hist),
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
            array=np.array(hi_B_hist) - np.array(means_B),
            arrayminus=np.array(means_B) - np.array(lo_B_hist),
        ),
    )
)

_max_step = len(A["history"]) - 1
_tick_vals = list(range(_max_step + 1))
_tick_text = ["Prior"] + [str(i) for i in range(1, _max_step + 1)]

fig_t.update_layout(
    template="plotly_dark",
    xaxis=dict(
        title="Batch",
        tickmode="array",
        tickvals=_tick_vals,
        ticktext=_tick_text,
        range=[-0.5, 0.5] if _max_step == 0 else None,
    ),
    yaxis=dict(title="Posterior mean", range=[0, 1]),
    margin=dict(l=40, r=40, t=40, b=40),
    height=400,
)

col_seq, col_seq_metrics = st.columns([3, 1])

with col_seq:
    st.subheader("Posterior mean evolution")
    st.plotly_chart(fig_t, width="stretch")

with col_seq_metrics:
    st.subheader("Sample metrics")

    n_batches = len(A["history"]) - 1

    cum_nA = int(
        round((A["alpha"] - A["history"][0][0]) + (A["beta"] - A["history"][0][1]))
    )
    cum_nB = int(
        round((B["alpha"] - B["history"][0][0]) + (B["beta"] - B["history"][0][1]))
    )

    if n_batches >= 1:
        last_nA = int(
            round(
                (A["history"][-1][0] - A["history"][-2][0])
                + (A["history"][-1][1] - A["history"][-2][1])
            )
        )
        last_nB = int(
            round(
                (B["history"][-1][0] - B["history"][-2][0])
                + (B["history"][-1][1] - B["history"][-2][1])
            )
        )
    else:
        last_nA = last_nB = None

    st.metric(
        "Batches",
        n_batches,
        help="Number of data batches added so far. Each batch is one step on the x-axis of the chart.",
    )
    st.metric(
        "Cumulative N (A)",
        cum_nA,
        delta=f"+{last_nA}" if last_nA is not None else None,
        delta_color="off",
        help="Total observations for arm A across all batches.",
    )
    st.metric(
        "Cumulative N (B)",
        cum_nB,
        delta=f"+{last_nB}" if last_nB is not None else None,
        delta_color="off",
        help="Total observations for arm B across all batches.",
    )
