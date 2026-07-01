"""
RiCode: live IDX risk tools (Streamlit).

Four pages over live Yahoo Finance data:
  Value at Risk       - historical / EWMA / parametric / Cornish-Fisher VaR,
                        CVaR, and an out-of-sample breach backtest
  Monte Carlo         - GBM (antithetic) or bootstrap price-path simulation
  Single-Factor Model - market-model beta, alpha, R2, systematic vs specific risk
  Reader's Guide      - explains each metric for readers new to finance

Run locally:   streamlit run app.py
Deploy:        push to GitHub, point Streamlit Community Cloud at app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ricode_core as vc

# --------------------------------------------------------------------------- #
# Palette + theming
# --------------------------------------------------------------------------- #
BG   = "#f3f3ea"; PANEL = "#ffffff"; LINE = "#e0e0d4"
INK  = "#23231f"; MUTED = "#8d8d82"; FAINT = "#b9b9ac"
GREEN= "#1c7d4d"; GOLD = "#a97b0a"; RED = "#ad2727"; BLUE = "#2a4fd0"; PURPLE = "#5a2fd6"
DARKRED = "#7a1a1a"
MONO = "JetBrains Mono, ui-monospace, monospace"

st.set_page_config(page_title="RiCode", page_icon="◱", layout="centered",
                   initial_sidebar_state="expanded")

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Space+Mono:wght@400;700&display=swap');
html, body, [class*="css"], .stApp, .stMarkdown, input, textarea, button, select {{
  font-family:{MONO} !important;
}}
.stApp {{
  background:{BG};
  background-image:
    linear-gradient(to right, rgba(0,0,0,.022) 1px, transparent 1px),
    linear-gradient(to bottom, rgba(0,0,0,.022) 1px, transparent 1px);
  background-size:34px 34px;
}}
#MainMenu, header[data-testid="stHeader"], [data-testid="stToolbar"], footer {{visibility:hidden; height:0;}}
.block-container {{padding-top:1.4rem; max-width:660px;}}
h1,h2,h3 {{font-family:'Space Mono',{MONO} !important; letter-spacing:-.01em;}}
[data-testid="stSidebar"] {{background:{PANEL}; border-right:1px solid {LINE};}}
[data-testid="stSidebar"] .stRadio label, [data-testid="stSidebar"] label {{font-size:13px;}}
.stButton>button {{
  width:100%; background:{GREEN}; color:#fff; border:none; border-radius:2px;
  font-weight:700; letter-spacing:.2em; text-transform:uppercase; padding:.7rem;
}}
.stButton>button:hover {{background:#18693f; color:#fff;}}
input, select, textarea, [data-baseweb="select"]>div {{border-radius:2px !important;}}

/* custom blocks */
.vhead {{display:flex; align-items:center; gap:14px; font-size:15px; letter-spacing:.28em;
  font-weight:700; border-bottom:1px solid {LINE}; padding-bottom:16px; margin-bottom:22px;}}
.vhead .v {{color:{GREEN};letter-spacing:.04em;}} .vhead .s {{color:{FAINT}; font-weight:400;}}
.vhead .t {{color:{MUTED}; font-weight:400; letter-spacing:.22em;}}
.eyebrow {{font-size:12px; letter-spacing:.26em; color:{GREEN}; font-weight:500;
  text-transform:uppercase; margin:26px 0 6px;}}
.lede {{color:{MUTED}; font-size:14px; line-height:1.55; margin:0 0 8px;}}
.sgrid {{display:grid; grid-template-columns:1fr 1fr; gap:12px; margin:8px 0 6px;}}
.scard {{background:{PANEL}; border:1px solid {LINE}; border-radius:2px; padding:18px;}}
.scard.wide {{grid-column:1 / -1;}}
.scard .k {{font-size:11px; letter-spacing:.2em; color:{MUTED}; text-transform:uppercase; margin:0 0 9px;}}
.scard .n {{font-family:'Space Mono',{MONO}; font-weight:700; font-size:27px; line-height:1; margin:0;}}
.scard .sub {{font-size:12px; color:{MUTED}; margin:8px 0 0;}}
.note {{background:{PANEL}; border:1px solid {LINE}; border-left:4px solid {LINE};
  border-radius:2px; padding:18px 20px; margin:14px 0;}}
.note h4 {{font-size:14px; margin:0 0 8px;}} .note p {{font-size:13.5px; line-height:1.6; margin:0; color:{INK};}}
.gterm {{background:{PANEL}; border:1px solid {LINE}; border-radius:2px;
  padding:15px 18px; margin:0 0 10px;}}
.gterm h4 {{font-size:12.5px; letter-spacing:.14em; text-transform:uppercase; margin:0 0 7px;}}
.gterm p {{font-size:13.5px; line-height:1.62; margin:0; color:{INK};}}
.gterm p i {{color:{MUTED}; font-style:normal;}}
.foot {{border-top:1px solid {LINE}; margin-top:34px; padding-top:18px; display:flex;
  justify-content:space-between; font-size:12px; color:{FAINT};}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def header(tool: str):
    st.markdown(f'<div class="vhead"><span class="v">RiCode</span>'
                f'<span class="s">/</span><span class="t">{tool}</span></div>',
                unsafe_allow_html=True)


def stat_grid(items):
    """items: list of (label, value, sub, color, wide?)"""
    cells = ""
    for label, value, sub, color, *rest in items:
        wide = " wide" if (rest and rest[0]) else ""
        cells += (f'<div class="scard{wide}"><p class="k">{label}</p>'
                  f'<p class="n" style="color:{color}">{value}</p>'
                  f'<p class="sub">{sub}</p></div>')
    st.markdown(f'<div class="sgrid">{cells}</div>', unsafe_allow_html=True)


def note(color, title, body):
    st.markdown(f'<div class="note" style="border-left-color:{color}">'
                f'<h4 style="color:{color}">{title}</h4><p>{body}</p></div>',
                unsafe_allow_html=True)


def eyebrow(text):
    st.markdown(f'<p class="eyebrow">{text}</p>', unsafe_allow_html=True)


def term(color, title, body):
    st.markdown(f'<div class="gterm"><h4 style="color:{color}">{title}</h4>'
                f'<p>{body}</p></div>', unsafe_allow_html=True)


def theme_fig(fig, height=360, ylab="", xlab=""):
    fig.update_layout(
        height=height, paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(family=MONO, size=12, color=INK),
        margin=dict(l=10, r=10, t=18, b=10), showlegend=False,
        xaxis_title=xlab, yaxis_title=ylab,
    )
    fig.update_xaxes(gridcolor="#eeeee4", zeroline=False, linecolor=LINE,
                     title_font=dict(size=11, color=FAINT))
    fig.update_yaxes(gridcolor="#eeeee4", zeroline=False, linecolor=LINE,
                     title_font=dict(size=11, color=FAINT))
    return fig


def show(fig):
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def footer():
    st.markdown('<div class="foot"><span>Educational only. Not financial advice.</span>'
                '<span>github.com/richardohugo</span></div>', unsafe_allow_html=True)


@st.cache_data(ttl=1800, show_spinner=False)
def load_prices(ticker, period):
    return vc.get_prices(ticker, period)


PERIODS = {"1 year": "1y", "2 years": "2y", "3 years": "3y", "5 years": "5y"}
CONFS = [0.90, 0.95, 0.975, 0.99]


def conf_label(conf: float) -> str:
    c = conf * 100
    return f"{c:.0f}" if c % 1 == 0 else f"{c:g}"


# --------------------------------------------------------------------------- #
# Tool 1: Value at Risk
# --------------------------------------------------------------------------- #
def page_var():
    header("VALUE AT RISK")
    st.title("Value at Risk")
    st.markdown('<p class="lede">Pull a live price history, then estimate the loss level '
                'that only the worst days cross. Four methods, plus the average loss on '
                'those worst days, and a backtest of how often the line was actually '
                'crossed.</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("**Inputs**")
        ticker = st.text_input("Ticker", "BBCA.JK").strip()
        period = st.selectbox("History", list(PERIODS), index=1)
        conf = st.select_slider("Confidence", CONFS, value=0.95)
        horizon = st.number_input("Horizon (days)", 1, 60, 1)
        ccy = st.text_input("Currency label", "Rp")
        position = st.number_input(f"Position value ({ccy}, 0 = skip)", 0.0, value=0.0,
                                   step=1e6, format="%.0f")
        go_btn = st.button("Run")

    if not (go_btn or "var_done" in st.session_state):
        st.info("Set a ticker in the sidebar and press **Run**.")
        return
    st.session_state["var_done"] = True

    try:
        px = load_prices(ticker, PERIODS[period])
    except Exception as e:
        st.error(f"{e}")
        return
    rets = vc.simple_returns(px)
    rep = vc.var_report(rets, conf=conf, horizon=int(horizon), position=position)
    cpct = conf_label(conf)

    st.markdown(f'<div class="lede" style="margin-top:2px">● <b style="color:{INK}">{ticker.upper()}</b> '
                f'&nbsp; {rep["n"]:,} days / {period} / {int(horizon)}d horizon</div>',
                unsafe_allow_html=True)
    if rep["n"] < 120:
        st.warning("Short sample: every number below is noisy. Prefer 1 year of history or more.")

    def pc(x): return f"{x*100:.2f}%"
    hsub = "empirical quantile" if horizon == 1 else "compounded from real daily returns"
    stat_grid([
        (f"VaR {cpct}% · historical", pc(rep["hist"]), hsub, RED),
        (f"VaR {cpct}% · EWMA", pc(rep["ewma"]), "weights recent volatility most", PURPLE),
        (f"VaR {cpct}% · parametric", pc(rep["param"]), "normal assumption", GOLD),
        (f"VaR {cpct}% · Cornish-Fisher", pc(rep["cf"]), "adjusted for skew + fat tails", BLUE),
        (f"CVaR {cpct}% · expected shortfall", pc(rep["cvar"]),
         "average loss beyond the VaR line", DARKRED, True),
    ])

    # all methods on one scale; position is the channel the eye reads best
    methods = [
        ("Historical", rep["hist"], RED),
        ("EWMA", rep["ewma"], PURPLE),
        ("Parametric", rep["param"], GOLD),
        ("Cornish-Fisher", rep["cf"], BLUE),
        ("CVaR", rep["cvar"], DARKRED),
    ]
    methods.sort(key=lambda m: m[1])
    dot = go.Figure()
    for name, val, color in methods:
        dot.add_trace(go.Scatter(x=[0, val * 100], y=[name, name], mode="lines",
                                 line=dict(color="#eeeee4", width=2), hoverinfo="skip"))
        dot.add_trace(go.Scatter(x=[val * 100], y=[name], mode="markers+text",
                                 marker=dict(color=color, size=9),
                                 text=[f" {val*100:.2f}%"], textposition="middle right",
                                 textfont=dict(size=11, color=color),
                                 hovertemplate=f"{name} %{{x:.2f}}%<extra></extra>"))
    dot.update_yaxes(showgrid=False)
    dot.update_xaxes(ticksuffix="%", range=[0, methods[-1][1] * 100 * 1.25])
    eyebrow(f"All methods on one scale · {int(horizon)}d")
    fig = theme_fig(dot, 210)
    fig.update_layout(margin=dict(l=10, r=64, t=8, b=10))
    show(fig)

    # distribution with the 1-day VaR / CVaR thresholds marked
    r = rets.to_numpy()
    counts, edges = np.histogram(r, bins=46)
    centers = (edges[:-1] + edges[1:]) / 2
    thresh = -rep["hist1"]
    colors = [RED if c <= thresh else "rgba(28,125,77,0.45)" for c in centers]
    fig = go.Figure(go.Bar(x=centers, y=counts, marker_color=colors,
                           marker_line_width=0, hovertemplate="%{x:.2%}<extra></extra>"))
    fig.add_vline(x=thresh, line=dict(color=RED, dash="dash", width=1.5),
                  annotation_text=f"VaR {cpct}%", annotation_position="top left",
                  annotation_font_color=RED)
    fig.add_vline(x=-rep["cvar1"], line=dict(color=DARKRED, dash="dot", width=1.5),
                  annotation_text="CVaR", annotation_position="bottom left",
                  annotation_font_color=DARKRED)
    eyebrow("Daily return distribution · 1-day thresholds")
    show(theme_fig(fig, 330, xlab="daily return"))

    # rolling VaR with real out-of-sample breaches marked on the same surface
    window = min(250, max(60, len(rets) // 3))
    rv = vc.rolling_var(rets, window=window, conf=conf)
    bt = vc.var_backtest(rets, window=window, conf=conf)
    if len(rv):
        line = go.Figure(go.Scatter(x=rv.index, y=rv.values * 100, mode="lines",
                                    line=dict(color=RED, width=1.6),
                                    hovertemplate="%{x|%d %b %y} · VaR %{y:.2f}%<extra></extra>"))
        if bt and bt["breaches"]:
            line.add_trace(go.Scatter(
                x=bt["breach_dates"], y=-bt["breach_returns"].to_numpy() * 100,
                mode="markers", marker=dict(color=DARKRED, size=6, symbol="x"),
                hovertemplate="%{x|%d %b %y} · loss %{y:.2f}%<extra></extra>"))
        eyebrow(f"Rolling 1-day VaR ({window}d window) · × = day the loss exceeded it")
        show(theme_fig(line, 260, ylab="loss %"))
    if bt:
        verdict = ("breach count consistent with the target"
                   if bt["kupiec_p"] >= 0.05 else
                   "breach count off target, treat the VaR level with suspicion")
        st.markdown(f'<p class="lede">Backtest: <b>{bt["breaches"]}</b> breaches vs '
                    f'<b>{bt["expected"]:.1f}</b> expected over {bt["n"]:,} days '
                    f'(Kupiec p = {bt["kupiec_p"]:.2f}): {verdict}.</p>',
                    unsafe_allow_html=True)

    # cash + extras
    extra = f"Ann. vol <b>{rep['vol_ann']*100:.1f}%</b> (EWMA <b>{rep['vol_ewma_ann']*100:.1f}%</b>) " \
            f"&nbsp;·&nbsp; worst day <b>{rep['worst_day']*100:.1f}%</b> " \
            f"&nbsp;·&nbsp; skew <b>{rep['skew']:+.2f}</b> &nbsp;·&nbsp; excess kurt <b>{rep['exkurt']:+.2f}</b>"
    if position:
        extra += f"<br>Cash VaR (hist) <b>{ccy}{rep['hist_cash']:,.0f}</b> &nbsp;·&nbsp; " \
                 f"CVaR <b>{ccy}{rep['cvar_cash']:,.0f}</b> on a {ccy}{position:,.0f} position"
    st.markdown(f'<p class="lede" style="margin-top:6px">{extra}</p>', unsafe_allow_html=True)

    note(GOLD, "What VaR leaves out",
         "VaR is a threshold, not a worst case. It says nothing about how bad losses get once "
         "you cross it; that is what CVaR (expected shortfall) is for. All of it is backward-looking: "
         "it assumes tomorrow resembles the window you measured. Multi-day historical figures here "
         "compound real daily returns rather than stretching the 1-day number. Even so, calm "
         "windows produce comfortable-looking VaR right up until the storm, which is what the "
         "breach backtest above is watching for.")
    footer()


# --------------------------------------------------------------------------- #
# Tool 2: Monte Carlo
# --------------------------------------------------------------------------- #
def page_mc():
    header("MONTE CARLO")
    st.title("Monte Carlo")
    st.markdown('<p class="lede">Pull a live price history, estimate its drift and volatility, '
                'then simulate thousands of possible futures and read the probability fan. '
                'Two engines: a smooth model of returns, or resampling the days the stock '
                'has actually had.</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("**Inputs**")
        ticker = st.text_input("Ticker", "BBCA.JK").strip()
        period = st.selectbox("History", list(PERIODS), index=1)
        days = st.number_input("Horizon (days)", 1, 504, 30)
        n_paths = st.selectbox("Simulations", [1000, 5000, 10000, 20000], index=1,
                               format_func=lambda n: f"{n:,}")
        engine = st.radio("Engine", ["Model (GBM)", "Bootstrap (real days)"])
        conf = st.select_slider("Confidence", CONFS, value=0.95)
        seed_in = st.number_input("Seed (0 = new each run)", 0, 999_999_999, 0)
        ccy = st.text_input("Currency label", "Rp")
        position = st.number_input(f"Position value ({ccy}, 0 = skip)", 0.0, value=0.0,
                                   step=1e6, format="%.0f")
        with st.expander("Override drift / vol"):
            manual = st.checkbox("Set manually (GBM only)")
            mu_man = st.number_input("Drift %/yr", -100.0, 200.0, 8.0, 0.5)
            sig_man = st.number_input("Vol %/yr", 0.0, 300.0, 30.0, 0.5)
        go_btn = st.button("Run")

    if not (go_btn or "mc_done" in st.session_state):
        st.info("Set a ticker in the sidebar and press **Run**.")
        return
    st.session_state["mc_done"] = True

    try:
        px = load_prices(ticker, PERIODS[period])
    except Exception as e:
        st.error(f"{e}")
        return
    rets = vc.simple_returns(px)
    spot = float(px.iloc[-1])
    mu_est, sig_est = vc.annualized_mu_sigma(px)

    # freeze the seed per Run press so widget tweaks don't reshuffle the fan
    if seed_in:
        seed = int(seed_in)
    else:
        if go_btn or "mc_seed" not in st.session_state:
            st.session_state["mc_seed"] = int(np.random.default_rng().integers(1, 1e9))
        seed = st.session_state["mc_seed"]

    days_i, conf_i = int(days), float(conf)
    try:
        if engine.startswith("Model"):
            mu_use = mu_man / 100 if manual else mu_est
            sig_use = sig_man / 100 if manual else sig_est
            paths = vc.mc_paths_gbm(spot, mu_use, sig_use, days_i, int(n_paths), seed=seed)
            eng_note = f"μ {mu_use*100:+.1f}%/yr · σ {sig_use*100:.1f}%/yr" + \
                       (" (manual)" if manual else " (measured)")
        else:
            paths = vc.mc_paths_bootstrap(spot, rets, days_i, int(n_paths), seed=seed)
            eng_note = f"resampling {len(rets):,} real trading days"
    except ValueError as e:
        st.error(f"{e}")
        return
    summ = vc.mc_summary(paths, conf=conf_i)
    cpct = conf_label(conf_i)

    st.markdown(f'<div class="lede" style="margin-top:2px">● <b style="color:{INK}">{ticker.upper()}</b>'
                f' &nbsp; {int(n_paths):,} paths / {days_i}d / seed {seed} &nbsp;·&nbsp; {eng_note}</div>',
                unsafe_allow_html=True)

    # ---- fan chart: quiet bands, one loud median, labels at the data ---- #
    b = summ["bands"]
    xs = np.arange(days_i + 1)
    fig = go.Figure()

    # sample paths, colored by where they end (kept very faint: context, not signal)
    stride = max(1, len(paths) // 100)
    sample = paths[::stride][:100]
    up = sample[sample[:, -1] >= spot]
    dn = sample[sample[:, -1] < spot]
    for mat, color in ((up, "rgba(28,125,77,0.09)"), (dn, "rgba(173,39,39,0.09)")):
        if len(mat):
            xcat, ycat = [], []
            for row in mat:
                xcat.extend(xs.tolist() + [None])
                ycat.extend(row.tolist() + [None])
            fig.add_trace(go.Scatter(x=xcat, y=ycat, mode="lines",
                                     line=dict(color=color, width=1),
                                     hoverinfo="skip"))
    # percentile bands
    fig.add_trace(go.Scatter(x=xs, y=b[0], mode="lines",
                             line=dict(width=0), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=b[4], mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(120,120,180,0.07)",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=b[1], mode="lines",
                             line=dict(width=0), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=b[3], mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(120,120,180,0.11)",
                             hoverinfo="skip"))
    for arr, color, dash, wd in ((b[4], GREEN, "dash", 1.4), (b[3], FAINT, "dot", 1),
                                 (b[1], FAINT, "dot", 1), (b[0], RED, "dash", 1.4),
                                 (b[2], BLUE, None, 2.4)):
        fig.add_trace(go.Scatter(x=xs, y=arr, mode="lines",
                                 line=dict(color=color, dash=dash, width=wd),
                                 hovertemplate="day %{x} · %{y:,.0f}<extra></extra>"))
    # entry marker + direct labels at the right edge (no legend to shuttle to)
    fig.add_trace(go.Scatter(x=[0], y=[spot], mode="markers",
                             marker=dict(color=GOLD, size=8), hoverinfo="skip"))
    for qi, lab, color, bold in ((4, "P95", GREEN, True), (3, "P75", "#9a9a90", False),
                                 (2, "P50", BLUE, True), (1, "P25", "#9a9a90", False),
                                 (0, "P5", RED, True)):
        fig.add_annotation(x=days_i, y=b[qi, -1], xanchor="left", xshift=6,
                           text=f"{lab} {b[qi, -1]:,.0f}", showarrow=False,
                           font=dict(size=11, color=color,
                                     family="JetBrains Mono, monospace"),
                           bgcolor="rgba(255,255,255,0.6)")
    fig = theme_fig(fig, 430, xlab="trading days")
    fig.update_layout(margin=dict(l=10, r=92, t=18, b=10))
    fig.update_yaxes(tickformat="~s")
    eyebrow("Probability fan")
    show(fig)

    med_pct = (summ["median"] / spot - 1) * 100
    money = lambda v: f"{ccy}{v:,.0f}"
    stat_grid([
        ("Median", money(summ["median"]), f"{med_pct:+.1f}% vs entry {money(spot)}",
         GREEN if med_pct >= 0 else RED),
        ("Prob. above entry", f"{summ['prob_up']*100:.1f}%", "share of paths ending up", GOLD),
        ("P5", money(summ["p5"]), "only 1 in 20 ends lower", RED),
        ("P95", money(summ["p95"]), "only 1 in 20 ends higher", BLUE),
        (f"VaR {cpct}% · {days_i}d · simulated", f"{summ['var']*100:.2f}%",
         f"CVaR {summ['cvar']*100:.2f}% · average loss in the worst tail", DARKRED, True),
    ])
    if position:
        st.markdown(f'<p class="lede">Cash VaR <b>{ccy}{summ["var"]*position:,.0f}</b> &nbsp;·&nbsp; '
                    f'CVaR <b>{ccy}{summ["cvar"]*position:,.0f}</b> on a {ccy}{position:,.0f} position</p>',
                    unsafe_allow_html=True)

    # ---- terminal distribution ---- #
    term = summ["terminal"]
    counts, edges = np.histogram(term, bins=46)
    centers = (edges[:-1] + edges[1:]) / 2
    colors = ["rgba(173,39,39,0.55)" if c < spot else "rgba(28,125,77,0.5)" for c in centers]
    hist = go.Figure(go.Bar(x=centers, y=counts, marker_color=colors, marker_line_width=0,
                            hovertemplate="%{x:,.0f}<extra></extra>"))
    hist.add_vline(x=spot, line=dict(color=GOLD, dash="dash", width=1.5),
                   annotation_text="entry", annotation_position="top left",
                   annotation_font_color=GOLD)
    hist.add_vline(x=spot * (1 - summ["var"]), line=dict(color=RED, dash="dot", width=1.5),
                   annotation_text=f"VaR {cpct}%", annotation_position="bottom left",
                   annotation_font_color=RED)
    hist = theme_fig(hist, 260, xlab=f"price after {days_i} days")
    hist.update_xaxes(tickformat="~s")
    eyebrow(f"Where the {int(n_paths):,} paths end")
    show(hist)

    note(PURPLE, "Results shift between runs",
         "With seed 0 every press of Run draws fresh randomness, so the fan wobbles slightly; "
         "more simulations make it steadier. Set a nonzero seed to freeze a run so a colleague "
         "can reproduce your exact numbers.")
    note(RED, "Read it as a range, not a forecast",
         "The smooth engine assumes volatility never changes and extreme days are rarer than "
         "they really are. The bootstrap engine replays days the stock has actually had, fat "
         "tails included, but shuffles them independently, so it misses the way storms cluster. "
         "Neither knows anything the price history doesn't.")
    footer()


# --------------------------------------------------------------------------- #
# Tool 3: Single-Factor Model
# --------------------------------------------------------------------------- #
def page_factor():
    header("SINGLE-FACTOR MODEL")
    st.title("Single-Factor Model")
    st.markdown('<p class="lede">Regress a stock\'s excess returns on the market. What comes back '
                'is beta, alpha, and the split between risk you share with the market (systematic) '
                'and risk that is yours alone (specific).</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("**Inputs**")
        stock = st.text_input("Stock", "BBCA.JK").strip()
        factor = st.text_input("Factor / benchmark", "^JKSE").strip()
        period = st.selectbox("History", list(PERIODS), index=1)
        rf = st.number_input("Risk-free %/yr", 0.0, 20.0, 0.0, 0.25) / 100
        go_btn = st.button("Run")

    if not (go_btn or "fm_done" in st.session_state):
        st.info("Set a stock and benchmark in the sidebar and press **Run**. "
                "Default benchmark ^JKSE is the Jakarta Composite.")
        return
    st.session_state["fm_done"] = True

    try:
        ps = load_prices(stock, PERIODS[period])
        pf = load_prices(factor, PERIODS[period])
        rs, rm = vc.simple_returns(ps), vc.simple_returns(pf)
        fm = vc.factor_model(rs, rm, rf_annual=rf)
    except Exception as e:
        st.error(f"{e}")
        return

    st.markdown(f'<div class="lede" style="margin-top:2px">● <b style="color:{INK}">{stock.upper()}</b> '
                f'vs <b style="color:{INK}">{factor.upper()}</b> &nbsp; {fm["n"]:,} days / {period}</div>',
                unsafe_allow_html=True)

    a_sig = "significant" if fm["p_alpha"] < 0.05 else "not significant"
    stat_grid([
        ("Beta", f"{fm['beta']:.2f}", f"t = {fm['t_beta']:.1f}", BLUE),
        ("Alpha /yr", f"{fm['alpha_ann']*100:+.1f}%", f"{a_sig} (p={fm['p_alpha']:.2f})",
         GREEN if fm['alpha_ann'] >= 0 else RED),
        ("R²", f"{fm['r2']*100:.0f}%", "variance from the market", GOLD),
        ("Idiosyncratic vol", f"{fm['resid_vol_ann']*100:.1f}%", "stock-specific, annualized", PURPLE),
    ])

    # scatter + fit, with the slope labeled at the data instead of a legend
    x, y = fm["x"] * 100, fm["y"] * 100
    xs = np.linspace(x.min(), x.max(), 50)
    ys = (fm["alpha_daily"] + fm["beta"] * xs / 100) * 100
    sc = go.Figure()
    sc.add_trace(go.Scatter(x=x, y=y, mode="markers",
                            marker=dict(color="rgba(42,79,208,0.35)", size=5),
                            hovertemplate="mkt %{x:.2f}%<br>stock %{y:.2f}%<extra></extra>"))
    sc.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=BLUE, width=2.2),
                            hoverinfo="skip"))
    sc.add_annotation(x=xs[-1], y=ys[-1], text=f"beta {fm['beta']:.2f}",
                      showarrow=False, xanchor="right", yshift=12,
                      font=dict(size=11, color=BLUE))
    eyebrow("Excess returns · stock vs market")
    show(theme_fig(sc, 340, ylab=f"{stock.upper()} %", xlab=f"{factor.upper()} %"))

    # variance decomposition
    dec = go.Figure()
    dec.add_trace(go.Bar(y=["risk"], x=[fm["sys_share"]*100], orientation="h",
                         marker_color=BLUE, name="systematic",
                         hovertemplate="systematic %{x:.0f}%<extra></extra>"))
    dec.add_trace(go.Bar(y=["risk"], x=[fm["spec_share"]*100], orientation="h",
                         marker_color=GOLD, name="specific",
                         hovertemplate="specific %{x:.0f}%<extra></extra>"))
    dec.update_layout(barmode="stack")
    dec.update_yaxes(showticklabels=False)
    dec.update_xaxes(range=[0, 100], ticksuffix="%")
    eyebrow("Variance decomposition")
    show(theme_fig(dec, 130))
    st.markdown(f'<p class="lede" style="margin-top:-6px">'
                f'<span style="color:{BLUE}">■</span> systematic {fm["sys_share"]*100:.0f}% &nbsp;&nbsp;'
                f'<span style="color:{GOLD}">■</span> specific {fm["spec_share"]*100:.0f}% &nbsp;·&nbsp; '
                f'total vol {fm["total_vol_ann"]*100:.1f}%/yr</p>', unsafe_allow_html=True)

    # rolling beta
    rb = vc.rolling_beta(rs, rm, window=63)
    if len(rb):
        bl = go.Figure(go.Scatter(x=rb.index, y=rb.values, mode="lines",
                                  line=dict(color=BLUE, width=1.6)))
        bl.add_hline(y=fm["beta"], line=dict(color=FAINT, dash="dash", width=1))
        eyebrow("Rolling 63-day beta")
        show(theme_fig(bl, 230, ylab="beta"))

    note(PURPLE, "Reading the split",
         "R² is the share of the stock's variance explained by the market. The rest is "
         "idiosyncratic, and by construction the two add to the total. Beta is estimated tightly; "
         "alpha usually is not. A large t-stat on beta and a near-zero one on alpha is the norm, "
         "and a reminder that most measured alpha is noise, not skill. Watch the rolling beta "
         "wander: a single number hides real instability.")
    footer()


# --------------------------------------------------------------------------- #
# Tool 4: Reader's Guide
# --------------------------------------------------------------------------- #
def page_guide():
    header("READER'S GUIDE")
    st.title("Reader's Guide")
    st.markdown('<p class="lede">What each number on the other pages means and how to '
                'read it. None of these figures predict the future. They measure how '
                'the stock has behaved, and what that implies if the pattern '
                'continues.</p>', unsafe_allow_html=True)

    eyebrow("Value at Risk")
    term(RED, "VaR 95%",
         "The daily loss you should stay under on 19 days out of 20. If the card says "
         "2.85%, then on a Rp100 million position, about one trading day in 20 will "
         "lose <b>more</b> than Rp2.85 million. Every other day should lose less. It "
         "marks where the bad zone starts, not how deep it goes.")
    term(DARKRED, "CVaR · expected shortfall",
         "The average loss on the days that cross the VaR line. It is always larger "
         "than VaR. Think of VaR as the height of a flood barrier and CVaR as the "
         "average depth of the water on the days it overflows. A wide gap between the "
         "two means the rare days tend to be severe.")
    term(GREEN, "Why four versions of VaR",
         "Four estimates of the same line. <b>Historical</b> replays the actual past "
         "days. <b>EWMA</b> does the same but gives recent weeks more weight, so it "
         "reacts faster when markets turn rough. <b>Parametric</b> assumes returns "
         "follow a bell curve, which is simple but often too optimistic. "
         "<b>Cornish-Fisher</b> starts from the bell curve and widens it, since real "
         "markets produce more extreme days than a bell curve allows. "
         "<i>When EWMA sits far above Historical, recent conditions are rougher than "
         "the long-run average, and the bigger number is the safer guide.</i>")
    term(GOLD, "Confidence & Horizon",
         "Confidence sets how rare the bad day is: 95% means the line is crossed about "
         "1 trading day in 20, and 99% means about 1 in 100, so the number grows. "
         "Horizon is how long you hold the position: 1 day measures overnight risk, 10 "
         "days measures a two-week hold. A longer horizon gives a bigger number.")
    term(RED, "Rolling VaR chart & the × marks",
         "The red line is the VaR estimate as it moved through time. Each × marks a "
         "day when the actual loss was worse than the line had indicated. A few "
         "scattered × are normal. Clusters of them mean the estimate was lagging "
         "behind reality.")
    term(PURPLE, "The backtest sentence",
         "A check on the model itself. At 95% confidence, about 1 day in 20 should "
         "breach the line, so the sentence compares the expected number of breaches "
         "with the number that actually happened. The p-value is the chance that a "
         "correct model would show a gap this large by luck. Below 0.05 the model is "
         "counted as <b>off target</b> and its headline VaR is probably too low.")
    term(MUTED, "The small print: vol, worst day, skew, kurtosis",
         "<b>Annualized volatility</b> measures how bumpy the ride is: around 10% is "
         "calm, around 30% is rough. <b>Worst day</b> is the largest single-day drop "
         "in the window. Negative <b>skew</b> means the surprises lean to the "
         "downside. <b>Excess kurtosis</b> above zero means extreme days happen more "
         "often than a bell curve would suggest.")

    eyebrow("Monte Carlo")
    term(BLUE, "The fan itself",
         "Thousands of simulated futures for the price, each built from the stock's "
         "measured drift and volatility. The blue center line is the median path and "
         "the shaded band holds the middle 90% of outcomes. The fan widens with time "
         "because uncertainty compounds. Read it as a range of outcomes, not a "
         "prediction.")
    term(GREEN, "Median & Prob. above entry",
         "The median is the middle ending: half the simulated futures finish above it "
         "and half below. Prob. above entry is the share of futures that end above "
         "today's price. A value of 36% means the position ends down in roughly 2 runs "
         "out of 3, assuming the measured pattern holds.")
    term(RED, "P5 and P95",
         "The pessimistic and optimistic brackets. Only 1 future in 20 ends below P5, "
         "and only 1 in 20 ends above P95. Size the position so the P5 ending is "
         "survivable.")
    term(DARKRED, "Simulated VaR / CVaR",
         "The same reading as on the Value at Risk page, the loss line and the average "
         "loss beyond it, but measured over the whole horizon from the simulated "
         "endings instead of directly from history.")
    term(GOLD, "Engine & Seed",
         "Two ways to generate the futures. <b>Model</b> uses smooth textbook "
         "randomness, which is tidy but understates wild days. <b>Bootstrap</b> "
         "re-deals the stock's actual past days in random order, so the wild days it "
         "has lived through stay in the deck. The <b>seed</b> is the shuffle number: "
         "the same seed reproduces the identical run, which lets a colleague verify "
         "your exact chart.")

    eyebrow("Single-Factor Model")
    term(BLUE, "Beta",
         "How much the stock tends to move when the market moves 1%. A beta of 0.9 "
         "means it moves about 0.9%, slightly less than the market. A beta of 1.5 "
         "amplifies every market swing by half again, and a beta near 0 barely follows "
         "the market at all. This is the part of the risk shared with everyone holding "
         "the index.")
    term(GREEN, "Alpha /yr",
         "The return left over after the market's influence is removed. It is usually "
         "labeled <b>not significant</b>, which means the data cannot distinguish it "
         "from zero. Most measured alpha is luck rather than skill, so a "
         "non-significant alpha is the normal result, not a failure.")
    term(GOLD, "R²",
         "The share of the stock's day-to-day movement that mirrors the market, from 0 "
         "to 100%. At 42%, a bit under half of every move follows the market. The rest "
         "comes from the stock itself: earnings, news, sentiment.")
    term(PURPLE, "Idiosyncratic vol & the variance split",
         "The stock's own turbulence, unrelated to the market. Holding many stocks "
         "cancels this part out, which is why it is called diversifiable. The market "
         "share of the bar cannot be diversified away.")
    term(BLUE, "Rolling beta",
         "Beta re-measured over a sliding 3-month window. The headline beta is a "
         "single number; this line shows how it drifts. If it wanders far from the "
         "dashed line, the relationship is unstable and the headline number deserves "
         "less trust.")

    note(GOLD, "How to use these numbers",
         "Quote CVaR together with VaR: the first says where trouble starts, the "
         "second says how deep it runs. When the backtest says off target, believe it "
         "over the headline number. And remember that every figure here is measured "
         "from the past. Markets can behave differently tomorrow.")
    footer()


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown(f'<div style="font-weight:700;letter-spacing:.06em;color:{GREEN};'
                f'margin-bottom:8px">RiCode</div>', unsafe_allow_html=True)
    tool = st.radio("Tool", ["Value at Risk", "Monte Carlo", "Single-Factor Model",
                             "Reader's Guide"],
                    label_visibility="collapsed")
    st.divider()

if tool == "Value at Risk":
    page_var()
elif tool == "Monte Carlo":
    page_mc()
elif tool == "Single-Factor Model":
    page_factor()
else:
    page_guide()
