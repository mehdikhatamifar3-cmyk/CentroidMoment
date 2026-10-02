import base64
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import sympy as sp
from matplotlib.patches import Rectangle

st.set_page_config(
    page_title="Area Properties Integration Tutor",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
LOGO_PATH = APP_DIR / "assets" / "jcu_logo.png"
LOGO_B64 = (
    base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    if LOGO_PATH.exists()
    else ""
)

x, y = sp.symbols("x y", real=True)
SYSTEM_RNG = random.SystemRandom()

st.markdown(
    """
    <style>
      :root {
        color-scheme: light;
        --text: #172033;
        --muted: #526174;
        --border: #cbd5e1;
        --accent: #c8102e;
      }
      html, body, [data-testid="stAppViewContainer"], .stApp {
        background: #ffffff !important;
        color: var(--text) !important;
      }
      .block-container { max-width: 1450px; padding-top: 1.0rem; padding-bottom: 2.4rem; }
      [data-testid="stSidebar"] { background: #eef1f5 !important; border-right: 1px solid #dde2e8; }
      .app-header {
        display: grid; grid-template-columns: 230px 1fr 210px;
        gap: 24px; align-items: center; padding: 14px 10px 20px;
      }
      .app-logo { display:flex; justify-content:center; align-items:center; }
      .app-logo img { width:210px; max-height:92px; object-fit:contain; }
      .app-title h1 { margin:0; font-size:34px; line-height:1.05; color:#0f172a; }
      .app-title p { margin:8px 0 0; color:#334155; line-height:1.45; }
      .app-author { text-align:right; color:#334155; font-size:13px; line-height:1.5; }
      .divider { height:1px; background:#d9dee5; margin:2px 0 18px; }
      .info-box { background:#e8f2ff; border:1px solid #bfdbfe; border-radius:10px; padding:13px 15px; margin:8px 0 16px; }
      .hint-box { background:#ecfdf3; border:1px solid #bbf7d0; border-radius:10px; padding:12px 14px; margin:8px 0 14px; }
      .warning-box { background:#fff8e8; border:1px solid #f3d69b; border-radius:10px; padding:12px 14px; margin:8px 0 14px; }
      .method-card { background:#fff; border:1px solid #dfe4ea; border-left:5px solid #0b5fa5; border-radius:10px; padding:14px 16px; margin:10px 0; }
      .step-card { background:#f8fafc; border:1px solid #dbe3ec; border-radius:10px; padding:12px 14px; margin:8px 0; }
      .good { background:#ecfdf3; border:1px solid #bbf7d0; color:#166534; border-radius:9px; padding:10px 12px; margin:6px 0; }
      .bad { background:#fff1f2; border:1px solid #fecdd3; color:#9f1239; border-radius:9px; padding:10px 12px; margin:6px 0; }
      [data-baseweb="tab"][aria-selected="true"] p { color:var(--accent) !important; font-weight:700; }
      .stButton > button[kind="primary"] { background:#b91c1c !important; color:#fff !important; border-color:#991b1b !important; }
      @media (max-width: 980px) {
        .app-header { grid-template-columns:1fr; text-align:center; }
        .app-author { text-align:center; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_header():
    logo = (
        f'<img src="data:image/png;base64,{LOGO_B64}" alt="James Cook University">'
        if LOGO_B64
        else '<div style="font-size:25px;font-weight:800;color:#0b5fa5">AREA<br>PROPERTIES</div>'
    )
    st.markdown(
        f"""
        <div class="app-header">
          <div class="app-logo">{logo}</div>
          <div class="app-title">
            <h1>Area Properties Integration Tutor</h1>
            <p>Learn centroid and second moment of area by constructing vertical and horizontal differential strips.</p>
          </div>
          <div class="app-author">
            Designed for first-year engineering<br>
            <b>Dr. Mehdi Khatamifar</b><br>
            James Cook University
          </div>
        </div>
        <div class="divider"></div>
        """,
        unsafe_allow_html=True,
    )


def fmt(value, decimals=5):
    value = float(sp.N(value))
    if abs(value - round(value)) < 1e-10:
        return str(int(round(value)))
    return f"{value:.{decimals}f}".rstrip("0").rstrip(".")


def make_answer(label, value, unit):
    value = float(sp.N(value))
    return {
        "label": label,
        "value": value,
        "unit": unit,
        "tolerance": max(0.015, 0.01 * max(1.0, abs(value))),
    }


def solve_vertical(top, bottom, lower, upper):
    top, bottom = sp.sympify(top), sp.sympify(bottom)
    height = sp.simplify(top - bottom)
    area = sp.simplify(sp.integrate(height, (x, lower, upper)))
    x_tilde = x
    y_tilde = sp.simplify((top + bottom) / 2)
    q_y = sp.simplify(sp.integrate(x_tilde * height, (x, lower, upper)))
    q_x = sp.simplify(sp.integrate(y_tilde * height, (x, lower, upper)))
    x_bar = sp.simplify(q_y / area)
    y_bar = sp.simplify(q_x / area)
    i_x = sp.simplify(sp.integrate((top**3 - bottom**3) / 3, (x, lower, upper)))
    i_y = sp.simplify(sp.integrate(x**2 * height, (x, lower, upper)))
    return {
        "cut": "vertical", "top": top, "bottom": bottom,
        "lower": sp.sympify(lower), "upper": sp.sympify(upper),
        "dA": height, "area": area, "x_tilde": x_tilde, "y_tilde": y_tilde,
        "q_x": q_x, "q_y": q_y, "x_bar": x_bar, "y_bar": y_bar,
        "i_x": i_x, "i_y": i_y,
        "i_x_bar": sp.simplify(i_x - area * y_bar**2),
        "i_y_bar": sp.simplify(i_y - area * x_bar**2),
    }


def solve_horizontal(right, left, lower, upper):
    right, left = sp.sympify(right), sp.sympify(left)
    width = sp.simplify(right - left)
    area = sp.simplify(sp.integrate(width, (y, lower, upper)))
    x_tilde = sp.simplify((right + left) / 2)
    y_tilde = y
    q_y = sp.simplify(sp.integrate(x_tilde * width, (y, lower, upper)))
    q_x = sp.simplify(sp.integrate(y_tilde * width, (y, lower, upper)))
    x_bar = sp.simplify(q_y / area)
    y_bar = sp.simplify(q_x / area)
    i_x = sp.simplify(sp.integrate(y**2 * width, (y, lower, upper)))
    i_y = sp.simplify(sp.integrate((right**3 - left**3) / 3, (y, lower, upper)))
    return {
        "cut": "horizontal", "right": right, "left": left,
        "lower": sp.sympify(lower), "upper": sp.sympify(upper),
        "dA": width, "area": area, "x_tilde": x_tilde, "y_tilde": y_tilde,
        "q_x": q_x, "q_y": q_y, "x_bar": x_bar, "y_bar": y_bar,
        "i_x": i_x, "i_y": i_y,
        "i_x_bar": sp.simplify(i_x - area * y_bar**2),
        "i_y_bar": sp.simplify(i_y - area * x_bar**2),
    }


def generate_problem(level, seed):
    rng = random.Random(seed)
    if level == 1:
        width, height = rng.choice([4, 5, 6]), rng.choice([2, 3, 4])
        data = solve_vertical(height, 0, 0, width)
        asks = ["area", "x_bar", "y_bar"]
        title, difficulty = "Rectangle using a vertical strip", "Foundation"
        purpose = "Identify the strip dimensions, write dA, and verify the centroid by integration."
    elif level == 2:
        width, height = rng.choice([3, 4, 5, 6]), rng.choice([3, 4, 5, 6])
        data = solve_vertical(height * (1 - x / width), 0, 0, width)
        asks = ["area", "x_bar", "y_bar"]
        title, difficulty = "Triangle below a straight line", "Developing"
        purpose = "Find the area and centroid by integration rather than by memorised triangle formulas."
    elif level == 3:
        width, height = rng.choice([2, 3, 4]), rng.choice([3, 4, 5, 6])
        data = solve_vertical(height * (1 - (x / width) ** 2), 0, 0, width)
        asks = ["area", "x_bar", "y_bar", "i_x", "i_y"]
        title, difficulty = "Area below a parabola", "Intermediate"
        purpose = "Use a vertical strip to determine the centroid and second moments about the shown axes."
    elif level == 4:
        height, width = rng.choice([2, 3, 4]), rng.choice([3, 4, 5, 6])
        data = solve_horizontal(width * (1 - (y / height) ** 2), 0, 0, height)
        asks = ["area", "x_bar", "y_bar", "i_x", "i_y"]
        title, difficulty = "Region described by x as a function of y", "Intermediate"
        purpose = "Use a horizontal strip because the boundaries are naturally expressed as functions of y."
    elif level == 5:
        width, height = rng.choice([3, 4, 5]), rng.choice([3, 4, 5])
        data = solve_vertical(height, height * (x / width) ** 2, 0, width)
        asks = ["area", "x_bar", "y_bar", "i_x_bar", "i_y_bar"]
        title, difficulty = "Area between a horizontal line and a parabola", "Challenging"
        purpose = "Determine the centroid and centroidal second moments for a region between two boundaries."
    else:
        width, height = rng.choice([4, 6, 8]), rng.choice([4, 5, 6])
        data = solve_vertical(height * (1 - (x / width) ** 2), height * x / (3 * width), 0, width / 2)
        asks = ["area", "x_bar", "y_bar", "i_x", "i_y", "i_x_bar", "i_y_bar"]
        title, difficulty = "Region between two varying curves", "Advanced"
        purpose = "Build top-minus-bottom carefully and determine all reference-axis and centroidal properties."
    return {
        "title": title, "difficulty": difficulty, "purpose": purpose,
        "data": data, "asks": asks, "uid": f"L{level}_{seed}",
    }


def region_figure(data, strip_position, show_area_centroid=False, title=None, stage="all"):
    """Draw the region from its exact equations with transparent, non-overlapping annotations."""
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    blue, navy, green = "#60a5fa", "#1d4ed8", "#047857"
    orange, purple, red = "#f97316", "#7c3aed", "#dc2626"
    lower, upper = map(float, (data["lower"], data["upper"]))
    span = max(upper - lower, 1.0)
    bbox = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.82)

    if data["cut"] == "vertical":
        values = np.linspace(lower, upper, 500)
        top = np.asarray(sp.lambdify(x, data["top"], "numpy")(values), dtype=float) + np.zeros_like(values)
        bottom = np.asarray(sp.lambdify(x, data["bottom"], "numpy")(values), dtype=float) + np.zeros_like(values)
        ax.fill_between(values, bottom, top, color=blue, alpha=0.28, label="Complete region A")
        ax.plot(values, top, color=navy, lw=2.6, label=r"Upper boundary $y_T(x)$")
        ax.plot(values, bottom, color=green, lw=2.3, label=r"Lower boundary $y_B(x)$")
        s = float(strip_position)
        top_s = float(data["top"].subs(x, s)); bottom_s = float(data["bottom"].subs(x, s))
        half = max(span * 0.016, 0.018)
        ax.add_patch(Rectangle((s-half, bottom_s), 2*half, top_s-bottom_s,
                               facecolor=orange, edgecolor="#c2410c", lw=1.4,
                               alpha=0.58, zorder=5, label="Differential strip"))
        strip_centroid = (s, (top_s+bottom_s)/2)
        ax.annotate(r"$dx$", xy=(s, bottom_s), xytext=(0, -25), textcoords="offset points",
                    ha="center", color="#9a3412", weight="bold", bbox=bbox,
                    arrowprops=dict(arrowstyle="-[,widthB=1.0", color="#c2410c", alpha=.75))
        ax.annotate(r"height $=y_T-y_B$", xy=strip_centroid, xytext=(42, 0),
                    textcoords="offset points", va="center", color="#9a3412", weight="bold",
                    bbox=bbox, arrowprops=dict(arrowstyle="->", color=orange, alpha=.8))
        boundary_note = rf"$y_T={sp.latex(data['top'])}$" + r"\n" + rf"$y_B={sp.latex(data['bottom'])}$"
        ax.text(0.02, 0.98, boundary_note, transform=ax.transAxes, va="top", ha="left",
                color="#0f172a", fontsize=10, bbox=bbox)
    else:
        values = np.linspace(lower, upper, 500)
        right = np.asarray(sp.lambdify(y, data["right"], "numpy")(values), dtype=float) + np.zeros_like(values)
        left = np.asarray(sp.lambdify(y, data["left"], "numpy")(values), dtype=float) + np.zeros_like(values)
        ax.fill_betweenx(values, left, right, color=blue, alpha=0.28, label="Complete region A")
        ax.plot(right, values, color=navy, lw=2.6, label=r"Right boundary $x_R(y)$")
        ax.plot(left, values, color=green, lw=2.3, label=r"Left boundary $x_L(y)$")
        s = float(strip_position)
        right_s = float(data["right"].subs(y, s)); left_s = float(data["left"].subs(y, s))
        half = max(span * 0.016, 0.018)
        ax.add_patch(Rectangle((left_s, s-half), right_s-left_s, 2*half,
                               facecolor=orange, edgecolor="#c2410c", lw=1.4,
                               alpha=0.58, zorder=5, label="Differential strip"))
        strip_centroid = ((right_s+left_s)/2, s)
        ax.annotate(r"$dy$", xy=(left_s, s), xytext=(-28, 0), textcoords="offset points",
                    va="center", color="#9a3412", weight="bold", bbox=bbox,
                    arrowprops=dict(arrowstyle="-[,widthB=1.0", color="#c2410c", alpha=.75))
        ax.annotate(r"width $=x_R-x_L$", xy=strip_centroid, xytext=(0, 34),
                    textcoords="offset points", ha="center", color="#9a3412", weight="bold",
                    bbox=bbox, arrowprops=dict(arrowstyle="->", color=orange, alpha=.8))
        boundary_note = rf"$x_R={sp.latex(data['right'])}$" + r"\n" + rf"$x_L={sp.latex(data['left'])}$"
        ax.text(0.02, 0.98, boundary_note, transform=ax.transAxes, va="top", ha="left",
                color="#0f172a", fontsize=10, bbox=bbox)

    if stage in ("strip", "centroid", "all"):
        ax.scatter(*strip_centroid, s=90, color=purple, edgecolor="white", linewidth=1.0, zorder=8)
        ax.annotate(r"strip centroid" + r"\n" + r"$(\tilde{x},\tilde{y})$", xy=strip_centroid,
                    xytext=(-62, 35), textcoords="offset points", ha="center",
                    color=purple, weight="bold", bbox=bbox,
                    arrowprops=dict(arrowstyle="->", color=purple, alpha=.8))
    if show_area_centroid or stage in ("centroid", "all_centroid"):
        whole = (float(data["x_bar"]), float(data["y_bar"]))
        ax.scatter(*whole, marker="X", s=180, color=red, edgecolor="white", linewidth=1.2, zorder=9)
        ax.annotate(r"complete-area centroid" + r"\n" + r"$(\bar{x},\bar{y})$", xy=whole,
                    xytext=(62, -38), textcoords="offset points", ha="center",
                    color=red, weight="bold", bbox=bbox,
                    arrowprops=dict(arrowstyle="->", color=red, alpha=.8))

    ax.axhline(0, color="#64748b", lw=1); ax.axvline(0, color="#64748b", lw=1)
    ax.grid(alpha=0.12); ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper right", fontsize=9, framealpha=.88)
    if title: ax.set_title(title, weight="bold", pad=12)
    fig.tight_layout(pad=1.5)
    return fig

def render_interactive_diagram(problem, key_prefix):
    data = problem["data"]
    lower, upper = map(float, (data["lower"], data["upper"]))
    position = st.slider(
        "Move the differential strip",
        min_value=lower,
        max_value=upper,
        value=(lower + upper) / 2,
        key=f"{key_prefix}_strip_slider",
    )
    show = st.checkbox(
        "Reveal the centroid of the complete area",
        key=f"{key_prefix}_area_centroid",
    )
    st.pyplot(region_figure(data, position, show), use_container_width=True)
    st.caption("Blue: complete region | Orange: differential strip | Purple: strip centroid | Red X: complete-area centroid")


def render_static_diagram(data, title, show_centroid=True):
    position = float((data["lower"] + data["upper"]) / 2)
    st.pyplot(region_figure(data, position, show_centroid, title), use_container_width=True)


def render_solution_process_figures(data):
    """Use three coded figures to visually build the solution without repeated widgets."""
    position = float(data["lower"] + 0.58 * (data["upper"] - data["lower"]))
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("#### Figure A: Boundaries and differential strip")
        st.pyplot(region_figure(data, position, False,
                  "1. Identify boundaries and form dA", stage="strip"),
                  use_container_width=True)
    with c2:
        st.markdown("#### Figure B: Strip-centroid coordinates")
        st.pyplot(region_figure(data, position, False,
                  "2. Locate the centroid of the moving strip", stage="centroid"),
                  use_container_width=True)
    st.markdown("#### Figure C: Complete-area centroid and reference axes")
    st.pyplot(region_figure(data, position, True,
              "3. Integrate over the complete region and locate the area centroid", stage="all"),
              use_container_width=True)


def render_learn_tab():
    st.markdown("## Learn the method")
    st.markdown(
        '<div class="info-box"><b>Geometry first, integration second.</b> '
        'Choose a strip, write its dimensions, locate the strip centroid, establish the limits, and only then integrate.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### 1. Integration refresher")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("#### Reverse power rule")
        st.latex(r"\int x^n\,dx=\frac{x^{n+1}}{n+1}+C,\qquad n\ne-1")
        st.write("Increase the power by one, then divide by the new power.")
        st.latex(r"\int 3x^2\,dx=x^3+C")
    with c2:
        st.markdown("#### Definite integral")
        st.latex(r"\int_a^b f(x)\,dx=F(b)-F(a)")
        st.write("Find the antiderivative first. Substitute the upper limit, then subtract the lower-limit value.")
        st.latex(r"\int_0^2 3x^2\,dx=[x^3]_0^2=8")

    st.markdown("### 2. Choose and define the differential strip")
    vertical_col, horizontal_col = st.columns(2, gap="large")
    with vertical_col:
        st.markdown("#### Vertical cutting")
        demo = solve_vertical(4 - x**2 / 4, 0, 0, 4)
        render_static_diagram(demo, "Vertical strip: thickness dx, height top minus bottom", False)
        st.latex(r"dA=[y_T(x)-y_B(x)]\,dx")
        st.latex(r"\tilde{x}=x,\qquad \tilde{y}=\frac{y_T+y_B}{2}")
        st.write("Choose this cut when the upper and lower boundaries are naturally written as functions of x. The integration limits are x-coordinates.")
        st.latex(r"I_y=\int x^2[y_T-y_B]\,dx")
        st.latex(r"I_x=\int\frac{y_T^3-y_B^3}{3}\,dx")
    with horizontal_col:
        st.markdown("#### Horizontal cutting")
        demo = solve_horizontal(4 - y**2 / 4, 0, 0, 4)
        render_static_diagram(demo, "Horizontal strip: thickness dy, width right minus left", False)
        st.latex(r"dA=[x_R(y)-x_L(y)]\,dy")
        st.latex(r"\tilde{x}=\frac{x_R+x_L}{2},\qquad \tilde{y}=y")
        st.write("Choose this cut when the right and left boundaries are naturally written as functions of y. The integration limits are y-coordinates.")
        st.latex(r"I_x=\int y^2[x_R-x_L]\,dy")
        st.latex(r"I_y=\int\frac{x_R^3-x_L^3}{3}\,dy")

    st.markdown("### 3. Area, centroid, and centroidal axes")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.latex(r"A=\int dA")
        st.latex(r"\bar{x}=\frac{\int\tilde{x}\,dA}{A}")
        st.latex(r"\bar{y}=\frac{\int\tilde{y}\,dA}{A}")
        st.write("The tilde coordinates belong to one moving strip. The bar coordinates belong to the complete area.")
    with c2:
        st.latex(r"I_{\bar{x}}=I_x-A\bar{y}^{\,2}")
        st.latex(r"I_{\bar{y}}=I_y-A\bar{x}^{\,2}")
        st.write("First calculate about the shown reference axes. Then move to parallel centroidal axes using the parallel-axis theorem.")

    st.markdown(
        '<div class="warning-box"><b>Important:</b> for a deep vertical strip, '
        r'$I_x$ is not generally obtained from only $\tilde{y}^{2}dA$. '
        'The strip has its own local second moment, so integrate through its depth or use '
        r'$dI_x=(y_T^3-y_B^3)dx/3$.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### 4. Fully worked teaching example")
    example = solve_vertical(4 - x, 0, 0, 4)
    left, right = st.columns(2, gap="large")
    with left:
        render_static_diagram(example, "Example: triangular region under y = 4 - x", True)
    with right:
        st.write("The region lies below y = 4 - x, above y = 0, from x = 0 to x = 4.")
        st.markdown("**Step 1: Differential strip**")
        st.latex(r"dA=[(4-x)-0]dx=(4-x)dx")
        st.markdown("**Step 2: Strip centroid**")
        st.latex(r"\tilde{x}=x,\qquad\tilde{y}=\frac{(4-x)+0}{2}=\frac{4-x}{2}")
        st.markdown("**Step 3: Area**")
        st.latex(r"A=\int_0^4(4-x)dx=8")
        st.markdown("**Step 4: Complete-area centroid**")
        st.latex(r"\bar{x}=\frac{\int_0^4x(4-x)dx}{8}=\frac{4}{3}")
        st.latex(r"\bar{y}=\frac{\int_0^4\frac{4-x}{2}(4-x)dx}{8}=\frac{4}{3}")
        st.markdown("**Step 5: Second moments about the shown axes**")
        st.latex(r"I_x=\int_0^4\frac{(4-x)^3}{3}dx=\frac{64}{3}")
        st.latex(r"I_y=\int_0^4x^2(4-x)dx=\frac{64}{3}")
        st.success("Check: the centroid lies inside the triangle and second-moment units are length to the fourth power.")


def render_detailed_solution(problem):
    data = problem["data"]
    vertical = data["cut"] == "vertical"
    differential = "dx" if vertical else "dy"
    variable = "x" if vertical else "y"

    st.markdown("## Detailed step-by-step solution")
    st.markdown('<div class="info-box">The figures below build the method visually: boundaries → differential strip → strip centroid → complete-area centroid.</div>', unsafe_allow_html=True)
    render_solution_process_figures(data)

    st.markdown("### Step 1: Choose the cut and write the boundaries")
    if vertical:
        st.write("Use a vertical strip because the upper and lower boundaries are functions of x.")
        st.latex(rf"y_T(x)={sp.latex(data['top'])},\qquad y_B(x)={sp.latex(data['bottom'])}")
        st.write("The strip height is upper boundary minus lower boundary, so:")
    else:
        st.write("Use a horizontal strip because the right and left boundaries are functions of y.")
        st.latex(rf"x_R(y)={sp.latex(data['right'])},\qquad x_L(y)={sp.latex(data['left'])}")
        st.write("The strip width is right boundary minus left boundary, so:")
    st.latex(rf"dA=\left({sp.latex(data['dA'])}\right){differential}")

    st.markdown("### Step 2: Locate the centroid of the differential strip")
    st.write("The tilde coordinates describe the centroid of one moving strip, not the centroid of the complete area.")
    st.latex(rf"\tilde{{x}}={sp.latex(data['x_tilde'])},\qquad\tilde{{y}}={sp.latex(data['y_tilde'])}")

    st.markdown("### Step 3: Establish the integration limits")
    st.write(f"The strip sweeps over the complete region from {variable}={sp.latex(data['lower'])} to {variable}={sp.latex(data['upper'])}.")
    st.latex(rf"{sp.latex(data['lower'])}\le {variable}\le {sp.latex(data['upper'])}")

    st.markdown("### Step 4: Calculate the total area")
    st.latex(rf"A=\int_{{{sp.latex(data['lower'])}}}^{{{sp.latex(data['upper'])}}}\left({sp.latex(data['dA'])}\right){differential}")
    st.latex(rf"A={sp.latex(data['area'])}\;\mathrm{{units}}^2\approx {fmt(data['area'])}\;\mathrm{{units}}^2")

    st.markdown("### Step 5: Calculate the centroid using the full centroid equations")
    st.write("Substitute the strip-centroid coordinate and dA directly into each centroid equation.")
    st.latex(rf"\bar{{x}}=\frac{{\int_{{{sp.latex(data['lower'])}}}^{{{sp.latex(data['upper'])}}}\tilde{{x}}\,dA}}{{\int_{{{sp.latex(data['lower'])}}}^{{{sp.latex(data['upper'])}}}dA}}")
    st.latex(rf"\bar{{x}}=\frac{{\int_{{{sp.latex(data['lower'])}}}^{{{sp.latex(data['upper'])}}}\left({sp.latex(data['x_tilde'])}\right)\left({sp.latex(data['dA'])}\right){differential}}}{{{sp.latex(data['area'])}}}={sp.latex(data['x_bar'])}\approx {fmt(data['x_bar'])}\;\mathrm{{units}}")
    st.latex(rf"\bar{{y}}=\frac{{\int_{{{sp.latex(data['lower'])}}}^{{{sp.latex(data['upper'])}}}\tilde{{y}}\,dA}}{{\int_{{{sp.latex(data['lower'])}}}^{{{sp.latex(data['upper'])}}}dA}}")
    st.latex(rf"\bar{{y}}=\frac{{\int_{{{sp.latex(data['lower'])}}}^{{{sp.latex(data['upper'])}}}\left({sp.latex(data['y_tilde'])}\right)\left({sp.latex(data['dA'])}\right){differential}}}{{{sp.latex(data['area'])}}}={sp.latex(data['y_bar'])}\approx {fmt(data['y_bar'])}\;\mathrm{{units}}")

    st.markdown("### Step 6: Calculate second moments about the shown axes")
    if vertical:
        st.write("For Iy, the thin strip is located at x. For Ix, integrate through the full strip depth from yB to yT.")
        st.latex(rf"I_y=\int_{{{sp.latex(data['lower'])}}}^{{{sp.latex(data['upper'])}}}x^2[y_T-y_B]dx={sp.latex(data['i_y'])}\;\mathrm{{units}}^4")
        st.latex(rf"I_x=\int_{{{sp.latex(data['lower'])}}}^{{{sp.latex(data['upper'])}}}\frac{{y_T^3-y_B^3}}{{3}}dx={sp.latex(data['i_x'])}\;\mathrm{{units}}^4")
    else:
        st.write("For Ix, the thin strip is located at y. For Iy, integrate through the full strip width from xL to xR.")
        st.latex(rf"I_x=\int_{{{sp.latex(data['lower'])}}}^{{{sp.latex(data['upper'])}}}y^2[x_R-x_L]dy={sp.latex(data['i_x'])}\;\mathrm{{units}}^4")
        st.latex(rf"I_y=\int_{{{sp.latex(data['lower'])}}}^{{{sp.latex(data['upper'])}}}\frac{{x_R^3-x_L^3}}{{3}}dy={sp.latex(data['i_y'])}\;\mathrm{{units}}^4")

    st.markdown("### Step 7: Shift to centroidal axes when required")
    st.latex(rf"I_{{\bar{{x}}}}=I_x-A\bar{{y}}^2={sp.latex(data['i_x_bar'])}\;\mathrm{{units}}^4")
    st.latex(rf"I_{{\bar{{y}}}}=I_y-A\bar{{x}}^2={sp.latex(data['i_y_bar'])}\;\mathrm{{units}}^4")

    st.markdown("### Final checks")
    c1, c2, c3 = st.columns(3)
    c1.metric("Area", f"{fmt(data['area'])} units²")
    c2.metric("x̄", f"{fmt(data['x_bar'])} units")
    c3.metric("ȳ", f"{fmt(data['y_bar'])} units")
    st.markdown('<div class="hint-box"><b>Checks:</b> area is positive; the centroid lies inside the region; second moments are positive; centroid units are length; second-moment units are length⁴.</div>', unsafe_allow_html=True)


render_header()

st.sidebar.markdown("## Problem setup")
level_names = {
    1: "1 · Rectangle",
    2: "2 · Triangle",
    3: "3 · Parabola",
    4: "4 · Horizontal strip",
    5: "5 · Between curves",
    6: "6 · Advanced",
}
level = st.sidebar.selectbox("Level", list(level_names), format_func=level_names.get)

if "current_seed" not in st.session_state:
    st.session_state.current_seed = SYSTEM_RNG.randrange(1, 2**63)
if st.session_state.get("current_level") != level:
    st.session_state.current_level = level
    st.session_state.current_seed = SYSTEM_RNG.randrange(1, 2**63)
if st.sidebar.button("🎲 New random question", use_container_width=True):
    st.session_state.current_seed = SYSTEM_RNG.randrange(1, 2**63)

problem = generate_problem(level, st.session_state.current_seed)
data = problem["data"]
st.sidebar.caption(f"Current problem: {problem['title']}")
st.sidebar.markdown("---")
st.sidebar.markdown("### Colour key")
st.sidebar.markdown("🔵 Complete region  \n🟧 Differential strip  \n🟣 Strip centroid  \n❌ Complete-area centroid")

labels = {
    "area": ("Area A", "units²"),
    "x_bar": ("Centroid x̄", "units"),
    "y_bar": ("Centroid ȳ", "units"),
    "i_x": ("Iₓ about shown x-axis", "units⁴"),
    "i_y": ("Iᵧ about shown y-axis", "units⁴"),
    "i_x_bar": ("Iₓ̄ about centroidal axis", "units⁴"),
    "i_y_bar": ("Iᵧ̄ about centroidal axis", "units⁴"),
}

tabs = st.tabs([
    "0. Learn the topic",
    "1. Problem",
    "2. Build the method",
    "3. Solve and check",
    "4. Detailed solution",
])

with tabs[0]:
    render_learn_tab()

with tabs[1]:
    st.markdown("## Current problem")
    left, right = st.columns([0.9, 1.1], gap="large")
    with left:
        st.caption(f"Level {level} | Difficulty: {problem['difficulty']}")
        st.markdown(f"### {problem['title']}")
        st.write(problem["purpose"])
        if data["cut"] == "vertical":
            st.latex(rf"y_T(x)={sp.latex(data['top'])},\qquad y_B(x)={sp.latex(data['bottom'])}")
            st.latex(rf"{sp.latex(data['lower'])}\le x\le {sp.latex(data['upper'])}")
        else:
            st.latex(rf"x_R(y)={sp.latex(data['right'])},\qquad x_L(y)={sp.latex(data['left'])}")
            st.latex(rf"{sp.latex(data['lower'])}\le y\le {sp.latex(data['upper'])}")
        st.markdown(
            '<div class="method-card"><b>Your task</b><br>'
            '1. Choose the cut.<br>2. Write dA.<br>3. Locate the strip centroid.<br>'
            '4. Establish the limits.<br>5. Build and evaluate the integrals.<br>'
            '6. Check units and physical reasonableness.</div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("### Interactive coded diagram")
        render_interactive_diagram(problem, key_prefix=f"problem_{problem['uid']}")

with tabs[2]:
    st.markdown("## Build the method")
    st.write("Make the geometric decisions first. The feedback identifies exactly where the setup needs correction.")
    expected_cut = "Vertical" if data["cut"] == "vertical" else "Horizontal"
    expected_differential = "dx" if data["cut"] == "vertical" else "dy"
    c1, c2 = st.columns(2)
    cut_choice = c1.radio("1. Cutting direction", ["Vertical", "Horizontal"], key=f"cut_{problem['uid']}")
    differential = c2.radio("2. Strip thickness", ["dx", "dy"], key=f"diff_{problem['uid']}")
    lower = st.number_input("3. Lower integration limit", value=float(data["lower"]), key=f"lower_{problem['uid']}")
    upper = st.number_input("4. Upper integration limit", value=float(data["upper"]), key=f"upper_{problem['uid']}")
    if st.button("Check my setup", type="primary", key=f"method_check_{problem['uid']}"):
        correct = True
        if cut_choice != expected_cut:
            st.error(f"Use a {expected_cut.lower()} cut because the supplied boundaries are most direct in that orientation.")
            correct = False
        if differential != expected_differential:
            st.error(f"A {expected_cut.lower()} strip has thickness {expected_differential}.")
            correct = False
        if not math.isclose(lower, float(data["lower"]), abs_tol=1e-8):
            st.error("Recheck the lower coordinate reached by the integration variable.")
            correct = False
        if not math.isclose(upper, float(data["upper"]), abs_tol=1e-8):
            st.error("Recheck the upper coordinate reached by the integration variable.")
            correct = False
        if correct:
            st.success("Correct setup. The strip, differential, and limits are consistent.")
            st.latex(rf"dA=\left({sp.latex(data['dA'])}\right){expected_differential}")
            st.latex(rf"\tilde{{x}}={sp.latex(data['x_tilde'])},\qquad\tilde{{y}}={sp.latex(data['y_tilde'])}")

with tabs[3]:
    st.markdown("## Solve and check")
    st.markdown(
        '<div class="info-box">Enter final numerical values. Keep full precision during calculation and round only at the end.</div>',
        unsafe_allow_html=True,
    )
    entered = {}
    columns = st.columns(2)
    for index, key in enumerate(problem["asks"]):
        label, unit = labels[key]
        with columns[index % 2]:
            entered[key] = st.number_input(
                f"{label} ({unit})",
                value=0.0,
                format="%.5f",
                key=f"answer_{key}_{problem['uid']}",
            )
    if st.button("Check final answers", type="primary", key=f"answer_check_{problem['uid']}"):
        all_correct = True
        for key in problem["asks"]:
            answer = make_answer(*labels[key], data[key]) if False else None
            expected = float(sp.N(data[key]))
            tolerance = max(0.015, 0.01 * max(1.0, abs(expected)))
            result = abs(entered[key] - expected) <= tolerance
            all_correct = all_correct and result
            if result:
                st.markdown(f'<div class="good">{labels[key][0]}: correct.</div>', unsafe_allow_html=True)
            else:
                detail = f" Expected approximately {fmt(expected)} {labels[key][1]}."
                st.markdown(
                    f'<div class="bad">{labels[key][0]}: check the strip setup, limits, formula, and arithmetic.{detail}</div>',
                    unsafe_allow_html=True,
                )
        if all_correct:
            st.success("Excellent. All requested area properties are correct.")
    st.markdown("### Reasonableness checks")
    st.write("• Area must be positive.  \n• The centroid should lie inside the region.  \n• Second moments must be positive.  \n• Centroid units are length; second-moment units are length⁴.")

with tabs[4]:
    render_detailed_solution(problem)
