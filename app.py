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
      .block-container { max-width: 1280px; padding-top: 1.0rem; padding-bottom: 2.4rem; }
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
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    blue, navy, green = "#38bdf8", "#0047AB", "#00A651"
    orange, purple, red = "#FF7A00", "#7A00CC", "#E00034"
    lower, upper = map(float, (data["lower"], data["upper"]))
    span = max(upper - lower, 1.0)
    bbox = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.82)

    if data["cut"] == "vertical":
        values = np.linspace(lower, upper, 500)
        top = np.asarray(sp.lambdify(x, data["top"], "numpy")(values), dtype=float) + np.zeros_like(values)
        bottom = np.asarray(sp.lambdify(x, data["bottom"], "numpy")(values), dtype=float) + np.zeros_like(values)
        ax.fill_between(values, bottom, top, color=blue, alpha=0.38, label="Complete region A")
        ax.plot(values, top, color=navy, lw=3.2, label=r"Upper boundary $y_T(x)$")
        ax.plot(values, bottom, color=green, lw=3.0, label=r"Lower boundary $y_B(x)$")
        s = float(strip_position)
        top_s = float(data["top"].subs(x, s)); bottom_s = float(data["bottom"].subs(x, s))
        half = max(span * 0.016, 0.018)
        ax.add_patch(Rectangle((s-half, bottom_s), 2*half, top_s-bottom_s,
                               facecolor=orange, edgecolor="#9A3412", lw=2.0,
                               alpha=0.68, zorder=5, label="Differential strip"))
        strip_centroid = (s, (top_s+bottom_s)/2)
        ax.annotate(r"$dx$", xy=(s, bottom_s), xytext=(0, -25), textcoords="offset points",
                    ha="center", color="#9a3412", weight="bold", bbox=bbox,
                    arrowprops=dict(arrowstyle="-[,widthB=1.0", color="#c2410c", alpha=.75))
        ax.annotate(r"height $=y_T-y_B$", xy=strip_centroid, xytext=(42, 0),
                    textcoords="offset points", va="center", color="#9a3412", weight="bold",
                    bbox=bbox, arrowprops=dict(arrowstyle="->", color=orange, alpha=.8))
        boundary_note = rf"$y_T={sp.latex(data['top'])}$" + "\n" + rf"$y_B={sp.latex(data['bottom'])}$"
        ax.text(0.02, 0.98, boundary_note, transform=ax.transAxes, va="top", ha="left",
                color="#0f172a", fontsize=10, bbox=bbox)
    else:
        values = np.linspace(lower, upper, 500)
        right = np.asarray(sp.lambdify(y, data["right"], "numpy")(values), dtype=float) + np.zeros_like(values)
        left = np.asarray(sp.lambdify(y, data["left"], "numpy")(values), dtype=float) + np.zeros_like(values)
        ax.fill_betweenx(values, left, right, color=blue, alpha=0.38, label="Complete region A")
        ax.plot(right, values, color=navy, lw=3.2, label=r"Right boundary $x_R(y)$")
        ax.plot(left, values, color=green, lw=3.0, label=r"Left boundary $x_L(y)$")
        s = float(strip_position)
        right_s = float(data["right"].subs(y, s)); left_s = float(data["left"].subs(y, s))
        half = max(span * 0.016, 0.018)
        ax.add_patch(Rectangle((left_s, s-half), right_s-left_s, 2*half,
                               facecolor=orange, edgecolor="#9A3412", lw=2.0,
                               alpha=0.68, zorder=5, label="Differential strip"))
        strip_centroid = ((right_s+left_s)/2, s)
        ax.annotate(r"$dy$", xy=(left_s, s), xytext=(-28, 0), textcoords="offset points",
                    va="center", color="#9a3412", weight="bold", bbox=bbox,
                    arrowprops=dict(arrowstyle="-[,widthB=1.0", color="#c2410c", alpha=.75))
        ax.annotate(r"width $=x_R-x_L$", xy=strip_centroid, xytext=(0, 34),
                    textcoords="offset points", ha="center", color="#9a3412", weight="bold",
                    bbox=bbox, arrowprops=dict(arrowstyle="->", color=orange, alpha=.8))
        boundary_note = rf"$x_R={sp.latex(data['right'])}$" + "\n" + rf"$x_L={sp.latex(data['left'])}$"
        ax.text(0.02, 0.98, boundary_note, transform=ax.transAxes, va="top", ha="left",
                color="#0f172a", fontsize=10, bbox=bbox)

    if stage in ("strip", "centroid", "all"):
        ax.scatter(*strip_centroid, s=90, color=purple, edgecolor="white", linewidth=1.0, zorder=8)
        ax.annotate(r"strip centroid" + "\n" + r"$(\tilde{x},\tilde{y})$", xy=strip_centroid,
                    xytext=(-62, 35), textcoords="offset points", ha="center",
                    color=purple, weight="bold", bbox=bbox,
                    arrowprops=dict(arrowstyle="->", color=purple, alpha=.8))
    if show_area_centroid or stage in ("centroid", "all_centroid"):
        whole = (float(data["x_bar"]), float(data["y_bar"]))
        ax.scatter(*whole, marker="X", s=180, color=red, edgecolor="white", linewidth=1.2, zorder=9)
        ax.annotate(r"complete-area centroid" + "\n" + r"$(\bar{x},\bar{y})$", xy=whole,
                    xytext=(62, -38), textcoords="offset points", ha="center",
                    color=red, weight="bold", bbox=bbox,
                    arrowprops=dict(arrowstyle="->", color=red, alpha=.8))

    ax.axhline(0, color="#111827", lw=1.4); ax.axvline(0, color="#111827", lw=1.4)
    ax.grid(alpha=0.12); ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper right", fontsize=8, framealpha=.90)
    if title: ax.set_title(title, weight="bold", pad=9, fontsize=11)
    fig.subplots_adjust(left=0.08,right=0.98,top=0.90,bottom=0.10)
    return fig

def render_colour_key():
    items = [
        ("#38bdf8", "Complete region"),
        ("#0047AB", "Upper / right boundary"),
        ("#00A651", "Lower / left boundary"),
        ("#FF7A00", "Differential strip"),
        ("#7A00CC", "Strip centroid"),
        ("#E00034", "Complete-area centroid"),
        ("#111827", "Reference x- and y-axes"),
    ]
    html = "<div style='display:grid;gap:7px'>"
    for colour, text in items:
        html += (
            "<div style='display:flex;align-items:center;gap:9px'>"
            f"<span style='display:inline-block;width:18px;height:18px;border-radius:4px;"
            f"background:{colour};border:1px solid #334155'></span>"
            f"<span>{text}</span></div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


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
    st.caption("Cyan: complete region | Royal blue: upper/right boundary | Bright green: lower/left boundary | Orange: differential strip | Purple: strip centroid | Crimson X: complete-area centroid | Black: reference axes")


def render_static_diagram(data, title, show_centroid=True):
    position = float((data["lower"] + data["upper"]) / 2)
    left_space, figure_col, right_space = st.columns([0.15, 2.2, 0.15])
    with figure_col:
        st.pyplot(region_figure(data, position, show_centroid, title), use_container_width=True)


def render_solution_process_figures(data):
    """Show a compact visual sequence without adding duplicate interactive widgets."""
    position = float(data["lower"] + 0.58 * (data["upper"] - data["lower"]))
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown("#### 1. Boundaries and strip")
        st.pyplot(
            region_figure(data, position, False, "Form the differential area", stage="strip"),
            use_container_width=True,
        )
    with c2:
        st.markdown("#### 2. Strip centroid")
        st.pyplot(
            region_figure(data, position, False, "Locate the strip centroid", stage="centroid"),
            use_container_width=True,
        )
    st.markdown("#### 3. Complete-area centroid")
    st.pyplot(region_figure(data, position, True, "Integrate over the region", stage="all"), use_container_width=True)

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
        st.write("You should usually choose this cut when the upper and lower boundaries are already given as functions of x. The integration limits are x-coordinates.")
        st.markdown("##### How the second-moment formulas are obtained")
        st.write("For the moment about the y-axis, every point in the thin vertical strip is at approximately the same x-coordinate. Starting from the definition:")
        st.latex(r"I_y=\int_A x^2\,dA")
        st.latex(r"dI_y=x^2dA=x^2[y_T(x)-y_B(x)]dx")
        st.latex(r"I_y=\int_a^b x^2[y_T(x)-y_B(x)]\,dx")
        st.write("For the moment about the x-axis, y varies through the full depth of the strip. Use a small element dA = dy dx inside the strip and integrate first from y_B to y_T:")
        st.latex(r"dI_x=\int_{y_B}^{y_T}y^2\,dy\,dx")
        st.latex(r"dI_x=\left[\frac{y^3}{3}\right]_{y_B}^{y_T}dx=\frac{y_T^3-y_B^3}{3}dx")
        st.latex(r"I_x=\int_a^b\frac{y_T^3-y_B^3}{3}\,dx")
    with horizontal_col:
        st.markdown("#### Horizontal cutting")
        demo = solve_horizontal(4 - y**2 / 4, 0, 0, 4)
        render_static_diagram(demo, "Horizontal strip: thickness dy, width right minus left", False)
        st.latex(r"dA=[x_R(y)-x_L(y)]\,dy")
        st.latex(r"\tilde{x}=\frac{x_R+x_L}{2},\qquad \tilde{y}=y")
        st.write("You should usually choose this cut when the right and left boundaries are already given as functions of y. The integration limits are y-coordinates.")
        st.markdown("##### How the second-moment formulas are obtained")
        st.write("For the moment about the x-axis, every point in the thin horizontal strip is at approximately the same y-coordinate. Starting from the definition:")
        st.latex(r"I_x=\int_A y^2\,dA")
        st.latex(r"dI_x=y^2dA=y^2[x_R(y)-x_L(y)]dy")
        st.latex(r"I_x=\int_c^d y^2[x_R(y)-x_L(y)]\,dy")
        st.write("For the moment about the y-axis, x varies through the full width of the strip. Use dA = dx dy and integrate first from x_L to x_R:")
        st.latex(r"dI_y=\int_{x_L}^{x_R}x^2\,dx\,dy")
        st.latex(r"dI_y=\left[\frac{x^3}{3}\right]_{x_L}^{x_R}dy=\frac{x_R^3-x_L^3}{3}dy")
        st.latex(r"I_y=\int_c^d\frac{x_R^3-x_L^3}{3}\,dy")

    st.markdown("### Two valid ways to obtain the strip second moment")
    st.markdown('<div class="info-box"><b>Recommended study sequence:</b> Start with the parallel-axis theorem because it connects directly to the lecture notes and the second moment of a rectangle. After you understand that method, use direct integration as a verification step. Both methods produce exactly the same result.</div>', unsafe_allow_html=True)
    pat_left, pat_right = st.columns(2, gap="large")
    with pat_left:
        st.markdown("#### Vertical strip: parallel-axis theorem")
        st.write("Let the strip height be h = yT − yB and its thickness be dx. The local half-height is h/2, while the global strip-centroid coordinate from the x-axis is ỹ = yB + h/2 = (yT + yB)/2.")
        st.latex(r"dA=h\,dx")
        st.latex(r"dI_{x,c}=\frac{1}{12}(dx)h^3")
        st.latex(r"dI_x=dI_{x,c}+dA\,\tilde{y}^{\,2}")
        st.latex(r"dI_x=\frac{1}{12}h^3dx+h\,dx\left(\frac{y_T+y_B}{2}\right)^2")
        st.latex(r"dI_x=\frac{y_T^3-y_B^3}{3}dx")
        st.write("If yB = 0 and yT = h, this becomes dIx = h³dx/3, exactly as in the lecture notes.")
    with pat_right:
        st.markdown("#### Horizontal strip: parallel-axis theorem")
        st.write("Let the strip width be w = xR − xL and its thickness be dy. The local half-width is w/2, while the global strip-centroid coordinate from the y-axis is x̃ = xL + w/2 = (xR + xL)/2.")
        st.latex(r"dA=w\,dy")
        st.latex(r"dI_{y,c}=\frac{1}{12}(dy)w^3")
        st.latex(r"dI_y=dI_{y,c}+dA\,\tilde{x}^{\,2}")
        st.latex(r"dI_y=\frac{1}{12}w^3dy+w\,dy\left(\frac{x_R+x_L}{2}\right)^2")
        st.latex(r"dI_y=\frac{x_R^3-x_L^3}{3}dy")
        st.write("If xL = 0 and xR = w, this becomes dIy = w³dy/3.")
    st.markdown('<div class="hint-box"><b>Which method should you learn first?</b> Most students find the parallel-axis approach easier at first because the differential strip behaves like a thin rectangle. Once the idea is clear, direct integration helps explain where the formula comes from. You should be comfortable with both approaches because they are mathematically equivalent.</div>', unsafe_allow_html=True)

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
        st.write("First calculate about the shown reference axes. If the required axis is the parallel axis through the complete-area centroid, use the parallel-axis theorem to shift to that centroidal axis.")

    st.markdown(
        '<div class="warning-box"><b>Important:</b> for a deep vertical strip, '
        r'$I_x$ is not generally obtained from only $\tilde{y}^{2}dA$. '
        'The strip has its own local second moment, so integrate through its depth or use '
        r'$dI_x=(y_T^3-y_B^3)dx/3$.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### 4. Fully worked teaching examples")
    st.markdown('<div class="info-box">The same triangular region is solved with a vertical strip and a horizontal strip. Each example is arranged as: geometry, strip definition, centroid equations, then second moments.</div>', unsafe_allow_html=True)

    st.markdown("#### Example A: Vertical-strip solution")
    example_v = solve_vertical(4 - x, 0, 0, 4)
    render_static_diagram(example_v, "Vertical strip for y = 4 - x", True)
    st.markdown("##### A1. Geometry and differential area")
    c1, c2 = st.columns(2)
    with c1:
        st.latex(r"y_T=4-x,\qquad y_B=0,\qquad 0\le x\le4")
        st.latex(r"dA=(y_T-y_B)dx=(4-x)dx")
    with c2:
        st.latex(r"\tilde{x}=x")
        st.latex(r"\tilde{y}=\frac{y_T+y_B}{2}=\frac{4-x}{2}")
    st.markdown("##### A2. Area and centroid")
    st.latex(r"A=\int_0^4(4-x)dx=8")
    st.latex(r"\bar{x}=\frac{\int_0^4x(4-x)dx}{\int_0^4(4-x)dx}=\frac{4}{3}")
    st.latex(r"\bar{y}=\frac{\int_0^4\left(\frac{4-x}{2}\right)(4-x)dx}{\int_0^4(4-x)dx}=\frac{4}{3}")
    st.markdown("##### A3. Second moments")
    st.latex(r"I_y=\int_0^4x^2(4-x)dx=\frac{64}{3}")
    st.latex(r"I_x=\int_0^4\frac{(4-x)^3}{3}dx=\frac{64}{3}")

    st.markdown("---")
    st.markdown("#### Example B: Horizontal-strip solution of the same region")
    example_h = solve_horizontal(4 - y, 0, 0, 4)
    render_static_diagram(example_h, "Horizontal strip for x = 4 - y", True)
    st.markdown("##### B1. Geometry and differential area")
    st.write("Rearrange y = 4 - x as x = 4 - y.")
    c1, c2 = st.columns(2)
    with c1:
        st.latex(r"x_R=4-y,\qquad x_L=0,\qquad 0\le y\le4")
        st.latex(r"dA=(x_R-x_L)dy=(4-y)dy")
    with c2:
        st.latex(r"\tilde{x}=\frac{x_R+x_L}{2}=\frac{4-y}{2}")
        st.latex(r"\tilde{y}=y")
    st.markdown("##### B2. Area and centroid")
    st.latex(r"A=\int_0^4(4-y)dy=8")
    st.latex(r"\bar{x}=\frac{\int_0^4\left(\frac{4-y}{2}\right)(4-y)dy}{\int_0^4(4-y)dy}=\frac{4}{3}")
    st.latex(r"\bar{y}=\frac{\int_0^4y(4-y)dy}{\int_0^4(4-y)dy}=\frac{4}{3}")
    st.markdown("##### B3. Second moments")
    st.latex(r"I_x=\int_0^4y^2(4-y)dy=\frac{64}{3}")
    st.latex(r"I_y=\int_0^4\frac{(4-y)^3}{3}dy=\frac{64}{3}")
    st.success("Cross-check passed: both strip directions give A = 8, x̄ = 4/3, ȳ = 4/3, Ix = 64/3, and Iy = 64/3.")

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
    st.write("The tilde coordinates describe the global coordinates of one moving strip centroid, measured from the shown reference axes.")
    render_strip_centroid_note("vertical" if vertical else "horizontal")
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
        st.write("Start from Ix = ∫A y²dA and Iy = ∫A x²dA. For a vertical strip, dA = dy dx.")
        st.write("Because x is effectively constant across the thin strip:")
        st.latex(r"dI_y=x^2dA=x^2(y_T-y_B)dx")
        st.latex(rf"I_y=\int_{{{sp.latex(data['lower'])}}}^{{{sp.latex(data['upper'])}}}x^2[y_T-y_B]dx={sp.latex(data['i_y'])}\;\mathrm{{units}}^4")
        st.write("For Ix, y changes from yB to yT inside the strip, so perform the inner y-integration first:")
        st.latex(r"dI_x=\int_{y_B}^{y_T}y^2dy\,dx=\frac{y_T^3-y_B^3}{3}dx")
        st.latex(rf"I_x=\int_{{{sp.latex(data['lower'])}}}^{{{sp.latex(data['upper'])}}}\frac{{y_T^3-y_B^3}}{{3}}dx={sp.latex(data['i_x'])}\;\mathrm{{units}}^4")
    else:
        st.write("Start from Ix = ∫A y²dA and Iy = ∫A x²dA. For a horizontal strip, dA = dx dy.")
        st.write("Because y is effectively constant across the thin strip:")
        st.latex(r"dI_x=y^2dA=y^2(x_R-x_L)dy")
        st.latex(rf"I_x=\int_{{{sp.latex(data['lower'])}}}^{{{sp.latex(data['upper'])}}}y^2[x_R-x_L]dy={sp.latex(data['i_x'])}\;\mathrm{{units}}^4")
        st.write("For Iy, x changes from xL to xR inside the strip, so perform the inner x-integration first:")
        st.latex(r"dI_y=\int_{x_L}^{x_R}x^2dx\,dy=\frac{x_R^3-x_L^3}{3}dy")
        st.latex(rf"I_y=\int_{{{sp.latex(data['lower'])}}}^{{{sp.latex(data['upper'])}}}\frac{{x_R^3-x_L^3}}{{3}}dy={sp.latex(data['i_y'])}\;\mathrm{{units}}^4")

    st.markdown("#### Parallel-axis theorem alternative for the differential strip")
    if vertical:
        st.write("This is the lecture-note approach. Treat the differential strip as a thin rectangle of height h = yT - yB and width dx.")
        st.latex(r"dI_{x,c}=\frac{1}{12}(dx)h^3,\qquad dA=h\,dx,\qquad \tilde{y}=\frac{y_T+y_B}{2}")
        st.latex(r"dI_x=dI_{x,c}+dA\tilde{y}^{\,2}")
        st.latex(r"dI_x=\frac{1}{12}h^3dx+h\,dx\left(\frac{y_T+y_B}{2}\right)^2=\frac{y_T^3-y_B^3}{3}dx")
    else:
        st.write("Treat the differential strip as a thin rectangle of width w = xR - xL and height dy.")
        st.latex(r"dI_{y,c}=\frac{1}{12}(dy)w^3,\qquad dA=w\,dy,\qquad \tilde{x}=\frac{x_R+x_L}{2}")
        st.latex(r"dI_y=dI_{y,c}+dA\tilde{x}^{\,2}")
        st.latex(r"dI_y=\frac{1}{12}w^3dy+w\,dy\left(\frac{x_R+x_L}{2}\right)^2=\frac{x_R^3-x_L^3}{3}dy")
    st.info("The direct-integration and parallel-axis approaches give the same differential-strip formula. Use the parallel-axis route when following the lecture notes; use direct integration as verification.")

    st.markdown("### Step 7: Shift to centroidal axes when required")
    st.latex(rf"I_{{\bar{{x}}}}=I_x-A\bar{{y}}^2={sp.latex(data['i_x_bar'])}\;\mathrm{{units}}^4")
    st.latex(rf"I_{{\bar{{y}}}}=I_y-A\bar{{x}}^2={sp.latex(data['i_y_bar'])}\;\mathrm{{units}}^4")

    st.markdown("### Final checks")
    c1, c2, c3 = st.columns(3)
    c1.metric("Area", f"{fmt(data['area'])} units²")
    c2.metric("x̄", f"{fmt(data['x_bar'])} units")
    c3.metric("ȳ", f"{fmt(data['y_bar'])} units")
    st.markdown('<div class="hint-box"><b>Checks:</b> area is positive; the centroid lies inside the region; second moments are positive; centroid units are length; second-moment units are length⁴.</div>', unsafe_allow_html=True)



def render_theory_tab():
    st.markdown("## Learn the theory")
    st.markdown('<div class="info-box"><b>Your learning goal:</b> understand how a region is replaced by many thin strips, then use integration to calculate area, centroid, and second moment of area.</div>', unsafe_allow_html=True)
    st.markdown("### 1. Essential integration")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("#### Reverse power rule")
        st.latex(r"\int x^n\,dx=\frac{x^{n+1}}{n+1}+C,\qquad n\neq -1")
        st.write("Increase the power by one and divide by the new power.")
    with c2:
        st.markdown("#### Definite integral")
        st.latex(r"\int_a^b f(x)\,dx=F(b)-F(a)")
        st.write("Substitute the upper limit first, then subtract the value at the lower limit.")
    st.markdown("### 2. Area and centroid")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.latex(r"A=\int dA")
        st.latex(r"\bar{x}=\frac{\int \tilde{x}\,dA}{\int dA}")
        st.latex(r"\bar{y}=\frac{\int \tilde{y}\,dA}{\int dA}")
    with c2:
        st.write("Use tildes for the centroid of one moving strip.")
        st.write("Use bars for the centroid of the complete area.")
        st.write("The integration limits must match the variable in the differential, dx or dy.")
    st.markdown("### 3. Second moment of area")
    st.latex(r"I_x=\int_A y^2\,dA,\qquad I_y=\int_A x^2\,dA")
    st.write("The distance from the reference axis is squared. Therefore, area farther from an axis contributes much more strongly.")
    st.markdown("### Coordinate convention used throughout this app")
    st.write("Boundary symbols such as yT, yB, xR, and xL are coordinates measured from the shown reference axes. Therefore, midpoint coordinates use the average of the two boundary coordinates.")
    st.latex(r"\tilde{y}=\frac{y_T+y_B}{2},\qquad \tilde{x}=\frac{x_R+x_L}{2}")
    st.write("The half-height (yT − yB)/2 and half-width (xR − xL)/2 are local distances measured from one boundary, not global coordinates unless that boundary is at zero.")

    st.markdown("### 4. Centroidal axes")
    st.latex(r"I_{\bar{x}}=I_x-A\bar{y}^{\,2},\qquad I_{\bar{y}}=I_y-A\bar{x}^{\,2}")
    st.markdown('<div class="hint-box"><b>Recommended order:</b> identify the geometry, define dA, locate the strip centroid, set the limits, calculate area and centroid, then calculate second moments.</div>', unsafe_allow_html=True)


def strip_coordinate_figure(cut):
    """Visualise local half-dimension versus global centroid coordinate."""
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    bbox = dict(boxstyle="round,pad=0.25", fc="white", ec="#cbd5e1", alpha=0.92)
    if cut == "vertical":
        y_bottom, y_top = 2.0, 6.0
        x0, width = 2.0, 0.42
        y_centroid = (y_top + y_bottom) / 2
        ax.add_patch(Rectangle((x0-width/2, y_bottom), width, y_top-y_bottom,
                               facecolor="#FF7A00", edgecolor="#9A3412", lw=2.2, alpha=0.60))
        ax.axhline(0, color="#111827", lw=1.5)
        ax.hlines([y_bottom, y_top], 0.5, 3.5, colors=["#00A651", "#0047AB"], lw=2.5)
        ax.scatter(x0, y_centroid, s=115, color="#7A00CC", edgecolor="white", zorder=5)
        ax.annotate(r"lower boundary $y_B$", (0.55, y_bottom), xytext=(0, -25),
                    textcoords="offset points", color="#00823B", weight="bold", bbox=bbox)
        ax.annotate(r"upper boundary $y_T$", (0.55, y_top), xytext=(0, 9),
                    textcoords="offset points", color="#0047AB", weight="bold", bbox=bbox)
        ax.annotate(r"strip centroid $\tilde{y}$", (x0, y_centroid), xytext=(45, 0),
                    textcoords="offset points", color="#7A00CC", weight="bold", bbox=bbox,
                    arrowprops=dict(arrowstyle="->", color="#7A00CC"))
        ax.annotate("local half-height\n$(y_T-y_B)/2$", (x0-width/2, y_centroid),
                    xytext=(-105, 0), textcoords="offset points", ha="center",
                    color="#9A3412", weight="bold", bbox=bbox,
                    arrowprops=dict(arrowstyle="<->", color="#FF7A00"))
        ax.annotate("global coordinate\n$y_B+(y_T-y_B)/2$", (x0+width/2, y_centroid),
                    xytext=(102, -55), textcoords="offset points", ha="center",
                    color="#E00034", weight="bold", bbox=bbox,
                    arrowprops=dict(arrowstyle="->", color="#E00034"))
        ax.text(0.15, -0.12, "x-axis reference", color="#111827", weight="bold", bbox=bbox)
        ax.set_xlim(0, 4.2); ax.set_ylim(-0.5, 7.0); ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_title("Vertical strip: local distance versus global y-coordinate", weight="bold")
    else:
        x_left, x_right = 2.0, 6.0
        y0, height = 2.0, 0.42
        x_centroid = (x_right + x_left) / 2
        ax.add_patch(Rectangle((x_left, y0-height/2), x_right-x_left, height,
                               facecolor="#FF7A00", edgecolor="#9A3412", lw=2.2, alpha=0.60))
        ax.axvline(0, color="#111827", lw=1.5)
        ax.vlines([x_left, x_right], 0.5, 3.5, colors=["#00A651", "#0047AB"], lw=2.5)
        ax.scatter(x_centroid, y0, s=115, color="#7A00CC", edgecolor="white", zorder=5)
        ax.annotate(r"left boundary $x_L$", (x_left, 0.55), xytext=(-25, -22),
                    textcoords="offset points", color="#00823B", weight="bold", bbox=bbox)
        ax.annotate(r"right boundary $x_R$", (x_right, 0.55), xytext=(-25, -22),
                    textcoords="offset points", color="#0047AB", weight="bold", bbox=bbox)
        ax.annotate(r"strip centroid $\tilde{x}$", (x_centroid, y0), xytext=(0, 42),
                    textcoords="offset points", ha="center", color="#7A00CC", weight="bold", bbox=bbox,
                    arrowprops=dict(arrowstyle="->", color="#7A00CC"))
        ax.annotate("local half-width\n$(x_R-x_L)/2$", (x_centroid, y0-height/2),
                    xytext=(0, -62), textcoords="offset points", ha="center",
                    color="#9A3412", weight="bold", bbox=bbox,
                    arrowprops=dict(arrowstyle="<->", color="#FF7A00"))
        ax.annotate("global coordinate\n$x_L+(x_R-x_L)/2$", (x_centroid, y0+height/2),
                    xytext=(85, 32), textcoords="offset points", ha="center",
                    color="#E00034", weight="bold", bbox=bbox,
                    arrowprops=dict(arrowstyle="->", color="#E00034"))
        ax.text(-0.15, 0.15, "y-axis reference", rotation=90, color="#111827", weight="bold", bbox=bbox)
        ax.set_xlim(-0.5, 7.0); ax.set_ylim(0, 4.2); ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_title("Horizontal strip: local distance versus global x-coordinate", weight="bold")
    ax.grid(alpha=0.10)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.86, bottom=0.15)
    return fig


def parallel_axis_strip_figure(cut):
    """Show the two terms in the parallel-axis theorem for a differential strip."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 3.9))
    if cut == "vertical":
        yb, yt, xc = 1.5, 5.5, 2.0
        yc = (yb + yt) / 2
        for ax in (ax1, ax2):
            ax.add_patch(Rectangle((xc-0.18, yb), 0.36, yt-yb,
                                   facecolor="#FF7A00", edgecolor="#9A3412", lw=2, alpha=.58))
            ax.scatter(xc, yc, s=90, color="#7A00CC", edgecolor="white", zorder=5)
            ax.set_xlim(0.5, 3.5); ax.set_ylim(0, 6.5); ax.set_aspect("equal"); ax.grid(alpha=.10)
        ax1.axhline(yc, color="#7A00CC", ls="--", lw=1.7)
        ax1.set_title(r"Local term: $dI_{x,c}=\frac{1}{12}(dx)h^3$", fontsize=10, weight="bold")
        ax2.axhline(0, color="#111827", lw=1.5)
        ax2.annotate(r"global distance $\tilde{y}$", (xc, yc), xytext=(38, -8),
                     textcoords="offset points", color="#E00034", weight="bold",
                     arrowprops=dict(arrowstyle="->", color="#E00034"))
        ax2.set_title(r"Shift term: $dA\tilde{y}^{2}$", fontsize=10, weight="bold")
    else:
        xl, xr, yc = 1.5, 5.5, 2.0
        xc = (xl + xr) / 2
        for ax in (ax1, ax2):
            ax.add_patch(Rectangle((xl, yc-0.18), xr-xl, 0.36,
                                   facecolor="#FF7A00", edgecolor="#9A3412", lw=2, alpha=.58))
            ax.scatter(xc, yc, s=90, color="#7A00CC", edgecolor="white", zorder=5)
            ax.set_xlim(0, 6.5); ax.set_ylim(0.5, 3.5); ax.set_aspect("equal"); ax.grid(alpha=.10)
        ax1.axvline(xc, color="#7A00CC", ls="--", lw=1.7)
        ax1.set_title(r"Local term: $dI_{y,c}=\frac{1}{12}(dy)w^3$", fontsize=10, weight="bold")
        ax2.axvline(0, color="#111827", lw=1.5)
        ax2.annotate(r"global distance $\tilde{x}$", (xc, yc), xytext=(0, 38),
                     textcoords="offset points", ha="center", color="#E00034", weight="bold",
                     arrowprops=dict(arrowstyle="->", color="#E00034"))
        ax2.set_title(r"Shift term: $dA\tilde{x}^{2}$", fontsize=10, weight="bold")
    for ax in (ax1, ax2):
        ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.suptitle("Parallel-axis theorem: local strip contribution + axis-shift contribution", weight="bold")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.79, bottom=0.14, wspace=0.28)
    return fig


def render_visual_figure(fig, caption):
    left, centre, right = st.columns([0.12, 1.9, 0.12])
    with centre:
        st.pyplot(fig, use_container_width=True)
        st.caption(caption)


def render_strip_centroid_note(cut):
    """Clarify local half-dimension versus global centroid coordinate."""
    render_visual_figure(
        strip_coordinate_figure(cut),
        "The orange strip is shown away from the reference axis so the local half-dimension and the global centroid coordinate are visibly different.",
    )
    if cut == "vertical":
        st.markdown(
            '<div class="warning-box"><b>Do not confuse two different distances.</b><br>'
            'The distance from the lower boundary to the strip centroid is '
            '<b>(yT − yB)/2</b>. However, the global y-coordinate measured from the x-axis is '
            '<b>yB + (yT − yB)/2 = (yT + yB)/2</b>.</div>',
            unsafe_allow_html=True,
        )
        st.latex(r"\underbrace{\frac{y_T-y_B}{2}}_{\text{local distance from }y_B}\qquad"
                 r"\underbrace{\tilde{y}=y_B+\frac{y_T-y_B}{2}=\frac{y_T+y_B}{2}}_{\text{global coordinate from the x-axis}}")
    else:
        st.markdown(
            '<div class="warning-box"><b>Do not confuse two different distances.</b><br>'
            'The distance from the left boundary to the strip centroid is '
            '<b>(xR − xL)/2</b>. However, the global x-coordinate measured from the y-axis is '
            '<b>xL + (xR − xL)/2 = (xR + xL)/2</b>.</div>',
            unsafe_allow_html=True,
        )
        st.latex(r"\underbrace{\frac{x_R-x_L}{2}}_{\text{local distance from }x_L}\qquad"
                 r"\underbrace{\tilde{x}=x_L+\frac{x_R-x_L}{2}=\frac{x_R+x_L}{2}}_{\text{global coordinate from the y-axis}}")


def render_vertical_walkthrough():
    st.markdown("## Vertical strip walkthrough")
    st.markdown('<div class="info-box"><b>Use a vertical strip</b> when the region is most naturally described by an upper boundary and a lower boundary, both written as functions of x.</div>', unsafe_allow_html=True)
    data = solve_vertical(4 - x**2 / 4, 0, 0, 4)
    render_static_diagram(data, "Vertical strip: upper minus lower", False)
    st.markdown("### Step 1: Identify the strip")
    st.latex(r"\text{thickness}=dx,\qquad \text{height}=y_T(x)-y_B(x)")
    st.latex(r"dA=[y_T(x)-y_B(x)]dx")
    st.markdown("### Step 2: Locate the strip centroid")
    render_strip_centroid_note("vertical")
    st.latex(r"\tilde{x}=x,\qquad \tilde{y}=\frac{y_T+y_B}{2}")
    st.markdown("### Step 3: Set the x-limits")
    st.write("Read the leftmost and rightmost x-coordinates of the complete region.")
    st.latex(r"x=a\quad\text{to}\quad x=b")
    st.markdown("### Step 4: Build the required integrals")
    st.latex(r"A=\int_a^b [y_T-y_B]dx")
    st.latex(r"\bar{x}=\frac{\int_a^b x[y_T-y_B]dx}{\int_a^b[y_T-y_B]dx}")
    st.latex(r"\bar{y}=\frac{\int_a^b \left(\frac{y_T+y_B}{2}\right)[y_T-y_B]dx}{\int_a^b[y_T-y_B]dx}")
    st.latex(r"I_y=\int_a^b x^2[y_T-y_B]dx")
    st.latex(r"I_x=\int_a^b\frac{y_T^3-y_B^3}{3}dx")
    st.warning("For Ix, do not use only tilde-y squared times dA. A vertical strip has a finite height and therefore has its own local second moment.")


def render_horizontal_walkthrough():
    st.markdown("## Horizontal strip walkthrough")
    st.markdown('<div class="info-box"><b>Use a horizontal strip</b> when the region is most naturally described by a right boundary and a left boundary, both written as functions of y.</div>', unsafe_allow_html=True)
    data = solve_horizontal(4 - y**2 / 4, 0, 0, 4)
    render_static_diagram(data, "Horizontal strip: right minus left", False)
    st.markdown("### Step 1: Identify the strip")
    st.latex(r"\text{thickness}=dy,\qquad \text{width}=x_R(y)-x_L(y)")
    st.latex(r"dA=[x_R(y)-x_L(y)]dy")
    st.markdown("### Step 2: Locate the strip centroid")
    render_strip_centroid_note("horizontal")
    st.latex(r"\tilde{x}=\frac{x_R+x_L}{2},\qquad \tilde{y}=y")
    st.markdown("### Step 3: Set the y-limits")
    st.write("Read the lowest and highest y-coordinates of the complete region.")
    st.latex(r"y=c\quad\text{to}\quad y=d")
    st.markdown("### Step 4: Build the required integrals")
    st.latex(r"A=\int_c^d [x_R-x_L]dy")
    st.latex(r"\bar{x}=\frac{\int_c^d \left(\frac{x_R+x_L}{2}\right)[x_R-x_L]dy}{\int_c^d[x_R-x_L]dy}")
    st.latex(r"\bar{y}=\frac{\int_c^d y[x_R-x_L]dy}{\int_c^d[x_R-x_L]dy}")
    st.latex(r"I_x=\int_c^d y^2[x_R-x_L]dy")
    st.latex(r"I_y=\int_c^d\frac{x_R^3-x_L^3}{3}dy")
    st.warning("For Iy, do not use only tilde-x squared times dA. A horizontal strip has a finite width and therefore has its own local second moment.")


def render_formula_tab():
    st.markdown("## Why the formulas work")
    st.markdown('<div class="info-box"><b>Two valid routes:</b> the parallel-axis theorem matches your lecture notes and is usually the easier starting point. Direct integration explains the same result from the definition. Both routes are equivalent.</div>', unsafe_allow_html=True)
    st.markdown("### Vertical strip: obtaining dIx")
    render_strip_centroid_note("vertical")
    render_visual_figure(
        parallel_axis_strip_figure("vertical"),
        "The local rectangle term accounts for the strip about its own centroidal axis; the shift term moves that result to the global x-axis.",
    )
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("#### Parallel-axis theorem")
        st.latex(r"h=y_T-y_B,\qquad dA=h\,dx,\qquad \tilde{y}=\frac{y_T+y_B}{2}")
        st.latex(r"dI_{x,c}=\frac{1}{12}(dx)h^3")
        st.latex(r"dI_x=dI_{x,c}+dA\tilde{y}^{\,2}")
        st.latex(r"dI_x=\frac{1}{12}h^3dx+h\,dx\left(\frac{y_T+y_B}{2}\right)^2")
        st.latex(r"dI_x=\frac{y_T^3-y_B^3}{3}dx")
    with c2:
        st.markdown("#### Direct integration")
        st.latex(r"dA=dy\,dx")
        st.latex(r"dI_x=\int_{y_B}^{y_T}y^2\,dy\,dx")
        st.latex(r"dI_x=\left[\frac{y^3}{3}\right]_{y_B}^{y_T}dx")
        st.latex(r"dI_x=\frac{y_T^3-y_B^3}{3}dx")
    st.markdown("### Horizontal strip: obtaining dIy")
    render_strip_centroid_note("horizontal")
    render_visual_figure(
        parallel_axis_strip_figure("horizontal"),
        "The local rectangle term accounts for the strip about its own centroidal axis; the shift term moves that result to the global y-axis.",
    )
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("#### Parallel-axis theorem")
        st.latex(r"w=x_R-x_L,\qquad dA=w\,dy,\qquad \tilde{x}=\frac{x_R+x_L}{2}")
        st.latex(r"dI_{y,c}=\frac{1}{12}(dy)w^3")
        st.latex(r"dI_y=dI_{y,c}+dA\tilde{x}^{\,2}")
        st.latex(r"dI_y=\frac{x_R^3-x_L^3}{3}dy")
    with c2:
        st.markdown("#### Direct integration")
        st.latex(r"dA=dx\,dy")
        st.latex(r"dI_y=\int_{x_L}^{x_R}x^2\,dx\,dy")
        st.latex(r"dI_y=\left[\frac{x^3}{3}\right]_{x_L}^{x_R}dy")
        st.latex(r"dI_y=\frac{x_R^3-x_L^3}{3}dy")
    st.success("You may use either route. Start with the parallel-axis theorem if that is the method used in your lecture notes, and use direct integration to check your understanding.")


def render_complete_examples():
    st.markdown("## Complete worked examples")
    st.markdown('<div class="info-box">The same triangular region is solved with both cutting directions. Compare the setup carefully and confirm that both methods give the same final values.</div>', unsafe_allow_html=True)
    st.markdown("### Example A: Vertical strip")
    data_v = solve_vertical(4 - x, 0, 0, 4)
    render_static_diagram(data_v, "Vertical strip for y = 4 - x", True)
    st.latex(r"dA=(4-x)dx,\qquad \tilde{x}=x,\qquad \tilde{y}=\frac{4-x}{2}")
    st.latex(r"A=\int_0^4(4-x)dx=8")
    st.latex(r"\bar{x}=\frac{\int_0^4x(4-x)dx}{8}=\frac43")
    st.latex(r"\bar{y}=\frac{\int_0^4\left(\frac{4-x}{2}\right)(4-x)dx}{8}=\frac43")
    st.latex(r"I_x=\int_0^4\frac{(4-x)^3}{3}dx=\frac{64}{3},\qquad I_y=\int_0^4x^2(4-x)dx=\frac{64}{3}")
    st.markdown("---")
    st.markdown("### Example B: Horizontal strip")
    data_h = solve_horizontal(4 - y, 0, 0, 4)
    render_static_diagram(data_h, "Horizontal strip for x = 4 - y", True)
    st.latex(r"dA=(4-y)dy,\qquad \tilde{x}=\frac{4-y}{2},\qquad \tilde{y}=y")
    st.latex(r"A=\int_0^4(4-y)dy=8")
    st.latex(r"\bar{x}=\frac{\int_0^4\left(\frac{4-y}{2}\right)(4-y)dy}{8}=\frac43")
    st.latex(r"\bar{y}=\frac{\int_0^4y(4-y)dy}{8}=\frac43")
    st.latex(r"I_x=\int_0^4y^2(4-y)dy=\frac{64}{3},\qquad I_y=\int_0^4\frac{(4-y)^3}{3}dy=\frac{64}{3}")
    st.success("Both directions give A = 8, x-bar = 4/3, y-bar = 4/3, Ix = 64/3, and Iy = 64/3.")


def render_interactive_tab(problem):
    data = problem["data"]
    st.markdown("## Interactive exploration")
    st.markdown('<div class="info-box"><b>Try this:</b> move the strip across the region. Watch how the strip dimensions and strip-centroid position change. Then reveal the complete-area centroid.</div>', unsafe_allow_html=True)
    left, right = st.columns([0.9, 1.25], gap="large")
    with left:
        st.caption(f"Level {level} | {problem['difficulty']}")
        st.markdown(f"### {problem['title']}")
        st.write(problem["purpose"])
        if data["cut"] == "vertical":
            st.latex(rf"y_T(x)={sp.latex(data['top'])},\qquad y_B(x)={sp.latex(data['bottom'])}")
            st.latex(rf"dA=\left({sp.latex(data['dA'])}\right)dx")
            st.latex(rf"\tilde{{x}}={sp.latex(data['x_tilde'])},\qquad \tilde{{y}}={sp.latex(data['y_tilde'])}")
        else:
            st.latex(rf"x_R(y)={sp.latex(data['right'])},\qquad x_L(y)={sp.latex(data['left'])}")
            st.latex(rf"dA=\left({sp.latex(data['dA'])}\right)dy")
            st.latex(rf"\tilde{{x}}={sp.latex(data['x_tilde'])},\qquad \tilde{{y}}={sp.latex(data['y_tilde'])}")
        st.markdown("#### Final calculated properties")
        st.latex(rf"A={sp.latex(data['area'])}")
        st.latex(rf"\bar{{x}}={sp.latex(data['x_bar'])},\qquad \bar{{y}}={sp.latex(data['y_bar'])}")
        st.latex(rf"I_x={sp.latex(data['i_x'])},\qquad I_y={sp.latex(data['i_y'])}")
    with right:
        render_interactive_diagram(problem, key_prefix=f"interactive_{problem['uid']}")

render_header()

st.sidebar.markdown("## Teaching example")
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
if st.sidebar.button("🎲 Generate another example", use_container_width=True):
    st.session_state.current_seed = SYSTEM_RNG.randrange(1, 2**63)

problem = generate_problem(level, st.session_state.current_seed)
data = problem["data"]
st.sidebar.caption(f"Current example: {problem['title']}")
st.sidebar.markdown("---")
st.sidebar.markdown("### Colour key")
with st.sidebar:
    render_colour_key()

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
    "1. Learn the theory",
    "2. Vertical strip walkthrough",
    "3. Horizontal strip walkthrough",
    "4. Why the formulas work",
    "5. Complete worked examples",
    "6. Interactive exploration",
])

with tabs[0]:
    render_theory_tab()
with tabs[1]:
    render_vertical_walkthrough()
with tabs[2]:
    render_horizontal_walkthrough()
with tabs[3]:
    render_formula_tab()
with tabs[4]:
    render_complete_examples()
with tabs[5]:
    render_interactive_tab(problem)
