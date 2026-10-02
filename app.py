
import base64
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon
import numpy as np
import streamlit as st
import sympy as sp


st.set_page_config(
    page_title="MathQuest",
    page_icon="➗",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
LOGO_PATH = APP_DIR / "assets" / "jcu_logo.png"
LOGO_B64 = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii") if LOGO_PATH.exists() else ""

SEED_RNG = random.SystemRandom()
RNG = SEED_RNG
x = sp.symbols("x", real=True)
t = sp.symbols("t", real=True)

st.markdown(
    """
    <style>
      /*
       * Keep MathQuest readable regardless of the Streamlit/browser theme.
       * Previously only the backgrounds were forced to light colours. A user
       * with Streamlit's dark theme therefore received white text on a white
       * page. These variables and selectors make the whole app consistently
       * high contrast.
       */
      :root {
        color-scheme: light;
        --mq-text: #172033;
        --mq-muted: #526174;
        --mq-border: #cbd5e1;
        --mq-sidebar: #eef1f5;
        --mq-white: #ffffff;
        --mq-accent: #c8102e;
      }

      html, body, [data-testid="stAppViewContainer"], .stApp {
        background:#ffffff !important;
        color:var(--mq-text) !important;
      }

      /* Native Streamlit text, Markdown and KaTeX inherit the same dark text. */
      .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5,
      .stApp h6, .stApp p, .stApp li, .stApp label,
      .stApp [data-testid="stMarkdownContainer"],
      .stApp [data-testid="stCaptionContainer"],
      .stApp [data-testid="stText"],
      .stApp .katex, .stApp .katex-display {
        color:var(--mq-text) !important;
      }

      .stApp [data-testid="stCaptionContainer"],
      .stApp small {
        color:var(--mq-muted) !important;
      }

      .block-container {
        max-width:1450px;
        padding-top:1.1rem;
        padding-bottom:2.2rem;
      }

      [data-testid="stSidebar"] {
        background:var(--mq-sidebar) !important;
        border-right:1px solid #dde2e8;
      }

      [data-testid="stSidebar"] > div {
        background:var(--mq-sidebar) !important;
      }

      [data-testid="stSidebar"] h1,
      [data-testid="stSidebar"] h2,
      [data-testid="stSidebar"] h3,
      [data-testid="stSidebar"] p,
      [data-testid="stSidebar"] label,
      [data-testid="stSidebar"] span,
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color:var(--mq-text) !important;
      }

      /* Radio controls: visible labels and borders in either system theme. */
      [data-testid="stRadio"] label,
      [data-testid="stRadio"] label p {
        color:var(--mq-text) !important;
      }

      [data-testid="stRadio"] [role="radiogroup"] label > div:first-child {
        border-color:#475569 !important;
      }

      /* Select boxes and number inputs. */
      [data-baseweb="select"] > div,
      [data-testid="stNumberInput"] input {
        background:var(--mq-white) !important;
        color:var(--mq-text) !important;
        border-color:var(--mq-border) !important;
      }

      [data-baseweb="select"] *,
      [data-testid="stNumberInput"] input,
      [data-testid="stNumberInput"] button,
      [data-testid="stNumberInput"] button * {
        color:var(--mq-text) !important;
      }

      [data-baseweb="popover"],
      [data-baseweb="menu"],
      [role="listbox"],
      [role="option"] {
        background:var(--mq-white) !important;
        color:var(--mq-text) !important;
      }

      /* Tabs need explicit colours because Streamlit theme values vary by user. */
      [data-baseweb="tab-list"] {
        border-bottom-color:var(--mq-border) !important;
      }

      [data-baseweb="tab"] p {
        color:#475569 !important;
      }

      [data-baseweb="tab"][aria-selected="true"] p {
        color:var(--mq-accent) !important;
        font-weight:700;
      }

      [data-baseweb="tab-highlight"] {
        background-color:var(--mq-accent) !important;
      }

      /* Stable high-contrast button styling. */
      .stButton > button {
        background:#ffffff !important;
        color:#172033 !important;
        border:1px solid #64748b !important;
      }

      .stButton > button p,
      .stButton > button span {
        color:inherit !important;
      }

      .stButton > button:hover {
        background:#f1f5f9 !important;
        border-color:#334155 !important;
      }

      .stButton > button[kind="primary"] {
        background:#b91c1c !important;
        color:#ffffff !important;
        border-color:#991b1b !important;
      }

      .stButton > button[kind="primary"]:hover {
        background:#991b1b !important;
      }

      /* Streamlit success/warning/error/info components. */
      [data-testid="stAlert"] {
        color:var(--mq-text) !important;
      }

      [data-testid="stAlert"] p,
      [data-testid="stAlert"] span {
        color:inherit !important;
      }

      [data-testid="stSidebar"] .block-container {
        padding-top:3rem;
      }

      header[data-testid="stHeader"] { background:transparent; }
      #MainMenu, footer { visibility:hidden; }

      .app-header {
        display:grid;
        grid-template-columns:230px 1fr 190px;
        gap:24px;
        align-items:center;
        padding:16px 10px 20px;
      }

      .app-header-logo {
        display:flex;
        justify-content:center;
        align-items:center;
      }

      .app-header-logo img {
        width:210px;
        max-height:92px;
        object-fit:contain;
      }

      .app-header-title h1 {
        margin:0;
        font-size:34px;
        line-height:1.05;
        color:#0f172a;
      }

      .app-header-title p {
        margin:7px 0 0;
        color:#334155;
        line-height:1.45;
        font-size:15px;
      }

      .app-header-author {
        text-align:right;
        color:#334155;
        font-size:13px;
        line-height:1.5;
      }

      .app-divider {
        height:1px;
        background:#d9dee5;
        margin:2px 0 18px;
      }

      .info {
        background:#e8f2ff;
        border-radius:10px;
        padding:13px 15px;
        color:#0f4c81;
        margin:8px 0 16px;
      }

      .hint {
        background:#ecfdf3;
        border:1px solid #bbf7d0;
        border-radius:10px;
        padding:11px 13px;
        color:#166534;
        margin:8px 0 14px;
      }

      .warning {
        background:#fff8e8;
        border:1px solid #f3d69b;
        border-radius:10px;
        padding:11px 13px;
        color:#7c4a03;
        margin:8px 0 14px;
      }

      .good {
        background:#ecfdf3;
        border:1px solid #bbf7d0;
        color:#166534;
        border-radius:9px;
        padding:10px 12px;
        margin:6px 0;
      }

      .bad {
        background:#fff1f2;
        border:1px solid #fecdd3;
        color:#9f1239;
        border-radius:9px;
        padding:10px 12px;
        margin:6px 0;
      }

      .card {
        border:1px solid #dfe4ea;
        border-radius:10px;
        padding:14px 16px;
        margin-bottom:14px;
        background:#fff;
      }

      .skill {
        color:#7c8798;
        font-size:13px;
        margin-top:10px;
      }

      @media (max-width:980px) {
        .app-header {
          grid-template-columns:1fr;
          text-align:center;
        }

        .app-header-author { text-align:center; }
        .app-header-logo img { width:190px; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


TOPICS = {
    "Linear equations": [1, 2, 3, 4],
    "Trigonometry": [1, 2, 3],
    "Vector components": [1, 2, 3, 4],
    "Differentiation": [1, 2, 3, 4, 5],
    "Integration": [1, 2, 3, 4, 5, 6],
    "Centroid and second moment of area": [1, 2, 3, 4, 5, 6],
}


def fmt(value, decimals=4):
    if isinstance(value, sp.Basic):
        value = float(value)
    if abs(float(value) - round(float(value))) < 1e-10:
        return str(int(round(float(value))))
    return f"{float(value):.{decimals}f}".rstrip("0").rstrip(".")


def render_header():
    st.markdown(
        f"""
        <div class="app-header">
          <div class="app-header-logo">
            {f'<img src="data:image/png;base64,{LOGO_B64}" alt="James Cook University">' if LOGO_B64 else '<b style="font-size:24px;color:#0f4c81">MathQuest</b>'}
          </div>
          <div class="app-header-title">
            <h1>MathQuest</h1>
            <p>
              Foundation mathematics practice for EG1011 Statics and Dynamics:
              equations, trigonometry, vectors, differentiation and integration.
            </p>
          </div>
          <div class="app-header-author">
            Designed for EG1011<br>
            <b>Dr. Mehdi Khatamifar</b><br>
            James Cook University
          </div>
        </div>
        <div class="app-divider"></div>
        """,
        unsafe_allow_html=True,
    )


def make_answer(label, value, unit="", tolerance=0.015):
    return {
        "label": label,
        "value": float(value),
        "unit": unit,
        "tolerance": max(tolerance, 0.01 * max(1.0, abs(float(value)))),
    }


def close_enough(user_value, answer):
    return abs(float(user_value) - answer["value"]) <= answer["tolerance"]


def unique_problem(generator, topic, level):
    seen = st.session_state.setdefault("seen_problem_signatures", set())

    for _ in range(500):
        problem = generator(level)
        signature = f"{topic}|{level}|{problem['signature']}"
        if signature not in seen:
            seen.add(signature)
            problem["uid"] = f"{topic}_{level}_{len(seen)}"
            return problem

    # Very unlikely fallback.
    problem = generator(level)
    problem["uid"] = f"{topic}_{level}_{RNG.randrange(10**9)}"
    return problem


def generate_linear(level):
    if level == 1:
        a = RNG.choice([i for i in range(-9, 10) if i not in (0, 1, -1)])
        x_sol = RNG.randint(-12, 12)
        b = RNG.randint(-15, 15)
        c = a * x_sol + b

        equation = sp.Eq(a * x + b, c)

        return {
            "title": "One-step linear equation",
            "difficulty": "Foundation",
            "prompt": "Solve the following equation for x.",
            "display_equation": sp.latex(equation),
            "formula_question": {
                "question": "What should be done first?",
                "options": [
                    f"Subtract {b} from both sides",
                    f"Divide both sides by {a} immediately",
                    "Square both sides",
                    "Differentiate both sides",
                ],
                "correct": f"Subtract {b} from both sides",
                "explanation": "Undo the added constant first, then divide by the coefficient of x.",
            },
            "answers": [make_answer("x", x_sol)],
            "solution": [
                ("text", "Start with the original equation."),
                ("latex", sp.latex(equation)),
                ("text", f"Subtract {b} from both sides so that the x-term is isolated."),
                ("latex", sp.latex(sp.Eq(a * x, c - b))),
                ("text", f"Divide both sides by {a}."),
                ("latex", sp.latex(sp.Eq(x, sp.Rational(c - b, a)))),
                ("text", f"Therefore x = {x_sol}."),
            ],
            "signature": f"L1:{a}:{b}:{c}",
            "diagram": None,
            "teaching_point": "Whatever operation is performed on one side of an equation must also be performed on the other side.",
        }

    if level == 2:
        a = RNG.choice([2, 3, 4, 5, 6, -2, -3, -4])
        shift = RNG.randint(-8, 8)
        x_sol = RNG.randint(-10, 10)
        c = a * (x_sol + shift)

        equation = sp.Eq(a * (x + shift), c)

        return {
            "title": "Equation with brackets",
            "difficulty": "Easy",
            "prompt": "Solve the following equation for x.",
            "display_equation": sp.latex(equation),
            "formula_question": {
                "question": "Which method is cleanest here?",
                "options": [
                    f"Divide both sides by {a}, then remove the shift",
                    "Differentiate the brackets",
                    "Take the square root of both sides",
                    "Convert the equation into a triangle",
                ],
                "correct": f"Divide both sides by {a}, then remove the shift",
                "explanation": "The entire bracket is multiplied by a, so undo that multiplication first.",
            },
            "answers": [make_answer("x", x_sol)],
            "solution": [
                ("text", "The whole bracket is multiplied by the outside coefficient."),
                ("latex", sp.latex(equation)),
                ("text", f"Divide both sides by {a}."),
                ("latex", sp.latex(sp.Eq(x + shift, sp.Rational(c, a)))),
                ("text", f"Now subtract {shift} from both sides."),
                ("latex", sp.latex(sp.Eq(x, sp.Rational(c, a) - shift))),
                ("text", f"Therefore x = {x_sol}."),
            ],
            "signature": f"L2:{a}:{shift}:{c}",
            "diagram": None,
            "teaching_point": "Treat the bracket as one object until the outside multiplication has been removed.",
        }

    if level == 3:
        a = RNG.choice([i for i in range(-8, 9) if i not in (0,)])
        ccoef = RNG.choice([i for i in range(-8, 9) if i not in (0, a)])
        x_sol = RNG.randint(-10, 10)
        b = RNG.randint(-12, 12)
        d = a * x_sol + b - ccoef * x_sol

        equation = sp.Eq(a * x + b, ccoef * x + d)

        return {
            "title": "Variables on both sides",
            "difficulty": "Moderate",
            "prompt": "Solve the following equation for x.",
            "display_equation": sp.latex(equation),
            "formula_question": {
                "question": "What is the key first step?",
                "options": [
                    "Bring all x-terms to one side and constants to the other",
                    "Multiply both sides by x",
                    "Use sine and cosine",
                    "Integrate both sides",
                ],
                "correct": "Bring all x-terms to one side and constants to the other",
                "explanation": "Collect like terms before dividing by the final coefficient of x.",
            },
            "answers": [make_answer("x", x_sol)],
            "solution": [
                ("text", "Start by collecting all x-terms on the left."),
                ("latex", sp.latex(equation)),
                ("latex", sp.latex(sp.Eq((a - ccoef) * x + b, d))),
                ("text", f"Move the constant {b} to the right side."),
                ("latex", sp.latex(sp.Eq((a - ccoef) * x, d - b))),
                ("text", f"Divide by the coefficient {a - ccoef}."),
                ("latex", sp.latex(sp.Eq(x, sp.Rational(d - b, a - ccoef)))),
                ("text", f"Therefore x = {x_sol}."),
            ],
            "signature": f"L3:{a}:{b}:{ccoef}:{d}",
            "diagram": None,
            "teaching_point": "Collect like terms before dividing. Do not divide until only one x-term remains.",
        }

    # Level 4: simultaneous equations
    while True:
        a, b, c, d = [RNG.choice([i for i in range(-6, 7) if i != 0]) for _ in range(4)]
        determinant = a * d - b * c
        if determinant != 0:
            break

    x_sol = RNG.randint(-6, 6)
    y_sol = RNG.randint(-6, 6)
    e = a * x_sol + b * y_sol
    f = c * x_sol + d * y_sol

    return {
        "title": "Two simultaneous linear equations",
        "difficulty": "Challenging",
        "prompt": "Solve the two equations for x and y.",
        "display_equation": (
            r"\begin{cases}"
            + sp.latex(a * x + b * sp.Symbol("y"))
            + "="
            + sp.latex(e)
            + r"\\"
            + sp.latex(c * x + d * sp.Symbol("y"))
            + "="
            + sp.latex(f)
            + r"\end{cases}"
        ),
        "formula_question": {
            "question": "Which method is appropriate?",
            "options": [
                "Elimination or substitution",
                "Differentiate both equations",
                "Use the Pythagorean theorem",
                "Take inverse sine",
            ],
            "correct": "Elimination or substitution",
            "explanation": "Two unknowns require two independent equations. Elimination removes one unknown first.",
        },
        "answers": [
            make_answer("x", x_sol),
            make_answer("y", y_sol),
        ],
        "solution": [
            ("text", "Write the two equations clearly."),
            (
                "latex",
                r"\begin{aligned}"
                + sp.latex(a * x + b * sp.Symbol("y"))
                + "&="
                + sp.latex(e)
                + r"\\"
                + sp.latex(c * x + d * sp.Symbol("y"))
                + "&="
                + sp.latex(f)
                + r"\end{aligned}",
            ),
            ("text", f"Multiply the first equation by {c} and the second equation by {a}, so the x-terms become equal."),
            (
                "latex",
                r"\begin{aligned}"
                + sp.latex(a * c * x + b * c * sp.Symbol("y"))
                + "&="
                + sp.latex(c * e)
                + r"\\"
                + sp.latex(a * c * x + a * d * sp.Symbol("y"))
                + "&="
                + sp.latex(a * f)
                + r"\end{aligned}",
            ),
            ("text", "Subtract the second new equation from the first."),
            (
                "latex",
                sp.latex((b * c - a * d) * sp.Symbol("y"))
                + "="
                + sp.latex(c * e - a * f),
            ),
            (
                "latex",
                sp.latex(
                    sp.Eq(
                        sp.Symbol("y"),
                        sp.Rational(c * e - a * f, b * c - a * d),
                    )
                ),
            ),
            ("text", f"Therefore y = {y_sol}. Substitute this into the first original equation."),
            (
                "latex",
                sp.latex(sp.Eq(a * x + b * y_sol, e)),
            ),
            (
                "latex",
                sp.latex(sp.Eq(x, sp.Rational(e - b * y_sol, a))),
            ),
            ("text", f"Therefore x = {x_sol} and y = {y_sol}."),
        ],
        "signature": f"L4:{a}:{b}:{c}:{d}:{e}:{f}",
        "diagram": None,
        "teaching_point": "Two independent equations are needed to solve two unknowns. Eliminate one variable first.",
    }


def trig_triangle_figure(data):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    A = np.array([0.8, 0.6])
    B = np.array([5.2, 0.6])
    C = np.array([5.2, 3.4])

    ax.add_patch(
        Polygon([A, B, C], closed=True, fill=False, linewidth=2.2)
    )
    ax.plot([B[0] - 0.3, B[0] - 0.3, B[0]], [B[1], B[1] + 0.3, B[1] + 0.3], linewidth=1.2)

    ax.text(3.0, 0.25, data.get("adj_label", "Adjacent"), ha="center")
    ax.text(5.55, 2.0, data.get("opp_label", "Opposite"), va="center", rotation=90)
    ax.text(2.9, 2.2, data.get("hyp_label", "Hypotenuse"), rotation=32, ha="center")
    ax.text(1.15, 0.78, rf"$\theta={data['theta']}^\circ$")

    ax.set_xlim(0.2, 6.2)
    ax.set_ylim(0.0, 4.0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


def generate_trigonometry(level):
    angle = RNG.choice([20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70])
    rad = math.radians(angle)

    if level == 1:
        hyp = RNG.choice([8, 10, 12, 15, 18, 20, 24])
        opp = hyp * math.sin(rad)
        adj = hyp * math.cos(rad)

        return {
            "title": "Find sides using sine and cosine",
            "difficulty": "Foundation",
            "prompt": "A right triangle has the angle and hypotenuse shown. Find the opposite and adjacent sides.",
            "display_equation": None,
            "formula_question": {
                "question": "Which formulas should be used?",
                "options": [
                    "Opposite = H sin θ and Adjacent = H cos θ",
                    "Opposite = H cos θ and Adjacent = H sin θ",
                    "Opposite = H tan θ and Adjacent = H/tan θ",
                    "Use differentiation",
                ],
                "correct": "Opposite = H sin θ and Adjacent = H cos θ",
                "explanation": "SOH gives sin θ = O/H and CAH gives cos θ = A/H.",
            },
            "answers": [
                make_answer("Opposite side", opp, "units", 0.03),
                make_answer("Adjacent side", adj, "units", 0.03),
            ],
            "solution": [
                ("text", "Use SOH and CAH."),
                ("latex", r"\sin\theta=\frac{\text{opposite}}{\text{hypotenuse}}"),
                ("latex", r"\cos\theta=\frac{\text{adjacent}}{\text{hypotenuse}}"),
                ("text", "Rearrange each equation."),
                ("latex", rf"\text{{opposite}}=({hyp})\sin({angle}^\circ)={opp:.4f}"),
                ("latex", rf"\text{{adjacent}}=({hyp})\cos({angle}^\circ)={adj:.4f}"),
                ("text", "Round only at the end."),
            ],
            "signature": f"T1:{angle}:{hyp}",
            "diagram": ("trig", {
                "theta": angle,
                "hyp_label": f"H = {hyp}",
                "opp_label": "Opposite = ?",
                "adj_label": "Adjacent = ?",
            }),
            "teaching_point": "Identify opposite, adjacent and hypotenuse relative to the stated angle before choosing a formula.",
        }

    if level == 2:
        adjacent = RNG.choice([5, 7, 9, 12, 14, 16, 20])
        opposite = adjacent * math.tan(rad)
        hyp = adjacent / math.cos(rad)

        return {
            "title": "Find an unknown side using tangent",
            "difficulty": "Easy",
            "prompt": "A right triangle has the angle and adjacent side shown. Find the opposite side and the hypotenuse.",
            "display_equation": None,
            "formula_question": {
                "question": "Which ratio directly connects opposite and adjacent?",
                "options": [
                    "tan θ = opposite/adjacent",
                    "sin θ = adjacent/hypotenuse",
                    "cos θ = opposite/hypotenuse",
                    "tan θ = hypotenuse/opposite",
                ],
                "correct": "tan θ = opposite/adjacent",
                "explanation": "TOA gives tangent as opposite divided by adjacent.",
            },
            "answers": [
                make_answer("Opposite side", opposite, "units", 0.03),
                make_answer("Hypotenuse", hyp, "units", 0.03),
            ],
            "solution": [
                ("latex", r"\tan\theta=\frac{\text{opposite}}{\text{adjacent}}"),
                ("latex", rf"\text{{opposite}}=({adjacent})\tan({angle}^\circ)={opposite:.4f}"),
                ("text", "Now use cosine to find the hypotenuse."),
                ("latex", r"\cos\theta=\frac{\text{adjacent}}{\text{hypotenuse}}"),
                ("latex", rf"\text{{hypotenuse}}=\frac{{{adjacent}}}{{\cos({angle}^\circ)}}={hyp:.4f}"),
            ],
            "signature": f"T2:{angle}:{adjacent}",
            "diagram": ("trig", {
                "theta": angle,
                "hyp_label": "H = ?",
                "opp_label": "Opposite = ?",
                "adj_label": f"Adjacent = {adjacent}",
            }),
            "teaching_point": "Use the ratio containing the known side and the required side. Avoid introducing unnecessary unknowns.",
        }

    # Level 3: inverse trig
    adjacent = RNG.choice([4, 5, 6, 8, 10, 12, 15])
    opposite = RNG.choice([3, 4, 5, 7, 9, 11, 13])
    theta = math.degrees(math.atan2(opposite, adjacent))
    hyp = math.hypot(opposite, adjacent)

    return {
        "title": "Find an angle using inverse tangent",
        "difficulty": "Moderate",
        "prompt": "The opposite and adjacent sides are given. Find the angle θ and the hypotenuse.",
        "display_equation": None,
        "formula_question": {
            "question": "How should the angle be found?",
            "options": [
                "θ = tan⁻¹(opposite/adjacent)",
                "θ = tan(opposite/adjacent)",
                "θ = opposite × adjacent",
                "θ = sin⁻¹(adjacent/opposite)",
            ],
            "correct": "θ = tan⁻¹(opposite/adjacent)",
            "explanation": "The inverse tangent converts a side ratio into an angle.",
        },
        "answers": [
            make_answer("Angle θ", theta, "degrees", 0.05),
            make_answer("Hypotenuse", hyp, "units", 0.03),
        ],
        "solution": [
            ("latex", r"\tan\theta=\frac{\text{opposite}}{\text{adjacent}}"),
            ("latex", rf"\tan\theta=\frac{{{opposite}}}{{{adjacent}}}"),
            ("latex", rf"\theta=\tan^{{-1}}\left(\frac{{{opposite}}}{{{adjacent}}}\right)={theta:.4f}^\circ"),
            ("text", "Use Pythagoras for the hypotenuse."),
            ("latex", r"H^2=O^2+A^2"),
            ("latex", rf"H=\sqrt{{{opposite}^2+{adjacent}^2}}={hyp:.4f}"),
            ("text", "Make sure the calculator is in degree mode."),
        ],
        "signature": f"T3:{opposite}:{adjacent}",
        "diagram": ("trig", {
            "theta": round(theta, 1),
            "hyp_label": "H = ?",
            "opp_label": f"Opposite = {opposite}",
            "adj_label": f"Adjacent = {adjacent}",
        }),
        "teaching_point": "Use an inverse trigonometric function when the angle is unknown.",
    }


def vector_figure(data):
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    origin = np.array([0.8, 0.8])
    theta = math.radians(data["angle_from_x"])
    length = 3.5
    end = origin + length * np.array([math.cos(theta), math.sin(theta)])

    arrow = FancyArrowPatch(
        origin,
        end,
        arrowstyle="->",
        mutation_scale=18,
        linewidth=2.2,
    )
    ax.add_patch(arrow)

    ax.plot([origin[0], end[0]], [origin[1], origin[1]], linestyle="--", linewidth=1.2)
    ax.plot([end[0], end[0]], [origin[1], end[1]], linestyle="--", linewidth=1.2)

    ax.text((origin[0] + end[0]) / 2, origin[1] - 0.28, data["x_label"], ha="center")
    ax.text(end[0] + 0.18, (origin[1] + end[1]) / 2, data["y_label"], va="center")
    ax.text((origin[0] + end[0]) / 2, (origin[1] + end[1]) / 2 + 0.25, data["vector_label"], ha="center")
    ax.text(origin[0] + 0.55, origin[1] + 0.18, data["angle_label"])

    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)
    ax.set_xlim(-0.5, 5.3)
    ax.set_ylim(-0.3, 4.7)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


def direction_triangle_figure(data):
    """
    Draw a force vector with:
    - dashed rectangular x/y component construction;
    - a small rise-run-hypotenuse triangle placed on the force vector;
    - clear labels that match the calculation F_x=F(run/L), F_y=F(rise/L).
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.5))

    run = float(data["run"])
    rise = float(data["rise"])
    magnitude = float(data["magnitude"])

    direction_length = math.hypot(run, rise)
    ux = run / direction_length
    uy = rise / direction_length

    origin = np.array([0.75, 0.65])
    force_draw_length = 4.35
    end = origin + force_draw_length * np.array([ux, uy])
    x_corner = np.array([end[0], origin[1]])

    force_colour = "#ef4444"
    triangle_colour = "#2563eb"
    angle_colour = "#16a34a"

    # Main force vector.
    force_arrow = FancyArrowPatch(
        origin,
        end,
        arrowstyle="->",
        mutation_scale=18,
        linewidth=2.8,
        color=force_colour,
    )
    ax.add_patch(force_arrow)

    # Dashed component construction.
    x_arrow = FancyArrowPatch(
        origin,
        x_corner,
        arrowstyle="->",
        mutation_scale=15,
        linewidth=1.9,
        linestyle="--",
        color=force_colour,
    )
    y_arrow = FancyArrowPatch(
        x_corner,
        end,
        arrowstyle="->",
        mutation_scale=15,
        linewidth=1.9,
        linestyle="--",
        color=force_colour,
    )
    ax.add_patch(x_arrow)
    ax.add_patch(y_arrow)

    # Main vector and component labels.
    ax.text(
        end[0] + 0.08,
        end[1] + 0.10,
        rf"$F={fmt(magnitude)}\ \mathrm{{N}}$",
        color=force_colour,
        fontsize=11,
        weight="bold",
    )
    ax.text(
        (origin[0] + x_corner[0]) / 2,
        origin[1] - 0.38,
        r"$F_x$",
        color=force_colour,
        fontsize=11,
        ha="center",
    )
    ax.text(
        x_corner[0] + 0.18,
        (origin[1] + end[1]) / 2,
        r"$F_y$",
        color=force_colour,
        fontsize=11,
        va="center",
    )

    # Small direction triangle placed on the vector.
    scale = 1.15 / direction_length
    small_run = run * scale
    small_rise = rise * scale

    tri_start = origin + 0.50 * (end - origin)
    tri_horizontal_end = tri_start + np.array([small_run, 0.0])
    tri_end = tri_horizontal_end + np.array([0.0, small_rise])

    ax.plot(
        [tri_start[0], tri_horizontal_end[0]],
        [tri_start[1], tri_horizontal_end[1]],
        color=triangle_colour,
        linewidth=2.0,
    )
    ax.plot(
        [tri_horizontal_end[0], tri_end[0]],
        [tri_horizontal_end[1], tri_end[1]],
        color=triangle_colour,
        linewidth=2.0,
    )
    ax.plot(
        [tri_start[0], tri_end[0]],
        [tri_start[1], tri_end[1]],
        color=triangle_colour,
        linewidth=2.0,
    )

    ax.text(
        (tri_start[0] + tri_horizontal_end[0]) / 2,
        tri_start[1] - 0.22,
        fmt(run),
        color=triangle_colour,
        fontsize=10,
        ha="center",
        weight="bold",
    )
    ax.text(
        tri_horizontal_end[0] + 0.12,
        (tri_horizontal_end[1] + tri_end[1]) / 2,
        fmt(rise),
        color=triangle_colour,
        fontsize=10,
        va="center",
        weight="bold",
    )
    ax.text(
        (tri_start[0] + tri_end[0]) / 2 - 0.02,
        (tri_start[1] + tri_end[1]) / 2 + 0.17,
        rf"$L=\sqrt{{{fmt(run)}^2+{fmt(rise)}^2}}$",
        color=triangle_colour,
        fontsize=9,
        ha="center",
        rotation=math.degrees(math.atan2(rise, run)),
    )

    # Angle arc at the origin.
    theta = math.atan2(rise, run)
    arc_angles = np.linspace(0.0, theta, 60)
    arc_radius = 0.58
    ax.plot(
        origin[0] + arc_radius * np.cos(arc_angles),
        origin[1] + arc_radius * np.sin(arc_angles),
        color=angle_colour,
        linewidth=1.8,
    )
    ax.text(
        origin[0] + 0.68 * math.cos(theta / 2),
        origin[1] + 0.68 * math.sin(theta / 2),
        r"$\theta$",
        color=angle_colour,
        fontsize=10,
        weight="bold",
    )

    # Light axes for orientation.
    ax.plot(
        [origin[0] - 0.2, end[0] + 0.75],
        [origin[1], origin[1]],
        color="#cbd5e1",
        linewidth=0.9,
    )
    ax.plot(
        [origin[0], origin[0]],
        [origin[1] - 0.2, end[1] + 0.65],
        color="#cbd5e1",
        linewidth=0.9,
    )

    ax.set_xlim(0.1, max(6.1, end[0] + 1.1))
    ax.set_ylim(0.05, max(4.5, end[1] + 0.9))
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


def generate_vector(level):
    magnitude = RNG.choice([40, 50, 60, 75, 80, 90, 100, 120, 150])
    angle = RNG.choice([20, 25, 30, 35, 40, 45, 50, 55, 60])

    if level == 1:
        fx = magnitude * math.cos(math.radians(angle))
        fy = magnitude * math.sin(math.radians(angle))

        return {
            "title": "Vector components from an angle above +x",
            "difficulty": "Foundation",
            "prompt": "Resolve the vector into x- and y-components.",
            "display_equation": None,
            "formula_question": {
                "question": "Which component equations are correct?",
                "options": [
                    "Fₓ = F cos θ, Fᵧ = F sin θ",
                    "Fₓ = F sin θ, Fᵧ = F cos θ",
                    "Fₓ = F tan θ, Fᵧ = F/tan θ",
                    "Fₓ = F + θ, Fᵧ = F - θ",
                ],
                "correct": "Fₓ = F cos θ, Fᵧ = F sin θ",
                "explanation": "The angle is measured from the x-axis, so x is adjacent and y is opposite.",
            },
            "answers": [
                make_answer("Fₓ", fx, "N", 0.05),
                make_answer("Fᵧ", fy, "N", 0.05),
            ],
            "solution": [
                ("text", "The angle is measured from the positive x-axis."),
                ("latex", rf"F_x=F\cos\theta=({magnitude})\cos({angle}^\circ)={fx:.4f}\ \text{{N}}"),
                ("latex", rf"F_y=F\sin\theta=({magnitude})\sin({angle}^\circ)={fy:.4f}\ \text{{N}}"),
                ("text", "Both components are positive because the vector is in the first quadrant."),
                ("latex", rf"\mathbf F=({fx:.4f})\mathbf i+({fy:.4f})\mathbf j\ \text{{N}}"),
            ],
            "signature": f"V1:{magnitude}:{angle}",
            "diagram": ("vector", {
                "angle_from_x": angle,
                "x_label": r"$F_x$",
                "y_label": r"$F_y$",
                "vector_label": f"F = {magnitude} N",
                "angle_label": rf"${angle}^\circ$",
            }),
            "teaching_point": "When the angle is measured from x, cosine belongs to x and sine belongs to y.",
        }

    if level == 2:
        # Angle from vertical, first quadrant.
        fx = magnitude * math.sin(math.radians(angle))
        fy = magnitude * math.cos(math.radians(angle))
        angle_from_x = 90 - angle

        return {
            "title": "Vector components when angle is measured from vertical",
            "difficulty": "Easy",
            "prompt": "The angle is measured from the positive y-axis. Resolve the vector into x- and y-components.",
            "display_equation": None,
            "formula_question": {
                "question": "Which equations are correct when θ is measured from y?",
                "options": [
                    "Fₓ = F sin θ, Fᵧ = F cos θ",
                    "Fₓ = F cos θ, Fᵧ = F sin θ",
                    "Fₓ = F tan θ, Fᵧ = F",
                    "Fₓ = F/θ, Fᵧ = Fθ",
                ],
                "correct": "Fₓ = F sin θ, Fᵧ = F cos θ",
                "explanation": "Relative to the angle from y, the y-component is adjacent and the x-component is opposite.",
            },
            "answers": [
                make_answer("Fₓ", fx, "N", 0.05),
                make_answer("Fᵧ", fy, "N", 0.05),
            ],
            "solution": [
                ("text", "Because the angle is measured from the y-axis, the roles of sine and cosine swap."),
                ("latex", rf"F_x=F\sin\theta=({magnitude})\sin({angle}^\circ)={fx:.4f}\ \text{{N}}"),
                ("latex", rf"F_y=F\cos\theta=({magnitude})\cos({angle}^\circ)={fy:.4f}\ \text{{N}}"),
                ("latex", rf"\mathbf F=({fx:.4f})\mathbf i+({fy:.4f})\mathbf j\ \text{{N}}"),
            ],
            "signature": f"V2:{magnitude}:{angle}",
            "diagram": ("vector", {
                "angle_from_x": angle_from_x,
                "x_label": r"$F_x$",
                "y_label": r"$F_y$",
                "vector_label": f"F = {magnitude} N",
                "angle_label": rf"${angle}^\circ$ from +y",
            }),
            "teaching_point": "Always identify the axis from which the angle is measured before assigning sine and cosine.",
        }

    if level == 3:
        # Quadrant II or IV.
        quadrant = RNG.choice([2, 4])

        if quadrant == 2:
            angle_from_x = 180 - angle
            fx = -magnitude * math.cos(math.radians(angle))
            fy = magnitude * math.sin(math.radians(angle))
            direction_text = f"{angle}° above the negative x-axis"
            sign_note = "x is negative and y is positive."
        else:
            angle_from_x = 360 - angle
            fx = magnitude * math.cos(math.radians(angle))
            fy = -magnitude * math.sin(math.radians(angle))
            direction_text = f"{angle}° below the positive x-axis"
            sign_note = "x is positive and y is negative."

        return {
            "title": "Vector components with signs",
            "difficulty": "Moderate",
            "prompt": f"A vector of magnitude {magnitude} N acts {direction_text}. Find its signed components.",
            "display_equation": None,
            "formula_question": {
                "question": "What determines the component signs?",
                "options": [
                    "The quadrant and actual vector direction",
                    "The vector magnitude only",
                    "Sine is always negative",
                    "Cosine is always positive",
                ],
                "correct": "The quadrant and actual vector direction",
                "explanation": "First calculate component magnitudes, then assign signs from the direction.",
            },
            "answers": [
                make_answer("Fₓ", fx, "N", 0.05),
                make_answer("Fᵧ", fy, "N", 0.05),
            ],
            "solution": [
                ("text", "Calculate the component magnitudes using the acute reference angle."),
                ("latex", rf"|F_x|=({magnitude})\cos({angle}^\circ)={abs(fx):.4f}\ \text{{N}}"),
                ("latex", rf"|F_y|=({magnitude})\sin({angle}^\circ)={abs(fy):.4f}\ \text{{N}}"),
                ("text", f"From the direction, {sign_note}"),
                ("latex", rf"\mathbf F=({fx:.4f})\mathbf i+({fy:.4f})\mathbf j\ \text{{N}}"),
            ],
            "signature": f"V3:{magnitude}:{angle}:{quadrant}",
            "diagram": ("vector", {
                "angle_from_x": angle_from_x,
                "x_label": r"$F_x$",
                "y_label": r"$F_y$",
                "vector_label": f"F = {magnitude} N",
                "angle_label": direction_text,
            }),
            "teaching_point": "Trigonometry gives magnitudes. The diagram and quadrant determine signs.",
        }

    # Level 4: direction triangle shown directly on the vector.
    run, rise = RNG.choice(
        [
            (3, 4),
            (4, 3),
            (5, 2),
            (5, 3),
            (7, 4),
            (8, 5),
        ]
    )
    hyp = math.hypot(run, rise)
    fx = magnitude * run / hyp
    fy = magnitude * rise / hyp
    theta = math.degrees(math.atan2(rise, run))

    return {
        "title": "Force components from a small direction triangle",
        "difficulty": "Challenging",
        "prompt": (
            "The small triangle drawn on the force vector gives its direction. "
            "Use the triangle ratios to calculate the x- and y-components."
        ),
        "display_equation": None,
        "formula_question": {
            "question": "What do the numbers in the small triangle represent?",
            "options": [
                "Direction ratios used to form a unit vector",
                "The actual force components in newtons",
                "The x- and y-coordinates of the force application point",
                "The force magnitude and moment",
            ],
            "correct": "Direction ratios used to form a unit vector",
            "explanation": (
                "The small triangle describes direction only. Divide its horizontal "
                "and vertical sides by its hypotenuse, then multiply by F."
            ),
        },
        "answers": [
            make_answer("Fₓ", fx, "N", 0.05),
            make_answer("Fᵧ", fy, "N", 0.05),
        ],
        "solution": [
            (
                "text",
                "The small triangle is similar to the large component triangle. "
                "Its sides give the direction ratios.",
            ),
            (
                "latex",
                rf"L=\sqrt{{({run})^2+({rise})^2}}={hyp:.4f}",
            ),
            (
                "text",
                "The horizontal direction ratio is run/L and the vertical direction "
                "ratio is rise/L.",
            ),
            (
                "latex",
                rf"\cos\theta=\frac{{{run}}}{{{hyp:.4f}}},"
                rf"\qquad \sin\theta=\frac{{{rise}}}{{{hyp:.4f}}}",
            ),
            (
                "latex",
                rf"F_x=F\frac{{{run}}}{{L}}"
                rf"=({magnitude})\frac{{{run}}}{{{hyp:.4f}}}"
                rf"={fx:.4f}\ \text{{N}}",
            ),
            (
                "latex",
                rf"F_y=F\frac{{{rise}}}{{L}}"
                rf"=({magnitude})\frac{{{rise}}}{{{hyp:.4f}}}"
                rf"={fy:.4f}\ \text{{N}}",
            ),
            (
                "text",
                "This is equivalent to using cosine and sine with the angle shown.",
            ),
            (
                "latex",
                rf"\theta=\tan^{{-1}}\left(\frac{{{rise}}}{{{run}}}\right)"
                rf"={theta:.4f}^\circ",
            ),
            (
                "latex",
                rf"\mathbf F=({fx:.4f})\mathbf i+({fy:.4f})\mathbf j\ \text{{N}}",
            ),
        ],
        "signature": f"V4:{magnitude}:{run}:{rise}",
        "diagram": (
            "direction_triangle",
            {
                "magnitude": magnitude,
                "run": run,
                "rise": rise,
            },
        ),
        "teaching_point": (
            "The small triangle gives direction ratios. Its side numbers are not "
            "the force components until they are divided by the triangle hypotenuse "
            "and multiplied by the force magnitude."
        ),
    }



def generate_differentiation(level):
    if level == 1:
        coefficient = RNG.choice([2, 3, 4, 5, 6, -2, -3, -4])
        power = RNG.randint(2, 6)
        expr = coefficient * x**power
        derivative = sp.diff(expr, x)
        x_value = RNG.randint(1, 4)
        answer_value = derivative.subs(x, x_value)

        return {
            "title": "Power-rule differentiation",
            "difficulty": "Foundation",
            "prompt": f"Differentiate the function, then evaluate the derivative at x = {x_value}.",
            "display_equation": rf"y={sp.latex(expr)}",
            "formula_question": {
                "question": "Which rule is required?",
                "options": [
                    "d(xⁿ)/dx = n xⁿ⁻¹",
                    "∫xⁿdx = xⁿ⁺¹/(n+1)",
                    "sin θ = opposite/hypotenuse",
                    "ax + b = c",
                ],
                "correct": "d(xⁿ)/dx = n xⁿ⁻¹",
                "explanation": "Multiply by the power and reduce the power by one.",
            },
            "answers": [
                make_answer(f"dy/dx at x = {x_value}", answer_value),
            ],
            "solution": [
                ("latex", rf"y={sp.latex(expr)}"),
                ("latex", r"\frac{d}{dx}\left(ax^n\right)=anx^{n-1}"),
                ("latex", rf"\frac{{dy}}{{dx}}={sp.latex(derivative)}"),
                ("latex", rf"\left.\frac{{dy}}{{dx}}\right|_{{x={x_value}}}={sp.latex(derivative.subs(x, x_value))}"),
            ],
            "signature": f"D1:{coefficient}:{power}:{x_value}",
            "diagram": None,
            "teaching_point": "The derivative gives the rate of change or slope of a function.",
        }

    if level == 2:
        a = RNG.choice([2, 3, 4, -2, -3])
        b = RNG.choice([2, 3, 5, -2, -4])
        n = RNG.randint(3, 5)
        m = RNG.randint(1, n - 1)
        c = RNG.randint(-8, 8)
        expr = a * x**n + b * x**m + c
        derivative = sp.diff(expr, x)
        x_value = RNG.randint(1, 3)
        answer_value = derivative.subs(x, x_value)

        return {
            "title": "Differentiate a polynomial",
            "difficulty": "Easy",
            "prompt": f"Differentiate term-by-term, then evaluate at x = {x_value}.",
            "display_equation": rf"y={sp.latex(expr)}",
            "formula_question": {
                "question": "What happens to a constant during differentiation?",
                "options": [
                    "Its derivative is zero",
                    "It remains unchanged",
                    "It becomes x",
                    "It becomes infinity",
                ],
                "correct": "Its derivative is zero",
                "explanation": "A constant does not change with x, so its rate of change is zero.",
            },
            "answers": [
                make_answer(f"dy/dx at x = {x_value}", answer_value),
            ],
            "solution": [
                ("text", "Differentiate each term separately."),
                ("latex", rf"y={sp.latex(expr)}"),
                ("latex", rf"\frac{{dy}}{{dx}}={sp.latex(derivative)}"),
                ("text", "The constant term disappears because its derivative is zero."),
                ("latex", rf"\left.\frac{{dy}}{{dx}}\right|_{{x={x_value}}}={sp.latex(answer_value)}"),
            ],
            "signature": f"D2:{a}:{n}:{b}:{m}:{c}:{x_value}",
            "diagram": None,
            "teaching_point": "Differentiate one term at a time. Constants differentiate to zero.",
        }

    if level == 3:
        a = RNG.choice([1, 2, 3])
        b = RNG.choice([-4, -3, -2, 2, 3, 4])
        c = RNG.choice([-5, -3, 2, 4, 6])
        d = RNG.randint(-5, 5)
        s_expr = a * t**3 + b * t**2 + c * t + d
        v_expr = sp.diff(s_expr, t)
        acc_expr = sp.diff(v_expr, t)
        time_value = RNG.randint(1, 4)
        v_value = v_expr.subs(t, time_value)
        acc_value = acc_expr.subs(t, time_value)

        return {
            "title": "Position → velocity → acceleration",
            "difficulty": "Moderate",
            "prompt": f"The position function is given. Find velocity and acceleration at t = {time_value} s.",
            "display_equation": rf"s(t)={sp.latex(s_expr)}",
            "formula_question": {
                "question": "Which relationships are correct?",
                "options": [
                    "v = ds/dt and a = dv/dt",
                    "v = ∫s dt and a = ∫v dt",
                    "v = s/t and a = v/t for every problem",
                    "v = sin s and a = cos v",
                ],
                "correct": "v = ds/dt and a = dv/dt",
                "explanation": "Velocity is the time derivative of position, and acceleration is the time derivative of velocity.",
            },
            "answers": [
                make_answer(f"Velocity at t = {time_value}", v_value, "m/s"),
                make_answer(f"Acceleration at t = {time_value}", acc_value, "m/s²"),
            ],
            "solution": [
                ("latex", rf"s(t)={sp.latex(s_expr)}"),
                ("latex", r"v(t)=\frac{ds}{dt}"),
                ("latex", rf"v(t)={sp.latex(v_expr)}"),
                ("latex", r"a(t)=\frac{dv}{dt}"),
                ("latex", rf"a(t)={sp.latex(acc_expr)}"),
                ("latex", rf"v({time_value})={sp.latex(v_value)}\ \text{{m/s}}"),
                ("latex", rf"a({time_value})={sp.latex(acc_value)}\ \text{{m/s}}^2"),
            ],
            "signature": f"D3:{a}:{b}:{c}:{d}:{time_value}",
            "diagram": None,
            "teaching_point": "In kinematics, differentiate position once for velocity and twice for acceleration.",
        }


    if level == 4:
        trig_kind = RNG.choice(["sin", "cos"])
        amplitude = RNG.choice([2, 3, 4, 5, 6])
        frequency = RNG.choice([1, 2, 3, 4])
        target_angle = RNG.choice(
            [
                sp.Integer(0),
                sp.pi / 6,
                sp.pi / 4,
                sp.pi / 3,
            ]
        )
        x_value = sp.simplify(target_angle / frequency)

        if trig_kind == "sin":
            expr = amplitude * sp.sin(frequency * x)
            derivative = amplitude * frequency * sp.cos(frequency * x)
            rule_text = r"\frac{d}{dx}\sin(kx)=k\cos(kx)"
            correct_option = "d[sin(kx)]/dx = k cos(kx)"
            options = [
                correct_option,
                "d[sin(kx)]/dx = -k cos(kx)",
                "d[sin(kx)]/dx = cos(kx)/k",
                "d[sin(kx)]/dx = k sin(kx)",
            ]
        elif trig_kind == "cos":
            expr = amplitude * sp.cos(frequency * x)
            derivative = -amplitude * frequency * sp.sin(frequency * x)
            rule_text = r"\frac{d}{dx}\cos(kx)=-k\sin(kx)"
            correct_option = "d[cos(kx)]/dx = -k sin(kx)"
            options = [
                "d[cos(kx)]/dx = k sin(kx)",
                correct_option,
                "d[cos(kx)]/dx = sin(kx)/k",
                "d[cos(kx)]/dx = -cos(kx)/k",
            ]

        derivative_value = sp.simplify(derivative.subs(x, x_value))

        return {
            "title": "Differentiate sine or cosine",
            "difficulty": "Moderate",
            "prompt": (
                "Differentiate the trigonometric function and evaluate the "
                "derivative at the stated x-value."
            ),
            "display_equation": (
                rf"y={sp.latex(expr)},"
                rf"\qquad x={sp.latex(x_value)}\ \text{{rad}}"
            ),
            "formula_question": {
                "question": "Which derivative rule is correct for this function?",
                "options": options,
                "correct": correct_option,
                "explanation": (
                    "Use the trigonometric derivative and multiply by the derivative "
                    "of the inside function kx. This is the chain rule."
                ),
            },
            "answers": [
                make_answer(
                    f"dy/dx at x = {sp.latex(x_value)} rad",
                    derivative_value,
                ),
            ],
            "solution": [
                (
                    "text",
                    "Trigonometric differentiation uses radians. First identify the "
                    "outer trigonometric function and the inside function.",
                ),
                ("latex", rf"y={sp.latex(expr)}"),
                ("latex", rule_text),
                (
                    "text",
                    f"The derivative of the inside function {frequency}x is "
                    f"{frequency}.",
                ),
                ("latex", rf"\frac{{dy}}{{dx}}={sp.latex(derivative)}"),
                (
                    "latex",
                    rf"\left.\frac{{dy}}{{dx}}\right|_{{x={sp.latex(x_value)}}}"
                    rf"={sp.latex(derivative_value)}"
                    rf"\approx {float(derivative_value):.5f}",
                ),
            ],
            "signature": (
                f"D4:{trig_kind}:{amplitude}:{frequency}:"
                f"{sp.srepr(target_angle)}"
            ),
            "diagram": None,
            "teaching_point": (
                "For sin(kx) and cos(kx), the factor k appears because of the "
                "chain rule. Calculus trigonometric formulas assume radians."
            ),
        }

    # Level 5: sinusoidal motion.
    motion_kind = RNG.choice(["sin", "cos"])
    amplitude = RNG.choice([2, 3, 4, 5])
    omega = RNG.choice([1, 2, 3, 4])
    target_angle = RNG.choice(
        [
            sp.Integer(0),
            sp.pi / 6,
            sp.pi / 4,
            sp.pi / 3,
            sp.pi / 2,
        ]
    )
    time_value = sp.simplify(target_angle / omega)

    if motion_kind == "sin":
        s_expr = amplitude * sp.sin(omega * t)
    else:
        s_expr = amplitude * sp.cos(omega * t)

    v_expr = sp.diff(s_expr, t)
    a_expr = sp.diff(v_expr, t)
    v_value = sp.simplify(v_expr.subs(t, time_value))
    a_value = sp.simplify(a_expr.subs(t, time_value))

    return {
        "title": "Differentiate sinusoidal motion",
        "difficulty": "Challenging",
        "prompt": (
            "The particle position is sinusoidal. Find the velocity and "
            "acceleration at the stated time."
        ),
        "display_equation": (
            rf"s(t)={sp.latex(s_expr)}\ \text{{m}},"
            rf"\qquad t={sp.latex(time_value)}\ \text{{s}}"
        ),
        "formula_question": {
            "question": "Which kinematic relationships are required?",
            "options": [
                "v = ds/dt and a = d²s/dt²",
                "v = ∫s dt and a = ∫v dt",
                "v = s/t and a = v/t in every case",
                "v = d²s/dt² and a = ds/dt",
            ],
            "correct": "v = ds/dt and a = d²s/dt²",
            "explanation": (
                "Differentiate position once for velocity and twice for acceleration."
            ),
        },
        "answers": [
            make_answer(
                f"Velocity at t = {sp.latex(time_value)}",
                v_value,
                "m/s",
            ),
            make_answer(
                f"Acceleration at t = {sp.latex(time_value)}",
                a_value,
                "m/s²",
            ),
        ],
        "solution": [
            (
                "text",
                "The angular argument is in radians. Differentiate the position "
                "function using the chain rule.",
            ),
            ("latex", rf"s(t)={sp.latex(s_expr)}"),
            ("latex", r"v(t)=\frac{ds}{dt}"),
            ("latex", rf"v(t)={sp.latex(v_expr)}"),
            ("latex", r"a(t)=\frac{dv}{dt}=\frac{d^2s}{dt^2}"),
            ("latex", rf"a(t)={sp.latex(a_expr)}"),
            (
                "text",
                "The answer has been obtained by differentiating twice. "
                "For this sinusoidal function only, we can now check that "
                "a(t)=-ω²s(t).",
            ),
            (
                "latex",
                rf"a(t)=-({omega})^2s(t)=-{omega**2}s(t)",
            ),
            (
                "latex",
                rf"({sp.latex(a_expr)})+({omega})^2"
                rf"({sp.latex(s_expr)})=0",
            ),
            (
                "latex",
                rf"v({sp.latex(time_value)})={sp.latex(v_value)}"
                rf"\approx {float(v_value):.5f}\ \text{{m/s}}",
            ),
            (
                "latex",
                rf"a({sp.latex(time_value)})={sp.latex(a_value)}"
                rf"\approx {float(a_value):.5f}\ \text{{m/s}}^2",
            ),
        ],
        "signature": (
            f"D5:{motion_kind}:{amplitude}:{omega}:"
            f"{sp.srepr(target_angle)}"
        ),
        "diagram": None,
        "teaching_point": (
            "For s=A sin(ωt) or A cos(ωt), each differentiation introduces a "
            "factor ω. The acceleration is opposite in sign to displacement: "
            "a=-ω²s."
        ),
    }



def generate_integration(level):
    if level == 1:
        coefficient = RNG.choice([2, 3, 4, 5, 6, -2, -3])
        power = RNG.randint(1, 5)
        upper = RNG.randint(2, 5)
        expr = coefficient * x**power
        antiderivative = sp.integrate(expr, x)
        value = sp.integrate(expr, (x, 0, upper))

        return {
            "title": "Basic definite integration",
            "difficulty": "Foundation",
            "prompt": f"Evaluate the definite integral from x = 0 to x = {upper}.",
            "display_equation": rf"\int_0^{{{upper}}}{sp.latex(expr)}\,dx",
            "formula_question": {
                "question": "Which integration rule is required?",
                "options": [
                    "∫xⁿdx = xⁿ⁺¹/(n+1)",
                    "d(xⁿ)/dx = n xⁿ⁻¹",
                    "Differentiate xⁿ using the power rule",
                    "Add the upper and lower limits directly",
                ],
                "correct": "∫xⁿdx = xⁿ⁺¹/(n+1)",
                "explanation": "Increase the power by one, then divide by the new power.",
            },
            "answers": [
                make_answer("Integral value", value),
            ],
            "solution": [
                ("latex", rf"\int {sp.latex(expr)}\,dx={sp.latex(antiderivative)}+C"),
                ("text", "For a definite integral, evaluate the antiderivative at the upper and lower limits."),
                ("latex", rf"\left[{sp.latex(antiderivative)}\right]_0^{{{upper}}}"),
                ("latex", rf"{sp.latex(antiderivative.subs(x, upper))}-{sp.latex(antiderivative.subs(x, 0))}={sp.latex(value)}"),
            ],
            "signature": f"I1:{coefficient}:{power}:{upper}",
            "diagram": None,
            "teaching_point": "Integration reverses differentiation. For definite integrals, subtract the lower-limit value from the upper-limit value.",
        }

    if level == 2:
        a = RNG.choice([1, 2, 3, -1, -2])
        b = RNG.choice([2, 3, 4, -2, -3])
        c = RNG.randint(-5, 5)
        lower = RNG.randint(0, 2)
        upper = RNG.randint(lower + 2, lower + 5)
        expr = a * x**2 + b * x + c
        antiderivative = sp.integrate(expr, x)
        value = sp.integrate(expr, (x, lower, upper))

        return {
            "title": "Integrate a polynomial",
            "difficulty": "Easy",
            "prompt": f"Evaluate the polynomial integral from x = {lower} to x = {upper}.",
            "display_equation": rf"\int_{{{lower}}}^{{{upper}}}\left({sp.latex(expr)}\right)\,dx",
            "formula_question": {
                "question": "How should a polynomial be integrated?",
                "options": [
                    "Integrate each term separately",
                    "Multiply all powers together",
                    "Differentiate the entire expression",
                    "Use the Pythagorean theorem",
                ],
                "correct": "Integrate each term separately",
                "explanation": "Integration distributes across sums and differences.",
            },
            "answers": [
                make_answer("Integral value", value),
            ],
            "solution": [
                ("text", "Integrate every term separately."),
                ("latex", rf"\int\left({sp.latex(expr)}\right)\,dx={sp.latex(antiderivative)}+C"),
                ("latex", rf"\left[{sp.latex(antiderivative)}\right]_{{{lower}}}^{{{upper}}}"),
                ("latex", rf"{sp.latex(antiderivative.subs(x, upper))}-{sp.latex(antiderivative.subs(x, lower))}={sp.latex(value)}"),
            ],
            "signature": f"I2:{a}:{b}:{c}:{lower}:{upper}",
            "diagram": None,
            "teaching_point": "Integrate term-by-term and apply the limits only after obtaining the antiderivative.",
        }

    if level == 3:
        a = RNG.choice([1, 2, 3, 4])
        b = RNG.choice([-3, -2, 0, 2, 3])
        v0 = RNG.randint(-5, 8)
        time_value = RNG.randint(2, 5)
        acc_expr = a * t + b
        v_expr = sp.integrate(acc_expr, t) + sp.Symbol("C")
        constant = v0
        v_exact = sp.integrate(acc_expr, (t, 0, t)) + v0
        answer_value = v_exact.subs(t, time_value)

        return {
            "title": "Acceleration → velocity using an initial condition",
            "difficulty": "Moderate",
            "prompt": f"Acceleration is given and v(0) = {v0} m/s. Find v({time_value}).",
            "display_equation": rf"a(t)={sp.latex(acc_expr)}",
            "formula_question": {
                "question": "Which relationship should be used?",
                "options": [
                    "v(t) = v₀ + ∫₀ᵗ a(τ)dτ",
                    "v(t) = da/dt",
                    "v(t) = a(t)/t",
                    "v(t) = sin a",
                ],
                "correct": "v(t) = v₀ + ∫₀ᵗ a(τ)dτ",
                "explanation": "Velocity change equals the time integral of acceleration.",
            },
            "answers": [
                make_answer(f"v({time_value})", answer_value, "m/s"),
            ],
            "solution": [
                ("latex", r"v(t)=v_0+\int_0^t a(\tau)\,d\tau"),
                ("latex", rf"v(t)={v0}+\int_0^t\left({sp.latex(acc_expr.subs(t, sp.Symbol('tau')))}\right)\,d\tau"),
                ("latex", rf"v(t)={sp.latex(v_exact)}"),
                ("latex", rf"v({time_value})={sp.latex(answer_value)}\ \text{{m/s}}"),
                ("text", "The initial velocity determines the integration constant."),
            ],
            "signature": f"I3:{a}:{b}:{v0}:{time_value}",
            "diagram": None,
            "teaching_point": "An indefinite integral needs a constant. An initial condition determines that constant.",
        }

    if level == 4:
        a = RNG.choice([1, 2, 3])
        b = RNG.choice([-4, -2, 1, 3, 5])
        s0 = RNG.randint(-5, 10)
        time_value = RNG.randint(2, 5)
        velocity = a * t**2 + b
        position = sp.integrate(velocity, (t, 0, t)) + s0
        answer_value = position.subs(t, time_value)

        return {
            "title": "Velocity → position using an initial condition",
            "difficulty": "Challenging",
            "prompt": f"Velocity is given and s(0) = {s0} m. Find s({time_value}).",
            "display_equation": rf"v(t)={sp.latex(velocity)}",
            "formula_question": {
                "question": "Which relationship should be used?",
                "options": [
                    "s(t) = s₀ + ∫₀ᵗ v(τ)dτ",
                    "s(t) = dv/dt",
                    "s(t) = v(t)/t",
                    "s(t) = cos v",
                ],
                "correct": "s(t) = s₀ + ∫₀ᵗ v(τ)dτ",
                "explanation": "Position change equals the time integral of velocity.",
            },
            "answers": [
                make_answer(f"s({time_value})", answer_value, "m"),
            ],
            "solution": [
                ("latex", r"s(t)=s_0+\int_0^t v(\tau)\,d\tau"),
                ("latex", rf"s(t)={s0}+\int_0^t\left({sp.latex(velocity.subs(t, sp.Symbol('tau')))}\right)\,d\tau"),
                ("latex", rf"s(t)={sp.latex(position)}"),
                ("latex", rf"s({time_value})={sp.latex(answer_value)}\ \text{{m}}"),
                ("text", "The initial position supplies the integration constant."),
            ],
            "signature": f"I4:{a}:{b}:{s0}:{time_value}",
            "diagram": None,
            "teaching_point": "Integrating velocity gives displacement. Add the initial position to obtain absolute position.",
        }


    if level == 5:
        trig_kind = RNG.choice(["sin", "cos"])
        amplitude = RNG.choice([2, 3, 4, 5, 6])
        frequency = RNG.choice([1, 2, 3, 4])

        target_angle = RNG.choice(
            [
                sp.pi / 6,
                sp.pi / 4,
                sp.pi / 3,
                sp.pi / 2,
            ]
        )

        upper = sp.simplify(target_angle / frequency)

        if trig_kind == "sin":
            expr = amplitude * sp.sin(frequency * x)
            antiderivative = -sp.Rational(amplitude, frequency) * sp.cos(
                frequency * x
            )
            rule_text = (
                r"\int A\sin(kx)\,dx="
                r"-\frac{A}{k}\cos(kx)+C"
            )
            correct_option = "∫A sin(kx)dx = -(A/k) cos(kx) + C"
            options = [
                correct_option,
                "∫A sin(kx)dx = A k cos(kx) + C",
                "∫A sin(kx)dx = (A/k) sin(kx) + C",
                "∫A sin(kx)dx = A sec²(kx) + C",
            ]
        elif trig_kind == "cos":
            expr = amplitude * sp.cos(frequency * x)
            antiderivative = sp.Rational(amplitude, frequency) * sp.sin(
                frequency * x
            )
            rule_text = (
                r"\int A\cos(kx)\,dx="
                r"\frac{A}{k}\sin(kx)+C"
            )
            correct_option = "∫A cos(kx)dx = (A/k) sin(kx) + C"
            options = [
                "∫A cos(kx)dx = -(A/k) cos(kx) + C",
                correct_option,
                "∫A cos(kx)dx = A k sin(kx) + C",
                "∫A cos(kx)dx = -(A/k) ln|cos(kx)| + C",
            ]

        exact_value = sp.simplify(
            antiderivative.subs(x, upper) - antiderivative.subs(x, 0)
        )

        return {
            "title": "Integrate sine or cosine",
            "difficulty": "Moderate",
            "prompt": (
                "Evaluate the definite trigonometric integral. The angle "
                "arguments are in radians."
            ),
            "display_equation": (
                rf"\int_0^{{{sp.latex(upper)}}}"
                rf"\left({sp.latex(expr)}\right)\,dx"
            ),
            "formula_question": {
                "question": "Which antiderivative is correct for this function?",
                "options": options,
                "correct": correct_option,
                "explanation": (
                    "Use the trigonometric antiderivative and divide by the "
                    "inside coefficient k. This reverses the chain rule."
                ),
            },
            "answers": [
                make_answer("Integral value", exact_value),
            ],
            "solution": [
                (
                    "text",
                    "Trigonometric integration uses radians. The inside function is "
                    f"{frequency}x, so the antiderivative contains a factor "
                    f"1/{frequency}.",
                ),
                ("latex", rule_text),
                (
                    "latex",
                    rf"\int\left({sp.latex(expr)}\right)\,dx"
                    rf"={sp.latex(antiderivative)}+C",
                ),
                (
                    "latex",
                    rf"\left[{sp.latex(antiderivative)}\right]_0^"
                    rf"{{{sp.latex(upper)}}}",
                ),
                (
                    "latex",
                    rf"={sp.latex(exact_value)}"
                    rf"\approx {float(exact_value):.5f}",
                ),
            ],
            "signature": (
                f"I5:{trig_kind}:{amplitude}:{frequency}:"
                f"{sp.srepr(target_angle)}"
            ),
            "diagram": None,
            "teaching_point": (
                "Integration reverses the chain rule: a function of kx normally "
                "introduces a factor 1/k in the antiderivative."
            ),
        }

    # Level 6: integrate sinusoidal acceleration to velocity.
    motion_kind = RNG.choice(["sin", "cos"])
    amplitude = RNG.choice([2, 3, 4, 5])
    omega = RNG.choice([1, 2, 3, 4])
    v0 = RNG.randint(-4, 6)
    target_angle = RNG.choice(
        [
            sp.pi / 6,
            sp.pi / 4,
            sp.pi / 3,
            sp.pi / 2,
        ]
    )
    time_value = sp.simplify(target_angle / omega)

    if motion_kind == "sin":
        acceleration = amplitude * sp.sin(omega * t)
    else:
        acceleration = amplitude * sp.cos(omega * t)

    tau = sp.symbols("tau", real=True)
    velocity = sp.simplify(
        v0
        + sp.integrate(
            acceleration.subs(t, tau),
            (tau, 0, t),
        )
    )
    velocity_value = sp.simplify(velocity.subs(t, time_value))

    return {
        "title": "Integrate sinusoidal acceleration",
        "difficulty": "Challenging",
        "prompt": (
            f"The acceleration function is given and v(0) = {v0} m/s. "
            "Find the velocity at the stated time."
        ),
        "display_equation": (
            rf"a(t)={sp.latex(acceleration)}\ \text{{m/s}}^2,"
            rf"\qquad t={sp.latex(time_value)}\ \text{{s}}"
        ),
        "formula_question": {
            "question": "Which initial-condition form should be used?",
            "options": [
                "v(t) = v₀ + ∫₀ᵗ a(τ)dτ",
                "v(t) = da/dt",
                "v(t) = a(t)/t",
                "v(t) = ∫₀ᵗ v(τ)dτ",
            ],
            "correct": "v(t) = v₀ + ∫₀ᵗ a(τ)dτ",
            "explanation": (
                "The velocity change is the integral of acceleration. Add the "
                "known initial velocity."
            ),
        },
        "answers": [
            make_answer(
                f"Velocity at t = {sp.latex(time_value)}",
                velocity_value,
                "m/s",
            ),
        ],
        "solution": [
            (
                "text",
                "Use a dummy integration variable τ so the upper limit can remain t.",
            ),
            ("latex", r"v(t)=v_0+\int_0^t a(\tau)\,d\tau"),
            (
                "latex",
                rf"v(t)={v0}+\int_0^t"
                rf"\left({sp.latex(acceleration.subs(t, tau))}\right)d\tau",
            ),
            ("latex", rf"v(t)={sp.latex(velocity)}"),
            (
                "latex",
                rf"v({sp.latex(time_value)})={sp.latex(velocity_value)}"
                rf"\approx {float(velocity_value):.5f}\ \text{{m/s}}",
            ),
        ],
        "signature": (
            f"I6:{motion_kind}:{amplitude}:{omega}:{v0}:"
            f"{sp.srepr(target_angle)}"
        ),
        "diagram": None,
        "teaching_point": (
            "Integrating acceleration gives the change in velocity, not the final "
            "velocity by itself. The initial velocity must be included."
        ),
    }




# --- Centroid and second moment of area ---------------------------------------
y = sp.symbols("y", real=True)

def area_properties(cut, a, b, first, second):
    if cut == "vertical":
        yt, yb = map(sp.sympify, (first, second)); h=sp.simplify(yt-yb)
        A=sp.integrate(h,(x,a,b)); xt=x; ytld=sp.simplify((yt+yb)/2)
        Qy=sp.integrate(xt*h,(x,a,b)); Qx=sp.integrate(ytld*h,(x,a,b))
        Ix=sp.integrate((yt**3-yb**3)/3,(x,a,b)); Iy=sp.integrate(x**2*h,(x,a,b))
        data=dict(cut=cut,a=a,b=b,top=yt,bottom=yb,dA=h)
    else:
        xr, xl = map(sp.sympify, (first, second)); w=sp.simplify(xr-xl)
        A=sp.integrate(w,(y,a,b)); xt=sp.simplify((xr+xl)/2); ytld=y
        Qy=sp.integrate(xt*w,(y,a,b)); Qx=sp.integrate(ytld*w,(y,a,b))
        Ix=sp.integrate(y**2*w,(y,a,b)); Iy=sp.integrate((xr**3-xl**3)/3,(y,a,b))
        data=dict(cut=cut,a=a,b=b,right=xr,left=xl,dA=w)
    xb=sp.simplify(Qy/A); ybbar=sp.simplify(Qx/A)
    sol=dict(A=sp.simplify(A),xb=xb,yb=ybbar,Ix=sp.simplify(Ix),Iy=sp.simplify(Iy),
             Ixc=sp.simplify(Ix-A*ybbar**2),Iyc=sp.simplify(Iy-A*xb**2),xt=xt,yt=ytld)
    data['sol']=sol; return data

def centroid_figure(data,pos,show_whole=False):
    fig,ax=plt.subplots(figsize=(7.5,5.2)); a,b=map(float,(data['a'],data['b']))
    blue='#93c5fd'; orange='#f97316'; purple='#7c3aed'; red='#dc2626'
    if data['cut']=='vertical':
        q=np.linspace(a,b,500); ft=sp.lambdify(x,data['top'],'numpy'); fb=sp.lambdify(x,data['bottom'],'numpy')
        top=np.asarray(ft(q),float)+np.zeros_like(q); bot=np.asarray(fb(q),float)+np.zeros_like(q)
        ax.fill_between(q,bot,top,color=blue,alpha=.65); ax.plot(q,top,color='#1e3a8a',lw=2,label='$y_T$'); ax.plot(q,bot,color='#0f766e',lw=2,label='$y_B$')
        z=float(pos); t=float(data['top'].subs(x,z)); d=float(data['bottom'].subs(x,z)); eps=max((b-a)*.018,.02)
        ax.fill_between([z-eps,z+eps],[d,d],[t,t],color=orange,alpha=.95,zorder=4)
        c=(z,(t+d)/2); ax.text(z,d-.08*max(1,t-d),'$dx$',ha='center',color=orange,weight='bold')
        ax.annotate('$y_T-y_B$',xy=(z,(t+d)/2),xytext=(z+.15*(b-a),(t+d)/2),arrowprops=dict(arrowstyle='->',color=orange),color=orange,weight='bold')
    else:
        q=np.linspace(a,b,500); fr=sp.lambdify(y,data['right'],'numpy'); fl=sp.lambdify(y,data['left'],'numpy')
        right=np.asarray(fr(q),float)+np.zeros_like(q); left=np.asarray(fl(q),float)+np.zeros_like(q)
        ax.fill_betweenx(q,left,right,color=blue,alpha=.65); ax.plot(right,q,color='#1e3a8a',lw=2,label='$x_R$'); ax.plot(left,q,color='#0f766e',lw=2,label='$x_L$')
        z=float(pos); r=float(data['right'].subs(y,z)); l=float(data['left'].subs(y,z)); eps=max((b-a)*.018,.02)
        ax.fill_betweenx([z-eps,z+eps],[l,l],[r,r],color=orange,alpha=.95,zorder=4)
        c=((r+l)/2,z); ax.text(l-.08*max(1,r-l),z,'$dy$',va='center',color=orange,weight='bold')
        ax.annotate('$x_R-x_L$',xy=((r+l)/2,z),xytext=((r+l)/2,z+.16*(b-a)),ha='center',arrowprops=dict(arrowstyle='->',color=orange),color=orange,weight='bold')
    ax.scatter(*c,s=85,color=purple,zorder=7); ax.annotate('strip centroid $(\\tilde{x},\\tilde{y})$',xy=c,xytext=(10,20),textcoords='offset points',arrowprops=dict(arrowstyle='->',color=purple),color=purple,weight='bold')
    if show_whole:
        cc=(float(data['sol']['xb']),float(data['sol']['yb'])); ax.scatter(*cc,marker='X',s=150,color=red,zorder=8); ax.annotate('area centroid $(\\bar{x},\\bar{y})$',xy=cc,xytext=(12,-28),textcoords='offset points',arrowprops=dict(arrowstyle='->',color=red),color=red,weight='bold')
    ax.axhline(0,color='#64748b'); ax.axvline(0,color='#64748b'); ax.grid(alpha=.15); ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_aspect('equal',adjustable='datalim'); ax.legend(); fig.tight_layout(); return fig

def generate_centroid(level):
    if level==1:
        b,h=RNG.choice([3,4,5,6]),RNG.choice([2,3,4,5]); data=area_properties('vertical',0,b,h,0); asks=['A']; title='Identify a vertical differential strip'; diff='Foundation'
    elif level==2:
        b,h=RNG.choice([3,4,5,6]),RNG.choice([3,4,5,6]); data=area_properties('vertical',0,b,h*(1-x/b),0); asks=['A','xb','yb']; title='Triangle beneath a straight line'; diff='Easy'
    elif level==3:
        b,h=RNG.choice([2,3,4]),RNG.choice([3,4,5,6]); data=area_properties('vertical',0,b,h*(1-(x/b)**2),0); asks=['A','xb','yb']; title='Area beneath a parabola'; diff='Moderate'
    elif level==4:
        h,b=RNG.choice([2,3,4]),RNG.choice([3,4,5,6]); data=area_properties('horizontal',0,h,b*(1-(y/h)**2),0); asks=['A','xb','yb']; title='Horizontal strip from x=f(y)'; diff='Moderate'
    elif level==5:
        b,h=RNG.choice([3,4,5]),RNG.choice([3,4,5]); data=area_properties('vertical',0,b,h*(1-x/b),0); asks=['A','xb','yb','Ix','Iy']; title='Second moments of a triangular area'; diff='Challenging'
    else:
        b,h=RNG.choice([2,3,4]),RNG.choice([3,4,5]); data=area_properties('vertical',0,b,h,h*(x/b)**2); asks=['A','xb','yb','Ixc','Iyc']; title='Centroidal moments between two curves'; diff='Advanced'
    sol=data['sol']; labels={'A':('Area A','units²'),'xb':('Centroid x̄','units'),'yb':('Centroid ȳ','units'),'Ix':('Iₓ about x-axis','units⁴'),'Iy':('Iᵧ about y-axis','units⁴'),'Ixc':('Iₓ̄ centroidal','units⁴'),'Iyc':('Iᵧ̄ centroidal','units⁴')}
    correct='Vertical strip (dx)' if data['cut']=='vertical' else 'Horizontal strip (dy)'; other='Horizontal strip (dy)' if data['cut']=='vertical' else 'Vertical strip (dx)'
    variable=x if data['cut']=='vertical' else y; d='dx' if data['cut']=='vertical' else 'dy'
    steps=[('text',f"Use a {data['cut']} strip with thickness {d}."),('latex',rf"dA=({sp.latex(data['dA'])})\,{d}"),('latex',rf"\tilde{{x}}={sp.latex(sol['xt'])},\quad\tilde{{y}}={sp.latex(sol['yt'])}"),('latex',rf"A=\int_{{{data['a']}}}^{{{data['b']}}}({sp.latex(data['dA'])})\,{d}={sp.latex(sol['A'])}"),('latex',rf"\bar{{x}}={sp.latex(sol['xb'])},\quad\bar{{y}}={sp.latex(sol['yb'])}"),('latex',rf"I_x={sp.latex(sol['Ix'])},\quad I_y={sp.latex(sol['Iy'])}"),('latex',rf"I_{{\bar x}}={sp.latex(sol['Ixc'])},\quad I_{{\bar y}}={sp.latex(sol['Iyc'])}")]
    return {'title':title,'difficulty':diff,'prompt':'Build the differential strip, establish the limits, then calculate the requested area properties.','display_equation':None,'formula_question':{'question':'Which cut is most direct?','options':[correct,other,'Use dA=dx dy','No integration'],'correct':correct,'explanation':f"The supplied boundaries are most direct with a {data['cut']} strip."},'answers':[make_answer(labels[k][0],sol[k],labels[k][1]) for k in asks],'solution':steps,'signature':f"C{level}:{b}:{h}",'diagram':('centroid',data),'teaching_point':'Keep (x̃,ỹ), the moving strip centroid, separate from (x̄,ȳ), the centroid of the whole area.'}

def render_centroid_learning():
    st.markdown('### Geometry first, integration second')
    st.info('x and y locate boundaries; x̃ and ỹ locate one differential strip; x̄ and ȳ locate the whole-area centroid.')
    a,b=st.columns(2)
    with a:
        st.markdown('#### Vertical strip'); st.latex(r'dA=[y_T-y_B]dx'); st.latex(r'\tilde{x}=x,\quad\tilde{y}=(y_T+y_B)/2')
    with b:
        st.markdown('#### Horizontal strip'); st.latex(r'dA=[x_R-x_L]dy'); st.latex(r'\tilde{x}=(x_R+x_L)/2,\quad\tilde{y}=y')
    st.latex(r'A=\int dA,\quad\bar{x}=\frac{\int\tilde{x}dA}{A},\quad\bar{y}=\frac{\int\tilde{y}dA}{A}')
    st.latex(r'I_x=\int y^2dA,\quad I_y=\int x^2dA')
    st.warning('For a deep vertical strip, Iₓ is not generally ỹ²dA alone. The strip local second moment must also be included.')

GENERATORS = {
    "Linear equations": generate_linear,
    "Trigonometry": generate_trigonometry,
    "Vector components": generate_vector,
    "Differentiation": generate_differentiation,
    "Integration": generate_integration,
    "Centroid and second moment of area": generate_centroid,
}


def render_diagram(problem):
    if not problem.get("diagram"):
        return

    kind, data = problem["diagram"]

    if kind == "trig":
        st.pyplot(trig_triangle_figure(data), use_container_width=True)
    elif kind == "vector":
        st.pyplot(vector_figure(data), use_container_width=True)
    elif kind == "direction_triangle":
        st.pyplot(direction_triangle_figure(data), use_container_width=True)
    elif kind == "centroid":
        a, b = map(float, (data["a"], data["b"]))
        pos = st.slider("Move the differential strip", a, b, (a+b)/2, key=f"{problem['uid']}_strip")
        show = st.checkbox("Reveal whole-area centroid", key=f"{problem['uid']}_whole")
        st.pyplot(centroid_figure(data, pos, show), use_container_width=True)


def render_learning(topic):
    st.markdown("## Main ideas and formulas")
    st.markdown(
        """
        <div class="info">
          Start here when the mathematics is unfamiliar. The formulas are written
          in the same notation used later in Statics and Dynamics.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if topic == "Linear equations":
        st.markdown("### Balancing an equation")
        st.latex(r"ax+b=c")
        st.latex(r"ax=c-b")
        st.latex(r"x=\frac{c-b}{a}")
        st.markdown(
            """
            **Main rule:** whatever operation is applied to one side must also be
            applied to the other side.

            **For simultaneous equations:** two unknowns need two independent
            equations. Use substitution or elimination.
            """
        )
        st.markdown("### Common mistakes")
        st.markdown(
            """
            - Changing a sign incorrectly when moving a term.
            - Dividing only one term instead of the whole side.
            - Stopping before the unknown is fully isolated.
            - Using one equation to solve two unknowns.
            """
        )

    elif topic == "Trigonometry":
        st.markdown("### SOH–CAH–TOA")
        st.latex(r"\sin\theta=\frac{\text{opposite}}{\text{hypotenuse}}")
        st.latex(r"\cos\theta=\frac{\text{adjacent}}{\text{hypotenuse}}")
        st.latex(r"\tan\theta=\frac{\text{opposite}}{\text{adjacent}}")
        st.markdown("### Finding an angle")
        st.latex(r"\theta=\sin^{-1}\left(\frac{O}{H}\right)")
        st.latex(r"\theta=\cos^{-1}\left(\frac{A}{H}\right)")
        st.latex(r"\theta=\tan^{-1}\left(\frac{O}{A}\right)")
        st.markdown(
            """
            **Important:** opposite and adjacent depend on which angle is being used.

            **Calculator:** use degree mode when the angle is given in degrees.
            """
        )
        st.markdown("### Common mistakes")
        st.markdown(
            """
            - Choosing opposite and adjacent before identifying the angle.
            - Using sine instead of inverse sine when finding an angle.
            - Leaving the calculator in radian mode.
            - Rounding intermediate values too early.
            """
        )

    elif topic == "Vector components":
        st.markdown("### Angle measured from the x-axis")
        st.latex(r"F_x=F\cos\theta,\qquad F_y=F\sin\theta")
        st.markdown("### Angle measured from the y-axis")
        st.latex(r"F_x=F\sin\theta,\qquad F_y=F\cos\theta")
        st.markdown("### Direction triangle")
        st.latex(
            r"\mathbf u=\frac{\Delta x\,\mathbf i+\Delta y\,\mathbf j}"
            r"{\sqrt{(\Delta x)^2+(\Delta y)^2}}"
        )
        st.latex(r"\mathbf F=F\mathbf u")
        st.markdown(
            """
            Trigonometry gives component magnitudes. The vector direction and
            quadrant determine the signs.
            """
        )
        st.markdown("### Common mistakes")
        st.markdown(
            """
            - Using cosine for x without checking where the angle is measured.
            - Forgetting negative signs for leftward or downward components.
            - Treating the small direction triangle sides as the actual components.
            - Adding component magnitudes instead of using vector notation.
            """
        )

    elif topic == "Centroid and second moment of area":
        render_centroid_learning()
    elif topic == "Differentiation":
        st.markdown("### Power rule")
        st.latex(r"\frac{d}{dx}(x^n)=nx^{n-1}")
        st.latex(r"\frac{d}{dx}(C)=0")

        st.markdown("### Trigonometric derivatives")
        st.latex(r"\frac{d}{dx}\sin(kx)=k\cos(kx)")
        st.latex(r"\frac{d}{dx}\cos(kx)=-k\sin(kx)")
        st.markdown(
            """
            The factor $k$ comes from the chain rule because the inside
            function is $kx$. These calculus formulas assume the angle is in
            **radians**.
            """
        )

        st.markdown("### Kinematics")
        st.latex(r"v(t)=\frac{ds}{dt}")
        st.latex(r"a(t)=\frac{dv}{dt}=\frac{d^2s}{dt^2}")

        st.markdown("### Special case: sinusoidal motion only")
        st.write(
            "The following relationship is correct only when position is a sine "
            "or cosine function with constant amplitude and angular frequency."
        )
        st.latex(r"s(t)=A\sin(\omega t)")
        st.latex(r"v(t)=\frac{ds}{dt}=A\omega\cos(\omega t)")
        st.latex(
            r"a(t)=\frac{d^2s}{dt^2}"
            r"=-A\omega^2\sin(\omega t)"
        )
        st.latex(r"a(t)=-\omega^2s(t)")
        st.markdown(
            """
            The same final relationship is obtained for
            $s(t)=A cos(ωt)$. It is **not** a general formula for every
            kinematics problem. In the questions, velocity and acceleration are
            calculated by differentiation first; $a=-ω²s$ is used only
            as a check for sinusoidal motion.
            """
        )

        st.markdown("### Common mistakes")
        st.markdown(
            """
            - Reducing the power but forgetting to multiply by the original power.
            - Differentiating a constant as though it contains x.
            - Forgetting the negative sign in the derivative of cosine.
            - Forgetting the inside-function factor k or ω.
            - Using degree-mode ideas inside calculus instead of radians.
            - Substituting the numerical x or t value before differentiating.
            """
        )

    else:
        st.markdown("### Reverse power rule")
        st.latex(r"\int x^n\,dx=\frac{x^{n+1}}{n+1}+C,\qquad n\ne-1")

        st.markdown("### Trigonometric antiderivatives")
        st.latex(r"\int\sin(kx)\,dx=-\frac{1}{k}\cos(kx)+C")
        st.latex(r"\int\cos(kx)\,dx=\frac{1}{k}\sin(kx)+C")
        st.markdown(
            """
            The factor $1/k$ reverses the chain rule. Trigonometric calculus
            uses radians.
            """
        )

        st.markdown("### Kinematics")
        st.latex(r"v(t)=v_0+\int_0^t a(\tau)\,d\tau")
        st.latex(r"s(t)=s_0+\int_0^t v(\tau)\,d\tau")
        st.markdown(
            """
            The constant of integration is essential. Initial conditions such as
            $v(0)$ or $s(0)$ determine its value.
            """
        )

        st.markdown("### Common mistakes")
        st.markdown(
            """
            - Forgetting to increase the power before dividing.
            - Forgetting the constant of integration.
            - Forgetting the factor 1/k for `sin(kx)` or `cos(kx)`.
            - Confusing a trigonometric derivative with its antiderivative.
            - Applying limits before finding the antiderivative.
            - Forgetting to add the initial velocity or initial position.
            """
        )


def award_xp(problem_id, points):
    completed = st.session_state.setdefault("completed_checks", set())
    if problem_id not in completed:
        completed.add(problem_id)
        st.session_state["xp"] = st.session_state.get("xp", 0) + points


# -----------------------------------------------------------------------------
# Persistent question state
# -----------------------------------------------------------------------------
# Streamlit session_state is intentionally temporary. If a student's browser is
# inactive long enough for the Streamlit session to expire, session_state can be
# recreated and the old app would therefore generate a different random problem.
#
# To make a question survive that reconnect, MathQuest stores only a compact
# deterministic random seed (plus topic/level and a short question history) in
# the page URL. The complete problem is regenerated from that seed on every run.
# No answer or worked solution is placed in the URL.
QUESTION_STATE_PARAM = "mq_q"
QUESTION_STATE_VERSION = 1
MAX_QUESTION_HISTORY = 30
TOPIC_ORDER = list(TOPICS.keys())


def _query_param_value(name):
    """Read one query parameter without depending on a particular Streamlit API."""
    try:
        if hasattr(st, "query_params"):
            value = st.query_params.get(name, "")
        else:
            value = st.experimental_get_query_params().get(name, "")
    except Exception:
        return ""

    if isinstance(value, (list, tuple)):
        value = value[-1] if value else ""
    return str(value or "")


def _set_query_param(name, value):
    """Update our query parameter while leaving any unrelated parameters alone."""
    try:
        if hasattr(st, "query_params"):
            st.query_params[name] = value
        else:
            params = st.experimental_get_query_params()
            params[name] = value
            st.experimental_set_query_params(**params)
    except Exception:
        # The app still works with normal session_state behaviour if a very old
        # Streamlit deployment does not permit query-parameter updates.
        pass


def _encode_question_state(state):
    topic_index = TOPIC_ORDER.index(state["topic"])
    payload = {
        "v": QUESTION_STATE_VERSION,
        "t": topic_index,
        "l": int(state["level"]),
        "i": int(state["index"]),
        "s": [int(seed) for seed in state["seeds"]],
    }
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_question_state(token):
    if not token:
        return None

    try:
        padded = token + "=" * (-len(token) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))

        if int(payload.get("v", -1)) != QUESTION_STATE_VERSION:
            return None

        topic_index = int(payload["t"])
        if topic_index < 0 or topic_index >= len(TOPIC_ORDER):
            return None

        topic = TOPIC_ORDER[topic_index]
        level = int(payload["l"])
        if level not in TOPICS[topic]:
            return None

        seeds = [int(seed) for seed in payload.get("s", [])]
        seeds = [seed for seed in seeds if 0 < seed < 2**63]
        if not seeds:
            return None

        # Keep URLs compact even if an older version somehow saved more history.
        if len(seeds) > MAX_QUESTION_HISTORY:
            seeds = seeds[-MAX_QUESTION_HISTORY:]

        index = int(payload.get("i", len(seeds) - 1))
        index = max(0, min(index, len(seeds) - 1))

        return {
            "topic": topic,
            "level": level,
            "seeds": seeds,
            "index": index,
        }
    except Exception:
        return None


def _save_question_state(state):
    """Trim and persist the current question/history in the browser URL."""
    seeds = list(state["seeds"])
    index = int(state["index"])

    if len(seeds) > MAX_QUESTION_HISTORY:
        remove_count = len(seeds) - MAX_QUESTION_HISTORY
        seeds = seeds[remove_count:]
        index = max(0, index - remove_count)

    state = {
        "topic": state["topic"],
        "level": int(state["level"]),
        "seeds": seeds,
        "index": max(0, min(index, len(seeds) - 1)),
    }
    _set_query_param(QUESTION_STATE_PARAM, _encode_question_state(state))
    return state


def _fresh_seed(existing_seeds=()):
    """Create a new seed, avoiding exact seed reuse in the visible history."""
    used = set(int(seed) for seed in existing_seeds)
    for _ in range(100):
        seed = SEED_RNG.randrange(1, 2**63)
        if seed not in used:
            return seed
    return SEED_RNG.randrange(1, 2**63)


def _new_question_state(topic, level):
    return {
        "topic": topic,
        "level": int(level),
        "seeds": [_fresh_seed()],
        "index": 0,
    }


def build_problem_from_seed(topic, level, seed):
    """Regenerate exactly the same problem from a persistent random seed."""
    global RNG

    previous_rng = RNG
    RNG = random.Random(int(seed))
    try:
        problem = GENERATORS[topic](int(level))
    finally:
        RNG = previous_rng

    # A seed-based uid is stable across Streamlit reconnects, but different for
    # each generated question. It also keeps widget keys tied to the question.
    problem["uid"] = f"q_{TOPIC_ORDER.index(topic)}_{int(level)}_{int(seed)}"
    problem["seed"] = int(seed)
    return problem


render_header()

st.sidebar.markdown("## Mode")
mode = st.sidebar.radio(
    "Choose activity",
    ["Practice", "Self-test"],
    help=(
        "Practice gives formula clues and immediate feedback. "
        "Self-test hides the worked solution until it is revealed."
    ),
)

st.sidebar.markdown("---")
st.sidebar.markdown("## Practice setup")

# Restore the question after a Streamlit inactivity timeout/reconnect.
question_state = _decode_question_state(
    _query_param_value(QUESTION_STATE_PARAM)
)
if question_state is None:
    default_topic = TOPIC_ORDER[0]
    question_state = _save_question_state(
        _new_question_state(default_topic, TOPICS[default_topic][0])
    )

restored_topic = question_state["topic"]
topic = st.sidebar.radio(
    "Topic",
    TOPIC_ORDER,
    index=TOPIC_ORDER.index(restored_topic),
)

level_options = TOPICS[topic]
if topic == question_state["topic"] and question_state["level"] in level_options:
    restored_level = question_state["level"]
else:
    restored_level = level_options[0]

level = st.sidebar.selectbox(
    "Level",
    level_options,
    index=level_options.index(restored_level),
)
level = int(level)

# Changing topic/level starts a fresh history for that setup. This mirrors the
# old app behaviour while ensuring that the selected question itself persists.
if topic != question_state["topic"] or level != question_state["level"]:
    question_state = _save_question_state(_new_question_state(topic, level))

new_question = st.sidebar.button(
    "🎲 New random question",
    use_container_width=True,
)

previous_col, next_col = st.sidebar.columns(2)
previous_question = previous_col.button(
    "← Previous",
    use_container_width=True,
    disabled=question_state["index"] <= 0,
    help="Return to the previous generated question.",
)
next_question = next_col.button(
    "Next →",
    use_container_width=True,
    disabled=question_state["index"] >= len(question_state["seeds"]) - 1,
    help="Move forward again after using Previous.",
)

if previous_question:
    question_state["index"] -= 1
    question_state = _save_question_state(question_state)
elif next_question:
    question_state["index"] += 1
    question_state = _save_question_state(question_state)
elif new_question:
    # If the student went back first, a new question starts a new branch and the
    # old forward history is discarded, which is the usual browser-history model.
    question_state["seeds"] = question_state["seeds"][: question_state["index"] + 1]
    question_state["seeds"].append(
        _fresh_seed(question_state["seeds"])
    )
    question_state["index"] = len(question_state["seeds"]) - 1
    question_state = _save_question_state(question_state)

current_seed = question_state["seeds"][question_state["index"]]
problem = build_problem_from_seed(topic, level, current_seed)

st.sidebar.caption(f"Current problem: {problem['title']}")
st.sidebar.caption(
    f"Question {question_state['index'] + 1} of {len(question_state['seeds'])} in this history"
)
st.sidebar.caption(
    "The current question is saved in this page URL, so it can be restored after an inactivity timeout or reconnect."
)

st.sidebar.markdown("---")
st.sidebar.markdown("## Progress")
xp = st.session_state.get("xp", 0)

if xp >= 350:
    badge = "🏆 Maths Master"
elif xp >= 220:
    badge = "⭐ Strong Foundations"
elif xp >= 100:
    badge = "✅ Developing Well"
elif xp > 0:
    badge = "🛠 Getting Started"
else:
    badge = "None yet"

st.sidebar.markdown(f"**XP**  \n{xp}")
st.sidebar.markdown(f"**Badge**  \n{badge}")
st.sidebar.markdown(
    "<div class='skill'>New questions are kept in a short browser-linked history. "
    "Use Previous/Next to revisit them; the current question is also restored after a reconnect.</div>",
    unsafe_allow_html=True,
)

tabs = st.tabs(
    [
        "0. Learn the topic",
        "1. Problem",
        "2. Choose the method",
        "3. Solve and check",
        "4. Solution",
    ]
)

with tabs[0]:
    render_learning(topic)

with tabs[1]:
    left, right = st.columns([1.0, 1.05], gap="large")

    with left:
        st.markdown("## Current problem")
        st.caption(
            f"{topic} | Level {level} | Difficulty: {problem['difficulty']}"
        )
        st.markdown(f"### {problem['title']}")
        st.write(problem["prompt"])

        if problem.get("display_equation"):
            st.latex(problem["display_equation"])

        st.markdown(
            """
            <div class="hint">
              Work carefully and keep full calculator precision until the final line.
              Use the New random question button for another problem.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        if problem.get("diagram"):
            st.markdown("### Diagram")
            render_diagram(problem)
        else:
            st.markdown("### Suggested working method")
            st.markdown(
                """
                <div class="card">
                  1. Write the relevant formula or equation.<br>
                  2. Substitute the known values with signs.<br>
                  3. Rearrange carefully.<br>
                  4. Calculate.<br>
                  5. Check whether the answer is reasonable.
                </div>
                """,
                unsafe_allow_html=True,
            )

with tabs[2]:
    st.markdown("## Choose the method or formula")

    fq = problem["formula_question"]
    selected = st.radio(
        fq["question"],
        fq["options"],
        key=f"{problem['uid']}_formula",
    )

    if st.button(
        "Check method",
        key=f"{problem['uid']}_check_formula",
    ):
        if selected == fq["correct"]:
            st.success("Correct method.")
            if mode == "Practice":
                st.write(fq["explanation"])
            award_xp(f"{problem['uid']}_method", 10)
        else:
            st.warning("That is not the best method for this question.")
            if mode == "Practice":
                st.write(fq["explanation"])

with tabs[3]:
    st.markdown("## Solve and check")

    if mode == "Practice":
        st.markdown(
            """
            <div class="info">
              Enter the final numerical values. Signs and units matter.
              The detailed calculation appears in the Solution tab.
            </div>
            """,
            unsafe_allow_html=True,
        )

    entered = {}

    columns = st.columns(min(2, len(problem["answers"])))

    for index, answer in enumerate(problem["answers"]):
        with columns[index % len(columns)]:
            label = answer["label"]
            if answer["unit"]:
                label += f" ({answer['unit']})"

            entered[index] = st.number_input(
                label,
                value=0.0,
                format="%.5f",
                key=f"{problem['uid']}_answer_{index}",
            )

    if st.button(
        "Check final answer",
        type="primary",
        key=f"{problem['uid']}_check_answers",
    ):
        results = [
            close_enough(entered[i], answer)
            for i, answer in enumerate(problem["answers"])
        ]

        for result, answer in zip(results, problem["answers"]):
            if result:
                st.markdown(
                    f"<div class='good'>{answer['label']}: correct.</div>",
                    unsafe_allow_html=True,
                )
            else:
                if mode == "Practice":
                    expected = fmt(answer["value"], 5)
                    unit = f" {answer['unit']}" if answer["unit"] else ""
                    st.markdown(
                        f"<div class='bad'>{answer['label']}: check again. "
                        f"Expected approximately {expected}{unit}.</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<div class='bad'>{answer['label']}: check the formula, "
                        f"signs and arithmetic.</div>",
                        unsafe_allow_html=True,
                    )

        if all(results):
            st.success("Excellent — all answers are correct.")
            award_xp(f"{problem['uid']}_answers", 35)

    st.markdown("### Reasonableness check")
    st.write(problem["teaching_point"])

with tabs[4]:
    st.markdown("## Detailed worked solution")

    revealed = mode == "Practice" or st.session_state.get(
        f"reveal_{problem['uid']}",
        False,
    )

    if not revealed:
        st.warning(
            "The solution is hidden in Self-test mode until you finish your attempt."
        )
        if st.button(
            "Reveal worked solution",
            key=f"reveal_button_{problem['uid']}",
        ):
            st.session_state[f"reveal_{problem['uid']}"] = True
            st.rerun()
    else:
        st.markdown("### Original question")
        st.markdown(f"**{problem['title']}**")
        st.write(problem["prompt"])

        if problem.get("display_equation"):
            st.latex(problem["display_equation"])

        if problem.get("diagram"):
            render_diagram(problem)

        st.markdown("---")
        st.markdown("### Step-by-step solution")
        st.markdown(
            """
            <div class="info">
              Each line explains what is being done and why. Follow the same
              sequence in your written working.
            </div>
            """,
            unsafe_allow_html=True,
        )

        for kind, content in problem["solution"]:
            if kind == "latex":
                st.latex(content)
            else:
                st.write(content)

        st.markdown("### Final learning point")
        st.markdown(
            f"<div class='hint'>{problem['teaching_point']}</div>",
            unsafe_allow_html=True,
        )
