# ============================================================
# 🥇Tanara AI
# 10Alytics Global Hackathon 2026
# SDG 3: ASEAN Public Health Early-Warning System
# ============================================================

from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="🥇Tanara AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. FILE PATHS
# ============================================================

DATA_FILE = "asean_health_risk_readiness_priority.csv"
MODEL_FILE = "healthpulse_final_model.pkl"
FEATURE_FILE = "healthpulse_model_features.pkl"
LABEL_FILE = "healthpulse_risk_label_mapping.pkl"
LOGO_FILE = Path("assets/10alytics_logo.png")


# ============================================================
# 3. BRAND COLOURS
# ============================================================

BRAND = {
    "navy": "#071B3A",
    "blue": "#0B3C68",
    "cyan": "#00B4D8",
    "teal": "#00A896",
    "green": "#2E7D32",
    "gold": "#F9C74F",
    "orange": "#F8961E",
    "red": "#D62828",
    "purple": "#7C3AED",
    "light_blue": "#EEF8FF",
    "cream": "#F7F9FC",
    "white": "#FFFFFF",
    "muted": "#6B7280",
    "dark": "#111827",
    "border": "#E5E7EB",
}

RISK_COLORS = {
    "Low": BRAND["green"],
    "Medium": BRAND["gold"],
    "High": BRAND["red"],
}

READINESS_COLORS = {
    "Low": BRAND["red"],
    "Medium": BRAND["gold"],
    "High": BRAND["green"],
}

PRIORITY_COLORS = {
    "Emergency Priority": BRAND["red"],
    "High Priority": BRAND["orange"],
    "Preventive Priority": BRAND["gold"],
    "Monitor Closely": BRAND["cyan"],
    "Moderate Priority": BRAND["blue"],
    "Capacity Building Needed": BRAND["purple"],
    "Controlled Risk": BRAND["teal"],
    "Stable but Needs Support": "#60A5FA",
    "Stable / Resilient": BRAND["green"],
}


# ============================================================
# 4. CUSTOM CSS
# ============================================================

st.markdown(
    f"""
<style>
    .stApp {{
        background:
            radial-gradient(
                circle at 95% 5%,
                rgba(0,180,216,0.08),
                transparent 28%
            ),
            linear-gradient(
                180deg,
                #F7F9FC 0%,
                #FFFFFF 45%,
                #F7F9FC 100%
            );
    }}

    section[data-testid="stSidebar"] {{
        background: linear-gradient(
            180deg,
            {BRAND["navy"]} 0%,
            {BRAND["blue"]} 100%
        );
    }}

    section[data-testid="stSidebar"] * {{
        color: white;
    }}

    section[data-testid="stSidebar"]
    div[role="radiogroup"] label {{
        padding: 8px 4px;
    }}

    #MainMenu {{
        visibility: hidden;
    }}

    footer {{
        visibility: hidden;
    }}

    .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }}

    .hero {{
        padding: 36px 40px;
        border-radius: 26px;
        background: linear-gradient(
            135deg,
            {BRAND["navy"]} 0%,
            {BRAND["blue"]} 55%,
            {BRAND["cyan"]} 125%
        );
        color: white;
        margin-bottom: 28px;
        box-shadow: 0 18px 45px rgba(7,27,58,0.20);
    }}

    .hero h1 {{
        color: white;
        font-size: 48px;
        font-weight: 900;
        margin: 0;
        letter-spacing: -1.5px;
    }}

    .hero p {{
        color: #E5F7FF;
        font-size: 18px;
        line-height: 1.6;
        max-width: 1050px;
        margin-top: 12px;
        margin-bottom: 0;
    }}

    .hero-badge {{
        display: inline-block;
        padding: 8px 14px;
        border-radius: 999px;
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.28);
        color: white;
        font-size: 13px;
        font-weight: 800;
        margin-bottom: 14px;
    }}

    .metric-card {{
        background: white;
        padding: 22px;
        border-radius: 18px;
        border: 1px solid {BRAND["border"]};
        box-shadow: 0 10px 28px rgba(17,24,39,0.06);
        min-height: 126px;
        overflow-wrap: anywhere;
    }}

    .metric-label {{
        color: {BRAND["muted"]};
        font-size: 12px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.7px;
        margin-bottom: 8px;
    }}

    .metric-value {{
        color: {BRAND["navy"]};
        font-size: 31px;
        line-height: 1.15;
        font-weight: 900;
        margin-bottom: 6px;
    }}

    .metric-note {{
        color: {BRAND["muted"]};
        font-size: 13px;
    }}

    .section-title {{
        font-size: 27px;
        font-weight: 900;
        color: {BRAND["navy"]};
        margin-top: 14px;
        margin-bottom: 4px;
    }}

    .section-subtitle {{
        color: {BRAND["muted"]};
        font-size: 15px;
        margin-bottom: 18px;
    }}

    .pill {{
        display: inline-block;
        padding: 7px 12px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 800;
        margin-right: 6px;
        margin-top: 10px;
        margin-bottom: 10px;
    }}

    .pill-high {{
        background: rgba(214,40,40,0.12);
        color: {BRAND["red"]};
    }}

    .pill-medium {{
        background: rgba(249,199,79,0.24);
        color: #946200;
    }}

    .pill-low {{
        background: rgba(46,125,50,0.12);
        color: {BRAND["green"]};
    }}

    .recommendation-header {{
        padding: 18px 22px;
        border-radius: 16px;
        background: linear-gradient(
            135deg,
            #F5FCFF,
            #EAF7FF
        );
        border-left: 6px solid {BRAND["cyan"]};
        margin-bottom: 16px;
    }}

    .recommendation-header h3 {{
        margin: 0;
        color: {BRAND["navy"]};
        font-size: 24px;
    }}

    .method-number {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 34px;
        height: 34px;
        margin-right: 10px;
        border-radius: 50%;
        background: {BRAND["blue"]};
        color: white;
        font-weight: 900;
    }}

    div[data-testid="stDataFrame"] {{
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid {BRAND["border"]};
    }}

    div[data-testid="stMetric"] {{
        background: white;
        border: 1px solid {BRAND["border"]};
        padding: 16px;
        border-radius: 16px;
    }}

    /* Keep all normal page text clearly readable */
    .stApp [data-testid="stMarkdownContainer"] {{
        color: #0A1931;
    }}

    .stApp [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stMarkdownContainer"] li,
    .stApp [data-testid="stMarkdownContainer"] span {{
        color: inherit;
    }}

    .stApp label,
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stWidgetLabel"] {{
        color: #0A1931 !important;
        font-weight: 600;
    }}

    /* Prevent accidental browser selection from covering text in blue */
    .stApp ::selection {{
        background: transparent !important;
        color: inherit !important;
    }}

    .stApp ::-moz-selection {{
        background: transparent !important;
        color: inherit !important;
    }}

    /* Keep every sidebar navigation label bright and readable */
    section[data-testid="stSidebar"] div[role="radiogroup"] label,
    section[data-testid="stSidebar"] div[role="radiogroup"] label p,
    section[data-testid="stSidebar"] div[role="radiogroup"] label span,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
        color: #FFFFFF !important;
        opacity: 1 !important;
    }}

    @media (max-width: 900px) {{
        .hero h1 {{
            font-size: 36px;
        }}

        .hero {{
            padding: 28px 24px;
        }}

        .metric-value {{
            font-size: 25px;
        }}
    }}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 5. LOAD DATA AND MODEL
# ============================================================

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)

    df["year"] = pd.to_numeric(
        df["year"],
        errors="coerce",
    ).astype("Int64")

    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)

    for col in [
        "health_risk_level",
        "readiness_level",
        "priority_category",
    ]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df


@st.cache_resource
def load_model():
    final_model = joblib.load(MODEL_FILE)
    model_features = joblib.load(FEATURE_FILE)
    label_mapping = joblib.load(LABEL_FILE)

    return final_model, model_features, label_mapping


try:
    health_df = load_data()

    model, final_features, reverse_risk_mapping = (
        load_model()
    )

except FileNotFoundError as error:
    st.error(
        f"Required file not found: {error.filename}. "
        "Confirm that all CSV and PKL files are in "
        "the same project folder."
    )
    st.stop()

except Exception as error:
    st.error(f"Application loading failed: {error}")
    st.stop()


# ============================================================
# 6. INDICATOR GROUPS
# ============================================================

RISK_INDICATORS = [
    "infant_mortality_rate",
    "under_5_mortality_rate",
    "maternal_mortality_rate",
    "malaria_prevalence",
    "tb_prevalence",
    "undernourished_population",
]

READINESS_INDICATORS = [
    "government_health_expenditure",
    "dpt_immunization",
    "measles_immunization",
    "nurses_midwives_density",
    "physicians_density",
]

indicator_medians = health_df[
    final_features
].median(numeric_only=True)


# ============================================================
# 7. HELPER FUNCTIONS
# ============================================================

def render_hero() -> None:
    hero_html = (
        '<div class="hero">'
        '<div class="hero-badge">'
        '10Alytics Global Hackathon 2026 · SDG 3'
        '</div>'
        '<h1>🥇Tanara AI</h1>'
        '<p>'
        'An AI-powered public health early-warning and response system for ASEAN. '
        '🥇Tanara AI:combines health risk scoring, system readiness, machine '
        'learning and actionable recommendations to identify where intervention '
        'is needed before a health crisis escalates.'
        '</p>'
        '</div>'
    )

    st.markdown(
        hero_html,
        unsafe_allow_html=True,
    )


def section_header(
    title: str,
    subtitle: str = "",
) -> None:
    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True,
    )

    if subtitle:
        st.markdown(
            f'<div class="section-subtitle">{subtitle}</div>',
            unsafe_allow_html=True,
        )


def metric_card(
    label: str,
    value,
    note: str = "",
) -> None:
    st.markdown(
        f"""
<div class="metric-card">
    <div class="metric-label">{label}</div>
    <div class="metric-value">{value}</div>
    <div class="metric-note">{note}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def risk_badge(level: str) -> str:
    if level == "High":
        return (
            '<span class="pill pill-high">'
            "🔴 High Risk"
            "</span>"
        )

    if level == "Medium":
        return (
            '<span class="pill pill-medium">'
            "🟡 Medium Risk"
            "</span>"
        )

    return (
        '<span class="pill pill-low">'
        "🟢 Low Risk"
        "</span>"
    )


def readiness_badge(level: str) -> str:
    if level == "High":
        return (
            '<span class="pill pill-low">'
            "🟢 High Readiness"
            "</span>"
        )

    if level == "Medium":
        return (
            '<span class="pill pill-medium">'
            "🟡 Medium Readiness"
            "</span>"
        )

    return (
        '<span class="pill pill-high">'
        "🔴 Low Readiness"
        "</span>"
    )


def readable_name(column: str) -> str:
    custom_names = {
        "dpt_immunization": "DPT immunization",
        "tb_prevalence": "TB prevalence",
        "under_5_mortality_rate":
            "Under-5 mortality rate",
    }

    return custom_names.get(
        column,
        column.replace("_", " ").capitalize(),
    )


def style_plotly(
    fig,
    title: str,
    height: int = 500,
    legend_position: str = "top",
):
    if legend_position == "top":
        legend_settings = {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "title": None,
        }

    elif legend_position == "bottom":
        legend_settings = {
            "orientation": "h",
            "yanchor": "top",
            "y": -0.18,
            "xanchor": "center",
            "x": 0.5,
            "title": None,
        }

    else:
        legend_settings = {
            "orientation": "v",
            "yanchor": "top",
            "y": 1,
            "xanchor": "left",
            "x": 1.02,
            "title": None,
        }

    fig.update_layout(
        height=height,
        title={
            "text": title,
            "font": {
                "size": 21,
                "color": BRAND["navy"],
                "family": "Arial Black",
            },
            "x": 0.01,
            "xanchor": "left",
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={
            "family": "Arial",
            "color": "#0A1931",
            "size": 14,
        },
        margin={
            "l": 30,
            "r": 35,
            "t": (
                105
                if legend_position == "top"
                else 75
            ),
            "b": (
                80
                if legend_position == "bottom"
                else 45
            ),
        },
        legend={
            **legend_settings,
            "font": {"color": "#0A1931", "size": 13},
        },
        hoverlabel={
            "bgcolor": "white",
            "font_size": 13,
        },
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        title_font={"size": 14, "color": "#0A1931"},
        tickfont={"size": 13, "color": "#0A1931"},
        linecolor="rgba(10,25,49,0.35)",
    )

    fig.update_yaxes(
        gridcolor="rgba(107,114,128,0.18)",
        zeroline=False,
        title_font={"size": 14, "color": "#0A1931"},
        tickfont={"size": 13, "color": "#0A1931"},
        linecolor="rgba(10,25,49,0.35)",
    )

    fig.update_annotations(
        font={"color": "#0A1931", "size": 12}
    )

    return fig


def predict_next_risk(
    selected_row: pd.DataFrame,
):
    prediction_data = selected_row[
        final_features
    ].copy()

    predicted_code = model.predict(
        prediction_data
    )[0]

    predicted_level = reverse_risk_mapping[
        int(predicted_code)
    ]

    predicted_probabilities = (
        model.predict_proba(prediction_data)[0]
    )

    class_names = [
        reverse_risk_mapping[int(class_code)]
        for class_code in model.classes_
    ]

    probability_dictionary = {
        class_names[index]:
            float(predicted_probabilities[index])
        for index in range(len(class_names))
    }

    return predicted_level, probability_dictionary


def identify_concerns(
    row: pd.Series,
    top_n: int = 5,
):
    concerns = []

    for indicator in RISK_INDICATORS:
        value = row[indicator]
        median_value = indicator_medians[indicator]

        if value > median_value:
            if median_value != 0:
                relative_gap = (
                    value - median_value
                ) / abs(median_value)
            else:
                relative_gap = (
                    value - median_value
                )

            concerns.append(
                {
                    "indicator": indicator,
                    "message": (
                        f"{readable_name(indicator)} "
                        "is above the ASEAN median."
                    ),
                    "severity": relative_gap,
                }
            )

    for indicator in READINESS_INDICATORS:
        value = row[indicator]
        median_value = indicator_medians[indicator]

        if value < median_value:
            if median_value != 0:
                relative_gap = (
                    median_value - value
                ) / abs(median_value)
            else:
                relative_gap = (
                    median_value - value
                )

            concerns.append(
                {
                    "indicator": indicator,
                    "message": (
                        f"{readable_name(indicator)} "
                        "is below the ASEAN median."
                    ),
                    "severity": relative_gap,
                }
            )

    concerns = sorted(
        concerns,
        key=lambda item: item["severity"],
        reverse=True,
    )

    return concerns[:top_n]


def generate_actions(concerns):
    indicators = {
        concern["indicator"]
        for concern in concerns
    }

    actions = []

    if {
        "under_5_mortality_rate",
        "infant_mortality_rate",
    } & indicators:
        actions.append(
            "Expand child health services, immunization "
            "outreach, early disease screening and "
            "community child-survival programmes."
        )

    if "maternal_mortality_rate" in indicators:
        actions.append(
            "Strengthen emergency obstetric care, "
            "skilled birth attendance, maternal referral "
            "systems and midwife-led services."
        )

    if "malaria_prevalence" in indicators:
        actions.append(
            "Expand malaria surveillance, rapid testing, "
            "vector control and community prevention campaigns."
        )

    if "tb_prevalence" in indicators:
        actions.append(
            "Scale up TB screening, early case detection, "
            "treatment-adherence support and community monitoring."
        )

    if "undernourished_population" in indicators:
        actions.append(
            "Integrate nutrition support with maternal "
            "and child health services in vulnerable communities."
        )

    if {
        "dpt_immunization",
        "measles_immunization",
    } & indicators:
        actions.append(
            "Launch targeted vaccine catch-up campaigns "
            "and strengthen routine immunization delivery."
        )

    if "nurses_midwives_density" in indicators:
        actions.append(
            "Increase the deployment and retention of "
            "nurses and midwives in underserved and rural facilities."
        )

    if "physicians_density" in indicators:
        actions.append(
            "Improve physician access using mobile clinics, "
            "telemedicine and rural-service incentives."
        )

    if "government_health_expenditure" in indicators:
        actions.append(
            "Increase targeted public health financing "
            "for frontline services and high-burden priorities."
        )

    if not actions:
        actions.append(
            "Maintain current health investments, strengthen "
            "surveillance and continue monitoring early-warning indicators."
        )

    return actions


def render_recommendation(
    row: pd.Series,
    predicted_level: str,
    probability_dictionary: dict,
) -> None:
    concerns = identify_concerns(
        row,
        top_n=5,
    )

    actions = generate_actions(concerns)

    probability_high = probability_dictionary.get(
        "High",
        0.0,
    )

    st.markdown(
        """
<div class="recommendation-header">
    <h3>AI Public Health Recommendation</h3>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        summary_col1, summary_col2 = st.columns(2)

        with summary_col1:
            st.markdown(
                f"**Country assessed:** {row['country']}"
            )
            st.markdown(
                f"**Assessment year:** {int(row['year'])}"
            )

        with summary_col2:
            st.markdown(
                f"**Predicted next-year risk:** "
                f"{predicted_level}"
            )
            st.markdown(
                f"**Probability of high risk:** "
                f"{probability_high:.1%}"
            )

        st.divider()

        st.markdown("#### Main risk signals")

        if concerns:
            for concern in concerns:
                st.markdown(
                    f"- {concern['message']}"
                )
        else:
            st.success(
                "No major indicator exceeded the current "
                "ASEAN risk and readiness reference thresholds."
            )

        st.markdown("#### Recommended actions")

        for action_number, action in enumerate(
            actions,
            start=1,
        ):
            st.markdown(
                f"{action_number}. {action}"
            )


def render_method_step(
    number: int,
    title: str,
    explanation: str,
) -> None:
    with st.container(border=True):
        st.markdown(
            f"""
<span class="method-number">{number}</span>
<span style="
    color:{BRAND['navy']};
    font-size:22px;
    font-weight:900;
">{title}</span>
""",
            unsafe_allow_html=True,
        )

        st.markdown(explanation)


# ============================================================
# 8. SIDEBAR
# ============================================================

with st.sidebar:
    if LOGO_FILE.exists():
        st.image(
            str(LOGO_FILE),
            use_container_width=True,
        )

    else:
        st.markdown(
            """
<div style="font-size:30px;font-weight:900;">
    10Alytics
</div>

<div style="
    font-size:13px;
    opacity:0.85;
    margin-bottom:18px;
">
    Global Hackathon 2026
</div>
""",
            unsafe_allow_html=True,
        )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Executive Overview",
            "Risk Intelligence",
            "Readiness & Priority",
            "AI Prediction Copilot",
            "Country Deep Dive",
            "Methodology",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.caption("🥇Tanara AI")
    st.caption("SDG 3 · ASEAN Public Health")


# ============================================================
# 9. EXECUTIVE OVERVIEW
# ============================================================

if page == "Executive Overview":
    render_hero()

    latest_year = int(
        health_df["year"].max()
    )

    latest_df = health_df[
        health_df["year"] == latest_year
    ].copy()

    average_risk = (
        latest_df["health_risk_score"].mean()
    )

    emergency_count = (
        latest_df["priority_category"]
        == "Emergency Priority"
    ).sum()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card(
            "Countries covered",
            health_df["country"].nunique(),
            "ASEAN member countries",
        )

    with col2:
        metric_card(
            "Historical period",
            (
                f"{health_df['year'].min()}"
                f"–{health_df['year'].max()}"
            ),
            "Country-year panel dataset",
        )

    with col3:
        metric_card(
            "Average risk",
            f"{average_risk:.1f}",
            f"Regional score in {latest_year}",
        )

    with col4:
        metric_card(
            "Emergency cases",
            int(emergency_count),
            (
                "Emergency-priority countries "
                f"in {latest_year}"
            ),
        )

    st.write("")

    section_header(
        "ASEAN Health Risk Snapshot",
        (
            "Countries ranked from the highest to "
            f"the lowest Health Risk Score in {latest_year}."
        ),
    )

    ranked_latest = latest_df.sort_values(
        "health_risk_score",
        ascending=True,
    )

    risk_bar = px.bar(
        ranked_latest,
        x="health_risk_score",
        y="country",
        orientation="h",
        color="health_risk_level",
        color_discrete_map=RISK_COLORS,
        text="health_risk_score",
        category_orders={
            "country":
                ranked_latest["country"].tolist(),
            "health_risk_level":
                ["High", "Medium", "Low"],
        },
        labels={
            "health_risk_score":
                "Health risk score",
            "country": "",
            "health_risk_level":
                "Risk level",
        },
    )

    risk_bar.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside",
        cliponaxis=False,
        marker_line_width=0,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Health risk score: %{x:.1f}"
            "<extra></extra>"
        ),
    )

    risk_bar.update_xaxes(
        title="Health risk score",
        range=[
            0,
            (
                ranked_latest[
                    "health_risk_score"
                ].max()
                * 1.16
            ),
        ],
    )

    risk_bar.update_yaxes(
        title=None,
        autorange="reversed",
    )

    risk_bar = style_plotly(
        risk_bar,
        (
            "Health Risk Scores Across "
            f"ASEAN Countries, {latest_year}"
        ),
        height=570,
        legend_position="top",
    )

    st.plotly_chart(
        risk_bar,
        use_container_width=True,
        config={"displayModeBar": False},
    )

    section_header(
        "Regional Risk Category Trend",
        (
            "Share of ASEAN countries classified "
            "as Low, Medium or High risk over time."
        ),
    )

    risk_share = (
        health_df
        .groupby(
            [
                "year",
                "health_risk_level",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={"size": "count"}
        )
    )

    risk_share["share"] = (
        risk_share
        .groupby("year")["count"]
        .transform(
            lambda values:
                values / values.sum() * 100
        )
    )

    area_chart = px.area(
        risk_share,
        x="year",
        y="share",
        color="health_risk_level",
        color_discrete_map=RISK_COLORS,
        category_orders={
            "health_risk_level":
                ["High", "Medium", "Low"]
        },
        labels={
            "year": "Year",
            "share":
                "Share of countries (%)",
            "health_risk_level":
                "Risk level",
        },
    )

    area_chart.update_traces(
        hovertemplate=(
            "<b>%{fullData.name}</b><br>"
            "Year: %{x}<br>"
            "Share: %{y:.1f}%"
            "<extra></extra>"
        )
    )

    area_chart = style_plotly(
        area_chart,
        (
            "Health Risk Levels Across "
            f"ASEAN, {health_df['year'].min()}–{health_df['year'].max()}"
        ),
        height=480,
        legend_position="top",
    )

    area_chart.update_yaxes(
        range=[0, 100],
        ticksuffix="%",
    )

    area_chart.update_xaxes(
        dtick=1,
        tickmode="linear",
    )

    st.plotly_chart(
        area_chart,
        use_container_width=True,
        config={"displayModeBar": False},
    )


# ============================================================
# 10. RISK INTELLIGENCE
# ============================================================

elif page == "Risk Intelligence":
    render_hero()

    section_header(
        "Risk Intelligence",
        (
            "Explore countries with the greatest health "
            "burden and compare changes across time."
        ),
    )

    selected_year = st.selectbox(
        "Select year",
        sorted(
            health_df["year"].unique()
        ),
        index=(
            len(
                health_df["year"].unique()
            )
            - 1
        ),
    )

    year_df = health_df[
        health_df["year"] == selected_year
    ].copy()

    left_column, right_column = st.columns(
        [1.45, 0.85],
        gap="large",
    )

    with left_column:
        sorted_year_df = year_df.sort_values(
            "health_risk_score",
            ascending=True,
        )

        year_bar = px.bar(
            sorted_year_df,
            x="health_risk_score",
            y="country",
            orientation="h",
            color="health_risk_level",
            color_discrete_map=RISK_COLORS,
            text="health_risk_score",
            category_orders={
                "country":
                    sorted_year_df[
                        "country"
                    ].tolist(),
                "health_risk_level":
                    ["High", "Medium", "Low"],
            },
            labels={
                "health_risk_score":
                    "Health risk score",
                "country": "",
                "health_risk_level":
                    "Risk level",
            },
        )

        year_bar.update_traces(
            texttemplate="%{text:.1f}",
            textposition="outside",
            cliponaxis=False,
            marker_line_width=0,
        )

        year_bar.update_xaxes(
            range=[
                0,
                (
                    sorted_year_df[
                        "health_risk_score"
                    ].max()
                    * 1.18
                ),
            ]
        )

        year_bar.update_yaxes(
            title=None,
            autorange="reversed",
        )

        year_bar = style_plotly(
            year_bar,
            (
                "Health Risk Scores by "
                f"Country, {selected_year}"
            ),
            height=565,
            legend_position="top",
        )

        st.plotly_chart(
            year_bar,
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with right_column:
        risk_counts = (
            year_df["health_risk_level"]
            .value_counts()
            .reindex(
                ["High", "Medium", "Low"],
                fill_value=0,
            )
            .rename_axis("risk_level")
            .reset_index(name="count")
        )

        risk_counts["percentage"] = (
            risk_counts["count"]
            / risk_counts["count"].sum()
            * 100
        )

        risk_donut = px.pie(
            risk_counts,
            names="risk_level",
            values="count",
            hole=0.60,
            color="risk_level",
            color_discrete_map=RISK_COLORS,
            category_orders={
                "risk_level":
                    ["High", "Medium", "Low"]
            },
        )

        risk_donut.update_traces(
            textposition="inside",
            texttemplate="%{percent:.0%}",
            textfont={
                "size": 17,
                "color": "white",
                "family": "Arial Black",
            },
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Share: %{percent:.1%}<br>"
                "Countries: %{value}"
                "<extra></extra>"
            ),
        )

        risk_donut.add_annotation(
            text=(
                f"<b>{len(year_df)}</b>"
                "<br>countries"
            ),
            x=0.5,
            y=0.5,
            showarrow=False,
            font={
                "size": 16,
                "color": BRAND["navy"],
            },
        )

        risk_donut = style_plotly(
            risk_donut,
            (
                "Risk Category Mix, "
                f"{selected_year}"
            ),
            height=565,
            legend_position="bottom",
        )

        st.plotly_chart(
            risk_donut,
            use_container_width=True,
            config={"displayModeBar": False},
        )

    section_header(
        "Country Risk Trends",
        (
            "Select countries to compare their "
            "Health Risk Scores over time."
        ),
    )

    country_options = sorted(
        health_df["country"].unique()
    )

    default_countries = country_options[:4]

    selected_countries = st.multiselect(
        "Countries to compare",
        options=country_options,
        default=default_countries,
    )

    if selected_countries:
        trend_df = health_df[
            health_df["country"].isin(
                selected_countries
            )
        ].copy()

        risk_line = px.line(
            trend_df,
            x="year",
            y="health_risk_score",
            color="country",
            markers=True,
            labels={
                "year": "Year",
                "health_risk_score":
                    "Health risk score",
                "country": "Country",
            },
        )

        risk_line.update_traces(
            line={"width": 3},
            marker={"size": 7},
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Year: %{x}<br>"
                "Risk score: %{y:.1f}"
                "<extra></extra>"
            ),
        )

        risk_line.update_xaxes(dtick=1)

        risk_line = style_plotly(
            risk_line,
            "Health Risk Trend by Country",
            height=520,
            legend_position="top",
        )

        st.plotly_chart(
            risk_line,
            use_container_width=True,
            config={"displayModeBar": False},
        )

    else:
        st.info(
            "Select at least one country "
            "to display the trend chart."
        )


# ============================================================
# 11. READINESS AND PRIORITY
# ============================================================

elif page == "Readiness & Priority":
    render_hero()

    section_header(
        "Readiness and Priority Matrix",
        (
            "Identify countries combining high health "
            "burden with weak health-system capacity."
        ),
    )

    selected_year = st.selectbox(
        "Select year",
        sorted(
            health_df["year"].unique()
        ),
        index=(
            len(
                health_df["year"].unique()
            )
            - 1
        ),
    )

    matrix_df = health_df[
        health_df["year"] == selected_year
    ].copy()

    left_column, right_column = st.columns(
        [1.25, 0.95],
        gap="large",
    )

    with left_column:
        matrix_chart = px.scatter(
            matrix_df,
            x="readiness_score",
            y="health_risk_score",
            color="priority_category",
            size="health_risk_score",
            text="country",
            hover_name="country",
            color_discrete_map=PRIORITY_COLORS,
            size_max=35,
            labels={
                "readiness_score":
                    "Health-system readiness score",
                "health_risk_score":
                    "Health risk score",
                "priority_category":
                    "Priority category",
            },
        )

        matrix_chart.update_traces(
            textposition="top center",
            textfont={"size": 11, "color": "#0A1931"},
            marker={
                "line": {
                    "width": 1.2,
                    "color": "white",
                }
            },
            hovertemplate=(
                "<b>%{hovertext}</b><br>"
                "Readiness: %{x:.1f}<br>"
                "Health risk: %{y:.1f}"
                "<extra></extra>"
            ),
        )

        matrix_chart.add_vline(
            x=(
                matrix_df[
                    "readiness_score"
                ].median()
            ),
            line_dash="dash",
            line_color=BRAND["muted"],
            annotation_text="Median readiness",
            annotation_position="top",
        )

        matrix_chart.add_hline(
            y=(
                matrix_df[
                    "health_risk_score"
                ].median()
            ),
            line_dash="dash",
            line_color=BRAND["muted"],
            annotation_text="Median risk",
            annotation_position="right",
        )

        matrix_chart = style_plotly(
            matrix_chart,
            (
                "Risk–Readiness Priority "
                f"Matrix, {selected_year}"
            ),
            height=600,
            legend_position="bottom",
        )

        st.plotly_chart(
            matrix_chart,
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with right_column:
        priority_counts = (
            matrix_df["priority_category"]
            .value_counts()
            .rename_axis(
                "priority_category"
            )
            .reset_index(name="count")
            .sort_values(
                "count",
                ascending=True,
            )
        )

        priority_bar = px.bar(
            priority_counts,
            x="count",
            y="priority_category",
            orientation="h",
            color="priority_category",
            color_discrete_map=PRIORITY_COLORS,
            text="count",
            category_orders={
                "priority_category":
                    priority_counts[
                        "priority_category"
                    ].tolist()
            },
            labels={
                "count":
                    "Number of countries",
                "priority_category": "",
            },
        )

        priority_bar.update_traces(
            textposition="outside",
            cliponaxis=False,
            marker_line_width=0,
        )

        priority_bar.update_xaxes(
            range=[
                0,
                max(
                    (
                        priority_counts[
                            "count"
                        ].max()
                        * 1.35
                    ),
                    1,
                ),
            ]
        )

        priority_bar.update_yaxes(
            title=None,
            autorange="reversed",
        )

        priority_bar = style_plotly(
            priority_bar,
            "Priority Category Distribution",
            height=600,
            legend_position="none",
        )

        priority_bar.update_layout(
            showlegend=False
        )

        st.plotly_chart(
            priority_bar,
            use_container_width=True,
            config={"displayModeBar": False},
        )

    section_header(
        "Priority Table",
        (
            "Country results sorted from higher "
            "risk to weaker readiness."
        ),
    )

    priority_table = matrix_df[
        [
            "country",
            "health_risk_score",
            "health_risk_level",
            "readiness_score",
            "readiness_level",
            "priority_category",
        ]
    ].sort_values(
        by=[
            "health_risk_score",
            "readiness_score",
        ],
        ascending=[False, True],
    )

    priority_table = priority_table.rename(
        columns={
            "country": "Country",
            "health_risk_score":
                "Risk score",
            "health_risk_level":
                "Risk level",
            "readiness_score":
                "Readiness score",
            "readiness_level":
                "Readiness level",
            "priority_category":
                "Priority category",
        }
    )

    st.dataframe(
        priority_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Risk score":
                st.column_config.NumberColumn(
                    format="%.1f"
                ),
            "Readiness score":
                st.column_config.NumberColumn(
                    format="%.1f"
                ),
        },
    )


# ============================================================
# 12. AI PREDICTION COPILOT
# ============================================================

elif page == "AI Prediction Copilot":
    render_hero()

    section_header(
        "AI Prediction Copilot",
        (
            "Use current-year health indicators to "
            "predict the following year's risk level "
            "and recommended response."
        ),
    )

    selection_column, result_column = st.columns(
        [0.75, 1.25],
        gap="large",
    )

    with selection_column:
        with st.container(border=True):
            st.markdown(
                "### Select an assessment"
            )

            selected_country = st.selectbox(
                "Country",
                sorted(
                    health_df[
                        "country"
                    ].unique()
                ),
            )

            country_df = health_df[
                health_df["country"]
                == selected_country
            ].copy()

            selected_year = st.selectbox(
                "Assessment year",
                sorted(
                    country_df[
                        "year"
                    ].unique()
                ),
            )

            selected_row = country_df[
                country_df["year"]
                == selected_year
            ].copy()

            st.caption(
                "The model uses the selected year's "
                "health indicators to estimate the "
                "following year's risk category."
            )

    if selected_row.empty:
        st.warning(
            "No data is available for the "
            "selected country and year."
        )

    else:
        (
            predicted_level,
            probability_dictionary,
        ) = predict_next_risk(
            selected_row
        )

        selected_series = (
            selected_row.iloc[0]
        )

        with result_column:
            (
                result_col1,
                result_col2,
                result_col3,
            ) = st.columns(3)

            with result_col1:
                metric_card(
                    "Current risk",
                    selected_series[
                        "health_risk_level"
                    ],
                    (
                        "Score: "
                        f"{selected_series['health_risk_score']:.1f}"
                    ),
                )

            with result_col2:
                metric_card(
                    "Current readiness",
                    selected_series[
                        "readiness_level"
                    ],
                    (
                        "Score: "
                        f"{selected_series['readiness_score']:.1f}"
                    ),
                )

            with result_col3:
                metric_card(
                    "Predicted risk",
                    predicted_level,
                    "Following-year classification",
                )

            st.markdown(
                (
                    risk_badge(
                        selected_series[
                            "health_risk_level"
                        ]
                    )
                    + readiness_badge(
                        selected_series[
                            "readiness_level"
                        ]
                    )
                ),
                unsafe_allow_html=True,
            )

        probability_df = pd.DataFrame(
            {
                "Risk level": [
                    "Low",
                    "Medium",
                    "High",
                ],
                "Probability": [
                    probability_dictionary.get(
                        "Low",
                        0,
                    ),
                    probability_dictionary.get(
                        "Medium",
                        0,
                    ),
                    probability_dictionary.get(
                        "High",
                        0,
                    ),
                ],
            }
        )

        probability_chart = px.bar(
            probability_df,
            x="Risk level",
            y="Probability",
            color="Risk level",
            color_discrete_map=RISK_COLORS,
            text="Probability",
            category_orders={
                "Risk level": [
                    "Low",
                    "Medium",
                    "High",
                ]
            },
            labels={
                "Risk level": "",
                "Probability":
                    "Prediction probability",
            },
        )

        probability_chart.update_traces(
            texttemplate="%{text:.1%}",
            textposition="outside",
            cliponaxis=False,
            marker_line_width=0,
        )

        probability_chart.update_yaxes(
            tickformat=".0%",
            range=[0, 1.12],
        )

        probability_chart = style_plotly(
            probability_chart,
            (
                "Next-Year Risk "
                "Prediction Probability"
            ),
            height=430,
            legend_position="none",
        )

        probability_chart.update_layout(
            showlegend=False
        )

        st.plotly_chart(
            probability_chart,
            use_container_width=True,
            config={"displayModeBar": False},
        )

        render_recommendation(
            selected_series,
            predicted_level,
            probability_dictionary,
        )


# ============================================================
# 13. COUNTRY DEEP DIVE
# ============================================================

elif page == "Country Deep Dive":
    render_hero()

    section_header(
        "Country Deep Dive",
        (
            "Explore one country's health risk, "
            "readiness and priority pathway across "
            "the historical period."
        ),
    )

    selected_country = st.selectbox(
        "Select country",
        sorted(
            health_df["country"].unique()
        ),
    )

    country_df = (
        health_df[
            health_df["country"]
            == selected_country
        ]
        .sort_values("year")
        .copy()
    )

    latest_row = country_df.iloc[-1]

    col1, col2, col3 = st.columns(3)

    with col1:
        metric_card(
            "Latest risk score",
            (
                f"{latest_row['health_risk_score']:.1f}"
            ),
            latest_row[
                "health_risk_level"
            ],
        )

    with col2:
        metric_card(
            "Latest readiness",
            (
                f"{latest_row['readiness_score']:.1f}"
            ),
            latest_row[
                "readiness_level"
            ],
        )

    with col3:
        metric_card(
            "Latest priority",
            latest_row[
                "priority_category"
            ],
            (
                "Assessment year: "
                f"{latest_row['year']}"
            ),
        )

    trend_chart = go.Figure()

    trend_chart.add_trace(
        go.Scatter(
            x=country_df["year"],
            y=country_df[
                "health_risk_score"
            ],
            mode="lines+markers",
            name="Health risk score",
            line={
                "color": BRAND["red"],
                "width": 4,
            },
            marker={
                "size": 8,
                "color": BRAND["red"],
            },
            hovertemplate=(
                "Year: %{x}<br>"
                "Risk score: %{y:.1f}"
                "<extra></extra>"
            ),
        )
    )

    trend_chart.add_trace(
        go.Scatter(
            x=country_df["year"],
            y=country_df[
                "readiness_score"
            ],
            mode="lines+markers",
            name="Readiness score",
            line={
                "color": BRAND["teal"],
                "width": 4,
            },
            marker={
                "size": 8,
                "color": BRAND["teal"],
            },
            hovertemplate=(
                "Year: %{x}<br>"
                "Readiness score: %{y:.1f}"
                "<extra></extra>"
            ),
        )
    )

    trend_chart.update_xaxes(
        title="Year",
        dtick=1,
    )

    trend_chart.update_yaxes(
        title="Score",
    )

    trend_chart = style_plotly(
        trend_chart,
        (
            "Health Risk and Readiness "
            f"Trend: {selected_country}"
        ),
        height=530,
        legend_position="top",
    )

    st.plotly_chart(
        trend_chart,
        use_container_width=True,
        config={"displayModeBar": False},
    )

    section_header(
        "Country-Year Records",
        (
            "Historical score and "
            "priority classification."
        ),
    )

    country_table = country_df[
        [
            "year",
            "health_risk_score",
            "health_risk_level",
            "readiness_score",
            "readiness_level",
            "priority_category",
        ]
    ].rename(
        columns={
            "year": "Year",
            "health_risk_score":
                "Risk score",
            "health_risk_level":
                "Risk level",
            "readiness_score":
                "Readiness score",
            "readiness_level":
                "Readiness level",
            "priority_category":
                "Priority category",
        }
    )

    st.dataframe(
        country_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Risk score":
                st.column_config.NumberColumn(
                    format="%.1f"
                ),
            "Readiness score":
                st.column_config.NumberColumn(
                    format="%.1f"
                ),
        },
    )


# ============================================================
# 14. METHODOLOGY
# ============================================================

elif page == "Methodology":
    render_hero()

    section_header(
        "Methodology",
        (
            "How 🥇VitaGuard AI converts fragmented "
            "indicators into decision intelligence."
        ),
    )

    render_method_step(
        1,
        "Health Risk Index",
        """
The **Health Risk Index** measures direct public-health burden using:

- Infant mortality rate
- Under-5 mortality rate
- Maternal mortality rate
- Malaria prevalence
- Tuberculosis prevalence
- Undernourished population

Each indicator was normalised to a common scale. The indicators were
combined into a score from **0 to 100** and classified as **Low,
Medium or High risk**.
""",
    )

    render_method_step(
        2,
        "Health System Readiness Score",
        """
The **Readiness Score** measures the capacity of a health system to
prevent and respond to health challenges. It uses:

- Government health expenditure
- DPT immunisation
- Measles immunisation
- Nurses and midwives density
- Physician density

A higher score represents stronger prevention and response capacity.
""",
    )

    render_method_step(
        3,
        "Risk–Readiness Priority Matrix",
        """
The risk and readiness dimensions are analysed separately and then
combined.

This identifies country-year observations such as:

- **High risk + Low readiness:** Emergency Priority
- **High risk + Medium readiness:** High Priority
- **Medium risk + Low readiness:** Preventive Priority
- **Low risk + High readiness:** Stable / Resilient

The matrix helps determine where limited resources should be deployed first.
""",
    )

    render_method_step(
        4,
        "Machine Learning Early-Warning Model",
        """
The machine-learning dataset was structured so that indicators from one
year predict the **following year's Health Risk Level**.

For example:

- 2004 indicators predict the 2005 risk level
- 2005 indicators predict the 2006 risk level
- 2024 indicators predict the 2025 risk level

The selected raw-indicator Logistic Regression model achieved strong
time-based validation performance while retaining interpretability.
""",
    )

    render_method_step(
        5,
        "AI Recommendation Layer",
        """
After predicting the next-year risk level, the recommendation layer:

1. Compares the selected country's indicators with ASEAN reference medians.
2. Identifies mortality, disease, nutrition and readiness gaps.
3. Converts those gaps into targeted public-health actions.
4. Presents the result in language suitable for policy and resource-allocation decisions.

The recommendations support human decision-makers; they do not replace
clinical or governmental judgement.
""",
    )

    render_method_step(
        6,
        "Data Limitation",
        """
The dataset covers **10 ASEAN countries from 2004 to 2025**. The original cleaned data covered 2004–2014, while 2015–2025 values are projected estimates for prototype analysis.
Some missing values were estimated using within-country interpolation and
median imputation. The `underweight_children` variable was excluded from
the final model because of its high proportion of missing observations.

Future versions should incorporate more recent data, subnational
information, climate indicators and real-time surveillance feeds.
""",
    )