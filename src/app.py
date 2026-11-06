# Askar Turanazarov

# src/app.py
# Streamlit UI for the flight delay risk model (CatBoost pipeline + SHAP + economic impact).
# Run from the project root:  streamlit run src/app.py
# UI languages: English (default), Uzbek, Russian - auto-selected from the browser locale.

import os
import sys
import base64
import calendar
from datetime import date

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.analytics import FlightAnalytics            # noqa: E402
from src.model_loader import (WEATHER_VALUES, available_variants, default_variant,  # noqa: E402
                              load_calibration, resolve_model_path, uses_weather)
from src.i18n import LANGS, SESSION_KEY, t, detect_language  # noqa: E402

LOOKUP_PATH = os.path.join(BASE_DIR, "models", "lookup_data.joblib")
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGOS_DIR = os.path.join(BASE_DIR, "logos")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
DATA_YEAR = 2015  # the dataset covers US flights of 2015
MODEL_KEY = "model"  # session state key and URL parameter (?model=weather|compact|full)

# Metrics of the three variants. Each comes from its own test, so rows compare with care: the diploma models
# were checked on a random 20 % of flights from dates they had seen in training (test="random"), the weather
# model on 73 whole days it never saw (test="days"); auc_new is the comparable row. Sources of truth:
# reports/metrics/model_results.txt + recall085_results.txt (full), compact_model_results.txt (compact),
# weather_model_results.txt (weather, plus auc_new of the diploma models).
MODEL_METRICS = {
    "weather": dict(test="days", auc="0.698", auc_new="0.698", thr="0.20", recall="0.54", precision="0.31",
                    size="40 MB", ram="0.4 GB", ctr="2", script="—",
                    band_delay=(0.083, 0.161, 0.254, 0.485)),  # actual delay share per risk band, 73 hidden days
    "compact": dict(test="random", auc="0.779", auc_new="0.637", thr="0.37", recall="0.82", precision="0.30",
                    size="114 MB", ram="0.6 GB", ctr="3", script="train_compact.py"),
    "full":    dict(test="random", auc="0.786", auc_new="0.630", thr="0.37", recall="0.82", precision="0.31",
                    size="433 MB", ram="1.5 GB", ctr="4", script="train_final.py"),
}

AUTHOR = "Askar Turanazarov"
GITHUB_URL = "https://github.com/Askar-Turanazarov/FlightDelay-Service"
HF_URL = "https://huggingface.co/AskarT/flight-delay-catboost"
PORTFOLIO_URL = os.getenv("PORTFOLIO_URL", "")  # shown in the sidebar footer once the portfolio site is live

st.set_page_config(
    page_title="Flight Delay Risk Analytics System", page_icon="✈️", layout="wide",
    menu_items={
        "Get help": GITHUB_URL,
        "Report a bug": GITHUB_URL + "/issues",
        "About": (
            "**Flight Delay Risk Analytics System**  \n"
            "CatBoost + SHAP + FAA cost model, Streamlit UI (EN / UZ / RU).  \n"
            f"© 2026 {AUTHOR} · [GitHub]({GITHUB_URL}) · [Hugging Face]({HF_URL})"
        ),
    },
)

# Streamlit has no config option for the top padding (the header bar is 60 px and opaque, so the
# page content cannot start higher than ~3rem). Pull the main block and the sidebar content up so
# the title and the first sidebar label sit on one line just below the header.
st.html(
    "<style>"
    "[data-testid='stMainBlockContainer']{padding-top:3rem !important;}"
    "[data-testid='stSidebarUserContent']{margin-top:-0.5rem;}"
    "</style>"
)


def ui_theme() -> dict:
    """Colours that live in code (map, risk labels, inline HTML), matched to the active theme.
    Surfaces, text and accent come from .streamlit/config.toml; this only mirrors them."""
    try:
        dark = st.context.theme.type != "light"
    except Exception:  # no browser context (tests, old Streamlit)
        dark = True
    if dark:
        return dict(
            dark=True, muted="#A3ACBF", arrow="#6FA8FF",
            risk={"low": "#3DD68C", "moderate": "#FFC53D", "high": "#FF8C42", "critical": "#FF6B6B"},
            map_land="#243049", map_border="#3A4760", map_route="#FF6B6B",
            map_airport="#6FA8FF", map_airport_edge="#141B2B",
        )
    return dict(
        dark=False, muted="#6B7280", arrow="#1A6FD1",
        risk={"low": "#1E9E5A", "moderate": "#B97A00", "high": "#D9641B", "critical": "#C62828"},
        map_land="#E9EDF3", map_border="#C5CCD8", map_route="#E03E3E",
        map_airport="#1A6FD1", map_airport_edge="#FFFFFF",
    )


THEME = ui_theme()

# Language: detected once from the browser locale, then driven by the sidebar selector.
if SESSION_KEY not in st.session_state:
    st.session_state[SESSION_KEY] = detect_language()


# --- helpers ---------------------------------------------------------------

def format_duration(minutes) -> str:
    h, m = divmod(int(minutes), 60)
    return t("duration.h_m", h=h, m=m) if h > 0 else t("duration.m", m=m)


def calculate_economic_impact(prob, distance_miles):
    """Expected loss of one flight, based on FAA cost assumptions (USD)."""
    COST_PER_MIN_AIRLINE = 110.0        # airline operating cost per delay minute
    PASSENGER_TIME_VAL_HOUR = 62.50     # FAA value of travel time savings per passenger-hour
    pax = 150 if distance_miles < 1000 else 280
    avg_delay = 52                      # average delay length, minutes

    direct_flight_cost = avg_delay * COST_PER_MIN_AIRLINE
    pax_loss = (avg_delay / 60) * PASSENGER_TIME_VAL_HOUR * pax

    # Knock-on (reactionary) delays: total event cost is 2.2x the direct losses
    total_event_cost = (direct_flight_cost + pax_loss) * 2.2
    network_impact = total_event_cost - (direct_flight_cost + pax_loss)

    return {
        "expected": prob * total_event_cost,
        "total_event": total_event_cost,
        "direct": direct_flight_cost,
        "pax": pax_loss,
        "network": network_impact,
    }


RISK_LEVELS = ("low", "moderate", "high", "critical")


def get_risk_level(band: int):
    """Label and colour of a risk band (0..3). The band edges belong to the model (FlightAnalytics.bands):
    0.20 / 0.37 / 0.55 on the raw probability of the diploma models, 0.12 / 0.20 / 0.35 on the calibrated
    probability of the weather model. The high band starts at each model's decision threshold."""
    level = RISK_LEVELS[band]
    return t(f"risk.{level}"), THEME["risk"][level]


@st.cache_data(show_spinner=False)
def logo_data_uri(path: str) -> str:
    with open(path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode()


def airline_logo(path: str) -> None:
    """Full-width logo. Airline logos are drawn for light backgrounds, so in the dark theme they sit on
    a light rounded plate; the plate is pulled up 10 px so the logo keeps its place next to the name."""
    if THEME["dark"]:
        box = "background:#EEF1F6;border-radius:12px;padding:8px;margin-top:-10px;"
    else:
        box = "padding:0 8px;"
    st.markdown(
        f'<div style="{box}display:flex;align-items:center;justify-content:center;">'
        f'<img src="{logo_data_uri(path)}" alt="airline logo" style="width:100%;height:auto;display:block;"></div>',
        unsafe_allow_html=True,
    )


def feature_label(name: str) -> str:
    """Human-readable, translated feature name (falls back to the technical name)."""
    key = f"feature.{name}"
    label = t(key)
    return name if label == key else label


# --- resources ---------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_reference():
    data = joblib.load(LOOKUP_PATH)
    try:
        air_df = pd.read_csv(os.path.join(DATA_DIR, "airports.csv"))
        air_info = air_df.set_index("IATA_CODE").to_dict("index")
        line_names = pd.read_csv(os.path.join(DATA_DIR, "airlines.csv")).set_index("IATA_CODE")["AIRLINE"].to_dict()
    except Exception:
        air_info, line_names = {}, {}
    return data, air_info, line_names


@st.cache_resource(show_spinner=False)
def load_model(variant: str):
    """One cache entry per variant, shared by every visitor of this server process: a variant is loaded the
    first time anyone picks it (local models/<file>.pkl, otherwise a download from the Hugging Face Hub) and
    stays in memory, so later switches are instant. Both demo variants together take about 0.8 GB."""
    return joblib.load(resolve_model_path(variant)), load_calibration(variant)


data, air_info, line_names = load_reference()

# Model: from the URL (?model=...) on the first visit, then driven by the sidebar selector.
OFFERED = available_variants()
if st.session_state.get(MODEL_KEY) not in OFFERED:
    asked = st.query_params.get(MODEL_KEY, "")
    st.session_state[MODEL_KEY] = asked if asked in OFFERED else default_variant()

# --- sidebar -----------------------------------------------------------------

with st.sidebar:
    st.selectbox(f":material/language: {t('sidebar.language')}", options=list(LANGS), format_func=LANGS.get, key=SESSION_KEY)
    st.selectbox(t("sidebar.model"), OFFERED, format_func=lambda v: t(f"model.name.{v}"), key=MODEL_KEY)
    variant = st.session_state[MODEL_KEY]
    st.query_params[MODEL_KEY] = variant
    st.caption(t(f"model.caption.{variant}"))
    st.header(t("sidebar.header"))

    unknown = t("sidebar.unknown")
    airline_code = st.selectbox(
        f":material/airlines: {t('sidebar.airline')}", data["airlines"],
        format_func=lambda x: f"{line_names.get(x, unknown)} ({x})",
    )
    origin_code = st.selectbox(
        f":material/flight_takeoff: {t('sidebar.origin')}", data["airports"],
        format_func=lambda x: f"{x} - {air_info.get(x, {}).get('AIRPORT', unknown)}",
    )
    dest_list = data["lookup"][data["lookup"]["ORIGIN_AIRPORT"] == origin_code]["DESTINATION_AIRPORT"].unique()
    dest_code = st.selectbox(
        f":material/flight_land: {t('sidebar.destination')}", sorted(dest_list),
        format_func=lambda x: f"{x} - {air_info.get(x, {}).get('AIRPORT', unknown)}",
    )

    st.divider()
    month = st.slider(t("sidebar.month"), 1, 12, 3)
    days_in_month = calendar.monthrange(DATA_YEAR, month)[1]
    day = st.slider(t("sidebar.day"), 1, days_in_month, min(15, days_in_month))
    dep_hour = st.slider(t("sidebar.dep_hour"), 0, 23, 12)

    # Weather at both ends of the route, only for the model that was trained on it. The widget keys
    # do not depend on the airports, so the choice survives a change of route or a model switch.
    weather = {}
    if uses_weather(variant):
        st.divider()
        st.subheader(t("weather.header"))
        st.caption(t("weather.hint"))
        for tab, end in zip(st.tabs([f"🛫 {origin_code}", f"🛬 {dest_code}"]), ("ORIGIN", "DEST")):
            with tab:
                for kind, values in WEATHER_VALUES.items():
                    weather[f"{end}_{kind}"] = st.selectbox(
                        t(f"weather.{kind}"), values, key=f"weather_{end}_{kind}",
                        format_func=lambda v, k=kind: t(f"weather.{k}.{v}"),
                    )

    st.divider()
    links = [f"[GitHub]({GITHUB_URL})", f"[Hugging Face]({HF_URL})"]
    if PORTFOLIO_URL:
        links.append(f"[{t('footer.portfolio')}]({PORTFOLIO_URL})")
    st.caption(f"© 2026 {AUTHOR}  \n" + " · ".join(links))

# --- main --------------------------------------------------------------------

st.title(t("app.title"))

try:
    with st.spinner(t("model.loading", name=t(f"model.name.{variant}"))):
        model, calibration = load_model(variant)
except Exception as error:  # Hub unreachable or rate-limited: say so instead of a stack trace
    st.error(t("model.failed", name=t(f"model.name.{variant}"), error=type(error).__name__))
    st.stop()
analytics = FlightAnalytics(model, calibration)
tab1, tab2 = st.tabs([t("tab.analysis"), t("tab.model")])

with tab1:
    route = data["lookup"][(data["lookup"]["ORIGIN_AIRPORT"] == origin_code) &
                           (data["lookup"]["DESTINATION_AIRPORT"] == dest_code)].iloc[0]
    origin_full = air_info.get(origin_code, {})
    dest_full = air_info.get(dest_code, {})
    origin_city = origin_full.get("CITY", origin_code)
    dest_city = dest_full.get("CITY", dest_code)

    # Top panel: route, flight details, airline
    c1, _gap, c2, c3 = st.columns([1, 0.2, 0.6, 1.1])
    with c1:
        st.markdown(t("route.header"))
        st.markdown(f"""
            <div style="display:flex; align-items:flex-start; justify-content:space-between; margin-top:-20px;">
                <div style="flex:1;">
                    <h1 style="margin:0; font-family:serif; font-size:2.3rem; line-height:1;">{origin_code}</h1>
                    <div style="font-weight:bold; font-size:0.85rem; margin-top:2px; font-family:sans-serif;">{origin_full.get('AIRPORT', unknown)}</div>
                    <div style="color:{THEME['muted']}; font-size:0.75rem; font-family:sans-serif;">{origin_full.get('CITY', '')}, {origin_full.get('STATE', '')}</div>
                </div>
                <div style="flex:0.2; text-align:center; padding-top:22px; font-size:1.5rem; color:{THEME['arrow']};">➔</div>
                <div style="flex:1; text-align:right;">
                    <h1 style="margin:0; font-family:serif; font-size:2.3rem; line-height:1;">{dest_code}</h1>
                    <div style="font-weight:bold; font-size:0.85rem; margin-top:2px; font-family:sans-serif;">{dest_full.get('AIRPORT', unknown)}</div>
                    <div style="color:{THEME['muted']}; font-size:0.75rem; font-family:sans-serif;">{dest_full.get('CITY', '')}, {dest_full.get('STATE', '')}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(t("flight.header"))
        dist_km = f"{int(route['DISTANCE'] * 1.609):,}".replace(",", " ")
        st.write(t("flight.distance", km=dist_km))
        st.write(t("flight.time", duration=format_duration(route["SCHEDULED_TIME"])))

    with c3:
        st.markdown(t("airline.header"))
        text_col, logo_col = st.columns([0.75, 1.25])
        with logo_col:
            logo_path = os.path.join(LOGOS_DIR, f"{airline_code}.png")
            if os.path.isfile(logo_path):
                airline_logo(logo_path)
            else:
                st.write("🛩")
        with text_col:
            st.write(line_names.get(airline_code, t("airline.unknown")))
            st.caption(t("airline.iata", code=airline_code))

    st.divider()
    col_map, col_res = st.columns([1.3, 1])

    with col_map:
        if origin_code in air_info and dest_code in air_info:
            fig = go.Figure(go.Scattergeo(
                locationmode="USA-states",
                lon=[air_info[origin_code]["LONGITUDE"], air_info[dest_code]["LONGITUDE"]],
                lat=[air_info[origin_code]["LATITUDE"], air_info[dest_code]["LATITUDE"]],
                mode="lines+markers",
                line=dict(width=5, color=THEME["map_route"]),
                marker=dict(size=15, color=THEME["map_airport"], line=dict(width=2, color=THEME["map_airport_edge"])),
                hoverinfo="skip",
            ))
            # The map takes the page colours: land one step lighter than the page, water = page background
            fig.update_geos(
                scope="usa", bgcolor="rgba(0,0,0,0)", showland=True, landcolor=THEME["map_land"],
                showlakes=True, lakecolor="rgba(0,0,0,0)", subunitcolor=THEME["map_border"],
                countrycolor=THEME["map_border"], coastlinecolor=THEME["map_border"], showframe=False,
            )
            fig.update_layout(height=400, margin={"r": 0, "t": 0, "l": 0, "b": 0},
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")

    impact = None
    with col_res:
        if st.button(t("button.analyze"), type="primary", width="stretch"):
            # 1. Model input: weekday from the date (2015), arrival from the scheduled flight time
            flight_date = date(DATA_YEAR, month, day)
            day_of_week = flight_date.isoweekday()          # 1 = Monday ... 7 = Sunday (dataset convention)
            sched_time = int(route["SCHEDULED_TIME"])
            arr_minutes = (dep_hour * 60 + sched_time) % (24 * 60)
            input_df = pd.DataFrame([{
                "MONTH": month, "DAY": day, "DAY_OF_WEEK": day_of_week, "AIRLINE": airline_code,
                "ORIGIN_AIRPORT": origin_code, "DESTINATION_AIRPORT": dest_code,
                "SCHEDULED_DEPARTURE": dep_hour * 100, "DISTANCE": route["DISTANCE"],
                "SCHEDULED_TIME": route["SCHEDULED_TIME"],
                "SCHEDULED_ARRIVAL": (arr_minutes // 60) * 100 + arr_minutes % 60,
                "DEP_HOUR": dep_hour, "ARR_HOUR": arr_minutes // 60,
                "IS_WEEKEND": int(day_of_week in (6, 7)),
                **weather,  # eight weather columns after the flight columns, the order the weather model expects
            }])

            # 2. Delay probability: calibrated for the weather model, raw for the diploma models.
            #    It drives the verdict and the economics.
            prob = analytics.probability(input_df)
            delay_predicted = analytics.is_delay_predicted(prob)

            # 3. Risk index: visual stretch for the bar only (not a calibration)
            risk_idx = analytics.risk_index(prob)

            # 4. Comparison with the base delay rate of the model's data
            ratio, rel_status = analytics.relative_risk(prob)

            # 5. Economics from the same probability
            impact = calculate_economic_impact(prob, route["DISTANCE"])

            # --- results ---
            band = analytics.risk_band(prob)
            risk_label, color = get_risk_level(band)
            st.markdown(f"### {t('risk.level')}: <span style='color:{color}'>{risk_label}</span>",
                        unsafe_allow_html=True)
            band_delay = MODEL_METRICS[variant].get("band_delay")
            if band_delay:
                st.caption(t("risk.band_fact", share=f"{band_delay[band]:.0%}"))
            st.progress(risk_idx, text=t("risk.index_bar", idx=f"{risk_idx:.0%}"))

            col_p1, col_p2 = st.columns(2)
            col_p1.metric(t("metric.prob"), f"{prob:.1%}",
                          help=t("metric.prob_help_cal" if analytics.calibrated else "metric.prob_help",
                                 thr=f"{analytics.threshold:.2f}"))
            col_p2.metric(t("metric.rel", base=f"{analytics.base_rate:.1%}"), f"{ratio:.1f}×",
                          help=t(f"rel.{rel_status}"))

            thr_text = f"{analytics.threshold:.2f}"
            if delay_predicted:
                st.error(t("verdict.delay", thr=thr_text))
            else:
                st.success(t("verdict.ontime", thr=thr_text))

            st.warning(t("econ.risk_value", value=f"{impact['expected']:,.0f}"))

            # --- SHAP summary ---
            shap_data = analytics.get_local_shap(input_df)
            shap_data["label"] = shap_data["feature"].map(feature_label)
            top_factor = shap_data.iloc[0]
            direction = t("direction.up") if top_factor["contribution"] > 0 else t("direction.down")
            st.info(t("insight", feature=top_factor["label"], direction=direction))

    # Detailed panels appear only after an analysis has been run
    if impact is not None:
        details1, details2 = st.columns(2)

        with details1:
            with st.expander(t("econ.expander")):
                st.write(t("econ.total", value=f"{impact['total_event']:,.0f}"))
                st.write(t("econ.formula",
                           expected=f"{impact['expected']:,.0f}",
                           prob=f"{prob:.1%}",
                           total=f"{impact['total_event']:,.0f}"))
                st.markdown(f"""
                ---
                * **{t('econ.network')}:** `${impact['network']:,.0f}`
                * **{t('econ.direct')}:** `${impact['direct']:,.0f}`
                * **{t('econ.pax')}:** `${impact['pax']:,.0f}`
                """)
                st.caption(t("econ.method"))

        with details2:
            with st.expander(t("shap.expander")):
                st.write(t("shap.intro", origin=origin_city, dest=dest_city))
                fig_shap = px.bar(
                    shap_data, x="contribution", y="label", orientation="h",
                    color="contribution", color_continuous_scale="RdYlGn_r", text_auto=".2f",
                    labels={"contribution": t("shap.axis_contrib"), "label": t("shap.axis_feature")},
                )
                fig_shap.update_layout(height=300, showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_shap, width="stretch")
                st.caption(t("shap.caption"))

with tab2:
    current = MODEL_METRICS[variant]
    st.header(t("metrics.header"))
    st.caption(t(f"metrics.loaded.{variant}", size=current["size"]))
    m1, m2 = st.columns(2)
    m1.info(t(f"metrics.auc.{current['test']}", auc=current["auc"]))
    m2.info(t("metrics.recall", thr=current["thr"], recall=current["recall"], precision=current["precision"]))

    st.subheader(t("metrics.weather_header"))
    st.markdown(t("metrics.weather_text"))

    st.subheader(t("metrics.variants_header"))
    st.markdown(t("metrics.variants_text"))
    columns = ("weather", "compact", "full")
    words = {  # rows whose cells are words rather than numbers
        "features": ("features_weather", "features_flight", "features_flight"),
        "train": ("train_weather", "train_diploma", "train_diploma"),
        "test": ("test_days", "test_random", "test_random"),
        "prob": ("prob_cal", "prob_raw", "prob_raw"),
    }
    rows = [("features", "row_features"), ("train", "row_train"), ("test", "row_test"), ("auc", "row_auc"),
            ("auc_new", "row_auc_new"), ("prob", "row_prob"), ("thr", "row_threshold"), ("recall", "row_recall"),
            ("precision", "row_precision"), ("size", "row_size"), ("ram", "row_ram"), ("ctr", "row_ctr"),
            ("script", "row_script")]
    table = ["| " + " | ".join([t("metrics.col_metric")] + [t(f"metrics.col_{c}") for c in columns]) + " |",
             "|---|---|---|---|"]
    for key, label in rows:
        if key in words:
            cells = [t(f"metrics.val.{v}") for v in words[key]]
        elif key == "auc_new":
            cells = [t("metrics.val.auc_days", auc=MODEL_METRICS["weather"]["auc_new"])] + \
                    [t("metrics.val.auc_months", auc=MODEL_METRICS[c]["auc_new"]) for c in columns[1:]]
        else:
            cells = [MODEL_METRICS[c][key] for c in columns]
        table.append(f"| {t(f'metrics.{label}')} | " + " | ".join(cells) + " |")
    st.markdown("\n".join(table))

    st.caption(t("metrics.figures_note"))
    f1, f2 = st.columns(2)
    with f1:
        st.image(os.path.join(FIGURES_DIR, "pr_curve_optimal_threshold.png"), caption=t("metrics.pr_caption"))
    with f2:
        st.image(os.path.join(FIGURES_DIR, "CatBoost Feature Importance.jpg"), caption=t("metrics.fi_caption"))
