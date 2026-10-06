"""
Hotel Bookings Explorer
------------------------
A local Streamlit app with three parts:
  1. Dashboard      - KPIs and charts over the booking data
  2. Search & Browse - filter / search / export the raw records
  3. Predictor       - a trained model that predicts booking cancellation risk

Run with:  streamlit run app.py
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Hotel Bookings Explorer",
    page_icon="🏨",
    layout="wide",
)

THEMES = {
    "Dark": dict(
        bg_gradient="linear-gradient(135deg, #1e293b, #0f172a, #1e1b4b, #0f172a, #1e293b)",
        text_color="#f1f5f9", metric_bg="rgba(255,255,255,0.06)", metric_border="rgba(255,255,255,0.12)",
        metric_value="#38bdf8", sidebar_bg="rgba(15, 23, 42, 0.85)", tab_bg="rgba(255,255,255,0.06)",
        tab_text="#cbd5e1", tab_hover_bg="rgba(56, 189, 248, 0.12)", tab_hover_text="#e0f2fe",
        tab_active_bg="rgba(56, 189, 248, 0.18)", tab_active_text="#38bdf8", form_bg="rgba(255,255,255,0.05)",
        form_border="rgba(255,255,255,0.1)", hr_color="rgba(255,255,255,0.15)", tile_bg="rgba(255,255,255,0.045)",
        tile_border="rgba(255,255,255,0.10)", tile_shadow="rgba(0,0,0,0.22)",
        tile_hover_shadow="rgba(56, 189, 248, 0.20)", tile_hover_border="rgba(56, 189, 248, 0.35)",
        dim_color="#334155", plotly_template="plotly_dark",
    ),
    "Light": dict(
        bg_gradient="linear-gradient(135deg, #f8fafc, #e2e8f0, #eef2ff, #e2e8f0, #f8fafc)",
        text_color="#0f172a", metric_bg="rgba(15, 23, 42, 0.04)", metric_border="rgba(15, 23, 42, 0.12)",
        metric_value="#0284c7", sidebar_bg="rgba(255, 255, 255, 0.9)", tab_bg="rgba(15, 23, 42, 0.05)",
        tab_text="#334155", tab_hover_bg="rgba(2, 132, 199, 0.10)", tab_hover_text="#075985",
        tab_active_bg="rgba(2, 132, 199, 0.15)", tab_active_text="#0284c7", form_bg="rgba(15, 23, 42, 0.03)",
        form_border="rgba(15, 23, 42, 0.10)", hr_color="rgba(15, 23, 42, 0.15)", tile_bg="rgba(15, 23, 42, 0.035)",
        tile_border="rgba(15, 23, 42, 0.10)", tile_shadow="rgba(15, 23, 42, 0.08)",
        tile_hover_shadow="rgba(2, 132, 199, 0.18)", tile_hover_border="rgba(2, 132, 199, 0.35)",
        dim_color="#cbd5e1", plotly_template="plotly_white",
    ),
}

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Dark"

with st.sidebar:
    st.markdown("### 🎨 Appearance")
    theme_mode = st.radio("Theme", ["Dark", "Light"], horizontal=True, key="theme_mode")

T = THEMES[theme_mode]
DIM_COLOR = T["dim_color"]
pio.templates.default = T["plotly_template"]

st.markdown(
    f"""
    <style>
    @keyframes gradientShift {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(14px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    .stApp {{
        background: {T['bg_gradient']};
        background-size: 300% 300%;
        animation: gradientShift 18s ease infinite;
    }}
    [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
    }}
    h1, h2, h3, h4, p, span, label, .stMarkdown, .stCaption {{
        color: {T['text_color']} !important;
    }}
    .block-container {{
        animation: fadeInUp 0.6s ease-out;
    }}
    [data-testid="stMetric"] {{
        background: {T['metric_bg']};
        border: 1px solid {T['metric_border']};
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.18);
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        animation: fadeInUp 0.5s ease-out;
    }}
    [data-testid="stMetric"]:hover {{
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 10px 26px {T['tile_hover_shadow']};
        border-color: {T['tile_hover_border']};
    }}
    [data-testid="stMetricValue"] {{
        color: {T['metric_value']} !important;
        transition: color 0.3s ease;
    }}
    [data-testid="stSidebar"] {{
        background: {T['sidebar_bg']};
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: {T['tab_bg']};
        border-radius: 10px 10px 0 0;
        padding: 8px 18px;
        color: {T['tab_text']};
        transition: background 0.25s ease, color 0.25s ease;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        background: {T['tab_hover_bg']};
        color: {T['tab_hover_text']};
    }}
    .stTabs [aria-selected="true"] {{
        background: {T['tab_active_bg']} !important;
        color: {T['tab_active_text']} !important;
    }}
    div[data-testid="stForm"] {{
        background: {T['form_bg']};
        border-radius: 14px;
        padding: 20px;
        border: 1px solid {T['form_border']};
        animation: fadeInUp 0.6s ease-out;
    }}
    div.stButton > button, div.stFormSubmitButton > button {{
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    div.stButton > button:hover, div.stFormSubmitButton > button:hover {{
        transform: translateY(-2px) scale(1.03);
        box-shadow: 0 6px 16px {T['tile_hover_shadow']};
    }}
    .stDataFrame {{
        border-radius: 10px;
        overflow: hidden;
        animation: fadeInUp 0.5s ease-out;
    }}
    hr {{
        border-color: {T['hr_color']} !important;
    }}
    span[data-baseweb="tag"] {{
        background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
        border-radius: 8px !important;
        color: #ffffff !important;
        border: none !important;
    }}
    span[data-baseweb="tag"] svg {{
        fill: #ffffff !important;
    }}
    .js-plotly-plot .plotly .cursor-crosshair,
    .js-plotly-plot .plotly .cursor-pointer {{
        cursor: pointer !important;
    }}
    [data-testid="stVerticalBlockBorderWrapper"] {{
        background: {T['tile_bg']};
        border-radius: 14px !important;
        border: 1px solid {T['tile_border']} !important;
        box-shadow: 0 4px 16px {T['tile_shadow']};
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        padding: 6px 6px 0 6px;
    }}
    [data-testid="stVerticalBlockBorderWrapper"]:hover {{
        transform: translateY(-3px);
        box-shadow: 0 12px 28px {T['tile_hover_shadow']};
        border-color: {T['tile_hover_border']} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "hotel_bookings_updated_2024.csv")

MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

DEPOSIT_LABELS = {
    "No Deposit": "Cash",
    "Non Refund": "UPI",
    "Refundable": "Card",
}
DEPOSIT_LABELS_REVERSE = {v: k for k, v in DEPOSIT_LABELS.items()}

ROOM_TYPE_LABELS = {
    "A": "AC", "B": "AC", "C": "AC", "D": "AC", "E": "AC",
    "F": "Non-AC", "G": "Non-AC", "H": "Non-AC", "L": "Non-AC", "P": "Non-AC",
}

FRIENDLY_BASE = {
    "hotel_type": "Hotel type",
    "arrival_date_month": "Arrival month",
    "meal": "Meal plan",
    "market_segment": "Market segment",
    "distribution_channel": "Distribution channel",
    "room_ac_type": "Room type",
    "deposit_type": "Payment method",
    "customer_type": "Customer type",
}
FRIENDLY_NUM = {
    "lead_time": "Lead time (days)",
    "arrival_date_week_number": "Arrival week number",
    "stays_in_weekend_nights": "Weekend nights booked",
    "stays_in_week_nights": "Weekday nights booked",
    "adults": "Number of adults",
    "children": "Number of children",
    "babies": "Number of babies",
    "is_repeated_guest": "Repeat guest",
    "previous_cancellations": "Previous cancellations",
    "previous_bookings_not_canceled": "Previous bookings kept",
    "booking_changes": "Booking changes made",
    "days_in_waiting_list": "Days on waiting list",
    "adr": "Average daily rate",
    "required_car_parking_spaces": "Parking requested",
    "total_of_special_requests": "Special requests made",
}


def prettify_feature(col: str) -> str:
    if col in FRIENDLY_NUM:
        return FRIENDLY_NUM[col]
    for base, friendly in FRIENDLY_BASE.items():
        prefix = base + "_"
        if col.startswith(prefix):
            val = col[len(prefix):]
            if base == "deposit_type":
                val = DEPOSIT_LABELS.get(val, val)
            return f"{friendly}: {val}"
    return col


FEATURE_COLUMNS_NUM = [
    "lead_time", "arrival_date_week_number", "stays_in_weekend_nights",
    "stays_in_week_nights", "adults", "children", "babies",
    "is_repeated_guest", "previous_cancellations", "previous_bookings_not_canceled",
    "booking_changes", "days_in_waiting_list", "adr",
    "required_car_parking_spaces", "total_of_special_requests",
]
FEATURE_COLUMNS_CAT = [
    "hotel_type", "arrival_date_month", "meal", "market_segment",
    "distribution_channel", "room_ac_type", "deposit_type", "customer_type",
]
FEATURE_COLUMNS = FEATURE_COLUMNS_NUM + FEATURE_COLUMNS_CAT


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["hotel_type"] = df["hotel"].str.split(" - ").str[0]
    df["room_ac_type"] = df["reserved_room_type"].map(ROOM_TYPE_LABELS).fillna("Non-AC")
    df["children"] = df["children"].fillna(0)
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["revenue"] = df["adr"] * df["total_nights"]
    df["arrival_date_month"] = pd.Categorical(df["arrival_date_month"], categories=MONTH_ORDER, ordered=True)
    return df


@st.cache_resource(show_spinner="Training cancellation model...")
def train_model(df: pd.DataFrame):
    data = df.copy()
    data["arrival_date_month"] = data["arrival_date_month"].astype(str)
    X = pd.get_dummies(data[FEATURE_COLUMNS], columns=FEATURE_COLUMNS_CAT, drop_first=False)
    y = data["is_canceled"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=200, max_depth=14, min_samples_leaf=3,
        n_jobs=-1, random_state=42, class_weight="balanced",
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "auc": roc_auc_score(y_test, proba),
        "cm": confusion_matrix(y_test, preds),
    }
    importances = (
        pd.Series(model.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
        .head(15)
    )
    return model, list(X.columns), metrics, importances


df = load_data()

st.title("🏨 Hotel Bookings Explorer")

tab_dashboard, tab_search, tab_predict = st.tabs(
    ["📊 Dashboard", "🔍 Search & Browse", "🤖 Booking Predictor"]
)

# --------------------------------------------------------------------------
# Shared sidebar filters (used by Dashboard + Search tabs)
# --------------------------------------------------------------------------
st.sidebar.header("Filters")

hotel_types = sorted(df["hotel_type"].unique())
sel_hotel_types = st.sidebar.multiselect("Hotel type", hotel_types, default=hotel_types)

cities = sorted(df["city"].dropna().unique())
sel_cities = st.sidebar.multiselect("City", cities, default=[])

years = sorted(df["arrival_date_year"].unique())
sel_years = st.sidebar.multiselect("Year", years, default=years)

sel_months = st.sidebar.multiselect("Month", MONTH_ORDER, default=[])

status_options = sorted(df["reservation_status"].unique())
sel_status = st.sidebar.multiselect("Reservation status", status_options, default=[])

filtered = df[df["hotel_type"].isin(sel_hotel_types) & df["arrival_date_year"].isin(sel_years)]
if sel_cities:
    filtered = filtered[filtered["city"].isin(sel_cities)]
if sel_months:
    filtered = filtered[filtered["arrival_date_month"].astype(str).isin(sel_months)]
if sel_status:
    filtered = filtered[filtered["reservation_status"].isin(sel_status)]

st.sidebar.caption(f"{len(filtered):,} bookings match current filters")

# --------------------------------------------------------------------------
# TAB 1: Dashboard
# --------------------------------------------------------------------------
PBI_PALETTE = ["#118DFF", "#E66C37", "#12239E", "#6B007B", "#E044A7", "#744EC2", "#D9B300", "#00B294"]


def apply_cross_filters(base_df, filters, exclude=None):
    d = base_df
    for dim, val in filters.items():
        if dim == exclude or val is None:
            continue
        if dim == "arrival_date_month":
            d = d[d["arrival_date_month"].astype(str) == val]
        else:
            d = d[d[dim] == val]
    return d


def bar_colors(categories, selected_val, palette):
    return [
        (palette[i % len(palette)] if (not selected_val or cat == selected_val) else "#334155")
        for i, cat in enumerate(categories)
    ]


with tab_dashboard:
    if filtered.empty:
        st.warning("No bookings match the selected filters.")
    else:
        if "cross_filters" not in st.session_state:
            st.session_state.cross_filters = {
                "arrival_date_month": None, "hotel_type": None, "city": None,
                "market_segment": None,
            }
        cf = st.session_state.cross_filters
        cross = apply_cross_filters(filtered, cf)

        info_col, clear_col = st.columns([5, 1])
        with clear_col:
            if st.button("Clear cross-filter", use_container_width=True):
                st.session_state.cross_filters = {k: None for k in cf}
                st.rerun()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total bookings", f"{len(cross):,}")
        c2.metric("Cancellation rate", f"{cross['is_canceled'].mean() * 100:.1f}%")
        c3.metric("Average ADR", f"₹{cross['adr'].mean():,.0f}")
        c4.metric("Estimated revenue", f"₹{cross['revenue'].sum():,.0f}")

        st.divider()

        def handle_click(event, dim):
            points = (event or {}).get("selection", {}).get("points", [])
            if points:
                clicked = points[0].get("x") or points[0].get("label")
                new_val = None if clicked == cf[dim] else clicked
                if new_val != cf[dim]:
                    st.session_state.cross_filters[dim] = new_val
                    st.rerun()

        col1, col2 = st.columns(2)

        with col1, st.container(border=True):
            own = apply_cross_filters(filtered, cf, exclude="arrival_date_month")
            monthly = (
                own.groupby("arrival_date_month", observed=True).size()
                .reindex(MONTH_ORDER).fillna(0).reset_index(name="bookings")
            )
            fig = px.bar(monthly, x="arrival_date_month", y="bookings", title="Bookings by month")
            fig.update_traces(marker_color=bar_colors(monthly["arrival_date_month"], cf["arrival_date_month"], PBI_PALETTE))
            fig.update_layout(xaxis_title="", yaxis_title="Bookings", dragmode=False)
            event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points",
                                     key="month_chart", config={"displayModeBar": False})
            handle_click(event, "arrival_date_month")

        with col2, st.container(border=True):
            own = apply_cross_filters(filtered, cf, exclude="hotel_type")
            cancel_by_hotel = own.groupby("hotel_type")["is_canceled"].mean().reset_index()
            cancel_by_hotel["is_canceled"] *= 100
            fig = px.bar(cancel_by_hotel, x="hotel_type", y="is_canceled", title="Cancellation rate by hotel type")
            fig.update_traces(marker_color=bar_colors(cancel_by_hotel["hotel_type"], cf["hotel_type"], PBI_PALETTE))
            fig.update_layout(xaxis_title="", yaxis_title="Cancellation rate (%)", showlegend=False, dragmode=False)
            event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points",
                                     key="hotel_chart", config={"displayModeBar": False})
            handle_click(event, "hotel_type")

        col3, col4 = st.columns(2)

        with col3, st.container(border=True):
            own = apply_cross_filters(filtered, cf, exclude="city")
            top_cities = own.groupby("city").size().sort_values(ascending=False).head(10).reset_index(name="bookings")
            fig = px.bar(top_cities, x="bookings", y="city", orientation="h", title="Top 10 cities by bookings")
            fig.update_traces(marker_color=bar_colors(top_cities["city"], cf["city"], PBI_PALETTE))
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title="Bookings", yaxis_title="", dragmode=False)
            event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points",
                                     key="city_chart", config={"displayModeBar": False})
            handle_click(event, "city")

        with col4, st.container(border=True):
            fig = px.box(cross, x="hotel_type", y="adr", title="ADR distribution by hotel type", points=False,
                         color="hotel_type", color_discrete_sequence=PBI_PALETTE)
            fig.update_layout(xaxis_title="", yaxis_title="ADR", showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with st.container(border=True):
            own = apply_cross_filters(filtered, cf, exclude="market_segment")
            seg = own["market_segment"].value_counts().reset_index()
            seg.columns = ["market_segment", "bookings"]
            pull = [0.06 if (not cf["market_segment"] or s == cf["market_segment"]) else 0 for s in seg["market_segment"]]
            fig = px.pie(seg, names="market_segment", values="bookings", title="Bookings by market segment",
                         hole=0.4, color_discrete_sequence=PBI_PALETTE)
            fig.update_traces(pull=pull)
            event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points",
                                     key="segment_chart", config={"displayModeBar": False})
            points = (event or {}).get("selection", {}).get("points", [])
            if points:
                clicked = points[0].get("label")
                new_val = None if clicked == cf["market_segment"] else clicked
                if new_val != cf["market_segment"]:
                    st.session_state.cross_filters["market_segment"] = new_val
                    st.rerun()

# --------------------------------------------------------------------------
# TAB 2: Search & Browse
# --------------------------------------------------------------------------
with tab_search:
    st.subheader("Search & browse bookings")

    search_col1, search_col2, search_col3 = st.columns(3)
    with search_col1:
        city_filter = st.multiselect("City", sorted(df["city"].dropna().unique()), default=[])
    with search_col2:
        room_filter = st.multiselect("Room type", sorted(df["room_ac_type"].unique()), default=[])
    with search_col3:
        deposit_display = [DEPOSIT_LABELS.get(d, d) for d in sorted(df["deposit_type"].unique())]
        deposit_filter_display = st.multiselect("Payment method", deposit_display, default=[])
        deposit_filter = [DEPOSIT_LABELS_REVERSE.get(d, d) for d in deposit_filter_display]

    lead_time_range = st.slider(
        "Lead time (days between booking and arrival)",
        int(df["lead_time"].min()), int(df["lead_time"].max()),
        (int(df["lead_time"].min()), int(df["lead_time"].max())),
    )
    adr_range = st.slider(
        "ADR range",
        float(df["adr"].min()), float(df["adr"].max()),
        (float(df["adr"].min()), float(df["adr"].max())),
    )

    result = filtered.copy()
    if city_filter:
        result = result[result["city"].isin(city_filter)]
    if room_filter:
        result = result[result["room_ac_type"].isin(room_filter)]
    if deposit_filter:
        result = result[result["deposit_type"].isin(deposit_filter)]
    result = result[
        result["lead_time"].between(*lead_time_range) & result["adr"].between(*adr_range)
    ]

    st.write(f"**{len(result):,} bookings** match your search")

    display_cols = [
        "hotel_type", "city", "arrival_date_year", "arrival_date_month",
        "arrival_date_day_of_month", "adults", "children", "babies",
        "room_ac_type", "deposit_type", "customer_type", "lead_time",
        "adr", "total_of_special_requests", "reservation_status",
    ]
    result_display = result[display_cols].head(2000).copy()
    result_display["deposit_type"] = result_display["deposit_type"].map(DEPOSIT_LABELS).fillna(result_display["deposit_type"])
    result_display = result_display.rename(columns={"deposit_type": "payment_method", "room_ac_type": "room_type"})
    st.dataframe(result_display, use_container_width=True, height=450)
    if len(result) > 2000:
        st.caption("Showing first 2,000 of the matching rows. Narrow your filters or download the full result below.")

    export_df = result[display_cols].copy()
    export_df["deposit_type"] = export_df["deposit_type"].map(DEPOSIT_LABELS).fillna(export_df["deposit_type"])
    export_df = export_df.rename(columns={"deposit_type": "payment_method", "room_ac_type": "room_type"})
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered results as CSV", csv_bytes, "filtered_bookings.csv", "text/csv")

# --------------------------------------------------------------------------
# TAB 3: Cancellation Predictor
# --------------------------------------------------------------------------
with tab_predict:
    st.subheader("Predict booking outcome")
    st.caption("A Random Forest model trained on the full dataset (cached after first run).")

    model, model_columns, metrics, importances = train_model(df)

    m1, m2 = st.columns(2)
    m1.metric("Model accuracy (held-out test set)", f"{metrics['accuracy'] * 100:.1f}%")
    m2.metric("ROC-AUC", f"{metrics['auc']:.3f}")

    fig_imp = px.bar(
        importances.rename(prettify_feature).sort_values().reset_index(),
        x=0, y="index", orientation="h",
        title="What drives a booking's outcome", labels={"0": "Importance", "index": ""},
    )
    st.plotly_chart(fig_imp, use_container_width=True)


