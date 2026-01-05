import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

st.set_page_config(page_title="TFT Prediction Dashboard", layout="wide")

# ============================
# LOAD DATA
# ============================
@st.cache_data
def load_predictions():
    df = pd.read_csv("Data/predicted_tft.csv")

    # Normalisasi nama kolom (jaga2 biar nggak error)
    df.columns = [c.strip().lower() for c in df.columns]

    return df

df = load_predictions()

st.title("📊 TFT Prediction Dashboard")
st.write("Visualisasi hasil prediksi model Temporal Fusion Transformer (TFT).")
st.markdown("---")



# ============================
# SIDEBAR FILTERS
# ============================
with st.sidebar:
    st.header("Filter Data")

   
    # Auto-detect kolom yang mungkin ada
    available_cols = df.columns

    col_school = "school" if "school" in available_cols else None
    col_major  = "major" if "major" in available_cols else None
    col_year   = "year" if "year" in available_cols else None
    col_pred   = "prediction" if "prediction" in available_cols else df.columns[-1]


    # Filter school
    if col_school:
        school_list = sorted(df[col_school].dropna().unique())
        selected_school = st.selectbox("School", ["All"] + school_list)
        if selected_school != "All":
            df = df[df[col_school] == selected_school]

    # Filter major
    if col_major:
        major_list = sorted(df[col_major].dropna().unique())
        selected_major = st.selectbox("Major", ["All"] + major_list)
        if selected_major != "All":
            df = df[df[col_major] == selected_major]

    # Filter year
    if col_year:
        year_list = sorted(df[col_year].unique())
        selected_year = st.selectbox("Year", ["All"] + [int(y) for y in year_list])
        if selected_year != "All":
            df = df[df[col_year] == selected_year]

st.subheader("📈 Prediction Trend")

# ============================
# LINE CHART
# ============================
if col_year:
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X(col_year + ":O", title="Year"),
            y=alt.Y(col_pred + ":Q", title="Prediction Value"),
            tooltip=list(df.columns),
        )
        .properties(height=350)
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("Kolom 'year' tidak ditemukan. Tidak bisa menampilkan line chart.")

st.markdown("---")
st.subheader("📄 Data Table")

st.dataframe(df, use_container_width=True)

st.markdown("### Download Data")
st.download_button(
    label="Download Filtered CSV",
    data=df.to_csv(index=False),
    file_name="filtered_predictions.csv",
    mime="text/csv"
)
