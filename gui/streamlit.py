import streamlit as st 
import pandas as pd
import altair as alt

st.set_page_config(layout="wide")
st.write('## ECM P&W1000g models')

@st.cache_data
def get_data():
  return pd.read_csv("https://drive.google.com/uc?export=view&id=1wCaZH0A-r6QRdMvAnwtKn8L1nUREWcn9", parse_dates=['reportts'])

fleet = [
  # "RA73439",
  # "RA73440",
  # "RA73441",
  # "RA73442",
  # "RA73443",
  # "RA73444",
  # "RA73445",
  # "RA73446",
  "VQ-BDU",
  "VQ-BGU",
  "VQ-BDI",
  "VQ-BDV",
  "VQ-BDW",
  "VQ-BGR",
  "VQ-BYI",
  "VQ-BYJ"
]

col1, col2 = st.columns([2, 10])

acnum = col1.selectbox("Select aircraft", fleet)
pos = col1.radio("Select engine position", [1, 2])

is_clicked = col1.button("Predict")

df = get_data().query(f'acnum=="{acnum}" and pos=={pos}')

egtm = alt.Chart(df, height=700) \
  .mark_line(interpolate="basis") \
  .encode(
    x=alt.X('reportts:T'),
    y=alt.Y('egtm'),
  )

delta = alt.Chart(df, height=700) \
  .mark_point(filled=True) \
  .encode(
    x=alt.X('reportts:T'),
    y=alt.Y('egt_delta'),
    opacity=alt.value(0.4),
    color=alt.ColorValue('orange')
  ).interactive()

delta_smooth = alt.Chart(df, height=700) \
  .mark_line(interpolate="basis") \
  .encode(
    x=alt.X('reportts:T'),
    y=alt.Y('egt_delta_smooth'),
    color=alt.ColorValue('orange')
  )


col2.altair_chart(egtm + delta + delta_smooth, use_container_width=True)
