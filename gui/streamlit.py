import streamlit as st 

st.write('## ECM P&W1000g models')

fleet = [
  "RA73439",
  "RA73440",
  "RA73441",
  "RA73442",
  "RA73443",
  "RA73444",
  "RA73445",
  "RA73446",
  "VQ-BDI",
  "VQ-BDU",
  "VQ-BDV",
  "VQ-BDW",
  "VQ-BGR",
  "VQ-BGU",
  "VQ-BYI",
  "VQ-BYJ"
]

acnum = st.selectbox("Select aircraft", fleet)
pos = st.radio("Select engine position", [1, 2])

st.write(f'Selected acnum {acnum} pos {pos}')

is_clicked = st.button("Predict")
