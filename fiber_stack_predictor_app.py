
import streamlit as st
import joblib
import numpy as np

model = joblib.load('fiber_stack_model.joblib')
st.title('Composite Fiber Stack Tensile Strength Predictor')
fiber_names = ['G', 'C', 'K']
fiber_ints = {'G': 0, 'C': 1, 'K': 2}

st.write('Select a 6-layer fiber stack. Each layer can be Glass (G), Carbon (C), or Kevlar (K).')
cols = st.columns(6)
selected = []
for i in range(6):
    sel = cols[i].selectbox(f'Layer {i+1}', fiber_names, index=0)
    selected.append(fiber_ints[sel])

if st.button('Predict'):
    stack = np.array(selected).reshape(1, -1)
    pred = model.predict(stack)[0]
    st.subheader('Predicted Properties:')
    st.write(f"**Tensile_MPa**: {pred[0]:.2f}")
    st.write(f"**Tensile_noisy_MPa**: {pred[1]:.2f}")
