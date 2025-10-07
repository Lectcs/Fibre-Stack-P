import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.title('Composite Fiber Stack Tensile Strength Predictor')

# Load data
@st.cache_data
def load_data():
    df = pd.read_excel('composites_data.xlsx')
    def extract_stack(type_str):
        import re
        items = re.findall(r'[CKG]', type_str)
        return items if len(items)==6 else None
    def to_int(f):
        return {'G': 0, 'C': 1, 'K': 2}[f]
    stacks = df['Type'].apply(extract_stack)
    valid_rows = stacks.notna()
    for i in range(6):
        df[f'f{i+1}'] = stacks.apply(lambda x: to_int(x[i]) if x else np.nan)
    df = df[valid_rows].reset_index(drop=True)
    X = df[[f'f{i+1}' for i in range(6)]]
    y = df[['Tensile_MPa', 'Tensile_noisy_MPa']]
    return X, y

X, y = load_data()

# Train model in app environment
@st.cache_resource
def train_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model(X, y)

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
