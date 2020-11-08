import streamlit as st

from ui.datasets import datasets
from ui.multiple_optimizations import multiple_optimizations
from ui.optimize import optimize
from ui.simulate import simulate

actions = {
    'Simulate (Circular)': simulate,
    'Datasets': datasets,
    'Cost Optimize': optimize,
    'Previous Optimizations': multiple_optimizations
}

if __name__ == '__main__':
    st.set_page_config(page_title='Graph Models')
    selected_action = st.sidebar.selectbox('Action', list(actions.keys()))
    st.sidebar.title('Parameters')
    actions[selected_action]()
