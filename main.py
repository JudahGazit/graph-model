import logging
import sys

import streamlit as st

from ui.datasets import datasets
from ui.multiple_optimizations import multiple_optimizations
from ui.optimize import optimize
from ui.simulate_circular import simulate
from ui.simulate_lattice import simulate_lattice

actions = {
    'Simulate (Circular)': simulate,
    'Simulate (Lattice)': simulate_lattice,
    'Datasets': datasets,
    'Cost Optimize': optimize,
    'Previous Optimizations': multiple_optimizations
}

logging.basicConfig(force=True, stream=sys.stdout, level=logging.INFO, format="%(asctime)s \t %(levelname)s \t %(name)s \t %(message)s")
logger = logging.getLogger()

if __name__ == '__main__':
    st.set_page_config(page_title='Graph Models')
    selected_action = st.sidebar.selectbox('Action', list(actions.keys()))
    st.sidebar.title('Parameters')
    logger.info('Selecting action "%s"', selected_action)
    actions[selected_action]()
