import streamlit as st
import networkx as nx

from graph.datasets_loader import DatasetsResultCache
from graph.graph_formatter import GraphFormatter
from ui.utils import *

dataset_result_cache = DatasetsResultCache()


def _select_option():
    all_options = dataset_result_cache.options()
    categories = [o['label'] for o in all_options]
    select_category = st.sidebar.radio('Category', categories)
    options = [o['options'] for o in all_options if o['label'] == select_category][0]
    selected_option = st.sidebar.selectbox('Dataset', [o['label'] for o in options])
    st.title(f'{select_category} - {selected_option}')
    selected_category_name = [o['name'] for o in all_options if o['label'] == select_category][0]
    selected_option_name = [o['name'] for o in options if o['label'] == selected_option][0]
    return selected_category_name, selected_option_name


def datasets():
    select_category, selected_option = _select_option()
    result = dataset_result_cache.get_results(f'{select_category}/{selected_option}')
    graph = nx.Graph()
    frame = pd.DataFrame(result['edges'])
    graph.add_weighted_edges_from(frame.values)
    # display_graph_naturally(graph)
    display_metrics(result['metric'])
    display_charts(result['chart'])
