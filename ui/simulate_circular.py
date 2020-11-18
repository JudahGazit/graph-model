from graph.graph_formatter import GraphFormatter
from graph.graph_simulator import GraphSimulator
from ui.utils import *


last_parameters = None
current_graph = None


def _get_parameters():
    N = st.sidebar.select_slider('Number of Leaves', [2 ** i for i in range(2, 10)], value=16)
    alpha = st.sidebar.slider('alpha', 0.0, 3.0, step=0.001)
    beta = st.sidebar.slider('beta', 0.0, 1.0, step=0.001)
    B = st.sidebar.slider('B', 0.0, 1.0, step=0.001)
    Amax = math.exp(alpha * 2 * math.pi / N) - B * math.exp(2 * beta * math.log2(N) + ((2 * alpha * math.pi) / N))
    A = st.sidebar.slider('A', 0.0, Amax, value=0.1, step=0.001)
    return N, A, B, alpha, beta


def _show_equation(A, B, alpha, beta):
    st.sidebar.latex(f'p_{{i, j}} = A \\cdot e^{{\\alpha\\cdot d_{{i_j}}}} + B \\cdot e^{{-\\beta\\cdot l_{{i_j}}}}')
    st.sidebar.latex(f'p_{{i, j}} = {A} \\cdot e^{{{alpha}\\cdot d_{{i_j}}}} + {B} \\cdot e^{{-{beta}\\cdot l_{{i_j}}}}')


def simulate():
    global last_parameters, current_graph
    N, A, B, alpha, beta = _get_parameters()
    _show_equation(A, B, alpha, beta)

    start_simulation = st.sidebar.button('Start Simulation')
    if start_simulation:
        last_parameters = (N, A, B, alpha, beta)
        current_graph = GraphSimulator(N, A, B, alpha, beta).simulate()

    if last_parameters == (N, A, B, alpha, beta):
        formatter = GraphFormatter(current_graph)
        display_graph_in_circle(current_graph.graph)
        display_metrics(formatter.format_metrics())
        display_charts(formatter.format_chart())
    else:
        display_graph_in_circle(nx.circular_ladder_graph(N))