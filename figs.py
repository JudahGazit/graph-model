import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from graph.multiple_simulations_loader import MultipleOptimizationsLoader

loader = MultipleOptimizationsLoader()

options = [opt for opt in loader.options if opt.cost_type == 'torus'
           and opt.nodes == '100'
           and opt.edges == '504']
graphs = []
for opt in options:
    try:
        data = loader.load('best', *opt)
        X, Y = data['chart']['edge-length-dist-dbins']['x'], data['chart']['edge-length-dist-dbins']['y']
        graphs.append([float(opt.wiring_factor), X, Y])
    except Exception as e:
        print(f'Error in {opt}')
        print(e)
graphs = sorted(graphs)


def display_two_dimentional(graphs):
    fig, ax = plt.subplots(1, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_xlabel('B')
    ax.set_ylabel('C')
    ax.set_title(f'N=100, K=10.0, A=1.0, C=?, B=?, F = (A (1-w) - 0.5 * (Br + Cf)) ^ 2')
    for g in graphs:
        r, f, x, y = g
        row_ax = ax.inset_axes(((0.055 + r) / 1.2, (0.055 + f) / 1.2, 0.9 / 12, 0.9 / 12))
        row_ax.set_xticks([])
        row_ax.set_yticks([])
        row_ax.bar(range(len(x)), y)
        args = fit_exponential_function_to_chart(range(len(x)), y)
        X1 = np.linspace(0, len(x) - 1, 100)
        Y1 = exponent(*args, X1)
        row_ax.plot(X1, Y1, 'red')


def exponent(A, B, lamda1, lamda2, x):
    return A * np.exp(- abs(lamda1) * x) + B * np.exp(abs(lamda2) * x)


def fit_exponential_function_to_chart(X, Y):
    def loss_exp(args, x, y):
        return exponent(*args, x) - y

    res_exp = least_squares(loss_exp, [1, 0.01, 0.5, 0.5], args=(range(len(X)), Y), ftol=1e-10)
    args = res_exp.x
    return args


def display_one_dimentional(graphs):
    fig, [ax1, ax2] = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 4]})

    ax1.set_title(f'N=100, K=10.0, B=1.0, C=1.0, A=?, regulation lambda 0.1')
    ax1.set_yticks([])
    lim = [0, 1]
    ax1.set_xlim(*lim)
    ax1.set_xticks(np.linspace(*lim, 11))
    ax1.set_xlabel('A')
    for spine in ax1.spines:
        if spine != 'bottom':
            ax1.spines[spine].set_visible(False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    for spine in ax2.spines:
        ax2.spines[spine].set_visible(False)
    for g in graphs:
        key, x, y = g
        key_x_pos = (key - lim[0]) / (lim[1] - lim[0])
        ax = ax2.inset_axes((key_x_pos - 0.045, 0.8, 0.09, 0.2))
        ax.bar(range(len(x)), y)
        ax.set_xticks([])
        ax.set_yticks([])

if __name__ == '__main__':
    display_one_dimentional(graphs)