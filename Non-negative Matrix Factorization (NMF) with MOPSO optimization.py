# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 20:21:15 2024

@author: Administrator

Non-negative Matrix Factorization (NMF) with MOPSO optimization for end‑member
unmixing of geochemical data. Outputs contribution proportions and saves them
to an Excel file.
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from tqdm import tqdm
import webbrowser
from matplotlib.colors import rgb2hex
from pyswarm import pso          # Multi‑objective particle swarm optimization
import holoviews as hv

hv.extension('bokeh')


# ---------- Helper functions ----------
def calculate_quantiles(series):
    """
    Return the 2.5th and 97.5th percentiles of a Series.
    """
    lower_quantile = series.quantile(0.025)
    upper_quantile = series.quantile(0.975)
    return lower_quantile, upper_quantile


def replace_outliers(data_frame):
    """
    Replace values outside the [2.5%, 97.5%] interval with the corresponding
    quantile bounds.
    """
    new_data_frame = data_frame.copy()
    for column in data_frame.columns:
        series = data_frame[column]
        lower_q, upper_q = calculate_quantiles(series)
        new_data_frame[column] = np.where(series < lower_q, lower_q,
                                          np.where(series > upper_q, upper_q, series))
    return new_data_frame


def initialize_nndsvda(V, n_components):
    """
    NNDSVDa initialization for NMF (modified version).
    """
    U, S, VT = np.linalg.svd(V, full_matrices=False)
    W = np.zeros((V.shape[0], n_components))
    H = np.zeros((n_components, V.shape[1]))

    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(VT[0, :])

    for r in range(1, n_components):
        ui = U[:, r]
        vi = VT[r, :]
        W[:, r] = np.sqrt(S[r]) * (np.abs(ui) + ui) / 2
        H[r, :] = np.sqrt(S[r]) * (np.abs(vi) + vi) / 2

    W = np.maximum(W, 0)
    H = np.maximum(H, 0)

    # Enforce row sums of W to be close to 1
    for i in range(W.shape[0]):
        row_sum = np.sum(W[i, :])
        if row_sum < 0.97 or row_sum > 1.02:
            scale_factor = 0.97 / row_sum if row_sum < 0.97 else 1.02 / row_sum
            W[i, :] *= scale_factor

    return W, H


def normalize_W(W):
    """
    Force each row of W to have a sum between 0.97 and 1.02.
    """
    for i in range(W.shape[0]):
        row_sum = np.sum(W[i, :])
        if row_sum < 0.97 or row_sum > 1.02:
            scale_factor = 0.97 / row_sum if row_sum < 0.97 else 1.02 / row_sum
            W[i, :] *= scale_factor
    return W


def constrained_nmf(V, n_components, reg_coeff, sparsity_coeff, max_iter=5000, tol=1e-5):
    """
    NMF with Tikhonov regularization and L1 sparsity penalty.
    """
    m, n = V.shape
    W, H = initialize_nndsvda(V, n_components)
    prev_error = float('inf')

    for iteration in tqdm(range(max_iter), desc='NMF Progress'):
        # Update H (non‑negative least squares)
        WTW = W.T @ W
        for j in range(n):
            H[:, j], _ = nnls(WTW + reg_coeff * np.eye(n_components), W.T @ V[:, j])
        H += 1e-7          # avoid zero columns

        # Update W
        HHT = H @ H.T
        for i in range(m):
            W[i, :], _ = nnls(HHT + reg_coeff * np.eye(n_components), H @ V[i, :].T)

        # Apply sparsity (soft thresholding)
        W = np.maximum(0, W - sparsity_coeff * np.sign(W))
        W += 1e-7          # avoid zero rows

        # Enforce row‑sum constraints
        W = normalize_W(W)

        # Reconstruction error with penalties
        reconstruction = W @ H
        error = np.linalg.norm(V - reconstruction) \
                + reg_coeff * (np.linalg.norm(W)**2) \
                + sparsity_coeff * np.sum(np.abs(W))

        # Convergence check
        if abs(prev_error - error) / prev_error < tol:
            break
        prev_error = error

    return W, H, error


def objective_function(params, V):
    """
    Objective function for MOPSO: combines reconstruction error,
    row‑sum deviation and sparsity.
    """
    reg_coeff, sparsity_coeff, n_comp = params
    n_comp = int(n_comp)
    W, H, err = constrained_nmf(V, n_comp, reg_coeff, sparsity_coeff, max_iter=2000)

    # Penalty for row sums not close to 1
    row_sum_penalty = np.sum(np.abs(np.sum(W, axis=1) - 1))

    # L1 penalty for sparsity
    sparsity_penalty = np.sum(np.abs(W))

    # Weight factors (can be adjusted)
    row_sum_weight = 10
    sparsity_weight = 8

    return err + row_sum_weight * row_sum_penalty + sparsity_weight * sparsity_penalty


def optimize_with_mopso(V_scaled):
    """
    Use MOPSO to find optimal regularization coefficient, sparsity coefficient,
    and number of end‑members.
    """
    lb = [1e-7, 0, 3]       # lower bounds
    ub = [1, 0.1, 4]        # upper bounds

    # Wrap objective to pass V_scaled
    def obj(params):
        return objective_function(params, V_scaled)

    best_params, _ = pso(obj, lb, ub, swarmsize=10, maxiter=20)
    return best_params


# ---------- Main workflow ----------
if __name__ == "__main__":

    # 1. Load data
    data = pd.read_excel('Data for mixing proportions.xlsx', header=0, index_col=0)

    # 2. Impute missing values (KNN)
    imputer = KNNImputer(n_neighbors=5, weights="uniform")
    data_filled = imputer.fit_transform(data)
    data_filled = pd.DataFrame(data_filled, columns=data.columns, index=data.index)

    # 3. Replace outliers
    data_filled = replace_outliers(data_filled)

    # 4. Scale to [0,1] by min‑max normalization
    min_vals = np.min(data_filled, axis=0)
    max_vals = np.max(data_filled, axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1
    data_scaled = (data_filled - min_vals) / ranges

    # 5. Optimize parameters using MOPSO
    best_params = optimize_with_mopso(data_scaled.values)
    best_reg, best_sparsity, best_n = best_params
    best_n = int(best_n)
    print(f"Optimized parameters: reg_coeff = {best_reg:.6f}, "
          f"sparsity_coeff = {best_sparsity:.6f}, n_components = {best_n}")

    # 6. Run NMF with optimized parameters
    W, H, final_error = constrained_nmf(data_scaled.values, best_n,
                                        best_reg, best_sparsity)

    # 7. Normalize W to represent percentages (row sum = 1)
    W_normalized = W / np.sum(W, axis=1, keepdims=True)

    # 8. Save mixing proportions to Excel
    #    Create a DataFrame with sample names as index and end‑member columns
    em_names = [f'EM{i+1}' for i in range(best_n)]
    proportion_df = pd.DataFrame(W_normalized,
                                 columns=em_names,
                                 index=data.index)

    proportion_df.to_excel('mixing_proportions.xlsx')
    print("Mixing proportions saved to 'mixing_proportions.xlsx'.")

    # 9. (Optional) Generate chord diagram using HoloViews
    #    Convert proportions to integer percentages for chord representation
    W_int = (W_normalized * 100).astype(int)

    color_palette = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00',
                     '#FF00FF', '#00FFFF']   # extend if more than 6 end‑members

    # Assign colors to each end‑member
    num_em = W_normalized.shape[1]
    colors = [color_palette[i % len(color_palette)] for i in range(num_em)]
    em_colors = {f'EM{i+1}': colors[i] for i in range(num_em)}

    # Build chord data (source = end‑member, target = sample, value = percentage)
    chord_list = []
    for i, sample in enumerate(data.index):
        for j in range(num_em):
            val = W_int[i, j]
            if val > 15:          # only show connections ≥15% for clarity
                chord_list.append({'source': f'EM{j+1}',
                                   'target': sample,
                                   'value': val})

    chord_data = pd.DataFrame(chord_list)

    # Create Chord plot
    chord = hv.Chord(chord_data)
    chord.opts(
        node_color='index',
        edge_color=hv.dim('source').str(),
        cmap=colors,
        labels='name',
        label_text_font_size=12,
        width=800,
        height=800
    )

    # Save and open the chord diagram
    renderer = hv.renderer('bokeh')
    renderer.save(chord, 'chord_diagram.html')
    webbrowser.open('chord_diagram.html')