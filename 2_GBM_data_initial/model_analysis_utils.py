"""
Model Analysis Utilities for Regression Models

Functions for model power, fit, and lift charts.
Based on: /Users/Mach/dev/aps/code/26Dmodelv1/lib/
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def gini(actual, pred):
    """Calculate Gini coefficient (0=no power, 1=perfect, typical: 0.3-0.7)"""
    actual, pred = np.array(actual), np.array(pred)
    sorted_indices = np.argsort(pred)
    sorted_actual = actual[sorted_indices]
    cumulative_actual = np.cumsum(sorted_actual) / np.sum(sorted_actual)
    cumulative_population = np.linspace(1/len(sorted_actual), 1, len(sorted_actual))
    area_under_curve = np.trapz(cumulative_actual, cumulative_population)
    return 1 - 2 * area_under_curve


def plot_lorenz_curve(actual, pred, title="Lorenz Curve"):
    """
    Plot Lorenz curve and calculate Gini coefficient.
    
    Parameters
    ----------
    actual : array-like
        Actual values
    pred : array-like
        Predicted values (used for ranking)
    title : str
        Plot title
    """
    actual = np.array(actual)
    pred = np.array(pred)
    
    # Sort by predictions
    sorted_indices = np.argsort(pred)
    sorted_actual = actual[sorted_indices]
    
    # Cumulative distributions
    cumulative_actual = np.cumsum(sorted_actual) / np.sum(sorted_actual)
    cumulative_population = np.linspace(1/len(sorted_actual), 1, len(sorted_actual))
    
    # Add (0,0) point
    cumulative_actual = np.insert(cumulative_actual, 0, 0)
    cumulative_population = np.insert(cumulative_population, 0, 0)
    
    # Calculate Gini
    gini_value = gini(actual, pred)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(cumulative_population, cumulative_actual, label='Lorenz Curve', color='blue', linewidth=2)
    plt.plot([0, 1], [0, 1], label='Line of Equality', color='black', linestyle='--')
    plt.fill_between(cumulative_population, cumulative_actual, cumulative_population, 
                     color='gray', alpha=0.3)
    plt.title(f"{title}\nGini Coefficient = {gini_value:.4f}")
    plt.xlabel("Cumulative Share of Population (sorted by prediction)")
    plt.ylabel("Cumulative Share of Actuals")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def create_lift_chart(y_actual, y_pred, weight=None, bins=10, title="Lift Chart", print_table=False):
    """
    Create lift chart with relativities.
    IMPORTANT: Sorted ascending by pred, so decile 10 = HIGHEST predictions.
    
    Returns
    -------
    pd.DataFrame
        Decile-level aggregated data with columns: decile, actual, pred, weight, act_rel, pred_rel
    """
    df = pd.DataFrame({
        'actual': np.array(y_actual),
        'pred': np.array(y_pred),
        'weight': np.ones(len(y_actual)) if weight is None else np.array(weight)
    })
    
    # Sort by pred (ascending) - decile 10 will have highest predictions
    df = df.sort_values('pred').reset_index(drop=True)
    
    # Weighted deciles
    w = df['weight'].astype(float)
    cum_w = w.cumsum() / w.sum()
    df['decile'] = np.ceil(cum_w * bins).astype(int).clip(1, bins)
    
    # Aggregate
    x = df.groupby('decile').agg({
        'actual': 'mean',
        'pred': 'mean',
        'weight': 'sum'
    }).reset_index()
    
    # Relativities (shape-only: normalize to pred mean)
    overall_pred = df['pred'].mean()
    x['act_rel'] = x['actual'] / overall_pred
    x['pred_rel'] = x['pred'] / overall_pred
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax2 = ax.twinx()
    
    weight_pct = (x['weight'] / x['weight'].sum() * 100)
    ax2.bar(x['decile'], weight_pct, alpha=0.3, color='gray', width=0.6)
    
    ax.plot(x['decile'], x['act_rel'], 'o-', linewidth=2, label='Actual', color='blue', markersize=8)
    ax.plot(x['decile'], x['pred_rel'], 's-', linewidth=2, label='Predicted', color='orange', markersize=8)
    ax.axhline(1.0, linestyle='--', color='black', alpha=0.5)
    
    y_max = max(x['act_rel'].max(), x['pred_rel'].max()) * 1.10
    y_min = min(x['act_rel'].min(), x['pred_rel'].min()) * 0.90
    if np.isclose(y_max, y_min): y_min, y_max = 0.95, 1.05
    
    ax.set_ylim(y_min, y_max)
    ax2.set_ylim(0, max(100, weight_pct.max() * 1.2))
    
    ax.set_xlabel('Decile (1=Low Risk, 10=High Risk)')
    ax.set_ylabel('Relativity (vs portfolio avg)')
    ax2.set_ylabel('Weight (%)')
    ax.set_title(title)
    ax.set_xticks(x['decile'])
    ax.legend(loc='upper left')
    ax2.legend(['Weight'], loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    if print_table:
        print(x[['decile', 'actual', 'pred', 'act_rel', 'pred_rel']])
    
    return x
