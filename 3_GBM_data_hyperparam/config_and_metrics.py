"""
Configuration and Custom Metrics for XGBoost Hyperparameter Tuning
This module provides configuration settings and custom metric functions.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'target_col': 'cc_col',
    'exposure_col': 'ee_col',
    'cols_to_remove': [
        'ep_bi', 'ep_col', 'ee_bi', 'ee_col',
        'incloss_bi', 'incloss_col', 'cc_bi', 'cc_col',
        'zip', 'pol_id', 'vin_id', 'Date'
    ]
}

# =============================================================================
# METRIC FUNCTIONS
# =============================================================================

def model_metrics(input_data, act, pred, weight_name):
    """
    Calculate custom insurance metrics (model power and fit quality).
    
    Parameters:
    -----------
    input_data : DataFrame
        Input data with actual, predicted, and weight columns
    act : str
        Column name for actual values
    pred : str
        Column name for predicted values
    weight_name : str
        Column name for weights (exposure)
    
    Returns:
    --------
    list : [model_power, fit_quality]
    """
    test_data = input_data.copy()
    bins = 10
    
    test_data['decile'] = (
        round(test_data.sort_values(by='pred')[weight_name].cumsum() / 
              test_data[weight_name].sum(), 2) * bins
    ).apply(np.floor)
    test_data['decile'] = np.where(
        test_data['decile'] + 1 > bins, 
        bins, 
        test_data['decile'] + 1
    )
    
    x = test_data.groupby(['decile'], dropna=False).agg({
        weight_name: 'sum', 
        act: 'sum', 
        pred: 'sum'
    }).reset_index()
    
    x['pp_act'] = x[act] / x[weight_name]
    x['pp_pred'] = x[pred] / x[weight_name]
    
    # Calculate fit quality
    x['decile_error'] = abs(x['pp_pred'] / x['pp_act'] - 1).replace(
        [-np.inf, np.inf], np.nan
    ).fillna(1)
    x['decile_error_sp'] = x['decile_error'] * x[weight_name]
    fit_quality = 1 - x['decile_error_sp'].sum() / x[weight_name].sum()
    
    # Calculate model power
    tot_pp = x[act].sum() / x[weight_name].sum()
    x['diff_unity'] = abs(x['pp_pred'] / tot_pp - 1)
    x['diff_unity'] = x['diff_unity'] * x[weight_name]
    model_power = x['diff_unity'].sum() / x[weight_name].sum()
    
    return [model_power, fit_quality]


def calculate_all_metrics(y_true, y_pred, exposure):
    """
    Calculate multiple metrics including standard and custom insurance metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted values
    exposure : array-like
        Exposure values
    
    Returns:
    --------
    dict : Dictionary of metric names and values
    """
    # Standard metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Prepare data for custom metrics
    eval_data = pd.DataFrame({
        'act': y_true,
        'pred': y_pred,
        'exposure': exposure
    })
    eval_data['act_weighted'] = eval_data['act'] * eval_data['exposure']
    eval_data['pred_weighted'] = eval_data['pred'] * eval_data['exposure']
    
    # Custom insurance metrics
    model_power, fit_quality = model_metrics(
        eval_data, 
        'act_weighted', 
        'pred_weighted', 
        'exposure'
    )
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Model_Power': model_power,
        'Fit_Quality': fit_quality
    }


def lift_chart_modified(test_data, weight_name, bins, print_table=False):
    """
    Generate a lift chart comparing actual vs predicted values.
    
    Parameters:
    -----------
    test_data : DataFrame
        Data containing 'act', 'pred', and weight columns
    weight_name : str
        Column name for weights (exposure)
    bins : int
        Number of bins/deciles to create
    print_table : bool
        Whether to print the summary table
    """
    import matplotlib.pyplot as plt
    
    test_data = test_data.copy()
    test_data['decile'] = (
        round(test_data.sort_values(by='pred', ascending=True)[weight_name].cumsum() / 
              test_data[weight_name].sum(), 2) * bins
    ).apply(np.floor)
    test_data['decile'] = np.where(
        test_data['decile'] + 1 > bins, 
        bins, 
        test_data['decile'] + 1
    )
    
    x = test_data.groupby(['decile'], dropna=False).agg({
        weight_name: 'sum', 
        'act_weighted': 'sum', 
        'pred_weighted': 'sum'
    }).reset_index()
    
    x['act'] = x['act_weighted'] / x[weight_name]
    x['pred'] = x['pred_weighted'] / x[weight_name]
    x.drop(columns=['act_weighted', 'pred_weighted'], inplace=True)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax2 = ax.twinx()
    
    y_max = max(x['act'].max(), x['pred'].max()) * 1.20
    ax2.set_ylim(0, y_max)
    
    x[weight_name].plot.bar(stacked=False, ax=ax, alpha=0.6, label='Exposure')
    x['act'].plot(kind='line', ax=ax2, marker='o', linewidth=2, label='Actual', color='blue')
    x['pred'].plot(kind='line', ax=ax2, marker='s', linewidth=2, label='Predicted', color='red')
    
    ax.set_xlabel('Decile')
    ax.set_ylabel('Exposure')
    ax2.set_ylabel('Average Value')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title('Lift Chart: Actual vs Predicted')
    plt.tight_layout()
    plt.show()
    
    if print_table:
        print("\nLift Chart Summary:")
        print(x.to_string(index=False))


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def prepare_data_for_modeling(df, config):
    """
    Prepare data for modeling based on configuration.
    
    Parameters:
    -----------
    df : DataFrame
        Raw input data
    config : dict
        Configuration dictionary with target_col, exposure_col, and cols_to_remove
    
    Returns:
    --------
    tuple : (X, y, exposure, feature_names)
    """
    # Extract target and exposure before removing columns
    target = df[config['target_col']].copy()
    exposure = df[config['exposure_col']].copy()
    
    # Remove specified columns
    cols_to_drop = [col for col in config['cols_to_remove'] if col in df.columns]
    df_features = df.drop(columns=cols_to_drop)
    
    # Get feature names
    feature_names = df_features.columns.tolist()
    
    return df_features, target, exposure, feature_names


def compare_models(original_metrics, tuned_metrics):
    """
    Compare original and tuned model metrics.
    
    Parameters:
    -----------
    original_metrics : dict
        Metrics from original model
    tuned_metrics : dict
        Metrics from tuned model
    
    Returns:
    --------
    DataFrame : Comparison table
    """
    comparison = pd.DataFrame({
        'Metric': list(original_metrics.keys()),
        'Original': list(original_metrics.values()),
        'Tuned': list(tuned_metrics.values())
    })
    
    # Calculate improvement
    comparison['Improvement'] = comparison.apply(
        lambda row: ((row['Tuned'] - row['Original']) / abs(row['Original']) * 100)
        if row['Metric'] != 'Model_Power' 
        else ((row['Original'] - row['Tuned']) / abs(row['Original']) * 100),
        axis=1
    )
    
    return comparison


# =============================================================================
# HYPERPARAMETER TUNING FUNCTIONS
# =============================================================================

def random_search_xgboost(X_train, y_train, param_grid, n_iter=50, cv=3, random_state=42):
    """
    Perform random hyperparameter search using XGBoost's native CV.
    
    Parameters:
    -----------
    X_train : DataFrame
        Training features
    y_train : Series
        Training target
    param_grid : dict
        Parameter grid to search
    n_iter : int
        Number of random combinations to try
    cv : int
        Number of cross-validation folds
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    tuple : (best_params, best_score, results)
    """
    np.random.seed(random_state)
    
    # Generate random parameter combinations
    param_combinations = []
    for _ in range(n_iter):
        params = {key: np.random.choice(values) for key, values in param_grid.items()}
        param_combinations.append(params)
    
    # Prepare DMatrix for XGBoost CV
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    # Initialize tracking variables
    best_score = float('inf')
    best_params = None
    results = []
    
    print("Starting random hyperparameter search...")
    print(f"Testing {n_iter} random combinations out of {np.prod([len(v) for v in param_grid.values()]):,} possible")
    
    for i, params in enumerate(param_combinations, 1):
        # Configure XGBoost CV parameters
        xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': params['max_depth'],
            'learning_rate': params['learning_rate'],
            'subsample': params['subsample'],
            'colsample_bytree': params['colsample_bytree'],
            'reg_alpha': params['reg_alpha'],
            'seed': random_state
        }
        
        # Perform cross-validation
        cv_results = xgb.cv(
            xgb_params,
            dtrain,
            num_boost_round=params['n_estimators'],
            nfold=cv,
            metrics='rmse',
            early_stopping_rounds=10,
            verbose_eval=False,
            seed=random_state
        )
        
        # Extract best RMSE score from CV results
        score = cv_results['test-rmse-mean'].min()
        
        # Store results for this combination
        results.append({
            'params': params.copy(),
            'score': score
        })
        
        # Update best parameters if this is better
        if score < best_score:
            best_score = score
            best_params = params.copy()
        
        # Print progress every 10 combinations
        if i % 10 == 0:
            print(f"Completed {i}/{n_iter} combinations (Best RMSE so far: {best_score:.4f})")
    
    return best_params, best_score, results


def grid_search_xgboost(X_train, y_train, param_grid, cv=3, random_state=42, progress_interval=100):
    """
    Perform EXHAUSTIVE grid search using XGBoost's native CV.
    Tests ALL possible parameter combinations in the grid.
    
    WARNING: This can take many hours depending on grid size!
    
    Parameters:
    -----------
    X_train : DataFrame
        Training features
    y_train : Series
        Training target
    param_grid : dict
        Parameter grid to search (all combinations will be tested)
    cv : int
        Number of cross-validation folds
    random_state : int
        Random state for reproducibility
    progress_interval : int
        Print progress every N combinations (default: 100)
    
    Returns:
    --------
    tuple : (best_params, best_score, results)
    
    Example:
    --------
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.1, 0.2]
    }
    # This will test 2 × 2 × 2 = 8 combinations
    """
    # Calculate total number of combinations
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    total_combinations = np.prod([len(v) for v in param_values])
    
    print("=" * 70)
    print("EXHAUSTIVE GRID SEARCH")
    print("=" * 70)
    print(f"Total combinations to test: {total_combinations:,}")
    print(f"Parameters: {param_names}")
    for name, values in param_grid.items():
        print(f"  {name}: {len(values)} options")
    print(f"\nEstimated time: {total_combinations * 2 / 3600:.1f} - {total_combinations * 3 / 3600:.1f} hours")
    print("=" * 70)
    
    # Prepare DMatrix for XGBoost CV
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    # Initialize tracking variables
    best_score = float('inf')
    best_params = None
    results = []
    
    # Generate all combinations using nested loops
    def generate_combinations(param_dict):
        """Generate all parameter combinations from a dictionary of lists."""
        keys = list(param_dict.keys())
        values = [param_dict[k] for k in keys]
        
        # Use recursive approach to generate all combinations
        def recurse(index, current):
            if index == len(keys):
                yield dict(zip(keys, current))
            else:
                for value in values[index]:
                    yield from recurse(index + 1, current + [value])
        
        return recurse(0, [])
    
    print("\nStarting exhaustive grid search...")
    start_time = time.time()
    
    # Iterate through all parameter combinations
    for i, params in enumerate(generate_combinations(param_grid), 1):
        # Configure XGBoost CV parameters
        xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': params['max_depth'],
            'learning_rate': params['learning_rate'],
            'subsample': params['subsample'],
            'colsample_bytree': params['colsample_bytree'],
            'reg_alpha': params['reg_alpha'],
            'seed': random_state
        }
        
        # Perform cross-validation
        cv_results = xgb.cv(
            xgb_params,
            dtrain,
            num_boost_round=params['n_estimators'],
            nfold=cv,
            metrics='rmse',
            early_stopping_rounds=10,
            verbose_eval=False,
            seed=random_state
        )
        
        # Extract best RMSE score from CV results
        score = cv_results['test-rmse-mean'].min()
        
        # Store results for this combination
        results.append({
            'params': params.copy(),
            'score': score
        })
        
        # Update best parameters if this is better
        if score < best_score:
            best_score = score
            best_params = params.copy()
            print(f"  → New best! RMSE: {best_score:.4f} at combination {i}/{total_combinations}")
        
        # Print progress at specified intervals
        if i % progress_interval == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = (total_combinations - i) * avg_time
            print(f"Progress: {i}/{total_combinations} ({i/total_combinations*100:.1f}%) | "
                  f"Elapsed: {elapsed/60:.1f}min | "
                  f"Remaining: {remaining/60:.1f}min | "
                  f"Best RMSE: {best_score:.4f}")
    
    # Final summary
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("GRID SEARCH COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Combinations tested: {total_combinations:,}")
    print(f"Best RMSE: {best_score:.4f}")
    print("=" * 70)
    
    return best_params, best_score, results


def train_xgboost_model(X_train, y_train, params=None):
    """
    Train an XGBoost model with given parameters.
    
    Parameters:
    -----------
    X_train : DataFrame
        Training features
    y_train : Series
        Training target
    params : dict, optional
        Model parameters. If None, uses default parameters.
    
    Returns:
    --------
    XGBRegressor : Trained model
    """
    if params is None:
        # Default baseline parameters
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0,
            'random_state': 42
        }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(model, X_test, y_test, exposure_test):
    """
    Evaluate model and return predictions and metrics.
    
    Parameters:
    -----------
    model : XGBRegressor
        Trained model
    X_test : DataFrame
        Test features
    y_test : Series
        Test target
    exposure_test : Series
        Test exposure
    
    Returns:
    --------
    tuple : (predictions, metrics_dict)
    """
    y_pred = model.predict(X_test)
    metrics = calculate_all_metrics(y_test, y_pred, exposure_test)
    
    return y_pred, metrics


def prepare_lift_chart_data(y_test, y_pred, exposure_test):
    """
    Prepare data for lift chart visualization.
    
    Parameters:
    -----------
    y_test : array-like
        True target values
    y_pred : array-like
        Predicted values
    exposure_test : array-like
        Exposure values
    
    Returns:
    --------
    DataFrame : Data ready for lift_chart_modified function
    """
    test_results = pd.DataFrame({
        'act': y_test,
        'pred': y_pred,
        'exposure': exposure_test
    })
    test_results['act_weighted'] = test_results['act'] * test_results['exposure']
    test_results['pred_weighted'] = test_results['pred'] * test_results['exposure']
    
    return test_results


def analyze_feature_importance(model_baseline, model_tuned, feature_names, top_n=10):
    """
    Analyze and compare feature importance between two models.
    
    Parameters:
    -----------
    model_baseline : XGBRegressor
        Baseline model
    model_tuned : XGBRegressor
        Tuned model
    feature_names : list
        List of feature names
    top_n : int
        Number of top features to display
    
    Returns:
    --------
    DataFrame : Feature importance comparison
    """
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_baseline': model_baseline.feature_importances_,
        'importance_tuned': model_tuned.feature_importances_
    }).sort_values('importance_tuned', ascending=False)
    
    feature_importance_df['change'] = (
        feature_importance_df['importance_tuned'] - 
        feature_importance_df['importance_baseline']
    )
    
    print(f"=== TOP {top_n} FEATURES (TUNED MODEL) ===")
    print(feature_importance_df.head(top_n).to_string(index=False))
    
    print(f"\n=== BIGGEST INCREASES IN IMPORTANCE ===")
    print(feature_importance_df.nlargest(5, 'change')[
        ['feature', 'importance_baseline', 'importance_tuned', 'change']
    ].to_string(index=False))
    
    print(f"\n=== BIGGEST DECREASES IN IMPORTANCE ===")
    print(feature_importance_df.nsmallest(5, 'change')[
        ['feature', 'importance_baseline', 'importance_tuned', 'change']
    ].to_string(index=False))
    
    return feature_importance_df


def display_comparison_tables(baseline_params, tuned_params, baseline_metrics, tuned_metrics, config, X_shape):
    """
    Display comprehensive comparison tables for hyperparameters and metrics.
    
    Parameters:
    -----------
    baseline_params : dict
        Baseline model parameters
    tuned_params : dict
        Tuned model parameters
    baseline_metrics : dict
        Baseline model metrics
    tuned_metrics : dict
        Tuned model metrics
    config : dict
        Configuration dictionary
    X_shape : tuple
        Shape of feature matrix (n_samples, n_features)
    """
    # Table 1: Hyperparameters
    hyperparam_data = []
    for param in sorted(baseline_params.keys()):
        if param != 'random_state':  # Skip random_state
            hyperparam_data.append({
                'Hyperparameter': param,
                'Baseline Model': baseline_params[param],
                'Tuned Model': tuned_params[param],
                'Change': f"{tuned_params[param] - baseline_params[param]:+g}"
            })
    
    hyperparam_df = pd.DataFrame(hyperparam_data)
    
    # Table 2: Metrics
    metrics_data = []
    for metric in baseline_metrics.keys():
        baseline_val = baseline_metrics[metric]
        tuned_val = tuned_metrics[metric]
        
        # Calculate improvement percentage
        if metric in ['RMSE', 'MAE', 'Model_Power']:
            pct_change = ((baseline_val - tuned_val) / abs(baseline_val)) * 100
            improved = tuned_val < baseline_val
        else:
            pct_change = ((tuned_val - baseline_val) / abs(baseline_val)) * 100
            improved = tuned_val > baseline_val
        
        metrics_data.append({
            'Metric': metric,
            'Baseline Model': f"{baseline_val:.4f}",
            'Tuned Model': f"{tuned_val:.4f}",
            'Change (%)': f"{pct_change:+.2f}%",
            'Status': '✅ Improved' if improved else '⚠️  Declined'
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Display tables
    print("=" * 90)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("=" * 90)
    print(f"\nDataset: {X_shape[0]:,} samples, {X_shape[1]} features")
    print(f"Target: {config['target_col']} | Exposure: {config['exposure_col']}")
    
    print("\n" + "=" * 90)
    print("TABLE 1: HYPERPARAMETERS COMPARISON")
    print("=" * 90)
    print(hyperparam_df.to_string(index=False))
    
    print("\n" + "=" * 90)
    print("TABLE 2: METRICS COMPARISON")
    print("=" * 90)
    print(metrics_df.to_string(index=False))
    
    print("\n" + "=" * 90)
    print("Notes:")
    print("  • Lower is better: RMSE, MAE, Model_Power")
    print("  • Higher is better: R2, Fit_Quality")
    print("  • Change (%) shows improvement direction for each metric")
    print("=" * 90)


def display_final_summary(feature_importance_df, baseline_metrics, tuned_metrics, config, X_shape):
    """
    Display final model summary and assessment.
    
    Parameters:
    -----------
    feature_importance_df : DataFrame
        Feature importance comparison
    baseline_metrics : dict
        Baseline model metrics
    tuned_metrics : dict
        Tuned model metrics
    config : dict
        Configuration dictionary
    X_shape : tuple
        Shape of feature matrix
    """
    print("=" * 70)
    print("FINAL MODEL SUMMARY")
    print("=" * 70)
    
    print(f"\nDataset: {X_shape[0]:,} samples, {X_shape[1]} features")
    print(f"Target: {config['target_col']}")
    print(f"Exposure: {config['exposure_col']}")
    
    print("\n" + "=" * 70)
    print("TOP 5 FEATURES")
    print("=" * 70)
    for i, (_, row) in enumerate(feature_importance_df.head(5).iterrows(), 1):
        print(f"{i}. {row['feature']}: {row['importance_tuned']:.4f}")
    
    # Overall improvement assessment
    rmse_improved = tuned_metrics['RMSE'] < baseline_metrics['RMSE']
    quality_improved = tuned_metrics['Fit_Quality'] > baseline_metrics['Fit_Quality']
    power_improved = tuned_metrics['Model_Power'] < baseline_metrics['Model_Power']
    
    improvements = sum([rmse_improved, quality_improved, power_improved])
    
    print("\n" + "=" * 70)
    print("OVERALL ASSESSMENT")
    print("=" * 70)
    print(f"{'✅' if rmse_improved else '⚠️ '} RMSE: {'Improved' if rmse_improved else 'No improvement'}")
    print(f"{'✅' if quality_improved else '⚠️ '} Fit Quality: {'Improved' if quality_improved else 'No improvement'}")
    print(f"{'✅' if power_improved else '⚠️ '} Model Power: {'Improved' if power_improved else 'No improvement'}")
    print(f"\n{'🎉 SUCCESS' if improvements >= 2 else '⚠️  MIXED RESULTS'}: "
          f"Tuned model improved {improvements}/3 key metrics")
    
    print("\n✅ Analysis complete! The tuned model is available as 'best_model'.")
