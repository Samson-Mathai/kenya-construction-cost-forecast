"""
Kenya Construction Input Price Index — Forecasting Analysis
===========================================================
Source: KNBS Construction Input Price Indices (Base: December 2019 = 100)
Published quarterly by the Kenya National Bureau of Statistics
https://www.knbs.or.ke/button/construction-input-pi/

This script:
1. Loads and validates the compiled CIPI dataset
2. Performs exploratory data analysis with visualizations
3. Fits ARIMA and Exponential Smoothing models
4. Evaluates forecast accuracy on a holdout set (last 4–8 quarters)
5. Produces forward forecasts with confidence intervals
6. Translates index movements to AHP unit cost sensitivity
"""

import os
import sys
import warnings

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from scipy import stats

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# AHP unit cost assumptions (Ksh) — derived from KNBS Economic Survey 2026
# Verify these against original sources before citing
AHP_UNIT_COSTS = {
    'Affordable (1BR)': 1_500_000,
    'Affordable (2BR)': 2_000_000,
    'Affordable (3BR)': 3_000_000,
    'Social': 1_500_000,
    'Institutional': 2_300_000,
}

# Holdout configuration
HOLDOUT_QUARTERS = 6  # Last 6 quarters for out-of-sample testing

# Plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
COLORS = {
    'overall': '#1a5276',
    'building': '#2e86c1',
    'civil': '#148f77',
    'forecast': '#e74c3c',
    'ci': '#fadbd8',
    'holdout': '#f39c12',
}


# =============================================================================
# 1. DATA LOADING AND VALIDATION
# =============================================================================
def load_data():
    """Load and prepare the CIPI dataset."""
    filepath = os.path.join(DATA_DIR, 'construction_input_price_index.csv')
    df = pd.read_csv(filepath)

    # Create proper datetime index
    quarter_map = {'Q1': 1, 'Q2': 4, 'Q3': 7, 'Q4': 10}
    df['date'] = pd.to_datetime(
        df.apply(lambda r: f"{r['year']}-{quarter_map[r['quarter']]:02d}-01", axis=1)
    )
    df = df.sort_values('date').reset_index(drop=True)

    # Set quarterly period index for time series
    df['period_idx'] = pd.PeriodIndex(df['date'], freq='Q')

    print(f"Dataset loaded: {len(df)} quarterly observations")
    print(f"Period: {df['period'].iloc[0]} to {df['period'].iloc[-1]}")
    print(f"Overall index range: {df['overall_index'].min():.2f} – {df['overall_index'].max():.2f}")

    # Validate against known values
    known_checks = {
        '2025-Q2': 119.75,
        '2025-Q3': 121.27,
        '2026-Q2': 126.36,
    }
    print("\nValidation against known values:")
    for period_str, expected in known_checks.items():
        actual = df.loc[df['period'] == period_str, 'overall_index'].values
        if len(actual) > 0:
            status = 'OK' if abs(actual[0] - expected) < 0.01 else 'MISMATCH'
            print(f"  [{status}] {period_str}: expected {expected:.2f}, got {actual[0]:.2f}")

    return df


# =============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# =============================================================================
def exploratory_analysis(df):
    """Generate EDA visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Kenya Construction Input Price Index — Exploratory Analysis',
                 fontsize=16, fontweight='bold', y=0.98)

    # --- Plot 1: All three indices ---
    ax = axes[0, 0]
    ax.plot(df['date'], df['overall_index'], color=COLORS['overall'],
            linewidth=2.5, label='Overall', marker='o', markersize=4)
    ax.plot(df['date'], df['building_index'], color=COLORS['building'],
            linewidth=1.8, label='Building Works', marker='s', markersize=3, alpha=0.8)
    ax.plot(df['date'], df['civil_engineering_index'], color=COLORS['civil'],
            linewidth=1.8, label='Civil Engineering', marker='^', markersize=3, alpha=0.8)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Base (Dec 2019)')
    ax.set_title('Construction Input Price Indices', fontweight='bold')
    ax.set_ylabel('Index (Dec 2019 = 100)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Plot 2: Quarter-on-Quarter changes ---
    ax = axes[0, 1]
    df_temp = df.copy()
    df_temp['qoq_change'] = df_temp['overall_index'].pct_change() * 100
    colors_bar = ['#e74c3c' if x > 0 else '#2ecc71' for x in df_temp['qoq_change'].fillna(0)]
    ax.bar(df_temp['date'], df_temp['qoq_change'], width=60, color=colors_bar, alpha=0.8)
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_title('Quarter-on-Quarter Change (%)', fontweight='bold')
    ax.set_ylabel('Change (%)')
    ax.grid(True, alpha=0.3)

    # Annotate the Q2 2026 spike
    q2_2026 = df_temp[df_temp['period'] == '2026-Q2']
    if len(q2_2026) > 0:
        val = q2_2026['qoq_change'].values[0]
        ax.annotate(f'Q2 2026\n+{val:.1f}%\n(Fuel spike)',
                    xy=(q2_2026['date'].values[0], val),
                    xytext=(q2_2026['date'].values[0] - pd.Timedelta(days=300), val + 1),
                    arrowprops=dict(arrowstyle='->', color='#e74c3c'),
                    fontsize=9, color='#e74c3c', fontweight='bold')

    # --- Plot 3: Year-on-Year inflation rate ---
    ax = axes[1, 0]
    df_temp['yoy_change'] = df_temp['overall_index'].pct_change(periods=4) * 100
    ax.plot(df_temp['date'], df_temp['yoy_change'], color=COLORS['overall'],
            linewidth=2, marker='o', markersize=4)
    ax.fill_between(df_temp['date'], 0, df_temp['yoy_change'], alpha=0.15, color=COLORS['overall'])
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_title('Year-on-Year Construction Inflation (%)', fontweight='bold')
    ax.set_ylabel('YoY Change (%)')
    ax.grid(True, alpha=0.3)

    # --- Plot 4: Rolling statistics ---
    ax = axes[1, 1]
    rolling_mean = df['overall_index'].rolling(window=4, min_periods=1).mean()
    rolling_std = df['overall_index'].rolling(window=4, min_periods=1).std()
    ax.plot(df['date'], df['overall_index'], color=COLORS['overall'],
            alpha=0.5, linewidth=1, label='Actual')
    ax.plot(df['date'], rolling_mean, color=COLORS['forecast'],
            linewidth=2, label='4Q Rolling Mean')
    ax2 = ax.twinx()
    ax2.fill_between(df['date'], rolling_std, alpha=0.2, color=COLORS['holdout'])
    ax2.set_ylabel('Rolling Std Dev', color=COLORS['holdout'])
    ax.set_title('Trend & Volatility', fontweight='bold')
    ax.set_ylabel('Index')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '01_eda_overview.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 01_eda_overview.png")

    return df_temp


def stationarity_analysis(df):
    """Run ADF test and plot ACF/PACF for model selection."""
    series = df['overall_index'].dropna()

    # ADF test on levels
    adf_result = adfuller(series, autolag='AIC')
    print("\n=== Augmented Dickey-Fuller Test (Levels) ===")
    print(f"  Test statistic: {adf_result[0]:.4f}")
    print(f"  p-value:        {adf_result[1]:.4f}")
    print(f"  Lags used:      {adf_result[2]}")
    for key, value in adf_result[4].items():
        print(f"  Critical ({key}): {value:.4f}")
    stationary_levels = adf_result[1] < 0.05

    # ADF test on first differences
    diff_series = series.diff().dropna()
    adf_diff = adfuller(diff_series, autolag='AIC')
    print("\n=== Augmented Dickey-Fuller Test (First Difference) ===")
    print(f"  Test statistic: {adf_diff[0]:.4f}")
    print(f"  p-value:        {adf_diff[1]:.4f}")
    stationary_diff = adf_diff[1] < 0.05

    # ACF / PACF plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Stationarity & Autocorrelation Analysis', fontsize=14, fontweight='bold')

    plot_acf(series, ax=axes[0, 0], lags=12, title='ACF — Levels')
    plot_pacf(series, ax=axes[0, 1], lags=12, title='PACF — Levels', method='ywm')
    plot_acf(diff_series, ax=axes[1, 0], lags=12, title='ACF — First Difference')
    plot_pacf(diff_series, ax=axes[1, 1], lags=12, title='PACF — First Difference', method='ywm')

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '02_stationarity_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 02_stationarity_analysis.png")

    return {
        'stationary_levels': stationary_levels,
        'stationary_diff': stationary_diff,
        'adf_levels_pvalue': adf_result[1],
        'adf_diff_pvalue': adf_diff[1],
    }


# =============================================================================
# 3. FORECASTING MODELS
# =============================================================================
def fit_models(df, holdout_n=HOLDOUT_QUARTERS):
    """Fit ARIMA and Exponential Smoothing, evaluate on holdout."""
    series = df['overall_index'].dropna().values
    dates = df['date'].values[:len(series)]
    n = len(series)

    train = series[:n - holdout_n]
    test = series[n - holdout_n:]
    train_dates = dates[:n - holdout_n]
    test_dates = dates[n - holdout_n:]

    print(f"\n=== Model Fitting ===")
    print(f"Training set: {len(train)} quarters ({df['period'].iloc[0]} to {df['period'].iloc[n - holdout_n - 1]})")
    print(f"Holdout set:  {len(test)} quarters ({df['period'].iloc[n - holdout_n]} to {df['period'].iloc[-1]})")

    results = {}

    # --- Model 1: ARIMA ---
    # Try multiple orders and select by AIC
    best_aic = np.inf
    best_order = None
    best_arima = None

    arima_orders = [
        (1, 1, 0), (1, 1, 1), (2, 1, 0), (2, 1, 1),
        (0, 1, 1), (0, 1, 2), (1, 0, 0), (1, 0, 1),
        (2, 0, 0), (2, 0, 1), (0, 2, 1), (1, 2, 1),
    ]

    print("\nARIMA model selection:")
    for order in arima_orders:
        try:
            model = ARIMA(train, order=order)
            fitted = model.fit()
            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_order = order
                best_arima = fitted
                print(f"  ARIMA{order}: AIC = {fitted.aic:.2f} ← best so far")
        except Exception:
            continue

    if best_arima is not None:
        arima_forecast = best_arima.forecast(steps=holdout_n)
        arima_mae = mean_absolute_error(test, arima_forecast)
        arima_rmse = np.sqrt(mean_squared_error(test, arima_forecast))
        arima_mape = mean_absolute_percentage_error(test, arima_forecast) * 100

        results['ARIMA'] = {
            'order': best_order,
            'aic': best_aic,
            'forecast': arima_forecast,
            'mae': arima_mae,
            'rmse': arima_rmse,
            'mape': arima_mape,
            'model': best_arima,
        }
        print(f"\nBest ARIMA{best_order}:")
        print(f"  AIC:  {best_aic:.2f}")
        print(f"  MAE:  {arima_mae:.4f}")
        print(f"  RMSE: {arima_rmse:.4f}")
        print(f"  MAPE: {arima_mape:.2f}%")

    # --- Model 2: Exponential Smoothing (Holt-Winters) ---
    # Try additive and multiplicative variants
    ets_configs = [
        {'trend': 'add', 'seasonal': None, 'name': 'Holt (Additive Trend)'},
        {'trend': 'mul', 'seasonal': None, 'name': 'Holt (Multiplicative Trend)'},
        {'trend': 'add', 'damped_trend': True, 'seasonal': None, 'name': 'Damped Holt'},
    ]

    # Try seasonal models if we have enough data (need >= 2 full seasons)
    if len(train) >= 8:
        ets_configs.extend([
            {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 4, 'name': 'HW Additive'},
            {'trend': 'add', 'seasonal': 'mul', 'seasonal_periods': 4, 'name': 'HW Multiplicative'},
        ])

    print("\nExponential Smoothing model selection:")
    best_ets_aic = np.inf
    best_ets = None
    best_ets_name = None

    for config in ets_configs:
        name = config.pop('name')
        try:
            model = ExponentialSmoothing(train, **config)
            fitted = model.fit(optimized=True)
            aic = fitted.aic
            if aic < best_ets_aic:
                best_ets_aic = aic
                best_ets = fitted
                best_ets_name = name
                print(f"  {name}: AIC = {aic:.2f} ← best so far")
            config['name'] = name
        except Exception:
            config['name'] = name
            continue

    if best_ets is not None:
        ets_forecast = best_ets.forecast(holdout_n)
        ets_mae = mean_absolute_error(test, ets_forecast)
        ets_rmse = np.sqrt(mean_squared_error(test, ets_forecast))
        ets_mape = mean_absolute_percentage_error(test, ets_forecast) * 100

        results['ETS'] = {
            'name': best_ets_name,
            'aic': best_ets_aic,
            'forecast': ets_forecast,
            'mae': ets_mae,
            'rmse': ets_rmse,
            'mape': ets_mape,
            'model': best_ets,
        }
        print(f"\nBest ETS ({best_ets_name}):")
        print(f"  AIC:  {best_ets_aic:.2f}")
        print(f"  MAE:  {ets_mae:.4f}")
        print(f"  RMSE: {ets_rmse:.4f}")
        print(f"  MAPE: {ets_mape:.2f}%")

    # --- Select winner ---
    if results:
        winner_key = min(results, key=lambda k: results[k]['rmse'])
        print(f"\n>>> Best model by RMSE: {winner_key} (RMSE = {results[winner_key]['rmse']:.4f})")

    return results, train, test, train_dates, test_dates


def plot_holdout_results(df, results, train, test, train_dates, test_dates):
    """Plot holdout forecast vs actuals."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Forecast vs Actuals — Holdout Evaluation', fontsize=14, fontweight='bold')

    for idx, (model_name, res) in enumerate(results.items()):
        ax = axes[idx] if len(results) > 1 else axes[0]
        ax.plot(train_dates, train, color=COLORS['overall'], linewidth=2,
                label='Training', marker='o', markersize=3)
        ax.plot(test_dates, test, color=COLORS['holdout'], linewidth=2,
                label='Actual (Holdout)', marker='s', markersize=5)
        ax.plot(test_dates, res['forecast'], color=COLORS['forecast'], linewidth=2,
                label=f'{model_name} Forecast', linestyle='--', marker='^', markersize=5)

        # Shade error
        ax.fill_between(test_dates, test, res['forecast'], alpha=0.15, color=COLORS['forecast'])

        ax.set_title(f'{model_name}\nRMSE={res["rmse"]:.2f} | MAPE={res["mape"]:.1f}%', fontweight='bold')
        ax.set_ylabel('Index (Dec 2019 = 100)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '03_holdout_evaluation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 03_holdout_evaluation.png")


# =============================================================================
# 4. FORWARD FORECAST
# =============================================================================
def forward_forecast(df, results, n_ahead=4):
    """Produce forward forecast from the best model, refit on full data."""
    series = df['overall_index'].dropna().values
    last_date = df['date'].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=3),
                                 periods=n_ahead, freq='QS')

    # Pick best model by holdout RMSE
    winner_key = min(results, key=lambda k: results[k]['rmse'])
    print(f"\n=== Forward Forecast ({n_ahead} quarters) using {winner_key} ===")

    forecasts = {}

    # Refit ARIMA on full data
    if 'ARIMA' in results:
        order = results['ARIMA']['order']
        model = ARIMA(series, order=order)
        fitted = model.fit()
        forecast_result = fitted.get_forecast(steps=n_ahead)
        arima_fwd = forecast_result.predicted_mean
        ci = forecast_result.conf_int(alpha=0.1)  # 90% CI
        forecasts['ARIMA'] = {
            'forecast': arima_fwd,
            'ci_lower': ci[:, 0],
            'ci_upper': ci[:, 1],
        }

    # Refit ETS on full data
    if 'ETS' in results:
        ets_name = results['ETS']['name']
        # Determine config from name
        if 'HW Multiplicative' in ets_name:
            config = {'trend': 'add', 'seasonal': 'mul', 'seasonal_periods': 4}
        elif 'HW Additive' in ets_name:
            config = {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 4}
        elif 'Damped' in ets_name:
            config = {'trend': 'add', 'damped_trend': True, 'seasonal': None}
        elif 'Multiplicative' in ets_name:
            config = {'trend': 'mul', 'seasonal': None}
        else:
            config = {'trend': 'add', 'seasonal': None}

        model = ExponentialSmoothing(series, **config)
        fitted = model.fit(optimized=True)
        ets_fwd = fitted.forecast(n_ahead)

        # Bootstrap CI (simple approach using training residuals)
        residuals = fitted.resid
        residual_std = np.std(residuals)
        ci_lower = ets_fwd - 1.645 * residual_std * np.sqrt(np.arange(1, n_ahead + 1))
        ci_upper = ets_fwd + 1.645 * residual_std * np.sqrt(np.arange(1, n_ahead + 1))

        forecasts['ETS'] = {
            'forecast': ets_fwd,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
        }

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(14, 7))

    # Historical
    ax.plot(df['date'], df['overall_index'], color=COLORS['overall'], linewidth=2.5,
            label='Historical', marker='o', markersize=4)

    # Forecasts
    line_styles = {'ARIMA': '--', 'ETS': '-.'}
    forecast_colors = {'ARIMA': '#e74c3c', 'ETS': '#8e44ad'}

    for model_name, fwd in forecasts.items():
        color = forecast_colors.get(model_name, COLORS['forecast'])
        ax.plot(future_dates, fwd['forecast'], color=color, linewidth=2.5,
                linestyle=line_styles.get(model_name, '--'),
                label=f'{model_name} Forecast', marker='D', markersize=5)
        ax.fill_between(future_dates, fwd['ci_lower'], fwd['ci_upper'],
                        alpha=0.15, color=color, label=f'{model_name} 90% CI')

    # Annotate last actual and first forecast
    last_actual = df['overall_index'].iloc[-1]
    last_actual_date = df['date'].iloc[-1]
    ax.annotate(f'{last_actual:.2f}', xy=(last_actual_date, last_actual),
                xytext=(-60, 15), textcoords='offset points',
                fontsize=10, fontweight='bold', color=COLORS['overall'],
                arrowprops=dict(arrowstyle='->', color=COLORS['overall']))

    for model_name, fwd in forecasts.items():
        last_fwd = fwd['forecast'][-1]
        color = forecast_colors.get(model_name, COLORS['forecast'])
        ax.annotate(f'{last_fwd:.2f}', xy=(future_dates[-1], last_fwd),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=10, fontweight='bold', color=color)

    ax.axvline(x=last_actual_date, color='gray', linestyle=':', alpha=0.5)
    ax.set_title('Kenya Construction Input Price Index — Forward Forecast',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Index (Dec 2019 = 100)', fontsize=12)
    ax.set_xlabel('Quarter', fontsize=12)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '04_forward_forecast.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 04_forward_forecast.png")

    return forecasts, future_dates


# =============================================================================
# 5. AHP COST SENSITIVITY
# =============================================================================
def cost_sensitivity(df, forecasts, future_dates):
    """Translate forecast index changes into AHP unit cost impacts."""
    last_actual = df['overall_index'].dropna().iloc[-1]
    last_period = df['period'].iloc[-1]

    print(f"\n=== AHP Cost Sensitivity Analysis ===")
    print(f"Reference: Overall index = {last_actual:.2f} as at {last_period}")
    print(f"Base assumption: Index movement proportionally affects construction cost\n")

    # Use the best available forecast
    winner_key = list(forecasts.keys())[0]  # First one available
    fwd = forecasts[winner_key]['forecast']

    sensitivity_rows = []

    for i, (date, forecast_val) in enumerate(zip(future_dates, fwd)):
        pct_change = ((forecast_val - last_actual) / last_actual) * 100
        quarter_label = f"{date.year}-Q{(date.month - 1) // 3 + 1}"

        for unit_type, base_cost in AHP_UNIT_COSTS.items():
            cost_impact = base_cost * (pct_change / 100)
            new_cost = base_cost + cost_impact

            sensitivity_rows.append({
                'forecast_quarter': quarter_label,
                'forecast_index': round(forecast_val, 2),
                'pct_change_from_current': round(pct_change, 2),
                'unit_type': unit_type,
                'base_cost_ksh': base_cost,
                'cost_impact_ksh': round(cost_impact),
                'projected_cost_ksh': round(new_cost),
            })

    sensitivity_df = pd.DataFrame(sensitivity_rows)
    sensitivity_df.to_csv(os.path.join(OUTPUT_DIR, 'cost_sensitivity.csv'), index=False)

    # Summary table
    print(f"{'Quarter':<12} {'Index':<10} {'Change':<10} {'Unit Type':<22} {'Base Cost':>14} {'Impact':>14} {'New Cost':>14}")
    print('-' * 96)
    for _, row in sensitivity_df.iterrows():
        print(f"{row['forecast_quarter']:<12} {row['forecast_index']:<10.2f} "
              f"{row['pct_change_from_current']:>+8.2f}%  {row['unit_type']:<22} "
              f"Ksh {row['base_cost_ksh']:>10,}  Ksh {row['cost_impact_ksh']:>+10,}  "
              f"Ksh {row['projected_cost_ksh']:>10,}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Index forecast with cost annotations
    ax = axes[0]
    all_dates = list(df['date']) + list(future_dates)
    all_vals = list(df['overall_index'].dropna()) + list(fwd)
    split_idx = len(df['overall_index'].dropna())

    ax.plot(all_dates[:split_idx], all_vals[:split_idx], color=COLORS['overall'],
            linewidth=2.5, marker='o', markersize=3, label='Historical')
    ax.plot(all_dates[split_idx:], all_vals[split_idx:], color=COLORS['forecast'],
            linewidth=2.5, marker='D', markersize=5, linestyle='--', label='Forecast')
    ax.axvline(x=df['date'].iloc[-1], color='gray', linestyle=':', alpha=0.5)
    ax.set_title('Index Trajectory', fontweight='bold')
    ax.set_ylabel('Index')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Cost impact by unit type for the last forecast quarter
    ax = axes[1]
    last_q = sensitivity_df[sensitivity_df['forecast_quarter'] == sensitivity_df['forecast_quarter'].iloc[-1]]
    colors_bar = ['#2e86c1', '#1a5276', '#148f77', '#27ae60', '#8e44ad']
    bars = ax.barh(last_q['unit_type'], last_q['cost_impact_ksh'],
                   color=colors_bar[:len(last_q)], alpha=0.85)
    ax.set_title(f'Cost Impact per Unit — {last_q["forecast_quarter"].iloc[0]}',
                 fontweight='bold')
    ax.set_xlabel('Cost Impact (Ksh)')
    ax.grid(True, alpha=0.3, axis='x')

    for bar, val in zip(bars, last_q['cost_impact_ksh']):
        label = f'Ksh {val:+,}'
        ax.text(bar.get_width() + abs(bar.get_width()) * 0.05, bar.get_y() + bar.get_height() / 2,
                label, va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '05_cost_sensitivity.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: 05_cost_sensitivity.png")

    return sensitivity_df


# =============================================================================
# 6. SUMMARY REPORT
# =============================================================================
def generate_summary(df, stationarity, results, forecasts, future_dates, sensitivity_df):
    """Generate a text summary report."""
    report_lines = [
        "=" * 80,
        "KENYA CONSTRUCTION INPUT PRICE INDEX — FORECAST REPORT",
        "=" * 80,
        "",
        "DATA SOURCE",
        "-" * 40,
        f"Publisher:      Kenya National Bureau of Statistics (KNBS)",
        f"Series:         Construction Input Price Indices",
        f"Base period:    December 2019 = 100",
        f"Frequency:      Quarterly",
        f"Observations:   {len(df)} quarters ({df['period'].iloc[0]} to {df['period'].iloc[-1]})",
        f"URL:            https://www.knbs.or.ke/button/construction-input-pi/",
        "",
        "STATIONARITY",
        "-" * 40,
        f"ADF test (levels):     p = {stationarity['adf_levels_pvalue']:.4f} "
        f"({'stationary' if stationarity['stationary_levels'] else 'non-stationary'})",
        f"ADF test (1st diff):   p = {stationarity['adf_diff_pvalue']:.4f} "
        f"({'stationary' if stationarity['stationary_diff'] else 'non-stationary'})",
        "",
        "MODEL COMPARISON",
        "-" * 40,
    ]

    for name, res in results.items():
        model_label = f"ARIMA{res['order']}" if name == 'ARIMA' else f"ETS ({res['name']})"
        report_lines.extend([
            f"  {model_label}:",
            f"    AIC:  {res['aic']:.2f}",
            f"    MAE:  {res['mae']:.4f}",
            f"    RMSE: {res['rmse']:.4f}",
            f"    MAPE: {res['mape']:.2f}%",
            "",
        ])

    winner_key = min(results, key=lambda k: results[k]['rmse'])
    report_lines.extend([
        f"  Selected model: {winner_key} (lowest holdout RMSE)",
        "",
        "FORWARD FORECAST",
        "-" * 40,
    ])

    for name, fwd in forecasts.items():
        for i, (date, val) in enumerate(zip(future_dates, fwd['forecast'])):
            q = f"{date.year}-Q{(date.month - 1) // 3 + 1}"
            ci_lo = fwd['ci_lower'][i]
            ci_hi = fwd['ci_upper'][i]
            report_lines.append(f"  {name} {q}: {val:.2f} (90% CI: {ci_lo:.2f} – {ci_hi:.2f})")
        report_lines.append("")

    report_lines.extend([
        "COST SENSITIVITY (AHP Units)",
        "-" * 40,
        f"  Reference index: {df['overall_index'].dropna().iloc[-1]:.2f} ({df['period'].iloc[-1]})",
        "",
    ])

    last_q_data = sensitivity_df[
        sensitivity_df['forecast_quarter'] == sensitivity_df['forecast_quarter'].iloc[-1]
    ]
    for _, row in last_q_data.iterrows():
        report_lines.append(
            f"  {row['unit_type']}: Ksh {row['base_cost_ksh']:,} → "
            f"Ksh {row['projected_cost_ksh']:,} ({row['pct_change_from_current']:+.2f}%)"
        )

    report_lines.extend([
        "",
        "=" * 80,
        "Note: This forecast uses a simple time-series model on ~26 quarterly points.",
        "The confidence intervals widen rapidly. Treat as a directional indicator,",
        "not a precise prediction. The Q2 2026 fuel spike (transport/fuel sub-index",
        "+21.47%) is an outlier that heavily influences short-term forecasts.",
        "=" * 80,
    ])

    report_text = '\n'.join(report_lines)
    report_path = os.path.join(OUTPUT_DIR, 'forecast_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\nSaved: forecast_report.txt")
    print("\n" + report_text)

    return report_text


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("Kenya Construction Cost Forecast — Analysis Pipeline")
    print("=" * 60)

    # 1. Load data
    df = load_data()

    # 2. EDA
    df_eda = exploratory_analysis(df)

    # 3. Stationarity
    stationarity = stationarity_analysis(df)

    # 4. Fit models & evaluate
    results, train, test, train_dates, test_dates = fit_models(df)

    # 5. Plot holdout
    if results:
        plot_holdout_results(df, results, train, test, train_dates, test_dates)

    # 6. Forward forecast
    if results:
        forecasts, future_dates = forward_forecast(df, results, n_ahead=4)

        # 7. Cost sensitivity
        sensitivity_df = cost_sensitivity(df, forecasts, future_dates)

        # 8. Summary report
        generate_summary(df, stationarity, results, forecasts, future_dates, sensitivity_df)

    # 9. Save processed data
    df_eda.to_csv(os.path.join(OUTPUT_DIR, 'processed_index_data.csv'), index=False)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("Done.")


if __name__ == '__main__':
    main()
