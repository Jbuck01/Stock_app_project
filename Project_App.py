# =============================================================================
# Interactive Portfolio Analytics Application
# Financial Data Analytics II
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import minimize
import datetime
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Portfolio Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================
st.markdown("""
<style>
    /* Main background and font */
    .stApp { background-color: #f7f6f2; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f3f0ec;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #01696f !important;
        color: white !important;
    }
    /* Metric cards */
    [data-testid="metric-container"] {
        background: #f9f8f5;
        border: 1px solid #dcd9d5;
        border-radius: 8px;
        padding: 12px;
    }
    /* Section divider */
    .section-divider { border-top: 1px solid #dcd9d5; margin: 16px 0; }
    /* Info box */
    .info-box {
        background: #cedcd8;
        border-left: 4px solid #01696f;
        padding: 12px 16px;
        border-radius: 0 6px 6px 0;
        margin: 12px 0;
        color: #0c4e54;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTS
# =============================================================================
TRADING_DAYS_PER_YEAR = 252   # Number of trading days used for annualization
BENCHMARK_TICKER = "^GSPC"   # S&P 500 ticker used as benchmark

# =============================================================================
# CACHED DATA FUNCTIONS
# =============================================================================

@st.cache_data(ttl=3600)
def download_price_data(
    ticker_list: list,          # List of user-provided ticker symbols
    benchmark_ticker: str,      # Benchmark ticker (^GSPC)
    start_date: datetime.date,  # Start date for the historical data
    end_date: datetime.date     # End date for the historical data
) -> tuple:
    """
    Downloads adjusted closing prices for all tickers and the benchmark.
    Returns a tuple of (prices_df, benchmark_series, failed_tickers_list, warnings_list).
    """
    all_tickers_to_download = ticker_list + [benchmark_ticker]
    failed_tickers_list = []       # Tickers that could not be downloaded
    warnings_list = []             # Non-fatal warnings to show the user

    try:
        raw_download = yf.download(
            all_tickers_to_download,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False
        )
    except Exception as download_error:
        return None, None, ticker_list, [f"Download error: {str(download_error)}"]

    # Extract Close prices from MultiIndex or flat DataFrame
    if isinstance(raw_download.columns, pd.MultiIndex):
        close_prices_raw = raw_download["Close"]
    else:
        close_prices_raw = raw_download[["Close"]] if "Close" in raw_download.columns else raw_download

    if close_prices_raw is None or close_prices_raw.empty:
        return None, None, ticker_list, ["No data returned. Check tickers and date range."]

    # ---- Validate each ticker ----
    valid_stock_tickers = []   # Tickers that pass validation checks
    for ticker_symbol in ticker_list:
        if ticker_symbol not in close_prices_raw.columns:
            failed_tickers_list.append(ticker_symbol)
            continue
        ticker_series = close_prices_raw[ticker_symbol].dropna()
        if ticker_series.empty or len(ticker_series) < 30:
            failed_tickers_list.append(ticker_symbol)
            warnings_list.append(f"⚠️ {ticker_symbol}: insufficient data (fewer than 30 trading days).")
            continue
        valid_stock_tickers.append(ticker_symbol)

    if not valid_stock_tickers:
        return None, None, failed_tickers_list, warnings_list + ["No valid tickers after validation."]

    # ---- Build stock price DataFrame from valid tickers ----
    stock_prices_df = close_prices_raw[valid_stock_tickers].copy()

    # ---- Handle partial data: drop tickers with >5% missing values ----
    total_rows = len(stock_prices_df)
    tickers_to_keep = []
    for ticker_symbol in valid_stock_tickers:
        missing_pct = stock_prices_df[ticker_symbol].isna().sum() / total_rows
        if missing_pct > 0.05:
            failed_tickers_list.append(ticker_symbol)
            warnings_list.append(
                f"⚠️ {ticker_symbol} dropped: {missing_pct:.1%} missing values exceed 5% threshold."
            )
        else:
            tickers_to_keep.append(ticker_symbol)

    if not tickers_to_keep:
        return None, None, failed_tickers_list, warnings_list + ["All tickers had too much missing data."]

    # Truncate to overlapping date range across all remaining tickers
    stock_prices_df = stock_prices_df[tickers_to_keep].dropna(how="any")
    if len(stock_prices_df) < 30:
        return None, None, failed_tickers_list, warnings_list + ["Insufficient overlapping data after alignment."]

    if len(tickers_to_keep) < len(valid_stock_tickers):
        warnings_list.append("ℹ️ Data was truncated to the overlapping date range for all remaining tickers.")

    # ---- Benchmark series ----
    if benchmark_ticker in close_prices_raw.columns:
        benchmark_price_series = close_prices_raw[benchmark_ticker].reindex(stock_prices_df.index).ffill()
    else:
        benchmark_price_series = None
        warnings_list.append("⚠️ S&P 500 benchmark data could not be downloaded.")

    return stock_prices_df, benchmark_price_series, failed_tickers_list, warnings_list


@st.cache_data(ttl=3600)
def compute_daily_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes simple (arithmetic) daily returns from a prices DataFrame.
    Returns a DataFrame of daily simple returns with NaN in the first row dropped.
    """
    daily_returns_df = prices_df.pct_change().dropna()
    return daily_returns_df


@st.cache_data(ttl=3600)
def compute_summary_statistics(
    daily_returns_df: pd.DataFrame,  # DataFrame of daily simple returns for stocks
    benchmark_returns_series: pd.Series,  # Series of daily returns for the benchmark
    annual_rf_rate: float             # Annualized risk-free rate (e.g., 0.02 for 2%)
) -> pd.DataFrame:
    """
    Computes annualized summary statistics for each stock and the benchmark.
    Returns a styled DataFrame with: annualized mean, annualized vol, skewness,
    excess kurtosis, min daily return, max daily return, Sharpe ratio, Sortino ratio.
    """
    stats_records = []
    all_return_series = dict(daily_returns_df.items())
    if benchmark_returns_series is not None:
        aligned_bench = benchmark_returns_series.reindex(daily_returns_df.index).dropna()
        all_return_series["S&P 500"] = aligned_bench

    daily_rf_rate = annual_rf_rate / TRADING_DAYS_PER_YEAR  # Convert annual RF to daily

    for asset_name, return_series in all_return_series.items():
        clean_series = return_series.dropna()
        ann_mean_return = clean_series.mean() * TRADING_DAYS_PER_YEAR
        ann_volatility = clean_series.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        skewness_val = clean_series.skew()
        excess_kurtosis_val = clean_series.kurt()   # pandas .kurt() returns excess kurtosis
        min_daily = clean_series.min()
        max_daily = clean_series.max()

        # Sharpe ratio: (annualized return - annual RF) / annualized volatility
        sharpe_ratio_val = (ann_mean_return - annual_rf_rate) / ann_volatility if ann_volatility > 0 else np.nan

        # Sortino ratio: denominator is downside deviation (returns below daily RF only)
        downside_returns = clean_series[clean_series < daily_rf_rate] - daily_rf_rate
        downside_deviation_ann = downside_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        sortino_ratio_val = (ann_mean_return - annual_rf_rate) / downside_deviation_ann if downside_deviation_ann > 0 else np.nan

        stats_records.append({
            "Ticker": asset_name,
            "Ann. Return": f"{ann_mean_return:.2%}",
            "Ann. Volatility": f"{ann_volatility:.2%}",
            "Skewness": f"{skewness_val:.3f}",
            "Excess Kurtosis": f"{excess_kurtosis_val:.3f}",
            "Min Daily Return": f"{min_daily:.2%}",
            "Max Daily Return": f"{max_daily:.2%}",
            "Sharpe Ratio": f"{sharpe_ratio_val:.3f}",
            "Sortino Ratio": f"{sortino_ratio_val:.3f}",
        })

    stats_df = pd.DataFrame(stats_records).set_index("Ticker")
    return stats_df


# =============================================================================
# PORTFOLIO CALCULATION HELPERS
# =============================================================================

def compute_portfolio_variance(
    weights_array: np.ndarray,    # Array of portfolio weights summing to 1
    cov_matrix_df: pd.DataFrame   # Covariance matrix of daily returns
) -> float:
    """Returns annualized portfolio variance using the quadratic form w'Σw."""
    portfolio_variance_daily = weights_array @ cov_matrix_df.values @ weights_array
    return portfolio_variance_daily * TRADING_DAYS_PER_YEAR


def compute_portfolio_return(
    weights_array: np.ndarray,      # Array of portfolio weights
    mean_daily_returns: np.ndarray  # Array of mean daily returns per asset
) -> float:
    """Returns annualized portfolio expected return."""
    return float(weights_array @ mean_daily_returns) * TRADING_DAYS_PER_YEAR


def compute_sharpe_ratio(
    weights_array: np.ndarray,      # Portfolio weights
    mean_daily_returns: np.ndarray, # Mean daily returns per asset
    cov_matrix_df: pd.DataFrame,    # Covariance matrix
    annual_rf_rate: float           # Annualized risk-free rate
) -> float:
    """Computes annualized Sharpe ratio for a given set of portfolio weights."""
    port_return = compute_portfolio_return(weights_array, mean_daily_returns)
    port_vol = np.sqrt(compute_portfolio_variance(weights_array, cov_matrix_df))
    return (port_return - annual_rf_rate) / port_vol if port_vol > 0 else -np.inf


def compute_sortino_ratio(
    weights_array: np.ndarray,      # Portfolio weights
    daily_returns_df: pd.DataFrame, # DataFrame of daily returns
    annual_rf_rate: float           # Annualized risk-free rate
) -> float:
    """Computes annualized Sortino ratio using downside deviation."""
    daily_rf_rate = annual_rf_rate / TRADING_DAYS_PER_YEAR
    portfolio_daily_returns = daily_returns_df.values @ weights_array
    ann_port_return = portfolio_daily_returns.mean() * TRADING_DAYS_PER_YEAR
    downside_returns = portfolio_daily_returns[portfolio_daily_returns < daily_rf_rate] - daily_rf_rate
    downside_dev_ann = downside_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    return (ann_port_return - annual_rf_rate) / downside_dev_ann if downside_dev_ann > 0 else np.nan


def compute_max_drawdown(portfolio_cumulative_returns: pd.Series) -> float:
    """
    Computes maximum drawdown from a cumulative wealth index series.
    Returns the maximum drawdown as a negative fraction (e.g., -0.35 for -35%).
    """
    rolling_peak = portfolio_cumulative_returns.cummax()
    drawdown_series = (portfolio_cumulative_returns - rolling_peak) / rolling_peak
    return float(drawdown_series.min())


def compute_drawdown_series(price_or_wealth_series: pd.Series) -> pd.Series:
    """Returns a time series of percentage drawdown from the rolling peak."""
    rolling_peak = price_or_wealth_series.cummax()
    drawdown = (price_or_wealth_series - rolling_peak) / rolling_peak
    return drawdown


def compute_portfolio_metrics(
    weights_array: np.ndarray,      # Portfolio weights
    daily_returns_df: pd.DataFrame, # DataFrame of daily simple returns
    cov_matrix_df: pd.DataFrame,    # Covariance matrix
    annual_rf_rate: float           # Annualized risk-free rate
) -> dict:
    """
    Returns a dictionary of all key portfolio performance metrics.
    Keys: ann_return, ann_volatility, sharpe, sortino, max_drawdown, weights.
    """
    mean_daily_returns = daily_returns_df.mean().values
    port_return = compute_portfolio_return(weights_array, mean_daily_returns)
    port_vol = np.sqrt(compute_portfolio_variance(weights_array, cov_matrix_df))
    sharpe = compute_sharpe_ratio(weights_array, mean_daily_returns, cov_matrix_df, annual_rf_rate)
    sortino = compute_sortino_ratio(weights_array, daily_returns_df, annual_rf_rate)

    portfolio_daily_returns_series = daily_returns_df.values @ weights_array
    portfolio_wealth_index = (1 + portfolio_daily_returns_series).cumprod()
    max_dd = compute_max_drawdown(pd.Series(portfolio_wealth_index))

    return {
        "ann_return": port_return,
        "ann_volatility": port_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "weights": weights_array
    }


@st.cache_data(ttl=3600)
def optimize_gmv_portfolio(
    mean_daily_returns_tuple: tuple,  # Mean daily returns (as tuple for caching)
    cov_matrix_tuple: tuple,          # Covariance matrix values (as tuple for caching)
    ticker_names_tuple: tuple         # Ticker symbols (as tuple for caching)
) -> tuple:
    """
    Computes Global Minimum Variance (GMV) portfolio weights.
    Uses scipy.optimize.minimize with no-short-selling constraints.
    Returns (weights_array, success_bool, message_str).
    """
    n_assets = len(ticker_names_tuple)
    cov_matrix_arr = np.array(cov_matrix_tuple)
    initial_weights = np.ones(n_assets) / n_assets  # Equal-weight starting point

    # Objective: minimize portfolio variance
    def gmv_objective(w):
        return w @ cov_matrix_arr @ w

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]  # Weights sum to 1
    bounds = [(0.0, 1.0)] * n_assets                                  # No short selling

    optimization_result = minimize(
        gmv_objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12}
    )

    if optimization_result.success:
        gmv_weights = np.maximum(optimization_result.x, 0)
        gmv_weights = gmv_weights / gmv_weights.sum()
        return gmv_weights, True, "GMV optimization successful."
    else:
        return initial_weights, False, f"GMV optimization failed: {optimization_result.message}"


@st.cache_data(ttl=3600)
def optimize_tangency_portfolio(
    mean_daily_returns_tuple: tuple,  # Mean daily returns (as tuple for caching)
    cov_matrix_tuple: tuple,          # Covariance matrix values (as tuple for caching)
    ticker_names_tuple: tuple,        # Ticker symbols (as tuple for caching)
    annual_rf_rate: float             # Annualized risk-free rate
) -> tuple:
    """
    Computes Maximum Sharpe Ratio (Tangency) portfolio weights.
    Minimizes the negative Sharpe ratio using scipy.optimize.minimize.
    Returns (weights_array, success_bool, message_str).
    """
    n_assets = len(ticker_names_tuple)
    mean_daily_arr = np.array(mean_daily_returns_tuple)
    cov_matrix_arr = np.array(cov_matrix_tuple)
    initial_weights = np.ones(n_assets) / n_assets

    # Objective: minimize negative Sharpe ratio (i.e., maximize Sharpe)
    def negative_sharpe_objective(w):
        port_return = float(w @ mean_daily_arr) * TRADING_DAYS_PER_YEAR
        port_var = w @ cov_matrix_arr @ w * TRADING_DAYS_PER_YEAR
        port_vol = np.sqrt(port_var)
        if port_vol == 0:
            return np.inf
        return -(port_return - annual_rf_rate) / port_vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.0, 1.0)] * n_assets

    optimization_result = minimize(
        negative_sharpe_objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12}
    )

    if optimization_result.success:
        tangency_weights = np.maximum(optimization_result.x, 0)
        tangency_weights = tangency_weights / tangency_weights.sum()
        return tangency_weights, True, "Tangency optimization successful."
    else:
        return initial_weights, False, f"Tangency optimization failed: {optimization_result.message}"


@st.cache_data(ttl=3600)
def compute_efficient_frontier(
    mean_daily_returns_tuple: tuple,  # Mean daily returns (as tuple for caching)
    cov_matrix_tuple: tuple,          # Covariance matrix values (as tuple for caching)
    ticker_names_tuple: tuple,        # Ticker symbols (as tuple for caching)
    n_points: int = 80                # Number of points to trace along the frontier
) -> tuple:
    """
    Computes the efficient frontier by solving minimum-variance optimization
    at each target return level. Returns (frontier_vols, frontier_returns) as arrays.
    """
    n_assets = len(ticker_names_tuple)
    mean_daily_arr = np.array(mean_daily_returns_tuple)
    cov_matrix_arr = np.array(cov_matrix_tuple)

    ann_mean_returns = mean_daily_arr * TRADING_DAYS_PER_YEAR

    # Define the range of target returns between min and max individual asset returns
    target_return_min = ann_mean_returns.min()
    target_return_max = ann_mean_returns.max()
    target_returns_array = np.linspace(target_return_min, target_return_max, n_points)

    frontier_vols = []     # Annualized volatility at each efficient point
    frontier_rets = []     # Annualized return at each efficient point
    initial_weights = np.ones(n_assets) / n_assets

    for target_ret in target_returns_array:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, r=target_ret: (w @ mean_daily_arr) * TRADING_DAYS_PER_YEAR - r}
        ]
        bounds = [(0.0, 1.0)] * n_assets
        result = minimize(
            lambda w: w @ cov_matrix_arr @ w,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-10}
        )
        if result.success:
            port_vol_ann = np.sqrt(result.fun * TRADING_DAYS_PER_YEAR)
            frontier_vols.append(port_vol_ann)
            frontier_rets.append(target_ret)

    return np.array(frontier_vols), np.array(frontier_rets)


def compute_risk_contribution(
    weights_array: np.ndarray,    # Portfolio weights
    cov_matrix_df: pd.DataFrame   # Covariance matrix of daily returns
) -> np.ndarray:
    """
    Computes Percentage Risk Contribution (PRC) for each asset.
    Formula: PRC_i = w_i * (Σw)_i / σ²_p
    Returns array summing to 1 (each element = fraction of total portfolio risk).
    """
    cov_matrix_arr = cov_matrix_df.values
    portfolio_variance_daily = weights_array @ cov_matrix_arr @ weights_array
    marginal_risk_contributions = cov_matrix_arr @ weights_array  # (Σw) vector
    risk_contributions = weights_array * marginal_risk_contributions
    prc_array = risk_contributions / portfolio_variance_daily
    return prc_array


# =============================================================================
# PLOTTING HELPERS
# =============================================================================

CHART_COLOR_SEQUENCE = px.colors.qualitative.Safe   # Consistent color palette for all charts
BENCHMARK_COLOR = "#636363"                          # Gray for the S&P 500 benchmark


def plot_cumulative_wealth(
    daily_returns_df: pd.DataFrame,      # Daily returns for selected stocks
    benchmark_returns_series: pd.Series, # Daily returns for S&P 500
    selected_tickers: list,              # Tickers to display (user-selected)
    initial_value: float = 10000.0       # Starting investment value in dollars
) -> go.Figure:
    """Creates a cumulative wealth index chart starting at initial_value."""
    fig = go.Figure()
    color_map = {t: CHART_COLOR_SEQUENCE[i % len(CHART_COLOR_SEQUENCE)] for i, t in enumerate(daily_returns_df.columns)}

    for ticker in selected_tickers:
        if ticker in daily_returns_df.columns:
            cumulative_wealth = initial_value * (1 + daily_returns_df[ticker]).cumprod()
            fig.add_trace(go.Scatter(
                x=cumulative_wealth.index,
                y=cumulative_wealth.values,
                name=ticker,
                line=dict(color=color_map[ticker], width=2),
                mode="lines"
            ))

    if benchmark_returns_series is not None:
        bench_aligned = benchmark_returns_series.reindex(daily_returns_df.index).dropna()
        bench_wealth = initial_value * (1 + bench_aligned).cumprod()
        fig.add_trace(go.Scatter(
            x=bench_wealth.index,
            y=bench_wealth.values,
            name="S&P 500",
            line=dict(color=BENCHMARK_COLOR, width=2, dash="dash"),
            mode="lines"
        ))

    fig.update_layout(
        title="Cumulative Wealth Index (Starting Value: $10,000)",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_distribution(
    return_series: pd.Series,  # Daily return series for the selected stock
    ticker_name: str,          # Name of the stock
    plot_type: str             # Either "Histogram" or "Q-Q Plot"
) -> go.Figure:
    """Creates either a histogram with normal overlay, or a Q-Q plot."""
    if plot_type == "Histogram":
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=return_series,
            nbinsx=60,
            name="Daily Returns",
            histnorm="probability density",
            marker_color="#01696f",
            opacity=0.7
        ))
        # Overlay fitted normal distribution curve
        x_range = np.linspace(return_series.min(), return_series.max(), 200)
        fitted_pdf = stats.norm.pdf(x_range, loc=return_series.mean(), scale=return_series.std())
        fig.add_trace(go.Scatter(
            x=x_range, y=fitted_pdf,
            name="Fitted Normal",
            line=dict(color="#a12c7b", width=2)
        ))
        fig.update_layout(
            title=f"{ticker_name} — Daily Return Distribution",
            xaxis_title="Daily Return",
            yaxis_title="Probability Density",
            template="plotly_white"
        )
    else:  # Q-Q Plot
        qq_osm, qq_osr = stats.probplot(return_series, dist="norm", fit=False)
        theoretical_quantiles = qq_osm   # Theoretical normal quantiles
        sample_quantiles = qq_osr        # Observed return quantiles

        # Reference line (perfect normal fit)
        min_q, max_q = min(theoretical_quantiles), max(theoretical_quantiles)
        ref_line_y = [min_q * return_series.std() + return_series.mean(),
                      max_q * return_series.std() + return_series.mean()]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=theoretical_quantiles, y=sample_quantiles,
            mode="markers",
            name="Observed Quantiles",
            marker=dict(color="#01696f", size=4, opacity=0.7)
        ))
        fig.add_trace(go.Scatter(
            x=[min_q, max_q], y=ref_line_y,
            mode="lines",
            name="Normal Reference Line",
            line=dict(color="#a12c7b", width=2, dash="dash")
        ))
        fig.update_layout(
            title=f"{ticker_name} — Q-Q Plot vs Normal Distribution",
            xaxis_title="Theoretical Normal Quantiles",
            yaxis_title="Sample Quantiles",
            template="plotly_white"
        )
    return fig


def plot_correlation_heatmap(daily_returns_df: pd.DataFrame) -> go.Figure:
    """Creates an annotated correlation heatmap with a diverging color scale."""
    correlation_matrix = daily_returns_df.corr()
    tickers_list = list(correlation_matrix.columns)
    corr_values = correlation_matrix.values

    annotations_text = [[f"{corr_values[r][c]:.2f}" for c in range(len(tickers_list))]
                        for r in range(len(tickers_list))]

    fig = go.Figure(data=go.Heatmap(
        z=corr_values,
        x=tickers_list,
        y=tickers_list,
        text=annotations_text,
        texttemplate="%{text}",
        colorscale="RdBu",     # Diverging scale: red=negative, white=zero, blue=positive
        zmid=0,                # Center the diverging scale at zero
        zmin=-1,
        zmax=1,
        colorbar=dict(title="Correlation")
    ))
    fig.update_layout(
        title="Pairwise Correlation Matrix of Daily Returns",
        xaxis_title="Ticker",
        yaxis_title="Ticker",
        template="plotly_white"
    )
    return fig


def plot_efficient_frontier_chart(
    frontier_vols: np.ndarray,           # Efficient frontier volatilities
    frontier_rets: np.ndarray,           # Efficient frontier returns
    portfolio_points: dict,              # Named portfolio points {name: (vol, ret)}
    individual_stock_points: dict,       # Individual stock points {ticker: (vol, ret)}
    benchmark_point: tuple,             # (vol, ret) for S&P 500
    tangency_vol: float,                # Tangency portfolio volatility
    tangency_ret: float,                # Tangency portfolio return
    annual_rf_rate: float               # Annualized risk-free rate
) -> go.Figure:
    """Creates the efficient frontier chart with all required overlay points and the CAL."""
    fig = go.Figure()

    # Efficient frontier curve
    fig.add_trace(go.Scatter(
        x=frontier_vols, y=frontier_rets,
        mode="lines",
        name="Efficient Frontier",
        line=dict(color="#01696f", width=3),
        hovertemplate="Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>"
    ))

    # Capital Allocation Line (CAL): from risk-free rate through tangency portfolio
    cal_x_end = max(frontier_vols) * 1.3
    cal_slope = (tangency_ret - annual_rf_rate) / tangency_vol if tangency_vol > 0 else 0
    cal_vols = np.linspace(0, cal_x_end, 100)
    cal_rets = annual_rf_rate + cal_slope * cal_vols
    fig.add_trace(go.Scatter(
        x=cal_vols, y=cal_rets,
        mode="lines",
        name="Capital Allocation Line (CAL)",
        line=dict(color="#d19900", width=2, dash="dash"),
        hovertemplate="Vol: %{x:.2%}<br>CAL Return: %{y:.2%}<extra></extra>"
    ))

    # Individual stock points
    stock_colors = {"individual": "#006494"}
    for stock_ticker, (stock_vol, stock_ret) in individual_stock_points.items():
        fig.add_trace(go.Scatter(
            x=[stock_vol], y=[stock_ret],
            mode="markers+text",
            name=stock_ticker,
            marker=dict(color="#006494", size=9, symbol="circle"),
            text=[stock_ticker],
            textposition="top right",
            textfont=dict(size=10)
        ))

    # S&P 500 benchmark point
    if benchmark_point is not None:
        fig.add_trace(go.Scatter(
            x=[benchmark_point[0]], y=[benchmark_point[1]],
            mode="markers+text",
            name="S&P 500",
            marker=dict(color=BENCHMARK_COLOR, size=12, symbol="diamond"),
            text=["S&P 500"],
            textposition="top right",
            textfont=dict(size=10)
        ))

    # Named portfolio overlay points
    overlay_styles = {
        "Equal-Weight": dict(color="#437a22", size=14, symbol="star"),
        "GMV Portfolio": dict(color="#7a39bb", size=14, symbol="square"),
        "Tangency Portfolio": dict(color="#da7101", size=14, symbol="star-triangle-up"),
        "Custom Portfolio": dict(color="#a12c7b", size=14, symbol="pentagon"),
    }
    for port_name, (port_vol, port_ret) in portfolio_points.items():
        style = overlay_styles.get(port_name, dict(color="#333333", size=12, symbol="circle"))
        fig.add_trace(go.Scatter(
            x=[port_vol], y=[port_ret],
            mode="markers+text",
            name=port_name,
            marker=style,
            text=[port_name],
            textposition="top left",
            textfont=dict(size=10)
        ))

    # Risk-free rate point
    fig.add_trace(go.Scatter(
        x=[0], y=[annual_rf_rate],
        mode="markers+text",
        name="Risk-Free Rate",
        marker=dict(color="#636363", size=10, symbol="x"),
        text=["Rf"],
        textposition="top right",
        textfont=dict(size=10)
    ))

    fig.update_layout(
        title="Efficient Frontier with Capital Allocation Line",
        xaxis_title="Annualized Volatility (σ)",
        yaxis_title="Annualized Expected Return",
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
        template="plotly_white",
        legend=dict(orientation="v", yanchor="top", y=0.95, xanchor="left", x=1.01)
    )
    return fig


# =============================================================================
# SIDEBAR: INPUTS AND CONFIGURATION
# =============================================================================

with st.sidebar:
    st.markdown("## ⚙️ Portfolio Configuration")
    st.markdown("---")

    # ---- Ticker Input ----
    raw_ticker_input = st.text_input(
        "Stock Tickers (3–10, comma-separated)",
        value="AAPL, MSFT, GOOGL, AMZN, NVDA",
        help="Enter 3 to 10 valid stock ticker symbols separated by commas."
    )

    # ---- Date Range ----
    today_date = datetime.date.today()
    default_start_date = today_date - datetime.timedelta(days=5 * 365)   # 5 years back by default
    default_end_date = today_date

    selected_start_date = st.date_input(
        "Start Date",
        value=default_start_date,
        max_value=today_date - datetime.timedelta(days=730)   # Force at least 2 years
    )
    selected_end_date = st.date_input(
        "End Date",
        value=default_end_date,
        max_value=today_date
    )

    # ---- Risk-Free Rate ----
    annual_rf_rate_pct = st.number_input(
        "Annualized Risk-Free Rate (%)",
        min_value=0.0,
        max_value=20.0,
        value=2.0,
        step=0.1,
        format="%.1f",
        help="Used in all Sharpe and Sortino ratio calculations."
    )
    annual_rf_rate = annual_rf_rate_pct / 100.0   # Convert percentage to decimal

    # ---- Load Data Button ----
    load_data_button = st.button("🔄 Load / Refresh Data", type="primary", use_container_width=True)

    st.markdown("---")

    # ---- About / Methodology Section ----
    with st.expander("📖 About & Methodology", expanded=False):
        st.markdown("""
        **Data Source**: Yahoo Finance via `yfinance`.
        Adjusted closing prices account for dividends and stock splits.

        **Returns**: Simple (arithmetic) daily returns: `r_t = (P_t / P_{t-1}) - 1`.
        Log returns are **not** used because they are not additive across assets.

        **Annualization**: 252 trading days per year.
        - Annual return = mean daily return × 252
        - Annual volatility = daily std dev × √252

        **Sharpe Ratio**: `(Ann. Return − Rf) / Ann. Volatility`

        **Sortino Ratio**: `(Ann. Return − Rf) / Downside Deviation`
        Downside deviation uses only returns below the daily risk-free rate.

        **Portfolio Variance**: Full quadratic form `w'Σw`.
        Weighted-average of individual variances is **not** used.

        **Optimization**: `scipy.optimize.minimize` (SLSQP method)
        with no-short-selling bounds `[0, 1]` and sum-to-one equality constraint.

        **Efficient Frontier**: Constrained optimization at each target return level.
        Random simulation is **not** used.

        **Risk Contribution (PRC)**:
        `PRC_i = w_i × (Σw)_i / σ²_p`. Values sum to 1.

        **Benchmark**: S&P 500 (^GSPC) — displayed for comparison only,
        never included in portfolio optimization.
        """)

# =============================================================================
# INPUT VALIDATION
# =============================================================================

# Parse and validate ticker symbols entered by the user
raw_ticker_parts = [t.strip().upper() for t in raw_ticker_input.split(",") if t.strip()]
user_tickers_list = list(dict.fromkeys(raw_ticker_parts))   # Remove duplicates, preserve order

input_validation_errors = []   # List of validation error messages

if len(user_tickers_list) < 3:
    input_validation_errors.append("❌ Please enter at least 3 ticker symbols.")
if len(user_tickers_list) > 10:
    input_validation_errors.append("❌ Please enter no more than 10 ticker symbols.")
if (selected_end_date - selected_start_date).days < 730:
    input_validation_errors.append("❌ Date range must be at least 2 years (730 days).")
if selected_end_date <= selected_start_date:
    input_validation_errors.append("❌ End date must be after start date.")

# =============================================================================
# MAIN APP HEADER
# =============================================================================
st.title("📈 Interactive Portfolio Analytics")
st.caption("Financial Data Analytics II | Build, analyze, and optimize equity portfolios in real time.")

# =============================================================================
# SESSION STATE MANAGEMENT
# =============================================================================
# We store loaded data in session_state so it persists across widget interactions.
if "prices_df" not in st.session_state:
    st.session_state["prices_df"] = None               # Adjusted closing prices DataFrame
if "benchmark_series" not in st.session_state:
    st.session_state["benchmark_series"] = None        # S&P 500 adjusted close prices Series
if "daily_returns_df" not in st.session_state:
    st.session_state["daily_returns_df"] = None        # Daily simple returns DataFrame
if "benchmark_returns_series" not in st.session_state:
    st.session_state["benchmark_returns_series"] = None  # S&P 500 daily returns Series
if "valid_tickers" not in st.session_state:
    st.session_state["valid_tickers"] = []             # Tickers successfully loaded
if "data_loaded" not in st.session_state:
    st.session_state["data_loaded"] = False            # Flag: has data been loaded?

# =============================================================================
# DATA LOADING
# =============================================================================

if input_validation_errors:
    for validation_error_msg in input_validation_errors:
        st.error(validation_error_msg)
    st.stop()

if load_data_button:
    with st.spinner("⏳ Downloading market data from Yahoo Finance…"):
        (
            loaded_prices_df,
            loaded_benchmark_series,
            failed_tickers_from_download,
            download_warnings_list
        ) = download_price_data(
            user_tickers_list,
            BENCHMARK_TICKER,
            selected_start_date,
            selected_end_date
        )

    # Show any download warnings
    for warning_msg in download_warnings_list:
        st.warning(warning_msg)

    # Show which tickers failed
    if failed_tickers_from_download:
        st.error(f"❌ The following tickers could not be loaded: **{', '.join(failed_tickers_from_download)}**. "
                 "They may be invalid, delisted, or have insufficient history.")

    if loaded_prices_df is None or loaded_prices_df.empty:
        st.error("❌ Could not load any valid data. Please check your tickers and date range.")
        st.stop()

    if len(loaded_prices_df.columns) < 3:
        st.error(f"❌ At least 3 valid tickers are required after data validation. "
                 f"Only {len(loaded_prices_df.columns)} loaded: {list(loaded_prices_df.columns)}.")
        st.stop()

    # Store in session state
    st.session_state["prices_df"] = loaded_prices_df
    st.session_state["benchmark_series"] = loaded_benchmark_series
    st.session_state["daily_returns_df"] = compute_daily_returns(loaded_prices_df)
    if loaded_benchmark_series is not None:
        st.session_state["benchmark_returns_series"] = compute_daily_returns(
            loaded_benchmark_series.to_frame("^GSPC")
        )["^GSPC"]
    else:
        st.session_state["benchmark_returns_series"] = None
    st.session_state["valid_tickers"] = list(loaded_prices_df.columns)
    st.session_state["data_loaded"] = True

    st.success(f"✅ Data loaded for: **{', '.join(st.session_state['valid_tickers'])}** "
               f"({len(loaded_prices_df)} trading days, "
               f"{loaded_prices_df.index[0].date()} → {loaded_prices_df.index[-1].date()})")

# ---- Guard: require data to be loaded before rendering any tab ----
if not st.session_state["data_loaded"]:
    st.info("👈 Configure your portfolio in the sidebar and click **Load / Refresh Data** to begin.")
    st.stop()

# =============================================================================
# Convenience aliases pointing to session state data
# =============================================================================
prices_df = st.session_state["prices_df"]                           # Adjusted closing prices
benchmark_series = st.session_state["benchmark_series"]             # S&P 500 closing prices
daily_returns_df = st.session_state["daily_returns_df"]             # Daily simple returns
benchmark_returns_series = st.session_state["benchmark_returns_series"]  # S&P 500 daily returns
valid_tickers = st.session_state["valid_tickers"]                   # List of valid ticker strings

# Pre-compute covariance matrix and mean returns (used in multiple tabs)
cov_matrix_df = daily_returns_df.cov()                              # Daily covariance matrix
mean_daily_returns_arr = daily_returns_df.mean().values             # Mean daily returns array

# =============================================================================
# TABS: Main navigation structure
# =============================================================================
(
    tab_exploratory,   # Tab 1: Return Computation & Exploratory Analysis
    tab_risk,          # Tab 2: Risk Analysis
    tab_correlation,   # Tab 3: Correlation & Covariance Analysis
    tab_portfolio,     # Tab 4: Portfolio Construction & Optimization
    tab_sensitivity    # Tab 5: Estimation Window Sensitivity
) = st.tabs([
    "📊 Exploratory Analysis",
    "⚠️ Risk Analysis",
    "🔗 Correlation & Covariance",
    "🏦 Portfolio Optimization",
    "🔬 Sensitivity Analysis"
])

# =============================================================================
# TAB 1: RETURN COMPUTATION & EXPLORATORY ANALYSIS (Section 2.2)
# =============================================================================
with tab_exploratory:
    st.header("Return Computation & Exploratory Analysis")

    # ---- 2.2.1 Summary Statistics Table ----
    st.subheader("Summary Statistics")
    with st.spinner("Computing summary statistics…"):
        summary_stats_df = compute_summary_statistics(
            daily_returns_df,
            benchmark_returns_series,
            annual_rf_rate
        )
    st.dataframe(summary_stats_df, use_container_width=True)

    st.markdown("---")

    # ---- 2.2.2 Cumulative Wealth Index ----
    st.subheader("Cumulative Wealth Index ($10,000 Starting Value)")

    wealth_ticker_selection = st.multiselect(
        "Select stocks to display:",
        options=valid_tickers,
        default=valid_tickers,
        key="wealth_ticker_multiselect"
    )

    if wealth_ticker_selection:
        wealth_chart = plot_cumulative_wealth(
            daily_returns_df,
            benchmark_returns_series,
            wealth_ticker_selection
        )
        st.plotly_chart(wealth_chart, use_container_width=True)
    else:
        st.warning("Select at least one ticker to display the wealth chart.")

    st.markdown("---")

    # ---- 2.2.3 Return Distribution Plot ----
    st.subheader("Return Distribution Analysis")

    dist_col1, dist_col2 = st.columns([1, 2])
    with dist_col1:
        dist_ticker_selection = st.selectbox(
            "Select a stock:",
            options=valid_tickers,
            key="dist_ticker_selectbox"
        )
        dist_plot_type = st.radio(
            "Plot type:",
            options=["Histogram", "Q-Q Plot"],
            key="dist_plot_type_radio",
            help="Q-Q plots are more effective at revealing fat tails and deviations from normality."
        )

    with dist_col2:
        if dist_ticker_selection:
            dist_fig = plot_distribution(
                daily_returns_df[dist_ticker_selection],
                dist_ticker_selection,
                dist_plot_type
            )
            st.plotly_chart(dist_fig, use_container_width=True)


# =============================================================================
# TAB 2: RISK ANALYSIS (Section 2.3)
# =============================================================================
with tab_risk:
    st.header("Risk Analysis")

    # ---- 2.3.1 Rolling Volatility ----
    st.subheader("Rolling Annualized Volatility")

    rolling_window_days = st.select_slider(
        "Rolling window (trading days):",
        options=[21, 30, 60, 90, 120],
        value=60,
        key="rolling_vol_window_slider"
    )

    rolling_vol_fig = go.Figure()
    for idx_ticker, ticker_name in enumerate(valid_tickers):
        rolling_vol_series = (
            daily_returns_df[ticker_name]
            .rolling(window=rolling_window_days)
            .std() * np.sqrt(TRADING_DAYS_PER_YEAR)   # Annualize rolling daily std
        )
        rolling_vol_fig.add_trace(go.Scatter(
            x=rolling_vol_series.index,
            y=rolling_vol_series.values,
            name=ticker_name,
            line=dict(color=CHART_COLOR_SEQUENCE[idx_ticker % len(CHART_COLOR_SEQUENCE)], width=2),
            mode="lines"
        ))
    rolling_vol_fig.update_layout(
        title=f"Rolling {rolling_window_days}-Day Annualized Volatility",
        xaxis_title="Date",
        yaxis_title="Annualized Volatility",
        yaxis_tickformat=".0%",
        template="plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(rolling_vol_fig, use_container_width=True)

    st.markdown("---")

    # ---- 2.3.2 Drawdown Analysis ----
    st.subheader("Drawdown Analysis")

    dd_ticker_selection = st.selectbox(
        "Select a stock for drawdown analysis:",
        options=valid_tickers,
        key="drawdown_ticker_selectbox"
    )

    if dd_ticker_selection:
        dd_price_series = prices_df[dd_ticker_selection]
        drawdown_series = compute_drawdown_series(dd_price_series)
        max_drawdown_value = drawdown_series.min()

        # Display max drawdown as a prominent metric
        st.metric(
            label=f"Maximum Drawdown — {dd_ticker_selection}",
            value=f"{max_drawdown_value:.2%}",
            delta=None,
            help="The largest peak-to-trough decline in the stock's price."
        )

        drawdown_fig = go.Figure()
        drawdown_fig.add_trace(go.Scatter(
            x=drawdown_series.index,
            y=drawdown_series.values,
            fill="tozeroy",
            name=f"{dd_ticker_selection} Drawdown",
            line=dict(color="#a12c7b", width=1.5),
            fillcolor="rgba(161,44,123,0.15)"
        ))
        drawdown_fig.update_layout(
            title=f"{dd_ticker_selection} — Drawdown from Rolling Peak",
            xaxis_title="Date",
            yaxis_title="Drawdown",
            yaxis_tickformat=".0%",
            template="plotly_white"
        )
        st.plotly_chart(drawdown_fig, use_container_width=True)

    st.markdown("---")

    # ---- 2.3.3 Risk-Adjusted Metrics Table ----
    st.subheader("Risk-Adjusted Metrics")

    risk_adj_records = []
    daily_rf_rate = annual_rf_rate / TRADING_DAYS_PER_YEAR

    all_risk_series = dict(daily_returns_df.items())
    if benchmark_returns_series is not None:
        bench_aligned_for_risk = benchmark_returns_series.reindex(daily_returns_df.index).dropna()
        all_risk_series["S&P 500"] = bench_aligned_for_risk

    for asset_name, return_series in all_risk_series.items():
        clean_series = return_series.dropna()
        ann_ret = clean_series.mean() * TRADING_DAYS_PER_YEAR
        ann_vol = clean_series.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        sharpe_val = (ann_ret - annual_rf_rate) / ann_vol if ann_vol > 0 else np.nan
        downside_rets = clean_series[clean_series < daily_rf_rate] - daily_rf_rate
        downside_dev = downside_rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        sortino_val = (ann_ret - annual_rf_rate) / downside_dev if downside_dev > 0 else np.nan
        risk_adj_records.append({
            "Ticker": asset_name,
            "Ann. Return": f"{ann_ret:.2%}",
            "Ann. Volatility": f"{ann_vol:.2%}",
            f"Sharpe Ratio (Rf={annual_rf_rate_pct:.1f}%)": f"{sharpe_val:.3f}",
            f"Sortino Ratio (Rf={annual_rf_rate_pct:.1f}%)": f"{sortino_val:.3f}",
        })

    risk_adj_df = pd.DataFrame(risk_adj_records).set_index("Ticker")
    st.dataframe(risk_adj_df, use_container_width=True)


# =============================================================================
# TAB 3: CORRELATION & COVARIANCE ANALYSIS (Section 2.4)
# =============================================================================
with tab_correlation:
    st.header("Correlation & Covariance Analysis")

    # ---- 2.4.1 Correlation Heatmap ----
    st.subheader("Pairwise Correlation Heatmap")
    corr_heatmap_fig = plot_correlation_heatmap(daily_returns_df)
    st.plotly_chart(corr_heatmap_fig, use_container_width=True)

    st.markdown("---")

    # ---- 2.4.2 Rolling Correlation ----
    st.subheader("Rolling Pairwise Correlation")

    rc_col1, rc_col2, rc_col3 = st.columns(3)
    with rc_col1:
        rolling_corr_ticker_a = st.selectbox(
            "First stock:",
            options=valid_tickers,
            index=0,
            key="rolling_corr_ticker_a_selectbox"
        )
    with rc_col2:
        remaining_tickers_for_b = [t for t in valid_tickers if t != rolling_corr_ticker_a]
        rolling_corr_ticker_b = st.selectbox(
            "Second stock:",
            options=remaining_tickers_for_b,
            index=0,
            key="rolling_corr_ticker_b_selectbox"
        )
    with rc_col3:
        rolling_corr_window = st.select_slider(
            "Rolling window (days):",
            options=[21, 30, 60, 90, 120],
            value=60,
            key="rolling_corr_window_slider"
        )

    if rolling_corr_ticker_a and rolling_corr_ticker_b:
        rolling_corr_series = (
            daily_returns_df[rolling_corr_ticker_a]
            .rolling(window=rolling_corr_window)
            .corr(daily_returns_df[rolling_corr_ticker_b])
        )
        rolling_corr_fig = go.Figure()
        rolling_corr_fig.add_trace(go.Scatter(
            x=rolling_corr_series.index,
            y=rolling_corr_series.values,
            name=f"Correlation: {rolling_corr_ticker_a} vs {rolling_corr_ticker_b}",
            line=dict(color="#01696f", width=2),
            mode="lines"
        ))
        rolling_corr_fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        rolling_corr_fig.update_layout(
            title=f"Rolling {rolling_corr_window}-Day Correlation: {rolling_corr_ticker_a} vs {rolling_corr_ticker_b}",
            xaxis_title="Date",
            yaxis_title="Correlation",
            yaxis_range=[-1, 1],
            template="plotly_white"
        )
        st.plotly_chart(rolling_corr_fig, use_container_width=True)

    st.markdown("---")

    # ---- 2.4.3 Covariance Matrix (behind expander) ----
    with st.expander("📋 View Covariance Matrix (Daily Returns)", expanded=False):
        st.caption("Daily covariance matrix. Multiply values by 252 to annualize.")
        cov_display_df = cov_matrix_df.copy()
        cov_display_df.columns = valid_tickers
        cov_display_df.index = valid_tickers
        st.dataframe(
            cov_display_df.style.format("{:.6f}"),
            use_container_width=True
        )


# =============================================================================
# TAB 4: PORTFOLIO CONSTRUCTION & OPTIMIZATION (Section 2.5)
# =============================================================================
with tab_portfolio:
    st.header("Portfolio Construction & Optimization")

    n_assets = len(valid_tickers)   # Number of assets in the portfolio

    # ---- Pre-compute all optimized portfolios ----
    equal_weight_arr = np.ones(n_assets) / n_assets   # Equal-weight (1/N) portfolio weights

    # Convert arrays to tuples for cache-compatible function calls
    mean_returns_tuple = tuple(mean_daily_returns_arr.tolist())
    cov_matrix_tuple = tuple(map(tuple, cov_matrix_df.values.tolist()))
    tickers_tuple = tuple(valid_tickers)

    with st.spinner("Running portfolio optimizations…"):
        gmv_weights_arr, gmv_success, gmv_message = optimize_gmv_portfolio(
            mean_returns_tuple, cov_matrix_tuple, tickers_tuple
        )
        tangency_weights_arr, tangency_success, tangency_message = optimize_tangency_portfolio(
            mean_returns_tuple, cov_matrix_tuple, tickers_tuple, annual_rf_rate
        )

    if not gmv_success:
        st.warning(f"⚠️ GMV: {gmv_message}")
    if not tangency_success:
        st.warning(f"⚠️ Tangency: {tangency_message}")

    # ---- Compute metrics for each named portfolio ----
    ew_metrics = compute_portfolio_metrics(equal_weight_arr, daily_returns_df, cov_matrix_df, annual_rf_rate)
    gmv_metrics = compute_portfolio_metrics(gmv_weights_arr, daily_returns_df, cov_matrix_df, annual_rf_rate)
    tangency_metrics = compute_portfolio_metrics(tangency_weights_arr, daily_returns_df, cov_matrix_df, annual_rf_rate)

    # ==========================================================================
    # SECTION: CUSTOM PORTFOLIO BUILDER
    # ==========================================================================
    st.subheader("🎛️ Custom Portfolio Builder")
    st.caption("Adjust sliders to set raw weights. Weights are automatically normalized to sum to 1.")

    # One slider per asset; default = equal weight (100 / n_assets for integer slider)
    slider_default_value = round(100 / n_assets)
    raw_slider_values = {}   # Raw (unnormalized) slider values per ticker
    slider_columns = st.columns(min(n_assets, 5))
    for asset_idx, ticker_name in enumerate(valid_tickers):
        col_idx = asset_idx % len(slider_columns)
        with slider_columns[col_idx]:
            raw_slider_values[ticker_name] = st.slider(
                ticker_name,
                min_value=0,
                max_value=100,
                value=slider_default_value,
                step=1,
                key=f"custom_weight_slider_{ticker_name}"
            )

    # Normalize custom weights: divide each by the sum of all slider values
    raw_weight_sum = sum(raw_slider_values.values())
    if raw_weight_sum == 0:
        st.error("All custom weights are 0. Please set at least one slider above 0.")
        custom_weights_arr = equal_weight_arr.copy()
    else:
        custom_weights_arr = np.array(
            [raw_slider_values[t] / raw_weight_sum for t in valid_tickers]
        )

    # Display normalized weights
    normalized_weights_display = pd.DataFrame({
        "Ticker": valid_tickers,
        "Normalized Weight": [f"{w:.2%}" for w in custom_weights_arr]
    }).set_index("Ticker").T
    st.dataframe(normalized_weights_display, use_container_width=True)

    # Compute custom portfolio metrics (dynamically updates with sliders)
    custom_metrics = compute_portfolio_metrics(custom_weights_arr, daily_returns_df, cov_matrix_df, annual_rf_rate)

    # Display custom portfolio performance metrics
    cp_metric_cols = st.columns(5)
    cp_metric_cols[0].metric("Ann. Return", f"{custom_metrics['ann_return']:.2%}")
    cp_metric_cols[1].metric("Ann. Volatility", f"{custom_metrics['ann_volatility']:.2%}")
    cp_metric_cols[2].metric("Sharpe Ratio", f"{custom_metrics['sharpe']:.3f}")
    cp_metric_cols[3].metric("Sortino Ratio", f"{custom_metrics['sortino']:.3f}")
    cp_metric_cols[4].metric("Max Drawdown", f"{custom_metrics['max_drawdown']:.2%}")

    st.markdown("---")

    # ==========================================================================
    # SECTION: PORTFOLIO WEIGHTS CHARTS
    # ==========================================================================
    st.subheader("Portfolio Weight Allocations")

    weight_chart_col1, weight_chart_col2 = st.columns(2)

    def make_weight_bar_chart(weights_array, title_text, tickers_list):
        """Helper: creates a horizontal bar chart of portfolio weights."""
        fig = go.Figure(go.Bar(
            x=weights_array,
            y=tickers_list,
            orientation="h",
            marker=dict(color=CHART_COLOR_SEQUENCE[:len(tickers_list)]),
            text=[f"{w:.1%}" for w in weights_array],
            textposition="outside"
        ))
        fig.update_layout(
            title=title_text,
            xaxis_title="Weight",
            yaxis_title="Ticker",
            xaxis_tickformat=".0%",
            template="plotly_white",
            height=300 + 20 * len(tickers_list)
        )
        return fig

    with weight_chart_col1:
        st.plotly_chart(
            make_weight_bar_chart(gmv_weights_arr, "GMV Portfolio Weights", valid_tickers),
            use_container_width=True
        )
    with weight_chart_col2:
        st.plotly_chart(
            make_weight_bar_chart(tangency_weights_arr, "Tangency Portfolio Weights", valid_tickers),
            use_container_width=True
        )

    st.markdown("---")

    # ==========================================================================
    # SECTION: RISK CONTRIBUTION (Section 2.5 — New)
    # ==========================================================================
    st.subheader("📊 Risk Contribution Decomposition (PRC)")
    st.markdown("""
    <div class="info-box">
    <strong>What is Percentage Risk Contribution (PRC)?</strong><br>
    PRC measures how much of the total portfolio volatility each asset is responsible for.
    An asset with a <strong>10% weight</strong> but a <strong>25% risk contribution</strong>
    is a disproportionate source of portfolio risk — it "punches above its weight" in terms of
    volatility. PRC values sum to 1 (100%).<br><br>
    Formula: <em>PRC_i = w_i × (Σw)_i / σ²_p</em>
    </div>
    """, unsafe_allow_html=True)

    prc_col1, prc_col2 = st.columns(2)

    gmv_prc_arr = compute_risk_contribution(gmv_weights_arr, cov_matrix_df)
    tangency_prc_arr = compute_risk_contribution(tangency_weights_arr, cov_matrix_df)

    def make_prc_comparison_chart(weights_array, prc_array, title_text, tickers_list):
        """Creates a grouped bar chart comparing portfolio weights vs. risk contributions."""
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Weight",
            x=tickers_list,
            y=weights_array,
            marker_color="#01696f",
            text=[f"{w:.1%}" for w in weights_array],
            textposition="outside"
        ))
        fig.add_trace(go.Bar(
            name="Risk Contribution (PRC)",
            x=tickers_list,
            y=prc_array,
            marker_color="#a12c7b",
            text=[f"{p:.1%}" for p in prc_array],
            textposition="outside"
        ))
        fig.update_layout(
            barmode="group",
            title=title_text,
            xaxis_title="Ticker",
            yaxis_title="Fraction",
            yaxis_tickformat=".0%",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig

    with prc_col1:
        st.plotly_chart(
            make_prc_comparison_chart(
                gmv_weights_arr, gmv_prc_arr,
                "GMV: Weights vs. Risk Contribution", valid_tickers
            ),
            use_container_width=True
        )
    with prc_col2:
        st.plotly_chart(
            make_prc_comparison_chart(
                tangency_weights_arr, tangency_prc_arr,
                "Tangency: Weights vs. Risk Contribution", valid_tickers
            ),
            use_container_width=True
        )

    st.markdown("---")

    # ==========================================================================
    # SECTION: EFFICIENT FRONTIER
    # ==========================================================================
    st.subheader("📉 Efficient Frontier & Capital Allocation Line (CAL)")
    st.markdown("""
    <div class="info-box">
    <strong>Efficient Frontier</strong>: The set of portfolios offering the highest expected
    return for any given level of risk (volatility). No portfolio inside the frontier
    can achieve a higher return at the same risk.<br><br>
    <strong>Capital Allocation Line (CAL)</strong>: A line from the risk-free asset
    through the Tangency portfolio. Combinations along the CAL — mixing the risk-free
    asset and the Tangency portfolio — dominate all other combinations of risky assets.
    The slope of the CAL is the maximum achievable Sharpe ratio.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Tracing efficient frontier (this may take a moment)…"):
        frontier_vols_arr, frontier_rets_arr = compute_efficient_frontier(
            mean_returns_tuple, cov_matrix_tuple, tickers_tuple, n_points=80
        )

    # Build individual stock vol/return points
    individual_stock_points_dict = {}
    for ticker_name in valid_tickers:
        s = daily_returns_df[ticker_name]
        individual_stock_points_dict[ticker_name] = (
            s.std() * np.sqrt(TRADING_DAYS_PER_YEAR),
            s.mean() * TRADING_DAYS_PER_YEAR
        )

    # Benchmark point
    if benchmark_returns_series is not None:
        bench_returns_clean = benchmark_returns_series.reindex(daily_returns_df.index).dropna()
        benchmark_point_tuple = (
            bench_returns_clean.std() * np.sqrt(TRADING_DAYS_PER_YEAR),
            bench_returns_clean.mean() * TRADING_DAYS_PER_YEAR
        )
    else:
        benchmark_point_tuple = None

    # Named portfolio vol/return overlay points
    portfolio_overlay_points = {
        "Equal-Weight":       (ew_metrics["ann_volatility"],       ew_metrics["ann_return"]),
        "GMV Portfolio":      (gmv_metrics["ann_volatility"],      gmv_metrics["ann_return"]),
        "Tangency Portfolio": (tangency_metrics["ann_volatility"], tangency_metrics["ann_return"]),
        "Custom Portfolio":   (custom_metrics["ann_volatility"],   custom_metrics["ann_return"]),
    }

    frontier_fig = plot_efficient_frontier_chart(
        frontier_vols=frontier_vols_arr,
        frontier_rets=frontier_rets_arr,
        portfolio_points=portfolio_overlay_points,
        individual_stock_points=individual_stock_points_dict,
        benchmark_point=benchmark_point_tuple,
        tangency_vol=tangency_metrics["ann_volatility"],
        tangency_ret=tangency_metrics["ann_return"],
        annual_rf_rate=annual_rf_rate
    )
    st.plotly_chart(frontier_fig, use_container_width=True)

    st.markdown("---")

    # ==========================================================================
    # SECTION: PORTFOLIO COMPARISON
    # ==========================================================================
    st.subheader("📋 Portfolio Comparison")

    # ---- Comparison table ----
    comparison_table_records = []
    portfolio_definitions = {
        "Equal-Weight (1/N)":   (equal_weight_arr,    ew_metrics),
        "Global Min. Variance": (gmv_weights_arr,     gmv_metrics),
        "Tangency (Max Sharpe)":(tangency_weights_arr,tangency_metrics),
        "Custom Portfolio":     (custom_weights_arr,  custom_metrics),
    }

    for port_label, (_, port_metrics) in portfolio_definitions.items():
        comparison_table_records.append({
            "Portfolio":       port_label,
            "Ann. Return":     f"{port_metrics['ann_return']:.2%}",
            "Ann. Volatility": f"{port_metrics['ann_volatility']:.2%}",
            "Sharpe Ratio":    f"{port_metrics['sharpe']:.3f}",
            "Sortino Ratio":   f"{port_metrics['sortino']:.3f}",
            "Max Drawdown":    f"{port_metrics['max_drawdown']:.2%}",
        })

    # Add S&P 500 benchmark row
    if benchmark_returns_series is not None:
        bench_clean = benchmark_returns_series.reindex(daily_returns_df.index).dropna()
        bench_ann_ret = bench_clean.mean() * TRADING_DAYS_PER_YEAR
        bench_ann_vol = bench_clean.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        bench_sharpe = (bench_ann_ret - annual_rf_rate) / bench_ann_vol if bench_ann_vol > 0 else np.nan
        bench_daily_rf = annual_rf_rate / TRADING_DAYS_PER_YEAR
        bench_downside = bench_clean[bench_clean < bench_daily_rf] - bench_daily_rf
        bench_down_dev = bench_downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        bench_sortino = (bench_ann_ret - annual_rf_rate) / bench_down_dev if bench_down_dev > 0 else np.nan
        bench_wealth = (1 + bench_clean).cumprod()
        bench_max_dd = compute_max_drawdown(bench_wealth)
        comparison_table_records.append({
            "Portfolio":       "S&P 500 (Benchmark)",
            "Ann. Return":     f"{bench_ann_ret:.2%}",
            "Ann. Volatility": f"{bench_ann_vol:.2%}",
            "Sharpe Ratio":    f"{bench_sharpe:.3f}",
            "Sortino Ratio":   f"{bench_sortino:.3f}",
            "Max Drawdown":    f"{bench_max_dd:.2%}",
        })

    comparison_df = pd.DataFrame(comparison_table_records).set_index("Portfolio")
    st.dataframe(comparison_df, use_container_width=True)

    st.markdown("---")

    # ---- Portfolio Cumulative Wealth Comparison Chart ----
    st.subheader("Portfolio Cumulative Wealth Comparison")

    portfolio_wealth_fig = go.Figure()
    INITIAL_PORTFOLIO_VALUE = 10000.0   # Starting value for all portfolio comparisons

    portfolio_wealth_colors = {
        "Equal-Weight (1/N)":    "#437a22",
        "Global Min. Variance":  "#7a39bb",
        "Tangency (Max Sharpe)": "#da7101",
        "Custom Portfolio":      "#a12c7b",
        "S&P 500":               BENCHMARK_COLOR,
    }

    for port_label, (port_weights, _) in portfolio_definitions.items():
        daily_port_returns = daily_returns_df.values @ port_weights
        port_wealth = INITIAL_PORTFOLIO_VALUE * (1 + daily_port_returns).cumprod()
        portfolio_wealth_fig.add_trace(go.Scatter(
            x=daily_returns_df.index,
            y=port_wealth,
            name=port_label,
            line=dict(color=portfolio_wealth_colors[port_label], width=2),
            mode="lines"
        ))

    # Add S&P 500 to wealth chart
    if benchmark_returns_series is not None:
        bench_for_chart = benchmark_returns_series.reindex(daily_returns_df.index).ffill().dropna()
        bench_wealth_series = INITIAL_PORTFOLIO_VALUE * (1 + bench_for_chart).cumprod()
        portfolio_wealth_fig.add_trace(go.Scatter(
            x=bench_wealth_series.index,
            y=bench_wealth_series.values,
            name="S&P 500",
            line=dict(color=BENCHMARK_COLOR, width=2, dash="dash"),
            mode="lines"
        ))

    portfolio_wealth_fig.update_layout(
        title="Portfolio Cumulative Wealth Comparison ($10,000 Starting Value)",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(portfolio_wealth_fig, use_container_width=True)

    # ---- Individual Portfolio Detail Sections ----
    st.markdown("---")
    st.subheader("Individual Portfolio Details")

    detail_tabs = st.tabs(["Equal-Weight", "GMV Portfolio", "Tangency Portfolio"])

    for detail_tab, (port_name_short, (port_w, port_m)) in zip(
        detail_tabs,
        [("Equal-Weight (1/N)", (equal_weight_arr, ew_metrics)),
         ("Global Min. Variance", (gmv_weights_arr, gmv_metrics)),
         ("Tangency (Max Sharpe)", (tangency_weights_arr, tangency_metrics))]
    ):
        with detail_tab:
            detail_metric_cols = st.columns(5)
            detail_metric_cols[0].metric("Ann. Return", f"{port_m['ann_return']:.2%}")
            detail_metric_cols[1].metric("Ann. Volatility", f"{port_m['ann_volatility']:.2%}")
            detail_metric_cols[2].metric("Sharpe Ratio", f"{port_m['sharpe']:.3f}")
            detail_metric_cols[3].metric("Sortino Ratio", f"{port_m['sortino']:.3f}")
            detail_metric_cols[4].metric("Max Drawdown", f"{port_m['max_drawdown']:.2%}")

            detail_weight_df = pd.DataFrame({
                "Ticker": valid_tickers,
                "Weight": [f"{w:.2%}" for w in port_w]
            }).set_index("Ticker")
            st.dataframe(detail_weight_df.T, use_container_width=True)


# =============================================================================
# TAB 5: ESTIMATION WINDOW SENSITIVITY (Section 2.6 — New)
# =============================================================================
with tab_sensitivity:
    st.header("Estimation Window Sensitivity Analysis")
    st.markdown("""
    <div class="info-box">
    <strong>Why does estimation window matter?</strong><br>
    Mean-variance optimization is highly sensitive to its inputs — small changes in
    estimated returns or the covariance matrix can produce dramatically different portfolio
    weights. This section lets you see that directly by comparing optimization results
    across different historical lookback windows.<br><br>
    <em>Historical optimization results are only as stable as the data window used to produce them.
    A portfolio that looks optimal over a 1-year window may look completely different
    using 5 years of data.</em>
    </div>
    """, unsafe_allow_html=True)

    # ---- Determine available lookback windows ----
    total_years_in_sample = (daily_returns_df.index[-1] - daily_returns_df.index[0]).days / 365.25

    # Only offer lookback options supported by the user's date range
    all_lookback_options = {
        "1 Year":    1,
        "3 Years":   3,
        "5 Years":   5,
        "Full Sample": None
    }
    available_lookback_options = {
        label: years
        for label, years in all_lookback_options.items()
        if years is None or years <= total_years_in_sample
    }

    selected_lookback_labels = st.multiselect(
        "Select lookback windows to compare:",
        options=list(available_lookback_options.keys()),
        default=list(available_lookback_options.keys()),
        key="sensitivity_lookback_multiselect"
    )

    if not selected_lookback_labels:
        st.warning("Please select at least one lookback window.")
        st.stop()

    sensitivity_results = []   # Records for the comparison table
    weight_traces_dict = {}    # Weight arrays per window for grouped bar chart

    with st.spinner("Running sensitivity analysis across lookback windows…"):
        for lookback_label in selected_lookback_labels:
            n_years = available_lookback_options[lookback_label]

            # Subset returns to the lookback window
            if n_years is None:
                returns_window_df = daily_returns_df.copy()     # Full sample
            else:
                cutoff_date = daily_returns_df.index[-1] - pd.DateOffset(years=n_years)
                returns_window_df = daily_returns_df[daily_returns_df.index >= cutoff_date]

            if len(returns_window_df) < 60:
                st.warning(f"Skipping '{lookback_label}': insufficient data ({len(returns_window_df)} days).")
                continue

            # Recompute mean returns and covariance for this window
            window_mean_returns = returns_window_df.mean().values
            window_cov_matrix = returns_window_df.cov()
            window_mean_tuple = tuple(window_mean_returns.tolist())
            window_cov_tuple = tuple(map(tuple, window_cov_matrix.values.tolist()))

            # Optimize GMV and Tangency for this window
            window_gmv_weights, window_gmv_ok, _ = optimize_gmv_portfolio(
                window_mean_tuple, window_cov_tuple, tickers_tuple
            )
            window_tang_weights, window_tang_ok, _ = optimize_tangency_portfolio(
                window_mean_tuple, window_cov_tuple, tickers_tuple, annual_rf_rate
            )

            # Compute metrics using the FULL-SAMPLE returns for apples-to-apples comparison
            window_gmv_m = compute_portfolio_metrics(window_gmv_weights, daily_returns_df, cov_matrix_df, annual_rf_rate)
            window_tang_m = compute_portfolio_metrics(window_tang_weights, daily_returns_df, cov_matrix_df, annual_rf_rate)

            # Build weight label for bar chart: (window, portfolio_type) -> weights array
            weight_traces_dict[f"{lookback_label} — GMV"] = window_gmv_weights
            weight_traces_dict[f"{lookback_label} — Tangency"] = window_tang_weights

            # Record GMV row
            gmv_row = {"Window": lookback_label, "Portfolio": "GMV"}
            for ticker_idx, t in enumerate(valid_tickers):
                gmv_row[t] = f"{window_gmv_weights[ticker_idx]:.1%}"
            gmv_row["Ann. Return"] = f"{window_gmv_m['ann_return']:.2%}"
            gmv_row["Ann. Volatility"] = f"{window_gmv_m['ann_volatility']:.2%}"
            gmv_row["Sharpe Ratio"] = "—"  # GMV is not optimized for Sharpe
            sensitivity_results.append(gmv_row)

            # Record Tangency row
            tang_row = {"Window": lookback_label, "Portfolio": "Tangency"}
            for ticker_idx, t in enumerate(valid_tickers):
                tang_row[t] = f"{window_tang_weights[ticker_idx]:.1%}"
            tang_row["Ann. Return"] = f"{window_tang_m['ann_return']:.2%}"
            tang_row["Ann. Volatility"] = f"{window_tang_m['ann_volatility']:.2%}"
            tang_row["Sharpe Ratio"] = f"{window_tang_m['sharpe']:.3f}"
            sensitivity_results.append(tang_row)

    # ---- Display comparison table ----
    if sensitivity_results:
        st.subheader("Sensitivity Comparison Table")
        sensitivity_df = pd.DataFrame(sensitivity_results).set_index(["Window", "Portfolio"])
        st.dataframe(sensitivity_df, use_container_width=True)

        st.markdown("---")

        # ---- Grouped bar chart: weights by window ----
        st.subheader("Portfolio Weights Across Lookback Windows")
        weight_bar_fig = go.Figure()

        for trace_label, w_arr in weight_traces_dict.items():
            weight_bar_fig.add_trace(go.Bar(
                name=trace_label,
                x=valid_tickers,
                y=w_arr,
                text=[f"{v:.1%}" for v in w_arr],
                textposition="outside"
            ))

        weight_bar_fig.update_layout(
            barmode="group",
            title="Portfolio Weights by Lookback Window (GMV & Tangency)",
            xaxis_title="Ticker",
            yaxis_title="Weight",
            yaxis_tickformat=".0%",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(weight_bar_fig, use_container_width=True)
    else:
        st.info("No sensitivity results to display. Try selecting different lookback windows.")

