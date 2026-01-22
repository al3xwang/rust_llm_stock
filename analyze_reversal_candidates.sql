-- Reversal Candidate Analysis Script
-- Uses full feature set in ml_training_dataset to identify stocks poised for a bounce/reversal
-- Date: 2026-01-15

WITH LatestData AS (
    SELECT MAX(trade_date) as max_date FROM ml_training_dataset
),
base AS (
    SELECT *
    FROM ml_training_dataset
    WHERE trade_date = (SELECT max_date FROM LatestData)
),
scored_stocks AS (
    SELECT 
        base.*,
        -- Reversal Score Calculation (Higher = Stronger Reversal Signal)
        (
            -- 1) Oversold signals
            (CASE WHEN COALESCE(rsi_14, 50) < 35 THEN (35 - rsi_14) * 1.2 ELSE 0 END) +
            (CASE WHEN COALESCE(kdj_j, 50) < 20 THEN (20 - kdj_j) * 0.6 ELSE 0 END) +
            (CASE WHEN COALESCE(williams_r_14, -50) < -80 THEN (-80 - williams_r_14) * 0.15 ELSE 0 END) +
            (CASE WHEN COALESCE(bb_percent_b, 0.5) < 0.10 THEN (0.10 - bb_percent_b) * 50 ELSE 0 END) +
            (CASE WHEN COALESCE(price_position_52w, 0.5) < 0.20 THEN (0.20 - price_position_52w) * 40 ELSE 0 END) +
            (CASE WHEN COALESCE(close_position_in_range, 50) < 20 THEN (20 - close_position_in_range) * 0.2 ELSE 0 END) +

            -- 2) Momentum & lagged return pressure (negative = oversold)
            (CASE WHEN COALESCE(return_lag_1, 0) < 0 THEN ABS(return_lag_1) * 50 ELSE 0 END) +
            (CASE WHEN COALESCE(return_lag_2, 0) < 0 THEN ABS(return_lag_2) * 30 ELSE 0 END) +
            (CASE WHEN COALESCE(return_lag_3, 0) < 0 THEN ABS(return_lag_3) * 20 ELSE 0 END) +
            (CASE WHEN COALESCE(price_roc_5, 0) < 0 THEN ABS(price_roc_5) * 20 ELSE 0 END) +
            (CASE WHEN COALESCE(price_roc_10, 0) < 0 THEN ABS(price_roc_10) * 15 ELSE 0 END) +
            (CASE WHEN COALESCE(price_roc_20, 0) < 0 THEN ABS(price_roc_20) * 10 ELSE 0 END) +
            (CASE WHEN COALESCE(price_momentum_5, 0) < 0 THEN ABS(price_momentum_5) * 15 ELSE 0 END) +
            (CASE WHEN COALESCE(price_momentum_10, 0) < 0 THEN ABS(price_momentum_10) * 10 ELSE 0 END) +
            (CASE WHEN COALESCE(price_momentum_20, 0) < 0 THEN ABS(price_momentum_20) * 8 ELSE 0 END) +

            -- 3) Trend strength & MACD posture
            (CASE WHEN COALESCE(trend_strength, 0) < 0 THEN ABS(trend_strength) * 5 ELSE 0 END) +
            (CASE WHEN COALESCE(adx_14, 0) > 25 THEN 2 ELSE 0 END) +
            (CASE WHEN COALESCE(macd_histogram, 0) < 0 THEN ABS(macd_histogram) * 5 ELSE 0 END) +
            (CASE WHEN COALESCE(macd_line, 0) > COALESCE(macd_signal, 0) THEN 2 ELSE 0 END) +
            (CASE WHEN COALESCE(ema_5, 0) > COALESCE(ema_10, 0) THEN 1 ELSE 0 END) +
            (CASE WHEN COALESCE(sma_5, 0) > COALESCE(sma_10, 0) THEN 1 ELSE 0 END) +

            -- 4) Volatility & capitulation
            (CASE WHEN COALESCE(vol_percentile, 0) > 0.80 THEN 5 ELSE 0 END) +
            (CASE WHEN COALESCE(high_vol_regime, 0) = 1 THEN 3 ELSE 0 END) +
            (CASE WHEN COALESCE(atr, 0) > 0 THEN LEAST(5, atr) ELSE 0 END) +
            (CASE WHEN COALESCE(hist_volatility_20, 0) > 0 THEN LEAST(5, hist_volatility_20) ELSE 0 END) +

            -- 5) Volume & flow
            (CASE WHEN COALESCE(volume_ratio, 0) > 1.2 THEN 3 ELSE 0 END) +
            (CASE WHEN COALESCE(volume_roc_5, 0) > 0 THEN 2 ELSE 0 END) +
            (CASE WHEN COALESCE(volume_spike, false) THEN 3 ELSE 0 END) +
            (CASE WHEN COALESCE(cmf_20, 0) > 0 THEN 2 ELSE 0 END) +

            -- 6) Candlestick patterns
            (CASE WHEN COALESCE(is_hammer, false) THEN 6 ELSE 0 END) +
            (CASE WHEN COALESCE(is_doji, false) THEN 3 ELSE 0 END) +
            (CASE WHEN COALESCE(is_shooting_star, false) THEN -2 ELSE 0 END) +

            -- 7) Valuation & liquidity
            (CASE WHEN COALESCE(pe_ttm, 0) BETWEEN 0 AND 25 THEN (25 - pe_ttm) * 0.2 ELSE 0 END) +
            (CASE WHEN COALESCE(pb, 0) BETWEEN 0 AND 2 THEN (2 - pb) * 2 ELSE 0 END) +
            (CASE WHEN COALESCE(ps, 0) BETWEEN 0 AND 2 THEN (2 - ps) * 1 ELSE 0 END) +
            (CASE WHEN COALESCE(dv_ratio, 0) > 0 THEN 1 ELSE 0 END) +
            (CASE WHEN COALESCE(turnover_rate, 0) > 0.5 THEN 1 ELSE 0 END) +

            -- 8) Market context (avoid broad selloff)
            (CASE WHEN COALESCE(index_csi300_pct_chg, 0) > -1 THEN 1 ELSE -2 END) +
            (CASE WHEN COALESCE(index_chinext_pct_chg, 0) > -1 THEN 1 ELSE -1 END) +
            (CASE WHEN COALESCE(index_xin9_pct_chg, 0) > -1 THEN 1 ELSE -1 END)
        ) as reversal_score
    FROM base
    WHERE COALESCE(close_pct, 0) < 8
      AND COALESCE(rsi_14, 50) < 55
)
SELECT 
    ts_code,
    industry,
    trade_date,
    ROUND(reversal_score::numeric, 2) as score,
    ROUND(rsi_14::numeric, 2) as rsi,
    ROUND(kdj_j::numeric, 2) as kdj_j,
    ROUND(bb_percent_b::numeric, 2) as "%b",
    consecutive_days as streak,
    ROUND(price_position_52w::numeric, 2) as pos_52w,
    ROUND(pe_ttm::numeric, 2) as pe_ttm,
    ROUND(pb::numeric, 2) as pb,
    ROUND(volume_ratio::numeric, 2) as vol_ratio,
    CASE WHEN is_hammer THEN 'HAMMER' WHEN is_doji THEN 'DOJI' WHEN is_shooting_star THEN 'SHOOTING_STAR' ELSE '' END as pattern,
    ROUND(cmf_20::numeric, 2) as cmf,
    ROUND(index_csi300_pct_chg::numeric, 2) as csi300,
    ROUND(index_chinext_pct_chg::numeric, 2) as chinext,
    ROUND(index_xin9_pct_chg::numeric, 2) as xin9
FROM scored_stocks
WHERE reversal_score >= 20
ORDER BY reversal_score DESC
LIMIT 50;
