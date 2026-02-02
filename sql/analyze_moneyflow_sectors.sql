-- Money Flow Sector Analysis (THS/DC concepts & industries)
-- Run: psql postgresql://postgres:12341234@127.0.0.1:5432/research -f analyze_moneyflow_sectors.sql

\echo '=== 1) THS Concepts: Top Net Inflow (5-day sum) ==='
WITH mf AS (
    SELECT ts_code, name, trade_date::date AS d, net_amount, pct_change
    FROM moneyflow_cnt_ths
    WHERE trade_date::date >= CURRENT_DATE - INTERVAL '10 days'
), ranked AS (
    SELECT ts_code, name,
           SUM(CASE WHEN d >= CURRENT_DATE - INTERVAL '4 days' THEN net_amount END) AS net_5d,
           SUM(CASE WHEN d >= CURRENT_DATE - INTERVAL '4 days' THEN pct_change END) AS pct_sum_5d,
           AVG(CASE WHEN d >= CURRENT_DATE - INTERVAL '4 days' THEN pct_change END) AS pct_avg_5d,
           COUNT(*) FILTER (WHERE d >= CURRENT_DATE - INTERVAL '4 days') AS cnt5,
           MAX(d) AS last_d
    FROM mf
    GROUP BY ts_code, name
)
SELECT ts_code, name,
    ROUND((net_5d/1e8)::numeric,2) AS net_5d_100m,
    ROUND((pct_sum_5d)::numeric,2) AS pct_sum_5d,
    ROUND((pct_avg_5d)::numeric,2) AS pct_avg_5d
FROM ranked
WHERE cnt5 >= 3
ORDER BY net_5d DESC
LIMIT 20;

\echo ''
\echo '=== 2) THS Concepts: Accelerating Net Inflow (3d vs prev 3d) ==='
WITH mf AS (
    SELECT ts_code, name, trade_date::date AS d, net_amount
    FROM moneyflow_cnt_ths
    WHERE trade_date::date >= CURRENT_DATE - INTERVAL '12 days'
), win AS (
    SELECT ts_code, name,
           AVG(CASE WHEN d >= CURRENT_DATE - INTERVAL '2 days' THEN net_amount END) AS avg_last3,
           AVG(CASE WHEN d BETWEEN CURRENT_DATE - INTERVAL '5 days' AND CURRENT_DATE - INTERVAL '3 days' THEN net_amount END) AS avg_prev3,
           SUM(CASE WHEN d >= CURRENT_DATE - INTERVAL '2 days' THEN net_amount END) AS sum_last3,
           SUM(CASE WHEN d BETWEEN CURRENT_DATE - INTERVAL '5 days' AND CURRENT_DATE - INTERVAL '3 days' THEN net_amount END) AS sum_prev3
    FROM mf
    GROUP BY ts_code, name
)
SELECT ts_code, name,
    ROUND((sum_last3/1e8)::numeric,2) AS net_last3_100m,
    ROUND((sum_prev3/1e8)::numeric,2) AS net_prev3_100m,
    ROUND(((sum_last3 - COALESCE(sum_prev3,0))/1e8)::numeric,2) AS acceleration_100m
FROM win
WHERE sum_last3 > 0 AND sum_last3 > COALESCE(sum_prev3,0)
ORDER BY acceleration_100m DESC
LIMIT 20;

\echo ''
\echo '=== 3) THS Concepts: Latest Day Strong (price up + big net inflow) ==='
WITH latest AS (
    SELECT * FROM moneyflow_cnt_ths m
    WHERE trade_date = (SELECT MAX(trade_date) FROM moneyflow_cnt_ths)
)
SELECT ts_code, name, lead_stock, close_price,
    ROUND((pct_change)::numeric,2) AS pct_change,
    ROUND((net_amount/1e8)::numeric,2) AS net_amount_100m
FROM latest
WHERE pct_change > 1.5 AND net_amount > 0
ORDER BY net_amount DESC, pct_change DESC
LIMIT 20;

\echo ''
\echo '=== 4) THS Concepts: Consistent Inflows (>=3 positive net days in last 5) ==='
WITH mf AS (
    SELECT ts_code, name, trade_date::date AS d, net_amount
    FROM moneyflow_cnt_ths
    WHERE trade_date::date >= CURRENT_DATE - INTERVAL '7 days'
), agg AS (
    SELECT ts_code, name,
           COUNT(*) AS days,
           SUM(CASE WHEN net_amount > 0 THEN 1 ELSE 0 END) AS pos_days,
           SUM(net_amount) AS net_sum
    FROM mf
    GROUP BY ts_code, name
)
SELECT ts_code, name,
       pos_days, days,
    ROUND((net_sum/1e8)::numeric,2) AS net_sum_100m
FROM agg
WHERE days >= 3 AND pos_days >= 3
ORDER BY net_sum DESC
LIMIT 20;

\echo ''
\echo '=== 5) THS Industries: Latest Net Inflow (moneyflow_ind_ths) ==='
WITH latest AS (
    SELECT * FROM moneyflow_ind_ths
    WHERE trade_date = (SELECT MAX(trade_date) FROM moneyflow_ind_ths)
)
SELECT ts_code, industry_name,
    ROUND((net_amount/1e8)::numeric,2) AS net_amount_100m,
    ROUND((net_buy_amount/1e8)::numeric,2) AS buy_100m,
    ROUND((net_sell_amount/1e8)::numeric,2) AS sell_100m
FROM latest
ORDER BY net_amount DESC
LIMIT 20;

\echo ''
\echo '=== 6) DC Industries: Latest Net Inflow (moneyflow_ind_dc) ==='
WITH latest AS (
    SELECT * FROM moneyflow_ind_dc
    WHERE trade_date = (SELECT MAX(trade_date) FROM moneyflow_ind_dc)
)
SELECT ts_code, industry_name,
    ROUND((net_amount/1e8)::numeric,2) AS net_amount_100m,
    ROUND((net_buy_amount/1e8)::numeric,2) AS buy_100m,
    ROUND((net_sell_amount/1e8)::numeric,2) AS sell_100m
FROM latest
ORDER BY net_amount DESC
LIMIT 20;

\echo ''
\echo '=== 7) Stock-level Moneyflow: Top Net Inflow Today (moneyflow) ==='
WITH latest AS (
    SELECT * FROM moneyflow
    WHERE trade_date = (SELECT MAX(trade_date) FROM moneyflow)
)
SELECT ts_code,
    ROUND((net_mf_amount/1e8)::numeric,2) AS net_amount_100m,
    ROUND((net_mf_vol/1e6)::numeric,2) AS net_vol_millions
FROM latest
WHERE net_mf_amount IS NOT NULL
ORDER BY net_mf_amount DESC
LIMIT 30;
