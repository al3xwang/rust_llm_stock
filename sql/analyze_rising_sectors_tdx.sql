-- Sector Analysis: Find Sectors with Strong Upward Momentum (TDX Data)

\echo '=== 1. Top Sectors by Recent Momentum (Last 5 Days) ==='
WITH recent_performance AS (
    SELECT 
        d.ts_code,
        i.name,
        d.trade_date,
        d.pct_change,
        d.up_num,
        d.down_num,
        d.turnover_rate,
        d.float_mv as total_mv,
        ROW_NUMBER() OVER (PARTITION BY d.ts_code ORDER BY d.trade_date DESC) as rn
    FROM tdx_daily d
    LEFT JOIN tdx_index i ON d.ts_code = i.ts_code AND d.trade_date = i.trade_date
    WHERE d.trade_date >= TO_CHAR(CURRENT_DATE - INTERVAL '10 days', 'YYYYMMDD')
)
SELECT 
    ts_code,
    name,
    ROUND(AVG(pct_change)::numeric, 2) as avg_pct_change_5d,
    ROUND(SUM(pct_change)::numeric, 2) as total_return_5d,
    ROUND(AVG(up_num)::numeric, 0) as avg_up_stocks,
    ROUND(AVG(down_num)::numeric, 0) as avg_down_stocks,
    ROUND(AVG(turnover_rate)::numeric, 2) as avg_turnover,
    ROUND(MAX(total_mv)::numeric/100000000, 2) as latest_mv_100m
FROM recent_performance
WHERE rn <= 5
GROUP BY ts_code, name
HAVING COUNT(*) = 5  -- Must have data for all 5 days
ORDER BY total_return_5d DESC
LIMIT 20;

\echo ''
\echo '=== 2. Sectors with Accelerating Momentum (Trend Improving) ==='
WITH daily_data AS (
    SELECT 
        d.ts_code,
        i.name,
        d.trade_date,
        d.pct_change,
        d.up_num - d.down_num as net_up,
        ROW_NUMBER() OVER (PARTITION BY d.ts_code ORDER BY d.trade_date DESC) as rn
    FROM tdx_daily d
    LEFT JOIN tdx_index i ON d.ts_code = i.ts_code AND d.trade_date = i.trade_date
    WHERE d.trade_date >= TO_CHAR(CURRENT_DATE - INTERVAL '10 days', 'YYYYMMDD')
),
trend_calc AS (
    SELECT 
        ts_code,
        name,
        AVG(CASE WHEN rn <= 3 THEN pct_change END) as recent_3d_avg,
        AVG(CASE WHEN rn BETWEEN 4 AND 6 THEN pct_change END) as prev_3d_avg,
        AVG(CASE WHEN rn <= 3 THEN net_up END) as recent_net_up,
        MAX(CASE WHEN rn = 1 THEN pct_change END) as latest_pct_change
    FROM daily_data
    WHERE rn <= 6
    GROUP BY ts_code, name
)
SELECT 
    ts_code,
    name,
    ROUND(latest_pct_change::numeric, 2) as today_pct,
    ROUND(recent_3d_avg::numeric, 2) as avg_3d_recent,
    ROUND(prev_3d_avg::numeric, 2) as avg_3d_prev,
    ROUND((recent_3d_avg - prev_3d_avg)::numeric, 2) as momentum_acceleration,
    ROUND(recent_net_up::numeric, 0) as avg_net_up_stocks
FROM trend_calc
WHERE recent_3d_avg > prev_3d_avg  -- Accelerating
  AND recent_3d_avg > 0  -- Positive momentum
  AND recent_net_up > 5  -- More stocks going up
ORDER BY momentum_acceleration DESC
LIMIT 20;

\echo ''
\echo '=== 3. Sectors with Most Rising Stocks ==='
WITH latest_date AS (
    SELECT MAX(trade_date) as max_date FROM tdx_daily
)
SELECT 
    d.ts_code,
    i.name,
    d.pct_change,
    d.up_num,
    d.down_num,
    d.limit_up_num,
    ROUND((d.up_num::float / NULLIF(d.up_num + d.down_num, 0) * 100)::numeric, 1) as pct_stocks_up,
    d.turnover_rate,
    d.vol_ratio
FROM tdx_daily d
LEFT JOIN tdx_index i ON d.ts_code = i.ts_code AND d.trade_date = i.trade_date
CROSS JOIN latest_date ld
WHERE d.trade_date = ld.max_date
  AND d.pct_change > 0     -- Sector positive
  AND d.up_num > d.down_num  -- More ups than downs
ORDER BY d.up_num DESC, d.pct_change DESC
LIMIT 20;

\echo ''
\echo '=== 4. High Volume Breakout Sectors ==='
WITH volume_stats AS (
    SELECT 
        d.ts_code,
        i.name,
        d.trade_date,
        d.turnover_rate,
        d.pct_change,
        d.vol_ratio,
        AVG(d.turnover_rate) OVER (
            PARTITION BY d.ts_code 
            ORDER BY d.trade_date 
            ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
        ) as avg_turnover_20d,
        ROW_NUMBER() OVER (PARTITION BY d.ts_code ORDER BY d.trade_date DESC) as rn
    FROM tdx_daily d
    LEFT JOIN tdx_index i ON d.ts_code = i.ts_code AND d.trade_date = i.trade_date
    WHERE d.trade_date >= TO_CHAR(CURRENT_DATE - INTERVAL '30 days', 'YYYYMMDD')
)
SELECT 
    ts_code,
    name,
    ROUND(pct_change::numeric, 2) as pct_change,
    ROUND(turnover_rate::numeric, 2) as today_turnover,
    ROUND(avg_turnover_20d::numeric, 2) as avg_turnover_20d,
    ROUND((turnover_rate / NULLIF(avg_turnover_20d, 0))::numeric, 2) as turnover_vs_avg,
    ROUND(vol_ratio::numeric, 2) as vol_ratio
FROM volume_stats
WHERE rn = 1  -- Latest day only
  AND pct_change > 0
  AND turnover_rate > avg_turnover_20d * 1.5  -- 50% above average
ORDER BY turnover_vs_avg DESC, pct_change DESC
LIMIT 20;

\echo ''
\echo '=== 5. Consistent Winners (Positive >60% of Last 10 Days) ==='
WITH daily_performance AS (
    SELECT 
        d.ts_code,
        i.name,
        d.trade_date,
        d.pct_change,
        CASE WHEN d.pct_change > 0 THEN 1 ELSE 0 END as positive_day,
        ROW_NUMBER() OVER (PARTITION BY d.ts_code ORDER BY d.trade_date DESC) as rn
    FROM tdx_daily d
    LEFT JOIN tdx_index i ON d.ts_code = i.ts_code AND d.trade_date = i.trade_date
    WHERE d.trade_date >= TO_CHAR(CURRENT_DATE - INTERVAL '15 days', 'YYYYMMDD')
)
SELECT 
    ts_code,
    name,
    COUNT(*) as trading_days,
    SUM(positive_day) as positive_days,
    ROUND((SUM(positive_day)::float / COUNT(*) * 100)::numeric, 1) as win_rate_pct,
    ROUND(AVG(pct_change)::numeric, 2) as avg_daily_change,
    ROUND(SUM(pct_change)::numeric, 2) as cumulative_return
FROM daily_performance
WHERE rn <= 10
GROUP BY ts_code, name
HAVING COUNT(*) >= 8  -- At least 8 trading days
   AND SUM(positive_day)::float / COUNT(*) >= 0.6  -- 60%+ win rate
ORDER BY win_rate_pct DESC, cumulative_return DESC
LIMIT 20;

\echo ''
\echo '=== 6. Composite Score: Best Overall Sectors ==='
WITH latest_date AS (
    SELECT MAX(trade_date) as max_date FROM tdx_daily
),
recent_stats AS (
    SELECT 
        ts_code,
        name,
        AVG(CASE WHEN rn <= 5 THEN pct_change END) as avg_5d_return,
        SUM(CASE WHEN rn <= 5 THEN pct_change END) as total_5d_return,
        AVG(CASE WHEN rn <= 5 THEN turnover_rate END) as avg_turnover,
        AVG(CASE WHEN rn <= 5 THEN up_num - down_num END) as avg_net_up,
        MAX(CASE WHEN rn = 1 THEN pct_change END) as latest_pct,
        MAX(CASE WHEN rn = 1 THEN limit_up_num END) as limit_up_stocks,
        MAX(CASE WHEN rn = 1 THEN vol_ratio END) as vol_ratio
    FROM (
        SELECT d.*, i.name,
               ROW_NUMBER() OVER (PARTITION BY d.ts_code ORDER BY d.trade_date DESC) as rn
        FROM tdx_daily d
        LEFT JOIN tdx_index i ON d.ts_code = i.ts_code AND d.trade_date = i.trade_date
        WHERE d.trade_date >= TO_CHAR(CURRENT_DATE - INTERVAL '10 days', 'YYYYMMDD')
    ) t
    WHERE rn <= 5
    GROUP BY ts_code, name
)
SELECT 
    ts_code,
    name,
    ROUND(latest_pct::numeric, 2) as today_pct,
    ROUND(total_5d_return::numeric, 2) as return_5d,
    ROUND(avg_turnover::numeric, 2) as avg_turnover,
    ROUND(avg_net_up::numeric, 0) as avg_net_up,
    ROUND(limit_up_stocks::numeric, 0) as limit_up_stocks,
    ROUND(vol_ratio::numeric, 2) as vol_ratio,
    -- Composite score: weighted combination of signals
    ROUND((
        total_5d_return * 2 +  -- 5-day return (weight 2)
        latest_pct * 3 +       -- Today's change (weight 3)
        avg_net_up * 0.1 +     -- Net up stocks (weight 0.1)
        COALESCE(limit_up_stocks, 0) * 2 +  -- Limit up stocks (weight 2)
        (avg_turnover - 1.0) * 5 +  -- Turnover above 1% (weight 5)
        COALESCE(vol_ratio, 1) * 2  -- Volume ratio (weight 2)
    )::numeric, 2) as composite_score
FROM recent_stats
WHERE total_5d_return > 0  -- Positive 5-day trend
  AND avg_net_up > 0       -- More ups than downs
ORDER BY composite_score DESC
LIMIT 30;

\echo ''
\echo '=== 7. Top Constituent Stocks in Rising Sectors ==='
WITH top_sectors AS (
    SELECT ts_code, pct_change
    FROM tdx_daily
    WHERE trade_date = (SELECT MAX(trade_date) FROM tdx_daily)
      AND pct_change > 2.0
      AND up_num > down_num
    ORDER BY pct_change DESC
    LIMIT 10
),
latest_stock_data AS (
    SELECT 
        d.*,
        ROW_NUMBER() OVER (PARTITION BY d.ts_code ORDER BY d.trade_date DESC) as rn
    FROM tdx_daily d
    INNER JOIN top_sectors ts ON d.ts_code = ts.ts_code
)
SELECT 
    d.ts_code as sector_code,
    m.con_name as stock_name,
    m.con_code,
    ROUND(d.pct_change::numeric, 2) as stock_pct_change,
    ROUND(d.close::numeric, 2) as close_price,
    ROUND(d.turnover_rate::numeric, 2) as turnover_rate,
    ROUND(d.vol::numeric/1000000, 2) as vol_millions
FROM latest_stock_data d
INNER JOIN tdx_member m ON d.ts_code = m.ts_code 
    AND d.trade_date = m.trade_date
    AND m.con_code = d.ts_code
WHERE d.rn = 1
  AND d.pct_change > 0
ORDER BY d.ts_code, d.pct_change DESC;
