-- Sector Analysis: Find Sectors with Strong Upward Momentum
-- Run with: psql postgresql://postgres:12341234@127.0.0.1:5432/research -f analyze_rising_sectors.sql

\echo '=== 1. Top Sectors by Recent Momentum (Last 5 Days) ==='
WITH recent_performance AS (
    SELECT 
        ts_code,
        name,
        trade_date,
        pct_change,
        up_num,
        down_num,
        turnover_rate,
        total_mv,
        ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date DESC) as rn
    FROM dc_index
    WHERE trade_date >= TO_CHAR(CURRENT_DATE - INTERVAL '10 days', 'YYYYMMDD')
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
        ts_code,
        name,
        trade_date,
        pct_change,
        up_num - down_num as net_up,
        ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date DESC) as rn
    FROM dc_index
    WHERE trade_date >= TO_CHAR(CURRENT_DATE - INTERVAL '10 days', 'YYYYMMDD')
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
\echo '=== 3. Sectors with Strong Leading Stocks ==='
WITH latest_date AS (
    SELECT MAX(trade_date) as max_date FROM dc_index
)
SELECT 
    i.ts_code,
    i.name,
    i.pct_change,
    i.leading,
    i.leading_code,
    i.leading_pct,
    i.up_num,
    i.down_num,
    ROUND((i.up_num::float / NULLIF(i.up_num + i.down_num, 0) * 100)::numeric, 1) as pct_stocks_up,
    i.turnover_rate
FROM dc_index i
CROSS JOIN latest_date ld
WHERE i.trade_date = ld.max_date
  AND i.leading_pct IS NOT NULL
  AND i.leading_pct > 3.0  -- Leading stock up >3%
  AND i.pct_change > 0     -- Sector positive
  AND i.up_num > i.down_num  -- More ups than downs
ORDER BY i.leading_pct DESC, i.pct_change DESC
LIMIT 20;

\echo ''
\echo '=== 4. High Volume Breakout Sectors ==='
WITH volume_stats AS (
    SELECT 
        ts_code,
        name,
        trade_date,
        turnover_rate,
        pct_change,
        AVG(turnover_rate) OVER (
            PARTITION BY ts_code 
            ORDER BY trade_date 
            ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
        ) as avg_turnover_20d,
        ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date DESC) as rn
    FROM dc_index
    WHERE trade_date >= TO_CHAR(CURRENT_DATE - INTERVAL '30 days', 'YYYYMMDD')
)
SELECT 
    ts_code,
    name,
    ROUND(pct_change::numeric, 2) as pct_change,
    ROUND(turnover_rate::numeric, 2) as today_turnover,
    ROUND(avg_turnover_20d::numeric, 2) as avg_turnover_20d,
    ROUND((turnover_rate / NULLIF(avg_turnover_20d, 0))::numeric, 2) as turnover_vs_avg
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
        ts_code,
        name,
        trade_date,
        pct_change,
        CASE WHEN pct_change > 0 THEN 1 ELSE 0 END as positive_day,
        ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date DESC) as rn
    FROM dc_index
    WHERE trade_date >= TO_CHAR(CURRENT_DATE - INTERVAL '15 days', 'YYYYMMDD')
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
    SELECT MAX(trade_date) as max_date FROM dc_index
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
        MAX(CASE WHEN rn = 1 THEN leading_pct END) as latest_leading_pct
    FROM (
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date DESC) as rn
        FROM dc_index
        WHERE trade_date >= TO_CHAR(CURRENT_DATE - INTERVAL '10 days', 'YYYYMMDD')
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
    ROUND(latest_leading_pct::numeric, 2) as leading_stock_pct,
    -- Composite score: weighted combination of signals
    ROUND((
        total_5d_return * 2 +  -- 5-day return (weight 2)
        latest_pct * 3 +       -- Today's change (weight 3)
        avg_net_up * 0.1 +     -- Net up stocks (weight 0.1)
        COALESCE(latest_leading_pct, 0) * 1 +  -- Leading stock (weight 1)
        (avg_turnover - 1.0) * 5  -- Turnover above 1% (weight 5)
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
    FROM dc_index
    WHERE trade_date = (SELECT MAX(trade_date) FROM dc_index)
      AND pct_change > 2.0
      AND up_num > down_num
    ORDER BY pct_change DESC
    LIMIT 10
),
latest_stock_data AS (
    SELECT 
        d.*,
        ROW_NUMBER() OVER (PARTITION BY d.ts_code ORDER BY d.trade_date DESC) as rn
    FROM dc_daily d
    INNER JOIN top_sectors ts ON d.ts_code = ts.ts_code
)
SELECT 
    d.ts_code as sector_code,
    m.name as stock_name,
    ROUND(d.pct_change::numeric, 2) as stock_pct_change,
    ROUND(d.close::numeric, 2) as close_price,
    ROUND(d.turnover_rate::numeric, 2) as turnover_rate,
    ROUND(d.vol::numeric/1000000, 2) as vol_millions
FROM latest_stock_data d
INNER JOIN dc_member m ON d.ts_code = m.ts_code AND m.con_code = d.ts_code
WHERE d.rn = 1
  AND d.pct_change > 0
ORDER BY d.ts_code, d.pct_change DESC;
