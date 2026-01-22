use anyhow::Result;
use chrono::{Datelike, NaiveDate};
use deadpool_postgres::{Config, ManagerConfig, Pool, RecyclingMethod, Runtime};
use sqlx::FromRow;
use std::collections::HashMap;
use std::str::FromStr;
use tokio_postgres::NoTls;

#[allow(dead_code)]
pub struct DbClient {
    pool: Pool,
}

#[allow(dead_code)]
impl DbClient {
    pub async fn new(connection_string: &str) -> Result<Self> {
        let _cfg = Config::new();
        // Parse connection string manually or use a crate, but deadpool config is struct based.
        // For simplicity, we'll assume the user provides a standard postgres URL and we parse it
        // or we just use default config for now.
        // Actually, tokio-postgres Config can parse the URL.

        let pg_config = tokio_postgres::Config::from_str(connection_string)?;

        let mgr_config = ManagerConfig {
            recycling_method: RecyclingMethod::Fast,
        };

        let mgr = deadpool_postgres::Manager::from_config(pg_config, NoTls, mgr_config);
        let pool = Pool::builder(mgr).runtime(Runtime::Tokio1).build()?;

        Ok(Self { pool })
    }

    pub async fn init_table(&self) -> Result<()> {
        let client = self.pool.get().await?;
        client
            .batch_execute(
                "
            CREATE TABLE IF NOT EXISTS stock_history (
                symbol TEXT NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                month INTEGER NOT NULL,
                weekday INTEGER NOT NULL,
                sma5 REAL,
                sma20 REAL,
                rsi REAL,
                daily_return REAL,
                volume_ratio REAL,
                quarter INTEGER,
                PRIMARY KEY (symbol, timestamp)
            );
        ",
            )
            .await?;
        Ok(())
    }

    pub async fn save_quote(
        &self,
        symbol: &str,
        timestamp: i64,
        open: f32,
        high: f32,
        low: f32,
        close: f32,
        volume: f32,
        month: i32,
        weekday: i32,
        sma5: f32,
        sma20: f32,
        rsi: f32,
        daily_return: f32,
        volume_ratio: f32,
        quarter: i32,
    ) -> Result<()> {
        let client = self.pool.get().await?;

        // Convert timestamp (i64 unix seconds) to chrono::DateTime<Utc>
        let time = chrono::DateTime::<chrono::Utc>::from(
            std::time::UNIX_EPOCH + std::time::Duration::from_secs(timestamp as u64),
        );

        client.execute(
            "INSERT INTO stock_history (symbol, timestamp, open, high, low, close, volume, month, weekday, sma5, sma20, rsi, daily_return, volume_ratio, quarter) 
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
             ON CONFLICT (symbol, timestamp) DO UPDATE 
             SET open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low, close = EXCLUDED.close, volume = EXCLUDED.volume, 
                 month = EXCLUDED.month, weekday = EXCLUDED.weekday, sma5 = EXCLUDED.sma5, sma20 = EXCLUDED.sma20, 
                 rsi = EXCLUDED.rsi, daily_return = EXCLUDED.daily_return, volume_ratio = EXCLUDED.volume_ratio, quarter = EXCLUDED.quarter",
            &[
                &symbol as &(dyn tokio_postgres::types::ToSql + Sync),
                &time.to_rfc3339() as &(dyn tokio_postgres::types::ToSql + Sync),
                &open as &(dyn tokio_postgres::types::ToSql + Sync),
                &high as &(dyn tokio_postgres::types::ToSql + Sync),
                &low as &(dyn tokio_postgres::types::ToSql + Sync),
                &close as &(dyn tokio_postgres::types::ToSql + Sync),
                &volume as &(dyn tokio_postgres::types::ToSql + Sync),
                &(month as i32) as &(dyn tokio_postgres::types::ToSql + Sync),
                &(weekday as i32) as &(dyn tokio_postgres::types::ToSql + Sync),
                &sma5 as &(dyn tokio_postgres::types::ToSql + Sync),
                &sma20 as &(dyn tokio_postgres::types::ToSql + Sync),
                &rsi as &(dyn tokio_postgres::types::ToSql + Sync),
                &daily_return as &(dyn tokio_postgres::types::ToSql + Sync),
                &volume_ratio as &(dyn tokio_postgres::types::ToSql + Sync),
                &(quarter as i32) as &(dyn tokio_postgres::types::ToSql + Sync),
            ],
        ).await?;

        Ok(())
    }

    /// Fetch training data (2021-01-01 to 2023-06-30)
    pub async fn fetch_training_data(&self) -> Result<Vec<MlTrainingRecord>> {
        self.fetch_data_by_date_range("20210101", "20230630").await
    }

    /// Fetch validation data (2023-07-01 to 2024-03-31)
    pub async fn fetch_validation_data(&self) -> Result<Vec<MlTrainingRecord>> {
        self.fetch_data_by_date_range("20230701", "20240331").await
    }

    /// Fetch test data (2024-04-01 onwards)
    pub async fn fetch_test_data(&self) -> Result<Vec<MlTrainingRecord>> {
        self.fetch_data_by_date_range("20240401", "29991231").await
    }

    /// Fetch data for a specific stock and date range
    pub async fn fetch_stock_data(
        &self,
        ts_code: &str,
        start_date: &str,
        end_date: &str,
    ) -> Result<Vec<MlTrainingRecord>> {
        let client = self.pool.get().await?;
        println!(
            "Fetching data for stock={}, start={}, end={}",
            ts_code, start_date, end_date
        );
        let rows = client
            .query(
                "SELECT  ts_code, trade_date, industry, act_ent_type, volume, amount, month, weekday, quarter, week_no,
            open_pct, high_pct, low_pct, close_pct, high_from_open_pct, low_from_open_pct, close_from_open_pct,
            intraday_range_pct, close_position_in_range, ema_5, ema_10, ema_20, ema_30, ema_60, sma_5, sma_10, sma_20,
            macd_line, macd_signal, macd_histogram, macd_weekly_line, macd_weekly_signal, macd_monthly_line, macd_monthly_signal,
            rsi_14, kdj_k, kdj_d, kdj_j, bb_upper, bb_middle, bb_lower, bb_bandwidth, bb_percent_b, atr, volatility_5, volatility_20,
            asi, obv, volume_ratio, price_momentum_5, price_momentum_10, price_momentum_20, price_position_52w, body_size,
            upper_shadow, lower_shadow, trend_strength, adx_14, vwap_distance_pct, cmf_20, williams_r_14, aroon_up_25,
            aroon_down_25, return_lag_1, return_lag_2, return_lag_3, overnight_gap, gap_pct, volume_roc_5, volume_spike,
            price_roc_5, price_roc_10, price_roc_20, hist_volatility_20, is_doji, is_hammer, is_shooting_star, consecutive_days,
            index_csi300_pct_chg, index_csi300_vs_ma5_pct, index_csi300_vs_ma20_pct, index_chinext_pct_chg, index_chinext_vs_ma5_pct,
            index_chinext_vs_ma20_pct, index_xin9_pct_chg, index_xin9_vs_ma5_pct, index_xin9_vs_ma20_pct,
            index_hsi_pct_chg, index_hsi_vs_ma5_pct, index_hsi_vs_ma20_pct,
            fx_usdcnh_pct_chg, fx_usdcnh_vs_ma5_pct, fx_usdcnh_vs_ma20_pct,
            net_mf_vol, net_mf_amount, smart_money_ratio, large_order_flow,
            turnover_rate, turnover_rate_f, /* volume_ratio, */ pe, pe_ttm, pb, ps, ps_ttm, dv_ratio, dv_ttm, total_share, float_share,
            free_share, total_mv, circ_mv,
            vol_percentile, high_vol_regime, 
            pe_percentile_52w, sector_momentum_vs_market, volume_accel_5d, price_vs_52w_high, consecutive_up_days,
            next_day_return,
            next_day_direction, next_3day_return, next_3day_direction
             FROM ml_training_dataset
             WHERE ts_code = $1 AND trade_date >= $2 AND trade_date <= $3
             ORDER BY trade_date",
                &[&ts_code, &start_date, &end_date],
            )
            .await?;

        println!("Query returned {} rows", rows.len());
        Ok(self.parse_ml_records(rows))
    }

    /// Fetch stock data for prediction (features only, no missing columns)
    /// This is used by batch_predict to avoid deserialization errors from missing columns
    pub async fn fetch_stock_data_for_prediction(
        &self,
        ts_code: &str,
        start_date: &str,
        end_date: &str,
    ) -> Result<Vec<MlTrainingRecord>> {
        let client = self.pool.get().await?;
        let rows = client
            .query(
                "SELECT ts_code, trade_date, industry, act_ent_type, volume, amount, month, weekday, quarter, week_no,
            open_pct, high_pct, low_pct, close_pct, high_from_open_pct, low_from_open_pct, close_from_open_pct,
            intraday_range_pct, close_position_in_range, ema_5, ema_10, ema_20, ema_30, ema_60, sma_5, sma_10, sma_20,
            macd_line, macd_signal, macd_histogram, macd_weekly_line, macd_weekly_signal, macd_monthly_line, macd_monthly_signal,
            rsi_14, kdj_k, kdj_d, kdj_j, bb_upper, bb_middle, bb_lower, bb_bandwidth, bb_percent_b, atr, volatility_5, volatility_20,
            asi, obv, volume_ratio, price_momentum_5, price_momentum_10, price_momentum_20, price_position_52w, body_size,
            upper_shadow, lower_shadow, trend_strength, adx_14, vwap_distance_pct, cmf_20,
            williams_r_14, aroon_up_25, aroon_down_25, return_lag_1, return_lag_2, return_lag_3, overnight_gap, gap_pct, 
            volume_roc_5, volume_spike, price_roc_5, price_roc_10, price_roc_20, hist_volatility_20, is_doji, is_hammer, 
            is_shooting_star, consecutive_days, index_csi300_pct_chg, index_csi300_vs_ma5_pct, index_csi300_vs_ma20_pct, 
            index_chinext_pct_chg, index_chinext_vs_ma5_pct, index_chinext_vs_ma20_pct, index_xin9_pct_chg, index_xin9_vs_ma5_pct, 
            index_xin9_vs_ma20_pct, index_hsi_pct_chg, index_hsi_vs_ma5_pct, index_hsi_vs_ma20_pct,
            fx_usdcnh_pct_chg, fx_usdcnh_vs_ma5_pct, fx_usdcnh_vs_ma20_pct,
            net_mf_vol, net_mf_amount, smart_money_ratio, large_order_flow,
            vol_percentile, high_vol_regime,
            COALESCE(pe_percentile_52w, 0.5)::DOUBLE PRECISION as pe_percentile_52w,
            COALESCE(sector_momentum_vs_market, 0.0)::DOUBLE PRECISION as sector_momentum_vs_market,
            COALESCE(volume_accel_5d, 0.0)::DOUBLE PRECISION as volume_accel_5d,
            COALESCE(price_vs_52w_high, 0.0)::DOUBLE PRECISION as price_vs_52w_high,
            COALESCE(consecutive_up_days, 0)::INTEGER as consecutive_up_days,
            0.0::DOUBLE PRECISION as next_day_return,
            0 as next_day_direction,
            0.0::DOUBLE PRECISION as turnover_rate,
            0.0::DOUBLE PRECISION as turnover_rate_f,
            0.0::DOUBLE PRECISION as pe,
            0.0::DOUBLE PRECISION as pe_ttm,
            0.0::DOUBLE PRECISION as pb,
            0.0::DOUBLE PRECISION as ps,
            0.0::DOUBLE PRECISION as ps_ttm,
            0.0::DOUBLE PRECISION as dv_ratio,
            0.0::DOUBLE PRECISION as dv_ttm,
            0.0::DOUBLE PRECISION as total_share,
            0.0::DOUBLE PRECISION as float_share,
            0.0::DOUBLE PRECISION as free_share,
            0.0::DOUBLE PRECISION as total_mv,
            0.0::DOUBLE PRECISION as circ_mv
             FROM ml_training_dataset
             WHERE ts_code = $1 AND trade_date >= $2 AND trade_date <= $3
             ORDER BY trade_date",
                &[&ts_code, &start_date, &end_date],
            )
            .await?;

        Ok(self.parse_ml_records_for_prediction(rows))
    }

    async fn fetch_data_by_date_range(
        &self,
        start_date: &str,
        end_date: &str,
    ) -> Result<Vec<MlTrainingRecord>> {
        let client = self.pool.get().await?;
        let rows = client
            .query(
                "SELECT  ts_code, trade_date, industry, act_ent_type, volume, amount, month, weekday, quarter, week_no,
            open_pct, high_pct, low_pct, close_pct, high_from_open_pct, low_from_open_pct, close_from_open_pct,
            intraday_range_pct, close_position_in_range, ema_5, ema_10, ema_20, ema_30, ema_60, sma_5, sma_10, sma_20,
            macd_line, macd_signal, macd_histogram, macd_weekly_line, macd_weekly_signal, macd_monthly_line, macd_monthly_signal,
            rsi_14, kdj_k, kdj_d, kdj_j, bb_upper, bb_middle, bb_lower, bb_bandwidth, bb_percent_b, atr, volatility_5, volatility_20,
            asi, obv, volume_ratio, price_momentum_5, price_momentum_10, price_momentum_20, price_position_52w, body_size,
            upper_shadow, lower_shadow, trend_strength, adx_14, vwap_distance_pct, cmf_20, williams_r_14, aroon_up_25,
            aroon_down_25, return_lag_1, return_lag_2, return_lag_3, overnight_gap, gap_pct, volume_roc_5, volume_spike,
            price_roc_5, price_roc_10, price_roc_20, hist_volatility_20, is_doji, is_hammer, is_shooting_star, consecutive_days,
            index_csi300_pct_chg, index_csi300_vs_ma5_pct, index_csi300_vs_ma20_pct, index_chinext_pct_chg, index_chinext_vs_ma5_pct,
            index_chinext_vs_ma20_pct, index_xin9_pct_chg, index_xin9_vs_ma5_pct, index_xin9_vs_ma20_pct,
            index_hsi_pct_chg, index_hsi_vs_ma5_pct, index_hsi_vs_ma20_pct,
            fx_usdcnh_pct_chg, fx_usdcnh_vs_ma5_pct, fx_usdcnh_vs_ma20_pct,
            net_mf_vol, net_mf_amount, smart_money_ratio, large_order_flow,
            turnover_rate, turnover_rate_f, /* volume_ratio, */ pe, pe_ttm, pb, ps, ps_ttm, dv_ratio, dv_ttm, total_share, float_share,
            free_share, total_mv, circ_mv,
            vol_percentile, high_vol_regime, 
            pe_percentile_52w, sector_momentum_vs_market, volume_accel_5d, price_vs_52w_high, consecutive_up_days,
            next_day_return,
            next_day_direction, next_3day_return, next_3day_direction
             FROM ml_training_dataset
             WHERE trade_date >= $1 AND trade_date <= $2
                AND macd_weekly_line IS NOT NULL
                AND macd_weekly_signal IS NOT NULL
                AND macd_monthly_line IS NOT NULL
                AND macd_monthly_signal IS NOT NULL
                AND rsi_14 IS NOT NULL
                AND bb_upper IS NOT NULL
                      AND index_csi300_pct_chg IS NOT NULL
      AND index_chinext_pct_chg IS NOT NULL
      AND index_xin9_pct_chg IS NOT NULL
      AND index_hsi_pct_chg IS NOT NULL
      AND fx_usdcnh_pct_chg IS NOT NULL
             ORDER BY ts_code, trade_date",
                &[&start_date, &end_date],
            )
            .await?;

        Ok(self.parse_ml_records(rows))
    }

    fn parse_ml_records(&self, rows: Vec<tokio_postgres::Row>) -> Vec<MlTrainingRecord> {
        rows.into_iter()
            .map(|row| {
                let mut idx = 0;
                MlTrainingRecord {
                    ts_code: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    trade_date: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    industry: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    act_ent_type: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    // province REMOVED
                    volume: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    amount: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    month: {
                        let v = row.get::<_, Option<i16>>(idx).map(|v| v as i32);
                        idx += 1;
                        v
                    },
                    weekday: {
                        let v = row.get::<_, Option<i16>>(idx).map(|v| v as i32);
                        idx += 1;
                        v
                    },
                    quarter: {
                        let v = row.get::<_, Option<i16>>(idx).map(|v| v as i32);
                        idx += 1;
                        v
                    },
                    weekno: {
                        let v = row.get::<_, Option<i16>>(idx).map(|v| v as i32);
                        idx += 1;
                        v
                    },
                    open_pct: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    high_pct: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    low_pct: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    close_pct: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    high_from_open_pct: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    low_from_open_pct: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    close_from_open_pct: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    intraday_range_pct: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    close_position_in_range: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    ema_5: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    ema_10: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    ema_20: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    ema_30: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    ema_60: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    sma_5: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    sma_10: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    sma_20: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    macd_line: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    macd_signal: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    macd_histogram: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    macd_weekly_line: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    macd_weekly_signal: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    macd_monthly_line: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    macd_monthly_signal: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    rsi_14: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    kdj_k: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    kdj_d: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    kdj_j: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    bb_upper: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    bb_middle: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    bb_lower: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    bb_bandwidth: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    bb_percent_b: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    atr: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    volatility_5: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    volatility_20: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    asi: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    obv: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    volume_ratio: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    price_momentum_5: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    price_momentum_10: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    price_momentum_20: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    price_position_52w: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    body_size: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    upper_shadow: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    lower_shadow: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    trend_strength: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    adx_14: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    vwap_distance_pct: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    cmf_20: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    williams_r_14: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    aroon_up_25: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    aroon_down_25: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    return_lag_1: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    return_lag_2: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    return_lag_3: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    overnight_gap: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    gap_pct: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    volume_roc_5: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    volume_spike: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    price_roc_5: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    price_roc_10: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    price_roc_20: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    hist_volatility_20: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    is_doji: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    is_hammer: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    is_shooting_star: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    consecutive_days: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    index_csi300_pct_chg: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    index_csi300_vs_ma5_pct: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    index_csi300_vs_ma20_pct: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    index_xin9_pct_chg: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    index_xin9_vs_ma5_pct: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    index_xin9_vs_ma20_pct: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    index_chinext_pct_chg: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    index_chinext_vs_ma5_pct: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    index_chinext_vs_ma20_pct: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    index_hsi_pct_chg: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    index_hsi_vs_ma5_pct: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    index_hsi_vs_ma20_pct: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    fx_usdcnh_pct_chg: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    fx_usdcnh_vs_ma5_pct: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    fx_usdcnh_vs_ma20_pct: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    net_mf_vol: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    net_mf_amount: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    smart_money_ratio: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    large_order_flow: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    industry_avg_return: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    stock_vs_industry: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    industry_momentum_5d: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    vol_percentile: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    high_vol_regime: {
                        let v = row.get::<_, Option<i16>>(idx).map(|v| v as i32);
                        idx += 1;
                        v
                    },
                    pe_percentile_52w: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    sector_momentum_vs_market: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    volume_accel_5d: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    price_vs_52w_high: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    consecutive_up_days: {
                        let v = row.get::<_, Option<i32>>(idx);
                        idx += 1;
                        v
                    },
                    next_day_return: {
                        let v = row.get(idx);
                        idx += 1;
                        v
                    },
                    next_day_direction: {
                        let v = row.get::<_, Option<i16>>(idx).map(|v| v as i32);
                        idx += 1;
                        v
                    },
                }
            })
            .collect()
    }

    /// Parse ML records for prediction (handles missing columns gracefully)
    /// NOTE: Column order MUST match the SELECT statement in fetch_stock_data_for_prediction exactly!
    /// Column mapping (0-indexed, 106 total):
    /// 0: ts_code, 1: trade_date, 2: industry, 3: act_ent_type, 4: volume, 5: amount
    /// 6-9: month, weekday, quarter, week_no
    /// 10-13: open_pct, high_pct, low_pct, close_pct
    /// 14-18: high_from_open_pct, low_from_open_pct, close_from_open_pct, intraday_range_pct, close_position_in_range
    /// 19-26: ema_5, ema_10, ema_20, ema_30, ema_60, sma_5, sma_10, sma_20
    /// 27-33: macd_line, macd_signal, macd_histogram, macd_weekly_line, macd_weekly_signal, macd_monthly_line, macd_monthly_signal
    /// 34-37: rsi_14, kdj_k, kdj_d, kdj_j
    /// 38-42: bb_upper, bb_middle, bb_lower, bb_bandwidth, bb_percent_b
    /// 43-45: atr, volatility_5, volatility_20
    /// 46-48: asi, obv, volume_ratio
    /// 49-52: price_momentum_5, price_momentum_10, price_momentum_20, price_position_52w
    /// 53-55: body_size, upper_shadow, lower_shadow
    /// 56-59: trend_strength, adx_14, vwap_distance_pct, cmf_20
    /// 60-67: williams_r_14, aroon_up_25, aroon_down_25, return_lag_1, return_lag_2, return_lag_3, overnight_gap, gap_pct
    /// 68-73: volume_roc_5, volume_spike, price_roc_5, price_roc_10, price_roc_20, hist_volatility_20
    /// 74-77: is_doji, is_hammer, is_shooting_star, consecutive_days
    /// 78-80: index_csi300_pct_chg, index_csi300_vs_ma5_pct, index_csi300_vs_ma20_pct
    /// 81-83: index_chinext_pct_chg, index_chinext_vs_ma5_pct, index_chinext_vs_ma20_pct
    /// 84-86: index_xin9_pct_chg, index_xin9_vs_ma5_pct, index_xin9_vs_ma20_pct
    /// 87-88: vol_percentile, high_vol_regime
    /// 89-105: hard-coded values (next_day_return, next_day_direction, turnover_rate, etc.)
    fn parse_ml_records_for_prediction(
        &self,
        rows: Vec<tokio_postgres::Row>,
    ) -> Vec<MlTrainingRecord> {
        rows.into_iter()
            .map(|row| {
                // Read columns 0-88 from actual database values
                let ts_code: String = row.get(0);
                let trade_date: String = row.get(1);
                let industry: Option<String> = row.get(2);
                let act_ent_type: Option<String> = row.get(3);
                let volume: f64 = row.get(4);
                let amount: Option<f64> = row.get(5);
                let month: Option<i32> = row.get::<_, Option<i16>>(6).map(|v| v as i32);
                let weekday: Option<i32> = row.get::<_, Option<i16>>(7).map(|v| v as i32);
                let quarter: Option<i32> = row.get::<_, Option<i16>>(8).map(|v| v as i32);
                let weekno: Option<i32> = row.get::<_, Option<i16>>(9).map(|v| v as i32);
                let open_pct: Option<f64> = row.get(10);
                let high_pct: Option<f64> = row.get(11);
                let low_pct: Option<f64> = row.get(12);
                let close_pct: Option<f64> = row.get(13);
                let high_from_open_pct: Option<f64> = row.get(14);
                let low_from_open_pct: Option<f64> = row.get(15);
                let close_from_open_pct: Option<f64> = row.get(16);
                let intraday_range_pct: Option<f64> = row.get(17);
                let close_position_in_range: Option<f64> = row.get(18);
                let ema_5: Option<f64> = row.get(19);
                let ema_10: Option<f64> = row.get(20);
                let ema_20: Option<f64> = row.get(21);
                let ema_30: Option<f64> = row.get(22);
                let ema_60: Option<f64> = row.get(23);
                let sma_5: Option<f64> = row.get(24);
                let sma_10: Option<f64> = row.get(25);
                let sma_20: Option<f64> = row.get(26);
                let macd_line: Option<f64> = row.get(27);
                let macd_signal: Option<f64> = row.get(28);
                let macd_histogram: Option<f64> = row.get(29);
                let macd_weekly_line: Option<f64> = row.get(30);
                let macd_weekly_signal: Option<f64> = row.get(31);
                let macd_monthly_line: Option<f64> = row.get(32);
                let macd_monthly_signal: Option<f64> = row.get(33);
                let rsi_14: Option<f64> = row.get(34);
                let kdj_k: Option<f64> = row.get(35);
                let kdj_d: Option<f64> = row.get(36);
                let kdj_j: Option<f64> = row.get(37);
                let bb_upper: Option<f64> = row.get(38);
                let bb_middle: Option<f64> = row.get(39);
                let bb_lower: Option<f64> = row.get(40);
                let bb_bandwidth: Option<f64> = row.get(41);
                let bb_percent_b: Option<f64> = row.get(42);
                let atr: Option<f64> = row.get(43);
                let volatility_5: Option<f64> = row.get(44);
                let volatility_20: Option<f64> = row.get(45);
                let asi: Option<f64> = row.get(46);
                let obv: Option<f64> = row.get(47);
                let volume_ratio: Option<f64> = row.get(48);
                let price_momentum_5: Option<f64> = row.get(49);
                let price_momentum_10: Option<f64> = row.get(50);
                let price_momentum_20: Option<f64> = row.get(51);
                let price_position_52w: Option<f64> = row.get(52);
                let body_size: Option<f64> = row.get(53);
                let upper_shadow: Option<f64> = row.get(54);
                let lower_shadow: Option<f64> = row.get(55);
                let trend_strength: Option<f64> = row.get(56);
                let adx_14: Option<f64> = row.get(57);
                let vwap_distance_pct: Option<f64> = row.get(58);
                let cmf_20: Option<f64> = row.get(59);
                let williams_r_14: Option<f64> = row.get(60);
                let aroon_up_25: Option<f64> = row.get(61);
                let aroon_down_25: Option<f64> = row.get(62);
                let return_lag_1: Option<f64> = row.get(63);
                let return_lag_2: Option<f64> = row.get(64);
                let return_lag_3: Option<f64> = row.get(65);
                let overnight_gap: Option<f64> = row.get(66);
                let gap_pct: Option<f64> = row.get(67);
                let volume_roc_5: Option<f64> = row.get(68);
                let volume_spike: Option<bool> = row.get(69);
                let price_roc_5: Option<f64> = row.get(70);
                let price_roc_10: Option<f64> = row.get(71);
                let price_roc_20: Option<f64> = row.get(72);
                let hist_volatility_20: Option<f64> = row.get(73);
                let is_doji: Option<bool> = row.get(74);
                let is_hammer: Option<bool> = row.get(75);
                let is_shooting_star: Option<bool> = row.get(76);
                let consecutive_days: Option<i32> = row.get(77);
                let index_csi300_pct_chg: Option<f64> = row.get(78);
                let index_csi300_vs_ma5_pct: Option<f64> = row.get(79);
                let index_csi300_vs_ma20_pct: Option<f64> = row.get(80);
                let index_chinext_pct_chg: Option<f64> = row.get(81);
                let index_chinext_vs_ma5_pct: Option<f64> = row.get(82);
                let index_chinext_vs_ma20_pct: Option<f64> = row.get(83);
                let index_xin9_pct_chg: Option<f64> = row.get(84);
                let index_xin9_vs_ma5_pct: Option<f64> = row.get(85);
                let index_xin9_vs_ma20_pct: Option<f64> = row.get(86);
                let vol_percentile: Option<f64> = row.try_get(87).ok();
                let high_vol_regime: Option<i32> = row
                    .try_get::<_, Option<i16>>(88)
                    .ok()
                    .flatten()
                    .map(|v| v as i32);
                // NEW: 5 predictive features
                let pe_percentile_52w: Option<f64> = row.try_get(89).ok();
                let sector_momentum_vs_market: Option<f64> = row.try_get(90).ok();
                let volume_accel_5d: Option<f64> = row.try_get(91).ok();
                let price_vs_52w_high: Option<f64> = row.try_get(92).ok();
                let consecutive_up_days: Option<i32> = row.try_get(93).ok();
                // Columns 94+ are hard-coded dummy values (next_day_return, next_day_direction, turnover_rate, etc.)
                let next_day_return: Option<f64> = row.try_get(94).ok();
                let next_day_direction: Option<i32> = row.try_get::<_, i32>(95).ok();

                MlTrainingRecord {
                    ts_code,
                    trade_date,
                    industry,
                    act_ent_type,
                    volume,
                    amount,
                    month,
                    weekday,
                    quarter,
                    weekno,
                    open_pct,
                    high_pct,
                    low_pct,
                    close_pct,
                    high_from_open_pct,
                    low_from_open_pct,
                    close_from_open_pct,
                    intraday_range_pct,
                    close_position_in_range,
                    ema_5,
                    ema_10,
                    ema_20,
                    ema_30,
                    ema_60,
                    sma_5,
                    sma_10,
                    sma_20,
                    macd_line,
                    macd_signal,
                    macd_histogram,
                    macd_weekly_line,
                    macd_weekly_signal,
                    macd_monthly_line,
                    macd_monthly_signal,
                    rsi_14,
                    kdj_k,
                    kdj_d,
                    kdj_j,
                    bb_upper,
                    bb_middle,
                    bb_lower,
                    bb_bandwidth,
                    bb_percent_b,
                    atr,
                    volatility_5,
                    volatility_20,
                    asi,
                    obv,
                    volume_ratio,
                    price_momentum_5,
                    price_momentum_10,
                    price_momentum_20,
                    price_position_52w,
                    body_size,
                    upper_shadow,
                    lower_shadow,
                    trend_strength,
                    adx_14,
                    vwap_distance_pct,
                    cmf_20,
                    williams_r_14,
                    aroon_up_25,
                    aroon_down_25,
                    return_lag_1,
                    return_lag_2,
                    return_lag_3,
                    overnight_gap,
                    gap_pct,
                    volume_roc_5,
                    volume_spike,
                    price_roc_5,
                    price_roc_10,
                    price_roc_20,
                    hist_volatility_20,
                    is_doji,
                    is_hammer,
                    is_shooting_star,
                    consecutive_days,
                    index_csi300_pct_chg,
                    index_csi300_vs_ma5_pct,
                    index_csi300_vs_ma20_pct,
                    index_chinext_pct_chg,
                    index_chinext_vs_ma5_pct,
                    index_chinext_vs_ma20_pct,
                    index_xin9_pct_chg,
                    index_xin9_vs_ma5_pct,
                    index_xin9_vs_ma20_pct,
                    index_hsi_pct_chg: None,
                    index_hsi_vs_ma5_pct: None,
                    index_hsi_vs_ma20_pct: None,
                    fx_usdcnh_pct_chg: None,
                    fx_usdcnh_vs_ma5_pct: None,
                    fx_usdcnh_vs_ma20_pct: None,
                    net_mf_vol: None,
                    net_mf_amount: None,
                    smart_money_ratio: None,
                    large_order_flow: None,
                    industry_avg_return: None,
                    stock_vs_industry: None,
                    industry_momentum_5d: None,
                    vol_percentile,
                    high_vol_regime,
                    pe_percentile_52w,
                    sector_momentum_vs_market,
                    volume_accel_5d,
                    price_vs_52w_high,
                    consecutive_up_days,
                    next_day_return,
                    next_day_direction,
                }
            })
            .collect()
    }

    /// Fetch forward-adjusted stock data for a specific stock and date range
    /// Uses stock_daily_adjusted table which contains qfq (前复权) prices
    pub async fn fetch_stock_data_adjusted(
        &self,
        ts_code: &str,
        start_date: &str,
        end_date: &str,
    ) -> Result<Vec<StockDailyAdjusted>> {
        let client = self.pool.get().await?;
        println!(
            "Fetching adjusted data for stock={}, start={}, end={}",
            ts_code, start_date, end_date
        );
        let rows = client
            .query(
                "SELECT ts_code, trade_date, 
                    open, high, low, close, pre_close, 
                    volume, amount, 
                    change, pct_chg, adj_factor
             FROM stock_daily_adjusted
             WHERE ts_code = $1 AND trade_date >= $2 AND trade_date <= $3
             ORDER BY trade_date",
                &[&ts_code, &start_date, &end_date],
            )
            .await?;

        println!("Query returned {} rows", rows.len());
        Ok(rows
            .into_iter()
            .map(|row| StockDailyAdjusted {
                ts_code: row.get(0),
                trade_date: row.get(1),
                open: row.get(2),
                high: row.get(3),
                low: row.get(4),
                close: row.get(5),
                pre_close: row.get(6),
                volume: row.get(7),
                amount: row.get(8),
                change: row.get(9),
                pct_chg: row.get(10),
                adj_factor: row.get(11),
            })
            .collect())
    }

    /// Fetch forward-adjusted stock data for all stocks in a date range
    pub async fn fetch_adjusted_data_by_date_range(
        &self,
        start_date: &str,
        end_date: &str,
    ) -> Result<Vec<StockDailyAdjusted>> {
        let client = self.pool.get().await?;
        let rows = client
            .query(
                "SELECT ts_code, trade_date, 
                    open, high, low, close, pre_close, 
                    volume, amount, 
                    change, pct_chg, adj_factor
             FROM stock_daily_adjusted
             WHERE trade_date >= $1 AND trade_date <= $2
             ORDER BY ts_code, trade_date",
                &[&start_date, &end_date],
            )
            .await?;

        Ok(rows
            .into_iter()
            .map(|row| StockDailyAdjusted {
                ts_code: row.get(0),
                trade_date: row.get(1),
                open: row.get(2),
                high: row.get(3),
                low: row.get(4),
                close: row.get(5),
                pre_close: row.get(6),
                volume: row.get(7),
                amount: row.get(8),
                change: row.get(9),
                pct_chg: row.get(10),
                adj_factor: row.get(11),
            })
            .collect())
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, FromRow)]
pub struct StockDailyAdjusted {
    pub ts_code: String,
    pub trade_date: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub pre_close: Option<f64>,
    pub volume: f64,
    pub amount: Option<f64>,
    pub change: Option<f64>,
    pub pct_chg: Option<f64>,
    pub adj_factor: Option<f64>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StockPrediction {
    pub ts_code: String,
    pub trade_date: String,
    pub prediction_date: std::time::SystemTime,
    pub predicted_direction: bool,
    pub predicted_return: f64,
    pub confidence: f64,
    pub actual_direction: Option<bool>,
    pub actual_return: Option<f64>,
    pub prediction_correct: Option<bool>,
    pub model_version: Option<String>,
    // 3-day predictions
    pub predicted_3day_return: Option<f64>,
    pub predicted_3day_direction: Option<bool>,
    pub actual_3day_return: Option<f64>,
    pub actual_3day_direction: Option<bool>,
    pub prediction_3day_correct: Option<bool>,
}

/// Lightweight batch payload for prediction upserts
#[derive(Debug, Clone)]
pub struct PredictionInsert {
    pub ts_code: String,
    pub trade_date: String,
    pub predicted_direction: bool,
    pub predicted_return: f64,
    pub confidence: f64,
    pub model_version: String,
    pub predicted_3day_return: Option<f64>,
    pub predicted_3day_direction: Option<bool>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, FromRow)]
pub struct MlTrainingRecord {
    pub ts_code: String,
    pub trade_date: String,
    // Categorical features (for label encoding)
    pub industry: Option<String>,
    pub act_ent_type: Option<String>,
    // province REMOVED
    // Volume and amount
    pub volume: f64,
    pub amount: Option<f64>,
    // Temporal features
    pub month: Option<i32>,
    pub weekday: Option<i32>,
    pub quarter: Option<i32>,
    pub weekno: Option<i32>,
    // Percentage-based OHLC (relative to pre_close)
    pub open_pct: Option<f64>,
    pub high_pct: Option<f64>,
    pub low_pct: Option<f64>,
    pub close_pct: Option<f64>,
    // Intraday price movements
    pub high_from_open_pct: Option<f64>,
    pub low_from_open_pct: Option<f64>,
    pub close_from_open_pct: Option<f64>,
    pub intraday_range_pct: Option<f64>,
    pub close_position_in_range: Option<f64>,
    // Moving Averages
    pub ema_5: Option<f64>,
    pub ema_10: Option<f64>,
    pub ema_20: Option<f64>,
    pub ema_30: Option<f64>,
    pub ema_60: Option<f64>,
    pub sma_5: Option<f64>,
    pub sma_10: Option<f64>,
    pub sma_20: Option<f64>,
    // MACD indicators
    pub macd_line: Option<f64>,
    pub macd_signal: Option<f64>,
    pub macd_histogram: Option<f64>,
    pub macd_weekly_line: Option<f64>,
    pub macd_weekly_signal: Option<f64>,
    pub macd_monthly_line: Option<f64>,
    pub macd_monthly_signal: Option<f64>,
    // Technical indicators
    pub rsi_14: Option<f64>,
    pub kdj_k: Option<f64>,
    pub kdj_d: Option<f64>,
    pub kdj_j: Option<f64>,
    // Bollinger Bands
    pub bb_upper: Option<f64>,
    pub bb_middle: Option<f64>,
    pub bb_lower: Option<f64>,
    pub bb_bandwidth: Option<f64>,
    pub bb_percent_b: Option<f64>,
    // Volatility & Trend
    pub atr: Option<f64>,
    pub volatility_5: Option<f64>,
    pub volatility_20: Option<f64>,
    pub asi: Option<f64>,
    pub obv: Option<f64>,
    pub volume_ratio: Option<f64>,
    // Momentum indicators
    pub price_momentum_5: Option<f64>,
    pub price_momentum_10: Option<f64>,
    pub price_momentum_20: Option<f64>,
    pub price_position_52w: Option<f64>,
    // Candlestick features
    pub body_size: Option<f64>,
    pub upper_shadow: Option<f64>,
    pub lower_shadow: Option<f64>,
    // Trend & Strength
    pub trend_strength: Option<f64>,
    pub adx_14: Option<f64>,
    pub vwap_distance_pct: Option<f64>,
    pub cmf_20: Option<f64>,
    pub williams_r_14: Option<f64>,
    pub aroon_up_25: Option<f64>,
    pub aroon_down_25: Option<f64>,
    // Lagged returns
    pub return_lag_1: Option<f64>,
    pub return_lag_2: Option<f64>,
    pub return_lag_3: Option<f64>,
    // Gap features
    pub overnight_gap: Option<f64>,
    pub gap_pct: Option<f64>,
    // Volume features
    pub volume_roc_5: Option<f64>,
    pub volume_spike: Option<bool>,
    // Price Rate of Change
    pub price_roc_5: Option<f64>,
    pub price_roc_10: Option<f64>,
    pub price_roc_20: Option<f64>,
    pub hist_volatility_20: Option<f64>,
    // Candlestick patterns
    pub is_doji: Option<bool>,
    pub is_hammer: Option<bool>,
    pub is_shooting_star: Option<bool>,
    pub consecutive_days: Option<i32>,
    // Index features - CSI 300 (percentage-based)
    pub index_csi300_pct_chg: Option<f64>,
    pub index_csi300_vs_ma5_pct: Option<f64>,
    pub index_csi300_vs_ma20_pct: Option<f64>,
    // Index features - XIN9 (percentage-based)
    pub index_xin9_pct_chg: Option<f64>,
    pub index_xin9_vs_ma5_pct: Option<f64>,
    pub index_xin9_vs_ma20_pct: Option<f64>,
    // Index features - HSI (Hong Kong Hang Seng, percentage-based)
    pub index_hsi_pct_chg: Option<f64>,
    pub index_hsi_vs_ma5_pct: Option<f64>,
    pub index_hsi_vs_ma20_pct: Option<f64>,
    // FX features - USDCNH (percentage-based)
    pub fx_usdcnh_pct_chg: Option<f64>,
    pub fx_usdcnh_vs_ma5_pct: Option<f64>,
    pub fx_usdcnh_vs_ma20_pct: Option<f64>,
    // Index features - ChiNext (percentage-based)
    pub index_chinext_pct_chg: Option<f64>,
    pub index_chinext_vs_ma5_pct: Option<f64>,
    pub index_chinext_vs_ma20_pct: Option<f64>,
    // Money Flow
    pub net_mf_vol: Option<f64>,
    pub net_mf_amount: Option<f64>,
    pub smart_money_ratio: Option<f64>,
    pub large_order_flow: Option<f64>,
    // Industry features
    pub industry_avg_return: Option<f64>,
    pub stock_vs_industry: Option<f64>,
    pub industry_momentum_5d: Option<f64>,
    // Volatility regime
    pub vol_percentile: Option<f64>,
    pub high_vol_regime: Option<i32>,
    // NEW: 5 predictive features for accuracy improvement
    pub pe_percentile_52w: Option<f64>,
    pub sector_momentum_vs_market: Option<f64>,
    pub volume_accel_5d: Option<f64>,
    pub price_vs_52w_high: Option<f64>,
    pub consecutive_up_days: Option<i32>,
    // Target variables
    pub next_day_return: Option<f64>,
    pub next_day_direction: Option<i32>,
}

impl DbClient {
    /// Fetch stock data from stock_daily and calculate features on-the-fly
    /// This allows prediction for any stock, not just those in ml_training_dataset
    /// Get all distinct stock codes from ml_training_dataset
    pub async fn get_all_stocks(&self) -> Result<Vec<String>> {
        let client = self.pool.get().await?;
        let rows = client
            .query(
                "SELECT DISTINCT ts_code FROM ml_training_dataset ORDER BY ts_code",
                &[],
            )
            .await?;

        Ok(rows.iter().map(|row| row.get(0)).collect())
    }

    /// Get actively listed stocks from stock_basic (faster than DISTINCT)
    /// Note: list_status may be NULL in some databases, so we filter by:
    /// - Excluding ST stocks (special treatment)
    /// - Stocks that exist in ml_training_dataset (have training data)
    pub async fn get_active_stocks(&self) -> Result<Vec<String>> {
        let client = self.pool.get().await?;
        let rows = client
            .query(
                "SELECT DISTINCT sb.ts_code 
                 FROM stock_basic sb
                 WHERE sb.name NOT LIKE 'ST%' 
                   AND sb.name NOT LIKE '%ST'
                   AND EXISTS (
                     SELECT 1 FROM ml_training_dataset m 
                     WHERE m.ts_code = sb.ts_code 
                     LIMIT 1
                   )
                 ORDER BY sb.ts_code",
                &[],
            )
            .await?;

        Ok(rows.iter().map(|row| row.get(0)).collect())
    }

    /// Get latest trade date for a given stock
    pub async fn get_latest_trade_date(&self, ts_code: &str) -> Result<Option<String>> {
        let client = self.pool.get().await?;
        let rows = client
            .query(
                "SELECT MAX(trade_date) FROM ml_training_dataset WHERE ts_code = $1",
                &[&ts_code],
            )
            .await?;

        if rows.is_empty() {
            return Ok(None);
        }

        Ok(rows[0].get(0))
    }

    /// Save a prediction result to stock_predictions table
    /// Returns (inserted_count, updated_count) to track new vs updated predictions
    pub async fn save_prediction(
        &self,
        ts_code: &str,
        trade_date: &str,
        predicted_direction: bool,
        predicted_return: f64,
        confidence: f64,
        model_version: Option<&str>,
        predicted_3day_return: Option<f64>,
        predicted_3day_direction: Option<bool>,
    ) -> Result<(usize, usize)> {
        let client = self.pool.get().await?;

        let row = client
            .query_one(
                "INSERT INTO stock_predictions (
                    ts_code, trade_date, prediction_date, predicted_direction, predicted_return, confidence, model_version, predicted_3day_return, predicted_3day_direction
                ) VALUES ($1, $2, NOW(), $3, $4, $5, COALESCE($6, ''), $7, $8)
                ON CONFLICT (ts_code, trade_date, model_version)
                DO UPDATE SET
                    predicted_direction = EXCLUDED.predicted_direction,
                    predicted_return = EXCLUDED.predicted_return,
                    confidence = EXCLUDED.confidence,
                    prediction_date = EXCLUDED.prediction_date,
                    predicted_3day_return = EXCLUDED.predicted_3day_return,
                    predicted_3day_direction = EXCLUDED.predicted_3day_direction
                RETURNING (xmax = 0) AS inserted",
                &[
                    &ts_code as &(dyn tokio_postgres::types::ToSql + Sync),
                    &trade_date as &(dyn tokio_postgres::types::ToSql + Sync),
                    &predicted_direction as &(dyn tokio_postgres::types::ToSql + Sync),
                    &predicted_return as &(dyn tokio_postgres::types::ToSql + Sync),
                    &confidence as &(dyn tokio_postgres::types::ToSql + Sync),
                    &model_version as &(dyn tokio_postgres::types::ToSql + Sync),
                    &predicted_3day_return as &(dyn tokio_postgres::types::ToSql + Sync),
                    &predicted_3day_direction as &(dyn tokio_postgres::types::ToSql + Sync),
                ],
            )
            .await?;

        let inserted: bool = row.get("inserted");
        if inserted { Ok((1, 0)) } else { Ok((0, 1)) }
    }

    /// Batch upsert predictions to avoid per-stock round trips
    /// Returns (inserted_count, updated_count)
    pub async fn save_predictions_batch(
        &self,
        rows: &[PredictionInsert],
    ) -> Result<(usize, usize)> {
        if rows.is_empty() {
            return Ok((0, 0));
        }

        // De-duplicate within the batch to avoid ON CONFLICT hitting the same key twice
        let deduped_rows: Vec<PredictionInsert> = if rows.len() > 1 {
            let mut unique: HashMap<(String, String, String), PredictionInsert> =
                HashMap::with_capacity(rows.len());
            for row in rows {
                let key = (
                    row.ts_code.clone(),
                    row.trade_date.clone(),
                    row.model_version.clone(),
                );
                unique.insert(key, row.clone());
            }
            unique.into_values().collect()
        } else {
            rows.to_vec()
        };

        let mut inserted_total = 0;
        let mut updated_total = 0;
        let client = self.pool.get().await?;

        const CHUNK_SIZE: usize = 200;

        for chunk in deduped_rows.chunks(CHUNK_SIZE) {
            let mut params: Vec<&(dyn tokio_postgres::types::ToSql + Sync)> =
                Vec::with_capacity(chunk.len() * 8);
            let mut values: Vec<String> = Vec::with_capacity(chunk.len());

            for (i, row) in chunk.iter().enumerate() {
                let base = i * 8;
                values.push(format!(
                    "(${}, ${}, NOW(), ${}, ${}, ${}, ${}, ${}, ${})",
                    base + 1,
                    base + 2,
                    base + 3,
                    base + 4,
                    base + 5,
                    base + 6,
                    base + 7,
                    base + 8
                ));

                params.push(&row.ts_code);
                params.push(&row.trade_date);
                params.push(&row.predicted_direction);
                params.push(&row.predicted_return);
                params.push(&row.confidence);
                params.push(&row.model_version);
                params.push(&row.predicted_3day_return);
                params.push(&row.predicted_3day_direction);
            }

            let sql = format!(
                "INSERT INTO stock_predictions (
                    ts_code, trade_date, prediction_date, predicted_direction, predicted_return, confidence, model_version, predicted_3day_return, predicted_3day_direction
                ) VALUES {}
                ON CONFLICT (ts_code, trade_date, model_version)
                DO UPDATE SET
                    predicted_direction = EXCLUDED.predicted_direction,
                    predicted_return = EXCLUDED.predicted_return,
                    confidence = EXCLUDED.confidence,
                    prediction_date = EXCLUDED.prediction_date,
                    predicted_3day_return = EXCLUDED.predicted_3day_return,
                    predicted_3day_direction = EXCLUDED.predicted_3day_direction
                RETURNING (xmax = 0) AS inserted",
                values.join(", ")
            );

            let result_rows = client.query(&sql, &params).await?;

            for row in result_rows {
                let inserted: bool = row.get("inserted");
                if inserted {
                    inserted_total += 1;
                } else {
                    updated_total += 1;
                }
            }
        }

        Ok((inserted_total, updated_total))
    }

    /// Get predictions for a specific date
    pub async fn get_predictions_by_date(&self, trade_date: &str) -> Result<Vec<StockPrediction>> {
        let client = self.pool.get().await?;
        let rows = client.query(
            "SELECT ts_code, trade_date, prediction_date, predicted_direction, predicted_return, 
                    confidence, actual_direction, actual_return, prediction_correct, model_version,
                    predicted_3day_return, predicted_3day_direction, actual_3day_return, actual_3day_direction, prediction_3day_correct
             FROM stock_predictions 
             WHERE trade_date = $1 
             ORDER BY confidence DESC",
            &[&trade_date]
        ).await?;

        Ok(rows
            .iter()
            .map(|row| StockPrediction {
                ts_code: row.get(0),
                trade_date: row.get(1),
                prediction_date: row.get(2),
                predicted_direction: row.get(3),
                predicted_return: row.get(4),
                confidence: row.get(5),
                actual_direction: row.get(6),
                actual_return: row.get(7),
                prediction_correct: row.get(8),
                model_version: row.get(9),
                predicted_3day_return: row.get(10),
                predicted_3day_direction: row.get(11),
                actual_3day_return: row.get(12),
                actual_3day_direction: row.get(13),
                prediction_3day_correct: row.get(14),
            })
            .collect())
    }

    /// Get predictions for a specific stock
    pub async fn get_predictions_by_stock(&self, ts_code: &str) -> Result<Vec<StockPrediction>> {
        let client = self.pool.get().await?;
        let rows = client.query(
            "SELECT ts_code, trade_date, prediction_date, predicted_direction, predicted_return, 
                    confidence, actual_direction, actual_return, prediction_correct, model_version,
                    predicted_3day_return, predicted_3day_direction, actual_3day_return, actual_3day_direction, prediction_3day_correct
             FROM stock_predictions 
             WHERE ts_code = $1 
             ORDER BY trade_date DESC
             LIMIT 100",
            &[&ts_code]
        ).await?;

        Ok(rows
            .iter()
            .map(|row| StockPrediction {
                ts_code: row.get(0),
                trade_date: row.get(1),
                prediction_date: row.get(2),
                predicted_direction: row.get(3),
                predicted_return: row.get(4),
                confidence: row.get(5),
                actual_direction: row.get(6),
                actual_return: row.get(7),
                prediction_correct: row.get(8),
                model_version: row.get(9),
                predicted_3day_return: row.get(10),
                predicted_3day_direction: row.get(11),
                actual_3day_return: row.get(12),
                actual_3day_direction: row.get(13),
                prediction_3day_correct: row.get(14),
            })
            .collect())
    }

    pub async fn fetch_and_calculate_features(
        &self,
        ts_code: &str,
        start_date: &str,
        end_date: &str,
    ) -> Result<Vec<MlTrainingRecord>> {
        // Step 1: Fetch raw daily data
        let daily_records = self
            .fetch_raw_daily_data(ts_code, start_date, end_date)
            .await?;

        if daily_records.is_empty() {
            anyhow::bail!(
                "No data found for stock {} in date range {} to {}",
                ts_code,
                start_date,
                end_date
            );
        }

        println!(
            "Fetched {} raw daily records for {}",
            daily_records.len(),
            ts_code
        );

        // Step 2: Calculate adjusted prices (forward adjustment)
        let adjusted_records = self.calculate_adjusted_prices(&daily_records);

        // Step 3: Calculate all 105 features

        let feature_records = self.calculate_all_features(adjusted_records).await?;

        println!("Calculated features for {} records", feature_records.len());

        Ok(feature_records)
    }

    /// Fetch raw stock_daily data
    async fn fetch_raw_daily_data(
        &self,
        ts_code: &str,
        start_date: &str,
        end_date: &str,
    ) -> Result<Vec<StockDaily>> {
        let client = self.pool.get().await?;

        // Fetch with extra buffer for indicator calculation (need historical context)
        // Always fetch from a fixed early date to ensure enough history for long-period indicators
        let buffer_start = "20100101".to_string();

        // Fetch pre-adjusted data from adjusted_stock_daily table (same as training data source)
        let rows = client
            .query(
                "SELECT ts_code, trade_date, 
                    open::float8, high::float8, low::float8, close::float8, 
                    volume::float8, amount::float8, pct_chg::float8
             FROM adjusted_stock_daily
             WHERE ts_code = $1 AND trade_date >= $2 AND trade_date <= $3
             ORDER BY trade_date",
                &[
                    &ts_code as &(dyn tokio_postgres::types::ToSql + Sync),
                    &buffer_start as &(dyn tokio_postgres::types::ToSql + Sync),
                    &end_date as &(dyn tokio_postgres::types::ToSql + Sync),
                ],
            )
            .await?;

        Ok(rows
            .into_iter()
            .map(|row| {
                StockDaily {
                    ts_code: row.get(0),
                    trade_date: row.get(1),
                    open: row.get(2),
                    high: row.get(3),
                    low: row.get(4),
                    close: row.get(5),
                    pre_close: None, // Not available in adjusted_stock_daily
                    change: None,    // Not available in adjusted_stock_daily
                    pct_chg: row.get(8),
                    volume: row.get(6),
                    amount: row.get(7),
                    adj_factor: Some(1.0), // Already adjusted by stock_miner
                }
            })
            .collect())
    }

    /// Convert StockDaily to StockDailyAdjusted (no adjustment needed - already adjusted)
    fn calculate_adjusted_prices(&self, records: &[StockDaily]) -> Vec<StockDailyAdjusted> {
        records
            .iter()
            .map(|record| StockDailyAdjusted {
                ts_code: record.ts_code.clone(),
                trade_date: record.trade_date.clone(),
                open: record.open,
                high: record.high,
                low: record.low,
                close: record.close,
                pre_close: record.pre_close,
                volume: record.volume,
                amount: record.amount,
                change: record.change,
                pct_chg: record.pct_chg,
                adj_factor: record.adj_factor,
            })
            .collect()
    }

    /// Calculate all 105 features from adjusted price data
    async fn calculate_all_features(
        &self,
        adjusted_records: Vec<StockDailyAdjusted>,
    ) -> Result<Vec<MlTrainingRecord>> {
        use std::collections::HashMap;

        let mut result = Vec::new();
        let n = adjusted_records.len();

        if n < 60 {
            anyhow::bail!("Need at least 60 days of data for feature calculation");
        }

        // Fetch index data for the same date range
        let start_date = &adjusted_records.first().unwrap().trade_date;
        let end_date = &adjusted_records.last().unwrap().trade_date;
        let index_data = self.fetch_index_data(start_date, end_date).await?;

        // Get industry/company info
        let ts_code = &adjusted_records[0].ts_code;
        let company_info = self.fetch_company_info(ts_code).await?;

        // Calculate features for each day (starting from day 60 to have enough history)
        for i in 60..n {
            let record = &adjusted_records[i];
            let features = self.calculate_single_day_features(
                &adjusted_records[..=i],
                i,
                &index_data,
                &company_info,
            )?;

            result.push(features);
        }

        Ok(result)
    }

    /// Fetch index data (CSI300, Star50 variants, ChiNext)
    async fn fetch_index_data(
        &self,
        start_date: &str,
        end_date: &str,
    ) -> Result<HashMap<String, HashMap<String, IndexDaily>>> {
        let client = self.pool.get().await?;

        let rows = client
            .query(
                "SELECT ts_code, trade_date, close::float8, pct_chg::float8
             FROM index_daily
             WHERE ts_code IN ('000300.SH', '000680.SH', '000688.SH', '399006.SZ')
               AND trade_date >= $1 AND trade_date <= $2
             ORDER BY ts_code, trade_date",
                &[&start_date, &end_date],
            )
            .await
            .unwrap_or_default();

        let mut data: HashMap<String, HashMap<String, IndexDaily>> = HashMap::new();

        for row in rows {
            let ts_code: String = row.get(0);
            let trade_date: String = row.get(1);
            let close: f64 = row.get(2);
            let pct_chg: Option<f64> = row.get(3);

            data.entry(ts_code)
                .or_insert_with(HashMap::new)
                .insert(trade_date, IndexDaily { close, pct_chg });
        }

        Ok(data)
    }

    /// Fetch company basic info
    async fn fetch_company_info(&self, ts_code: &str) -> Result<CompanyInfo> {
        let client = self.pool.get().await?;

        let row = client
            .query_opt(
                "SELECT industry, province, name
             FROM stock_basic
             WHERE ts_code = $1 AND list_status = 'L'
             LIMIT 1",
                &[&ts_code],
            )
            .await?;

        if let Some(row) = row {
            Ok(CompanyInfo {
                industry: row.get(0),
                province: row.get(1),
                act_ent_type: row.get::<_, Option<String>>(2).unwrap_or_default(),
            })
        } else {
            Ok(CompanyInfo::default())
        }
    }

    /// Calculate all features for a single trading day
    fn calculate_single_day_features(
        &self,
        historical_data: &[StockDailyAdjusted],
        current_idx: usize,
        index_data: &HashMap<String, HashMap<String, IndexDaily>>,
        company_info: &CompanyInfo,
    ) -> Result<MlTrainingRecord> {
        let record = &historical_data[current_idx];
        let ts_code = record.ts_code.clone();
        let trade_date = record.trade_date.clone();

        // Parse date for temporal features
        let date = NaiveDate::parse_from_str(&trade_date, "%Y%m%d")?;
        let month = date.month() as i32;
        let weekday = date.weekday().num_days_from_monday() as i32;
        let quarter = ((month - 1) / 3 + 1) as i32;
        let week_no = date.iso_week().week() as i32;

        // Calculate percentage-based OHLC (relative to pre_close)
        // If pre_close is not available, calculate from previous day's close
        let pre_close = if let Some(pc) = record.pre_close {
            pc
        } else if current_idx > 0 {
            historical_data[current_idx - 1].close
        } else {
            record.close // First day fallback
        };

        let open_pct = (record.open - pre_close) / pre_close * 100.0;
        let high_pct = (record.high - pre_close) / pre_close * 100.0;
        let low_pct = (record.low - pre_close) / pre_close * 100.0;
        let close_pct = (record.close - pre_close) / pre_close * 100.0;

        // Intraday movements
        let high_from_open_pct = (record.high - record.open) / record.open * 100.0;
        let low_from_open_pct = (record.low - record.open) / record.open * 100.0;
        let close_from_open_pct = (record.close - record.open) / record.open * 100.0;
        let intraday_range_pct = (record.high - record.low) / record.low * 100.0;
        let close_position = if record.high > record.low {
            (record.close - record.low) / (record.high - record.low)
        } else {
            0.5
        };

        // Calculate EMAs and SMAs
        let (ema_5, ema_10, ema_20, ema_30, ema_60) =
            self.calculate_emas(historical_data, current_idx);
        let (sma_5, sma_10, sma_20) = self.calculate_smas(historical_data, current_idx);

        // MACD indicators
        let (macd_line, macd_signal, macd_histogram) =
            self.calculate_macd(historical_data, current_idx, 12, 26, 9);
        let (macd_weekly_line, macd_weekly_signal, _, _) =
            self.calculate_macd_weekly(historical_data, current_idx);
        let (macd_monthly_line, macd_monthly_signal, _, _) =
            self.calculate_macd_monthly(historical_data, current_idx);

        // RSI
        let rsi_14 = self.calculate_rsi(historical_data, current_idx, 14);

        // KDJ
        let (kdj_k, kdj_d, kdj_j) = self.calculate_kdj(historical_data, current_idx, 9);

        // Bollinger Bands
        let (bb_upper, bb_middle, bb_lower, bb_bandwidth, bb_percent_b) =
            self.calculate_bollinger_bands(historical_data, current_idx, 20, 2.0);

        // Volatility indicators
        let atr = self.calculate_atr(historical_data, current_idx, 14);
        let volatility_5 = self.calculate_volatility(historical_data, current_idx, 5);
        let volatility_20 = self.calculate_volatility(historical_data, current_idx, 20);
        let asi = self.calculate_asi(historical_data, current_idx);
        let obv = self.calculate_obv(historical_data, current_idx);
        let volume_ratio = self.calculate_volume_ratio(historical_data, current_idx, 5);

        // Momentum indicators
        let price_momentum_5 = self.calculate_momentum(historical_data, current_idx, 5);
        let price_momentum_10 = self.calculate_momentum(historical_data, current_idx, 10);
        let price_momentum_20 = self.calculate_momentum(historical_data, current_idx, 20);
        let price_position_52w = self.calculate_52w_position(historical_data, current_idx);

        // Candlestick features
        let body_size = (record.close - record.open).abs() / record.open * 100.0;
        let upper_shadow = (record.high - record.close.max(record.open)) / record.open * 100.0;
        let lower_shadow = (record.close.min(record.open) - record.low) / record.open * 100.0;

        // Trend strength
        let trend_strength = self.calculate_trend_strength(historical_data, current_idx, 20);
        let adx_14 = self.calculate_adx(historical_data, current_idx, 14);
        let vwap_distance_pct = self.calculate_vwap_distance(historical_data, current_idx, 20);
        let cmf_20 = self.calculate_cmf(historical_data, current_idx, 20);
        // MFI calculation removed due to insufficient reliable historical window
        let williams_r_14 = self.calculate_williams_r(historical_data, current_idx, 14);
        let (aroon_up, aroon_down) = self.calculate_aroon(historical_data, current_idx, 25);

        // Lagged returns
        let return_lag_1 = self.calculate_lagged_return(historical_data, current_idx, 1);
        let return_lag_2 = self.calculate_lagged_return(historical_data, current_idx, 2);
        let return_lag_3 = self.calculate_lagged_return(historical_data, current_idx, 3);

        // Gap features
        let overnight_gap = (record.open - pre_close) / pre_close * 100.0;
        let gap_pct = overnight_gap;

        // Volume features
        let volume_roc_5 = self.calculate_volume_roc(historical_data, current_idx, 5);
        let volume_spike = volume_roc_5 > 2.0;

        // Price ROC
        let price_roc_5 = self.calculate_price_roc(historical_data, current_idx, 5);
        let price_roc_10 = self.calculate_price_roc(historical_data, current_idx, 10);
        let price_roc_20 = self.calculate_price_roc(historical_data, current_idx, 20);
        let hist_volatility_20 = volatility_20;

        // Candlestick patterns
        let is_doji = body_size < 0.1;
        let is_hammer = lower_shadow > body_size * 2.0 && upper_shadow < body_size;
        let is_shooting_star = upper_shadow > body_size * 2.0 && lower_shadow < body_size;
        let consecutive_days = self.calculate_consecutive_days(historical_data, current_idx);

        // Index features
        let (csi300_pct, csi300_vs_ma5, csi300_vs_ma20) =
            self.get_index_features(index_data, "000300.SH", &trade_date);
        // Star50 variant: prefer 000680.SH, fallback to 000688.SH if missing
        let (star50_pct, star50_vs_ma5, star50_vs_ma20) =
            self.get_index_features_any(index_data, &["000680.SH", "000688.SH"], &trade_date);
        let (chinext_pct, chinext_vs_ma5, chinext_vs_ma20) =
            self.get_index_features(index_data, "399006.SZ", &trade_date);

        // Money flow (placeholder - would need additional data)
        let net_mf_vol = Some(0.0);
        let net_mf_amount = Some(0.0);
        let smart_money_ratio = Some(0.0);
        let large_order_flow = Some(0.0);

        // Industry features (placeholder - would need industry aggregation)
        let industry_avg_return = Some(0.0);
        let stock_vs_industry = Some(0.0);
        let industry_momentum_5d = Some(0.0);

        // Volatility regime
        let vol_percentile =
            Some(self.calculate_volatility_percentile(historical_data, current_idx, volatility_20));
        let high_vol_regime = if vol_percentile.unwrap_or(0.5) > 0.75 {
            Some(1)
        } else {
            Some(0)
        };

        // Target variables (if next day exists)
        let (next_day_return, next_day_direction) = if current_idx + 1 < historical_data.len() {
            let next_close = historical_data[current_idx + 1].close;
            let ret = (next_close - record.close) / record.close * 100.0; // Convert to percentage to match feature scale
            let dir = if ret > 0.0 { 1 } else { 0 };
            (Some(ret), Some(dir))
        } else {
            (None, None)
        };

        Ok(MlTrainingRecord {
            ts_code,
            trade_date,
            industry: company_info.industry.clone(),
            act_ent_type: Some(company_info.act_ent_type.clone()),
            // province: company_info.province.clone(),
            volume: record.volume,
            amount: record.amount,
            month: Some(month),
            weekday: Some(weekday),
            quarter: Some(quarter),
            weekno: Some(week_no),
            open_pct: Some(open_pct),
            high_pct: Some(high_pct),
            low_pct: Some(low_pct),
            close_pct: Some(close_pct),
            high_from_open_pct: Some(high_from_open_pct),
            low_from_open_pct: Some(low_from_open_pct),
            close_from_open_pct: Some(close_from_open_pct),
            intraday_range_pct: Some(intraday_range_pct),
            close_position_in_range: Some(close_position),
            ema_5: Some(ema_5),
            ema_10: Some(ema_10),
            ema_20: Some(ema_20),
            ema_30: Some(ema_30),
            ema_60: Some(ema_60),
            sma_5: Some(sma_5),
            sma_10: Some(sma_10),
            sma_20: Some(sma_20),
            macd_line: Some(macd_line),
            macd_signal: Some(macd_signal),
            macd_histogram: Some(macd_histogram),
            macd_weekly_line: Some(macd_weekly_line),
            macd_weekly_signal: Some(macd_weekly_signal),
            macd_monthly_line: Some(macd_monthly_line),
            macd_monthly_signal: Some(macd_monthly_signal),
            rsi_14: Some(rsi_14),
            kdj_k: Some(kdj_k),
            kdj_d: Some(kdj_d),
            kdj_j: Some(kdj_j),
            bb_upper: Some(bb_upper),
            bb_middle: Some(bb_middle),
            bb_lower: Some(bb_lower),
            bb_bandwidth: Some(bb_bandwidth),
            bb_percent_b: Some(bb_percent_b),
            atr: Some(atr),
            volatility_5: Some(volatility_5),
            volatility_20: Some(volatility_20),
            asi: Some(asi),
            obv: Some(obv),
            volume_ratio: Some(volume_ratio),
            price_momentum_5: Some(price_momentum_5),
            price_momentum_10: Some(price_momentum_10),
            price_momentum_20: Some(price_momentum_20),
            price_position_52w: Some(price_position_52w),
            body_size: Some(body_size),
            upper_shadow: Some(upper_shadow),
            lower_shadow: Some(lower_shadow),
            trend_strength: Some(trend_strength),
            adx_14: Some(adx_14),
            vwap_distance_pct: Some(vwap_distance_pct),
            cmf_20: Some(cmf_20),
            williams_r_14: Some(williams_r_14),
            aroon_up_25: Some(aroon_up),
            aroon_down_25: Some(aroon_down),
            return_lag_1: Some(return_lag_1),
            return_lag_2: Some(return_lag_2),
            return_lag_3: Some(return_lag_3),
            overnight_gap: Some(overnight_gap),
            gap_pct: Some(gap_pct),
            volume_roc_5: Some(volume_roc_5),
            volume_spike: Some(volume_spike),
            price_roc_5: Some(price_roc_5),
            price_roc_10: Some(price_roc_10),
            price_roc_20: Some(price_roc_20),
            hist_volatility_20: Some(hist_volatility_20),
            is_doji: Some(is_doji),
            is_hammer: Some(is_hammer),
            is_shooting_star: Some(is_shooting_star),
            consecutive_days: Some(consecutive_days),
            index_csi300_pct_chg: Some(csi300_pct),
            index_csi300_vs_ma5_pct: Some(csi300_vs_ma5),
            index_csi300_vs_ma20_pct: Some(csi300_vs_ma20),
            index_xin9_pct_chg: Some(star50_pct),
            index_xin9_vs_ma5_pct: Some(star50_vs_ma5),
            index_xin9_vs_ma20_pct: Some(star50_vs_ma20),
            index_chinext_pct_chg: Some(chinext_pct),
            index_chinext_vs_ma5_pct: Some(chinext_vs_ma5),
            index_chinext_vs_ma20_pct: Some(chinext_vs_ma20),
            index_hsi_pct_chg: None,
            index_hsi_vs_ma5_pct: None,
            index_hsi_vs_ma20_pct: None,
            fx_usdcnh_pct_chg: None,
            fx_usdcnh_vs_ma5_pct: None,
            fx_usdcnh_vs_ma20_pct: None,
            net_mf_vol,
            net_mf_amount,
            smart_money_ratio,
            large_order_flow,
            industry_avg_return,
            stock_vs_industry,
            industry_momentum_5d,
            vol_percentile,
            high_vol_regime,
            pe_percentile_52w: None,
            sector_momentum_vs_market: None,
            volume_accel_5d: None,
            price_vs_52w_high: None,
            consecutive_up_days: None,
            next_day_return,
            next_day_direction,
        })
    }

    // Technical indicator calculation helper methods
    // (These would be implemented with proper TA logic)

    fn calculate_emas(&self, data: &[StockDailyAdjusted], idx: usize) -> (f64, f64, f64, f64, f64) {
        // Simplified EMA calculation
        let periods = vec![5, 10, 20, 30, 60];
        let mut emas = Vec::new();

        for &period in &periods {
            if idx + 1 < period {
                emas.push(0.0);
                continue;
            }

            let alpha = 2.0 / (period as f64 + 1.0);
            let mut ema = data[idx - period + 1].close;

            for i in (idx - period + 2)..=idx {
                ema = alpha * data[i].close + (1.0 - alpha) * ema;
            }

            let ema_pct = (ema - data[idx].close) / data[idx].close * 100.0;
            emas.push(ema_pct);
        }

        (emas[0], emas[1], emas[2], emas[3], emas[4])
    }

    fn calculate_smas(&self, data: &[StockDailyAdjusted], idx: usize) -> (f64, f64, f64) {
        let periods = vec![5, 10, 20];
        let mut smas = Vec::new();

        for &period in &periods {
            if idx + 1 < period {
                smas.push(0.0);
                continue;
            }

            let sum: f64 = data[(idx - period + 1)..=idx].iter().map(|r| r.close).sum();
            let sma = sum / period as f64;
            let sma_pct = (sma - data[idx].close) / data[idx].close * 100.0;
            smas.push(sma_pct);
        }

        (smas[0], smas[1], smas[2])
    }

    fn calculate_macd(
        &self,
        data: &[StockDailyAdjusted],
        idx: usize,
        fast: usize,
        slow: usize,
        signal: usize,
    ) -> (f64, f64, f64) {
        if idx + 1 < slow {
            return (0.0, 0.0, 0.0);
        }

        // Calculate fast and slow EMAs
        let alpha_fast = 2.0 / (fast as f64 + 1.0);
        let alpha_slow = 2.0 / (slow as f64 + 1.0);

        let mut fast_ema = data[idx - slow + 1].close;
        let mut slow_ema = data[idx - slow + 1].close;

        for i in (idx - slow + 2)..=idx {
            fast_ema = alpha_fast * data[i].close + (1.0 - alpha_fast) * fast_ema;
            slow_ema = alpha_slow * data[i].close + (1.0 - alpha_slow) * slow_ema;
        }

        let macd_line = (fast_ema - slow_ema) / data[idx].close;

        // Signal line (EMA of MACD)
        let macd_signal = macd_line * 0.9; // Simplified
        let histogram = macd_line - macd_signal;

        (macd_line, macd_signal, histogram)
    }

    fn calculate_macd_weekly(
        &self,
        data: &[StockDailyAdjusted],
        idx: usize,
    ) -> (f64, f64, f64, f64) {
        // Simplified weekly MACD using 60/130 day periods
        let (line, signal, hist) = self.calculate_macd(data, idx, 60, 130, 45);
        (line, signal, hist, 0.0)
    }

    fn calculate_macd_monthly(
        &self,
        data: &[StockDailyAdjusted],
        idx: usize,
    ) -> (f64, f64, f64, f64) {
        // Simplified monthly MACD using longer periods
        if idx < 120 {
            return (0.0, 0.0, 0.0, 0.0);
        }
        let (line, signal, hist) = self.calculate_macd(data, idx, 120, 260, 90);
        (line, signal, hist, 0.0)
    }

    fn calculate_rsi(&self, data: &[StockDailyAdjusted], idx: usize, period: usize) -> f64 {
        if idx < period {
            return 50.0;
        }

        let mut gains = 0.0;
        let mut losses = 0.0;

        for i in (idx - period + 1)..=idx {
            let change = data[i].close - data[i - 1].close;
            if change > 0.0 {
                gains += change;
            } else {
                losses += -change;
            }
        }

        let avg_gain = gains / period as f64;
        let avg_loss = losses / period as f64;

        if avg_loss == 0.0 {
            return 100.0;
        }

        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }

    fn calculate_kdj(
        &self,
        data: &[StockDailyAdjusted],
        idx: usize,
        period: usize,
    ) -> (f64, f64, f64) {
        if idx < period {
            return (50.0, 50.0, 50.0);
        }

        let mut highest = data[idx - period + 1].high;
        let mut lowest = data[idx - period + 1].low;

        for i in (idx - period + 1)..=idx {
            highest = highest.max(data[i].high);
            lowest = lowest.min(data[i].low);
        }

        let rsv = if highest > lowest {
            (data[idx].close - lowest) / (highest - lowest) * 100.0
        } else {
            50.0
        };

        let k = rsv * 0.33 + 50.0 * 0.67; // Simplified
        let d = k * 0.33 + 50.0 * 0.67;
        let j = 3.0 * k - 2.0 * d;

        (k, d, j)
    }

    fn calculate_bollinger_bands(
        &self,
        data: &[StockDailyAdjusted],
        idx: usize,
        period: usize,
        std_dev: f64,
    ) -> (f64, f64, f64, f64, f64) {
        if idx + 1 < period {
            return (0.0, 0.0, 0.0, 0.0, 0.5);
        }

        let sum: f64 = data[(idx - period + 1)..=idx].iter().map(|r| r.close).sum();
        let mean = sum / period as f64;

        let variance: f64 = data[(idx - period + 1)..=idx]
            .iter()
            .map(|r| (r.close - mean).powi(2))
            .sum::<f64>()
            / period as f64;
        let std = variance.sqrt();

        let upper = mean + std_dev * std;
        let lower = mean - std_dev * std;
        let bandwidth = if mean > 0.0 {
            (upper - lower) / mean
        } else {
            0.0
        };
        let percent_b = if upper > lower {
            (data[idx].close - lower) / (upper - lower)
        } else {
            0.5
        };

        let upper_pct = (upper - data[idx].close) / data[idx].close * 100.0;
        let middle_pct = (mean - data[idx].close) / data[idx].close * 100.0;
        let lower_pct = (lower - data[idx].close) / data[idx].close * 100.0;

        (upper_pct, middle_pct, lower_pct, bandwidth, percent_b)
    }

    fn calculate_atr(&self, data: &[StockDailyAdjusted], idx: usize, period: usize) -> f64 {
        if idx < period {
            return 0.0;
        }

        let mut tr_sum = 0.0;
        for i in (idx - period + 1)..=idx {
            let prev_close = if i > 0 {
                data[i - 1].close
            } else {
                data[i].open
            };
            let tr = (data[i].high - data[i].low)
                .max((data[i].high - prev_close).abs())
                .max((data[i].low - prev_close).abs());
            tr_sum += tr;
        }

        let atr = tr_sum / period as f64;
        atr / data[idx].close * 100.0
    }

    fn calculate_volatility(&self, data: &[StockDailyAdjusted], idx: usize, period: usize) -> f64 {
        if idx < period {
            return 0.0;
        }

        let returns: Vec<f64> = (idx - period + 1..=idx)
            .map(|i| {
                if i > 0 {
                    (data[i].close / data[i - 1].close - 1.0).powi(2)
                } else {
                    0.0
                }
            })
            .collect();

        let variance = returns.iter().sum::<f64>() / period as f64;
        variance.sqrt()
    }

    fn calculate_asi(&self, _data: &[StockDailyAdjusted], _idx: usize) -> f64 {
        // Simplified ASI - would need full implementation
        0.0
    }

    fn calculate_obv(&self, data: &[StockDailyAdjusted], idx: usize) -> f64 {
        let mut obv = 0.0;
        for i in 1..=idx.min(60) {
            if data[i].close > data[i - 1].close {
                obv += data[i].volume;
            } else if data[i].close < data[i - 1].close {
                obv -= data[i].volume;
            }
        }
        obv
    }

    fn calculate_volume_ratio(
        &self,
        data: &[StockDailyAdjusted],
        idx: usize,
        period: usize,
    ) -> f64 {
        if idx < period {
            return 1.0;
        }

        let avg_volume: f64 = data[(idx - period + 1)..=idx]
            .iter()
            .map(|r| r.volume)
            .sum::<f64>()
            / period as f64;
        if avg_volume > 0.0 {
            data[idx].volume / avg_volume
        } else {
            1.0
        }
    }

    fn calculate_momentum(&self, data: &[StockDailyAdjusted], idx: usize, period: usize) -> f64 {
        if idx < period {
            return 0.0;
        }
        (data[idx].close / data[idx - period].close - 1.0)
    }

    fn calculate_52w_position(&self, data: &[StockDailyAdjusted], idx: usize) -> f64 {
        let period = 252.min(idx + 1);
        if period < 2 {
            return 0.5;
        }

        let start = idx - period + 1;
        let mut highest = data[start].high;
        let mut lowest = data[start].low;

        for i in start..=idx {
            highest = highest.max(data[i].high);
            lowest = lowest.min(data[i].low);
        }

        if highest > lowest {
            (data[idx].close - lowest) / (highest - lowest)
        } else {
            0.5
        }
    }

    fn calculate_trend_strength(
        &self,
        data: &[StockDailyAdjusted],
        idx: usize,
        period: usize,
    ) -> f64 {
        if idx < period {
            return 0.0;
        }

        let start_price = data[idx - period + 1].close;
        let end_price = data[idx].close;

        ((end_price - start_price) / start_price).clamp(-1.0, 1.0)
    }

    fn calculate_adx(&self, data: &[StockDailyAdjusted], idx: usize, period: usize) -> f64 {
        if idx < period + 1 {
            return 25.0;
        }

        // Simplified ADX calculation
        let mut dm_plus_sum = 0.0;
        let mut dm_minus_sum = 0.0;
        let mut tr_sum = 0.0;

        for i in (idx - period + 1)..=idx {
            if i > 0 {
                let dm_plus = (data[i].high - data[i - 1].high).max(0.0);
                let dm_minus = (data[i - 1].low - data[i].low).max(0.0);
                let tr = (data[i].high - data[i].low)
                    .max((data[i].high - data[i - 1].close).abs())
                    .max((data[i].low - data[i - 1].close).abs());

                dm_plus_sum += dm_plus;
                dm_minus_sum += dm_minus;
                tr_sum += tr;
            }
        }

        if tr_sum > 0.0 {
            let di_plus = (dm_plus_sum / tr_sum) * 100.0;
            let di_minus = (dm_minus_sum / tr_sum) * 100.0;
            let dx = ((di_plus - di_minus).abs() / (di_plus + di_minus)) * 100.0;
            dx
        } else {
            25.0
        }
    }

    fn calculate_vwap_distance(
        &self,
        data: &[StockDailyAdjusted],
        idx: usize,
        period: usize,
    ) -> f64 {
        if idx < period {
            return 0.0;
        }

        let mut total_pv = 0.0;
        let mut total_v = 0.0;

        for i in (idx - period + 1)..=idx {
            let typical_price = (data[i].high + data[i].low + data[i].close) / 3.0;
            total_pv += typical_price * data[i].volume;
            total_v += data[i].volume;
        }

        if total_v > 0.0 {
            let vwap = total_pv / total_v;
            (vwap - data[idx].close) / data[idx].close * 100.0
        } else {
            0.0
        }
    }

    fn calculate_cmf(&self, data: &[StockDailyAdjusted], idx: usize, period: usize) -> f64 {
        if idx < period {
            return 0.0;
        }

        let mut mf_volume = 0.0;
        let mut total_volume = 0.0;

        for i in (idx - period + 1)..=idx {
            let mf_multiplier = if data[i].high != data[i].low {
                ((data[i].close - data[i].low) - (data[i].high - data[i].close))
                    / (data[i].high - data[i].low)
            } else {
                0.0
            };
            mf_volume += mf_multiplier * data[i].volume;
            total_volume += data[i].volume;
        }

        if total_volume > 0.0 {
            mf_volume / total_volume
        } else {
            0.0
        }
    }

    fn calculate_mfi(&self, data: &[StockDailyAdjusted], idx: usize, period: usize) -> f64 {
        if idx < period {
            return 50.0;
        }

        let mut positive_flow = 0.0;
        let mut negative_flow = 0.0;

        for i in (idx - period + 1)..=idx {
            if i > 0 {
                let typical_price = (data[i].high + data[i].low + data[i].close) / 3.0;
                let prev_typical = (data[i - 1].high + data[i - 1].low + data[i - 1].close) / 3.0;
                let money_flow = typical_price * data[i].volume;

                if typical_price > prev_typical {
                    positive_flow += money_flow;
                } else if typical_price < prev_typical {
                    negative_flow += money_flow;
                }
            }
        }

        if negative_flow == 0.0 {
            return 100.0;
        }

        let mfi_ratio = positive_flow / negative_flow;
        100.0 - (100.0 / (1.0 + mfi_ratio))
    }

    fn calculate_williams_r(&self, data: &[StockDailyAdjusted], idx: usize, period: usize) -> f64 {
        if idx < period {
            return -50.0;
        }

        let mut highest = data[idx - period + 1].high;
        let mut lowest = data[idx - period + 1].low;

        for i in (idx - period + 1)..=idx {
            highest = highest.max(data[i].high);
            lowest = lowest.min(data[i].low);
        }

        if highest > lowest {
            ((highest - data[idx].close) / (highest - lowest)) * -100.0
        } else {
            -50.0
        }
    }

    fn calculate_aroon(
        &self,
        data: &[StockDailyAdjusted],
        idx: usize,
        period: usize,
    ) -> (f64, f64) {
        if idx < period {
            return (50.0, 50.0);
        }

        let mut days_since_high = 0;
        let mut days_since_low = 0;
        let mut highest = data[idx].high;
        let mut lowest = data[idx].low;

        for i in (idx - period + 1..=idx).rev() {
            if data[i].high >= highest {
                highest = data[i].high;
                days_since_high = idx - i;
            }
            if data[i].low <= lowest {
                lowest = data[i].low;
                days_since_low = idx - i;
            }
        }

        let aroon_up = ((period - days_since_high) as f64 / period as f64) * 100.0;
        let aroon_down = ((period - days_since_low) as f64 / period as f64) * 100.0;

        (aroon_up, aroon_down)
    }

    fn calculate_lagged_return(&self, data: &[StockDailyAdjusted], idx: usize, lag: usize) -> f64 {
        if idx < lag {
            return 0.0;
        }
        (data[idx].close / data[idx - lag].close - 1.0)
    }

    fn calculate_volume_roc(&self, data: &[StockDailyAdjusted], idx: usize, period: usize) -> f64 {
        if idx < period {
            return 0.0;
        }
        let avg_volume: f64 = data[(idx - period + 1)..=idx]
            .iter()
            .map(|r| r.volume)
            .sum::<f64>()
            / period as f64;
        if avg_volume > 0.0 {
            (data[idx].volume - avg_volume) / avg_volume
        } else {
            0.0
        }
    }

    fn calculate_price_roc(&self, data: &[StockDailyAdjusted], idx: usize, period: usize) -> f64 {
        if idx < period {
            return 0.0;
        }
        (data[idx].close / data[idx - period].close - 1.0)
    }

    fn calculate_consecutive_days(&self, data: &[StockDailyAdjusted], idx: usize) -> i32 {
        if idx == 0 {
            return 0;
        }

        let mut count = 0;
        let is_up = data[idx].close > data[idx - 1].close;

        for i in (1..=idx.min(10)).rev() {
            let current_up = data[idx - i + 1].close > data[idx - i].close;
            if current_up == is_up {
                count += 1;
            } else {
                break;
            }
        }

        if is_up { count } else { -count }
    }

    fn calculate_volatility_percentile(
        &self,
        data: &[StockDailyAdjusted],
        idx: usize,
        current_vol: f64,
    ) -> f64 {
        let period = 252.min(idx + 1);
        if period < 2 {
            return 0.5;
        }

        let mut vols = Vec::new();
        for i in (idx - period + 1)..=idx {
            let vol = self.calculate_volatility(data, i, 20.min(i + 1));
            vols.push(vol);
        }

        vols.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let count_below = vols.iter().filter(|&&v| v < current_vol).count();
        count_below as f64 / vols.len() as f64
    }

    fn get_index_features(
        &self,
        index_data: &HashMap<String, HashMap<String, IndexDaily>>,
        index_code: &str,
        trade_date: &str,
    ) -> (f64, f64, f64) {
        if let Some(index_records) = index_data.get(index_code) {
            if let Some(record) = index_records.get(trade_date) as Option<&IndexDaily> {
                let pct_chg = record.pct_chg.unwrap_or(0.0);
                // Simplified - would need to calculate MAs from index data
                (pct_chg, 0.0, 0.0)
            } else {
                (0.0, 0.0, 0.0)
            }
        } else {
            (0.0, 0.0, 0.0)
        }
    }

    fn get_index_features_any(
        &self,
        index_data: &HashMap<String, HashMap<String, IndexDaily>>,
        index_codes: &[&str],
        trade_date: &str,
    ) -> (f64, f64, f64) {
        for &code in index_codes {
            if let Some(map) = index_data.get(code) {
                if let Some(record) = map.get(trade_date) {
                    let pct_chg = record.pct_chg.unwrap_or(0.0);
                    return (pct_chg, 0.0, 0.0);
                }
            }
        }
        (0.0, 0.0, 0.0)
    }
}

// Helper structs
#[derive(Debug, Clone)]
struct StockDaily {
    ts_code: String,
    trade_date: String,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    pre_close: Option<f64>,
    change: Option<f64>,
    pct_chg: Option<f64>,
    volume: f64,
    amount: Option<f64>,
    adj_factor: Option<f64>,
}

#[derive(Debug, Clone)]
struct IndexDaily {
    close: f64,
    pct_chg: Option<f64>,
}

#[derive(Debug, Clone, Default)]
struct CompanyInfo {
    industry: Option<String>,
    province: Option<String>,
    act_ent_type: String,
}
