// use yahoo_finance_api::Quote; // Disabled - not needed for database training
use crate::db::MlTrainingRecord;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct StockItem {
    // 105 features (updated for percentage-based schema):
    // ... (doc omitted for brevity)
    pub values: Vec<[f32; 105]>,
    // metadata for sample weighting
    pub last_trade_date: Option<String>, // YYYYMMDD of the last timestep in sequence
    pub dataset_last_date: Option<String>, // most recent date available for that stock dataset
    // next-day returns aligned with `values` entries (Option<f32> per timestep)
    pub next_day_returns: Vec<Option<f32>>,
} 

#[allow(dead_code)]
pub struct DummyStockDataset {
    size: usize,
    seq_len: usize,
}

#[allow(dead_code)]
impl DummyStockDataset {
    pub fn new(size: usize, seq_len: usize) -> Self {
        Self { size, seq_len }
    }
}

impl DummyStockDataset {
    pub fn get(&self, index: usize) -> Option<StockItem> {
        if index >= self.size {
            return None;
        }
        // Generate deterministic "random" float data (sine wave pattern)
        let mut values: Vec<[f32; 105]> = Vec::with_capacity(self.seq_len);
        for i in 0..self.seq_len {
            let t = (index * self.seq_len + i) as f32;
            // Base price movement
            let base = (t * 0.1).sin() + (t * 0.01).cos() * 0.5 + 10.0;

            // Generate OHLC relative to base
            let open = base + (t * 0.5).sin() * 0.1;
            let close = base + (t * 0.5 + 1.0).sin() * 0.1;
            let high = f32::max(open, close) + 0.2;
            let low = f32::min(open, close) - 0.2;
            let volume = (base * 1000.0).abs();
            let amount = volume * close;

            // Create 105-feature array for dummy data
            values.push([
                // [0-2] Categorical (dummy)
                0.0,
                0.0,
                0.0,
                // [3-4] Volume
                volume,
                amount,
                // [5-8] Temporal
                ((t / 30.0) % 12.0).floor() + 1.0, // month
                ((t / 7.0) % 7.0).floor(),         // weekday
                ((t / 90.0) % 4.0).floor() + 1.0,  // quarter
                ((t / 7.0) % 53.0).floor() + 1.0,  // week_no
                // [9-12] Price percentages
                if i > 0 {
                    (open - values[i - 1][12]) / values[i - 1][12] * 100.0
                } else {
                    0.0
                },
                if i > 0 {
                    (high - values[i - 1][12]) / values[i - 1][12] * 100.0
                } else {
                    0.0
                },
                if i > 0 {
                    (low - values[i - 1][12]) / values[i - 1][12] * 100.0
                } else {
                    0.0
                },
                if i > 0 {
                    (close - values[i - 1][12]) / values[i - 1][12] * 100.0
                } else {
                    0.0
                },
                // [13-17] Intraday movements
                0.0,
                0.0,
                0.0,
                0.0,
                0.5,
                // [18-25] Moving averages (EMAs + SMAs)
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                // [26-32] MACD
                (t * 0.05).sin() * 0.5,
                (t * 0.05).sin() * 0.45,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                // [33-36] Technical (RSI, KDJ)
                50.0,
                50.0,
                50.0,
                50.0,
                // [37-41] Bollinger Bands
                0.0,
                0.0,
                0.0,
                0.04,
                0.5,
                // [42-47] Volatility
                0.5,
                0.02,
                0.02,
                (t * 0.1).sin() * 50.0,
                volume * (t + 1.0),
                1.0,
                // [48-51] Momentum
                0.0,
                0.0,
                0.0,
                0.5,
                // [52-54] Candlestick sizes
                0.0,
                0.0,
                0.0,
                // [55-62] Trend & strength
                0.0,
                25.0,
                0.0,
                0.0,
                50.0,
                -50.0,
                50.0,
                50.0,
                // [63-65] Lagged returns
                0.0,
                0.0,
                0.0,
                // [66-67] Gap features
                0.0,
                0.0,
                // [68-69] Volume features
                0.0,
                0.0,
                // [70-73] Price ROC
                0.0,
                0.0,
                0.0,
                0.02,
                // [74-77] Candlestick patterns
                0.0,
                0.0,
                0.0,
                0.0,
                // [78-80] Index CSI300 (percentage-based)
                0.0,
                0.0,
                0.0,
                // [81-83] Index Star50
                0.0,
                0.0,
                0.0,
                // [84-86] Index ChiNext
                0.0,
                0.0,
                0.0,
                // [87-90] Money flow
                0.0,
                0.0,
                0.0,
                0.0,
                // [91-93] Industry
                0.0,
                0.0,
                0.0,
                // [94-95] Volatility regime
                0.5,
                0.0,
                // [96-104] Reserved
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]);
        }
        // For dummy data, no next_day_return available — fill with None
        let nd_returns = vec![None; values.len()];
        Some(StockItem { values, last_trade_date: None, dataset_last_date: None, next_day_returns: nd_returns })
    }

    pub fn len(&self) -> usize {
        self.size
    }
}

/*
// Yahoo Finance API disabled - using database instead
#[allow(dead_code)]
pub struct RealStockDataset {
    ticker: String,
    quotes: Vec<Quote>,
    china_quotes: Option<Vec<Quote>>,
    seq_len: usize,
}

#[allow(dead_code)]
impl RealStockDataset {
    pub fn new(ticker: &str, quotes: Vec<Quote>, seq_len: usize) -> Self {
        Self {
            ticker: ticker.to_string(),
            quotes,
            china_quotes: None,
            seq_len,
        }
    }

    pub fn with_market_data(ticker: &str, quotes: Vec<Quote>, china_quotes: Vec<Quote>, seq_len: usize) -> Self {
        Self {
            ticker: ticker.to_string(),
            quotes,
            china_quotes: Some(china_quotes),
            seq_len,
        }
    }
}

impl Dataset<StockItem> for RealStockDataset {
    fn get(&self, index: usize) -> Option<StockItem> {
        if index + self.seq_len > self.quotes.len() {
            return None;
        }

        let mut values = Vec::with_capacity(self.seq_len);
        for i in 0..self.seq_len {
            let current_index = index + i;
            let quote = &self.quotes[current_index];

            // Extract weekday from timestamp
            let datetime = chrono::DateTime::from_timestamp(quote.timestamp as i64, 0)
                .map(|dt| dt.naive_utc())
                .unwrap_or_else(|| chrono::DateTime::from_timestamp(0, 0).unwrap().naive_utc());
            let weekday = datetime.weekday().num_days_from_monday() as f32; // 0-6
            let _industry = 0.0; // Placeholder for industry classification

            // Calculate technical indicators
            let china_quotes_slice = self.china_quotes.as_ref().map(|cq| cq.as_slice());
            let indicators = calculate_indicators(&self.quotes, current_index, None, china_quotes_slice);

            let week_no = ((datetime.iso_week().week() - 1) % 53) as f32;

            values.push([
                quote.open as f32,
                quote.high as f32,
                quote.low as f32,
                quote.close as f32,
                quote.volume as f32,
                weekday,
                week_no,
                quote.close as f32, // EMA5 placeholder
                quote.close as f32, // EMA10 placeholder
                quote.close as f32, // EMA20 placeholder
                quote.close as f32, // EMA30 placeholder
                quote.close as f32, // EMA60 placeholder
                indicators.daily_return,
                indicators.volume_ratio,
                indicators.macd_daily.0,
                indicators.macd_daily.1,
            ]);
        }
        Some(StockItem { values })
    }

    fn len(&self) -> usize {
        if self.quotes.len() < self.seq_len {
            0
        } else {
            self.quotes.len() - self.seq_len
        }
    }
}
*/

/// Database-backed stock dataset from ml_training_dataset table
#[allow(dead_code)]
pub struct DbStockDataset {
    records: Vec<MlTrainingRecord>,
    seq_len: usize,
}

#[allow(dead_code)]
impl DbStockDataset {
    pub fn new(records: Vec<MlTrainingRecord>, seq_len: usize) -> Self {
        Self { records, seq_len }
    }

    /// Group records by stock symbol for proper sequence creation
    /// This ensures sequences never cross stock boundaries and are sorted by date
    /// Parallelized with Rayon for performance
    pub fn from_records_grouped(records: Vec<MlTrainingRecord>, seq_len: usize) -> Vec<Self> {
        use rayon::prelude::*;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let mut grouped: HashMap<String, Vec<MlTrainingRecord>> = HashMap::new();

        println!("╔══════════════════════════════════════════════════════════╗");
        println!("║          Dataset Creation & Grouping Progress           ║");
        println!("╚══════════════════════════════════════════════════════════╝");
        println!();
        println!(
            "📊 Grouping {} total records by stock symbol...",
            records.len()
        );

        for record in records {
            grouped
                .entry(record.ts_code.clone())
                .or_insert_with(Vec::new)
                .push(record);
        }

        let num_stocks = grouped.len();
        println!("✓ Found {} unique stocks", num_stocks);
        println!();

        // Calculate stats about record distribution
        let mut record_counts: Vec<usize> = grouped.values().map(|v| v.len()).collect();
        record_counts.sort_unstable();
        let min_records = record_counts.first().copied().unwrap_or(0);
        let max_records = record_counts.last().copied().unwrap_or(0);
        let avg_records = if !record_counts.is_empty() {
            record_counts.iter().sum::<usize>() / record_counts.len()
        } else {
            0
        };

        println!("📈 Record distribution:");
        println!("   • Min records per stock: {}", min_records);
        println!("   • Max records per stock: {}", max_records);
        println!("   • Avg records per stock: {}", avg_records);
        println!();

        // Convert to Vec for parallel processing
        println!("🔄 Converting to vector for parallel processing...");
        let grouped_vec: Vec<(String, Vec<MlTrainingRecord>)> = grouped.into_iter().collect();
        println!("✓ Ready for parallel dataset creation");
        println!();

        // Parallel processing of stock datasets with progress tracking
        println!(
            "⚡ Creating datasets in parallel across {} stocks...",
            num_stocks
        );
        let processed_count = AtomicUsize::new(0);

        let datasets: Vec<Self> = grouped_vec
            .into_par_iter()
            .map(|(_ts_code, mut stock_records)| {
                // Sort by trade_date to ensure chronological order
                stock_records.sort_by(|a, b| a.trade_date.cmp(&b.trade_date));

                let dataset = Self::new(stock_records, seq_len);

                // Progress reporting every 100 stocks
                let count = processed_count.fetch_add(1, Ordering::Relaxed) + 1;
                if count % 100 == 0 || count == num_stocks {
                    println!(
                        "   ⏳ Processed {}/{} stocks ({:.1}%)",
                        count,
                        num_stocks,
                        (count as f64 / num_stocks as f64) * 100.0
                    );
                }

                dataset
            })
            .collect();

        let total_seqs: usize = datasets.iter().map(|d| d.len()).sum();
        println!();
        println!("✅ Dataset creation complete!");
        println!("   • {} datasets created", datasets.len());
        println!("   • {} total sequences generated", total_seqs);
        println!(
            "   • Avg sequences per stock: {:.1}",
            total_seqs as f64 / datasets.len() as f64
        );

        datasets
    }
}

impl DbStockDataset {
    pub fn get(&self, index: usize) -> Option<StockItem> {
        // Boundary check
        if index + self.seq_len > self.records.len() {
            return None;
        }

        // CRITICAL: Verify all records in this sequence are from the same stock
        let first_ts_code = &self.records[index].ts_code;
        for i in 0..self.seq_len {
            if &self.records[index + i].ts_code != first_ts_code {
                eprintln!(
                    "ERROR: Sequence at index {} crosses stock boundary! {} -> {}",
                    index,
                    first_ts_code,
                    &self.records[index + i].ts_code
                );
                return None;
            }
        }

        let mut values = Vec::with_capacity(self.seq_len);
        for i in 0..self.seq_len {
            let record = &self.records[index + i];

            // Map database fields to feature array [105 features]
            // Note: Categorical features will be encoded as placeholder indices here
            // They should be properly encoded during preprocessing
            values.push([
                // [0-2] Categorical (placeholder - will be encoded during training)
                0.0,
                0.0,
                0.0,
                // [3-4] Volume
                record.volume as f32,
                record.amount.unwrap_or(0.0) as f32,
                // [5-8] Temporal features
                record.month.unwrap_or(1) as f32,
                record.weekday.unwrap_or(0) as f32,
                record.quarter.unwrap_or(1) as f32,
                record.weekno.unwrap_or(0) as f32,
                // [9-12] Percentage-based OHLC
                record.open_pct.unwrap_or(0.0) as f32,
                record.high_pct.unwrap_or(0.0) as f32,
                record.low_pct.unwrap_or(0.0) as f32,
                record.close_pct.unwrap_or(0.0) as f32,
                // [13-17] Intraday movements
                record.high_from_open_pct.unwrap_or(0.0) as f32,
                record.low_from_open_pct.unwrap_or(0.0) as f32,
                record.close_from_open_pct.unwrap_or(0.0) as f32,
                record.intraday_range_pct.unwrap_or(0.0) as f32,
                record.close_position_in_range.unwrap_or(0.5) as f32,
                // [18-25] Moving averages (EMAs + SMAs)
                record.ema_5.unwrap_or(0.0) as f32,
                record.ema_10.unwrap_or(0.0) as f32,
                record.ema_20.unwrap_or(0.0) as f32,
                record.ema_30.unwrap_or(0.0) as f32,
                record.ema_60.unwrap_or(0.0) as f32,
                record.sma_5.unwrap_or(0.0) as f32,
                record.sma_10.unwrap_or(0.0) as f32,
                record.sma_20.unwrap_or(0.0) as f32,
                // [26-32] MACD
                record.macd_line.unwrap_or(0.0) as f32,
                record.macd_signal.unwrap_or(0.0) as f32,
                record.macd_histogram.unwrap_or(0.0) as f32,
                record.macd_weekly_line.unwrap_or(0.0) as f32,
                record.macd_weekly_signal.unwrap_or(0.0) as f32,
                record.macd_monthly_line.unwrap_or(0.0) as f32,
                record.macd_monthly_signal.unwrap_or(0.0) as f32,
                // [33-36] Technical indicators
                record.rsi_14.unwrap_or(50.0) as f32,
                record.kdj_k.unwrap_or(50.0) as f32,
                record.kdj_d.unwrap_or(50.0) as f32,
                record.kdj_j.unwrap_or(50.0) as f32,
                // [37-41] Bollinger Bands
                record.bb_upper.unwrap_or(0.0) as f32,
                record.bb_middle.unwrap_or(0.0) as f32,
                record.bb_lower.unwrap_or(0.0) as f32,
                record.bb_bandwidth.unwrap_or(0.0) as f32,
                record.bb_percent_b.unwrap_or(0.5) as f32,
                // [42-47] Volatility
                record.atr.unwrap_or(0.0) as f32,
                record.volatility_5.unwrap_or(0.0) as f32,
                record.volatility_20.unwrap_or(0.0) as f32,
                record.asi.unwrap_or(0.0) as f32,
                record.obv.unwrap_or(0.0) as f32,
                record.volume_ratio.unwrap_or(1.0) as f32,
                // [48-51] Momentum
                record.price_momentum_5.unwrap_or(0.0) as f32,
                record.price_momentum_10.unwrap_or(0.0) as f32,
                record.price_momentum_20.unwrap_or(0.0) as f32,
                record.price_position_52w.unwrap_or(0.5) as f32,
                // [52-54] Candlestick sizes
                record.body_size.unwrap_or(0.0) as f32,
                record.upper_shadow.unwrap_or(0.0) as f32,
                record.lower_shadow.unwrap_or(0.0) as f32,
                // [55-62] Trend & strength
                record.trend_strength.unwrap_or(0.0) as f32,
                record.adx_14.unwrap_or(25.0) as f32,
                record.vwap_distance_pct.unwrap_or(0.0) as f32,
                record.cmf_20.unwrap_or(0.0) as f32,
                50.0 as f32,
                record.williams_r_14.unwrap_or(-50.0) as f32,
                record.aroon_up_25.unwrap_or(50.0) as f32,
                record.aroon_down_25.unwrap_or(50.0) as f32,
                // [63-65] Lagged returns
                record.return_lag_1.unwrap_or(0.0) as f32,
                record.return_lag_2.unwrap_or(0.0) as f32,
                record.return_lag_3.unwrap_or(0.0) as f32,
                // [66-67] Gap features
                record.overnight_gap.unwrap_or(0.0) as f32,
                record.gap_pct.unwrap_or(0.0) as f32,
                // [68-69] Volume features
                record.volume_roc_5.unwrap_or(0.0) as f32,
                if record.volume_spike.unwrap_or(false) {
                    1.0
                } else {
                    0.0
                },
                // [70-73] Price ROC & volatility
                record.price_roc_5.unwrap_or(0.0) as f32,
                record.price_roc_10.unwrap_or(0.0) as f32,
                record.price_roc_20.unwrap_or(0.0) as f32,
                record.hist_volatility_20.unwrap_or(0.0) as f32,
                // [74-77] Candlestick patterns
                if record.is_doji.unwrap_or(false) {
                    1.0
                } else {
                    0.0
                },
                if record.is_hammer.unwrap_or(false) {
                    1.0
                } else {
                    0.0
                },
                if record.is_shooting_star.unwrap_or(false) {
                    1.0
                } else {
                    0.0
                },
                record.consecutive_days.unwrap_or(0) as f32,
                // [78-80] Index CSI300 (percentage-based)
                record.index_csi300_pct_chg.unwrap_or(0.0) as f32,
                record.index_csi300_vs_ma5_pct.unwrap_or(0.0) as f32,
                record.index_csi300_vs_ma20_pct.unwrap_or(0.0) as f32,
                // [81-83] Index XIN9
                record.index_xin9_pct_chg.unwrap_or(0.0) as f32,
                record.index_xin9_vs_ma5_pct.unwrap_or(0.0) as f32,
                record.index_xin9_vs_ma20_pct.unwrap_or(0.0) as f32,
                // [84-86] Index ChiNext
                record.index_chinext_pct_chg.unwrap_or(0.0) as f32,
                record.index_chinext_vs_ma5_pct.unwrap_or(0.0) as f32,
                record.index_chinext_vs_ma20_pct.unwrap_or(0.0) as f32,
                // [87-90] Money flow
                record.net_mf_vol.unwrap_or(0.0) as f32,
                record.net_mf_amount.unwrap_or(0.0) as f32,
                record.smart_money_ratio.unwrap_or(0.0) as f32,
                record.large_order_flow.unwrap_or(0.0) as f32,
                // [91-93] Industry features
                record.industry_avg_return.unwrap_or(0.0) as f32,
                record.stock_vs_industry.unwrap_or(0.0) as f32,
                record.industry_momentum_5d.unwrap_or(0.0) as f32,
                // [94-95] Volatility regime
                record.vol_percentile.unwrap_or(0.5) as f32,
                if record.high_vol_regime.unwrap_or(0) != 0 {
                    1.0
                } else {
                    0.0
                },
                // [96-104] Reserved for future use
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]);
        }
        // Build next_day_returns aligned with values: for each position i in values,
        // next_day_returns[i] corresponds to the `next_day_return` field of the record at index + i
        let mut nd_returns: Vec<Option<f32>> = Vec::with_capacity(self.seq_len);
        for i in 0..self.seq_len {
            let r = self.records[index + i].next_day_return.map(|v| v as f32);
            nd_returns.push(r);
        }

        Some(StockItem { values, last_trade_date: None, dataset_last_date: None, next_day_returns: nd_returns })
    }

    pub fn len(&self) -> usize {
        if self.records.len() < self.seq_len {
            0
        } else {
            self.records.len() - self.seq_len
        }
    }
}
