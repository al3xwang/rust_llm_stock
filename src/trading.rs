/// Trading types and structures for daily predictions and signal generation
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StockPrediction {
    pub ts_code: String,
    pub trade_date: String,
    pub pred_return: f32,            // Expected % return (next day)
    pub direction: i32,              // 1=up, -1=down, 0=neutral
    pub confidence: f32,             // 0-1, model confidence
    pub reversal_score: Option<f32>, // From SQL analysis, 0-100
    pub rsi_14: Option<f32>,         // Supporting data
    pub close_pct: Option<f32>,      // Current day close % change
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TradeAction {
    Buy,
    Sell,
    Hold,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    pub ts_code: String,
    pub trade_date: String,
    pub action: TradeAction,
    pub predicted_return: f32,
    pub confidence: f32,
    pub reversal_score: f32,
    pub signal_strength: f32,         // Composite score 0-1
    pub reason: String,               // Why this signal
    pub suggested_qty: Option<f32>,   // Number of shares (if implemented)
    pub suggested_price: Option<f32>, // Suggested entry price
}

#[derive(Debug, Clone)]
pub struct SignalParams {
    pub min_confidence: f32,       // e.g., 0.6
    pub min_reversal_score: f32,   // e.g., 60.0
    pub max_vol_percentile: f32,   // e.g., 80.0 (skip high vol)
    pub max_pe_ratio: Option<f32>, // e.g., 50.0
    pub min_close_pct: f32,        // e.g., -2.0 (reject strong up moves)
    pub max_close_pct: f32,        // e.g., 3.0 (reject strong down moves)
    pub top_n_stocks: usize,       // e.g., 20
    pub require_reversal: bool,    // Must have reversal indicators
}

impl Default for SignalParams {
    fn default() -> Self {
        Self {
            min_confidence: 0.55,
            min_reversal_score: 50.0,
            max_vol_percentile: 75.0,
            max_pe_ratio: Some(60.0),
            min_close_pct: -3.0,
            max_close_pct: 3.0,
            top_n_stocks: 15,
            require_reversal: true,
        }
    }
}
