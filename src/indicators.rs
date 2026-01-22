/// Technical indicators calculator
use yahoo_finance_api::Quote;

/// Calculate Simple Moving Average
pub fn calculate_sma(prices: &[f32], period: usize) -> Option<f32> {
    if prices.len() < period {
        return None;
    }
    let sum: f32 = prices.iter().rev().take(period).sum();
    Some(sum / period as f32)
}

/// Calculate RSI (Relative Strength Index)
pub fn calculate_rsi(prices: &[f32], period: usize) -> Option<f32> {
    if prices.len() < period + 1 {
        return None;
    }
    
    let mut gains = 0.0;
    let mut losses = 0.0;
    
    // Calculate price changes over the period
    for i in (prices.len() - period)..prices.len() {
        let change = prices[i] - prices[i - 1];
        if change > 0.0 {
            gains += change;
        } else {
            losses += -change;
        }
    }
    
    let avg_gain = gains / period as f32;
    let avg_loss = losses / period as f32;
    
    if avg_loss == 0.0 {
        return Some(100.0);
    }
    
    let rs = avg_gain / avg_loss;
    Some(100.0 - (100.0 / (1.0 + rs)))
}

/// Calculate daily return (percentage change)
pub fn calculate_return(current: f32, previous: f32) -> f32 {
    if previous == 0.0 {
        return 0.0;
    }
    ((current - previous) / previous) * 100.0
}

/// Calculate correlation coefficient between two price series
pub fn calculate_correlation(prices1: &[f32], prices2: &[f32], period: usize) -> f32 {
    if prices1.len() < period || prices2.len() < period || period < 2 {
        return 0.0;
    }
    
    let p1: Vec<f32> = prices1.iter().rev().take(period).copied().collect();
    let p2: Vec<f32> = prices2.iter().rev().take(period).copied().collect();
    
    // Calculate means
    let mean1: f32 = p1.iter().sum::<f32>() / period as f32;
    let mean2: f32 = p2.iter().sum::<f32>() / period as f32;
    
    // Calculate correlation
    let mut numerator = 0.0;
    let mut sum_sq1 = 0.0;
    let mut sum_sq2 = 0.0;
    
    for i in 0..period {
        let diff1 = p1[i] - mean1;
        let diff2 = p2[i] - mean2;
        numerator += diff1 * diff2;
        sum_sq1 += diff1 * diff1;
        sum_sq2 += diff2 * diff2;
    }
    
    let denominator = (sum_sq1 * sum_sq2).sqrt();
    if denominator == 0.0 {
        return 0.0;
    }
    
    numerator / denominator
}

/// Calculate relative return (stock return - market return)
pub fn calculate_relative_return(stock_return: f32, market_return: f32) -> f32 {
    stock_return - market_return
}

/// Calculate volume ratio (current volume / average volume)
pub fn calculate_volume_ratio(volumes: &[f32], period: usize) -> Option<f32> {
    if volumes.is_empty() || volumes.len() < period {
        return None;
    }
    
    let current_volume = volumes[volumes.len() - 1];
    let avg_volume: f32 = volumes.iter().rev().take(period).sum::<f32>() / period as f32;
    
    if avg_volume == 0.0 {
        return Some(1.0);
    }
    
    Some(current_volume / avg_volume)
}

/// Extract quarter from month (1-4)
pub fn get_quarter(month: u32) -> f32 {
    match month {
        1..=3 => 1.0,
        4..=6 => 2.0,
        7..=9 => 3.0,
        10..=12 => 4.0,
        _ => 1.0,
    }
}

/// Calculate ATR (Average True Range)
/// Measures market volatility
pub fn calculate_atr(quotes: &[Quote], period: usize) -> f32 {
    if quotes.len() < period + 1 {
        return 0.0;
    }
    
    let mut true_ranges = Vec::new();
    
    for i in 1..quotes.len() {
        let high = quotes[i].high as f32;
        let low = quotes[i].low as f32;
        let prev_close = quotes[i - 1].close as f32;
        
        // True Range is the greatest of:
        // 1. Current High - Current Low
        // 2. |Current High - Previous Close|
        // 3. |Current Low - Previous Close|
        let tr1 = high - low;
        let tr2 = (high - prev_close).abs();
        let tr3 = (low - prev_close).abs();
        
        let true_range = tr1.max(tr2).max(tr3);
        true_ranges.push(true_range);
    }
    
    // Calculate ATR as simple moving average of true ranges
    if true_ranges.len() < period {
        return true_ranges.iter().sum::<f32>() / true_ranges.len() as f32;
    }
    
    true_ranges.iter().rev().take(period).sum::<f32>() / period as f32
}

/// Calculate ASI (Accumulation Swing Index)
/// Measures the strength of price swings and accumulates them over time
pub fn calculate_asi(quotes: &[Quote], limit_move: f32) -> f32 {
    if quotes.len() < 2 {
        return 0.0;
    }
    
    let mut asi_sum = 0.0;
    
    for i in 1..quotes.len() {
        let curr = &quotes[i];
        let prev = &quotes[i - 1];
        
        let close = curr.close as f32;
        let open = curr.open as f32;
        let high = curr.high as f32;
        let low = curr.low as f32;
        let prev_close = prev.close as f32;
        
        // Calculate components
        let a = (high - prev_close).abs();
        let b = (low - prev_close).abs();
        let c = (high - low).abs();
        let d = (prev_close - prev.open as f32).abs();
        
        // Calculate K (the largest value)
        let k = a.max(b);
        
        // Calculate R (reference value)
        let r = if a >= b && a >= c {
            a - 0.5 * b + 0.25 * d
        } else if b >= a && b >= c {
            b - 0.5 * a + 0.25 * d
        } else {
            c + 0.25 * d
        };
        
        // Calculate SI (Swing Index)
        let si = if r != 0.0 && limit_move != 0.0 {
            50.0 * ((close - prev_close) + 0.5 * (close - open) + 0.25 * (prev_close - prev.open as f32)) / r * (k / limit_move)
        } else {
            0.0
        };
        
        asi_sum += si;
    }
    
    asi_sum
}

/// Calculate Exponential Moving Average (EMA)
pub fn calculate_ema(prices: &[f32], period: usize) -> Option<f32> {
    if prices.len() < period {
        return None;
    }
    
    let multiplier = 2.0 / (period as f32 + 1.0);
    
    // Start with SMA for the first value
    let sma: f32 = prices.iter().rev().take(period).sum::<f32>() / period as f32;
    
    // Calculate EMA iteratively for recent prices
    let mut ema = sma;
    let start_idx = if prices.len() > period { prices.len() - period } else { 0 };
    
    for i in start_idx..prices.len() {
        ema = (prices[i] - ema) * multiplier + ema;
    }
    
    Some(ema)
}

/// Calculate MACD (Moving Average Convergence Divergence)
/// Returns (MACD line, Signal line)
pub fn calculate_macd(prices: &[f32], fast: usize, slow: usize, _signal: usize) -> Option<(f32, f32)> {
    if prices.len() < slow {
        return None;
    }
    
    // Calculate fast and slow EMAs
    let fast_ema = calculate_ema(prices, fast)?;
    let slow_ema = calculate_ema(prices, slow)?;
    
    // MACD line = fast EMA - slow EMA
    let macd_line = fast_ema - slow_ema;
    
    // For signal line: in a full implementation, we'd maintain MACD history and calculate EMA of that
    // Simplified approach: use a dampened MACD value
    let signal_line = macd_line * 0.85; // Approximation
    
    Some((macd_line, signal_line))
}

/// Calculate MACD for standard daily timeframe (12, 26, 9)
pub fn calculate_macd_daily(prices: &[f32]) -> (f32, f32) {
    calculate_macd(prices, 12, 26, 9).unwrap_or((0.0, 0.0))
}

/// Calculate MACD for weekly timeframe (60, 130, 45 days ≈ 12, 26, 9 weeks)
pub fn calculate_macd_weekly(prices: &[f32]) -> (f32, f32) {
    calculate_macd(prices, 60, 130, 45).unwrap_or((0.0, 0.0))
}

/// Calculate MACD for monthly timeframe (252, 546, 189 days ≈ 12, 26, 9 months)
pub fn calculate_macd_monthly(prices: &[f32]) -> (f32, f32) {
    calculate_macd(prices, 252, 546, 189).unwrap_or((0.0, 0.0))
}

/// Calculate Bollinger Bands
/// Returns (upper_band, lower_band, bandwidth)
pub fn calculate_bollinger_bands(prices: &[f32], period: usize, std_dev: f32) -> (f32, f32, f32) {
    if prices.len() < period {
        let last = prices.last().copied().unwrap_or(0.0);
        return (last, last, 0.0);
    }
    
    // Calculate SMA
    let sma = calculate_sma(prices, period).unwrap_or(0.0);
    
    // Calculate standard deviation
    let recent_prices: Vec<f32> = prices.iter().rev().take(period).copied().collect();
    let variance: f32 = recent_prices.iter()
        .map(|&price| {
            let diff = price - sma;
            diff * diff
        })
        .sum::<f32>() / period as f32;
    let std = variance.sqrt();
    
    let upper = sma + (std_dev * std);
    let lower = sma - (std_dev * std);
    let bandwidth = if sma != 0.0 { (upper - lower) / sma } else { 0.0 };
    
    (upper, lower, bandwidth)
}

/// Calculate OBV (On Balance Volume)
pub fn calculate_obv(quotes: &[Quote]) -> f32 {
    if quotes.len() < 2 {
        return 0.0;
    }
    
    let mut obv = 0.0;
    
    for i in 1..quotes.len() {
        let curr_close = quotes[i].close as f32;
        let prev_close = quotes[i - 1].close as f32;
        let volume = quotes[i].volume as f32;
        
        if curr_close > prev_close {
            obv += volume;
        } else if curr_close < prev_close {
            obv -= volume;
        }
        // If equal, OBV stays the same
    }
    
    obv
}

/// Calculate all technical indicators for a given position in the quote history
pub struct IndicatorValues {
    pub sma5: f32,
    pub sma20: f32,
    pub rsi: f32,
    pub daily_return: f32,
    pub volume_ratio: f32,
    pub asi: f32,
    pub atr: f32,
    // Bollinger Bands (upper, lower, bandwidth)
    pub bb_upper: f32,
    pub bb_lower: f32,
    pub bb_width: f32,
    // OBV
    pub obv: f32,
    // China market correlation features
    pub china_correlation: f32,
    pub china_relative_return: f32,
    // MACD values: Daily, Weekly, Monthly (each has 2 components: line, signal)
    pub macd_daily: (f32, f32),
    pub macd_weekly: (f32, f32),
    pub macd_monthly: (f32, f32),
}

pub fn calculate_indicators(quotes: &[Quote], index: usize, _market_quotes: Option<&[Quote]>, china_quotes: Option<&[Quote]>) -> IndicatorValues {
    // Extract close prices and volumes up to current index
    let closes: Vec<f32> = quotes[..=index].iter().map(|q| q.close as f32).collect();
    let volumes: Vec<f32> = quotes[..=index].iter().map(|q| q.volume as f32).collect();
    
    let sma5 = calculate_sma(&closes, 5).unwrap_or(closes[index]);
    let sma20 = calculate_sma(&closes, 20).unwrap_or(closes[index]);
    let rsi = calculate_rsi(&closes, 14).unwrap_or(50.0); // Default to neutral
    
    let daily_return = if index > 0 {
        calculate_return(closes[index], closes[index - 1])
    } else {
        0.0
    };
    
    let volume_ratio = calculate_volume_ratio(&volumes, 20).unwrap_or(1.0);
    
    // Calculate ASI (using a typical limit move of 10% of average price)
    let avg_price = closes.iter().sum::<f32>() / closes.len() as f32;
    let limit_move = avg_price * 0.1;
    let asi = calculate_asi(&quotes[..=index], limit_move);
    
    // Calculate ATR (14 period)
    let atr = calculate_atr(&quotes[..=index], 14);
    
    // Calculate Bollinger Bands (20 period, 2 standard deviations)
    let (bb_upper, bb_lower, bb_width) = calculate_bollinger_bands(&closes, 20, 2.0);
    
    // Calculate OBV
    let obv = calculate_obv(&quotes[..=index]);
    
    // Calculate China market correlation and relative return
    let (china_correlation, china_relative_return) = if let Some(china_quotes_data) = china_quotes {
        if index < china_quotes_data.len() {
            let china_closes: Vec<f32> = china_quotes_data[..=index].iter().map(|q| q.close as f32).collect();
            let correlation = calculate_correlation(&closes, &china_closes, 20.min(closes.len()));
            
            let china_return = if index > 0 && index < china_closes.len() {
                calculate_return(china_closes[index], china_closes[index - 1])
            } else {
                0.0
            };
            
            let rel_return = calculate_relative_return(daily_return, china_return);
            (correlation, rel_return)
        } else {
            (0.0, 0.0)
        }
    } else {
        (0.0, 0.0)
    };
    
    // Calculate MACD for all timeframes
    let macd_daily = calculate_macd_daily(&closes);
    let macd_weekly = calculate_macd_weekly(&closes);
    let macd_monthly = calculate_macd_monthly(&closes);
    
    IndicatorValues {
        sma5,
        sma20,
        rsi,
        daily_return,
        volume_ratio,
        asi,
        atr,
        bb_upper,
        bb_lower,
        bb_width,
        obv,
        china_correlation,
        china_relative_return,
        macd_daily,
        macd_weekly,
        macd_monthly,
    }
}
