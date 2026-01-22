use std::fmt;

use ta::errors::{ Result, TaError };
use ta::{ Close, Next, Reset };
#[cfg(feature = "serde")]
use serde::{ Deserialize, Serialize };

/// MACD (Moving Average Convergence Divergence)
///
/// # Formula
///
/// MACD Line = EMA(12) - EMA(26)
/// Signal Line = EMA(9) of MACD Line
/// Histogram = MACD Line - Signal Line
///
/// Where:
///
/// * _EMA_ - Exponential Moving Average
/// * _MACD Line_ - difference between fast and slow EMA
/// * _Signal Line_ - EMA of MACD Line
/// * _Histogram_ - difference between MACD Line and Signal Line
///
/// # Parameters
///
/// * _fast_period_ - period for fast EMA (typically 12)
/// * _slow_period_ - period for slow EMA (typically 26)
/// * _signal_period_ - period for signal line EMA (typically 9)
///
/// # Example
///
/// ```
/// use stock_rust::macd::MACDIndicator;
/// use ta::Next;
///
/// let mut macd = MACDIndicator::new(12, 26, 9).unwrap();
/// let output = macd.next(10.0);
/// println!("MACD: {}, Signal: {}, Histogram: {}", output.macd, output.signal, output.histogram);
/// ```
///
/// # Links
///
/// * [MACD, Investopedia](https://www.investopedia.com/terms/m/macd.asp)
/// * [MACD, Wikipedia](https://en.wikipedia.org/wiki/MACD)
///
#[doc(alias = "MACD")]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct MACDIndicator {
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
    fast_ema: f64,
    slow_ema: f64,
    signal_ema: f64,
    fast_multiplier: f64,
    slow_multiplier: f64,
    signal_multiplier: f64,
    count: usize,
    initialized: bool,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct MacdOutput {
    pub macd: f64,
    pub signal: f64,
    pub histogram: f64,
}

impl MACDIndicator {
    pub fn new(fast_period: usize, slow_period: usize, signal_period: usize) -> Result<Self> {
        if fast_period == 0 || slow_period == 0 || signal_period == 0 {
            return Err(TaError::InvalidParameter);
        }
        if fast_period >= slow_period {
            return Err(TaError::InvalidParameter);
        }

        Ok(Self {
            fast_period,
            slow_period,
            signal_period,
            fast_ema: 0.0,
            slow_ema: 0.0,
            signal_ema: 0.0,
            fast_multiplier: 2.0 / ((fast_period + 1) as f64),
            slow_multiplier: 2.0 / ((slow_period + 1) as f64),
            signal_multiplier: 2.0 / ((signal_period + 1) as f64),
            count: 0,
            initialized: false,
        })
    }

    fn calculate_ema(&self, value: f64, prev_ema: f64, multiplier: f64) -> f64 {
        (value - prev_ema) * multiplier + prev_ema
    }
}

impl Next<f64> for MACDIndicator {
    type Output = MacdOutput;

    fn next(&mut self, input: f64) -> Self::Output {
        self.count += 1;

        if !self.initialized {
            self.fast_ema = input;
            self.slow_ema = input;
            self.initialized = true;
        } else {
            self.fast_ema = self.calculate_ema(input, self.fast_ema, self.fast_multiplier);
            self.slow_ema = self.calculate_ema(input, self.slow_ema, self.slow_multiplier);
        }

        let macd_line = self.fast_ema - self.slow_ema;

        if self.count == 1 {
            self.signal_ema = macd_line;
        } else {
            self.signal_ema = self.calculate_ema(macd_line, self.signal_ema, self.signal_multiplier);
        }

        let histogram = macd_line - self.signal_ema;

        MacdOutput {
            macd: macd_line,
            signal: self.signal_ema,
            histogram,
        }
    }
}

impl<T: Close> Next<&T> for MACDIndicator {
    type Output = MacdOutput;

    fn next(&mut self, input: &T) -> Self::Output {
        self.next(input.close())
    }
}

impl Reset for MACDIndicator {
    fn reset(&mut self) {
        self.fast_ema = 0.0;
        self.slow_ema = 0.0;
        self.signal_ema = 0.0;
        self.count = 0;
        self.initialized = false;
    }
}

impl Default for MACDIndicator {
    fn default() -> Self {
        Self::new(12, 26, 9).unwrap()
    }
}

impl fmt::Display for MACDIndicator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "MACD({}, {}, {})",
            self.fast_period,
            self.slow_period,
            self.signal_period
        )
    }
}

impl fmt::Display for MacdOutput {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "MACD: {:.4}, Signal: {:.4}, Histogram: {:.4}",
            self.macd,
            self.signal,
            self.histogram
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ta::Next;

    #[derive(Debug)]
    struct TestBar {
        close: f64,
    }

    impl Close for TestBar {
        fn close(&self) -> f64 {
            self.close
        }
    }

    #[test]
    fn test_new() {
        assert!(MACDIndicator::new(12, 26, 9).is_ok());
        assert!(MACDIndicator::new(0, 26, 9).is_err());
        assert!(MACDIndicator::new(12, 0, 9).is_err());
        assert!(MACDIndicator::new(12, 26, 0).is_err());
        assert!(MACDIndicator::new(26, 12, 9).is_err()); // fast >= slow
    }

    #[test]
    fn test_new_custom_periods() {
        // Test various valid period combinations
        assert!(MACDIndicator::new(5, 10, 5).is_ok());
        assert!(MACDIndicator::new(8, 17, 9).is_ok());
        assert!(MACDIndicator::new(1, 2, 1).is_ok());
        
        // Test invalid combinations
        assert!(MACDIndicator::new(10, 10, 5).is_err()); // fast == slow
        assert!(MACDIndicator::new(15, 10, 5).is_err()); // fast > slow
    }

    #[test]
    fn test_next() {
        let mut macd = MACDIndicator::new(12, 26, 9).unwrap();

        let test_data = vec![
            10.0, 11.0, 12.0, 11.5, 13.0, 14.0, 13.5, 15.0, 16.0, 15.5,
            17.0, 18.0, 17.5, 19.0, 20.0, 19.5, 21.0, 22.0, 21.5, 23.0,
            24.0, 23.5, 25.0, 26.0, 25.5, 27.0, 28.0,
        ];

        for price in test_data {
            let output = macd.next(price);
            assert!(output.macd.is_finite());
            assert!(output.signal.is_finite());
            assert!(output.histogram.is_finite());
        }
    }

    #[test]
    fn test_next_with_struct() {
        let mut macd = MACDIndicator::new(12, 26, 9).unwrap();
        
        let bars = vec![
            TestBar { close: 100.0 },
            TestBar { close: 101.0 },
            TestBar { close: 102.0 },
        ];

        for bar in &bars {
            let output = macd.next(bar);
            assert!(output.macd.is_finite());
        }
    }

    #[test]
    fn test_reset() {
        let mut macd = MACDIndicator::new(12, 26, 9).unwrap();

        let _ = macd.next(10.0);
        let _ = macd.next(11.0);

        macd.reset();

        assert_eq!(macd.count, 0);
        assert_eq!(macd.fast_ema, 0.0);
        assert_eq!(macd.slow_ema, 0.0);
        assert_eq!(macd.signal_ema, 0.0);
        assert!(!macd.initialized);
    }

    #[test]
    fn test_reset_produces_same_results() {
        let mut macd1 = MACDIndicator::new(12, 26, 9).unwrap();
        let mut macd2 = MACDIndicator::new(12, 26, 9).unwrap();

        let prices = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        
        // First run
        let mut results1 = vec![];
        for &price in &prices {
            results1.push(macd1.next(price));
        }

        // Reset and second run
        macd1.reset();
        let mut results2 = vec![];
        for &price in &prices {
            results2.push(macd1.next(price));
        }

        // Fresh indicator
        let mut results3 = vec![];
        for &price in &prices {
            results3.push(macd2.next(price));
        }

        // Results after reset should match fresh indicator
        for i in 0..prices.len() {
            assert_eq!(results2[i].macd, results3[i].macd);
            assert_eq!(results2[i].signal, results3[i].signal);
            assert_eq!(results2[i].histogram, results3[i].histogram);
        }
    }

    #[test]
    fn test_display() {
        let macd = MACDIndicator::new(12, 26, 9).unwrap();
        assert_eq!(format!("{}", macd), "MACD(12, 26, 9)");
    }

    #[test]
    fn test_output_display() {
        let output = MacdOutput {
            macd: 1.2345,
            signal: 0.9876,
            histogram: 0.2469,
        };
        let display = format!("{}", output);
        assert!(display.contains("MACD: 1.2345"));
        assert!(display.contains("Signal: 0.9876"));
        assert!(display.contains("Histogram: 0.2469"));
    }

    #[test]
    fn test_default() {
        let macd = MACDIndicator::default();
        assert_eq!(format!("{}", macd), "MACD(12, 26, 9)");
    }

    #[test]
    fn test_histogram() {
        let mut macd = MACDIndicator::new(12, 26, 9).unwrap();

        let prices = vec![22.0, 22.5, 23.0, 23.5, 24.0];
        let mut last_output = None;

        for price in prices {
            last_output = Some(macd.next(price));
        }

        if let Some(output) = last_output {
            assert!((output.histogram - (output.macd - output.signal)).abs() < 1e-10);
        }
    }

    #[test]
    fn test_histogram_consistency() {
        let mut macd = MACDIndicator::new(12, 26, 9).unwrap();

        let prices = vec![
            10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
            20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
        ];

        for price in prices {
            let output = macd.next(price);
            // Histogram should always equal MACD line - Signal line
            let calculated_histogram = output.macd - output.signal;
            assert!((output.histogram - calculated_histogram).abs() < 1e-10,
                "Histogram mismatch: {} != {}", output.histogram, calculated_histogram);
        }
    }

    #[test]
    fn test_bullish_crossover() {
        let mut macd = MACDIndicator::new(5, 10, 5).unwrap();

        // Simulate downtrend then reversal
        let prices = vec![
            100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0, 92.0, 91.0,
            // Reversal starts
            92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0,
        ];

        let mut prev_histogram = 0.0;
        let mut crossover_detected = false;

        for price in prices {
            let output = macd.next(price);
            
            // Bullish crossover: histogram crosses from negative to positive
            if prev_histogram < 0.0 && output.histogram > 0.0 {
                crossover_detected = true;
            }
            
            prev_histogram = output.histogram;
        }

        assert!(crossover_detected, "Should detect bullish crossover in reversal");
    }

    #[test]
    fn test_bearish_crossover() {
        let mut macd = MACDIndicator::new(5, 10, 5).unwrap();

        // Simulate uptrend then reversal
        let prices = vec![
            100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0,
            // Reversal starts
            108.0, 107.0, 106.0, 105.0, 104.0, 103.0, 102.0, 101.0, 100.0, 99.0,
        ];

        let mut prev_histogram = 0.0;
        let mut crossover_detected = false;

        for price in prices {
            let output = macd.next(price);
            
            // Bearish crossover: histogram crosses from positive to negative
            if prev_histogram > 0.0 && output.histogram < 0.0 {
                crossover_detected = true;
            }
            
            prev_histogram = output.histogram;
        }

        assert!(crossover_detected, "Should detect bearish crossover in reversal");
    }

    #[test]
    fn test_uptrend_positive_macd() {
        let mut macd = MACDIndicator::new(5, 10, 5).unwrap();

        // Strong uptrend
        let mut prices = vec![];
        for i in 0..30 {
            prices.push(100.0 + (i as f64) * 0.5);
        }

        let mut last_output = None;
        for price in prices {
            last_output = Some(macd.next(price));
        }

        // In a strong uptrend, MACD line should eventually be positive
        if let Some(output) = last_output {
            assert!(output.macd > 0.0, "MACD should be positive in uptrend");
        }
    }

    #[test]
    fn test_downtrend_negative_macd() {
        let mut macd = MACDIndicator::new(5, 10, 5).unwrap();

        // Strong downtrend
        let mut prices = vec![];
        for i in 0..30 {
            prices.push(100.0 - (i as f64) * 0.5);
        }

        let mut last_output = None;
        for price in prices {
            last_output = Some(macd.next(price));
        }

        // In a strong downtrend, MACD line should eventually be negative
        if let Some(output) = last_output {
            assert!(output.macd < 0.0, "MACD should be negative in downtrend");
        }
    }

    #[test]
    fn test_flat_market() {
        let mut macd = MACDIndicator::new(12, 26, 9).unwrap();

        // Flat market - same price
        let prices = vec![100.0; 50];

        let mut outputs = vec![];
        for price in prices {
            outputs.push(macd.next(price));
        }

        // In flat market, MACD should converge to zero
        let last_output = outputs.last().unwrap();
        assert!(last_output.macd.abs() < 1e-10, "MACD should be near 0 in flat market");
        assert!(last_output.signal.abs() < 1e-10, "Signal should be near 0 in flat market");
        assert!(last_output.histogram.abs() < 1e-10, "Histogram should be near 0 in flat market");
    }

    #[test]
    fn test_first_value_initialization() {
        let mut macd = MACDIndicator::new(12, 26, 9).unwrap();

        let first_output = macd.next(100.0);
        
        // First value: both EMAs initialize to first price, so MACD = 0
        assert_eq!(first_output.macd, 0.0);
        assert_eq!(first_output.signal, 0.0);
        assert_eq!(first_output.histogram, 0.0);
    }

    #[test]
    fn test_macd_output_clone() {
        let output1 = MacdOutput {
            macd: 1.5,
            signal: 1.2,
            histogram: 0.3,
        };
        let output2 = output1.clone();
        
        assert_eq!(output1, output2);
    }

    #[test]
    fn test_macd_indicator_clone() {
        let macd1 = MACDIndicator::new(12, 26, 9).unwrap();
        let macd2 = macd1.clone();
        
        assert_eq!(macd1.fast_period, macd2.fast_period);
        assert_eq!(macd1.slow_period, macd2.slow_period);
        assert_eq!(macd1.signal_period, macd2.signal_period);
    }

    #[test]
    fn test_signal_follows_macd() {
        let mut macd = MACDIndicator::new(5, 10, 3).unwrap();

        // Generate uptrend
        let prices = vec![
            100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0,
        ];

        let mut outputs = vec![];
        for price in prices {
            outputs.push(macd.next(price));
        }

        // Signal line should lag behind MACD line
        // In uptrend, MACD > Signal most of the time
        let last_output = outputs.last().unwrap();
        assert!(last_output.macd > last_output.signal, 
            "In uptrend, MACD should be above signal line");
    }

    #[test]
    fn test_different_period_configurations() {
        let prices = vec![
            100.0, 101.0, 102.0, 103.0, 102.0, 101.0, 100.0, 99.0, 100.0, 101.0,
        ];

        let mut macd_fast = MACDIndicator::new(5, 10, 5).unwrap();
        let mut macd_slow = MACDIndicator::new(12, 26, 9).unwrap();

        let mut fast_outputs = vec![];
        let mut slow_outputs = vec![];

        for &price in &prices {
            fast_outputs.push(macd_fast.next(price));
            slow_outputs.push(macd_slow.next(price));
        }

        // Fast MACD should be more responsive (larger absolute values)
        let fast_last = fast_outputs.last().unwrap();
        let slow_last = slow_outputs.last().unwrap();
        
        // Values should be different
        assert_ne!(fast_last.macd, slow_last.macd);
    }

    #[test]
    fn test_macd_line_calculation() {
        let mut macd = MACDIndicator::new(3, 5, 3).unwrap();

        // Simple test data
        let prices = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0];

        for price in prices {
            let output = macd.next(price);
            
            // MACD line should be the difference between fast and slow EMA
            // This is implicitly tested by the calculation, but we verify it's a valid number
            assert!(output.macd.is_finite());
            assert!(!output.macd.is_nan());
        }
    }
}
