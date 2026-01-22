use std::fmt;

use ta::errors::Result;
use ta::{Close, Next, Period, Reset};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Bollinger Bands (BB)
///
/// Bollinger Bands consist of three lines:
/// - Middle Band: Simple Moving Average (SMA)
/// - Upper Band: SMA + (k * Standard Deviation)
/// - Lower Band: SMA - (k * Standard Deviation)
///
/// # Formula
///
/// Middle Band = SMA(price, period)
/// Upper Band = Middle Band + (k * σ)
/// Lower Band = Middle Band - (k * σ)
///
/// where:
/// - k is typically 2
/// - σ is the standard deviation
///
/// # Links
///
/// * [Bollinger Bands, Wikipedia](https://en.wikipedia.org/wiki/Bollinger_Bands)
/// * [Bollinger Bands, Investopedia](https://www.investopedia.com/terms/b/bollingerbands.asp)
///
#[doc(alias = "BB")]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct BollingerBands {
    period: usize,
    multiplier: f64,
    prices: Vec<f64>,
    index: usize,
    count: usize,
}

impl BollingerBands {
    pub fn new(period: usize, multiplier: f64) -> Result<Self> {
        Ok(Self {
            period,
            multiplier,
            prices: vec![0.0; period],
            index: 0,
            count: 0,
        })
    }

    fn calculate_sma(&self) -> f64 {
        let sum: f64 = self.prices.iter().sum();
        sum / (self.count.min(self.period) as f64)
    }

    fn calculate_std_dev(&self, sma: f64) -> f64 {
        let variance: f64 = self.prices
            .iter()
            .take(self.count.min(self.period))
            .map(|&price| {
                let diff = price - sma;
                diff * diff
            })
            .sum::<f64>() / (self.count.min(self.period) as f64);
        
        variance.sqrt()
    }
}

impl Period for BollingerBands {
    fn period(&self) -> usize {
        self.period
    }
}

impl Next<f64> for BollingerBands {
    type Output = BollingerBandsOutput;

    fn next(&mut self, input: f64) -> Self::Output {
        self.prices[self.index] = input;
        self.index = (self.index + 1) % self.period;
        self.count += 1;

        let middle = self.calculate_sma();
        let std_dev = self.calculate_std_dev(middle);
        let upper = middle + (self.multiplier * std_dev);
        let lower = middle - (self.multiplier * std_dev);

        BollingerBandsOutput {
            upper,
            middle,
            lower,
            bandwidth: if middle != 0.0 {
                ((upper - lower) / middle) * 100.0
            } else {
                0.0
            },
            percent_b: if upper != lower {
                (input - lower) / (upper - lower)
            } else {
                0.5
            },
        }
    }
}

impl<T: Close> Next<&T> for BollingerBands {
    type Output = BollingerBandsOutput;

    fn next(&mut self, input: &T) -> Self::Output {
        self.next(input.close())
    }
}

impl Reset for BollingerBands {
    fn reset(&mut self) {
        self.index = 0;
        self.count = 0;
        for i in 0..self.period {
            self.prices[i] = 0.0;
        }
    }
}

impl Default for BollingerBands {
    fn default() -> Self {
        Self::new(20, 2.0).unwrap()
    }
}

impl fmt::Display for BollingerBands {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "BB({}, {})", self.period, self.multiplier)
    }
}

/// Output structure for Bollinger Bands
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BollingerBandsOutput {
    /// Upper band value
    pub upper: f64,
    /// Middle band value (SMA)
    pub middle: f64,
    /// Lower band value
    pub lower: f64,
    /// Bandwidth: ((upper - lower) / middle) * 100
    pub bandwidth: f64,
    /// %B: (price - lower) / (upper - lower)
    pub percent_b: f64,
}

impl fmt::Display for BollingerBandsOutput {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "BB(upper: {:.2}, middle: {:.2}, lower: {:.2}, bandwidth: {:.2}%, %B: {:.3})",
            self.upper, self.middle, self.lower, self.bandwidth, self.percent_b
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let bb = BollingerBands::new(20, 2.0);
        assert!(bb.is_ok());
        let bb = bb.unwrap();
        assert_eq!(bb.period(), 20);
    }

    #[test]
    fn test_new_different_configs() {
        assert!(BollingerBands::new(10, 1.5).is_ok());
        assert!(BollingerBands::new(5, 3.0).is_ok());
        assert!(BollingerBands::new(50, 2.5).is_ok());
    }

    #[test]
    fn test_default() {
        let bb = BollingerBands::default();
        assert_eq!(bb.period(), 20);
    }

    #[test]
    fn test_next() {
        let mut bb = BollingerBands::new(5, 2.0).unwrap();
        
        let prices = vec![10.0, 11.0, 12.0, 11.0, 10.0, 9.0, 10.0, 11.0, 12.0, 13.0];
        let mut outputs = Vec::new();
        
        for price in prices {
            outputs.push(bb.next(price));
        }

        // After 5 periods, we should have meaningful values
        let output = &outputs[4];
        assert!(output.upper > output.middle);
        assert!(output.middle > output.lower);
        assert!(output.bandwidth > 0.0);
        
        // Middle should be close to SMA
        let expected_sma = (10.0 + 11.0 + 12.0 + 11.0 + 10.0) / 5.0;
        assert!((output.middle - expected_sma).abs() < 0.01);
    }

    #[test]
    fn test_next_with_struct() {
        let mut bb = BollingerBands::new(5, 2.0).unwrap();
        
        let bars = vec![
            TestBar { close: 100.0 },
            TestBar { close: 101.0 },
            TestBar { close: 102.0 },
        ];

        for bar in &bars {
            let output = bb.next(bar);
            assert!(output.upper.is_finite());
            assert!(output.middle.is_finite());
            assert!(output.lower.is_finite());
        }
    }

    #[test]
    fn test_bands_relationship() {
        let mut bb = BollingerBands::new(10, 2.0).unwrap();
        
        let prices = vec![
            100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.0, 103.0, 102.0, 101.0,
        ];

        for price in prices {
            let output = bb.next(price);
            // Upper band should always be >= middle band
            assert!(output.upper >= output.middle);
            // Middle band should always be >= lower band
            assert!(output.middle >= output.lower);
        }
    }

    #[test]
    fn test_percent_b() {
        let mut bb = BollingerBands::new(3, 2.0).unwrap();
        
        // Use consistent prices first
        bb.next(10.0);
        bb.next(10.0);
        let output = bb.next(10.0);
        
        // Price at middle should give %B around 0.5
        // (with std dev = 0, %B calculation uses else branch)
        assert!((output.percent_b - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_percent_b_at_boundaries() {
        let mut bb = BollingerBands::new(5, 2.0).unwrap();
        
        // Create a stable baseline
        for _ in 0..5 {
            bb.next(100.0);
        }

        // Price variation
        bb.next(95.0);
        bb.next(105.0);
        bb.next(100.0);
        bb.next(98.0);
        let output = bb.next(102.0);

        // %B should be between 0 and 1 for prices within bands
        assert!(output.percent_b >= 0.0);
        assert!(output.percent_b <= 1.0);
    }

    #[test]
    fn test_percent_b_above_upper() {
        let mut bb = BollingerBands::new(5, 2.0).unwrap();
        
        // Create a stable baseline with small variation
        let prices = vec![100.0, 101.0, 100.5, 99.5, 100.0];
        for price in prices {
            bb.next(price);
        }

        // Get the current bands
        let baseline = bb.next(100.0);
        let upper_band = baseline.upper;
        
        // Now test with a price significantly above the upper band
        // Use the previous upper band value + extra to ensure it's above
        let high_price = upper_band + 5.0;
        let output = bb.next(high_price);

        // %B should be high when price is above typical range
        // Note: The extreme value becomes part of rolling window, so it may not be > 1.0
        // but it should be significantly high (> 0.8)
        assert!(output.percent_b > 0.8, "Expected %B > 0.8, got {}", output.percent_b);
    }

    #[test]
    fn test_percent_b_below_lower() {
        let mut bb = BollingerBands::new(5, 2.0).unwrap();
        
        // Create a stable baseline with small variation
        let prices = vec![100.0, 99.0, 100.5, 99.5, 100.0];
        for price in prices {
            bb.next(price);
        }

        // Get the current bands
        let baseline = bb.next(100.0);
        let lower_band = baseline.lower;
        
        // Now test with a price significantly below the lower band
        // Use the previous lower band value - extra to ensure it's below
        let low_price = lower_band - 5.0;
        let output = bb.next(low_price);

        // %B should be low when price is below typical range
        // Note: The extreme value becomes part of rolling window, so it may not be < 0.0
        // but it should be significantly low (< 0.2)
        assert!(output.percent_b < 0.2, "Expected %B < 0.2, got {}", output.percent_b);
    }

    #[test]
    fn test_reset() {
        let mut bb = BollingerBands::new(5, 2.0).unwrap();
        
        bb.next(10.0);
        bb.next(11.0);
        bb.next(12.0);
        
        bb.reset();
        
        assert_eq!(bb.index, 0);
        assert_eq!(bb.count, 0);
    }

    #[test]
    fn test_reset_produces_same_results() {
        let mut bb1 = BollingerBands::new(5, 2.0).unwrap();
        let mut bb2 = BollingerBands::new(5, 2.0).unwrap();

        let prices = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        
        // First run
        let mut results1 = vec![];
        for &price in &prices {
            results1.push(bb1.next(price));
        }

        // Reset and second run
        bb1.reset();
        let mut results2 = vec![];
        for &price in &prices {
            results2.push(bb1.next(price));
        }

        // Fresh indicator
        let mut results3 = vec![];
        for &price in &prices {
            results3.push(bb2.next(price));
        }

        // Results after reset should match fresh indicator
        for i in 0..prices.len() {
            assert_eq!(results2[i].upper, results3[i].upper);
            assert_eq!(results2[i].middle, results3[i].middle);
            assert_eq!(results2[i].lower, results3[i].lower);
        }
    }

    #[test]
    fn test_display() {
        let bb = BollingerBands::new(20, 2.0).unwrap();
        let display = format!("{}", bb);
        assert_eq!(display, "BB(20, 2)");
        
        let output = BollingerBandsOutput {
            upper: 110.0,
            middle: 100.0,
            lower: 90.0,
            bandwidth: 20.0,
            percent_b: 0.5,
        };
        let display = format!("{}", output);
        assert!(display.contains("110.00"));
        assert!(display.contains("100.00"));
        assert!(display.contains("90.00"));
    }

    #[test]
    fn test_bandwidth() {
        let mut bb = BollingerBands::new(5, 2.0).unwrap();
        
        // Low volatility
        for _ in 0..5 {
            bb.next(10.0);
        }
        let output1 = bb.next(10.0);
        
        bb.reset();
        
        // High volatility
        let volatile_prices = vec![5.0, 15.0, 5.0, 15.0, 5.0, 15.0];
        let mut output2 = BollingerBandsOutput {
            upper: 0.0,
            middle: 0.0,
            lower: 0.0,
            bandwidth: 0.0,
            percent_b: 0.0,
        };
        for price in volatile_prices {
            output2 = bb.next(price);
        }
        
        // High volatility should have larger bandwidth
        assert!(output2.bandwidth > output1.bandwidth);
    }

    #[test]
    fn test_bandwidth_calculation() {
        let mut bb = BollingerBands::new(5, 2.0).unwrap();
        
        let prices = vec![100.0, 102.0, 98.0, 101.0, 99.0];
        let mut last_output = None;
        
        for price in prices {
            last_output = Some(bb.next(price));
        }

        if let Some(output) = last_output {
            // Bandwidth = ((upper - lower) / middle) * 100
            let expected_bandwidth = ((output.upper - output.lower) / output.middle) * 100.0;
            assert!((output.bandwidth - expected_bandwidth).abs() < 0.01);
        }
    }

    #[test]
    fn test_zero_volatility() {
        let mut bb = BollingerBands::new(5, 2.0).unwrap();
        
        // All same prices = zero volatility
        for _ in 0..10 {
            bb.next(100.0);
        }
        let output = bb.next(100.0);

        // With zero std dev, upper and lower should equal middle
        assert_eq!(output.upper, output.middle);
        assert_eq!(output.lower, output.middle);
        assert_eq!(output.bandwidth, 0.0);
        assert_eq!(output.percent_b, 0.5); // Uses else branch
    }

    #[test]
    fn test_squeeze_and_expansion() {
        let mut bb = BollingerBands::new(5, 2.0).unwrap();
        
        // Squeeze phase (low volatility)
        for _ in 0..5 {
            bb.next(100.0);
        }
        let squeeze_output = bb.next(100.0);
        let squeeze_bandwidth = squeeze_output.bandwidth;

        // Expansion phase (high volatility)
        bb.next(110.0);
        bb.next(90.0);
        bb.next(115.0);
        bb.next(85.0);
        let expansion_output = bb.next(105.0);
        let expansion_bandwidth = expansion_output.bandwidth;

        // Bandwidth should increase during expansion
        assert!(expansion_bandwidth > squeeze_bandwidth);
    }

    #[test]
    fn test_middle_band_is_sma() {
        let mut bb = BollingerBands::new(5, 2.0).unwrap();
        
        let prices = vec![10.0, 12.0, 14.0, 16.0, 18.0];
        let mut last_output = None;
        
        for price in &prices {
            last_output = Some(bb.next(*price));
        }

        // Middle band should equal SMA of last 5 prices
        let expected_sma: f64 = prices.iter().sum::<f64>() / prices.len() as f64;
        if let Some(output) = last_output {
            assert!((output.middle - expected_sma).abs() < 0.01);
        }
    }

    #[test]
    fn test_multiplier_effect() {
        let mut bb1 = BollingerBands::new(5, 1.0).unwrap();
        let mut bb2 = BollingerBands::new(5, 3.0).unwrap();
        
        let prices = vec![100.0, 105.0, 95.0, 102.0, 98.0];
        
        let mut output1 = None;
        let mut output2 = None;
        
        for price in prices {
            output1 = Some(bb1.next(price));
            output2 = Some(bb2.next(price));
        }

        let out1 = output1.unwrap();
        let out2 = output2.unwrap();

        // Higher multiplier should create wider bands
        let width1 = out1.upper - out1.lower;
        let width2 = out2.upper - out2.lower;
        
        assert!(width2 > width1);
        // Middle bands should be the same (same SMA)
        assert!((out1.middle - out2.middle).abs() < 0.01);
    }

    #[test]
    fn test_uptrend() {
        let mut bb = BollingerBands::new(10, 2.0).unwrap();
        
        // Steady uptrend
        let mut prices = vec![];
        for i in 0..20 {
            prices.push(100.0 + (i as f64) * 0.5);
        }

        let mut last_output = None;
        for price in prices {
            last_output = Some(bb.next(price));
        }

        // In uptrend, recent prices should be near upper band
        if let Some(output) = last_output {
            // %B should be relatively high (> 0.5)
            assert!(output.percent_b > 0.5);
        }
    }

    #[test]
    fn test_downtrend() {
        let mut bb = BollingerBands::new(10, 2.0).unwrap();
        
        // Steady downtrend
        let mut prices = vec![];
        for i in 0..20 {
            prices.push(100.0 - (i as f64) * 0.5);
        }

        let mut last_output = None;
        for price in prices {
            last_output = Some(bb.next(price));
        }

        // In downtrend, recent prices should be near lower band
        if let Some(output) = last_output {
            // %B should be relatively low (< 0.5)
            assert!(output.percent_b < 0.5);
        }
    }

    #[test]
    fn test_period_effect() {
        let mut bb_short = BollingerBands::new(5, 2.0).unwrap();
        let mut bb_long = BollingerBands::new(20, 2.0).unwrap();
        
        let prices = vec![
            100.0, 102.0, 98.0, 103.0, 97.0, 104.0, 96.0, 105.0, 95.0, 106.0,
        ];

        let mut short_output = None;
        let mut long_output = None;
        
        for price in prices {
            short_output = Some(bb_short.next(price));
            long_output = Some(bb_long.next(price));
        }

        // Shorter period should be more responsive
        // Both should have valid outputs
        assert!(short_output.is_some());
        assert!(long_output.is_some());
    }

    #[test]
    fn test_clone() {
        let bb1 = BollingerBands::new(20, 2.0).unwrap();
        let bb2 = bb1.clone();
        
        assert_eq!(bb1.period(), bb2.period());
        assert_eq!(bb1.multiplier, bb2.multiplier);
    }

    #[test]
    fn test_output_clone() {
        let output1 = BollingerBandsOutput {
            upper: 110.0,
            middle: 100.0,
            lower: 90.0,
            bandwidth: 20.0,
            percent_b: 0.5,
        };
        let output2 = output1.clone();
        
        assert_eq!(output1, output2);
    }

    #[test]
    fn test_rolling_window() {
        let mut bb = BollingerBands::new(3, 2.0).unwrap();
        
        // Fill the window
        bb.next(10.0);
        bb.next(11.0);
        bb.next(12.0);
        
        // Add more values - should use rolling window
        let output1 = bb.next(13.0); // Window: [11, 12, 13]
        let output2 = bb.next(14.0); // Window: [12, 13, 14]
        
        // Middle should change as window rolls
        assert_ne!(output1.middle, output2.middle);
        assert!(output2.middle > output1.middle); // Increasing trend
    }

    #[test]
    fn test_bandwidth_zero_middle() {
        let mut bb = BollingerBands::new(5, 2.0).unwrap();
        
        // Edge case: prices at zero
        for _ in 0..5 {
            bb.next(0.0);
        }
        let output = bb.next(0.0);
        
        // Should handle division by zero in bandwidth calculation
        assert_eq!(output.bandwidth, 0.0);
        assert_eq!(output.middle, 0.0);
    }

    #[test]
    fn test_all_values_finite() {
        let mut bb = BollingerBands::new(10, 2.0).unwrap();
        
        let prices = vec![
            100.0, 101.0, 99.0, 102.0, 98.0, 103.0, 97.0, 104.0, 96.0, 105.0,
            95.0, 106.0, 94.0, 107.0, 93.0,
        ];

        for price in prices {
            let output = bb.next(price);
            
            assert!(output.upper.is_finite());
            assert!(output.middle.is_finite());
            assert!(output.lower.is_finite());
            assert!(output.bandwidth.is_finite());
            assert!(output.percent_b.is_finite());
            
            assert!(!output.upper.is_nan());
            assert!(!output.middle.is_nan());
            assert!(!output.lower.is_nan());
            assert!(!output.bandwidth.is_nan());
            assert!(!output.percent_b.is_nan());
        }
    }
}
