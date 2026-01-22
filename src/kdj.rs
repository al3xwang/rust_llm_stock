use core::{ f32, f64 };
use std::f64::{ INFINITY };
use std::fmt;

use ta::errors::{ Result, TaError };
use ta::{ Close, High, Low, Next, Period, Reset };
#[cfg(feature = "serde")]
use serde::{ Deserialize, Serialize };

///
/// # Links
///
/// * [Simple Moving Average, Wikipedia](https://en.wikipedia.org/wiki/Moving_average#Simple_moving_average)
///
#[doc(alias = "KDJ")]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct KDJIndicator {
    period: usize,

    highest_index: usize,
    lowest_index: usize,
    rsv: f64,
    output: KdjOutput,
    count: usize,
    cur_index: usize,
    high_deque: Box<[f64]>,
    low_deque: Box<[f64]>,
}

impl KDJIndicator {
    pub fn new(period: usize) -> Result<Self> {
        // match period {
        //     0 => Err(TaError::InvalidParameter),
        //     _ =>
        Ok(Self {
            period,
            count: 0,
            cur_index: 0,
            highest_index: 0,
            lowest_index: 0,
            rsv: 0.0,
            output: KdjOutput {
                k: 50.0,
                d: 50.0,
                j: 0.0,
            },

            high_deque: vec![-INFINITY; period].into_boxed_slice(),
            low_deque: vec![INFINITY; period].into_boxed_slice(),
        })
        // }
    }
    fn find_highest_index(&mut self) -> usize {
        for (i, &val) in self.high_deque.iter().enumerate() {
            if val > self.high_deque[self.highest_index] {
                self.highest_index = i;
            }
        }
        self.highest_index
    }

    fn find_lowest_index(&mut self) -> usize {
        for (i, &val) in self.low_deque.iter().enumerate() {
            if val < self.low_deque[self.lowest_index] {
                self.lowest_index = i;
            }
        }
        self.lowest_index
    }
}

impl Period for KDJIndicator {
    fn period(&self) -> usize {
        self.period
    }
}

impl<T: Close + High + Low> Next<&T> for KDJIndicator {
    type Output = KdjOutput;

    fn next(&mut self, input: &T) -> Self::Output {
        self.high_deque[self.cur_index] = input.high();
        self.low_deque[self.cur_index] = input.low();
        self.count += 1;

        if self.count >= self.period {
            let _ = self.find_highest_index();
            let _ = self.find_lowest_index();

            self.rsv =
                ((input.close() - self.low_deque[self.lowest_index]) /
                    (self.high_deque[self.highest_index] - self.low_deque[self.lowest_index])) *
                100.0;

            self.output.k =
                (2.0 / ((self.period as f64) + 1.0)) * self.rsv +
                (1.0 - 2.0 / ((self.period as f64) + 1.0)) * self.output.k;
            self.output.d =
                (2.0 / ((self.period as f64) + 1.0)) * self.output.k +
                (1.0 - 2.0 / ((self.period as f64) + 1.0)) * self.output.d;

            self.output.j = 3.0 * self.output.k - 2.0 * self.output.d;
        }

        self.cur_index = if self.cur_index + 1 < self.period { self.cur_index + 1 } else { 0 };

        self.output
    }
}

impl Reset for KDJIndicator {
    fn reset(&mut self) {
        self.cur_index = 0;
        self.count = 0;
        self.highest_index = 0;
        self.lowest_index = 0;
        self.rsv = 0.0;
        self.output = KdjOutput {
            k: 50.0,
            d: 50.0,
            j: 0.0,
        };
        for i in 0..self.period {
            self.high_deque[i] = -INFINITY;
            self.low_deque[i] = INFINITY;
        }
    }
}

impl Default for KDJIndicator {
    fn default() -> Self {
        Self::new(9).unwrap()
    }
}

impl fmt::Display for KDJIndicator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "KDJ(K:{:2.2} D:{:2.2} J:{:2.2})", self.output.k, self.output.d, self.output.j)
    }
}

#[derive(Debug, Clone, PartialEq, Copy)]
pub struct KdjOutput {
    pub k: f64,
    pub d: f64,
    pub j: f64,
}

// %D = Moving Average of %K
// %J = (3%K) – (2%D)
// Where:

// Close is the closing price
// Low(N) is the lowest low over N periods
// High(N) is the highest high over N periods

#[cfg(test)]
mod tests {
    use super::*;
    use ta::Next;

    #[derive(Debug)]
    struct TestBar {
        high: f64,
        low: f64,
        close: f64,
    }

    impl High for TestBar {
        fn high(&self) -> f64 {
            self.high
        }
    }

    impl Low for TestBar {
        fn low(&self) -> f64 {
            self.low
        }
    }

    impl Close for TestBar {
        fn close(&self) -> f64 {
            self.close
        }
    }

    #[test]
    fn test_kdj_new() {
        let kdj = KDJIndicator::new(9).unwrap();
        assert_eq!(kdj.period(), 9);
    }

    #[test]
    fn test_kdj_default() {
        let kdj = KDJIndicator::default();
        assert_eq!(kdj.period(), 9);
        assert_eq!(kdj.output.k, 50.0);
        assert_eq!(kdj.output.d, 50.0);
        assert_eq!(kdj.output.j, 0.0);
    }

    #[test]
    fn test_kdj_initial_values() {
        let mut kdj = KDJIndicator::new(3).unwrap();
        
        // First bar - should return initial values until period is reached
        let bar1 = TestBar { high: 110.0, low: 100.0, close: 105.0 };
        let result1 = kdj.next(&bar1);
        assert_eq!(result1.k, 50.0);
        assert_eq!(result1.d, 50.0);
        assert_eq!(result1.j, 0.0);
    }

    #[test]
    fn test_kdj_calculation_simple() {
        let mut kdj = KDJIndicator::new(3).unwrap();
        
        // Simple uptrend data
        let bars = vec![
            TestBar { high: 110.0, low: 100.0, close: 105.0 },
            TestBar { high: 115.0, low: 105.0, close: 112.0 },
            TestBar { high: 120.0, low: 110.0, close: 118.0 },
            TestBar { high: 125.0, low: 115.0, close: 122.0 },
        ];

        let mut results = vec![];
        for bar in bars {
            let result = kdj.next(&bar);
            results.push(result);
        }

        // After period (3 bars), KDJ should start calculating real values
        let last_result = results.last().unwrap();
        
        // In an uptrend, K should be > 50 and increasing
        assert!(last_result.k > 50.0, "K should be > 50 in uptrend, got {}", last_result.k);
        assert!(last_result.d > 50.0, "D should be > 50 in uptrend, got {}", last_result.d);
        
        // J = 3K - 2D
        let expected_j = 3.0 * last_result.k - 2.0 * last_result.d;
        assert!((last_result.j - expected_j).abs() < 0.01, 
            "J should equal 3K - 2D, expected {}, got {}", expected_j, last_result.j);
    }

    #[test]
    fn test_kdj_calculation_downtrend() {
        let mut kdj = KDJIndicator::new(3).unwrap();
        
        // Simple downtrend data
        let bars = vec![
            TestBar { high: 120.0, low: 110.0, close: 115.0 },
            TestBar { high: 115.0, low: 105.0, close: 108.0 },
            TestBar { high: 110.0, low: 100.0, close: 102.0 },
            TestBar { high: 105.0, low: 95.0, close: 98.0 },
        ];

        let mut results = vec![];
        for bar in bars {
            let result = kdj.next(&bar);
            results.push(result);
        }

        let last_result = results.last().unwrap();
        
        // In a downtrend, K should be < 50 and decreasing
        assert!(last_result.k < 50.0, "K should be < 50 in downtrend, got {}", last_result.k);
        assert!(last_result.d < 50.0, "D should be < 50 in downtrend, got {}", last_result.d);
    }

    #[test]
    fn test_kdj_boundary_conditions() {
        let mut kdj = KDJIndicator::new(3).unwrap();
        
        // Test with close at highest high
        let bars = vec![
            TestBar { high: 100.0, low: 90.0, close: 95.0 },
            TestBar { high: 105.0, low: 95.0, close: 100.0 },
            TestBar { high: 110.0, low: 100.0, close: 110.0 }, // Close at highest
        ];

        for bar in bars {
            kdj.next(&bar);
        }
        
        // When close is at highest high, RSV should be 100
        // So K and D should trend towards 100
        let bar4 = TestBar { high: 115.0, low: 110.0, close: 115.0 };
        let result = kdj.next(&bar4);
        assert!(result.k > 80.0, "K should be high when close at highest, got {}", result.k);
    }

    #[test]
    fn test_kdj_boundary_at_lowest() {
        let mut kdj = KDJIndicator::new(3).unwrap();
        
        // Test with close at lowest low
        let bars = vec![
            TestBar { high: 110.0, low: 100.0, close: 105.0 },
            TestBar { high: 105.0, low: 95.0, close: 100.0 },
            TestBar { high: 100.0, low: 90.0, close: 90.0 }, // Close at lowest
        ];

        for bar in bars {
            kdj.next(&bar);
        }
        
        // When close is at lowest low, RSV should be 0
        // So K and D should trend towards 0
        let bar4 = TestBar { high: 95.0, low: 85.0, close: 85.0 };
        let result = kdj.next(&bar4);
        assert!(result.k < 20.0, "K should be low when close at lowest, got {}", result.k);
    }

    #[test]
    fn test_kdj_j_calculation() {
        let mut kdj = KDJIndicator::new(9).unwrap();
        
        // Real-world-like data
        let bars = vec![
            TestBar { high: 50.0, low: 45.0, close: 48.0 },
            TestBar { high: 51.0, low: 46.0, close: 49.0 },
            TestBar { high: 52.0, low: 47.0, close: 50.0 },
            TestBar { high: 53.0, low: 48.0, close: 51.0 },
            TestBar { high: 54.0, low: 49.0, close: 52.0 },
            TestBar { high: 55.0, low: 50.0, close: 53.0 },
            TestBar { high: 56.0, low: 51.0, close: 54.0 },
            TestBar { high: 57.0, low: 52.0, close: 55.0 },
            TestBar { high: 58.0, low: 53.0, close: 56.0 },
            TestBar { high: 59.0, low: 54.0, close: 57.0 },
        ];

        let mut last_result = KdjOutput { k: 0.0, d: 0.0, j: 0.0 };
        for bar in bars {
            last_result = kdj.next(&bar);
        }
        
        // Verify J = 3K - 2D formula
        let expected_j = 3.0 * last_result.k - 2.0 * last_result.d;
        assert!((last_result.j - expected_j).abs() < 0.001, 
            "J calculation incorrect: expected {}, got {}", expected_j, last_result.j);
    }

    #[test]
    fn test_kdj_reset() {
        let mut kdj = KDJIndicator::new(3).unwrap();
        
        // Add some data
        let bar1 = TestBar { high: 110.0, low: 100.0, close: 105.0 };
        let bar2 = TestBar { high: 115.0, low: 105.0, close: 112.0 };
        kdj.next(&bar1);
        kdj.next(&bar2);
        
        // Reset
        kdj.reset();
        
        // After reset, should behave like new indicator
        assert_eq!(kdj.cur_index, 0);
        assert_eq!(kdj.count, 0);
    }

    #[test]
    fn test_kdj_display() {
        let kdj = KDJIndicator::new(9).unwrap();
        let display = format!("{}", kdj);
        assert!(display.contains("KDJ"));
        assert!(display.contains("K:"));
        assert!(display.contains("D:"));
        assert!(display.contains("J:"));
    }

    #[test]
    fn test_kdj_different_periods() {
        let mut kdj5 = KDJIndicator::new(5).unwrap();
        let mut kdj14 = KDJIndicator::new(14).unwrap();
        
        assert_eq!(kdj5.period(), 5);
        assert_eq!(kdj14.period(), 14);
        
        // Same data should produce different results with different periods
        let bars = vec![
            TestBar { high: 110.0, low: 100.0, close: 105.0 },
            TestBar { high: 115.0, low: 105.0, close: 112.0 },
            TestBar { high: 120.0, low: 110.0, close: 118.0 },
            TestBar { high: 125.0, low: 115.0, close: 122.0 },
            TestBar { high: 130.0, low: 120.0, close: 128.0 },
            TestBar { high: 135.0, low: 125.0, close: 132.0 },
        ];

        let mut result5 = KdjOutput { k: 0.0, d: 0.0, j: 0.0 };
        let mut result14 = KdjOutput { k: 0.0, d: 0.0, j: 0.0 };
        
        for bar in &bars {
            result5 = kdj5.next(bar);
            result14 = kdj14.next(bar);
        }
        
        // Results should be different (shorter period is more responsive)
        assert_ne!(result5.k, result14.k);
    }

    #[test]
    fn test_kdj_overbought_oversold() {
        let mut kdj = KDJIndicator::new(9).unwrap();
        
        // Create overbought scenario (strong uptrend)
        let mut bars = vec![];
        for i in 0..15 {
            bars.push(TestBar {
                high: 100.0 + (i as f64) * 2.0,
                low: 95.0 + (i as f64) * 2.0,
                close: 99.0 + (i as f64) * 2.0,
            });
        }

        let mut last_result = KdjOutput { k: 0.0, d: 0.0, j: 0.0 };
        for bar in bars {
            last_result = kdj.next(&bar);
        }
        
        // In overbought conditions, K and D should be high
        assert!(last_result.k > 70.0, "K should be > 70 in overbought, got {}", last_result.k);
        assert!(last_result.d > 70.0, "D should be > 70 in overbought, got {}", last_result.d);
    }

    #[test]
    fn test_kdj_clone() {
        let kdj1 = KDJIndicator::new(9).unwrap();
        let kdj2 = kdj1.clone();
        
        assert_eq!(kdj1.period(), kdj2.period());
        assert_eq!(kdj1.output.k, kdj2.output.k);
        assert_eq!(kdj1.output.d, kdj2.output.d);
        assert_eq!(kdj1.output.j, kdj2.output.j);
    }

    #[test]
    fn test_kdj_output_clone() {
        let output1 = KdjOutput { k: 75.0, d: 70.0, j: 85.0 };
        let output2 = output1.clone();
        
        assert_eq!(output1, output2);
    }
}
