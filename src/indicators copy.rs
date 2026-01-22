use std::f64::INFINITY;
use std::fmt;

use ta::errors::{ Result, TaError };
use ta::{ Close, High, Low, Next, Period, Reset };
#[cfg(feature = "serde")]
use serde::{ Deserialize, Serialize };

/// Simple moving average (SMA).
///
/// # Formula
///
/// ![SMA](https://wikimedia.org/api/rest_v1/media/math/render/svg/e2bf09dc6deaf86b3607040585fac6078f9c7c89)
///
/// Where:
///
/// * _SMA<sub>t</sub>_ - value of simple moving average at a point of time _t_
/// * _period_ - number of periods (period)
/// * _p<sub>t</sub>_ - input value at a point of time _t_
///
/// # Parameters
///
/// * _period_ - number of periods (integer greater than 0)
///
/// # Example
///
/// ```
/// use ta::indicators::SimpleMovingAverage;
/// use ta::Next;
///
/// let mut sma = SimpleMovingAverage::new(3).unwrap();
/// assert_eq!(sma.next(10.0), 10.0);
/// assert_eq!(sma.next(11.0), 10.5);
/// assert_eq!(sma.next(12.0), 11.0);
/// assert_eq!(sma.next(13.0), 12.0);
/// ```
///
/// # Links
///
/// * [Simple Moving Average, Wikipedia](https://en.wikipedia.org/wiki/Moving_average#Simple_moving_average)
///
#[doc(alias = "WR")]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct WilliamRateIndicator {
    period: usize,
    highest: f64,
    lowest: f64,
    highest_index: usize,
    lowest_index: usize,
    cur_index: usize,
    close: f64,
    count: usize,
    r: f64,
    wr_deque: Box<[f64]>,
    high_deque: Box<[f64]>,
    low_deque: Box<[f64]>,
}

impl WilliamRateIndicator {
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
            highest: 0f64,
            lowest: 0f64,
            close: 0f64,
            r: 0f64,
            wr_deque: vec![0.0; period].into_boxed_slice(),
            high_deque: vec![-INFINITY; period].into_boxed_slice(),
            low_deque: vec![INFINITY; period].into_boxed_slice(),
        })
        // }
    }
    fn find_highest_index(&mut self) -> usize {
        for (i, &val) in self.high_deque.iter().enumerate() {
            if val > self.high_deque[self.highest_index] {
                self.highest = val;
                self.highest_index = i;
            }
        }
        self.highest_index
    }

    fn find_lowest_index(&mut self) -> usize {
        for (i, &val) in self.low_deque.iter().enumerate() {
            if val < self.low_deque[self.lowest_index] {
                self.lowest = val;
                self.lowest_index = i;
            }
        }
        self.lowest_index
    }
}

impl Period for WilliamRateIndicator {
    fn period(&self) -> usize {
        self.period
    }
}

// impl Next<f64> for WilliamRateIndicator {
//     type Output = f64;

//     fn next(&mut self, input: f64) -> Self::Output {
//         self.deque[self.index] = input;

//         self.index = if self.index + 1 < self.period { self.index + 1 } else { 0 };

//         if self.count < self.period {
//             self.count += 1;
//         }

//         self.r = self.r - old_val + input;
//         self.r / (self.count as f64)
//     }
// }

impl<T: Close + High + Low> Next<&T> for WilliamRateIndicator {
    type Output = f64;

    fn next(&mut self, input: &T) -> Self::Output {
        self.high_deque[self.cur_index] = input.high();
        self.low_deque[self.cur_index] = input.low();
        self.close = input.close();
        self.count += 1;
        if self.count >= self.period {
            let _ = self.find_highest_index();
            let _ = self.find_lowest_index();
            self.r =
                ((self.high_deque[self.highest_index] - input.close()) /
                    (self.high_deque[self.highest_index] - self.low_deque[self.lowest_index])) *
                100.0;
            // -100.0;

            self.wr_deque[self.cur_index] = self.r;
        }
        self.cur_index = if self.cur_index + 1 < self.period { self.cur_index + 1 } else { 0 };

        self.r
    }
}

impl Reset for WilliamRateIndicator {
    fn reset(&mut self) {
        self.cur_index = 0;
        self.highest = 0.0;
        self.lowest = 0.0;
        self.r = 0.0;
        for i in 0..self.period {
            self.wr_deque[i] = 0.0;
            self.high_deque[i] = INFINITY;
            self.low_deque[i] = -INFINITY;
        }
    }
}

impl Default for WilliamRateIndicator {
    fn default() -> Self {
        Self::new(11).unwrap()
    }
}

impl fmt::Display for WilliamRateIndicator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "WR({})", self.period)
    }
}
