pub const FEATURE_SIZE: usize = 105;

// Calibration: for percentage-based features, we want denormalized values in ±15% range.
// tanh(x) ≈ ±0.95 at x ≈ ±3, so we use a scale that maps 0.95 -> ~10-12%.
// With SOFT_SCALE = 3.0: atanh(0.95) * 3 ≈ 3.0 * 3.0 = 9.0 (within ±15%)
const PCT_LIMIT: f32 = 15.0;
const PCT_SOFT_SCALE: f32 = 3.0;
const ATANH_GUARD: f32 = 0.999;

#[inline]
fn normalize_pct_feature(value: f32) -> f32 {
    (value / PCT_SOFT_SCALE).tanh()
}

#[inline]
pub fn denormalize_pct_feature(value: f32) -> f32 {
    let clamped = value.clamp(-ATANH_GUARD, ATANH_GUARD);
    let denorm = clamped.atanh() * PCT_SOFT_SCALE;
    denorm.clamp(-PCT_LIMIT, PCT_LIMIT)
}

/// Normalize all 105 features for percentage-based schema.
/// Prices are already percentage changes from pre_close, so we mainly clamp/scale.
pub fn normalize_features(mut features: [f32; FEATURE_SIZE], _reference_close_pct: f32) -> [f32; FEATURE_SIZE] {
    // Replace NaN/Inf with sane defaults to avoid propagating invalid values.
    for i in 0..features.len() {
        if !features[i].is_finite() {
            features[i] = 0.0;
        }
    }

    let mut normalized = features;

    // [0-2] Categorical features (already encoded, keep within 0-1 range)
    normalized[0] = features[0].clamp(0.0, 100.0) / 100.0;
    normalized[1] = features[1].clamp(0.0, 100.0) / 100.0;
    normalized[2] = features[2].clamp(0.0, 100.0) / 100.0;

    // [3-4] Volume/amount using log scale
    normalized[3] = (features[3].max(1.0)).ln().clamp(-10.0, 30.0) / 20.0;
    normalized[4] = (features[4].max(1.0)).ln().clamp(-10.0, 30.0) / 20.0;

    // [5-8] Temporal features normalized to 0-1
    normalized[5] = features[5] / 12.0;
    normalized[6] = features[6] / 7.0;
    normalized[7] = features[7] / 4.0;
    normalized[8] = features[8] / 53.0;

    // [9-12] Percentage-based OHLC, clamp to +/-10%
    normalized[9] = normalize_pct_feature(features[9]);
    normalized[10] = normalize_pct_feature(features[10]);
    normalized[11] = normalize_pct_feature(features[11]);
    normalized[12] = normalize_pct_feature(features[12]);

    // [13-17] Intraday movements
    normalized[13] = normalize_pct_feature(features[13]);
    normalized[14] = normalize_pct_feature(features[14]);
    normalized[15] = normalize_pct_feature(features[15]);
    normalized[16] = features[16].clamp(0.0, 20.0) / 20.0;
    normalized[17] = features[17].clamp(0.0, 1.0);

    // [18-25] Moving averages (percentages)
    for i in 18..=25 {
        normalized[i] = normalize_pct_feature(features[i]);
    }

    // [26-32] MACD indicators
    for i in 26..=32 {
        normalized[i] = features[i].clamp(-1.0, 1.0);
    }

    // [33-36] RSI/KDJ style oscillators
    normalized[33] = features[33] / 100.0;
    normalized[34] = features[34].clamp(0.0, 100.0) / 100.0;
    normalized[35] = features[35].clamp(0.0, 100.0) / 100.0;
    normalized[36] = (features[36].clamp(-20.0, 120.0) + 20.0) / 140.0;

    // [37-41] Bollinger values
    for i in 37..=39 {
        normalized[i] = normalize_pct_feature(features[i]);
    }
    normalized[40] = features[40].clamp(0.0, 1.0);
    normalized[41] = features[41].clamp(0.0, 1.0);

    // [42-47] Volatility metrics
    normalized[42] = features[42].clamp(0.0, 10.0) / 10.0;
    normalized[43] = features[43].clamp(0.0, 1.0);
    normalized[44] = features[44].clamp(0.0, 1.0);
    normalized[45] = features[45].clamp(-2000.0, 12000.0) / 14000.0 + 0.14;
    normalized[46] = if features[46].abs() < 1.0 {
        0.0
    } else {
        (features[46].abs().ln() / 20.0 * features[46].signum()).clamp(-1.0, 1.0)
    };
    normalized[47] = features[47].clamp(0.0, 10.0) / 10.0;

    // [48-51] Momentum values
    normalized[48] = features[48].clamp(-0.2, 0.2) / 0.2;
    normalized[49] = features[49].clamp(-0.3, 0.3) / 0.3;
    normalized[50] = features[50].clamp(-0.5, 0.5) / 0.5;
    normalized[51] = features[51].clamp(0.0, 1.0);

    // [52-54] Candle sizes
    normalized[52] = normalize_pct_feature(features[52]);
    normalized[53] = features[53].clamp(0.0, 10.0) / 10.0;
    normalized[54] = features[54].clamp(0.0, 10.0) / 10.0;

    // [55-62] Trend & strength block
    normalized[55] = features[55].clamp(-1.0, 1.0);
    normalized[56] = features[56].clamp(0.0, 100.0) / 100.0;
    normalized[57] = normalize_pct_feature(features[57]);
    normalized[58] = features[58].clamp(-1.0, 1.0);
    normalized[59] = features[59].clamp(0.0, 100.0) / 100.0;
    normalized[60] = (features[60].clamp(-100.0, 0.0) + 100.0) / 100.0;
    normalized[61] = features[61].clamp(0.0, 100.0) / 100.0;
    normalized[62] = features[62].clamp(0.0, 100.0) / 100.0;

    // [63-65] Lagged returns
    for i in 63..=65 {
        normalized[i] = features[i].clamp(-0.1, 0.1) / 0.1;
    }

    // [66-67] Gaps
    normalized[66] = normalize_pct_feature(features[66]);
    normalized[67] = normalize_pct_feature(features[67]);

    // [68-69] Volume signals
    normalized[68] = features[68].clamp(-5.0, 5.0) / 5.0;
    normalized[69] = features[69];

    // [70-73] Price ROC & hist volatility
    normalized[70] = features[70].clamp(-0.2, 0.2) / 0.2;
    normalized[71] = features[71].clamp(-0.3, 0.3) / 0.3;
    normalized[72] = features[72].clamp(-0.5, 0.5) / 0.5;
    normalized[73] = features[73].clamp(0.0, 1.0);

    // [74-77] Pattern flags
    normalized[74] = features[74];
    normalized[75] = features[75];
    normalized[76] = features[76];
    normalized[77] = normalize_pct_feature(features[77]);

    // [78-86] Index exposures
    for i in 78..=86 {
        normalized[i] = normalize_pct_feature(features[i]);
    }

    // [87-90] Money flow (log scale with sign)
    normalized[87] = (features[87].abs().max(1.0)).ln() / 20.0 * features[87].signum();
    normalized[88] = (features[88].abs().max(1.0)).ln() / 20.0 * features[88].signum();
    normalized[89] = features[89].clamp(-1.0, 1.0);
    normalized[90] = (features[90].abs().max(1.0)).ln() / 20.0 * features[90].signum();

    // [91-93] Industry features
    normalized[91] = features[91].clamp(-0.1, 0.1) / 0.1;
    normalized[92] = features[92].clamp(-0.2, 0.2) / 0.2;
    normalized[93] = features[93].clamp(-0.2, 0.2) / 0.2;

    // [94-95] Volatility regime flags
    normalized[94] = features[94].clamp(0.0, 1.0);
    normalized[95] = features[95];

    // [96-104] Reserved - keep zeroed
    for i in 96..FEATURE_SIZE {
        normalized[i] = 0.0;
    }

    normalized
}
