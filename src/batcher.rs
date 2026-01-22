use burn::tensor::{backend::Backend, Tensor, TensorData};
use burn::data::dataloader::batcher::Batcher;
use crate::dataset::StockItem;

#[derive(Clone, Debug)]
pub struct StockBatch<B: Backend> {
    pub inputs: Tensor<B, 3>,
    pub targets: Tensor<B, 3>,
}

pub struct StockBatcher<B: Backend> {
    _b: std::marker::PhantomData<B>,
}

impl<B: Backend> StockBatcher<B> {
    pub fn new(_device: B::Device) -> Self {
        Self { _b: std::marker::PhantomData }
    }
}

impl<B: Backend> Batcher<B, StockItem, StockBatch<B>> for StockBatcher<B> {
    fn batch(&self, items: Vec<StockItem>, device: &B::Device) -> StockBatch<B> {
        let batch_size = items.len();
        // We expect items to have length seq_len + 1 so we can split into input and target
        let item_len = items.first().map(|i| i.values.len()).unwrap_or(0);
        let seq_len = if item_len > 0 { item_len - 1 } else { 0 };
        let feature_dim = 19; // OHLCV(5) + Weekday(1) + Week_No(1) + EMA5/10/20/30/60(5) + Volume Ratio(1) + MACD(2) + ASI(1) + OBV(1) + Amount(1) + pct_change(1)

        let mut inputs_data = Vec::with_capacity(batch_size * seq_len * feature_dim);
        let mut targets_data = Vec::with_capacity(batch_size * seq_len * feature_dim);
        
        for item in &items {
            // Input: 0..N-1
            for i in 0..seq_len {
                let mut features = item.values[i];
                // Normalize features to prevent gradient explosion:
                // OHLC (0-3): normalize by close price
                let close = features[3].max(1.0); // Avoid division by zero
                features[0] /= close; // open
                features[1] /= close; // high
                features[2] /= close; // low
                features[3] = 1.0;    // close becomes 1.0
                // Volume (4): log scale normalization
                features[4] = (features[4].max(1.0)).ln() / 20.0; // Log scale, typical range 10-25
                // Weekday (5): already 0-6, divide by 7 for 0-1 range
                features[5] /= 7.0;
                // Week_No (6): divide by 53 for 0-1 range
                features[6] /= 53.0;
                // EMA5/10/20/30/60 (7-11): normalize by close price
                features[7] /= close;  // EMA5
                features[8] /= close;  // EMA10
                features[9] /= close;  // EMA20
                features[10] /= close; // EMA30
                features[11] /= close; // EMA60
                // volume_ratio (12), MACD (13-14): clamp to reasonable range
                features[12] = features[12].clamp(0.0, 10.0); // Volume ratio 0-10x
                features[13] = features[13].clamp(-1.0, 1.0); // MACD line
                features[14] = features[14].clamp(-1.0, 1.0); // MACD signal
                features[15] = features[15].clamp(-2000.0, 12000.0) / 1000.0; // ASI normalized to ~±2 to 12 range
                features[16] = (features[16].abs().max(1.0)).ln() / 18.0 * features[16].signum(); // OBV log-normalized with sign
                features[17] = (features[17].max(1.0)).ln() / 18.0; // Amount log-normalized (ln(90M)≈18.3)
                features[18] = features[18].clamp(-10.0, 10.0) / 10.0; // pct_change normalized to ±1 range
                inputs_data.extend_from_slice(&features);
            }
            // Target: 1..N (apply same normalization)
            for i in 1..item_len {
                let mut features = item.values[i];
                let close = features[3].max(1.0);
                features[0] /= close;
                features[1] /= close;
                features[2] /= close;
                features[3] = 1.0;
                features[4] = (features[4].max(1.0)).ln() / 20.0;
                features[5] /= 7.0;
                features[6] /= 53.0;
                features[7] /= close;
                features[8] /= close;
                features[9] /= close;
                features[10] /= close;
                features[11] /= close;
                features[12] = features[12].clamp(0.0, 10.0);
                features[13] = features[13].clamp(-1.0, 1.0);
                features[14] = features[14].clamp(-1.0, 1.0);
                features[15] = features[15].clamp(-2000.0, 12000.0) / 1000.0;
                features[16] = (features[16].abs().max(1.0)).ln() / 18.0 * features[16].signum();
                features[17] = (features[17].max(1.0)).ln() / 18.0;
                features[18] = features[18].clamp(-10.0, 10.0) / 10.0;
                targets_data.extend_from_slice(&features);
            }
        }
        
        let shape = [batch_size, seq_len, feature_dim];

        let inputs = Tensor::<B, 3>::from_floats(
            TensorData::new(inputs_data, shape.to_vec()),
            device,
        );
        
        let targets = Tensor::<B, 3>::from_floats(
            TensorData::new(targets_data, shape.to_vec()),
            device,
        );

        StockBatch { inputs, targets }
    }
}
