use tch::{Tensor, nn, nn::Module, nn::RNN};

pub struct TorchStockModel {
    input_proj: nn::Linear,
    lstm: Option<nn::LSTM>,
    lstm_to_transformer: Option<nn::Linear>,
    transformer: TransformerEncoder,
    output_proj_1day: nn::Linear,     // 1-day prediction head
    output_proj_3day: nn::Linear,     // 3-day prediction head (NEW)
    confidence_head_1day: nn::Linear, // 1-day confidence
    confidence_head_3day: nn::Linear, // 3-day confidence (NEW)
    dropout: f32,
}

struct TransformerEncoder {
    layers: Vec<TransformerEncoderLayer>,
}

struct TransformerEncoderLayer {
    self_attn: nn::Linear,
    feed_forward: nn::Sequential,
    norm1: nn::LayerNorm,
    norm2: nn::LayerNorm,
}

impl TorchStockModel {
    pub fn new(vs: &nn::Path, config: &ModelConfig) -> Self {
        let linear_config = nn::LinearConfig {
            ws_init: nn::Init::Randn {
                mean: 0.0,
                stdev: 0.01,
            },
            bs_init: Some(nn::Init::Const(0.0)),
            bias: true,
        };

        let input_proj = nn::linear(vs / "input_proj", 105, config.d_model, linear_config);

        // LSTM layers for temporal feature extraction
        let (lstm, lstm_to_transformer) = if config.use_lstm {
            let lstm_config = nn::RNNConfig {
                num_layers: config.lstm_layers,
                dropout: 0.1,
                bidirectional: true,
                ..Default::default()
            };
            let lstm = nn::lstm(vs / "lstm", config.d_model, config.lstm_hidden, lstm_config);
            // Bidirectional LSTM outputs lstm_hidden * 2
            let proj = nn::linear(
                vs / "lstm_proj",
                config.lstm_hidden * 2,
                config.d_model,
                linear_config,
            );
            (Some(lstm), Some(proj))
        } else {
            (None, None)
        };

        let mut layers = Vec::new();
        for i in 0..config.n_layers {
            let layer_path = vs / format!("layer_{}", i);
            layers.push(TransformerEncoderLayer::new(
                &layer_path,
                config.d_model,
                config.n_heads,
                config.d_ff,
                config.dropout,
            ));
        }

        let transformer = TransformerEncoder { layers };

        // Dual prediction heads for 1-day and 3-day forecasts
        let output_proj_1day =
            nn::linear(vs / "output_proj_1day", config.d_model, 105, linear_config);
        let output_proj_3day =
            nn::linear(vs / "output_proj_3day", config.d_model, 105, linear_config);

        // Confidence heads: multi-layer networks for better confidence calibration
        // Uses residual path to prevent vanishing gradients
        let confidence_head_config = nn::LinearConfig {
            ws_init: nn::Init::Randn {
                mean: 0.0,
                stdev: 0.01,
            },
            bs_init: Some(nn::Init::Const(-0.5)), // Initialize biased towards higher confidence
            bias: true,
        };
        let confidence_head_1day = nn::linear(
            vs / "confidence_head_1day",
            config.d_model,
            1,
            confidence_head_config,
        );
        let confidence_head_3day = nn::linear(
            vs / "confidence_head_3day",
            config.d_model,
            1,
            confidence_head_config,
        );

        Self {
            input_proj,
            lstm,
            lstm_to_transformer,
            transformer,
            output_proj_1day,
            output_proj_3day,
            confidence_head_1day,
            confidence_head_3day,
            dropout: config.dropout,
        }
    }

    pub fn forward(&self, input: &Tensor, train: bool) -> Tensor {
        // For backward compatibility, forward() returns 1-day predictions only
        let (pred_1day, _pred_3day, _conf_1day, _conf_3day) = self.forward_dual(input, train);
        pred_1day
    }

    /// Forward pass that returns both predictions and confidence scores
    /// Dual prediction forward pass returning both 1-day and 3-day predictions with confidence
    /// Returns: (pred_1day, pred_3day, conf_1day, conf_3day) where confidence is in range [0, 1]
    pub fn forward_dual(&self, input: &Tensor, train: bool) -> (Tensor, Tensor, Tensor, Tensor) {
        let x = self.input_proj.forward(input);
        let x = if train { x.dropout(self.dropout, true) } else { x };

        // LSTM for temporal processing if enabled
        let x = if let (Some(lstm), Some(proj)) = (&self.lstm, &self.lstm_to_transformer) {
            // LSTM expects (seq_len, batch, features)
            let x_permuted = x.permute(&[1, 0, 2]);
            let lstm_out = lstm.seq(&x_permuted).0;
            // Convert back to (batch, seq_len, features)
            let lstm_out = lstm_out.permute(&[1, 0, 2]);
            let lstm_out = proj.forward(&lstm_out);
            if train {
                lstm_out.dropout(0.1, true)
            } else {
                lstm_out
            }
        } else {
            x
        };

        // Transformer for attention-based refinement (shared encoder)
        let features = self.transformer.forward(&x, train);

        // Dual prediction heads for 1-day and 3-day forecasts
        let pred_1day = self.output_proj_1day.forward(&features);
        let pred_3day = self.output_proj_3day.forward(&features);

        // Separate confidence heads for each horizon
        let conf_1day = self.confidence_head_1day.forward(&features).sigmoid();
        let conf_3day = self.confidence_head_3day.forward(&features).sigmoid();

        (pred_1day, pred_3day, conf_1day, conf_3day)
    }

    pub fn forward_with_confidence(&self, input: &Tensor, train: bool) -> (Tensor, Tensor) {
        let (pred_1day, _pred_3day, conf_1day, _conf_3day) = self.forward_dual(input, train);
        (pred_1day, conf_1day)
    }
}

impl TransformerEncoderLayer {
    fn new(vs: &nn::Path, d_model: i64, _n_heads: i64, d_ff: i64, dropout: f32) -> Self {
        let self_attn = nn::linear(vs / "attn", d_model, d_model, Default::default());

        let feed_forward = nn::seq()
            .add(nn::linear(vs / "ff1", d_model, d_ff, Default::default()))
            .add_fn(|x: &Tensor| x.relu())
            .add_fn(move |x: &Tensor| x.dropout(dropout, true))
            .add(nn::linear(vs / "ff2", d_ff, d_model, Default::default()));

        let norm1 = nn::layer_norm(vs / "norm1", vec![d_model], Default::default());
        let norm2 = nn::layer_norm(vs / "norm2", vec![d_model], Default::default());

        Self {
            self_attn,
            feed_forward,
            norm1,
            norm2,
        }
    }

    fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let attn_out = self.self_attn.forward(x);
        let attn_out = if train {
            attn_out.dropout(0.1, true)
        } else {
            attn_out
        };
        let x = (x + attn_out).apply(&self.norm1);
        let ff_out = self.feed_forward.forward(&x);
        (x + ff_out).apply(&self.norm2)
    }
}

impl TransformerEncoder {
    fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let mut output = x.shallow_clone();
        for layer in &self.layers {
            output = layer.forward(&output, train);
        }
        output
    }
}

pub struct ModelConfig {
    pub d_model: i64,
    pub n_heads: i64,
    pub n_layers: i64,
    pub d_ff: i64,
    pub lstm_hidden: i64,
    pub lstm_layers: i64,
    pub use_lstm: bool,
    pub confidence_threshold: f64,
    pub dropout: f32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            d_model: 384, // Balanced capacity
            n_heads: 8,
            n_layers: 5,               // Moderate depth
            d_ff: 1536,                // Balanced FF dimension
            lstm_hidden: 256,          // LSTM hidden dimension
            lstm_layers: 2,            // Number of LSTM layers
            use_lstm: true,            // Enable LSTM + Transformer hybrid
            confidence_threshold: 0.6, // Filter predictions below 60% confidence
            dropout: 0.1,
        }
    }
}
