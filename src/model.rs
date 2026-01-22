use crate::batcher::StockBatch;
use burn::{
    config::Config,
    module::Module,
    nn::{
        Linear, LinearConfig,
        loss::MseLoss,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    },
    tensor::{
        Tensor,
        backend::{AutodiffBackend, Backend},
    },
    train::{RegressionOutput, TrainOutput, TrainStep, ValidStep},
};

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = 256)]
    pub d_model: usize,
    #[config(default = 8)]
    pub n_head: usize,
    #[config(default = 4)]
    pub n_layer: usize,
    #[config(default = 1024)]
    pub d_ff: usize,
}

#[derive(Module, Debug)]
pub struct StockModel<B: Backend> {
    input_proj: Linear<B>,
    transformer: TransformerEncoder<B>,
    output_proj: Linear<B>,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> StockModel<B> {
        // Input projection: 19 features -> d_model
        let input_proj = LinearConfig::new(19, self.d_model).init(device);

        let transformer =
            TransformerEncoderConfig::new(self.d_model, self.d_ff, self.n_head, self.n_layer)
                .init(device);

        // Output projection: d_model -> 19 features
        let output_proj = LinearConfig::new(self.d_model, 19).init(device);

        StockModel {
            input_proj,
            transformer,
            output_proj,
        }
    }
}

impl<B: Backend> StockModel<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_batch_size, _seq_len, _features] = input.dims();

        let x = self.input_proj.forward(input);
        let x = self.transformer.forward(TransformerEncoderInput::new(x));
        self.output_proj.forward(x)
    }

    pub fn forward_regression(
        &self,
        input: Tensor<B, 3>,
        targets: Tensor<B, 3>,
    ) -> RegressionOutput<B> {
        let output = self.forward(input);

        let loss = MseLoss::new().forward(
            output.clone(),
            targets.clone(),
            burn::nn::loss::Reduction::Mean,
        );

        // RegressionOutput expects 2D tensors [batch_size, num_targets] usually,
        // but for sequence regression we might need to flatten or define our own output struct.
        // However, Burn's RegressionOutput is generic over B but fixed to Tensor<B, 2> for output/target in some versions?
        // Let's check if we can just flatten it for the metric calculation.

        let [batch, seq, features] = output.dims();
        let output_flat = output.reshape([batch * seq, features]);
        let targets_flat = targets.reshape([batch * seq, features]);

        RegressionOutput::new(loss, output_flat, targets_flat)
    }
}

impl<B: AutodiffBackend> TrainStep<StockBatch<B>, RegressionOutput<B>> for StockModel<B> {
    fn step(&self, batch: StockBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_regression(batch.inputs, batch.targets);
        let grads = item.loss.backward();

        TrainOutput::new(self, grads, item)
    }
}

impl<B: Backend> ValidStep<StockBatch<B>, RegressionOutput<B>> for StockModel<B> {
    fn step(&self, batch: StockBatch<B>) -> RegressionOutput<B> {
        self.forward_regression(batch.inputs, batch.targets)
    }
}

use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use uuid::Uuid;

#[derive(Debug, FromRow, Deserialize, Serialize)]
#[allow(non_snake_case)]
pub struct NoteModel {
    pub id: Uuid,
    pub title: String,
    pub content: String,
    pub category: Option<String>,
    pub published: Option<bool>,
    #[serde(rename = "createdAt")]
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(rename = "updatedAt")]
    pub updated_at: Option<chrono::DateTime<chrono::Utc>>,
}
