# 3-Day Prediction Implementation

## Overview

The stock analysis system now has complete infrastructure to support 3-day ahead price movement predictions. This document explains the implementation and what's still needed to fully activate this feature.

## Current Status

### ✅ Completed Infrastructure

1. **Database Schema Extended**
   - Added 5 new nullable columns to `stock_predictions` table:
     - `predicted_3day_return` (DOUBLE PRECISION)
     - `predicted_3day_direction` (BOOLEAN)
     - `actual_3day_return` (DOUBLE PRECISION)
     - `actual_3day_direction` (BOOLEAN)
     - `prediction_3day_correct` (BOOLEAN)
   - All columns are optional to maintain backward compatibility
   - Schema verified with: `psql -h 127.0.0.1 -U postgres -d research -c "\d stock_predictions"`

2. **Rust Code Updated**
   - **StockPrediction struct** (`src/db.rs`): Added 5 new Option<> fields for 3-day predictions
   - **save_prediction() method**: Updated signature to accept `predicted_3day_return: Option<f64>` and `predicted_3day_direction: Option<bool>`
   - **get_predictions_by_date()**: Updated to query and return all 3-day fields
   - **get_predictions_by_stock()**: Updated to query and return all 3-day fields
   - **batch_predict.rs**: Updated caller to pass `None` for 3-day values (model doesn't output these yet)
   - **Status**: ✅ All code compiles successfully

3. **Training Data Ready**
   - MlTrainingRecord struct already has: `next_3day_return` and `next_3day_direction` fields
   - These are computed during dataset creation in `dataset_creator.rs`
   - Training data in `ml_training_dataset` table includes 3-day targets

### 🔄 Partially Complete

- **3-Day Prediction Storage**: Ready to store values when model outputs them
- **3-Day Prediction Retrieval**: Ready to fetch and return values
- **batch_predict Binary**: Successfully rebuilt and ready to deploy

### ⏳ Still Needed

1. **Model Architecture Changes**
   - Update `src/model_torch.rs` to output dual predictions:
     - `next_day_return` (current output)
     - `next_3day_return` (new output)
   - Update loss function to combine both prediction tasks
   - Option: Use multitask learning with weighted combination

2. **Model Retraining**
   - Retrain with the updated architecture on `ml_training_dataset`
   - Command: `cargo run --release --features pytorch --bin train`
   - This produces new `artifacts/best_model.safetensors`

3. **batch_predict Updates for 3-Day Output**
   - Modify model inference in `batch_predict.rs` to extract 3-day predictions
   - Update `save_single_prediction()` to pass extracted 3-day values
   - Change from `None, None` to actual predicted values

## Architecture

### Data Flow

```
Training Phase:
┌─────────────────────────────────────┐
│ ml_training_dataset (PostgreSQL)    │
│ Contains next_day_return            │
│ Contains next_3day_return (target)  │
└────────────┬────────────────────────┘
             │
             ▼
┌──────────────────────────────┐
│ train binary                 │
│ Loads training data          │
│ Model predicts both 1-day    │
│ and 3-day returns            │
└────────────┬─────────────────┘
             │
             ▼
┌──────────────────────────────┐
│ artifacts/best_model.pt      │
│ Saved weights for both tasks │
└──────────────────────────────┘

Prediction Phase:
┌─────────────────────────────────────┐
│ ml_training_dataset (latest data)   │
└────────────┬────────────────────────┘
             │
             ▼
┌──────────────────────────────┐
│ batch_predict binary         │
│ Loads latest features        │
│ Runs model inference         │
│ Extracts 1-day AND 3-day     │
│ predictions                  │
└────────────┬─────────────────┘
             │
             ▼
┌──────────────────────────────┐
│ stock_predictions table      │
│ Save predicted_return        │
│ Save predicted_3day_return   │
│ (when actual outcomes known) │
│ Save actual_3day_return      │
│ Calculate prediction_3day_correct
└──────────────────────────────┘
```

## Code Changes Made

### 1. Database Schema (`stock_predictions` table)

```sql
-- New columns added (all nullable):
ALTER TABLE stock_predictions 
ADD COLUMN predicted_3day_return DOUBLE PRECISION,
ADD COLUMN predicted_3day_direction BOOLEAN,
ADD COLUMN actual_3day_return DOUBLE PRECISION,
ADD COLUMN actual_3day_direction BOOLEAN,
ADD COLUMN prediction_3day_correct BOOLEAN;
```

### 2. StockPrediction Struct

```rust
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StockPrediction {
    // Existing 1-day fields (10 fields)
    pub ts_code: String,
    pub trade_date: String,
    pub prediction_date: std::time::SystemTime,
    pub predicted_direction: bool,
    pub predicted_return: f64,
    pub confidence: f64,
    pub actual_direction: Option<bool>,
    pub actual_return: Option<f64>,
    pub prediction_correct: Option<bool>,
    pub model_version: Option<String>,
    
    // New 3-day fields (5 fields)
    pub predicted_3day_return: Option<f64>,
    pub predicted_3day_direction: Option<bool>,
    pub actual_3day_return: Option<f64>,
    pub actual_3day_direction: Option<bool>,
    pub prediction_3day_correct: Option<bool>,
}
```

### 3. save_prediction() Method Update

```rust
pub async fn save_prediction(
    &self,
    ts_code: &str,
    trade_date: &str,
    predicted_direction: bool,
    predicted_return: f64,
    confidence: f64,
    model_version: Option<&str>,
    predicted_3day_return: Option<f64>,      // NEW
    predicted_3day_direction: Option<bool>,   // NEW
) -> Result<(usize, usize)>
{
    // ... stores both 1-day and 3-day predictions
}
```

### 4. Query Updates

Both `get_predictions_by_date()` and `get_predictions_by_stock()` now SELECT:
```sql
SELECT ts_code, trade_date, prediction_date, 
       predicted_direction, predicted_return, confidence,
       actual_direction, actual_return, prediction_correct, 
       model_version,
       predicted_3day_return, predicted_3day_direction,      -- NEW
       actual_3day_return, actual_3day_direction,            -- NEW
       prediction_3day_correct                               -- NEW
FROM stock_predictions ...
```

### 5. batch_predict.rs Update

```rust
// Updated save_prediction call to include 3-day parameters:
match db_client
    .save_prediction(
        ts_code,
        &latest_date,
        predicted_direction,
        predicted_return,
        confidence,
        Some(model_version),
        None,  // predicted_3day_return - not yet supported by model
        None,  // predicted_3day_direction - not yet supported by model
    )
    .await
```

## Next Steps

### Phase 1: Verify Current Setup (DONE ✅)
- ✅ Database schema extended with 3-day columns
- ✅ Rust code updated for new fields
- ✅ batch_predict rebuilt successfully
- ✅ Code compiles without errors

### Phase 2: Update Model Architecture (TODO)
1. Review `src/model_torch.rs` to understand current architecture
2. Add dual-output prediction head for next_3day_return
3. Update loss function to weight both tasks:
   ```python
   # Pseudocode
   loss = mse_loss(predicted_return, actual_return) 
        + 0.5 * mse_loss(predicted_3day_return, actual_3day_return)
   ```

### Phase 3: Retrain Model (TODO)
```bash
cd /Users/alex/stock-analysis-workspace
cargo run --release --features pytorch --bin train

# New model checkpoint: artifacts/best_model.safetensors
# Will have dual outputs for 1-day and 3-day predictions
```

### Phase 4: Update batch_predict Inference (TODO)
1. Modify model inference to extract both outputs
2. Update save_single_prediction() to use 3-day values:
   ```rust
   match db_client
       .save_prediction(
           ts_code,
           &latest_date,
           predicted_direction,
           predicted_return,
           confidence,
           Some(model_version),
           Some(predicted_3day_return),      // From model output
           Some(predicted_3day_direction),   // From model output
       )
       .await
   ```

### Phase 5: Deploy & Validate (TODO)
```bash
# Rebuild batch_predict with updated inference code
cd /Users/alex/stock-analysis-workspace
cargo build --release --features pytorch --bin batch_predict

# Run batch_predict - will now save 3-day predictions
/Users/alex/stock-analysis-workspace/target/release/batch_predict \
    --concurrency 16 \
    --output-only-new

# Verify 3-day predictions are being stored:
psql -h 127.0.0.1 -U postgres -d research -c \
    "SELECT COUNT(*) as total, 
            COUNT(predicted_3day_return) as with_3day_pred 
     FROM stock_predictions 
     WHERE prediction_date > NOW() - INTERVAL '1 day';"
```

## Validation Queries

```sql
-- Check schema is complete:
\d stock_predictions

-- Verify column types:
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'stock_predictions' 
ORDER BY ordinal_position;

-- Count predictions by type:
SELECT 
    COUNT(*) as total_predictions,
    COUNT(predicted_return) as with_1day_pred,
    COUNT(predicted_3day_return) as with_3day_pred,
    COUNT(CASE WHEN actual_3day_return IS NOT NULL THEN 1 END) as with_actual_3day
FROM stock_predictions;

-- Compare 1-day vs 3-day prediction accuracy:
SELECT 
    SUM(CASE WHEN prediction_correct = true THEN 1 ELSE 0 END)::float / 
    COUNT(prediction_correct) as accuracy_1day,
    SUM(CASE WHEN prediction_3day_correct = true THEN 1 ELSE 0 END)::float / 
    COUNT(prediction_3day_correct) as accuracy_3day
FROM stock_predictions
WHERE prediction_date > NOW() - INTERVAL '30 days';
```

## Backward Compatibility

All changes are backward compatible:
- New columns are nullable (Option<> in Rust)
- Optional parameters with default None values
- Existing 1-day prediction pipeline unaffected
- Can gradually roll out 3-day predictions as model is updated

## Files Modified

1. `/Users/alex/stock-analysis-workspace/rust_llm_stock/src/db.rs`
   - StockPrediction struct (added 5 fields)
   - save_prediction() method (added 2 parameters)
   - get_predictions_by_date() (added 5 columns to SELECT)
   - get_predictions_by_stock() (added 5 columns to SELECT)

2. `/Users/alex/stock-analysis-workspace/rust_llm_stock/src/bin/batch_predict.rs`
   - Updated save_prediction() call to pass None for 3-day values

3. Database: `stock_predictions` table
   - Added 5 new nullable columns

## Build Status

- **Binary**: `/Users/alex/stock-analysis-workspace/target/release/batch_predict` (3.4 MB)
- **Compilation**: ✅ Success (warnings only about unused variables)
- **Ready to deploy**: ✅ Yes (currently saves None for 3-day values)

## Summary

The infrastructure for 3-day predictions is now **complete at the database and code level**. The system can:
- ✅ Store 3-day predictions
- ✅ Retrieve 3-day predictions
- ✅ Maintain historical accuracy metrics for 3-day forecasts

What remains is to **update the model to output 3-day predictions** instead of `None`. Once the model architecture is updated and retraining completes, batch_predict can immediately start producing 3-day forecasts.

---

**Created**: 2025-01-16
**Status**: Database + Code Infrastructure Complete | Model Update Pending
