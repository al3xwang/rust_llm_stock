# Next Steps: Update Batch Predict for Dual Predictions

## Phase 4: Inference Update (batch_predict.rs)

Now that the model outputs dual predictions, batch_predict.rs needs to be updated to:
1. Extract both 1-day and 3-day predictions from model output
2. Extract confidence scores for both horizons
3. Pass all values to the database save_prediction call

### Changes Required in batch_predict.rs

#### 1. Update Prediction Extraction

**Current Code:**
```rust
let (predictions, confidences) = model.forward_with_confidence(&input_tensor, false)?;
// Process only 1-day predictions
```

**Updated Code:**
```rust
let (pred_1day, pred_3day, conf_1day, conf_3day) = model.forward_dual(&input_tensor, false)?;
// Now have access to both 1-day and 3-day predictions with confidence scores
```

#### 2. Update Save Prediction Calls

**Current Code:**
```rust
db_client.save_prediction(
    &stock.ts_code,
    &trade_date,
    predicted_direction,
    predicted_return,
    confidence,
    Some(&model_version),
    None,                    // 3-day return not filled
    None,                    // 3-day direction not filled
).await?;
```

**Updated Code:**
```rust
db_client.save_prediction(
    &stock.ts_code,
    &trade_date,
    predicted_direction_1day,
    predicted_return_1day,
    confidence_1day,
    Some(&model_version),
    Some(predicted_return_3day),      // NEW: 3-day return
    Some(predicted_direction_3day),   // NEW: 3-day direction
).await?;
```

#### 3. Extract Values from Output Tensors

**For 1-Day Predictions:**
```rust
// Extract 1-day predictions (last timestep, feature index for return)
let pred_1day_tensor = &pred_1day[pred_1day.size()[0] - 1];  // Last timestep
let predicted_return_1day = pred_1day_tensor[12].double_value(&[]); // Feature 12 is close_pct
let predicted_direction_1day = if predicted_return_1day > 0.002 { 1 } else if predicted_return_1day < -0.002 { -1 } else { 0 };
let confidence_1day = conf_1day[conf_1day.size()[0] - 1].double_value(&[]);
```

**For 3-Day Predictions:**
```rust
// Extract 3-day predictions (last timestep, feature index for return)
let pred_3day_tensor = &pred_3day[pred_3day.size()[0] - 1];  // Last timestep
let predicted_return_3day = pred_3day_tensor[12].double_value(&[]); // Feature 12 is close_pct
let predicted_direction_3day = if predicted_return_3day > 0.005 { 1 } else if predicted_return_3day < -0.005 { -1 } else { 0 };
let confidence_3day = conf_3day[conf_3day.size()[0] - 1].double_value(&[]);
```

### Implementation Sequence

1. **Modify model loading** (if needed)
   ```rust
   let model = TorchStockModel::new(&model_path)?;
   // No changes needed - forward_dual() is already implemented
   ```

2. **Update inference loop**
   ```rust
   for stock in stocks.iter() {
       for date in dates.iter() {
           let (pred_1day, pred_3day, conf_1day, conf_3day) = 
               model.forward_dual(&input_tensor, false)?;
           
           // Extract values for both horizons
           let return_1day = extract_return(&pred_1day);
           let return_3day = extract_return(&pred_3day);
           
           // Save with all 4 predictions
           db_client.save_prediction(
               ts_code, date,
               direction_1day, return_1day, conf_1day,
               Some(return_3day), Some(direction_3day)
           ).await?;
       }
   }
   ```

3. **Rebuild binary**
   ```bash
   cargo build --release --features pytorch --bin batch_predict
   ```

4. **Test with single stock**
   ```bash
   # Test predictions for one stock
   ./target/release/batch_predict --stock 000001.SZ --date 20250101
   # Check stock_predictions table for both 1-day and 3-day columns
   ```

### Verification Checklist

- [ ] Compile batch_predict.rs with new model.forward_dual() calls
- [ ] Test prediction extraction for both 1-day and 3-day
- [ ] Verify database receives both predicted_3day_return and predicted_3day_direction
- [ ] Check confidence scores are reasonable (0-1 range)
- [ ] Monitor for NaN or invalid values in predictions
- [ ] Run sample predictions on test date
- [ ] Compare 1-day predictions with previous model for consistency

### Database Verification

After deployment, verify data is being saved:

```sql
-- Check 3-day predictions are populated
SELECT 
    ts_code, 
    trade_date,
    predicted_return,
    predicted_3day_return,
    confidence,
    COUNT(*) OVER (PARTITION BY ts_code, trade_date) as pred_count
FROM stock_predictions
WHERE trade_date = '20250103'
ORDER BY ts_code
LIMIT 20;

-- Check both predictions are reasonable
SELECT 
    COUNT(*) as total_rows,
    COUNT(predicted_3day_return) as filled_3day,
    AVG(ABS(predicted_return)) as avg_1day_return,
    AVG(ABS(predicted_3day_return)) as avg_3day_return
FROM stock_predictions
WHERE trade_date >= '20250101'
AND trade_date <= '20250105';
```

## Timeline

**Phase Completion Order:**
1. ✅ Phase 1: Database schema - COMPLETED
2. ✅ Phase 2: Code structure (StockPrediction struct) - COMPLETED
3. ✅ Phase 3: Model dual-output architecture - **JUST COMPLETED**
4. ⏳ Phase 4: Update batch_predict inference - READY TO START
5. ⏳ Phase 5: Model retraining - READY TO START
6. ⏳ Phase 6: Deployment and validation - AFTER TRAINING

## Commands to Execute

```bash
# Step 1: Verify compilation
cd /Users/alex/stock-analysis-workspace/rust_llm_stock
cargo build --release --features pytorch

# Step 2: Retrain model with dual predictions
cargo run --release --features pytorch --bin train

# Step 3: Monitor training progress
tail -f logs/training.log

# Step 4: Once trained, update batch_predict.rs
# (Update source files as described above)

# Step 5: Rebuild batch_predict binary
cargo build --release --features pytorch --bin batch_predict

# Step 6: Run batch predictions
./target/release/batch_predict --date 20250103
```

## Troubleshooting

### Issue: Model not loading
```rust
// Ensure model path is correct
let model = TorchStockModel::from_checkpoint("artifacts/best_model.safetensors")?;
```

### Issue: Tensor shape mismatch
```rust
// Verify input shape: [batch_size, seq_len, features]
assert_eq!(input_tensor.size(), vec![1, 30, 105]);

// Verify output shape: [batch_size, seq_len, features]
assert_eq!(pred_1day.size(), vec![1, 30, 105]);
assert_eq!(pred_3day.size(), vec![1, 30, 105]);
```

### Issue: NaN in predictions
```rust
// Check for NaN values
if pred_1day.isnan().any().int64_value(&[]) != 0 {
    eprintln!("Warning: NaN in 1-day predictions");
}

// Clamp extreme values
let pred_1day = pred_1day.clamp(-0.5, 0.5);  // Reasonable return bounds
```

## Success Criteria

✅ Model outputs both 1-day and 3-day predictions  
✅ Confidence scores are in [0, 1] range  
✅ Database receives all 6 prediction columns  
✅ No NaN or infinite values in predictions  
✅ 3-day predictions are different from 1-day (not identical)  
✅ Inference completes in <100ms per stock  
