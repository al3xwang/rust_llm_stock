-- Migration: Create stock_predictions table
-- Description: Store batch predictions for all stocks in ml_training_dataset
-- Date: 2025-12-27

CREATE TABLE IF NOT EXISTS stock_predictions (
    -- Primary identifiers
    ts_code VARCHAR(20) NOT NULL,
    trade_date VARCHAR(8) NOT NULL,
    prediction_date TIMESTAMP NOT NULL DEFAULT NOW(),
    
    -- Prediction results
    predicted_direction BOOLEAN NOT NULL,  -- TRUE = up, FALSE = down
    predicted_return DOUBLE PRECISION NOT NULL,  -- Predicted next day return (%)
    confidence DOUBLE PRECISION NOT NULL,  -- Model confidence (0-1)
    
    -- Actual results (to be filled later for backtesting)
    actual_direction BOOLEAN,
    actual_return DOUBLE PRECISION,
    prediction_correct BOOLEAN,
    
    -- Model info
    model_version VARCHAR(50),
    
    -- Constraints
    PRIMARY KEY (ts_code, trade_date, prediction_date)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_stock_predictions_ts_code ON stock_predictions(ts_code);
CREATE INDEX IF NOT EXISTS idx_stock_predictions_trade_date ON stock_predictions(trade_date);
CREATE INDEX IF NOT EXISTS idx_stock_predictions_prediction_date ON stock_predictions(prediction_date);
CREATE INDEX IF NOT EXISTS idx_stock_predictions_confidence ON stock_predictions(confidence);

-- Add comments
COMMENT ON TABLE stock_predictions IS 'Batch predictions for all stocks with confidence scores';
COMMENT ON COLUMN stock_predictions.ts_code IS 'Stock ticker symbol (e.g., 000001.SZ)';
COMMENT ON COLUMN stock_predictions.trade_date IS 'Trading date for which prediction is made (YYYYMMDD format)';
COMMENT ON COLUMN stock_predictions.prediction_date IS 'When the prediction was made';
COMMENT ON COLUMN stock_predictions.predicted_direction IS 'Predicted movement direction (TRUE=up, FALSE=down)';
COMMENT ON COLUMN stock_predictions.predicted_return IS 'Predicted next day return percentage';
COMMENT ON COLUMN stock_predictions.confidence IS 'Model confidence score (0-1, higher is more confident)';
COMMENT ON COLUMN stock_predictions.actual_direction IS 'Actual movement direction (filled after market close)';
COMMENT ON COLUMN stock_predictions.actual_return IS 'Actual next day return (filled after market close)';
COMMENT ON COLUMN stock_predictions.prediction_correct IS 'Whether prediction was correct (filled after validation)';
