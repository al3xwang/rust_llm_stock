# LightGBM Binary Classification Training Results

## Data Summary
- **Market**: KC (Shanghai & Shenzhen Tech)
- **Date Range**: 2022-01-04 to 2025-12-31
- **Target**: Binary classification (-1=down, 1=up)
- **Training Set**: 139,400 rows (60%, dates 20220104-20240531)
- **Validation Set**: 46,467 rows (20%, dates 20240531-20250319)
- **Test Set**: 46,465 rows (20%, dates 20250319-20251231)

## Features
- Total Features: 105
  - Numeric Features: 71
  - Categorical Features: 2  
  - Embedding Features: 32

## Model Configuration
- **Framework**: LightGBM (Gradient Boosting Decision Trees)
- **Objective**: Binary Classification
- **Boosting Type**: GBDT
- **Learning Rate**: 0.02
- **Max Leaves**: 25
- **Max Depth**: 7
- **Early Stopping**: Yes (patience=30)
- **Best Iteration**: 6 (stopped early)

## Performance Results

### Validation Set
- **Accuracy**: 49.74%
- **Precision**: 87.18%
- **Recall**: 0.44%
- **F1-Score**: 0.87%
- **AUC-ROC**: 0.5205- **AUC-ROC**: 0.5205- **AUC-ROC**: 0.5205- **AUC-ROC**: 0.52
-----------------------------------------------UC-RO-----------

##########################################: 0.7###################################**Test A##########################################: 0.7##   ##########################################: 0.7###############Down     22,610    38
Actual Up       23,378    439
```

## Analysi## Analysi## Analysi## Analysi## Analysi## Analysi## Analg AUC (0.7115) → Validation AUC (0.5205)
   - Model memorize   - Model memorize   - Modsn'   - Model memorize   Class    - Model memorize  on   - Model memorize   - Model memorize   - Modsn'   - Mod  -    - Model memorize   - Model memoriz (   - Model memorize  7 ups detected)
   
3. **Poor G3. **Poor G3. **P   -3. **Poor G3.0.5474 barely beats random chance (0.5)
   - Essentially non-predictive model

### Root Causes
- Stock price mo- Stock price mo-en- Stock price difficult- Stock price mo-ture engineering may not ca- Stock price mo- Stna- Stock price mo- Stock price mo-en- Stock price difficult- Stock price mo-turleakage or look-ahead bias
- Market-specific factors not captured in f- Market-specific factors not captured in f- Market-specific factors not captured in f- Market-specific factors not captured in f- Market-specific factors not captured in f- Market-specific factors not captured in f- Marketent

1. **Feature Engineering**
   - A   - A   - A   - A   - A   - A   - A   - A   - A   - A   - A   - A   - A   - A   - A   - A   - A   - A   - Ay    - A   - A   - A   - A   - A   - A   - A   - A   - A   - A   - A   - A   - A   - A   - A   - A   - A   - A   - Ay    - A   - A   - A   - A   - A   - A   - A   - A   - A   - A   - A   - A   - A   - A   - A   - A   - A   - A   - Ay   k-ahead   - A   - A   - A   - A   - A   - A   - A   - A   - A   - A   - A   - A   - A   -ide   - A   - A   - A   - A   - A ative Approaches**
   -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    - with stronger regularization to reduce overfitting
3. Try different hyperparameter combinations
4. Consider ensemble methods or alternative architectures
