#!/bin/bash
# Quick Reference: Daily Trading Pipeline Commands

echo "
╔═══════════════════════════════════════════════════════════════════╗
║              Daily Trading Pipeline - Quick Reference             ║
╚═══════════════════════════════════════════════════════════════════╝

📌 MOST COMMON COMMANDS:

🚀 Generate trading signals for TODAY (fast):
   ./daily_pipeline.sh --skip-ingest

🔄 Full pipeline with fresh data (daily):
   ./daily_pipeline.sh

📊 Analyze historical date:
   ./daily_pipeline.sh 20251225

⚡ Just refresh features (no ingestion):
   ./run_full_pipeline_llm.sh

📈 View today's trading signals:
   cat daily_report_\$(date +%Y%m%d).csv

═══════════════════════════════════════════════════════════════════

⏱️  TIMING GUIDE:

Quick signals only:       <1 second   ← Use for daily trading
Full pipeline:            10-15 min   ← Use once daily, overnight
Historical analysis:      <5 sec      ← Use for backtesting

═══════════════════════════════════════════════════════════════════

📁 OUTPUT FILES:

Daily reports:
   daily_report_YYYYMMDD.csv        ← Your trading signals!

Features/Data:
   data/training_data.csv           ← 1.1M stock records

Logs:
   pipeline_YYYYMMDD.log            ← Ingestion details

═══════════════════════════════════════════════════════════════════

🎯 INTERPRETATION:

   ts_code              = Stock symbol
   next_day_return      = Expected return % (★ KEY SIGNAL)
   rsi_14              = Momentum (30=oversold, 70=overbought)
   bb_bandwidth        = Volatility (wider = more risk/reward)
   ema_5               = Short-term trend

═══════════════════════════════════════════════════════════════════

⚙️  SETUP (One-time):

crontab -e
# Add: 0 4 * * * cd /path/to/rust_llm_stock && ./daily_pipeline.sh

═══════════════════════════════════════════════════════════════════

📖 FULL GUIDE: cat DAILY_PIPELINE_README.md
📝 VIEW SIGNALS: cat daily_report_\$(date +%Y%m%d).csv
📊 DATABASE STATUS: psql \$DATABASE_URL -c 'SELECT COUNT(*) FROM ml_training_dataset;'

═══════════════════════════════════════════════════════════════════
"
