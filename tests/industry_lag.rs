use rust_llm_stock::stock_db::get_connection;
use sqlx::Row;

#[tokio::test]
async fn test_industry_avg_is_prior_day() {
    let pool = get_connection().await.expect("DB connection failed");

    // Sample up to 50 rows with an industry assigned
    let rows = sqlx::query!(
        r#"
        SELECT ts_code, trade_date, industry, industry_avg_return
        FROM ml_training_dataset
        WHERE industry IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 50
        "#,
    )
    .fetch_all(&pool)
    .await
    .expect("Query failed");

    assert!(!rows.is_empty(), "No sample rows found in ml_training_dataset");

    for r in rows {
        let ts_code = r.ts_code;
        let trade_date = r.trade_date;
        let industry = r.industry;
        let observed = r.industry_avg_return;

        // Find most recent prior industry average (date < trade_date)
        let expected_row = sqlx::query!(
            r#"
            SELECT avg_return FROM (
                SELECT sd.trade_date, AVG(sd.pct_chg::DOUBLE PRECISION) AS avg_return
                FROM stock_daily sd
                JOIN stock_basic sb ON sd.ts_code = sb.ts_code
                WHERE sb.industry = $1 AND sd.trade_date < $2
                GROUP BY sd.trade_date
                ORDER BY sd.trade_date DESC
                LIMIT 1
            ) t
            "#,
            industry,
            trade_date
        )
        .fetch_optional(&pool)
        .await
        .expect("Expected query failed");

        match expected_row {
            Some(er) => {
                let expected = er.avg_return;
                // expected is f64; observed is Option<f64>
                assert!(observed.is_some(), "Observed industry_avg_return is NULL for {} {} but expected {}", ts_code, trade_date, expected);
                let obs_v = observed.unwrap();
                let diff = (obs_v - expected).abs();
                assert!(diff <= 1e-8, "Mismatch for {} {}: observed {} vs expected {} (diff {})", ts_code, trade_date, obs_v, expected, diff);
            }
            None => {
                // No prior industry avg exists (very first date for industry). We expect observed to be NULL.
                assert!(observed.is_none(), "Expected NULL industry_avg_return for {} {} (no prior industry data), but got {:?}", ts_code, trade_date, observed);
            }
        }

        // --- New: check industry_momentum_5d matches prior-day industry 5-day rolling avg ---
        let observed_mom = sqlx::query!(
            r#"SELECT industry_momentum_5d FROM ml_training_dataset WHERE ts_code = $1 AND trade_date = $2"#,
            ts_code,
            trade_date
        )
        .fetch_one(&pool)
        .await
        .expect("Failed to fetch observed momentum");

        // Find expected prior-day avg_5d from industry daily aggregation (rows date < trade_date)
        let expected_mom_row = sqlx::query!(
            r#"
            SELECT avg_5d FROM (
                SELECT sd.trade_date,
                    AVG(sd.pct_chg::DOUBLE PRECISION) AS avg_return,
                    AVG(AVG(sd.pct_chg::DOUBLE PRECISION)) OVER (PARTITION BY sb.industry ORDER BY sd.trade_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS avg_5d
                FROM stock_daily sd
                JOIN stock_basic sb ON sd.ts_code = sb.ts_code
                WHERE sb.industry = $1 AND sd.trade_date < $2
                GROUP BY sd.trade_date, sb.industry
                ORDER BY sd.trade_date DESC
                LIMIT 1
            ) t
            "#,
            industry,
            trade_date
        )
        .fetch_optional(&pool)
        .await
        .expect("Expected momentum query failed");

        match expected_mom_row {
            Some(em) => {
                let expected_mom = em.avg_5d;
                let obs_m = observed_mom.industry_momentum_5d;
                assert!(obs_m.is_some(), "Observed industry_momentum_5d is NULL for {} {} but expected {}", ts_code, trade_date, expected_mom);
                let diff = (obs_m.unwrap() - expected_mom).abs();
                assert!(diff <= 1e-8, "Momentum mismatch for {} {}: observed {} vs expected {} (diff {})", ts_code, trade_date, obs_m.unwrap(), expected_mom, diff);
            }
            None => {
                // No prior industry row: expect NULL
                let obs_m = observed_mom.industry_momentum_5d;
                assert!(obs_m.is_none(), "Expected NULL industry_momentum_5d for {} {} (no prior industry data), but got {:?}", ts_code, trade_date, obs_m);
            }
        }
    }
}
