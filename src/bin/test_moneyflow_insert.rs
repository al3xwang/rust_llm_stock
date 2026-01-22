fn main() {}
use rust_llm_stock::stock_db::get_connection;
use sqlx::Row;

#[tokio::test]
async fn test_moneyflow_features_inserted() {
    let pool = get_connection().await.expect("DB connection failed");
    // Pick a recent date and a known stock code for the test
    let ts_code = "000001.SZ";
    let row = sqlx::query(
        r#"SELECT net_mf_vol, net_mf_amount, smart_money_ratio, large_order_flow FROM ml_training_dataset WHERE ts_code = $1 ORDER BY trade_date DESC LIMIT 1"#
    )
    .bind(ts_code)
    .fetch_one(&pool)
    .await
    .expect("No row found for test stock");

    // Assert at least one of the moneyflow features is not null
    let has_some = row
        .try_get::<Option<f64>, _>("net_mf_vol")
        .unwrap()
        .is_some()
        || row
            .try_get::<Option<f64>, _>("net_mf_amount")
            .unwrap()
            .is_some()
        || row
            .try_get::<Option<f64>, _>("smart_money_ratio")
            .unwrap()
            .is_some()
        || row
            .try_get::<Option<f64>, _>("large_order_flow")
            .unwrap()
            .is_some();
    assert!(has_some, "All moneyflow features are null for {}", ts_code);
}
