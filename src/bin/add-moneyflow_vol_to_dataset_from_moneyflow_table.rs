use rust_llm_stock::stock_db::get_connection;
use sqlx::{Executor, Pool, Postgres, postgres::PgQueryResult};

#[tokio::main]
async fn main() {
    let pool = get_connection().await;

    let sql = r#"
        UPDATE ml_training_dataset t
        SET net_mf_vol = m.net_mf_vol
        FROM moneyflow m
        WHERE t.ts_code = m.ts_code
          AND t.trade_date = m.trade_date
    "#;

    let result: PgQueryResult = match (&pool).execute(sql).await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Update failed: {}", e);
            return;
        }
    };
    println!(
        "Updated {} rows in ml_training_dataset",
        result.rows_affected()
    );
}
