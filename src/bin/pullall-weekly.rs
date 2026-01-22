use std::{
    collections::HashMap, error::Error, fmt::Debug, num::NonZeroU32, thread, time::Duration,
};

use chrono::{Datelike, Days, Local, NaiveDate, Weekday};
use ratelimit_meter::{DirectRateLimiter, GCRA};
use rust_llm_stock::{
    stock_db::{create_req, get_connection},
    ts::{
        http::{Responsex, TradeRecord},
        model::{StockBasic, StockKey},
    },
};
use sqlx::{Postgres, QueryBuilder, postgres::PgPoolOptions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut limiter = DirectRateLimiter::<GCRA>::per_second(NonZeroU32::new(3).unwrap());

    // Get database connection
    let dbpool = get_connection().await;

    // Query all ts_code from stock_basic (only listed stocks)
    let stock_list: Vec<String> =
        sqlx::query_scalar("SELECT ts_code FROM stock_basic WHERE list_status = 'L'")
            .fetch_all(&dbpool)
            .await?;

    // Default starting date for weekly data (week of 2020-12-14)
    let default_start = "20201214".to_string();

    for ts_code in stock_list {
        loop {
            if limiter.check().is_ok() {
                break;
            } else {
                thread::sleep(Duration::from_millis(100));
            }
        }
        pull_insert(&dbpool, ts_code, default_start.clone()).await?;
    }

    Ok(())
}

pub async fn pull_insert(
    dbpool: &sqlx::PgPool,
    ts_code: String,
    date: String,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let client: reqwest::Client = reqwest::Client::new();
    println!(
        "start pulling weekly data for {} from {} ...",
        ts_code, &date
    );
    let mut params = HashMap::new();
    params.insert("ts_code".to_string(), ts_code.clone());
    params.insert("start_date".to_string(), date.clone());

    let response = client
        .post("http://api.tushare.pro")
        .body(create_req("weekly".to_string(), params))
        .send()
        .await?;

    println!("Status: {}", response.status());

    let response_body = response.text().await?;
    println!("Response body:\n{}", response_body);

    let repx: Responsex = serde_json::from_str(&response_body)?;
    println!("API returned {} items", repx.data.items.len());

    if repx.data.items.is_empty() {
        println!("No data returned for {}", ts_code);
        return Ok(());
    }

    // Use a transaction for batch insert
    let mut tx = dbpool.begin().await?;
    let mut inserted_count = 0;
    let mut skipped_count = 0;

    for item in &repx.data.items {
        let ts_code = item[0].as_str();
        let trade_date = item[1].as_str();
        let open = item[2].as_f64();
        let high = item[3].as_f64();
        let low = item[4].as_f64();
        let close = item[5].as_f64();
        let pre_close = item[6].as_f64();
        let change = item[7].as_f64();
        let pct_chg = item[8].as_f64();
        let vol = item[9].as_f64();
        let amount = item[10].as_f64();

        let query_result = sqlx::query(
            "INSERT INTO weekly (ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount)
            values ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
            ON CONFLICT (ts_code, trade_date) DO NOTHING")
            .bind(ts_code)
            .bind(trade_date)
            .bind(open)
            .bind(high)
            .bind(low)
            .bind(close)
            .bind(pre_close)
            .bind(change)
            .bind(pct_chg)
            .bind(vol)
            .bind(amount)
            .execute(&mut *tx)
            .await?;

        if query_result.rows_affected() > 0 {
            inserted_count += 1;
        } else {
            skipped_count += 1;
        }
    }

    tx.commit().await?;
    println!(
        "Completed for {}: inserted={}, skipped={}",
        ts_code, inserted_count, skipped_count
    );

    Ok(())
}

// 处理行业板块日交易数据
// select  industry as ts_code, trade_date, ROUND(AVG(open)::numeric, 2) as open, ROUND(AVG(high)::numeric, 2) AS high,ROUND(AVG(low)::numeric, 2) AS low,ROUND(AVG(close)::numeric, 2) AS close, ROUND(sum(vol)::numeric, 2) AS vol,
// ROUND(sum(amount)::numeric, 2) as amount, ROUND(AVG(change)::numeric, 2) as change, ROUND(AVG(pct_chg)::numeric, 2) as pct_chg, ROUND(AVG(pre_close)::numeric, 2) as pre_close
// INTO public.industry
// from stock_basic b, daily d where
// b.ts_code=d.ts_code and name not like 'ST%'
// group by trade_date,industry order by trade_date

// 处理市场板块日交易数据
// select  substr(b.ts_code,0,4) as ts_code, trade_date, AVG(open) as open, AVG(high) AS high,AVG(low) AS low,AVG(close) AS close, sum(vol) AS vol,
// sum(amount) as amount, AVG(change) as change, AVG(pct_chg) as pct_chg, AVG(pre_close) as pre_close
// INTO public.market
// from stock_basic b, daily d where
// b.ts_code=d.ts_code and name not like 'ST%'
// group by substr(b.ts_code,0,4), trade_date order by trade_date
