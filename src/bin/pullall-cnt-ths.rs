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
    dotenv::dotenv().ok();

    let mut limiter = DirectRateLimiter::<GCRA>::per_second(NonZeroU32::new(3).unwrap());

    println!("Connecting to database...");
    let dbpool = get_connection().await;
    println!("Connected to database");

    // --- Accept optional start_date from command line ---
    let args: Vec<String> = std::env::args().collect();
    let user_start_date = if args.len() > 1 {
        Some(args[1].clone())
    } else {
        None
    };

    // Get the latest date from moneyflow_cnt_ths table, or use default start date
    let latest_date: Option<String> = match sqlx::query_scalar::<_, Option<String>>(
        "SELECT MAX(trade_date) FROM moneyflow_cnt_ths WHERE trade_date ~ '^[0-9]{8}$'",
    )
    .fetch_optional(&dbpool)
    .await
    {
        Ok(result) => result.flatten(),
        Err(e) => {
            eprintln!("Database query error: {}", e);
            None
        }
    };

    let start_date = if let Some(user_date) = user_start_date {
        user_date
    } else if let Some(latest) = latest_date {
        // Start from the day after the latest date
        let date = NaiveDate::parse_from_str(&latest, "%Y%m%d")?;
        let next_day = date + chrono::Duration::days(1);
        next_day.format("%Y%m%d").to_string()
    } else {
        // No data exists, use default start date
        "20150101".to_string()
    };

    let today = Local::now().date_naive();
    let end_date = today.format("%Y%m%d").to_string();

    println!(
        "Pulling moneyflow_cnt_ths data from {} to {}",
        start_date, end_date
    );

    let mut current = NaiveDate::parse_from_str(&start_date, "%Y%m%d")?;

    if current > today {
        println!("No new data to pull (already up to date)");
        return Ok(());
    }

    let total_months =
        (today.year() - current.year()) * 12 + (today.month() as i32 - current.month() as i32) + 1;
    println!(
        "Processing approximately {} months of data",
        total_months.max(1)
    );

    let mut processed_months = 0;

    let mut processed_months = 0;

    while current <= today {
        // Wait for rate limiter
        while limiter.check().is_err() {
            thread::sleep(Duration::from_millis(100));
        }

        processed_months += 1;

        let start_date_month = current.format("%Y%m%d").to_string();
        // Calculate end of month or today, whichever is earlier
        let mut next_month = current.with_day(1).unwrap().succ_opt().unwrap();
        next_month = if current.month() == 12 {
            NaiveDate::from_ymd_opt(current.year() + 1, 1, 1).unwrap()
        } else {
            NaiveDate::from_ymd_opt(current.year(), current.month() + 1, 1).unwrap()
        };
        let end_date_month = std::cmp::min(next_month.pred_opt().unwrap(), today);
        let end_date_str = end_date_month.format("%Y%m%d").to_string();

        println!(
            "[{}/{}] Pulling moneyflow_cnt_ths from {} to {}",
            processed_months,
            total_months.max(1),
            start_date_month,
            end_date_str
        );
        pull_insert_range(&dbpool, start_date_month.clone(), end_date_str.clone()).await?;

        // Move to next month
        current = next_month;
    }

    println!("\n=== Completed ===");
    println!(
        "Processed {} months from {} to {}",
        processed_months,
        start_date,
        today.format("%Y%m%d")
    );

    Ok(())
}

// Pull and insert for a date range (month)
pub async fn pull_insert_range(
    dbpool: &sqlx::PgPool,
    start_date: String,
    end_date: String,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let client: reqwest::Client = reqwest::Client::new();
    let mut params = HashMap::new();
    params.insert("start_date".to_string(), start_date.clone());
    params.insert("end_date".to_string(), end_date.clone());

    println!(
        "start pulling moneyflow_cnt_ths data from {} to {} ...",
        start_date, end_date
    );

    let response = client
        .post("http://api.tushare.pro")
        .body(create_req("moneyflow_cnt_ths".to_string(), params))
        .send()
        .await?;

    println!("Status: {}", response.status());

    let response_body = response.text().await?;
    println!("Response body:\n{}", response_body);

    let repx: Responsex = serde_json::from_str(&response_body)?;
    println!("API returned {} items", repx.data.items.len());

    if repx.data.items.is_empty() {
        println!("No data returned for range {} - {}", start_date, end_date);
        return Ok(());
    }

    // Use a transaction for batch insert
    let mut tx = dbpool.begin().await?;
    let mut inserted_count = 0;
    let mut skipped_count = 0;
    let mut null_ts_code_count = 0;
    let mut looked_up_count = 0;

    for item in &repx.data.items {
        let trade_date = item[0].as_str();
        let mut ts_code = item[1].as_str();
        let name = item[2].as_str();
        let lead_stock = item[3].as_str();

        // If ts_code is null or empty, look it up from stock_basic table using lead_stock name
        let ts_code_value: String;
        if ts_code.is_none() || ts_code.unwrap_or("").is_empty() {
            if let Some(stock_name) = lead_stock {
                // Query stock_basic table to find ts_code by name (only listed stocks)
                let lookup_result: Option<(String,)> = sqlx::query_as(
                    "SELECT ts_code FROM stock_basic WHERE name = $1 AND list_status = 'L' LIMIT 1",
                )
                .bind(stock_name)
                .fetch_optional(&mut *tx)
                .await?;

                if let Some((found_ts_code,)) = lookup_result {
                    ts_code_value = found_ts_code;
                    looked_up_count += 1;
                } else {
                    // Still no ts_code found, skip this record
                    null_ts_code_count += 1;
                    continue;
                }
            } else {
                null_ts_code_count += 1;
                continue;
            }
        } else {
            ts_code_value = ts_code.unwrap().to_string();
        }

        let close_price = item[4].as_f64();
        let pct_change = item[5].as_f64();
        let industry_index = item[6].as_f64();
        let company_num = item[7].as_i64().map(|v| v as i32);
        let pct_change_stock = item[8].as_f64();
        let net_buy_amount = item[9].as_f64();
        let net_sell_amount = item[10].as_f64();
        let net_amount = item[11].as_f64();

        let query_result = sqlx::query(
            "INSERT INTO moneyflow_cnt_ths (trade_date,ts_code,name,lead_stock,close_price,pct_change,industry_index,company_num,pct_change_stock,net_buy_amount,net_sell_amount,net_amount)
            values ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
            ON CONFLICT (trade_date, ts_code) DO NOTHING")
            .bind(trade_date)
            .bind(&ts_code_value)
            .bind(name)
            .bind(lead_stock)
            .bind(close_price)
            .bind(pct_change)
            .bind(industry_index)
            .bind(company_num)
            .bind(pct_change_stock)
            .bind(net_buy_amount)
            .bind(net_sell_amount)
            .bind(net_amount)
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
        "Completed for range {} - {}: inserted={}, skipped={}, looked_up={}, null_ts_code={}",
        start_date, end_date, inserted_count, skipped_count, looked_up_count, null_ts_code_count
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
