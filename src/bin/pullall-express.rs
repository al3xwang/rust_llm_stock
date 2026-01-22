use std::{ collections::HashMap, error::Error, num::NonZeroU32, thread, time::Duration };

use chrono::{ Datelike, Days, Local, NaiveDate };
use sqlx::{ Postgres, QueryBuilder };
use rust_llm_stock::{
    stock_db::{ create_req, get_connection },
    ts::{ http::{ Forecast, Rep, Responsex }, model::{ StockBasic, StockKey } },
};
use ratelimit_meter::{ DirectRateLimiter, GCRA };

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let pool = get_connection().await;

    //初始化数据库连接池：
    // let dbpool =  pool.clone() ;
    // Define the rate limit parameters (e.g., 200 requests per minute)

    // Then, for each incoming HTTP request, you can check if it's within the rate limit

    //select list from db
    let init_list_query =
        "SELECT * FROM stock_basic
    WHERE not exists(select 1 from express where ts_code=stock_basic.ts_code and end_date< '20231230')  ";
    let init_list = sqlx::query_as::<_, StockBasic>(init_list_query).fetch_all(&pool).await;

    //select list from db
    let curr_date = Local::now();
    let curr_date = format!(
        "{:04}{:02}{:02}",
        curr_date.year(),
        curr_date.month(),
        curr_date.day()
    );

    let forecast_update_query = format!(
        "select * from (select distinct ts_code, max(end_date) as trade_date from express group by ts_code ) tmp where  trade_code < '{}' ",
        "20231230"
    );

    println!("sql is {}", forecast_update_query);
    let update_stock_list = sqlx
        ::query_as::<_, StockKey>(&forecast_update_query)
        .fetch_all(&pool).await;

    let mut limiter = DirectRateLimiter::<GCRA>::per_second(NonZeroU32::new(3).unwrap()); // Allow 3 units per second

    match init_list {
        Ok(list) => {
            let mut handles: Vec<
                tokio::task::JoinHandle<Result<(), Box<dyn Error + Sync + Send>>>
            > = vec![];
            // let mut limiter = DirectRateLimiter::<GCRA>::per_second(NonZeroU32::new(3).unwrap()); // Allow 3 units per second
            // let since = Instant::now();
            // let mut request_counter = 0;
            for s in list {
                loop {
                    if limiter.check().is_ok() {
                        break;
                    } else {
                        thread::sleep(Duration::from_millis(100));
                    }
                }
                let handle = tokio::spawn(async move { pull_forecast(s.ts_code).await });

                handles.push(handle);
            }

            for handle in handles {
                let _ = handle.await;
            }
        }
        Err(_e) => {}
    }

    match update_stock_list {
        Ok(list) => {
            let mut handles: Vec<
                tokio::task::JoinHandle<Result<(), Box<dyn Error + Sync + Send>>>
            > = vec![];
            // let mut limiter = DirectRateLimiter::<GCRA>::per_second(NonZeroU32::new(3).unwrap()); // Allow 3 units per second
            // let since = Instant::now();
            // let mut request_counter = 0;

            for s in list {
                loop {
                    if limiter.check().is_ok() {
                        break;
                    } else {
                        thread::sleep(Duration::from_millis(100));
                    }
                }

                // Parse the string into a Date value
                let parsed_date = NaiveDate::parse_from_str(
                    &s.trade_date.unwrap(),
                    "%Y%m%d"
                ).unwrap();

                // Add 1 day to the parsed date
                let next_day = parsed_date.checked_add_days(Days::new(1)).unwrap();
                let date_str = next_day.format("%Y%m%d").to_string();
                // Convert the result back to a string in 'YYYYMMDD' format
                // let handle = tokio::spawn(async move { pull_forecast(s.ts_code).await });
                // handles.push(handle);
                let handle = tokio::spawn(async move { pull_forecast(s.ts_code).await });
                handles.push(handle);
            }
            for handle in handles {
                let _ = handle.await;
            }
        }
        Err(_e) => {}
    }

    Ok(())
}

pub async fn pull_forecast(
    code: String
    // date: String
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let client: reqwest::Client = reqwest::Client::new();
    println!("start pulling forecast...");
    let mut params = HashMap::new();
    params.insert(String::from("ts_code".to_string()), String::from(code));

    let response = client
        .post("http://api.tushare.pro")
        .body(create_req("express".to_string(), params))
        .send().await?;

    println!("Status: {}", response.status());

    let response_body = response.text().await?;
    println!("Response body:\n{}", response_body);

    let repx: Responsex = serde_json::from_str(&response_body)?;

    let dbpool = get_connection().await;

    for item in &repx.data.items {
        let ts_code = item[0].as_str();
        let ann_date = item[1].as_str();
        let end_date = item[2].as_str();
        let revenue = item[3].as_f64();
        let operate_profit = item[4].as_f64();
        let total_profit = item[5].as_f64();
        let n_income = item[6].as_f64();
        let total_assets = item[7].as_f64();
        let total_hldr_eqy_exc_min_int = item[8].as_f64();
        let diluted_eps = item[9].as_f64();
        let diluted_roe = item[10].as_f64();
        let yoy_net_profit = item[11].as_f64();
        let bps = item[12].as_f64();
        let perf_summary = item[13].as_str();
        let update_flag = item[14].as_str();

        let query = sqlx::query(
            "INSERT INTO express ( 
                ts_code,
                ann_date,
                end_date,
                revenue,
                operate_profit,
                total_profit,
                n_income,
                total_assets,
                total_hldr_eqy_exc_min_int,
                diluted_eps,
                diluted_roe,
                yoy_net_profit,
                bps,
                perf_summary,
                update_flag)  values($1,$2,$3, $4, $5, $6, $7, $8,$9,$10 ,$11,$12, $13, $14, $15)")
            .bind(ts_code)
            .bind(ann_date)
            .bind(end_date)
            .bind(revenue)
            .bind(operate_profit)
            .bind(total_profit)
            .bind(n_income)
            .bind(total_assets)
            .bind(total_hldr_eqy_exc_min_int)
            .bind(diluted_eps)
            .bind(diluted_roe)
            .bind(yoy_net_profit)
            .bind(bps)
            .bind(perf_summary)
            .bind(update_flag);

        let query_result = query.execute(&dbpool).await;
        println!("{:#?}", query_result);
    }
    Ok(())
}
