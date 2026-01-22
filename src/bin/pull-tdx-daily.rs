use reqwest::Client;
use serde_json::json;
use sqlx::{PgPool, Row};
use tokio;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Step 1: Get trading calendar from Tushare (SSE, since 20150101)
    let client = Client::new();
    let url = "https://api.tushare.pro";
    let tushare_token = "43748a5da1339b43a4956da729ae791f4e25ce9a15a356060658ffe6";
    let database_url = "postgresql://postgres:12341234@127.0.0.1:5432/research";
    let pool = PgPool::connect(database_url).await?;

    // Create tdx_daily table with all fields from Tushare response (use double quotes for column names starting with digits)
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS tdx_daily (
            ts_code TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            close DOUBLE PRECISION,
            open DOUBLE PRECISION,
            high DOUBLE PRECISION,
            low DOUBLE PRECISION,
            pre_close DOUBLE PRECISION,
            change DOUBLE PRECISION,
            pct_change DOUBLE PRECISION,
            vol DOUBLE PRECISION,
            amount DOUBLE PRECISION,
            rise DOUBLE PRECISION,
            vol_ratio DOUBLE PRECISION,
            turnover_rate DOUBLE PRECISION,
            swing DOUBLE PRECISION,
            up_num DOUBLE PRECISION,
            down_num DOUBLE PRECISION,
            limit_up_num DOUBLE PRECISION,
            limit_down_num DOUBLE PRECISION,
            lu_days DOUBLE PRECISION,
            "3day" DOUBLE PRECISION,
            "5day" DOUBLE PRECISION,
            "10day" DOUBLE PRECISION,
            "20day" DOUBLE PRECISION,
            "60day" DOUBLE PRECISION,
            mtd DOUBLE PRECISION,
            ytd DOUBLE PRECISION,
            "1year" DOUBLE PRECISION,
            pe DOUBLE PRECISION,
            pb DOUBLE PRECISION,
            float_mv DOUBLE PRECISION,
            ab_total_mv DOUBLE PRECISION,
            float_share DOUBLE PRECISION,
            total_share DOUBLE PRECISION,
            bm_buy_net DOUBLE PRECISION,
            bm_buy_ratio DOUBLE PRECISION,
            bm_net DOUBLE PRECISION,
            bm_ratio DOUBLE PRECISION,
            PRIMARY KEY (ts_code, trade_date)
        )
        "#,
    )
    .execute(&pool)
    .await?;

    // Fetch trading calendar
    let calendar_payload = json!({
        "api_name": "trade_cal",
        "token": tushare_token,
        "params": {
            "exchange": "SSE",
            "start_date": "20150101",
            "end_date": chrono::Local::now().format("%Y%m%d").to_string()
        },
        "fields": "cal_date,is_open"
    });

    let cal_resp = client
        .post(url)
        .json(&calendar_payload)
        .send()
        .await?
        .json::<serde_json::Value>()
        .await?;

    let cal_items = cal_resp["data"]["items"]
        .as_array()
        .cloned()
        .unwrap_or_default();

    let mut trading_days = Vec::new();
    for item in &cal_items {
        let cal_date = item[0].as_str().unwrap();
        let is_open = match &item[1] {
            v if v.is_u64() => v.as_u64().unwrap() == 1,
            v if v.as_str().is_some() => v.as_str().unwrap() == "1",
            _ => false,
        };
        if is_open {
            trading_days.push(cal_date.to_string());
        }
    }
    println!("Found {} trading days since 20150101", trading_days.len());

    // Step 1b: Get max(trade_date) from tdx_daily table
    let max_trade_date_row = sqlx::query("SELECT MAX(trade_date) FROM tdx_daily")
        .fetch_one(&pool)
        .await?;
    let max_trade_date: Option<String> = max_trade_date_row.try_get(0)?;
    println!("Max trade_date in tdx_daily: {:?}", max_trade_date);

    // Step 1c: Filter trading_days to only those after max_trade_date
    let filtered_trading_days: Vec<String> = match max_trade_date {
        Some(ref max_date) => trading_days
            .iter()
            .filter(|d| *d > max_date)
            .cloned()
            .collect(),
        None => trading_days.clone(),
    };
    println!("Will pull {} trading days", filtered_trading_days.len());

    // Step 2: Pull tdx_daily for each trading day and insert
    let mut total_inserted = 0;
    for (idx, trade_date) in filtered_trading_days.iter().enumerate() {
        let payload = json!({
            "api_name": "tdx_daily",
            "token": tushare_token,
            "params": {
                "start_date": trade_date,
                "end_date": trade_date
            },
            "fields": "" // all fields
        });

        let resp = client
            .post(url)
            .json(&payload)
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;

        let fields_vec = resp["data"]["fields"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let items_vec = resp["data"]["items"]
            .as_array()
            .cloned()
            .unwrap_or_default();

        let fields = &fields_vec;
        let items = &items_vec;

        if items.is_empty() {
            println!(
                "[{}/{}] {}: No tdx_daily data",
                idx + 1,
                trading_days.len(),
                trade_date
            );
            continue;
        }

        let field_names: Vec<&str> = fields.iter().map(|v| v.as_str().unwrap()).collect();
        // Map Tushare fields to DB columns for problematic names (use quoted names for SQL)
        let db_field_names: Vec<String> = field_names
            .iter()
            .map(|&f| match f {
                "3day" | "5day" | "10day" | "20day" | "60day" | "1year" => format!("\"{}\"", f),
                _ => f.to_string(),
            })
            .collect();

        let placeholders: Vec<String> = (1..=db_field_names.len())
            .map(|i| format!("${}", i))
            .collect();
        let insert_sql = format!(
            "INSERT INTO tdx_daily ({}) VALUES ({}) ON CONFLICT (ts_code, trade_date) DO UPDATE SET {}",
            db_field_names.join(","),
            placeholders.join(","),
            db_field_names
                .iter()
                .filter(|&f| f != "ts_code" && f != "trade_date")
                .map(|f| format!("{} = EXCLUDED.{}", f, f))
                .collect::<Vec<_>>()
                .join(", ")
        );

        for item in items {
            let params: Vec<&serde_json::Value> = item.as_array().unwrap().iter().collect();
            let mut query = sqlx::query(&insert_sql);
            for (idx, v) in params.iter().enumerate() {
                let field = field_names[idx];
                match field {
                    // Numeric fields
                    "close" | "open" | "high" | "low" | "pre_close" | "change" | "pct_change"
                    | "vol" | "amount" | "rise" | "vol_ratio" | "turnover_rate" | "swing"
                    | "up_num" | "down_num" | "limit_up_num" | "limit_down_num" | "lu_days"
                    | "3day" | "5day" | "10day" | "20day" | "60day" | "mtd" | "ytd" | "1year"
                    | "pe" | "pb" | "float_mv" | "ab_total_mv" | "float_share" | "total_share"
                    | "bm_buy_net" | "bm_buy_ratio" | "bm_net" | "bm_ratio" => {
                        if v.is_null() {
                            query = query.bind(None::<f64>);
                        } else if let Some(f) = v.as_f64() {
                            query = query.bind(f);
                        } else if let Some(s) = v.as_str() {
                            query = query.bind(s.parse::<f64>().ok());
                        } else {
                            query = query.bind(None::<f64>);
                        }
                    }
                    // Text fields
                    _ => {
                        if v.is_null() {
                            query = query.bind(None::<String>);
                        } else if let Some(s) = v.as_str() {
                            query = query.bind(s);
                        } else {
                            query = query.bind(v.to_string());
                        }
                    }
                }
            }
            let result = query.execute(&pool).await?;
            total_inserted += result.rows_affected();
        }
        println!(
            "[{}/{}] {}: Inserted {} records",
            idx + 1,
            trading_days.len(),
            trade_date,
            items.len()
        );
    }

    println!("Inserted total {} rows into tdx_daily", total_inserted);
    Ok(())
}
