use std::collections::HashMap;

use rust_llm_stock::{
    stock_db::get_connection,
    ts::http::{Rep, Requestx, Responsex, StockInfoRecord},
};
use sqlx::{Postgres, QueryBuilder};
/**
 * 首先初始化股市的股票列表，从头开始发送给一个请求并更新入库；
 * 如果发现重复的则不会更新
 * 因该接口只能每个小时调用一次，所以该功能需要做限流处理
 * Response body中的错误：
 * {"code":40203,"msg":"抱歉，您每小时最多访问该接口1次，权限的具体详情访问：https://tushare.pro/document/1?doc_id=108。","message":"抱歉，您每小时最多访问该接口1次，权限的具体详情访问：https://tushare.pro/document/1?doc_id=108。","data":null}
 */
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut params = HashMap::new();
    // params.insert("ts_code".to_string(), "689009.SH".to_string());
    let req: Requestx = Requestx {
        api_name: "stock_basic".to_string(),
        token: "43748a5da1339b43a4956da729ae791f4e25ce9a15a356060658ffe6".to_string(),
        params: params,
    };

    // let database_url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let json: String = serde_json::to_string(&req).unwrap();
    println!("{}", json);

    let pool: sqlx::Pool<sqlx::Postgres> = get_connection().await;

    let client: reqwest::Client = reqwest::Client::new();

    let response = client
        .post("http://api.tushare.pro")
        .body(json.to_owned())
        .send()
        .await?;

    println!("Status: {}", response.status());

    let response_body = response.text().await?;
    println!("Response body:\n{}", response_body);

    let repx: Responsex = serde_json::from_str(&response_body)?;

    println!("将记录数连接为sql:\n{:?}", repx.data);

    for item in &repx.data.items {
        let ts_code = item[0].as_str();
        let symbol = item[1].as_str();
        let name = item[2].as_str();
        let area = item[3].as_str();
        let industry = item[4].as_str();
        let cnspell = item[5].as_str();
        let market = item[6].as_str();
        let list_date = item[7].as_str();
        let act_name: Option<&str> = item[8].as_str();
        let act_ent_type = item[9].as_str();

        let insert_query = sqlx::query(
            "INSERT INTO stock_basic (ts_code,symbol,name,industry,cnspell,market,list_date,act_name,
                act_ent_type)
            values ($1,$2,$3,$4,$5,$6,$7,$8,$9
            ) ON CONFLICT (ts_code) DO NOTHING")
            .bind(ts_code)
            .bind(symbol)
            .bind(name)
            .bind(area)
            .bind(industry)
            .bind(cnspell)
            .bind(market)
            .bind(list_date)
            .bind(act_name)
            .bind(act_ent_type);
        let query_result = insert_query.execute(&pool).await?;

        println!("{:#?}", query_result);
    }

    Ok(())
}
