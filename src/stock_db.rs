use std::{collections::HashMap, rc::Rc, sync::Arc};

use serde::Serialize;
use sqlx::{Pool, Postgres, QueryBuilder, postgres::PgPoolOptions};

use crate::ts::{
    http::{InterfaceType, Params, Rep, Requestx},
    model::AppState,
};

// pub fn create_req(code: String, date: String, interface: &str) -> String {
//     let req: Requestx = Requestx {
//         api_name: interface.to_string(),
//         token: "9009687b64bb934feeb28a7d1245fb4537047b280ea9cc400c812f3f".to_string(),

//         params: Some(Params {
//             ts_code: code,
//             start_date: date,
//         }),
//     };
//     serde_json::to_string(&req).unwrap()
// }

pub fn create_req(api_name: String, params: HashMap<String, String>) -> String {
    let req: Requestx = Requestx {
        api_name: api_name,
        token: "43748a5da1339b43a4956da729ae791f4e25ce9a15a356060658ffe6".to_string(),

        params: params,
    };
    serde_json::to_string(&req).unwrap()
}

pub async fn get_connection() -> sqlx::Pool<Postgres> {
    let database_url = "postgresql://postgres:12341234@127.0.0.1:5432/research?schema=public";

    let pool = match PgPoolOptions::new()
        .max_connections(128) // Increased from 64 to handle parallel batch processing
        .min_connections(10) // Keep minimum connections warm
        .acquire_timeout(std::time::Duration::from_secs(30))
        .idle_timeout(std::time::Duration::from_secs(300)) // 5 minute idle timeout
        .max_lifetime(std::time::Duration::from_secs(1800)) // 30 minute max connection lifetime
        .connect(database_url)
        .await
    {
        Ok(pool) => {
            // println!("✅Connection to the database is successful!");
            pool
        }
        Err(err) => {
            println!("🔥 Failed to connect to the database: {:?}", err);
            std::process::exit(1);
        }
    };

    pool
}
