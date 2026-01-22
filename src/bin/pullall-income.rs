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
    WHERE list_status = 'L' AND not exists(select 1 from income where ts_code=stock_basic.ts_code and end_date< '20231230')  ";
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
        "select * from (select distinct ts_code, max(end_date) as trade_date from income group by ts_code ) tmp where  trade_code < '{}' ",
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
    println!("start pulling income ...");
    let mut params = HashMap::new();
    params.insert(String::from("ts_code".to_string()), String::from(code));

    let response = client
        .post("http://api.tushare.pro")
        .body(create_req("income".to_string(), params))
        .send().await?;

    println!("Status: {}", response.status());

    let response_body = response.text().await?;
    println!("Response body:\n{}", response_body);

    let repx: Responsex = serde_json::from_str(&response_body)?;

    let dbpool = get_connection().await;

    for item in &repx.data.items {
        let ts_code = item[0].as_str();
        let ann_date = item[1].as_str();
        let f_ann_date = item[2].as_str();
        let end_date = item[3].as_str();
        let report_type = item[4].as_str();
        let comp_type = item[5].as_str();
        let end_type = item[6].as_str();
        let basic_eps = item[7].as_f64();
        let diluted_eps = item[8].as_f64();
        let total_revenue = item[9].as_f64();
        let revenue = item[10].as_f64();
        let int_income = item[11].as_f64();
        let prem_earned = item[12].as_f64();
        let comm_income = item[13].as_f64();
        let n_commis_income = item[14].as_f64();
        let n_oth_income = item[15].as_f64();
        let n_oth_b_income = item[16].as_f64();
        let prem_income = item[17].as_f64();
        let out_prem = item[18].as_f64();
        let une_prem_reser = item[19].as_f64();
        let reins_income = item[20].as_f64();
        let n_sec_tb_income = item[21].as_f64();
        let n_sec_uw_income = item[22].as_f64();
        let n_asset_mg_income = item[23].as_f64();
        let oth_b_income = item[24].as_f64();
        let fv_value_chg_gain = item[25].as_f64();
        let invest_income = item[26].as_f64();
        let ass_invest_income = item[27].as_f64();
        let forex_gain = item[28].as_f64();
        let total_cogs = item[29].as_f64();
        let oper_cost = item[30].as_f64();
        let int_exp = item[31].as_f64();
        let comm_exp = item[32].as_f64();
        let biz_tax_surchg = item[33].as_f64();
        let sell_exp = item[34].as_f64();
        let admin_exp = item[35].as_f64();
        let fin_exp = item[36].as_f64();
        let assets_impair_loss = item[37].as_f64();
        let prem_refund = item[38].as_f64();
        let compens_payout = item[39].as_f64();
        let reser_insur_liab = item[40].as_f64();
        let div_payt = item[41].as_f64();
        let reins_exp = item[42].as_f64();
        let oper_exp = item[43].as_f64();
        let compens_payout_refu = item[44].as_f64();
        let insur_reser_refu = item[45].as_f64();
        let reins_cost_refund = item[46].as_f64();
        let other_bus_cost = item[47].as_f64();
        let operate_profit = item[48].as_f64();
        let non_oper_income = item[49].as_f64();
        let non_oper_exp = item[50].as_f64();
        let nca_disploss = item[51].as_f64();
        let total_profit = item[52].as_f64();
        let income_tax = item[53].as_f64();
        let n_income = item[54].as_f64();
        let n_income_attr_p = item[55].as_f64();
        let minority_gain = item[56].as_f64();
        let oth_compr_income = item[57].as_f64();
        let t_compr_income = item[58].as_f64();
        let compr_inc_attr_p = item[59].as_f64();
        let compr_inc_attr_m_s = item[60].as_f64();
        let ebit = item[61].as_f64();
        let ebitda = item[62].as_f64();
        let insurance_exp = item[63].as_f64();
        let undist_profit = item[64].as_f64();
        let distable_profit = item[65].as_f64();
        let rd_exp = item[66].as_f64();
        let fin_exp_int_exp = item[67].as_f64();
        let fin_exp_int_inc = item[68].as_f64();
        let transfer_surplus_rese = item[69].as_f64();
        let transfer_housing_imprest = item[70].as_f64();
        let transfer_oth = item[71].as_f64();
        let adj_lossgain = item[72].as_f64();
        let withdra_legal_surplus = item[73].as_f64();
        let withdra_legal_pubfund = item[74].as_f64();
        let withdra_biz_devfund = item[75].as_f64();
        let withdra_rese_fund = item[76].as_f64();
        let withdra_oth_ersu = item[77].as_f64();
        let workers_welfare = item[78].as_f64();
        let distr_profit_shrhder = item[79].as_f64();
        let prfshare_payable_dvd = item[80].as_f64();
        let comshare_payable_dvd = item[81].as_f64();
        let capit_comstock_div = item[82].as_f64();
        let continued_net_profit = item[83].as_f64();
        let update_flag = item[84].as_str();

        let query = sqlx::query(
            "INSERT INTO income ( 
                ts_code,
                ann_date,
                f_ann_date,
                end_date,
                report_type,
                comp_type,
                end_type,
                basic_eps,
                diluted_eps,
                total_revenue,
                revenue,
                int_income,
                prem_earned,
                comm_income,
                n_commis_income,
                n_oth_income,
                n_oth_b_income,
                prem_income,
                out_prem,
                une_prem_reser,
                reins_income,
                n_sec_tb_income,
                n_sec_uw_income,
                n_asset_mg_income,
                oth_b_income,
                fv_value_chg_gain,
                invest_income,
                ass_invest_income,
                forex_gain,
                total_cogs,
                oper_cost,
                int_exp,
                comm_exp,
                biz_tax_surchg,
                sell_exp,
                admin_exp,
                fin_exp,
                assets_impair_loss,
                prem_refund,
                compens_payout,
                reser_insur_liab,
                div_payt,
                reins_exp,
                oper_exp,
                compens_payout_refu,
                insur_reser_refu,
                reins_cost_refund,
                other_bus_cost,
                operate_profit,
                non_oper_income,
                non_oper_exp,
                nca_disploss,
                total_profit,
                income_tax,
                n_income,
                n_income_attr_p,
                minority_gain,
                oth_compr_income,
                t_compr_income,
                compr_inc_attr_p,
                compr_inc_attr_m_s,
                ebit,
                ebitda,
                insurance_exp,
                undist_profit,
                distable_profit,
                rd_exp,
                fin_exp_int_exp,
                fin_exp_int_inc,
                transfer_surplus_rese,
                transfer_housing_imprest,
                transfer_oth,
                adj_lossgain,
                withdra_legal_surplus,
                withdra_legal_pubfund,
                withdra_biz_devfund,
                withdra_rese_fund,
                withdra_oth_ersu,
                workers_welfare,
                distr_profit_shrhder,
                prfshare_payable_dvd,
                comshare_payable_dvd,
                capit_comstock_div,
                continued_net_profit,
                update_flag
                )  values($1,$2,$3, $4, $5, $6, $7, $8,$9,$10 ,
                    $11,$12, $13, $14, $15,$16,$17, $18, $19, $20,
                    $21,$22,$23, $24, $25, $26, $27, $28,$29,$30,
                    $31,$32,$33, $34, $35, $36, $37, $38,$39,$40 ,
                    $41,$42,$43, $44, $45, $46, $47, $48,$49,$50 ,
                    $51,$52,$53, $54, $55, $56, $57, $58,$59,$60,
                    $61,$62,$63, $64, $65, $66, $67, $68,$69,$70,
                    $71,$72,$73, $74, $75, $76, $77, $78,$79,$80,
                    $81,$82,$83, $84, $85)")
            .bind(ts_code)
            .bind(ann_date)
            .bind(f_ann_date)
            .bind(end_date)
            .bind(report_type)
            .bind(comp_type)
            .bind(end_type)
            .bind(basic_eps)
            .bind(diluted_eps)
            .bind(total_revenue)
            .bind(revenue)
            .bind(int_income)
            .bind(prem_earned)
            .bind(comm_income)
            .bind(n_commis_income)
            .bind(n_oth_income)
            .bind(n_oth_b_income)
            .bind(prem_income)
            .bind(out_prem)
            .bind(une_prem_reser)
            .bind(reins_income)
            .bind(n_sec_tb_income)
            .bind(n_sec_uw_income)
            .bind(n_asset_mg_income)
            .bind(oth_b_income)
            .bind(fv_value_chg_gain)
            .bind(invest_income)
            .bind(ass_invest_income)
            .bind(forex_gain)
            .bind(total_cogs)
            .bind(oper_cost)
            .bind(int_exp)
            .bind(comm_exp)
            .bind(biz_tax_surchg)
            .bind(sell_exp)
            .bind(admin_exp)
            .bind(fin_exp)
            .bind(assets_impair_loss)
            .bind(prem_refund)
            .bind(compens_payout)
            .bind(reser_insur_liab)
            .bind(div_payt)
            .bind(reins_exp)
            .bind(oper_exp)
            .bind(compens_payout_refu)
            .bind(insur_reser_refu)
            .bind(reins_cost_refund)
            .bind(other_bus_cost)
            .bind(operate_profit)
            .bind(non_oper_income)
            .bind(non_oper_exp)
            .bind(nca_disploss)
            .bind(total_profit)
            .bind(income_tax)
            .bind(n_income)
            .bind(n_income_attr_p)
            .bind(minority_gain)
            .bind(oth_compr_income)
            .bind(t_compr_income)
            .bind(compr_inc_attr_p)
            .bind(compr_inc_attr_m_s)
            .bind(ebit)
            .bind(ebitda)
            .bind(insurance_exp)
            .bind(undist_profit)
            .bind(distable_profit)
            .bind(rd_exp)
            .bind(fin_exp_int_exp)
            .bind(fin_exp_int_inc)
            .bind(transfer_surplus_rese)
            .bind(transfer_housing_imprest)
            .bind(transfer_oth)
            .bind(adj_lossgain)
            .bind(withdra_legal_surplus)
            .bind(withdra_legal_pubfund)
            .bind(withdra_biz_devfund)
            .bind(withdra_rese_fund)
            .bind(withdra_oth_ersu)
            .bind(workers_welfare)
            .bind(distr_profit_shrhder)
            .bind(prfshare_payable_dvd)
            .bind(comshare_payable_dvd)
            .bind(capit_comstock_div)
            .bind(continued_net_profit)
            .bind(update_flag);

        let query_result = query.execute(&dbpool).await;
        println!("{:#?}", query_result);
    }
    Ok(())
}
