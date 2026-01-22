pub mod http {
    pub enum InterfaceType {
        Daily(String),
        Weekly(String),
        Monthly(String),
        Income(String),
        Forecast(String),
        Express(String),
    }
    use serde::{ Deserialize, Serialize };
    #[derive(Serialize)]
    pub struct Params {
        pub ts_code: String,
        pub start_date: String,
    }

    use std::collections::HashMap;
    #[derive(Serialize, Deserialize, Debug)]
    pub struct Requestx {
        pub api_name: String,
        pub token: String,
        pub params: HashMap<String, String>,
    }

    #[derive(Serialize, Deserialize, Debug)]
    pub struct Rep<T> {
        pub request_id: String,
        pub code: i8,
        pub msg: String,
        pub data: Data<T>,
    }

    #[derive(Debug, Deserialize, Serialize)]
    pub struct Datax {
        pub fields: Vec<String>,
        pub items: Vec<Vec<serde_json::Value>>,
        pub has_more: bool,
    }

    #[derive(Debug, Deserialize, Serialize)]
    pub struct Responsex {
        pub request_id: String,
        pub code: i32,
        pub msg: String,
        pub data: Datax,
    }

    #[derive(Serialize, Deserialize, Debug)]
    pub struct Express(
        pub String,
        pub String,
        pub String,
        pub f64,
        pub f64,
        pub f64,
        pub f64,
        pub f64,
        pub f64,
        pub f64,
        pub f64,
        pub f64,
        pub f64,
        pub String,
        pub String,
    );

    #[derive(Serialize, Deserialize, Debug)]
    pub struct TradeRecord(
        pub String,
        pub String,
        pub f64,
        pub f64,
        pub f64,
        pub f64,
        pub f64,
        pub f64,
        pub f64,
        pub f64,
        pub f64,
        pub f64,
    );

    #[derive(Serialize, Deserialize, Debug)]
    pub struct Forecast(
        pub String,
        pub String,
        pub String,
        pub String,
        pub f64,
        pub f64,
        pub f64,
        pub f64,
        pub f64,
        pub String,
        pub String,
        pub String,
        pub String,
    );

    #[derive(Serialize, Deserialize, Debug)]
    pub struct Data<T> {
        pub fields: Vec<String>,
        pub items: Vec<T>,
        pub has_more: bool,
    }

    #[derive(Serialize, Deserialize, Debug)]
    pub struct StockInfoRecord(
        pub String,
        pub String,
        pub String,
        pub String,
        pub String,
        pub String,
        pub String,
        pub String,
        pub Option<String>,
        pub Option<String>,
    );
}

pub mod model {
    pub struct AppState {
        pub db: Pool<Postgres>,
    }

    use serde::{ Deserialize, Serialize };
    use sqlx::{ FromRow, Postgres, Pool };
    #[derive(Debug, FromRow, Deserialize, Serialize)]
    pub struct StockBasic {
        pub ts_code: String,
        pub symbol: Option<String>,
        pub name: Option<String>,
        pub industry: Option<String>,
        pub market: Option<String>,
        pub list_date: Option<String>,
    }
    #[derive(Debug, FromRow, Deserialize, Serialize)]
    pub struct IndustryBasic {
        pub industry: String,
    }
    #[derive(Debug, FromRow, Deserialize, Serialize)]
    pub struct MarketBasic {
        pub market: String,
    }

    #[derive(Debug, FromRow, Deserialize, Serialize)]
    pub struct StockKey {
        pub ts_code: String,
        pub trade_date: Option<String>,
    }

    #[derive(Debug, FromRow, Deserialize, Serialize)]
    pub struct DailyModel {
        pub ts_code: String,
        pub trade_date: Option<String>,
        pub open: Option<f64>,
        pub close: Option<f64>,
        pub high: Option<f64>,
        pub low: Option<f64>,
        pub pre_close: Option<f64>,
        pub vol: Option<f64>,
        pub amount: Option<f64>,
        pub change: Option<f64>,
        pub pct_chg: Option<f64>,
    }

    #[derive(Debug, FromRow, Deserialize, Serialize)]
    pub struct WeeklyModel {
        pub id: i32,
        pub ts_code: String,
        pub trade_date: chrono::DateTime<chrono::Utc>,
        pub open: f64,
        pub close: f64,
        pub high: f64,
        pub low: f64,
        pub pre_close: f64,
        pub volume: f64,
        pub amount: f64,
        pub change: f64,
        pub pct_change: f64,
    }

    #[derive(Debug, FromRow, Deserialize, Serialize)]
    pub struct MonthlyModel {
        pub id: i32,
        pub ts_code: String,
        pub trade_date: chrono::DateTime<chrono::Utc>,
        pub open: f32,
        pub close: f32,
        pub high: f32,
        pub low: f32,
        pub pre_close: f32,
        pub volume: f32,
        pub amount: f32,
        pub change: f32,
        pub pct_change: f32,
    }

    #[derive(Debug, FromRow, Deserialize, Serialize)]
    pub struct CCiModel {
        pub id: i32,
        pub ts_code: String,
        pub cci_value: chrono::DateTime<chrono::Utc>,
        pub high: f32,
        pub low: f32,
        pub rt_cci: f32,
        pub alert_time: chrono::DateTime<chrono::Utc>,
        pub action_indicator: i8,
        pub trade_date: chrono::DateTime<chrono::Utc>,
        pub job_date: chrono::DateTime<chrono::Utc>,
    }

    //指数
    #[derive(Debug, FromRow, Deserialize, Serialize)]
    pub struct IndexModel {
        pub id: i32,
        pub ts_code: String,
        pub symbol: String,
        pub name: String,
        pub inustry: String,
        pub area: String,
        pub market: String,
        pub list_date: chrono::DateTime<chrono::Utc>,
    }

    #[derive(Debug, FromRow, Deserialize, Serialize)]
    pub struct IndexDailyModel {
        pub id: i32,
        pub ts_code: String,
        pub trade_date: chrono::DateTime<chrono::Utc>,
        pub open: f32,
        pub close: f32,
        pub high: f32,
        pub low: f32,
        pub pre_close: f32,
        pub volume: f32,
        pub amount: f32,
        pub change: f32,
        pub pct_change: f32,
    }

    #[derive(Debug, FromRow, Deserialize, Serialize)]
    pub struct IncomeModel {
        pub id: i32,
        pub ts_code: String,
        /*
        报告期间
         */
        // @JsonProperty("period")
        // private LocalDate period;
        /*
        公告日期
         */
        pub ann_date: chrono::DateTime<chrono::Utc>,
        //     最终报告日
        /**
         * f_ann_date
         */
        pub f_ann_date: chrono::DateTime<chrono::Utc>,
        //    end_date 公告结束日期
        pub end_date: chrono::DateTime<chrono::Utc>,
        //    report_type
        pub report_type: String,
        //公司类型：1一般工商业 2银行 3保险 4证券
        pub comp_type: String,
        //    basic_eps基本每股收益
        pub basic_eps: f32,
        //    diluted_eps	float	稀释每股收益
        pub diluted_eps: f32,
        //    total_revenue	float	营业总收入 (元，下同)
        pub total_revenue: f32,
        //    revenue	float	营业收入
        pub revenue: f32,
        //    int_income	float	利息收入
        pub int_income: f32,
        //    prem_earned	float	已赚保费
        pub prem_earned: f32,
        //    comm_income	float	手续费及佣金收入
        pub comm_income: f32,
        //    n_commis_income	float	手续费及佣金净收入
        pub n_commis_income: f32,
        //    n_oth_income	float	其他经营净收益
        pub n_oth_income: f32,
        //    n_oth_b_income	float	加:其他业务净收益
        pub n_oth_b_income: f32,
        //    prem_income	float	保险业务收入
        pub prem_income: f32,
        //    out_prem	float	减:分出保费
        pub out_prem: f32,
        //    une_prem_reser	float	提取未到期责任准备金
        pub une_prem_reser: f32,
        //    reins_income	float	其中:分保费收入
        pub reins_income: f32,
        //    n_sec_tb_income	float	代理买卖证券业务净收入
        pub n_sec_tb_income: f32,
        //    n_sec_uw_income	float	证券承销业务净收入
        pub n_sec_uw_income: f32,
        //    n_asset_mg_income	float	受托客户资产管理业务净收入
        pub n_asset_mg_income: f32,
        //    oth_b_income	float	其他业务收入
        pub oth_b_income: f32,
        //    fv_value_chg_gain	float	加:公允价值变动净收益
        pub fv_value_chg_gain: f32,
        //    invest_income	float	加:投资净收益
        pub invest_income: f32,
        //    ass_invest_income	float	其中:对联营企业和合营企业的投资收益
        pub ass_invest_income: f32,
        //    forex_gain	float	加:汇兑净收益
        pub forex_gain: f32,
        //    total_cogs	float	营业总成本
        pub total_cogs: f32,
        //    oper_cost	float	减:营业成本
        pub oper_cost: f32,
        //    int_exp	float	减:利息支出
        pub int_exp: f32,
        //    comm_exp	float	减:手续费及佣金支出
        pub comm_exp: f32,
        //    biz_tax_surchg	float	减:营业税金及附加
        pub biz_tax_surchg: f32,
        //    sell_exp	float	减:销售费用
        pub sell_exp: f32,

        //    admin_exp	float	减:管理费用
        pub admin_exp: f32,
        //    fin_exp	float	减:财务费用
        pub fin_exp: f32,
        //    assets_impair_loss	float	减:资产减值损失
        pub assets_impair_loss: f32,
        //    prem_refund	float	退保金
        pub prem_refund: f32,
        //    compens_payout	float	赔付总支出
        pub compens_payout: f32,
        //    reser_insur_liab	float	提取保险责任准备金
        pub reser_insur_liab: f32,
        //    div_payt	float	保户红利支出
        pub div_payt: f32,
        //    reins_exp	float	分保费用
        pub reins_exp: f32,
        //    oper_exp	float	营业支出
        pub oper_exp: f32,
        //    compens_payout_refu	float	减:摊回赔付支出
        pub compens_payout_refu: f32,
        //    insur_reser_refu	float	减:摊回保险责任准备金
        pub insur_reser_refu: f32,
        //    reins_cost_refund	float	减:摊回分保费用
        pub reins_cost_refund: f32,
        //    other_bus_cost	float	其他业务成本
        pub other_bus_cost: f32,
        //    operate_profit	float	营业利润
        pub operate_profit: f32,
        //    non_oper_income	float	加:营业外收入
        pub non_oper_income: f32,
        //    non_oper_exp	float	减:营业外支出
        pub non_oper_exp: f32,
        //    nca_disploss	float	其中:减:非流动资产处置净损失
        pub nca_disploss: f32,
        //    total_profit	float	利润总额
        pub total_profit: f32,
        //    income_tax	float	所得税费用
        pub income_tax: f32,
        //    n_income	float	净利润(含少数股东损益)
        pub n_income: f32,
        //    n_income_attr_p	float	净利润(不含少数股东损益)
        pub n_income_attr_p: f32,
        //    minority_gain	float	少数股东损益
        pub minority_gain: f32,
        //    oth_compr_income	float	其他综合收益
        pub oth_compr_income: f32,
        //    t_compr_income	float	综合收益总额
        pub t_compr_income: f32,
        //    compr_inc_attr_p	float	归属于母公司(或股东)的综合收益总额 Total comprehensive income attributable to the parent company (or shareholder)
        pub compr_inc_attr_p: f32,
        //    compr_inc_attr_m_s	float	归属于少数股东的综合收益总额
        pub compr_inc_attr_m_s: f32,
        //    ebit	float	息税前利润
        pub ebit: f32,
        //    ebitda	float	息税折旧摊销前利润
        pub ebitda: f32,
        //    insurance_exp	float	保险业务支出
        pub insurance_exp: f32,
        //    undist_profit	float	年初未分配利润
        pub undist_profit: f32,
        //    distable_profit	float	可分配利润
        pub distable_profit: f32,
        pub rd_exp: f32,
        pub fin_exp_int_exp: f32,
        pub fin_exp_int_inc: f32,
        pub transfer_surplus_rese: f32,
        pub transfer_housing_imprest: f32,
        pub transfer_oth: f32,
        pub adj_lossgain: f32,
        pub withdra_legal_surplus: f32,
        pub withdra_legal_pubfund: f32,
        pub withdra_biz_devfund: f32,
        pub withdra_rese_fund: f32,
        pub withdra_oth_ersu: f32,
        pub workers_welfare: f32,
        pub distr_profit_shrhder: f32,
        pub prfshare_payable_dvd: f32,
        pub comshare_payable_dvd: f32,
        pub capit_comstock_div: f32,
        pub continued_net_profit: f32,
        pub update_flag: f32,
    }

    #[derive(Debug, FromRow, Deserialize, Serialize)]
    pub struct MoneyflowCntThsModel {
        pub trade_date: String,
        pub ts_code: String,
        pub name: String,
        pub lead_stock: String,
        pub close_price: f64,
        pub pct_change: f64,
        pub industry_index: f64,
        pub company_num: i32,
        pub pct_change_stock: f64,
        pub net_buy_amount: f64,
        pub net_sell_amount: f64,
        pub net_amount: f64,
        pub created_at: Option<chrono::DateTime<chrono::Utc>>,
    }
}
