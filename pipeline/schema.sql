-- =====================================================================
-- portfolio-risk-analytics warehouse schema (DuckDB, star schema)
--
-- dbt-model mapping (each table below corresponds to one dbt model):
--   seeds/tickers.csv              -> stg_tickers  (seeded instrument master)
--   data/index.parquet             -> stg_market_index (synthetic benchmark)
--   dim_ticker                     -> dim_tickers.sql  (SCD-0 dimension)
--   dim_date                       -> dim_dates.sql    (calendar dimension)
--   fact_daily_prices              -> fact_daily_prices.sql
--   fact_returns                   -> fact_returns.sql (log returns, portfolio math)
-- =====================================================================

CREATE TABLE IF NOT EXISTS dim_ticker (
    ticker           VARCHAR PRIMARY KEY,
    sector           VARCHAR NOT NULL,
    beta_market      DOUBLE,
    annualised_drift DOUBLE,
    shares_outstanding BIGINT
);

CREATE TABLE IF NOT EXISTS dim_date (
    date             DATE PRIMARY KEY,
    year             INTEGER NOT NULL,
    month            INTEGER NOT NULL,
    day              INTEGER NOT NULL,
    day_of_week      INTEGER NOT NULL,
    is_month_end     BOOLEAN NOT NULL
);

CREATE TABLE IF NOT EXISTS fact_daily_prices (
    date             DATE NOT NULL REFERENCES dim_date(date),
    ticker           VARCHAR NOT NULL REFERENCES dim_ticker(ticker),
    close            DOUBLE NOT NULL,
    market_cap       DOUBLE,
    PRIMARY KEY (date, ticker)
);

CREATE TABLE IF NOT EXISTS fact_returns (
    date             DATE NOT NULL,
    ticker           VARCHAR NOT NULL REFERENCES dim_ticker(ticker),
    log_return       DOUBLE NOT NULL,
    simple_return    DOUBLE NOT NULL,
    PRIMARY KEY (date, ticker)
);

CREATE TABLE IF NOT EXISTS fact_market_index (
    date             DATE PRIMARY KEY REFERENCES dim_date(date),
    close            DOUBLE NOT NULL,
    log_return       DOUBLE
);
