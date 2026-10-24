IF SCHEMA_ID('fred') IS NULL EXEC('CREATE SCHEMA fred');
GO
DROP TABLE IF EXISTS fred.series_quarterly;
GO
CREATE TABLE fred.series_quarterly (
    obs_date  date          NOT NULL,  -- first day of the quarter
    series_id varchar(16)   NOT NULL,
    value     decimal(9, 4) NOT NULL,
    CONSTRAINT pk_series_quarterly PRIMARY KEY (series_id, obs_date),
    CONSTRAINT ck_value_nonneg CHECK (value >= 0),
    CONSTRAINT ck_quarter_start CHECK (DAY(obs_date) = 1 AND MONTH(obs_date) IN (1, 4, 7, 10))
);
GO
CREATE OR ALTER VIEW fred.v_model_frame AS
SELECT obs_date,
       MAX(CASE WHEN series_id = 'CORCCACBS' THEN value END) AS chargeoff_rate,
       MAX(CASE WHEN series_id = 'DRCCLACBS' THEN value END) AS delinquency_rate,
       MAX(CASE WHEN series_id = 'UNRATE'    THEN value END) AS unemployment_rate
FROM fred.series_quarterly
GROUP BY obs_date;
GO
