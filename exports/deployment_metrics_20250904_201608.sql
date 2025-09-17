-- SQL export for deployment_metrics
-- Generated: 2025-09-04 20:16:08.910126

CREATE TABLE IF NOT EXISTS deployment_metrics (
    time TIMESTAMP,
    measurement TEXT,
    field TEXT,
    value FLOAT,
    result TEXT,
    table INTEGER,
    environment TEXT,
    service TEXT,
    version TEXT
);

INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-16 16:09:29.046290+00:00', 'deployment', 'status', 1.0, '_result', 44, 'staging', 'web', 'v1.19.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-16 16:09:29.046290+00:00', 'deployment', 'duration_seconds', 176.6604582808917, '_result', 42, 'staging', 'web', 'v1.19.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-16 16:09:29.046290+00:00', 'deployment', 'rollback', 1.0, '_result', 43, 'staging', 'web', 'v1.19.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-17 16:09:29.046290+00:00', 'deployment', 'duration_seconds', 164.19113496662513, '_result', 51, 'staging', 'worker', 'v1.18.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-17 16:09:29.046290+00:00', 'deployment', 'status', 1.0, '_result', 53, 'staging', 'worker', 'v1.18.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-17 16:09:29.046290+00:00', 'deployment', 'rollback', 1.0, '_result', 52, 'staging', 'worker', 'v1.18.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-18 16:09:29.046290+00:00', 'deployment', 'duration_seconds', 139.0827146891829, '_result', 0, 'dev', 'api', 'v1.17.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-18 16:09:29.046290+00:00', 'deployment', 'status', 1.0, '_result', 2, 'dev', 'api', 'v1.17.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-18 16:09:29.046290+00:00', 'deployment', 'rollback', 1.0, '_result', 1, 'dev', 'api', 'v1.17.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-19 16:09:29.046290+00:00', 'deployment', 'duration_seconds', 226.6762175603647, '_result', 12, 'dev', 'worker', 'v1.16.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-19 16:09:29.046290+00:00', 'deployment', 'rollback', 0.0, '_result', 13, 'dev', 'worker', 'v1.16.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-19 16:09:29.046290+00:00', 'deployment', 'status', 0.0, '_result', 14, 'dev', 'worker', 'v1.16.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-20 16:09:29.046290+00:00', 'deployment', 'rollback', 1.0, '_result', 31, 'production', 'web', 'v1.15.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-20 16:09:29.046290+00:00', 'deployment', 'status', 1.0, '_result', 32, 'production', 'web', 'v1.15.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-20 16:09:29.046290+00:00', 'deployment', 'duration_seconds', 162.67285783164493, '_result', 30, 'production', 'web', 'v1.15.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-21 16:09:29.046290+00:00', 'deployment', 'rollback', 0.0, '_result', 40, 'staging', 'web', 'v1.14.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-21 16:09:29.046290+00:00', 'deployment', 'status', 0.0, '_result', 41, 'staging', 'web', 'v1.14.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-21 16:09:29.046290+00:00', 'deployment', 'duration_seconds', 284.67404774868066, '_result', 39, 'staging', 'web', 'v1.14.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-22 16:09:29.046290+00:00', 'deployment', 'duration_seconds', 198.17946543764108, '_result', 18, 'production', 'api', 'v1.13.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-22 16:09:29.046290+00:00', 'deployment', 'status', 0.0, '_result', 20, 'production', 'api', 'v1.13.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-22 16:09:29.046290+00:00', 'deployment', 'rollback', 0.0, '_result', 19, 'production', 'api', 'v1.13.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-23 16:09:29.046290+00:00', 'deployment', 'rollback', 0.0, '_result', 37, 'staging', 'scheduler', 'v1.12.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-23 16:09:29.046290+00:00', 'deployment', 'duration_seconds', 140.07399237880549, '_result', 36, 'staging', 'scheduler', 'v1.12.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-23 16:09:29.046290+00:00', 'deployment', 'status', 1.0, '_result', 38, 'staging', 'scheduler', 'v1.12.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-24 16:09:29.046290+00:00', 'deployment', 'rollback', 1.0, '_result', 10, 'dev', 'worker', 'v1.11.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-24 16:09:29.046290+00:00', 'deployment', 'duration_seconds', 173.4873555068998, '_result', 9, 'dev', 'worker', 'v1.11.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-24 16:09:29.046290+00:00', 'deployment', 'status', 1.0, '_result', 11, 'dev', 'worker', 'v1.11.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-25 16:09:29.046290+00:00', 'deployment', 'status', 0.0, '_result', 50, 'staging', 'worker', 'v1.10.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-25 16:09:29.046290+00:00', 'deployment', 'duration_seconds', 222.51803721701523, '_result', 48, 'staging', 'worker', 'v1.10.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-25 16:09:29.046290+00:00', 'deployment', 'rollback', 1.0, '_result', 49, 'staging', 'worker', 'v1.10.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-26 16:09:29.046290+00:00', 'deployment', 'duration_seconds', 46.56595875770826, '_result', 15, 'dev', 'worker', 'v1.9.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-26 16:09:29.046290+00:00', 'deployment', 'rollback', 0.0, '_result', 16, 'dev', 'worker', 'v1.9.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-26 16:09:29.046290+00:00', 'deployment', 'status', 1.0, '_result', 17, 'dev', 'worker', 'v1.9.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-27 16:09:29.046290+00:00', 'deployment', 'duration_seconds', 149.93201275695355, '_result', 33, 'production', 'worker', 'v1.8.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-27 16:09:29.046290+00:00', 'deployment', 'rollback', 0.0, '_result', 34, 'production', 'worker', 'v1.8.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-27 16:09:29.046290+00:00', 'deployment', 'status', 1.0, '_result', 35, 'production', 'worker', 'v1.8.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-28 16:09:29.046290+00:00', 'deployment', 'status', 1.0, '_result', 26, 'production', 'api', 'v1.7.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-28 16:09:29.046290+00:00', 'deployment', 'rollback', 1.0, '_result', 25, 'production', 'api', 'v1.7.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-28 16:09:29.046290+00:00', 'deployment', 'duration_seconds', 275.3572363934305, '_result', 24, 'production', 'api', 'v1.7.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-29 16:09:29.046290+00:00', 'deployment', 'duration_seconds', 262.0306501290205, '_result', 57, 'staging', 'worker', 'v1.6.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-29 16:09:29.046290+00:00', 'deployment', 'status', 1.0, '_result', 59, 'staging', 'worker', 'v1.6.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-29 16:09:29.046290+00:00', 'deployment', 'rollback', 0.0, '_result', 58, 'staging', 'worker', 'v1.6.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-30 16:09:29.046290+00:00', 'deployment', 'duration_seconds', 97.53018848935467, '_result', 54, 'staging', 'worker', 'v1.5.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-30 16:09:29.046290+00:00', 'deployment', 'rollback', 1.0, '_result', 55, 'staging', 'worker', 'v1.5.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-30 16:09:29.046290+00:00', 'deployment', 'status', 0.0, '_result', 56, 'staging', 'worker', 'v1.5.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-31 16:09:29.046290+00:00', 'deployment', 'duration_seconds', 266.8143422199007, '_result', 3, 'dev', 'web', 'v1.4.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-31 16:09:29.046290+00:00', 'deployment', 'status', 1.0, '_result', 5, 'dev', 'web', 'v1.4.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-08-31 16:09:29.046290+00:00', 'deployment', 'rollback', 0.0, '_result', 4, 'dev', 'web', 'v1.4.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-09-01 16:09:29.046290+00:00', 'deployment', 'status', 1.0, '_result', 23, 'production', 'api', 'v1.3.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-09-01 16:09:29.046290+00:00', 'deployment', 'rollback', 0.0, '_result', 22, 'production', 'api', 'v1.3.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-09-01 16:09:29.046290+00:00', 'deployment', 'duration_seconds', 147.42974252961227, '_result', 21, 'production', 'api', 'v1.3.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-09-02 16:09:29.046290+00:00', 'deployment', 'duration_seconds', 64.17857098813728, '_result', 45, 'staging', 'web', 'v1.2.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-09-02 16:09:29.046290+00:00', 'deployment', 'rollback', 0.0, '_result', 46, 'staging', 'web', 'v1.2.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-09-02 16:09:29.046290+00:00', 'deployment', 'status', 1.0, '_result', 47, 'staging', 'web', 'v1.2.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-09-03 16:09:29.046290+00:00', 'deployment', 'rollback', 0.0, '_result', 7, 'dev', 'worker', 'v1.1.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-09-03 16:09:29.046290+00:00', 'deployment', 'status', 1.0, '_result', 8, 'dev', 'worker', 'v1.1.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-09-03 16:09:29.046290+00:00', 'deployment', 'duration_seconds', 67.90013029607996, '_result', 6, 'dev', 'worker', 'v1.1.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-09-04 16:09:29.046290+00:00', 'deployment', 'rollback', 0.0, '_result', 28, 'production', 'web', 'v1.0.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-09-04 16:09:29.046290+00:00', 'deployment', 'duration_seconds', 179.28659732554084, '_result', 27, 'production', 'web', 'v1.0.0');
INSERT INTO deployment_metrics (time, measurement, field, value, result, table, environment, service, version) VALUES ('2025-09-04 16:09:29.046290+00:00', 'deployment', 'status', 1.0, '_result', 29, 'production', 'web', 'v1.0.0');
