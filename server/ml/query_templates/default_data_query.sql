-- Enrollment Predictor Custom Data Query Template
-- Update the WHERE clause or add joins to shape the training dataset.
-- Ensure the query returns all columns required by your chosen feature schema.
-- Example below limits data to terms with a year of 2023 or later.

SELECT *
FROM section_detail_report_sbussection_detail_report_sbus
WHERE (term / 100) >= 2023;
