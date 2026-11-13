# Table `patients` (read-only SQLite, built from the cleaned Stroke Prediction Dataset)

| column | type | notes |
|---|---|---|
| id | INTEGER | dataset row id (not a patient identifier) |
| gender | TEXT | Female / Male |
| age | REAL | years |
| hypertension | INTEGER | 0/1 |
| heart_disease | INTEGER | 0/1 |
| ever_married | TEXT | Yes / No |
| work_type | TEXT | Private / Self-employed / Govt_job / children / Never_worked |
| Residence_type | TEXT | Urban / Rural |
| avg_glucose_level | REAL | mg/dL |
| bmi | REAL | NULL when missing in the source |
| smoking_status | TEXT | never smoked / formerly smoked / smokes / Unknown |
| stroke | INTEGER | 0/1 dataset label (history, no timing) |

Example: `SELECT COUNT(*) AS n, SUM(stroke) AS strokes FROM patients WHERE age BETWEEN 60 AND 69 AND hypertension = 1`
