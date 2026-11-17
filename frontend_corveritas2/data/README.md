# Dataset source

The application and portable training launcher inspect the existing dataset at
`../../data/heart_disease_uci.csv` without changing it. It is not copied into
this folder. There are 920 rows, 16 source columns, and 13 model input features.

The three original scripts drop `id`, `dataset` and `num` from model inputs and
create the binary target `(num > 0)`: 509 positive and 411 negative source rows.
The CSV contains Cleveland (304), Hungary (293), VA Long Beach (200) and
Switzerland (123) records. These are dataset records, not newly collected users.

The verified input order is `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`,
`restecg`, `thalch`, `exang`, `oldpeak`, `slope`, `ca`, `thal`.

Missing source values occur in resting blood pressure (59), cholesterol (30),
fasting blood sugar (90), resting ECG (2), maximum heart rate (55), exercise
angina (55), ST depression (62), slope (309), vessels (611) and thal (486).
Age, sex and chest-pain category have no missing values in this CSV. The schema
therefore requires those three fields and accepts null for the remaining ten.

Zero blood pressure/cholesterol and negative ST depression occur in the CSV.
The prototype preserves the original training conventions; the saved models
were not trained with an additional retrospective cleaning policy. Missing
values must be entered as Unknown, never invented or silently replaced with 0.

The categorical spelling `reversable defect` is intentional and matches the
saved encoders. The interface displays “reversible defect”. Boolean source
values are transmitted as JSON `true` or `false`.

This repository does not include a dataset licence or original download
provenance manifest. Confirm the exact distribution's attribution and licence
before redistributing the CSV. The UCI Heart Disease dataset reference is
https://archive.ics.uci.edu/dataset/45/heart+disease.
