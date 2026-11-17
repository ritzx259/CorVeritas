# CDC codebook verification

Reviewed against the official August 2021–August 2023 codebooks on 2026-10-23.
The downloader additionally saves HTML snapshots and hashes, checks cycle and
column names, and validates each XPT schema before merging. Offline XPT-only use
relies on this recorded verification; absent HTML hashes are explicitly null.

| Module and official codebook | Variables used | Eligibility / interpretation |
|---|---|---|
| [MCQ_L](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/MCQ_L.htm) | `MCQ160c`, `MCQ160b`, `MCQ160e` | Each question: age 20+, ever told by a doctor/health professional of CHD, CHF, heart attack. 1 Yes; 2 No; 7 Refused; 9 Don't know; dot missing. Only 1/2 create labels. |
| [DEMO_L](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DEMO_L.htm) | `RIDAGEYR`, `RIAGENDR` | All participants; age is top-coded at 80. Sex/gender variable: 1 Male, 2 Female. `RIDSTATR` distinguishes interview-only and examined participants. `WTMEC2YR`, `SDMVSTRA`, `SDMVPSU` are audit-only survey metadata. |
| [BMX_L](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BMX_L.htm) | `BMXBMI` | BMI available from age 2+, kg/m²; measured during MEC exam. |
| [BPXO_L](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BPXO_L.htm) | `BPXOSY1..3`, `BPXODI1..3` | Age 8+, oscillometric MEC measurements, mmHg, three readings. CDC protocol requires systolic > diastolic, systolic ≤300, no diastolic without systolic. Average valid available readings independently; missing all readings remains missing. |
| [TCHOL_L](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/TCHOL_L.htm) | `LBXTC` | Examined age 6+, total cholesterol mg/dL. Do not feed mmol/L without conversion. `WTPH2YR` is an audit-only phlebotomy weight, recommended by CDC for blood-analyte population analyses. |
| [DIQ_L](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DIQ_L.htm) | `DIQ010` | Age 1+, ever told of diabetes; excludes pregnancy-only diabetes in adult females. 1 Yes, 2 No, 3 Borderline, 7 Refused, 9 Don't know. Borderline remains a distinct category. |
| [SMQ_L](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/SMQ_L.htm) | `SMQ020` | This specific question is age 18+ although the module includes younger participants. At least 100 cigarettes in lifetime: 1 Yes, 2 No, 7 Refused, 9 Don't know. |

`SEQN` is used only for one-to-one joins and reproducible participant partitions.
Every final predictor is explicitly allowlisted. No MCQ diagnosis, medication,
age-at-diagnosis, cardiovascular follow-up, survey ID or UCI-specific predictor
enters the model.

The smoking feature intentionally measures lifetime exposure only. `SMQ040`
would be needed to distinguish current/former smokers, but cessation can follow
a cardiovascular diagnosis. It is excluded to reduce that particular temporal
shortcut. Inference accepts “Former smoker” and “Current smoker” as aliases for
Yes with a warning that the frontend must confirm 100 lifetime cigarettes.
“Never smoker” maps to No. No means fewer than 100 lifetime cigarettes; it does
not necessarily mean zero tobacco exposure. Other tobacco use is not measured.

Clinical engineering bounds in `config.py` are deliberately broad fixed guards,
not CDC reference intervals: BMI 10–100, systolic 50–300, diastolic 20–200,
cholesterol 40–1000. Invalid observed inputs become missing during preparation
and are counted; inference rejects invalid supplied values. Each BP pair is
checked before averaging. Age 20–120 is accepted at inference and clipped to 80
to reproduce public-data top-coding. Missing age is rejected at inference.

Temporal leakage cannot be eliminated: BMI, BP and cholesterol may have been
measured after diagnosis and treatment; diabetes timing and lifetime smoking
timing relative to the outcomes are unknown. These features support a
cross-sectional association task only. Feature importance cannot establish
which factors caused a disease.

CDC recommends design variables and appropriate sample weights for population
analyses. This implementation retains MEC and phlebotomy weights for audit,
but deliberately reports **unweighted sample ML metrics**, including all
eligible complete-label adults and imputing unavailable measurements. It is
not a design-based population estimator. Adding population inference requires
a separate sampling/nonresponse analysis, appropriate weight selection,
stratum/PSU-aware uncertainty estimation and potentially clustered validation.
