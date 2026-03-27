# Causal Validation Dataset

## Overview

This repository contains two publicly released datasets designed for benchmarking causal inference methods. Both datasets come from a real-world product setting in which a new software update (`New_Feature`) was deployed to end-user devices, and its effect on device reliability outcomes was measured over a two-week study period.

The two datasets share an identical schema (44 columns) and differ only in how treatment was assigned:


| Dataset | File | Rows | Treatment Assignment |
|---|---|---|---|
| **Experimental** | `FINAL_PUBLIC_experimental.parquet` | 435,170 | Randomized (A/B test) |
| **Observed** | `FINAL_PUBLIC_observed.parquet` | 445,286 | Observational (self-selected adoption) |

Having both a randomized experiment and a parallel observational study on the same population enables researchers to validate observational causal inference estimators against the experimental ground truth.

## Variable Descriptions

### Outcome Variables

| Variable | Type | Description |
|---|---|---|
| `Outcome_Binary` | Integer (0/1) | A binary device reliability outcome measured during the two-week study period. 1 = event occurred, 0 = no event. |
| `Outcome_Continuous` | Float | A continuous device reliability outcome measured during the two-week study period (count). |

### Treatment Variable

| Variable | Type | Description |
|---|---|---|
| `New_Feature` | Boolean | Whether the device received a new software update. `True` = update applied (treated), `False` = update not applied (control). In the experimental dataset, this was randomly assigned; in the observed dataset, adoption was organic. |

### Device Usage Covariates

| Variable | Type | Description |
|---|---|---|
| `Device_Spec_A` | Boolean | A binary device technical specification. |
| `Total_Device_Usage` | Float | Total time the device was in use over the two-week study period, in seconds. |
| `Browser_1_Usage` | Float | Total time spent in browser 1, in seconds. |
| `Browser_2_Usage` | Float | Total time spent in browser 2, in seconds. |
| `Browser_3_Usage` | Float | Total time spent in browser 3, in seconds. |
| `Browser_4_Usage` | Float | Total time spent in browser 4, in seconds. |
| `Other_Browser_Usage` | Float | Total time spent in other browsers, in seconds. |

### Engagement Segment (One-Hot Encoded)

Ordinal usage intensity within a usage cohort. One-hot encoded with the first category dropped. Each device belongs to exactly one segment.

| Variable | Type | Description |
|---|---|---|
| `EngagementSegment_General` | Boolean | General engagement level. |
| `EngagementSegment_Highly Engaged` | Boolean | Highly engaged — the most active users. |
| `EngagementSegment_Inactive` | Boolean | Inactive — minimal device engagement. |
| `EngagementSegment_Low Engaged` | Boolean | Low engagement. |

> When all four indicators are `False`, the device belongs to the dropped reference category.

### Geographic Region (One-Hot Encoded, Scrambled)

Geographic region of the device. **Important**: Region labels have been randomly shuffled for anonymization — the displayed region names do **not** correspond to their true geographic locations.

| Variable | Type | Description |
|---|---|---|
| `A14Region_APAC` | Boolean | Scrambled region indicator. |
| `A14Region_CEE` | Boolean | Scrambled region indicator. |
| `A14Region_Canada` | Boolean | Scrambled region indicator. |
| `A14Region_France` | Boolean | Scrambled region indicator. |
| `A14Region_Germany` | Boolean | Scrambled region indicator. |
| `A14Region_Greater China` | Boolean | Scrambled region indicator. |
| `A14Region_India` | Boolean | Scrambled region indicator. |
| `A14Region_Japan` | Boolean | Scrambled region indicator. |
| `A14Region_Korea` | Boolean | Scrambled region indicator. |
| `A14Region_Latam` | Boolean | Scrambled region indicator. |
| `A14Region_MEA` | Boolean | Scrambled region indicator. |
| `A14Region_UK` | Boolean | Scrambled region indicator. |
| `A14Region_United States` | Boolean | Scrambled region indicator. |
| `A14Region_Western Europe` | Boolean | Scrambled region indicator. |

> When all fourteen indicators are `False`, the device belongs to the dropped reference region.

### App-Usage Cohort (One-Hot Encoded)

Categorical user profile based on application usage patterns. Each device is assigned to exactly one cohort. One-hot encoded with the first category dropped.

| Variable | Type | Description |
|---|---|---|
| `AppCategoryCohort_Developer` | Boolean | Developer usage profile. |
| `AppCategoryCohort_Gamer` | Boolean | Gamer usage profile. |
| `AppCategoryCohort_General` | Boolean | General usage profile. |
| `AppCategoryCohort_Media` | Boolean | Media usage profile. |
| `AppCategoryCohort_Productivity` | Boolean | Productivity usage profile. |

> When all five indicators are `False`, the device belongs to the dropped reference cohort.

### Device Manufacturer (One-Hot Encoded, Anonymized)

Device manufacturer, anonymized with numeric labels. One category represents an aggregation of smaller manufacturers ("Other"). One-hot encoded with the first category dropped.

| Variable | Type | Description |
|---|---|---|
| `Manufacturer_1` | Boolean | Anonymized manufacturer category 1. |
| `Manufacturer_2` | Boolean | Anonymized manufacturer category 2. |
| `Manufacturer_3` | Boolean | Anonymized manufacturer category 3. |
| `Manufacturer_4` | Boolean | Anonymized manufacturer category 4. |
| `Manufacturer_5` | Boolean | Anonymized manufacturer category 5. |
| `Manufacturer_6` | Boolean | Anonymized manufacturer category 6. |

> When all six indicators are `False`, the device belongs to the dropped reference manufacturer category.

### Device Spec B (One-Hot Encoded)

Categorical variable indicating type of device. Anonymized with numeric labels. One-hot encoded with the first category dropped.

| Variable | Type | Description |
|---|---|---|
| `Device_Spec_B_1` | Boolean | Anonymized device-type category 1. |
| `Device_Spec_B_2` | Boolean | Anonymized device-type category 2. |

> When both indicators are `False`, the device belongs to the dropped reference device-type category.

### Device Spec C (One-Hot Encoded)

Categorical device technical specification, anonymized with numeric labels. One-hot encoded with the first category dropped.

| Variable | Type | Description |
|---|---|---|
| `Device_Spec_C_1` | Boolean | Anonymized device spec C category 1. |
| `Device_Spec_C_2` | Boolean | Anonymized device spec C category 2. |
| `Device_Spec_C_3` | Boolean | Anonymized device spec C category 3. |

> When all three indicators are `False`, the device belongs to the dropped reference category.

## File Format

Both datasets are stored as Apache Parquet files, which can be loaded in Python with:

```python
import pandas as pd

observed = pd.read_parquet("FINAL_PUBLIC_observed.parquet")
experimental = pd.read_parquet("FINAL_PUBLIC_experimental.parquet")
```