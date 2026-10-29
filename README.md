# Kenya Construction Cost Forecast

Forecasting Kenya's construction input costs using the KNBS Construction Input Price Index, with application to the Affordable Housing Programme (AHP).

<img width="3400" height="3443" alt="image" src="https://github.com/user-attachments/assets/a478d010-5858-4bb4-accc-23d4ea03ae7c" />

## Overview

This project:

1. **Compiles** quarterly Construction Input Price Index (CIPI) data from the Kenya National Bureau of Statistics (KNBS)
2. **Analyses** trends in overall, building, and civil engineering construction costs (Q4 2019 – Q2 2026)
3. **Forecasts** future index values using ARIMA and Exponential Smoothing models
4. **Translates** forecast movements into cost sensitivity estimates for AHP-type housing units

## Data Source

| Field | Detail |
|---|---|
| **Publisher** | Kenya National Bureau of Statistics (KNBS) |
| **Series** | Construction Input Price Indices (CIPI) |
| **Base period** | December 2019 = 100 |
| **Frequency** | Quarterly |
| **Coverage** | Q4 2019 (base) through Q2 2026 (27 observations) |
| **Sub-indices** | Overall, Building Works, Civil Engineering, plus component sub-indices (materials, transport/fuel, labour, equipment) |
| **Access** | [KNBS Construction Input PI](https://www.knbs.or.ke/button/construction-input-pi/) |
| **Example release** | [Q2 2025 PDF](https://www.knbs.or.ke/wp-content/uploads/2025/07/Construction-Input-Price-Indices-Second-Quarter-2025.pdf) |

### Definitions

- **Overall Construction Input Price Index**: Weighted composite of building and civil engineering input costs
- **Building Cost Index**: Input cost variations for residential and non-residential building construction
- **Civil Engineering Cost Index**: Input cost variations for infrastructure, roads, earthworks, and heavy civil structures
- **Sub-indices**: Materials (cement, steel, sand, timber, etc.), Transport/Fuel/Lubricants, Labour (by skill category), Equipment (machinery rental/ownership)

### Key Context

- The index rose only 0.5% through 2025 (Business Daily), then spiked 5.73% in Q2 2026 alone, driven by a 21.47% surge in the transport/fuel sub-index (Capmad, July 2026)
- AHP units are priced at fixed ranges (e.g., 1BR Ksh 1.5M–2.1M), so construction cost volatility directly affects programme viability

## Methodology

### Models

1. **ARIMA**: Multiple orders tested (grid search on p, d, q), selected by AIC on training data
2. **Exponential Smoothing**: Holt's linear trend, damped trend, and Holt-Winters seasonal variants tested, selected by AIC

### Evaluation

- **Holdout**: Last 6 quarters withheld for out-of-sample testing
- **Metrics**: MAE, RMSE, MAPE
- **Forward forecast**: 4 quarters ahead with 90% confidence intervals

### Honest Limitations

- ~26 quarterly data points is a small sample for time series modelling
- The Q2 2026 fuel spike is a structural outlier that heavily influences recent forecasts
- Confidence intervals widen rapidly; treat forecasts as directional indicators, not precise predictions
- Sub-index data (fuel, materials, labour) is available as percentage changes from reports but not as complete absolute time series without PDF extraction from all individual quarterly releases

## Project Structure

```
kenya-construction-cost-forecast/
├── README.md                          ← This file
├── requirements.txt                   ← Python dependencies
├── forecast_analysis.py               ← Main analysis pipeline
├── data/
│   └── raw/
│       ├── construction_input_price_index.csv  ← Core dataset (27 quarterly observations)
│       └── sub_indices_detail.csv              ← Sub-index detail (Q4 2025 – Q2 2026)
└── output/
    ├── figures/
    │   ├── 01_eda_overview.png
    │   ├── 02_stationarity_analysis.png
    │   ├── 03_holdout_evaluation.png
    │   ├── 04_forward_forecast.png
    │   └── 05_cost_sensitivity.png
    ├── processed_index_data.csv
    ├── cost_sensitivity.csv
    └── forecast_report.txt
```

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full analysis pipeline
python forecast_analysis.py
```

## Sources

All data points are sourced from publicly available KNBS publications. Specific sources for each row are documented in the CSV files. Media sources used for cross-validation:

- Capmad (July 2026): [Construction costs spike](https://www.capmad.com/post/kenyas-construction-costs-spike-at-the-fastest-pace-in-four-years-as-fuel-prices-surge)
- Business Daily: [Construction inflation at 0.5% in 2025](https://www.mwanaspoti.co.tz/bd/markets/real-estate/developers-relief-on-softened-construction-inflation-5442188)
- Nation: [AHP unit prices](https://nation.africa/kenya/business/sh206-billion-poured-into-ruto-s-housing-fund-in-three-years-5557238)

## AHP Unit Cost Assumptions

Used for cost sensitivity translation (verify against original sources):

| Unit Type | Base Cost (Ksh) | Source |
|---|---|---|
| Affordable 1BR | 1,500,000 | Nation |
| Affordable 2BR | 2,000,000 | Nation |
| Affordable 3BR | 3,000,000 | Nation |
| Social | 1,500,000 | Derived from KNBS Economic Survey 2026 |
| Institutional | 2,300,000 | Derived from KNBS Economic Survey 2026 |

## License

This project uses publicly available government data for analytical purposes.
