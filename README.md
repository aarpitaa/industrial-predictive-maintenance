# Industrial Predictive Maintenance

Predicting Remaining Useful Life (RUL) of aircraft turbofan engines using the NASA C-MAPSS dataset.

## Dataset
This project uses NASA's C-MAPSS Turbofan Engine Degradation Simulation dataset.
Download it from the [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
and place the extracted files in `data/CMAPSSData/`.

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Project structure
- `notebooks/` — exploratory analysis and modeling notebooks
- `src/` — reusable data loading and feature engineering code
- `data/` — raw dataset (not tracked in git — see Dataset section above)

## Status
🚧 In progress — currently at EDA stage.
