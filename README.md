# Stock Price Analysis & Prediction Pipeline

An end-to-end machine learning pipeline for stock price analysis, feature engineering, and prediction using Gradient Boosting and ARIMA time series models.

## Project Overview

This project implements a complete workflow for stock market analysis, including data validation, cleaning, feature engineering, exploratory analysis, and predictive modeling. The pipeline processes historical stock data for five major companies and generates price predictions using machine learning.

## Stocks Analyzed

- **AAPL** - Apple Inc.
- **MSFT** - Microsoft Corporation
- **AMZN** - Amazon.com Inc.
- **TSLA** - Tesla Inc.
- **JPM** - JPMorgan Chase & Co.

## Project Structure

```
Stock Pipeline/
├── data/
│   ├── raw/                          # Raw CSV files from data sources
│   │   ├── AAPL_historical_data.csv
│   │   ├── MSFT_historical_data.csv
│   │   ├── AMZN_historical_data.csv
│   │   ├── TSLA_historical_data.csv
│   │   └── JPM_historical_data.csv
│   └── processed/                     # Cleaned and processed data
│       ├── *_clean.parquet           # Cleaned data files
│       ├── *_features.parquet        # Feature-engineered datasets
│       └── *_ml.parquet              # ML-ready datasets
│
├── scripts/                           # Python automation scripts
│   ├── 01_validate_raw_data.py       # Data validation and quality checks
│   ├── 02_clean_data.py              # Data cleaning and preprocessing
│   ├── 03_feature_engineering.py     # Technical indicators and features
│   ├── 04_prepare_ml_dataset.py      # ML dataset preparation
│   └── 05_pipeline.py                # Complete training pipeline
│
├── notebooks/                         # Jupyter notebooks for analysis
│   ├── 01_EDA_test.ipynb             # Initial EDA on Tesla data
│   ├── 02_EDA_all.ipynb              # Comprehensive EDA for all stocks
│   ├── 03_Traing testing.ipynb       # Model training and testing
│   └── 04_ARIMA.ipynb                # ARIMA time series modeling
│
├── models/                            # Saved trained models
│   ├── AAPL_gbr_model.pkl
│   ├── MSFT_gbr_model.pkl
│   ├── AMZN_gbr_model.pkl
│   ├── TSLA_gbr_model.pkl
│   └── JPM_gbr_model.pkl
│
└── results/                           # Prediction results
    ├── AAPL_predictions.parquet
    ├── MSFT_predictions.parquet
    ├── AMZN_predictions.parquet
    ├── TSLA_predictions.parquet
    └── JPM_predictions.parquet
```

## Pipeline Workflow

### 1. Data Validation (`01_validate_raw_data.py`)
- Validates raw CSV files for completeness
- Checks for required columns: Date, Open, High, Low, Close, Volume
- Identifies missing values and duplicate dates
- Verifies date range consistency
- Reports data quality metrics

### 2. Data Cleaning (`02_clean_data.py`)
- Converts dates to datetime format with timezone handling
- Handles missing values with forward-fill strategy
- Removes rows with negative prices
- Sorts data chronologically
- Converts to efficient Parquet format
- **Output**: `*_clean.parquet` files

### 3. Feature Engineering (`03_feature_engineering.py`)
Creates comprehensive technical indicators:

#### Price-Based Features:
- **Returns**: Daily percentage returns
- **Log Returns**: Logarithmic returns for normality
- **Lagged Prices**: Lag1, Lag3, Lag5 for temporal patterns
- **Moving Averages**: MA5, MA10, MA20

#### Volatility Indicators:
- **Rolling Volatility**: 5-day, 10-day, 20-day standard deviations
- **Volume Moving Averages**: VolMA5, VolMA10, VolMA20

#### Technical Indicators:
- **RSI (14)**: Relative Strength Index
- **MACD**: Moving Average Convergence Divergence
- **Signal Line**: 9-period EMA of MACD
- **MACD Histogram**: Difference between MACD and Signal

#### Target Variable:
- **Target**: Next day's closing price (shifted -1)

**Output**: `*_features.parquet` files with 28 columns

### 4. ML Dataset Preparation (`04_prepare_ml_dataset.py`)
- Removes rows with NaN values (due to rolling windows)
- Selects 24 features for modeling:
  - OHLCV (Open, High, Low, Close, Volume)
  - Returns and Log Returns
  - Lagged prices (Lag1, Lag3, Lag5)
  - Moving averages (MA5, MA10, MA20)
  - Volatility measures (3 periods)
  - Volume averages (3 periods)
  - RSI14
  - MACD indicators (MACD, Signal, MACD_Hist)
- **Output**: `*_ml.parquet` files ready for training

### 5. Model Training Pipeline (`05_pipeline.py`)

The training pipeline:
1. Loads ML-ready datasets
2. Performs 80/20 time-based train-test split
3. Trains Gradient Boosting Regressor
4. Evaluates using RMSE, MAE, and MAPE metrics
5. Saves trained models as `.pkl` files
6. Stores predictions alongside actual values

Model configuration:
```python
GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    random_state=42
)
```

## Jupyter Notebooks

### 01_EDA_test.ipynb
- Initial exploratory analysis on Tesla stock data
- Data loading and visualization
- Feature distribution analysis
- Time series plotting

### 02_EDA_all.ipynb
- Comprehensive EDA for all 5 stocks
- **Visualizations**:
  - Close price trends over time
  - Volume analysis
  - Moving averages (20-day and 50-day)
- Comparative analysis across stocks

### 03_Training_testing.ipynb
- Train/test split demonstration
- Gradient Boosting model training
- Model evaluation and metrics
- Feature importance analysis
- Top 15 most important features identified

### 04_ARIMA.ipynb
Time series forecasting using ARIMA:
- Stationarity testing with Augmented Dickey-Fuller test
- First-order differencing to achieve stationarity
- ACF/PACF analysis for model parameter selection
- ARIMA(1,1,1) model trained on 204 samples, tested on 46
- Results: RMSE: 20.05, MAE: 17.05, MAPE: 3.91%

## Technologies Used

### Core Libraries
- **pandas** (2.3.2): Data manipulation and analysis
- **numpy** (1.26.4): Numerical computations
- **scikit-learn** (1.5.1): Machine learning algorithms
- **statsmodels** (0.14.5): Statistical modeling and ARIMA

### Visualization
- **matplotlib** (3.10.6): Plotting and visualization
- **seaborn** (0.13.2): Statistical data visualization

### Data Storage
- **fastparquet** (2024.11.0): Efficient Parquet file handling
- **joblib**: Model serialization

### Other Tools
- **jupyter**: Interactive notebooks
- **git**: Version control

## Installation & Setup

### Prerequisites
- Python 3.12+ (recommended)
- Conda or pip package manager
- 2GB+ RAM recommended

### Environment Setup

1. **Clone the repository**:
```bash
git clone https://github.com/kartikeya-m1/Stock-Pipline.git
cd Stock-Pipeline
```

2. **Create virtual environment**:
```bash
# Using conda
conda create -n stock-pipeline python=3.12
conda activate stock-pipeline

# Or using venv
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**:
```bash
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn fastparquet joblib jupyter
```

### Alternative: Install from requirements.txt
If a requirements.txt is available:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Complete Pipeline

Execute scripts in order:

```bash
# Step 1: Validate raw data
python scripts/01_validate_raw_data.py

# Step 2: Clean the data
python scripts/02_clean_data.py

# Step 3: Engineer features
python scripts/03_feature_engineering.py

# Step 4: Prepare ML datasets
python scripts/04_prepare_ml_dataset.py

# Step 5: Train models and generate predictions
python scripts/05_pipeline.py
```

### Running Individual Scripts

**Data Validation**:
```bash
python scripts/01_validate_raw_data.py
```
Output: Console report of data quality metrics

**Data Cleaning**:
```bash
python scripts/02_clean_data.py
```
Output: `data/processed/*_clean.parquet` files

**Feature Engineering**:
```bash
python scripts/03_feature_engineering.py
```
Output: `data/processed/*_features.parquet` files

**ML Dataset Preparation**:
```bash
python scripts/04_prepare_ml_dataset.py
```
Output: `data/processed/*_ml.parquet` files

**Model Training**:
```bash
python scripts/05_pipeline.py
```
Output:
- `models/*_gbr_model.pkl` (trained models)
- `results/*_predictions.parquet` (predictions)
- Console: Evaluation metrics for each stock

### Using Jupyter Notebooks

1. **Start Jupyter**:
```bash
jupyter notebook
```

2. **Navigate to notebooks folder**

3. **Open desired notebook**:
   - `01_EDA_test.ipynb` - For initial data exploration
   - `02_EDA_all.ipynb` - For comprehensive visualization
   - `03_Traing testing.ipynb` - For model training experiments
   - `04_ARIMA.ipynb` - For time series forecasting

## Model Performance

### Gradient Boosting Regressor
- **Algorithm**: Gradient Boosting
- **Train/Test Split**: 80/20 time-based
- **Features**: 24 technical indicators
- **Metrics**: RMSE, MAE, MAPE

### ARIMA Model (Tesla Example)
- **Model**: ARIMA(1,1,1)
- **Training Size**: 204 days
- **Test Size**: 46 days
- **Performance**:
  - RMSE: 20.05
  - MAE: 17.05
  - MAPE: 3.91%

## Key Features

### Data Quality
- Comprehensive validation pipeline
- Automated missing value handling
- Timezone-aware datetime processing
- Duplicate detection and removal

### Feature Engineering
- 24+ technical indicators
- Multiple time-window features
- Price momentum indicators
- Volume-based features

### Modeling
- Multiple approaches: Gradient Boosting and ARIMA
- Time-based train/test splitting
- Model persistence for deployment
- Comprehensive evaluation metrics

### Visualization
- Time series trend analysis
- Moving average overlays
- Volume analysis charts
- ACF/PACF plots for time series

## Known Issues & Solutions

### PyArrow DLL Error (Windows)
If you encounter `DLL load failed while importing lib` error with pyarrow:

**Solution**: Use fastparquet instead
```bash
pip uninstall pyarrow -y
pip install fastparquet
```

### Pandas Future Warnings
Some pandas methods (like `fillna(method='ffill')`) show deprecation warnings in newer versions but don't affect functionality.

## Contributing

Contributions are welcome. Feel free to submit a Pull Request.

### Development Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## Contact

**Author**: Kartikeya Mudliyar  
**GitHub**: [@kartikeya-m1](https://github.com/kartikeya-m1)  
**Repository**: [Stock-Pipline](https://github.com/kartikeya-m1/Stock-Pipline)

## License

This project is open source and available for educational purposes.

## Acknowledgments

Thanks to the open-source community for the libraries used in this project, and to various financial data providers for historical stock data.

## Future Enhancements

- Add more technical indicators (Bollinger Bands, ATR)
- Implement LSTM/Deep Learning models
- Real-time data fetching integration
- Interactive dashboard with Plotly/Dash
- Sentiment analysis from news sources
- Portfolio optimization features
- Backtesting framework
- API deployment for predictions

