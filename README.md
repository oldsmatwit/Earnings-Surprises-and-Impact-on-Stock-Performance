# Analysis of Earnings Surprises and Their Impact on Stock Performance

## Project Overview
This project examines the relationship between quarterly corporate earnings surprises and subsequent stock performance among S&P 500 companies. It investigates whether larger positive or negative earnings surprises can help predict market-adjusted stock returns over different time horizons.

The analysis uses historical earnings announcements, stock price data, and machine learning models to evaluate predictive performance across 1-, 3-, 5-, and 20-trading-day periods. It also examines model performance during major market disruptions, including the 2008 financial crisis and the 2020 COVID-19 market crash.

## Repository Structure

### `Pipeline/`
Contains the Python scripts used to collect, process, analyze, and model the data.

The pipeline includes:
- S&P 500 company selection and sampling.
- Historical earnings data collection and earnings surprise calculations.
- Stock price and market benchmark collection.
- Dataset construction and exploratory data analysis.
- Machine learning model development and evaluation.
- Neural network implementation and model comparisons.
- Financial crisis stress testing and additional feature analysis.
- Results summaries and reporting tables.

### `Figures/`
Contains the visual outputs generated throughout the project, including exploratory data analysis, earnings surprise comparisons, model performance, feature importance, and crisis analysis.

### `S&P Data/`
Contains the CSV dataset used in the project, including the historical S&P 500 stock and earnings data used for analysis.

## Methodology
The project follows a multi-stage data pipeline:

1. **Company Selection:** Selects approximately 250 S&P 500 companies using sector, market capitalization, trading frequency, and company age to create a balanced sample.
2. **Data Collection:** Retrieves historical quarterly earnings announcements and daily stock prices.
3. **Feature Engineering:** Calculates earnings surprises, standardized unexpected earnings (SUE), historical company performance, and market-related features.
4. **Dataset Construction:** Aligns earnings announcements with trading days and calculates market-adjusted cumulative returns.
5. **Modeling:** Evaluates linear regression, logistic regression, random forests, and neural networks.
6. **Evaluation:** Compares predictive performance across multiple return windows using chronological training and testing periods.
7. **Stress Testing:** Examines whether earnings surprise relationships and model performance change during major financial crises.

## Return Measurements
Market-adjusted cumulative returns (CAR) are calculated over four periods:
- 1 trading day
- 3 trading days
- 5 trading days
- 20 trading days

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- yfinance
- Alpha Vantage
- SciPy

## Research Objective
The primary objective is to determine whether earnings surprises provide meaningful predictive information about future stock performance and whether more complex machine learning models improve upon traditional approaches.
