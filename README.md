# Hierarchical Reasoning Model (HRM) for Portfolio Allocation

This repository contains a deep-learning-based framework for portfolio allocation using a custom **Hierarchical Reasoning Model (HRM)**. The model is designed to allocate weights across the Dow Jones Industrial Average (DJ30) tickers by reasoning through asset-level features and market regimes.

To ensure stability and reduce the variance inherent in neural network training, the final portfolio weights are generated through an **ensemble average** of 50 independent training runs.

## Overview

The allocation process follows a structured pipeline:
1. **Data Acquisition**: Fetches historical price and volume data for DJ30 tickers (and optionally NASDAQ-100/S&P 500) via Yahoo Finance and Wikipedia.
2. **Feature Engineering**: Calculates technical indicators including RSI, Z-scores, moving averages (MA10, MA20, MA50), and multi-period returns.
3. **Hierarchical Modeling**: A three-level PyTorch neural network that:
    - Encodes individual asset features.
    - Predicts the market regime (Bull, Bear, or Neutral).
    - Allocates weights based on the combined understanding of assets and the predicted regime.
4. **Ensemble Averaging**: Runs the training process multiple times and averages the results to produce stable, reliable allocation weights.

## Code Structure

- **Data Fetching**: Uses `yfinance` for price data and `requests`/`pandas` to scrape ticker lists from Wikipedia.
- **Technical Indicators**: Implements custom functions for RSI and Z-score normalization.
- **Model Architecture (`HierarchicalFinanceHRM`)**: 
    - **Level 1 (Asset Encoder)**: Linear layers that process raw features into asset embeddings.
    - **Level 2 (Regime Predictor)**: Aggregates asset embeddings to identify the current market environment.
    - **Level 3 (Portfolio Allocator)**: A Softmax-based layer that outputs the final weights for each ticker.
- **Training Loop (`train_once`)**: Optimizes a custom objective function focusing on the **Reward-to-Downside Ratio** (mean return divided by downside deviation).
- **Ensemble Execution**: Iterates the training process (default 50 runs) and computes the mean weights across all runs.

## requirements.txt

These are the libraries required to run the notebook, including those used for data scraping, numerical analysis, and deep learning.

yfinance
pandas
numpy
torch
matplotlib
requests
lxml
beautifulsoup4
scipy



## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/hrm-portfolio-allocation.git](https://github.com/yourusername/hrm-portfolio-allocation.git)
   cd hrm-portfolio-allocation
