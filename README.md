The Research questions can be found in the paper "Investigating Correlations in the Stock Market Using Multi-Source Data and Tensor-Based Models"

# Investigating Correlations in the Stock Market Using Multi-Source Data and Tensor-Based Models

This repository contains the code and experimental pipeline for the Master's thesis:
> *Investigating Correlations in the Stock Market Using Multi-Source Data and Tensor-Based Models*  
> University of Amsterdam, Data Science Master's Programme, 2025  
> Author: Björn van Engelenburg

## Overview

This project explores whether **inter-stock correlations**, combined with **financial news sentiment**, can enhance **stock price prediction** using **tensor-based machine learning models**. It compares multiple modeling approaches, including:

- Linear Regression (Baseline & Extended)
- Random Walk
- LSTM (Baseline & Extended)
- Bayesian Tensor Regression (Baseline & Extended)

The extended models incorporate **rolling correlation features** across five major tickers: `AAPL`, `GOOGL`, `AMZN`, `MSFT`, `TSLA`.

## Key Features

-  Multi-source input: price data, news sentiment (via BERT), and dynamic inter-stock correlations
-  Tensor decomposition via CP/PARAFAC for interpretable latent structure
-  Benchmarks against traditional and deep learning models (e.g., LSTM)
-  Modular, reproducible code (Python, PyTorch, Scikit-learn, HuggingFace Transformers)


