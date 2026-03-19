# Colab notebooks

These notebooks are intended for Google Colab and cover the main DMN entrypoints, excluding `forecasting_with_lstm.py`.

- `colab_backtest.ipynb`: run the global backtest CLI
- `colab_optimize.ipynb`: run the optimization/grid-search CLI
- `colab_by_ticker.ipynb`: evaluate one strategy ticker by ticker
- `colab_live.ipynb`: train/predict persisted DMN LSTM artifacts with Google Drive

Recommended Google Drive layout:

- `MyDrive/DMN/configs/`: custom TOML/JSON configs used by Colab notebooks
- `MyDrive/DMN/artifacts/dmn/`: saved model artifacts
- `MyDrive/DMN/output/`: CSV outputs and predictions
