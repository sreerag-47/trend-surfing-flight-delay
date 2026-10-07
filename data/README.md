# Dataset Directory

This directory contains flight delay datasets for Trend Surfing analysis.

## Included Files
- `DelayedFlights_sample.csv`: 50,000-row sample for instant benchmarking, CLI commands, and automated test suite.
- `DelayedFlights_sample.zip`: ZIP archive containing the sample CSV, used to test and verify upload handling of compressed archives.

## Full Dataset
The full `DelayedFlights.csv` (1.93M rows, ~247 MB) exceeds GitHub's 100 MB file size limit and is excluded via `.gitignore`.

To use the full dataset:
1. Download `DelayedFlights.csv` from [Kaggle Airline Delay Causes](https://www.kaggle.com/datasets/giovannidata/delayedflights) or the Harvard Dataverse.
2. Place `DelayedFlights.csv` inside this `data/` folder.
3. Run Trend Surfing:
   ```bash
   python main.py surf data/DelayedFlights.csv --algorithm surfer2 --depth 3
   ```
