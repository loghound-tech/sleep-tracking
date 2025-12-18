# Sleep Tracking Analysis

This Julia script provides a comprehensive analysis of sleep data exported from Apple Health. It aggregates daily sleep stages, calculates bedtime and wake time trends, and provides a correlation matrix to help you understand your sleep patterns.

Designed to work with Apple Watch sleep data, it assumes you wear your watch at night.

## Data Export
The easiest way to get your sleep data out of Apple Health in CSV format is to use the **"Health Auto Export"** app for iOS. 

The included `csv_files/Sleep Analysis.csv` is an example of the format needed for the script to work properly.

## Installation

### 1. Install Julia
If you don't have Julia installed, download and install it from [julialang.org](https://julialang.org/downloads/).

### 2. Set Up the Project
Clone or download this repository, then navigate to the directory and run:
```bash
julia sleep_analysis.jl
```
This will install all necessary dependencies (CSV, DataFrames, PlotlyJS, etc.). and use the default example file (csv_files/Sleep Analysis.csv which is my sleep data from the last year)

## Usage

Run the script using the default example file:
```bash
julia sleep_analysis.jl
```

Analyze a specific CSV file:
```bash
julia sleep_analysis.jl your_sleep_data.csv
```

Specify a custom reference bedtime (default is 11:00 PM):
```bash
julia sleep_analysis.jl --bedtime 22:30
```

## Assumptions & Logic
- **Sleep Day**: A "Sleep Day" is defined as starting at 6:00 PM the previous evening.
- **Bedtime Window**: To filter out late naps or unusual data points, the script only includes days where the calculated bedtime is within **ref_bedtime +/- 2 hours**.
  - *Example*: With the default 11:00 PM reference, only bedtimes between 9:00 PM and 1:00 AM are analyzed.

## Interpretation Guide

### Correlation Matrix
The correlation matrix helps you see how different metrics affect each other.
- **How to read**: Find your metric on the X-axis and look up to the crossing cell. The number is the correlation coefficient (-1.0 to 1.0).
- **Negative Correlation**: e.g., `-0.5` between Bedtime and Total Sleep Time. 
  - *Meaning*: For every hour **earlier** you go to bed, you sleep on average **30 minutes more**.
- **Positive Correlation**: e.g., `+0.2` for Wake Time.
  - *Meaning*: For every hour you sleep **earlier** (lower bedtime offset), you wake up about **12 minutes earlier**.

### Visualizations
1.  **Daily Sleep Analysis**: A stacked bar chart showing Core, Deep, REM, and Awake durations.
2.  **Bedtime Trend**: A line chart below the bars showing your bedtime relative to midnight.
3.  **Distributions**: Histograms showing the frequency of your bedtimes and wake times.
