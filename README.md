# my-masters-aint-shit
basics, automization, gaming scripts and some financial modeling

## MNQ Opening Range Breakout Backtest

`mnq_orb_backtest.py` provides a simple opening range breakout backtest for the
Micro E-mini Nasdaq-100 (MNQ) futures contract.  The script can download data
via [yfinance](https://github.com/ranaroussi/yfinance) or use a local CSV file
with minute-level bars.

Example usage:

```bash
# Download and run a 15 minute opening range breakout test
python mnq_orb_backtest.py --download --interval 15m --orb-minutes 15

# Run on your own dataset
python mnq_orb_backtest.py --csv path/to/data.csv
```

The script prints per-day results and total profit and loss in index points.
For MNQ each point equals $2.
