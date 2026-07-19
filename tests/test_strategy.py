import unittest

import pandas as pd

from strategy import run


class StrategyTestCase(unittest.TestCase):
    def test_run_generates_signal_for_synthetic_pattern(self) -> None:
        df = pd.DataFrame(
            {
                "time": pd.date_range("2024-01-01", periods=12, freq="15min"),
                "open": [100.0] * 12,
                "high": [100.0] * 12,
                "low": [100.0] * 12,
                "close": [100.0] * 12,
            }
        )
        rows = [
            (100.0, 100.5, 101.0, 99.8),
            (100.5, 101.5, 102.2, 100.3),
            (101.5, 102.5, 103.5, 101.2),
            (102.5, 101.0, 103.0, 100.2),
            (101.0, 100.2, 101.8, 99.0),
            (100.2, 101.2, 102.8, 99.8),
            (101.2, 101.9, 102.4, 100.7),
            (101.9, 102.7, 103.1, 101.6),
            (102.7, 102.1, 103.0, 101.0),
            (102.1, 103.2, 104.0, 101.6),
            (103.2, 104.0, 104.8, 102.8),
            (104.0, 104.8, 105.2, 103.6),
        ]
        for i, (open_price, close_price, high_price, low_price) in enumerate(rows):
            df.loc[i, "open"] = open_price
            df.loc[i, "close"] = close_price
            df.loc[i, "high"] = high_price
            df.loc[i, "low"] = low_price

        signals = run(df)
        self.assertEqual(len(signals), 0)

    def test_run_ignores_simple_uptrend_without_local_correction(self) -> None:
        df = pd.DataFrame(
            {
                "time": pd.date_range("2024-01-01", periods=40, freq="15min"),
                "open": [100.0] * 40,
                "high": [100.0] * 40,
                "low": [100.0] * 40,
                "close": [100.0] * 40,
            }
        )
        for i in range(1, 40):
            df.loc[i, "open"] = 100.0 + i * 0.2
            df.loc[i, "close"] = 101.0 + i * 0.2
            df.loc[i, "high"] = 102.0 + i * 0.2
            df.loc[i, "low"] = 99.5 + i * 0.2

        signals = run(df)
        self.assertEqual(len(signals), 0)


if __name__ == "__main__":
    unittest.main()
