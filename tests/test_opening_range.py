import pandas as pd

from config import DEFAULT_PARAMS, StrategyParams
from strategies.opening_range import compute_opening_range, generate_signal
from tests.conftest import make_day_bars


def test_opening_range_high_low(long_breakout_day):
    orr = compute_opening_range(long_breakout_day, "TEST", or_minutes=15)
    assert orr is not None
    assert orr.or_high == 101.0
    assert orr.or_low == 99.0
    assert orr.or_mid == 100.0
    assert orr.or_width == 2.0


def test_incomplete_day_returns_none():
    # Only 5 bars -- can't form a 15-minute opening range.
    idx = pd.date_range("2024-01-10 09:30", periods=5, freq="1min", tz="America/New_York")
    df = pd.DataFrame(
        {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 100, "symbol": "TEST"}, index=idx
    )
    assert compute_opening_range(df, "TEST", or_minutes=15) is None


def test_long_breakout_detected(long_breakout_day):
    orr = compute_opening_range(long_breakout_day, "TEST", or_minutes=15)
    sig = generate_signal(long_breakout_day, orr, DEFAULT_PARAMS, volume_baseline=None)
    assert sig is not None
    assert sig.direction == "long"
    assert sig.signal_price > orr.or_high


def test_short_breakout_detected(short_breakout_day):
    orr = compute_opening_range(short_breakout_day, "TEST", or_minutes=15)
    sig = generate_signal(short_breakout_day, orr, DEFAULT_PARAMS, volume_baseline=None)
    assert sig is not None
    assert sig.direction == "short"
    assert sig.signal_price < orr.or_low


def test_no_breakout_no_signal(no_breakout_day):
    orr = compute_opening_range(no_breakout_day, "TEST", or_minutes=15)
    sig = generate_signal(no_breakout_day, orr, DEFAULT_PARAMS, volume_baseline=None)
    assert sig is None


def test_signal_is_the_first_breakout_only():
    """If price breaks out, pulls back inside the range, then breaks out again,
    the signal must be the FIRST breakout bar -- not a later, possibly more
    dramatic one. Anything else would implicitly use information (the shape of
    the whole day) not available at the time of the first breakout.
    """
    df = make_day_bars("2024-01-10", breakout_direction="long", breakout_minute_offset=20, post_breakout_drift=0.05)
    # Inject a second, larger breakout later that should NOT be selected.
    df.iloc[100, df.columns.get_loc("close")] = 150.0
    df.iloc[100, df.columns.get_loc("high")] = 150.5

    orr = compute_opening_range(df, "TEST", or_minutes=15)
    sig = generate_signal(df, orr, DEFAULT_PARAMS, volume_baseline=None)
    assert sig.signal_time == df.index[20]


def test_no_signal_after_cutoff_time():
    """A breakout occurring after the cutoff time must not generate a signal,
    even though it is a valid breakout in every other respect."""
    df = make_day_bars("2024-01-10", breakout_direction="long", breakout_minute_offset=100)  # 09:30+100min = 11:10
    orr = compute_opening_range(df, "TEST", or_minutes=15)
    params = StrategyParams(cutoff_time="11:00")
    sig = generate_signal(df, orr, params, volume_baseline=None)
    assert sig is None


def test_signal_scan_never_uses_future_bars():
    """Directly verifies the no-look-ahead property: generate_signal on a
    truncated version of the day (bars only up to and including the true
    breakout bar) must produce the IDENTICAL signal as on the full day --
    proving nothing after the signal bar can affect the signal itself.
    """
    df = make_day_bars("2024-01-10", breakout_direction="long", breakout_minute_offset=20)
    orr = compute_opening_range(df, "TEST", or_minutes=15)
    params = DEFAULT_PARAMS

    full_day_signal = generate_signal(df, orr, params, volume_baseline=None)
    truncated = df.iloc[:21]  # up to and including the breakout bar (index 20)
    truncated_signal = generate_signal(truncated, orr, params, volume_baseline=None)

    assert full_day_signal.signal_time == truncated_signal.signal_time
    assert full_day_signal.signal_price == truncated_signal.signal_price
    assert full_day_signal.direction == truncated_signal.direction


def test_relative_volume_threshold_filters_low_volume_days():
    df = make_day_bars("2024-01-10", breakout_direction="long", or_volume_per_min=1_000)
    orr = compute_opening_range(df, "TEST", or_minutes=15)
    params = StrategyParams(relative_volume_threshold=2.0)
    # baseline of 20,000 total OR volume vs actual 15,000 (1,000/min * 15) -> relative volume 0.75 < 2.0
    sig = generate_signal(df, orr, params, volume_baseline=20_000)
    assert sig is None


def test_relative_volume_threshold_passes_high_volume_days():
    df = make_day_bars("2024-01-10", breakout_direction="long", or_volume_per_min=10_000)
    orr = compute_opening_range(df, "TEST", or_minutes=15)
    params = StrategyParams(relative_volume_threshold=2.0)
    # actual OR volume = 150,000 vs baseline 20,000 -> relative volume 7.5 >= 2.0
    sig = generate_signal(df, orr, params, volume_baseline=20_000)
    assert sig is not None
