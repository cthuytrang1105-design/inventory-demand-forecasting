import numpy as np
import pandas as pd
import pytest
from src.data_processing import KEYS, load_sales, regularize_daily, split_dates, validate_sales
from src.feature_engineering import FEATURES, future_features, make_features


def test_schema_and_sort(sample, tmp_path):
    shuffled = sample.sample(frac=1, random_state=9)
    got = validate_sales(shuffled)
    pd.testing.assert_frame_equal(got, sample)
    path = tmp_path / "new.csv"
    shuffled.to_csv(path, index=False)
    pd.testing.assert_frame_equal(load_sales(path), sample)
    with pytest.raises(ValueError, match="sales"):
        validate_sales(sample.drop(columns="sales"))
    with pytest.raises(ValueError, match="Trùng"):
        validate_sales(pd.concat([sample, sample.head(1)]))


@pytest.mark.parametrize("column,value", [("sales", -1), ("sales", np.nan), ("sales", np.inf),
                                         ("date", "không phải ngày"), ("family", "  "),
                                         ("store_nbr", 1.5), ("onpromotion", -2)])
def test_invalid_data(sample, column, value):
    bad = sample.copy()
    bad[column] = bad[column].astype(object)
    bad.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_sales(bad)


def test_optional_promotion_missing(sample):
    no_promo = validate_sales(sample.drop(columns="onpromotion"))
    assert no_promo.onpromotion.eq(0).all()
    sample.loc[0, "onpromotion"] = np.nan
    assert validate_sales(sample).onpromotion.iloc[0] == 0


def test_no_future_or_current_leakage(sample):
    before = make_features(sample)
    cutoff = sample.date.min() + pd.Timedelta(days=80)
    modified = sample.copy()
    modified.loc[modified.date >= cutoff, "sales"] += 10000
    after = make_features(modified)
    pd.testing.assert_frame_equal(before.loc[before.date <= cutoff, FEATURES], after.loc[after.date <= cutoff, FEATURES])
    # Group isolation: sửa một nhóm không ảnh hưởng nhóm khác.
    modified = sample.copy()
    modified.loc[modified.store_nbr == 1, "sales"] += 10000
    other = make_features(modified)
    pd.testing.assert_frame_equal(before.loc[before.store_nbr == 2, FEATURES], other.loc[other.store_nbr == 2, FEATURES])


def test_missing_calendar_day_not_label(sample):
    group = sample[(sample.store_nbr == 1) & (sample.family == "BEVERAGES")].copy()
    missing_date = group.date.iloc[50]
    removed = group[group.date != missing_date]
    daily = regularize_daily(removed)
    inserted = daily[daily.date == missing_date].iloc[0]
    assert not inserted.is_observed
    assert inserted.sales == group.sales.iloc[49]
    features = make_features(removed)
    row = features[features.date == missing_date + pd.Timedelta(days=7)].iloc[0]
    assert row.lag_7 == inserted.sales


def test_training_inference_feature_parity(sample):
    end = sample.date.max()
    history = sample[sample.date < end]
    keys, states = [], []
    for key, group in history.groupby(KEYS, sort=True):
        keys.append(key)
        states.append(group.sales.tail(28).to_numpy())
    dynamic = future_features(pd.DataFrame(keys, columns=KEYS), end, np.vstack(states))
    train = make_features(sample)
    expected = train.loc[train.date == end, FEATURES].reset_index(drop=True)
    pd.testing.assert_frame_equal(dynamic, expected, check_dtype=False, atol=1e-8)


def test_temporal_split(sample):
    train, val, test = split_dates(sample, 30)
    assert train.date.max() < val.date.min() <= val.date.max() < test.date.min()
    assert (test.date.max() - test.date.min()).days == 29
    with pytest.raises(ValueError):
        split_dates(sample, 60)
