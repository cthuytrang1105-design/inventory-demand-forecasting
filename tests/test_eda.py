import pandas as pd
from src.eda import distribution_summary, holiday_effect, run_eda


def test_holidays_respect_locale_and_transferred(sample, tmp_path):
    date = sample.date.min()
    rows = [(date, "Holiday", "Local", "City1", False),
            (date + pd.Timedelta(days=1), "Holiday", "National", "Ecuador", True),
            (date + pd.Timedelta(days=2), "Work Day", "National", "Ecuador", False),
            (date + pd.Timedelta(days=3), "Holiday", "Regional", "State2", False)]
    holidays, stores = tmp_path / "holidays.csv", tmp_path / "stores.csv"
    pd.DataFrame(rows, columns=["date", "type", "locale", "locale_name", "transferred"]).to_csv(holidays, index=False)
    pd.DataFrame([(1, "City1", "State1"), (2, "City2", "State2"), (3, "City3", "State3")],
                 columns=["store_nbr", "city", "state"]).to_csv(stores, index=False)
    df, _ = holiday_effect(sample, holidays, stores)
    flagged = df[df.holiday]
    assert len(flagged) == 6
    assert flagged[flagged.date == date].store_nbr.eq(1).all()
    assert flagged[flagged.date == date + pd.Timedelta(days=3)].store_nbr.eq(2).all()


def test_eda_outputs(tmp_path):
    from src.demo_data import generate_demo
    path = tmp_path / "demo.csv"
    generate_demo(path, days=120)
    result = run_eda(path, tmp_path / "results")
    assert result["source"]["simulated"]
    assert result["rows"] == 1080
    assert len(list((tmp_path / "results/figures").glob("eda_*.html"))) == 6


def test_distribution_uses_all_values():
    frame = pd.DataFrame({"group": ["A"] * 5, "sales": [0, 1, 2, 3, 100]})
    summary = distribution_summary(frame, "group").iloc[0]
    assert summary["count"] == 5
    assert (summary.q1, summary["median"], summary.q3) == (1, 2, 3)
    assert summary.upperfence == 3
