"""Загрузка и подготовка временного ряда сальдо ликвидности.

- load_raw: чтение Excel-файла Проект.xlsx.
- to_business_days: фильтрация выходных и праздников РФ.
- time_split: разрезание на train/val/test по датам.
"""

from pathlib import Path

import holidays
import pandas as pd


def load_raw(path: str | Path) -> pd.DataFrame:
    """Прочитать лист Data и привести типы. Возвращает DataFrame [date, income, outcome, balance]."""
    df = pd.read_excel(path, sheet_name="Data")
    df.columns = ["date", "income", "outcome", "balance"]
    df["date"] = pd.to_datetime(df["date"])
    for c in ("income", "outcome", "balance"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return df.sort_values("date").reset_index(drop=True)


def to_business_days(df: pd.DataFrame, year_range=range(2017, 2025)) -> pd.DataFrame:
    """Отфильтровать выходные и официальные праздники РФ."""
    ru_holidays = set(holidays.RU(years=list(year_range)).keys())
    is_weekend = df["date"].dt.weekday >= 5
    is_holiday = df["date"].dt.date.isin(ru_holidays)
    return df.loc[~(is_weekend | is_holiday)].reset_index(drop=True)


def time_split(df: pd.DataFrame, train_end: str, val_end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Разрезать по двум календарным датам.

    train: date <= train_end
    val:   train_end < date <= val_end
    test:  date > val_end
    """
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)
    train = df[df["date"] <= train_end_ts]
    val = df[(df["date"] > train_end_ts) & (df["date"] <= val_end_ts)]
    test = df[df["date"] > val_end_ts]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
