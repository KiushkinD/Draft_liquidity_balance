"""Экзогенные факторы: налоговый календарь и макроданные ЦБ/MOEX.

Все макро-источники кешируются в parquet, чтобы повторные запуски не били по внешним API.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

TAX_DAYS_OF_MONTH = (25, 28)
CBR_KEY_RATE_URL = "https://www.cbr.ru/hd_base/KeyRate/"
CBR_RUONIA_URL = "https://www.cbr.ru/hd_base/ruonia/dynamics/"
CBR_XML_DYNAMIC_URL = "https://www.cbr.ru/scripts/XML_dynamic.asp"
MOEX_IMOEX_URL = "https://iss.moex.com/iss/history/engines/stock/markets/index/securities/IMOEX.json"
USDRUB_VAL_CODE = "R01235"
DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0 (academic data fetch)"}


def tax_calendar(start: str, end: str) -> pd.DataFrame:
    """Календарь налоговых дней РФ: флаг + расстояние до/после ближайшего."""
    dates = pd.date_range(start, end, freq="D")
    df = pd.DataFrame({"date": dates})
    df["is_tax_day"] = df["date"].dt.day.isin(TAX_DAYS_OF_MONTH).astype(int)

    tax_dates = df.loc[df["is_tax_day"] == 1, "date"].to_numpy()
    if len(tax_dates) == 0:
        df["days_to_next_tax"] = 30
        df["days_since_last_tax"] = 30
        return df

    all_dates = df["date"].to_numpy()
    next_idx = np.searchsorted(tax_dates, all_dates, side="left")
    prev_idx = np.searchsorted(tax_dates, all_dates, side="right") - 1

    days_to = np.full(len(all_dates), 30, dtype=int)
    has_next = next_idx < len(tax_dates)
    days_to[has_next] = (
        (tax_dates[next_idx[has_next]] - all_dates[has_next]).astype("timedelta64[D]").astype(int)
    )

    days_since = np.full(len(all_dates), 30, dtype=int)
    has_prev = prev_idx >= 0
    days_since[has_prev] = (
        (all_dates[has_prev] - tax_dates[prev_idx[has_prev]]).astype("timedelta64[D]").astype(int)
    )

    df["days_to_next_tax"] = days_to
    df["days_since_last_tax"] = days_since
    return df


def _parse_cbr_table_dates_values(html: str, value_col_index: int = 1) -> pd.DataFrame:
    """Достать пары (date, value) из таблицы на странице ЦБ."""
    rows = re.findall(r"<tr>(.*?)</tr>", html, re.DOTALL)
    out = []
    for row in rows:
        cells = re.findall(r"<td[^>]*>([^<]*)</td>", row)
        if len(cells) <= value_col_index:
            continue
        raw_date = cells[0].strip()
        raw_value = cells[value_col_index].strip()
        try:
            date = pd.to_datetime(raw_date, format="%d.%m.%Y")
        except ValueError:
            continue
        value = raw_value.replace(",", ".").replace("\xa0", "").strip()
        if value in {"", "—", "-"}:
            continue
        try:
            value_f = float(value)
        except ValueError:
            continue
        out.append((date, value_f))
    return pd.DataFrame(out, columns=["date", "value"]).sort_values("date").reset_index(drop=True)


def _fetch_key_rate(start: str, end: str) -> pd.DataFrame:
    """Скачать ключевую ставку с HTML-страницы ЦБ."""
    params = {
        "UniDbQuery.Posted": "True",
        "UniDbQuery.From": pd.Timestamp(start).strftime("%d.%m.%Y"),
        "UniDbQuery.To": pd.Timestamp(end).strftime("%d.%m.%Y"),
    }
    r = requests.get(CBR_KEY_RATE_URL, params=params, headers=DEFAULT_HEADERS, timeout=30)
    r.raise_for_status()
    df = _parse_cbr_table_dates_values(r.text, value_col_index=1)
    df = df.rename(columns={"value": "key_rate"})
    if df.empty:
        raise RuntimeError("CBR key rate parsing returned empty frame")
    # Заполняем все календарные дни в диапазоне (ставка действует до следующей даты).
    full = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    merged = full.merge(df, on="date", how="left").sort_values("date")
    merged["key_rate"] = merged["key_rate"].ffill().bfill()
    return merged.reset_index(drop=True)


def _fetch_ruonia(start: str, end: str) -> pd.DataFrame:
    """RUONIA с HTML-страницы ЦБ. Скачивается порциями по году, чтобы не упереться в лимит."""
    chunks = []
    cur = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    while cur <= end_ts:
        chunk_end = min(cur + pd.DateOffset(years=1) - pd.Timedelta(days=1), end_ts)
        params = {
            "UniDbQuery.Posted": "True",
            "UniDbQuery.From": cur.strftime("%d.%m.%Y"),
            "UniDbQuery.To": chunk_end.strftime("%d.%m.%Y"),
        }
        r = requests.get(CBR_RUONIA_URL, params=params, headers=DEFAULT_HEADERS, timeout=30)
        r.raise_for_status()
        df = _parse_cbr_table_dates_values(r.text, value_col_index=1)
        chunks.append(df)
        cur = chunk_end + pd.Timedelta(days=1)
    out = pd.concat(chunks, ignore_index=True).rename(columns={"value": "ruonia"})
    out = out.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return out


def _fetch_usdrub(start: str, end: str) -> pd.DataFrame:
    """Курс USD/RUB с XML-API ЦБ."""
    params = {
        "date_req1": pd.Timestamp(start).strftime("%d/%m/%Y"),
        "date_req2": pd.Timestamp(end).strftime("%d/%m/%Y"),
        "VAL_NM_RQ": USDRUB_VAL_CODE,
    }
    r = requests.get(CBR_XML_DYNAMIC_URL, params=params, headers=DEFAULT_HEADERS, timeout=30)
    r.raise_for_status()
    text = r.content.decode("windows-1251")
    rows = re.findall(r'<Record Date="([^"]+)"[^>]*>.*?<Value>([^<]+)</Value>', text, re.DOTALL)
    if not rows:
        raise RuntimeError("CBR USDRUB parsing returned empty")
    df = pd.DataFrame(
        [(pd.to_datetime(d, format="%d.%m.%Y"), float(v.replace(",", "."))) for d, v in rows],
        columns=["date", "usdrub"],
    )
    return df.sort_values("date").reset_index(drop=True)


def _fetch_moex(start: str, end: str) -> pd.DataFrame:
    """Закрытие индекса MOEX (IMOEX) — JSON-API биржи, постранично по 100 записей."""
    rows: list[list] = []
    cols: list[str] | None = None
    start_offset = 0
    while True:
        params = {"from": start, "till": end, "start": start_offset}
        r = requests.get(MOEX_IMOEX_URL, params=params, headers=DEFAULT_HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()["history"]
        if cols is None:
            cols = data["columns"]
        batch = data["data"]
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 100:
            break
        start_offset += len(batch)
    if not rows:
        raise RuntimeError("MOEX returned no data")
    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(df["TRADEDATE"])
    df = df[["date", "CLOSE"]].rename(columns={"CLOSE": "moex_close"}).dropna()
    return df.sort_values("date").reset_index(drop=True)


def _cached(fetch_fn, start: str, end: str, cache_path: str | Path) -> pd.DataFrame:
    cache_path = Path(cache_path)
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        cached_min, cached_max = df["date"].min(), df["date"].max()
        if cached_min <= pd.Timestamp(start) and cached_max >= pd.Timestamp(end):
            return df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)
    df = fetch_fn(start, end)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    return df


def load_key_rate(start: str, end: str, cache_path: str | Path = "data/key_rate.parquet") -> pd.DataFrame:
    return _cached(_fetch_key_rate, start, end, cache_path)


def load_ruonia(start: str, end: str, cache_path: str | Path = "data/ruonia.parquet") -> pd.DataFrame:
    return _cached(_fetch_ruonia, start, end, cache_path)


def load_usdrub(start: str, end: str, cache_path: str | Path = "data/usdrub.parquet") -> pd.DataFrame:
    return _cached(_fetch_usdrub, start, end, cache_path)


def load_moex(start: str, end: str, cache_path: str | Path = "data/moex.parquet") -> pd.DataFrame:
    return _cached(_fetch_moex, start, end, cache_path)


def build_exog(start: str, end: str, cache_dir: str | Path = "data") -> pd.DataFrame:
    """Собрать все экзогены: налоговый календарь + макро. Все merge — left, по date."""
    cache_dir = Path(cache_dir)
    cal = tax_calendar(start, end)
    kr = load_key_rate(start, end, cache_path=cache_dir / "key_rate.parquet")
    rn = load_ruonia(start, end, cache_path=cache_dir / "ruonia.parquet")
    mx = load_moex(start, end, cache_path=cache_dir / "moex.parquet")
    fx = load_usdrub(start, end, cache_path=cache_dir / "usdrub.parquet")
    out = cal
    for src in (kr, rn, mx, fx):
        out = out.merge(src, on="date", how="left")
    macro_cols = ["key_rate", "ruonia", "moex_close", "usdrub"]
    out[macro_cols] = out[macro_cols].ffill().bfill()
    out = out.dropna(subset=["key_rate"]).reset_index(drop=True)
    return out
