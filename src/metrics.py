# src/metrics.py
"""Бизнес-метрика (асимметричный P&L-loss) и MAE + scorer'ы для sklearn/Optuna."""

import numpy as np
from sklearn.metrics import make_scorer as sklearn_make_scorer
from functools import partial
def pnl_loss(y_true, y_pred, key_rate=None):
    """
    Бизнес-метрика, отражающая финансовые потери от ошибки прогноза сальдо.

    Параметры
    ----------
    y_true : array-like
        Фактические значения баланса.
    y_pred : array-like
        Прогнозные значения баланса.
    key_rate : float or array-like, optional
        Ключевая ставка (годовых). В текущей реализации не используется,
        оставлена для совместимости с сигнатурой из ТЗ.

    Возвращает
    -------
    float
        Средние дневные потери (в денежных единицах, нормированные на 1 рубль баланса).

    Формула потерь
    --------------
    - Перепрогноз (ŷ > y):   (ŷ - y) * 0.5% / 365
    - Недопрогноз (ŷ < y):   (y - ŷ) * 1.4% / 365
    - Иначе: 0

    Потери асимметричны: недопрогноз в 2,8 раза дороже перепрогноза.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    diff = y_pred - y_true

    # 0.5% годовых → дневная ставка
    over_penalty = 0.5 / 365.0
    # 1.4% годовых → дневная ставка
    under_penalty = 1.4 / 365.0

    loss = np.where(diff > 0,
                    diff * over_penalty,
                    np.where(diff < 0,
                             -diff * under_penalty,
                             0.0))
    return np.mean(loss)


def mae(y_true, y_pred):
    """Средняя абсолютная ошибка (MAE)."""
    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))


def make_scorer(key_rate=None):
    """
    Создаёт scorer на основе pnl_loss для использования в sklearn / Optuna.

    Параметры
    ----------
    key_rate : float, optional
        Значение ключевой ставки (фиксированное). В текущей реализации игнорируется.

    Возвращает
    -------
    callable
        Scorer с сигнатурой scorer(estimator, X, y) -> float.
    """
    return sklearn_make_scorer(
        partial(pnl_loss, key_rate=key_rate),
        greater_is_better=False
    )


# Дополнительно: готовый scorer для MAE
mae_scorer = sklearn_make_scorer(mae, greater_is_better=False)