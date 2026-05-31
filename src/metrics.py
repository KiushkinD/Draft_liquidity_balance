# src/metrics.py
"""Бизнес-метрика (асимметричный P&L-loss) и MAE + scorer'ы для sklearn/Optuna."""

import numpy as np
from sklearn.metrics import make_scorer as sklearn_make_scorer
from functools import partial
def pnl_loss(y_true, y_pred, key_rate=None):
    """
    Бизнес-метрика, отражающая финансовые потери от ошибки прогноза сальдо.

   
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

    return sklearn_make_scorer(
        partial(pnl_loss, key_rate=key_rate),
        greater_is_better=False
    )


# Дополнительно: готовый scorer для MAE
mae_scorer = sklearn_make_scorer(mae, greater_is_better=False)