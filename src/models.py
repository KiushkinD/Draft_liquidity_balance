# src/models.py
"""Модели: бейзлайны, SARIMAX-обёртка, ML-пайплайны и реестр для тюнинга."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")


# ========================
# 1. Бейзлайны (наивные модели)
# ========================

class NaiveLastValue(BaseEstimator, RegressorMixin):
    """
    Наивный прогноз: последнее известное значение целевой переменной.
    """
    def fit(self, X, y):
        self.last_value_ = y.iloc[-1] if hasattr(y, 'iloc') else y[-1]
        return self

    def predict(self, X):
        return np.full(len(X), self.last_value_)


class SeasonalNaive(BaseEstimator, RegressorMixin):
    """
    Сезонный наивный прогноз: значение за 5 бизнес-дней до прогнозируемого.
    Для многомерного прогноза (например, в бэктесте) требуется,
    чтобы в X присутствовал индекс или дата, по которым можно восстановить
    позицию в обучающей выборке. Упрощённо: при n > 1 предсказывается
    последнее известное сезонное значение (для статического прогноза на один шаг).
    """
    def fit(self, X, y):
        self.y_ = np.asarray(y)
        return self

    def predict(self, X):
        n = len(X)
        if len(self.y_) < 5:
            return np.full(n, np.mean(self.y_))
        # Для одиночного прогноза на следующий день
        if n == 1:
            return np.array([self.y_[-5]])
        else:
            # Для нескольких прогнозов используем последнее доступное сезонное значение
            # (в реальном бэктесте нужно подставлять соответствующие лаги)
            return np.full(n, self.y_[-5])


# ========================
# 2. SARIMAX wrapper (авторегрессия + экзогены)
# ========================

class SARIMAXWrapper(BaseEstimator, RegressorMixin):
    """
    Обёртка для SARIMAX, совместимая с интерфейсом sklearn.
    Параметры:
        order : (p,d,q)
        seasonal_order : (P,D,Q,s)
        trend : str, 'c', 't', 'ct' и т.д.
    """
    def __init__(self, order=(1,0,1), seasonal_order=(0,0,0,0), trend='c'):
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.model_ = None
        self.fit_result_ = None

    def fit(self, X, y):
        # X – экзогенные переменные (DataFrame или 2D массив)
        # y – целевая переменная (Series или 1D массив)
        self.model_ = SARIMAX(
            endog=y,
            exog=X,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.fit_result_ = self.model_.fit(disp=False)
        return self

    def predict(self, X):
        # X – экзогенные переменные для периода прогноза
        return self.fit_result_.forecast(steps=len(X), exog=X)

    def get_params(self, deep=True):
        return {
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'trend': self.trend
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


# ========================
# 3. ML-модели внутри sklearn.Pipeline
# ========================

def make_elasticnet_pipeline(alpha=1.0, l1_ratio=0.5):
    # ElasticNet штрафует коэффициенты -> признаки масштабируем всегда.
    return Pipeline([
        ('scaler', StandardScaler()),
        ('enet', ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)),
    ])


def make_rf_pipeline(n_estimators=100, max_depth=None, min_samples_split=2):
    """
    RandomForestRegressor.
    """
    return Pipeline([
        ('rf', RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1
        ))
    ])


def make_lgb_pipeline(n_estimators=100, learning_rate=0.1, num_leaves=31,
                      subsample=0.8, colsample_bytree=0.8,
                      reg_alpha=0.0, reg_lambda=0.0):
    """
    LightGBM – основной кандидат.
    """
    return Pipeline([
        ('lgb', lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ))
    ])


# ========================
# 4. Словарь всех моделей с сетками гиперпараметров
# ========================

def get_all_models():
    """
    Возвращает словарь:
        {имя_модели: (экземпляр_модели, словарь_параметров_для_поиска)}
    """
    models = {
        # Бейзлайны (без гиперпараметров)
        'naive': (NaiveLastValue(), {}),
        'seasonal_naive': (SeasonalNaive(), {}),

        # SARIMAX
        'sarimax': (
            SARIMAXWrapper(order=(1,0,1), seasonal_order=(0,0,0,0)),
            {
                'order': [(1,0,1), (2,0,1), (1,0,2), (2,0,2)],
                'seasonal_order': [(0,0,0,0), (1,0,1,5)]
            }
        ),

        # ElasticNet
        'elasticnet': (
            make_elasticnet_pipeline(),
            {
                'enet__alpha': [0.01, 0.1, 1.0, 10.0],
                'enet__l1_ratio': [0.2, 0.5, 0.8, 1.0],
            }
        ),

        # Random Forest
        'rf': (
            make_rf_pipeline(),
            {
                'rf__n_estimators': [50, 100, 200],
                'rf__max_depth': [None, 10, 20],
                'rf__min_samples_split': [2, 5, 10]
            }
        ),

        # LightGBM
        'lgb': (
            make_lgb_pipeline(),
            {
                'lgb__n_estimators': [50, 100, 200],
                'lgb__learning_rate': [0.01, 0.05, 0.1],
                'lgb__num_leaves': [15, 31, 63],
                'lgb__subsample': [0.7, 0.8, 1.0],
                'lgb__colsample_bytree': [0.7, 0.8, 1.0]
            }
        )
    }
    return models