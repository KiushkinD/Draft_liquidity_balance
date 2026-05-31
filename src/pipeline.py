from dataclasses import dataclass, field

import pandas as pd

from src.drift import DriftMonitor
from src.retrain import RetrainPolicy


@dataclass
class ForecastService:
    model: object
    feature_builder: object
    retrain_policy: RetrainPolicy = field(default_factory=RetrainPolicy)
    drift_monitor: DriftMonitor = field(default_factory=DriftMonitor)

    target_col: str = "balance"
    date_col: str = "date"

    fit_date: pd.Timestamp = None
    last_hyperopt_date: pd.Timestamp = None
    feature_set: list = None
    hyperparams: dict = field(default_factory=dict)

    predictions_log: list = field(default_factory=list)
    drift_log: list = field(default_factory=list)
    retrain_log: list = field(default_factory=list)

    def fit(self, data: pd.DataFrame, current_date=None):
        if current_date is None:
            current_date = data.index.max()

        current_date = pd.Timestamp(current_date)

        train_data = self.retrain_policy.get_training_slice(data, current_date)

        X, y = self.feature_builder.transform_train(
            train_data,
            target_col=self.target_col,
        )

        self.model.fit(X, y)

        self.fit_date = current_date

        self.retrain_log.append(
            {
                "date": current_date,
                "reason": "manual_or_initial_fit",
                "n_train": len(train_data),
            }
        )

        return self

    def predict(self, data: pd.DataFrame, date):
        date = pd.Timestamp(date)

        X_pred = self.feature_builder.transform_predict(
            data,
            date=date,
        )

        y_pred = float(self.model.predict(X_pred)[0])

        next_date = X_pred.index[0] if hasattr(X_pred, "index") else None

        self.predictions_log.append(
            {
                "prediction_date": date,
                "target_date": next_date,
                "y_pred": y_pred,
            }
        )

        return y_pred

    def update(
        self, data: pd.DataFrame, actual_date, actual_value, y_pred, is_holiday=False
    ):
        actual_date = pd.Timestamp(actual_date)

        drift_result = self.drift_monitor.update(
            date=actual_date,
            y_true=actual_value,
            y_pred=y_pred,
            is_holiday=is_holiday,
        )

        self.drift_log.append(drift_result)

        should_retrain, reason = self.retrain_policy.should_retrain(
            current_date=actual_date,
            last_fit_date=self.fit_date,
            drift_alarm=drift_result["drift_alarm"],
        )

        if should_retrain:
            self.refit(data, actual_date, reason)

        return {
            "drift": drift_result,
            "retrained": should_retrain,
            "reason": reason,
        }

    def refit(self, data: pd.DataFrame, current_date, reason: str):
        current_date = pd.Timestamp(current_date)

        train_data = self.retrain_policy.get_training_slice(data, current_date)

        X, y = self.feature_builder.transform_train(
            train_data,
            target_col=self.target_col,
        )

        self.model.fit(X, y)

        self.fit_date = current_date

        self.retrain_log.append(
            {
                "date": current_date,
                "reason": reason,
                "n_train": len(train_data),
            }
        )

    def evaluate(self, data: pd.DataFrame, start, end):
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)

        results = []

        eval_dates = data.loc[start:end].index

        for current_date in eval_dates:
            historical_data = data.loc[:current_date].copy()

            y_pred = self.predict(historical_data, current_date)

            actual_value = data.loc[current_date, self.target_col]

            is_holiday = False
            if "is_holiday" in data.columns:
                is_holiday = bool(data.loc[current_date, "is_holiday"])

            update_result = self.update(
                data=data.loc[:current_date].copy(),
                actual_date=current_date,
                actual_value=actual_value,
                y_pred=y_pred,
                is_holiday=is_holiday,
            )

            results.append(
                {
                    "date": current_date,
                    "y_true": actual_value,
                    "y_pred": y_pred,
                    "abs_error": abs(actual_value - y_pred),
                    "drift_alarm": update_result["drift"]["drift_alarm"],
                    "retrained": update_result["retrained"],
                    "retrain_reason": update_result["reason"],
                }
            )

        return pd.DataFrame(results).set_index("date")
