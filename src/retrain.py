from dataclasses import dataclass

import pandas as pd


@dataclass
class RetrainPolicy:
    calibration_frequency_days: int = 21
    hyperopt_frequency_days: int = 63
    min_train_days: int = 500
    training_window_days: int = None

    def should_retrain(
        self,
        current_date,
        last_fit_date,
        drift_alarm: bool = False,
    ) -> tuple[bool, str]:
        current_date = pd.Timestamp(current_date)

        if last_fit_date is None:
            return True, "initial_fit"

        last_fit_date = pd.Timestamp(last_fit_date)

        if drift_alarm:
            return True, "drift_alarm"

        days_since_fit = (current_date - last_fit_date).days

        if days_since_fit >= self.calibration_frequency_days:
            return True, "scheduled_retrain"

        return False, "no_retrain"

    def should_hyperopt(
        self,
        current_date,
        last_hyperopt_date,
    ) -> tuple[bool, str]:
        current_date = pd.Timestamp(current_date)

        if last_hyperopt_date is None:
            return True, "initial_hyperopt"

        last_hyperopt_date = pd.Timestamp(last_hyperopt_date)

        days_since_hyperopt = (current_date - last_hyperopt_date).days

        if days_since_hyperopt >= self.hyperopt_frequency_days:
            return True, "scheduled_hyperopt"

        return False, "no_hyperopt"

    def get_training_slice(self, data, current_date):
        current_date = pd.Timestamp(current_date)

        train_data = data.loc[data.index < current_date].copy()

        if self.training_window_days is not None:
            start_date = current_date - pd.Timedelta(days=self.training_window_days)
            train_data = train_data.loc[train_data.index >= start_date]

        return train_data
