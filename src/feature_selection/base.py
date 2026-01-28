"Base class for feature selectors."

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd


class BaseSelector(ABC):
    "Abstract base class for feature selectors."

    def __init__(self):
        self.selected_features_: list[str] = []
        self.removed_features_: list[str] = []
        self.is_fitted_: bool = False
        self._params: dict = {}

    @abstractmethod
    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[np.ndarray] = None
    ) -> "BaseSelector":
        "Fit the selector to identify features to keep/remove."
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        "Select features from DataFrame."
        self._check_is_fitted()
        available = [f for f in self.selected_features_ if f in X.columns]
        return X[available]

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        "Fit and transform in one step."
        return self.fit(X, y, sample_weight).transform(X)

    def get_selected_features(self) -> list[str]:
        "Get list of selected features."
        return self.selected_features_.copy()

    def get_removed_features(self) -> list[str]:
        "Get list of removed features."
        return self.removed_features_.copy()

    def get_params(self) -> dict:
        "Get selector parameters for serialization."
        return {
            "selected_features": self.selected_features_,
            "removed_features": self.removed_features_,
            **self._params,
        }

    def set_params(self, params: dict) -> "BaseSelector":
        "Set selector parameters from serialization."
        self.selected_features_ = params.get("selected_features", [])
        self.removed_features_ = params.get("removed_features", [])
        self._params = params
        self.is_fitted_ = True
        return self

    def _check_is_fitted(self) -> None:
        "Check if selector is fitted."
        if not self.is_fitted_:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before transform"
            )
