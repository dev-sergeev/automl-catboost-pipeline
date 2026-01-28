'Base transformer class for preprocessing.'

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class BaseTransformer(ABC):
    'Abstract base class for all transformers.'

    def __init__(self):
        self.is_fitted_: bool = False
        self._params: dict = {}

    @abstractmethod
    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> 'BaseTransformer':
        'Fit the transformer to the data.'
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        'Transform the data.'
        pass

    def fit_transform(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        'Fit and transform in one step.'
        return self.fit(df, y).transform(df)

    def get_params(self) -> dict:
        'Get transformer parameters for serialization.'
        return self._params.copy()

    def set_params(self, params: dict) -> 'BaseTransformer':
        'Set transformer parameters from serialization.'
        self._params = params.copy()
        self.is_fitted_ = True
        return self

    def _check_is_fitted(self) -> None:
        'Check if transformer is fitted.'
        if not self.is_fitted_:
            raise RuntimeError(
                f'{self.__class__.__name__} must be fitted before transform'
            )
