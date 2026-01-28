'AutoML CatBoost Pipeline for bank scoring tasks.'

from src.config import PipelineConfig
from src.pipeline import CatBoostPipeline
from src.utils import NoFeaturesRemainingError

__all__ = ['PipelineConfig', 'CatBoostPipeline', 'NoFeaturesRemainingError']
__version__ = '0.1.0'
