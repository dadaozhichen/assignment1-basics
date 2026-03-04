import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

from .pretokenization import TokenizerTrainer

__all__ = ['TokenizerTrainer']
