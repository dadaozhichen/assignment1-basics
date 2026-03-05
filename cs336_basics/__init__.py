import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

from .pretokenization import TokenizerTrainer
from .tokenizer import Tokenizer

__all__ = ['TokenizerTrainer','Tokenizer']
