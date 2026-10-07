import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_and_preprocess_data
from src.features import build_features


@pytest.fixture(scope="session")
def df_model_ready():
    """build_features() çıktısı -- testler arasında tekrar tekrar hesaplamamak için session-scoped."""
    return build_features(load_and_preprocess_data())
