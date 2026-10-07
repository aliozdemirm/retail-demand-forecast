"""Kaydedilmiş üretim modellerinin (models/) temel sağlamlık kontrolleri."""
import json
from pathlib import Path

import joblib
import pytest

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


@pytest.mark.skipif(not (MODELS_DIR / "ensemble_config.json").exists(), reason="models/ henüz eğitilmemiş")
def test_ensemble_weights_sum_to_one():
    with open(MODELS_DIR / "ensemble_config.json", encoding="utf-8") as f:
        config = json.load(f)
    toplam = sum(config["final_weights"].values())
    assert abs(toplam - 1.0) < 1e-6


@pytest.mark.skipif(not (MODELS_DIR / "ensemble_config.json").exists(), reason="models/ henüz eğitilmemiş")
def test_ensemble_weights_non_negative():
    with open(MODELS_DIR / "ensemble_config.json", encoding="utf-8") as f:
        config = json.load(f)
    for w in config["final_weights"].values():
        assert w >= 0


@pytest.mark.skipif(not (MODELS_DIR / "meta_model.joblib").exists(), reason="meta_model henüz eğitilmemiş")
def test_meta_model_coefficients_valid():
    """Stacking meta-modelinin katsayıları negatif olmamalı ve toplamı ~1 olmalı
    (bkz. 2026-09-14 degenere-ağırlık bugfix'i -- docs/degisiklik_gecmisi.md)."""
    meta_model = joblib.load(MODELS_DIR / "meta_model.joblib")
    coefs = meta_model.coef_
    assert (coefs >= -1e-9).all()
    assert abs(coefs.sum() - 1.0) < 1e-6
