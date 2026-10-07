"""tuning.py'nin temel altyapı fonksiyonlarının sağlamlık kontrolleri."""
from pathlib import Path

import numpy as np
import pytest

from src.tuning import calculate_metrics, rolling_origin_cv, split_train_val_test

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def test_split_no_overlap(df_model_ready):
    """train/val/test tarih aralıkları çakışmamalı (veri sızıntısı olmasın)."""
    tr, va, te = split_train_val_test(df_model_ready)
    assert tr["Date"].max() < va["Date"].min()
    assert va["Date"].max() < te["Date"].min()


def test_split_covers_all_rows(df_model_ready):
    """3 parçanın toplam satır sayısı orijinal veriyle eşit olmalı (kayıp/mükerrer satır yok)."""
    tr, va, te = split_train_val_test(df_model_ready)
    assert len(tr) + len(va) + len(te) == len(df_model_ready)


def test_calculate_metrics_wape():
    """Bilinen bir girdide WAPE'nin doğru hesaplandığını kontrol et."""
    gercek = np.array([100.0, 200.0, 300.0])
    tahmin = np.array([110.0, 190.0, 300.0])
    # toplam mutlak hata = 10+10+0 = 20, toplam gercek = 600 -> WAPE = %3.33
    sonuc = calculate_metrics(gercek, tahmin, "test")
    assert abs(sonuc["WAPE"] - 3.33) < 0.01


def test_calculate_metrics_bias_direction():
    """Hep fazla tahmin eden model pozitif Bias, hep az tahmin eden negatif Bias vermeli."""
    gercek = np.array([100.0, 100.0])
    fazla_tahmin = np.array([110.0, 110.0])
    az_tahmin = np.array([90.0, 90.0])
    assert calculate_metrics(gercek, fazla_tahmin)["Bias"] > 0
    assert calculate_metrics(gercek, az_tahmin)["Bias"] < 0


@pytest.mark.skipif(
    not (MODELS_DIR / "ensemble_config.json").exists(),
    reason="models/ henüz eğitilmemiş -- best_params/weights kaynağı yok",
)
def test_rolling_origin_cv_runs(df_model_ready, tmp_path):
    """Rolling-origin CV altyapısı (eğitim + recursive backtest + metrik) hatasız çalışmalı."""
    out_path = tmp_path / "cv_results_test.json"
    fold_df, ozet = rolling_origin_cv(df_model_ready, n_folds=1, out_path=out_path)
    assert len(fold_df) == 1
    assert 0 <= ozet["WAPE_ortalama"] < 200
    assert out_path.exists()
