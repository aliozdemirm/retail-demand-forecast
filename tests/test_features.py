"""features.py'nin ürettiği feature setinin temel sağlamlık kontrolleri."""
import json
from pathlib import Path

import pytest

from src.data_loader import get_iso_monday
from src.tuning import get_feature_cols

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def test_no_leakage_columns(df_model_ready):
    """Contemporaneous StokSayisi/MagazaSayisi modele girmemeli -- sadece lag'li
    versiyonları (lag_stock_1/lag_store_1) girer."""
    assert "StokSayisi" not in df_model_ready.columns
    assert "MagazaSayisi" not in df_model_ready.columns
    assert "lag_stock_1" in df_model_ready.columns
    assert "lag_store_1" in df_model_ready.columns


def test_no_missing_values(df_model_ready):
    """build_features() sonunda dropna() var -- eksik satır kalmamalı.

    İstisna: prophet_tahmin/patchtst_tahmin, walk-forward'ın MIN_TRAIN_WEEKS
    sınırından önceki satırlarda kasıtlı olarak NaN kalır (bkz. src/features.py
    Bölüm 8, src/aux_forecasters.py) -- dropna() bu iki sütunu hariç tutuyor.
    """
    diger_sutunlar = [c for c in df_model_ready.columns if c not in ("prophet_tahmin", "patchtst_tahmin")]
    assert df_model_ready[diger_sutunlar].isna().sum().sum() == 0


def test_categorical_dtypes(df_model_ready):
    """Gender/GroupDesc/StockGroupDesc LightGBM/XGBoost için 'category' tipinde olmalı."""
    for col in ["Gender", "GroupDesc", "StockGroupDesc"]:
        assert str(df_model_ready[col].dtype) == "category"


def test_series_count(df_model_ready):
    """Proje boyunca "8 seri" (Gender x StockGroupDesc) diye bahsedilen sayı değişmemeli."""
    n_series = df_model_ready[["Gender", "StockGroupDesc"]].drop_duplicates().shape[0]
    assert n_series == 8


def test_get_iso_monday_dogru_pazartesiyi_uretir():
    """Year+Week -> o ISO haftasının Pazartesi'si. Hafta başlangıcı kaymışsa
    tüm zaman ızgarası (ve dolayısıyla lag'ler) kayar -- ucuz ama kritik kontrol."""
    for yil, hafta in [(2023, 1), (2024, 1), (2025, 1), (2026, 1), (2026, 34)]:
        gun = get_iso_monday(yil, hafta)
        assert gun.weekday() == 0, f"{yil}-W{hafta} Pazartesi değil: {gun}"
        iso = gun.isocalendar()
        assert (iso.year, iso.week) == (yil, hafta)


@pytest.mark.skipif(
    not (MODELS_DIR / "ensemble_config.json").exists(), reason="models/ henüz eğitilmemiş"
)
def test_feature_set_kayitli_modelle_ayni(df_model_ready):
    """build_features()'in ürettiği feature listesi, models/ altındaki üretim
    modellerinin eğitildiği listeyle BİREBİR (sıra dahil) aynı olmalı.

    2026-09-14 denetiminde elle doğrulanan şey buydu; sessizce bozulmasın diye
    teste bağlandı. Bozulursa kayıtlı .joblib'ler yanlış sütunlarla beslenir.
    """
    with open(MODELS_DIR / "ensemble_config.json", encoding="utf-8") as f:
        kayitli = json.load(f)["feature_cols"]
    assert get_feature_cols(df_model_ready) == kayitli


@pytest.mark.skipif(
    not (MODELS_DIR / "ensemble_config.json").exists(), reason="models/ henüz eğitilmemiş"
)
def test_panel_feature_sozlugu_eksiksiz():
    """app.py'deki FEATURE_METADATA, üretim modelinin her feature'ını kapsamalı ve
    artık var olmayan bir feature'ı ("hayalet") içermemeli -- yoksa XAI sekmesi
    ya boş açıklama gösterir ya da olmayan bir değişkeni anlatır."""
    import re

    app_src = (Path(__file__).resolve().parent.parent / "app.py").read_text(encoding="utf-8")
    meta_bloku = app_src.split("FEATURE_METADATA = {", 1)[1].split("\n}", 1)[0]
    meta_keys = set(re.findall(r"^\s*'([A-Za-z0-9_]+)':\s*\{", meta_bloku, re.M))

    with open(MODELS_DIR / "ensemble_config.json", encoding="utf-8") as f:
        kayitli = set(json.load(f)["feature_cols"])

    assert not (kayitli - meta_keys), f"Açıklaması eksik feature: {sorted(kayitli - meta_keys)}"
    assert not (meta_keys - kayitli), f"Hayalet (artık kullanılmayan) feature: {sorted(meta_keys - kayitli)}"
