"""Projeye özel (veri setine bağlı) ayarlar.

Kategori adları gibi veri setine özgü değerler kodun içine gömülmez; burada
genel yer tutucularla tanımlanır. Gerçek değerler, git'e GİRMEYEN yerel
`local_config.json` dosyasından okunur (proje kökünde, ör.:
{"bts_segments": ["KATEGORI_1", "KATEGORI_2"]}). Dosya yoksa yer tutucular
kullanılır; kod yine çalışır, sadece ilgili etkinlik bayrağı hiçbir seride
tetiklenmez.
"""
import json
from pathlib import Path

_LOCAL_CONFIG_PATH = Path(__file__).resolve().parent.parent / "local_config.json"

_DEFAULTS = {
    # Okula dönüş (BTS) etkisinin uygulandığı ürün kategorileri (StockGroupDesc değerleri)
    "bts_segments": ["SEGMENT_A", "SEGMENT_B"],
}


def _load() -> dict:
    cfg = dict(_DEFAULTS)
    if _LOCAL_CONFIG_PATH.exists():
        with open(_LOCAL_CONFIG_PATH, encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg


_CFG = _load()

BTS_SEGMENTS = tuple(_CFG["bts_segments"])
