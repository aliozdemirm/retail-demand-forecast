"""Gerçek veriyle AYNI FORMATTA, tamamen uydurma örnek veri üretir.

Amaç: depoyu indiren biri gerçek (gizli) veri olmadan da pipeline'ı
çalıştırabilsin. Buradaki hiçbir sayı gerçek bir şirket verisine dayanmaz;
mevsimsellik/trend/gürültü rastgele üretilir.

Çalıştırma:  python scripts/generate_sample_data.py
Çıktı:       sample_data/{kadin,erkek}_{satis,stok}.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent.parent / "sample_data"
SEED = 42
LAST_YEAR, LAST_YEAR_WEEKS = 2026, 34          # son yıl yarım: 34. haftada biter
YEARS = [2023, 2024, 2025, LAST_YEAR]

# kategori -> (taban satış, mevsimsel genlik, tepe haftası, yıllık trend)
CATEGORIES = {
    "SEGMENT_A": (900, 0.25, 38, 0.10),   # okula dönüş civarı zirve
    "SEGMENT_B": (700, 0.30, 44, 0.05),   # kış zirvesi
    "SEGMENT_C": (500, 0.80, 27, 0.02),   # çok mevsimsel (yaz zirvesi)
    "SEGMENT_D": (1100, 0.10, 20, 0.15),  # sakin, büyüyen seri
}
BRANDS_PER_CATEGORY = 4


def _weeks():
    return [(y, w) for y in YEARS for w in range(1, (LAST_YEAR_WEEKS if y == LAST_YEAR else 52) + 1)]


def _gen(gender: str, rng: np.random.Generator):
    sales_rows, stock_rows = [], []
    gender_scale = 1.0 if gender == "Kadin" else 0.8
    weeks = _weeks()
    for cat, (base, amp, peak, trend) in CATEGORIES.items():
        for b in range(BRANDS_PER_CATEGORY):
            brand = f"BRAND_{cat[-1]}{b + 1}"
            brand_scale = rng.uniform(0.4, 1.6) / BRANDS_PER_CATEGORY
            stores0 = int(rng.integers(150, 350))
            stock_level = None
            for i, (y, w) in enumerate(weeks):
                season = 1 + amp * np.cos(2 * np.pi * (w - peak) / 52)
                growth = 1 + trend * i / 52
                promo = 1.35 if w in (47, 48) else 1.0       # kasım kampanyası benzeri
                noise = rng.lognormal(0, 0.12)
                sales = int(max(0, base * gender_scale * brand_scale * season * growth * promo * noise))
                sales_rows.append((y, w, gender, cat, brand, sales))
                target_stock = sales * rng.uniform(5, 9)          # ~5-9 haftalık örtü
                stock_level = target_stock if stock_level is None else 0.8 * stock_level + 0.2 * target_stock
                stores = stores0 + int(5 * np.sin(i / 20))
                stock_rows.append((y, w, gender, cat, brand, int(stock_level), stores))
    return sales_rows, stock_rows


def _thousands(n: int) -> str:
    """Gerçek dosya formatı: 6322 -> '6.322' (loader thousands='.' ile okur)."""
    return f"{n:,}".replace(",", ".")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    for gender, prefix in (("Kadin", "kadin"), ("Erkek", "erkek")):
        sales, stock = _gen(gender, rng)
        pd.DataFrame(sales, columns=["Year", "Week", "GroupDesc", "StockGroupDesc", "BrandDesc", "SatisSayisi"]) \
            .to_csv(OUT_DIR / f"{prefix}_satis.csv", sep=";", index=False)
        df_stock = pd.DataFrame(stock, columns=["Year", "Week", "GroupDesc", "StockGroupDesc", "BrandDesc",
                                                "StokSayisi", "MagazaSayisi"])
        df_stock["StokSayisi"] = df_stock["StokSayisi"].map(_thousands)
        df_stock.to_csv(OUT_DIR / f"{prefix}_stok.csv", sep=";", index=False)
    print(f"Örnek veri yazıldı: {OUT_DIR}")


if __name__ == "__main__":
    main()
