"""README'deki örnek grafiği üretir: örnek (uydurma) veride 12 haftalık backtest.

ÖNEMLİ: Grafik SADECE depodaki uydurma `sample_data/` ile üretilmelidir.
Gerçek `data/` klasörü varsa betik durur; gerçek veriyle grafik üretilip
yanlışlıkla yayınlanmasın diye.

Çalıştırma (repo kökünden, `data/` olmayan temiz bir kopyada):
    python scripts/make_readme_figure.py
Çıktı: assets/forecast_example.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_and_preprocess_data  # noqa: E402
from src.features import build_features  # noqa: E402
from src.model import train_and_validate_ensemble  # noqa: E402

if (ROOT / "data" / "kadin_satis.csv").exists():
    sys.exit("DURDUM: data/ (gerçek veri) bulunuyor. Grafik yalnızca uydurma sample_data ile üretilmeli.")

# Renkler: doğrulanmış kategorik palet, yuva 1 (mavi) ve 2 (turuncu); metin rengi ayrı
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
C_ACTUAL, C_FORECAST = "#2a78d6", "#eb6834"
HISTORY_WEEKS, TEST_WEEKS = 40, 12

df_features = build_features(load_and_preprocess_data())
_, res, _ = train_and_validate_ensemble(df_features, test_weeks=TEST_WEEKS)

gender = "Kadin"
segments = sorted(res.loc[res["Gender"] == gender, "StockGroupDesc"].astype(str).unique())

fig, axes = plt.subplots(2, 2, figsize=(11, 6.2), sharex=False, facecolor=SURFACE)
for ax, seg in zip(axes.ravel(), segments):
    hist = df_features[(df_features["Gender"] == gender) & (df_features["StockGroupDesc"].astype(str) == seg)]
    hist = hist.sort_values("Date")
    test = res[(res["Gender"] == gender) & (res["StockGroupDesc"].astype(str) == seg)].sort_values("Date")
    pre = hist[hist["Date"] < test["Date"].min()].tail(HISTORY_WEEKS)

    ax.set_facecolor(SURFACE)
    ax.axvspan(test["Date"].min(), test["Date"].max(), color=GRID, alpha=0.5, lw=0)
    ax.plot(np.concatenate([pre["Date"].values, test["Date"].values]),
            np.concatenate([pre["SatisSayisi"].values, test["SatisSayisi"].values]), color=C_ACTUAL, lw=2)
    ax.plot(test["Date"], test["Tahmin_Ensemble"], color=C_FORECAST, lw=2, ls=(0, (4, 2)))

    wape = np.abs(test["SatisSayisi"] - test["Tahmin_Ensemble"]).sum() / test["SatisSayisi"].sum() * 100
    ax.set_title(seg, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.text(0.99, 0.04, f"WAPE %{wape:.1f}".replace(".", ","), transform=ax.transAxes, ha="right",
            fontsize=9, color=INK_2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %y"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}".replace(",", ".")))

# Doğrudan etiket (yalnızca ilk panelde) + açıklama
ax0 = axes.ravel()[0]
t0 = res[(res["Gender"] == gender) & (res["StockGroupDesc"].astype(str) == segments[0])].sort_values("Date")
ax0.annotate("gerçek", (t0["Date"].iloc[len(t0) // 2], t0["SatisSayisi"].iloc[len(t0) // 2]),
             xytext=(0, 18), textcoords="offset points", ha="center", fontsize=9, color=INK_2)
ax0.annotate("tahmin", (t0["Date"].iloc[len(t0) // 2], t0["Tahmin_Ensemble"].iloc[len(t0) // 2]),
             xytext=(0, -20), textcoords="offset points", ha="center", fontsize=9, color=INK_2)

fig.suptitle("12 haftalık özyinelemeli backtest — gerçek satış (mavi) ve ensemble tahmini (turuncu, kesikli)",
             x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
fig.text(0.01, 0.005, "Uydurma (sentetik) örnek veri · gri alan = backtest dönemi · WAPE = ağırlıklı mutlak yüzde hata",
         ha="left", fontsize=8, color=INK_2)
fig.tight_layout(rect=(0, 0.03, 1, 0.97))

out = ROOT / "assets" / "forecast_example.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150, facecolor=SURFACE)
print("Yazıldı:", out)
