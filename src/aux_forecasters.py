"""
Prophet ve PatchTST'nin ana modele (LGB/XGB/CatBoost) feature olarak eklenen
tahminlerini uretir.

Izole gelistirme/dogrulama notebooks/exp_prophet.ipynb (Asama 12) ve
notebooks/exp_patchtst.ipynb (Asama 11)'de yapildi -- oradaki sizintisiz
walk-forward mantigi buraya, kalici/cagrilabilir fonksiyonlara tasindi.

Iki tur uretim var:
- Walk-forward (gecmis, egitim icin): genisleyen pencerede, sizintisiz.
  generate_prophet_walkforward() / generate_patchtst_walkforward()
- Production (gelecek, canli tahmin icin): tum gercek gecmisle egitilip
  gelecek N haftayi tahmin eder.
  generate_prophet_future() / generate_patchtst_future()

ONEMLI (bu oturumda kesfedildi): Prophet (cmdstanpy alt-surec) ve PatchTST
(PyTorch) ayni Python surecinde art arda calistirilinca, kok nedeni tam
netlesmemis bir sekilde donuyor. Bu yuzden bu iki model ailesinin
fonksiyonlari AYRI scriptlerde/sureçlerde cagrilmali -- ikisini ayni
oturumda/import zincirinde art arda calistirmayin.
"""
import numpy as np
import pandas as pd

from src.data_loader import get_iso_monday
from src.features import get_dynamic_events

PROPHET_CPS = 0.05
PROPHET_MODE = "additive"
PATCHTST_INPUT_SIZE = 52
PATCHTST_MAX_STEPS = 300
BLOCK_WEEKS = 12
MIN_TRAIN_WEEKS = 104


def _build_prophet_holidays(years):
    rows = []
    for yr in years:
        events = get_dynamic_events(int(yr))
        for name, weeks in events.items():
            for w in weeks:
                if w < 1:
                    continue
                rows.append({"holiday": name, "ds": get_iso_monday(int(yr), int(w))})
    return pd.DataFrame(rows)


def _gruplar(df_raw):
    return df_raw[["Gender", "GroupDesc", "StockGroupDesc"]].drop_duplicates().values.tolist()


def generate_prophet_walkforward(df_raw, block_weeks=BLOCK_WEEKS, min_train_weeks=MIN_TRAIN_WEEKS):
    """Gecmis icin sizintisiz walk-forward prophet_tahmin uretir (egitim feature'i).

    Genisleyen pencerede ilerler: her blok SADECE kendinden onceki haftalarla
    egitilmis bir Prophet modelinden gelir (OOF/rolling-origin CV'deki
    sizintisizlik mantiginin ayni).
    """
    from prophet import Prophet

    unique_dates = np.sort(df_raw["Date"].unique())
    holidays_df = _build_prophet_holidays(df_raw["Year"].unique())
    gruplar = _gruplar(df_raw)

    rows = []
    start = min_train_weeks
    while start < len(unique_dates):
        block_dates = unique_dates[start:start + block_weeks]
        fit_dates = unique_dates[:start]
        for gender, group_desc, stock_group in gruplar:
            mask = (
                (df_raw["Gender"] == gender)
                & (df_raw["GroupDesc"] == group_desc)
                & (df_raw["StockGroupDesc"] == stock_group)
            )
            g_df = df_raw[mask].sort_values("Date")
            g_fit = g_df[g_df["Date"].isin(fit_dates)]
            fit_df = g_fit[["Date", "SatisSayisi"]].rename(columns={"Date": "ds", "SatisSayisi": "y"})

            m = Prophet(holidays=holidays_df, changepoint_prior_scale=PROPHET_CPS, seasonality_mode=PROPHET_MODE)
            m.fit(fit_df)
            fc = m.predict(pd.DataFrame({"ds": block_dates}))
            pred = fc[["ds", "yhat"]].rename(columns={"yhat": "prophet_tahmin", "ds": "Date"})
            pred["prophet_tahmin"] = pred["prophet_tahmin"].clip(lower=0).round(1)
            pred["Gender"], pred["GroupDesc"], pred["StockGroupDesc"] = gender, group_desc, stock_group
            rows.append(pred)
        start += block_weeks

    return pd.concat(rows, ignore_index=True)


def generate_prophet_future(df_raw, horizon_weeks):
    """Tum gercek gecmisle egitip gelecek horizon_weeks haftayi tahmin eder (canli tahmin icin)."""
    from prophet import Prophet

    holidays_years = list(df_raw["Year"].unique()) + [int(df_raw["Year"].max()) + 1]
    holidays_df = _build_prophet_holidays(holidays_years)
    gruplar = _gruplar(df_raw)

    last_date = df_raw["Date"].max()
    future_dates = pd.date_range(start=pd.Timestamp(last_date) + pd.Timedelta(weeks=1),
                                  periods=horizon_weeks, freq="W-MON")

    rows = []
    for gender, group_desc, stock_group in gruplar:
        mask = (
            (df_raw["Gender"] == gender)
            & (df_raw["GroupDesc"] == group_desc)
            & (df_raw["StockGroupDesc"] == stock_group)
        )
        g_df = df_raw[mask].sort_values("Date")
        fit_df = g_df[["Date", "SatisSayisi"]].rename(columns={"Date": "ds", "SatisSayisi": "y"})

        m = Prophet(holidays=holidays_df, changepoint_prior_scale=PROPHET_CPS, seasonality_mode=PROPHET_MODE)
        m.fit(fit_df)
        fc = m.predict(pd.DataFrame({"ds": future_dates}))
        pred = fc[["ds", "yhat"]].rename(columns={"yhat": "prophet_tahmin", "ds": "Date"})
        pred["prophet_tahmin"] = pred["prophet_tahmin"].clip(lower=0).round(1)
        pred["Gender"], pred["GroupDesc"], pred["StockGroupDesc"] = gender, group_desc, stock_group
        rows.append(pred)

    return pd.concat(rows, ignore_index=True)


def generate_patchtst_walkforward(df_raw, block_weeks=BLOCK_WEEKS, min_train_weeks=MIN_TRAIN_WEEKS,
                                   input_size=PATCHTST_INPUT_SIZE, max_steps=PATCHTST_MAX_STEPS):
    """Gecmis icin sizintisiz walk-forward patchtst_tahmin uretir (egitim feature'i).

    Her blokta 8 grup icin 8 ayri model degil, tek pooled PatchTST modeli
    (unique_id ile channel independence) egitilir -- Asama 6/11'deki tasarim.
    """
    from neuralforecast import NeuralForecast
    from neuralforecast.models import PatchTST

    unique_dates = np.sort(df_raw["Date"].unique())
    gruplar = _gruplar(df_raw)

    rows = []
    start = min_train_weeks
    while start < len(unique_dates):
        block_dates = unique_dates[start:start + block_weeks]
        fit_dates = unique_dates[:start]

        parcalar = []
        for gender, group_desc, stock_group in gruplar:
            mask = (
                (df_raw["Gender"] == gender)
                & (df_raw["GroupDesc"] == group_desc)
                & (df_raw["StockGroupDesc"] == stock_group)
            )
            g_df = df_raw[mask].sort_values("Date")
            g_fit = g_df[g_df["Date"].isin(fit_dates)]
            parca = g_fit[["Date", "SatisSayisi"]].rename(columns={"Date": "ds", "SatisSayisi": "y"})
            parca["unique_id"] = f"{gender}_{group_desc}_{stock_group}"
            parcalar.append(parca[["unique_id", "ds", "y"]])
        fit_df = pd.concat(parcalar, ignore_index=True)

        m = PatchTST(h=len(block_dates), input_size=input_size, max_steps=max_steps,
                     accelerator="cpu", enable_progress_bar=False, logger=False, random_seed=1)
        nf = NeuralForecast(models=[m], freq="W-MON")
        nf.fit(df=fit_df)
        fc = nf.predict()

        for gender, group_desc, stock_group in gruplar:
            uid = f"{gender}_{group_desc}_{stock_group}"
            g_fc = fc[fc["unique_id"] == uid][["ds", "PatchTST"]].rename(
                columns={"ds": "Date", "PatchTST": "patchtst_tahmin"})
            g_fc["patchtst_tahmin"] = g_fc["patchtst_tahmin"].clip(lower=0).round(1)
            g_fc["Gender"], g_fc["GroupDesc"], g_fc["StockGroupDesc"] = gender, group_desc, stock_group
            rows.append(g_fc)

        start += block_weeks

    return pd.concat(rows, ignore_index=True)


def generate_patchtst_future(df_raw, horizon_weeks, input_size=PATCHTST_INPUT_SIZE, max_steps=PATCHTST_MAX_STEPS):
    """Tum gercek gecmisle tek pooled model egitip gelecek horizon_weeks haftayi tahmin eder."""
    from neuralforecast import NeuralForecast
    from neuralforecast.models import PatchTST

    gruplar = _gruplar(df_raw)

    parcalar = []
    for gender, group_desc, stock_group in gruplar:
        mask = (
            (df_raw["Gender"] == gender)
            & (df_raw["GroupDesc"] == group_desc)
            & (df_raw["StockGroupDesc"] == stock_group)
        )
        g_df = df_raw[mask].sort_values("Date")
        parca = g_df[["Date", "SatisSayisi"]].rename(columns={"Date": "ds", "SatisSayisi": "y"})
        parca["unique_id"] = f"{gender}_{group_desc}_{stock_group}"
        parcalar.append(parca[["unique_id", "ds", "y"]])
    fit_df = pd.concat(parcalar, ignore_index=True)

    m = PatchTST(h=horizon_weeks, input_size=input_size, max_steps=max_steps,
                 accelerator="cpu", enable_progress_bar=False, logger=False, random_seed=1)
    nf = NeuralForecast(models=[m], freq="W-MON")
    nf.fit(df=fit_df)
    fc = nf.predict()

    rows = []
    for gender, group_desc, stock_group in gruplar:
        uid = f"{gender}_{group_desc}_{stock_group}"
        g_fc = fc[fc["unique_id"] == uid][["ds", "PatchTST"]].rename(
            columns={"ds": "Date", "PatchTST": "patchtst_tahmin"})
        g_fc["patchtst_tahmin"] = g_fc["patchtst_tahmin"].clip(lower=0).round(1)
        g_fc["Gender"], g_fc["GroupDesc"], g_fc["StockGroupDesc"] = gender, group_desc, stock_group
        rows.append(g_fc)

    return pd.concat(rows, ignore_index=True)
