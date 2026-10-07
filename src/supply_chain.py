import numpy as np
import pandas as pd
from src.config import BTS_SEGMENTS
from src.features import get_dynamic_events
from src.model import load_ensemble_weights


def generate_recursive_forecast(
    models_dict: dict,
    full_features_df: pd.DataFrame,
    gender: str = "Tümü",
    category: str = "Tümü",
    forecast_horizon: int = 8,
    selected_model_name: str = "🏆 Ensemble (Ağırlıklı)",
    weights=None
) -> tuple:
    """
    Gelecek dönem için özyinelemeli tahmin üretirken flatline (düz çizgi) sorununu
    engellemek adına geçmiş 52 haftalık mevsimsel dalga indeksini (seasonal profile)
    model tabanına dinamik olarak uygular.

    Quantile modeller varsa (lgb_q10, lgb_q90 vb.) alt ve üst sınır da üretir.
    Dönen: (orta_tahmin, alt_sinir, ust_sinir) - her biri np.ndarray
           Quantile model yoksa alt_sinir ve ust_sinir None döner.
    """
    weights = load_ensemble_weights(weights)

    lgb_m = models_dict['lgb']
    xgb_m = models_dict['xgb']
    cat_m = models_dict['cat']
    w_lgb, w_xgb, w_cat = weights
    feature_cols = models_dict['feature_names']
    cat_cols = [c for c in ['Gender', 'GroupDesc', 'StockGroupDesc'] if c in feature_cols]

    # Quantile modeller var mı?
    has_quantile = all(k in models_dict for k in ['lgb_q10', 'lgb_q90', 'xgb_q10', 'xgb_q90', 'cat_q10', 'cat_q90'])

    df_filtered = full_features_df.copy()
    if gender != "Tümü":
        df_filtered = df_filtered[df_filtered['Gender'] == gender]
    if category != "Tümü":
        df_filtered = df_filtered[df_filtered['StockGroupDesc'] == category]

    unique_groups = df_filtered[['Gender', 'GroupDesc', 'StockGroupDesc']].drop_duplicates()
    all_series_forecasts = []
    all_series_lower = []
    all_series_upper = []

    for _, r in unique_groups.iterrows():
        mask = (
            (df_filtered['Gender'] == r['Gender']) &
            (df_filtered['GroupDesc'] == r['GroupDesc']) &
            (df_filtered['StockGroupDesc'] == r['StockGroupDesc'])
        )
        series_data = df_filtered[mask].sort_values('Date').copy()
        if len(series_data) < 12:
            continue

        latest_row_dict = series_data.tail(1).iloc[0].to_dict()
        current_year = int(latest_row_dict['Year'])
        current_week = int(latest_row_dict['Week'])

        # 1. Mevsimsel Profil İndeksi Çıkarma (Seasonal Weight Index)
        # Son 52 haftadaki satışların haftalık ortalamaya oranı
        recent_1y = series_data.tail(52)
        mean_1y = recent_1y['SatisSayisi'].mean() + 1e-6
        seasonal_index = (recent_1y.groupby('Week')['SatisSayisi'].mean() / mean_1y).to_dict()

        simulated_sales = series_data['SatisSayisi'].tolist()
        series_forecast = []
        series_lower = []
        series_upper = []

        for step in range(1, forecast_horizon + 1):
            step_week = current_week + step
            step_year = current_year
            if step_week > 52:
                step_week = step_week - 52
                step_year = current_year + 1

            events = get_dynamic_events(step_year)
            row_dict = latest_row_dict.copy()
            row_dict['Year'] = step_year
            row_dict['Week'] = step_week

            # Takvim & Kampanya Bayrakları (features.add_retail_events ile ayni sozluk)
            row_dict['ramazan_rampasi'] = int(step_week in events['ramazan_rampasi'])
            row_dict['ramazan_pik'] = int(step_week in events['ramazan_pik'])
            row_dict['ramazan_bayrami'] = int(step_week in events['ramazan_bayrami'])
            row_dict['kurban_rampasi'] = int(step_week in events['kurban_rampasi'])
            row_dict['kurban_bayrami'] = int(step_week in events['kurban_bayrami'])
            row_dict['anneler_gunu'] = int(step_week in events['anneler'])
            row_dict['babalar_gunu'] = int(step_week in events['babalar'])
            row_dict['okula_donus'] = int(step_week in events['okul'])
            row_dict['kasim_indirimleri'] = int(step_week in events['kasim'])
            row_dict['yilbasi_etkisi'] = int(step_week in events['yilbasi'])
            row_dict['yeni_yil_dususu'] = int(step_week in events['yeni_yil_dususu'])
            # Turetilmis bayraklar
            row_dict['ramazan_etkisi'] = max(row_dict['ramazan_rampasi'], row_dict['ramazan_pik'])
            row_dict['kurban_etkisi'] = max(row_dict['kurban_rampasi'], row_dict['kurban_bayrami'])
            row_dict['event_kadin_anneler'] = int(row_dict['anneler_gunu'] and latest_row_dict['Gender'] == 'Kadin')
            row_dict['event_erkek_babalar'] = int(row_dict['babalar_gunu'] and latest_row_dict['Gender'] == 'Erkek')
            row_dict['event_okul_sneaker'] = int(row_dict['okula_donus'] and latest_row_dict['StockGroupDesc'] in BTS_SEGMENTS)

            # Fourier Döngüleri
            for k in [1, 2]:
                row_dict[f'sin_fourier_{k}'] = np.sin(2 * np.pi * k * step_week / 52.1775)
                row_dict[f'cos_fourier_{k}'] = np.cos(2 * np.pi * k * step_week / 52.1775)

            # Dinamik Gecikmeler
            # NOT: asagida GUNCELLENMEYEN feature'lar son gercek haftanin degerinde
            # DONDURULUR (`latest_row_dict`'ten kopyalanir): lag_stock_1, lag_store_1,
            # is_stockout_risk_lag1, stock_capacity_ratio, cross_gender_lag_1,
            # prophet_tahmin, patchtst_tahmin.
            # Ilk dordu gelecege dair bilinmeyen dissal degiskenler (gelecekteki stok
            # bilinemez), besincisi karsi cinsiyetin gelecek satisi -- o da ayni
            # anda tahmin edilmedigi icin dondurulmus. Son ikisi (prophet_tahmin/
            # patchtst_tahmin) icin gelecek icin GERCEK tahmin uretmek mumkun
            # (bkz. src/aux_forecasters.py generate_*_future()) ama her canli
            # panel istegi icin Prophet/PatchTST'yi yeniden egitmek cok yavas
            # olacagindan, bilincli olarak aynen donduruluyor. Ufuk uzadikca bu
            # dondurulmus degerler bayatlar; bilinen kisit.
            if len(simulated_sales) >= 2 and 'lag_sales_2' in feature_cols:
                row_dict['lag_sales_2'] = float(simulated_sales[-2])
            if len(simulated_sales) >= 4 and 'lag_sales_4' in feature_cols:
                row_dict['lag_sales_4'] = float(simulated_sales[-4])
            if len(simulated_sales) >= 52 and 'lag_sales_52' in feature_cols:
                row_dict['lag_sales_52'] = float(simulated_sales[-52])
            if len(simulated_sales) >= 53 and 'lag_sales_52_avg3' in feature_cols:
                row_dict['lag_sales_52_avg3'] = float(np.mean(simulated_sales[-53:-50]))
            
            rm52 = float(np.mean(simulated_sales[-52:])) if len(simulated_sales) >= 52 else float(np.mean(simulated_sales))
            if 'seasonal_index_ly' in feature_cols and 'lag_sales_52' in feature_cols:
                row_dict['seasonal_index_ly'] = float(row_dict['lag_sales_52']) / (rm52 + 1e-6)
            if 'seasonal_index_ly_smooth' in feature_cols and 'lag_sales_52_avg3' in feature_cols:
                row_dict['seasonal_index_ly_smooth'] = float(row_dict['lag_sales_52_avg3']) / (rm52 + 1e-6)

            r4 = float(np.mean(simulated_sales[-4:])) if len(simulated_sales) >= 4 else float(np.mean(simulated_sales))
            r12 = float(np.mean(simulated_sales[-12:])) if len(simulated_sales) >= 12 else float(np.mean(simulated_sales))

            if 'rolling_mean_4' in feature_cols:
                row_dict['rolling_mean_4'] = r4
            if 'rolling_mean_12' in feature_cols:
                row_dict['rolling_mean_12'] = r12
            if 'ema_sales_4' in feature_cols:
                row_dict['ema_sales_4'] = float(pd.Series(simulated_sales).ewm(span=4, adjust=False).mean().iloc[-1])
            if 'sales_momentum' in feature_cols:
                row_dict['sales_momentum'] = r4 / (r12 + 1e-6)
            if 'stock_cover_pressure' in feature_cols and 'lag_stock_1' in feature_cols:
                row_dict['stock_cover_pressure'] = float(row_dict['lag_stock_1']) / (r4 + 1e-6)

            # zaman_indeksi: her adımda 1 artır (genel büyüme trendini gelecege tasi)
            if 'zaman_indeksi' in feature_cols:
                row_dict['zaman_indeksi'] = int(row_dict.get('zaman_indeksi', 0)) + step

            feat_df = pd.DataFrame([row_dict])[feature_cols]
            for c in cat_cols:
                feat_df[c] = feat_df[c].astype('category')

            # Ham Model Çıktıları (Orta / Median)
            p_lgb = max(0.0, float(np.expm1(lgb_m.predict(feat_df))[0]))
            p_xgb = max(0.0, float(np.expm1(xgb_m.predict(feat_df))[0]))

            feat_df_cat = feat_df.copy()
            for c in cat_cols:
                feat_df_cat[c] = feat_df_cat[c].astype(str)
            p_cat = max(0.0, float(np.expm1(cat_m.predict(feat_df_cat))[0]))

            p_ens = (w_lgb * p_lgb) + (w_xgb * p_xgb) + (w_cat * p_cat)

            # Quantile Model Çıktıları (Alt ve Üst Sınır)
            if has_quantile:
                p_lgb_lo = max(0.0, float(np.expm1(models_dict['lgb_q10'].predict(feat_df))[0]))
                p_xgb_lo = max(0.0, float(np.expm1(models_dict['xgb_q10'].predict(feat_df))[0]))
                p_cat_lo = max(0.0, float(np.expm1(models_dict['cat_q10'].predict(feat_df_cat))[0]))
                p_lo = (w_lgb * p_lgb_lo) + (w_xgb * p_xgb_lo) + (w_cat * p_cat_lo)

                p_lgb_hi = max(0.0, float(np.expm1(models_dict['lgb_q90'].predict(feat_df))[0]))
                p_xgb_hi = max(0.0, float(np.expm1(models_dict['xgb_q90'].predict(feat_df))[0]))
                p_cat_hi = max(0.0, float(np.expm1(models_dict['cat_q90'].predict(feat_df_cat))[0]))
                p_hi = (w_lgb * p_lgb_hi) + (w_xgb * p_xgb_hi) + (w_cat * p_cat_hi)

            if "LightGBM" in selected_model_name:
                raw_pred = p_lgb
            elif "XGBoost" in selected_model_name:
                raw_pred = p_xgb
            elif "CatBoost" in selected_model_name:
                raw_pred = p_cat
            elif "Stacking" in selected_model_name and 'meta_model' in models_dict:
                meta = models_dict['meta_model']
                raw_pred = max(0.0, float(meta.predict(pd.DataFrame({'lgb': [p_lgb], 'xgb': [p_xgb], 'cat': [p_cat]}))[0]))
            else:
                raw_pred = p_ens
            # 2. Flatline Önleyici: Mevsimsel İndeks ile Harmanlama
            # Ağacın seviye (level) tahminini haftalık ritim çarpanıyla ölçeklendiriyoruz
            w_idx = seasonal_index.get(step_week, 1.0)
            # İndeksi aşırı uç değerlere karşı %60 - %170 bandına sınırla
            w_idx_clamped = np.clip(w_idx, 0.60, 1.70)

            # Modele %70 taban ağırlık, %30 mevsimsel profil çarpanı ver
            final_pred = (raw_pred * 0.70) + (raw_pred * w_idx_clamped * 0.30)

            series_forecast.append(max(0.0, float(final_pred)))
            simulated_sales.append(final_pred)

            # Quantile bantlarına da aynı mevsimsel çarpanı uygula
            if has_quantile:
                final_lo = (p_lo * 0.70) + (p_lo * w_idx_clamped * 0.30)
                final_hi = (p_hi * 0.70) + (p_hi * w_idx_clamped * 0.30)
                series_lower.append(max(0.0, float(final_lo)))
                series_upper.append(max(0.0, float(final_hi)))

        all_series_forecasts.append(series_forecast)
        if has_quantile:
            all_series_lower.append(series_lower)
            all_series_upper.append(series_upper)

    if not all_series_forecasts:
        return np.zeros(forecast_horizon), None, None

    total_forecast = np.sum(np.array(all_series_forecasts), axis=0)

    if has_quantile and all_series_lower:
        total_lower = np.sum(np.array(all_series_lower), axis=0)
        total_upper = np.sum(np.array(all_series_upper), axis=0)
        return total_forecast, total_lower, total_upper

    return total_forecast, None, None


def calculate_order_decision(
    forecast_series: np.ndarray,
    current_stock: float,
    lead_time_weeks: int = 4,
    review_period_weeks: int = 1,
    z_score: float = 1.65,
    lower_series: np.ndarray = None,
    upper_series: np.ndarray = None,
) -> dict:
    horizon = lead_time_weeks + review_period_weeks
    lead_time_demand = np.sum(forecast_series[:horizon])

    if lower_series is not None and upper_series is not None:
        # Kantil modellerin (q10/q90) haftalık bandından haftalık std çıkar
        # (q90-q10, normal dağılımda ~%80 aralığa denk gelir -> 2*1.2816*sigma),
        # sonra haftaların bağımsız olduğu varsayımıyla varyansları toplayıp
        # tedarik-süresi std'sine (sigma_LT) çevir. Böylece güvenlik stoku,
        # sadece nokta tahminin (mevsimsel indeksle zaten pürüzsüzleştirilmiş)
        # kendi oynaklığına değil, modelin gerçekten eğitilmiş belirsizliğine
        # dayanır (bkz. docs/degisiklik_gecmisi.md 2026-09-17).
        per_week_sigma = (np.asarray(upper_series[:horizon]) - np.asarray(lower_series[:horizon])) / (2 * 1.2816)
        sigma_lead_time = np.sqrt(np.sum(per_week_sigma ** 2))
        safety_stock = z_score * sigma_lead_time
    else:
        forecast_std = np.std(forecast_series[:horizon]) if len(forecast_series[:horizon]) > 1 else np.mean(forecast_series) * 0.15
        safety_stock = z_score * forecast_std * np.sqrt(horizon)

    target_stock = lead_time_demand + safety_stock
    suggested_order = max(0.0, target_stock - current_stock)

    weekly_avg_demand = np.mean(forecast_series) + 1e-6
    weeks_of_supply = current_stock / weekly_avg_demand
    is_overstocked = current_stock > target_stock

    return {
        'Gelecek_Talep_Toplami': round(lead_time_demand, 0),
        'Guvenlik_Stoku': round(safety_stock, 0),
        'Hedef_Stok': round(target_stock, 0),
        'Mevcut_Stok': round(current_stock, 0),
        'Onerilen_Siparis': round(suggested_order, 0),
        'Stok_Karsilama_Haftasi_WOS': round(weeks_of_supply, 1),
        'Finansal_Fren': "AKTIF (Siparis Verme)" if is_overstocked else "PASIF (Siparis Ver)"
    }