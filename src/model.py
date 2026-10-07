import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# models/ klasoru kaydedilmis model yoksa kullanilacak yedek agirlik.
DEFAULT_WEIGHTS = (0.33, 0.33, 0.34)


def load_ensemble_weights(weights=None) -> tuple:
    """
    Ensemble harman agirligini (w_lgb, w_xgb, w_cat) olarak cozer.

    `weights` verilmisse aynen dondurur; verilmemisse
    models/ensemble_config.json -> "final_weights" okunur. Dosya yoksa/bozuksa
    DEFAULT_WEIGHTS'e duser (model.py, supply_chain.py ve feature-importance
    hesabi ayni kaynagi kullansin diye tek fonksiyona toplandi).
    """
    if weights is not None:
        return tuple(weights)
    conf_path = Path(__file__).resolve().parent.parent / "models" / "ensemble_config.json"
    try:
        with open(conf_path, encoding="utf-8") as cfile:
            w = json.load(cfile)["final_weights"]
        return (w["LightGBM"], w["XGBoost"], w["CatBoost"])
    except (OSError, ValueError, KeyError):
        return DEFAULT_WEIGHTS


def _update_features_from_chain(base_features, sim_sales):
    """Bir simülasyon zincirinden lag ve türev özellikleri günceller."""
    feat = base_features.copy()
    if len(sim_sales) >= 2 and 'lag_sales_2' in feat:
        feat['lag_sales_2'] = float(sim_sales[-2])
    if len(sim_sales) >= 4 and 'lag_sales_4' in feat:
        feat['lag_sales_4'] = float(sim_sales[-4])
    r4 = float(np.mean(sim_sales[-4:])) if len(sim_sales) >= 4 else float(np.mean(sim_sales))
    r12 = float(np.mean(sim_sales[-12:])) if len(sim_sales) >= 12 else float(np.mean(sim_sales))
    if 'rolling_mean_4' in feat:
        feat['rolling_mean_4'] = r4
    if 'rolling_mean_12' in feat:
        feat['rolling_mean_12'] = r12
    if 'ema_sales_4' in feat:
        feat['ema_sales_4'] = float(pd.Series(sim_sales).ewm(span=4, adjust=False).mean().iloc[-1])
    if 'sales_momentum' in feat:
        feat['sales_momentum'] = r4 / (r12 + 1e-6)
    if 'stock_cover_pressure' in feat and 'lag_stock_1' in feat:
        feat['stock_cover_pressure'] = float(feat['lag_stock_1']) / (r4 + 1e-6)
    return feat


def _make_feat_df(feat_dict, feature_cols, dtypes, cat_cols, for_catboost=False):
    """Özellik sözlüğünden model-uyumlu DataFrame üretir."""
    feat_df = pd.DataFrame([feat_dict])[feature_cols]
    for c in feature_cols:
        feat_df[c] = feat_df[c].astype(dtypes[c])
    if for_catboost:
        for c in cat_cols:
            feat_df[c] = feat_df[c].astype(str)
    return feat_df


def _run_recursive_chains(df_proc, feature_cols, cat_cols, target_col, test_weeks,
                          model_lgb, model_xgb, model_cat, weights, dtypes, meta_model=None):
    """
    Paylaşılan özyinelemeli (recursive) backtest döngüsü. Her seri (grup) için
    5 bağımsız zincir (lgb/xgb/cat/ens/stk) kendi tahminiyle beslenerek
    `test_weeks` haftayı simüle eder. `train_and_validate_ensemble` ve
    `run_recursive_backtest` bu döngüyü aynen paylaşır (tek kaynak).

    meta_model verilmezse stacking, ensemble ile aynı sonucu üretir (fallback).

    Donen: validated_test_df (Hata = SatisSayisi - Tahmin_Ensemble)
    """
    w_lgb, w_xgb, w_cat = weights
    unique_dates = np.sort(df_proc['Date'].unique())
    train_dates = unique_dates[:-test_weeks]
    test_dates = unique_dates[-test_weeks:]

    test_results_list = []
    grouped = df_proc.groupby(cat_cols, observed=True) if cat_cols else [('ALL', df_proc)]

    for _, group_df in grouped:
        group_df = group_df.sort_values('Date').copy()
        train_series = group_df[group_df['Date'].isin(train_dates)][target_col].tolist()
        test_sub = group_df[group_df['Date'].isin(test_dates)].copy()
        if len(test_sub) == 0:
            continue

        # Her model kendi bağımsız simülasyon zincirinden beslenir (Adil Metrik)
        chains = {k: list(train_series) for k in ['lgb', 'xgb', 'cat', 'ens', 'stk']}
        preds = {k: [] for k in ['lgb', 'xgb', 'cat', 'ens', 'stk']}

        for _, row in test_sub.iterrows():
            base = row[feature_cols].to_dict()

            # --- Bağımsız tekil model zincirleri ---
            f_lgb = _update_features_from_chain(base, chains['lgb'])
            df_lgb = _make_feat_df(f_lgb, feature_cols, dtypes, cat_cols)
            p_lgb = max(0.0, float(np.expm1(model_lgb.predict(df_lgb))[0]))

            f_xgb = _update_features_from_chain(base, chains['xgb'])
            df_xgb = _make_feat_df(f_xgb, feature_cols, dtypes, cat_cols)
            p_xgb = max(0.0, float(np.expm1(model_xgb.predict(df_xgb))[0]))

            f_cat = _update_features_from_chain(base, chains['cat'])
            df_cat = _make_feat_df(f_cat, feature_cols, dtypes, cat_cols, for_catboost=True)
            p_cat = max(0.0, float(np.expm1(model_cat.predict(df_cat))[0]))

            # --- Ensemble zinciri (kendi lag'larından 3 model çalıştırır) ---
            f_ens = _update_features_from_chain(base, chains['ens'])
            df_ens = _make_feat_df(f_ens, feature_cols, dtypes, cat_cols)
            df_ens_cat = _make_feat_df(f_ens, feature_cols, dtypes, cat_cols, for_catboost=True)
            p_ens = max(0.0,
                        w_lgb * max(0.0, float(np.expm1(model_lgb.predict(df_ens))[0])) +
                        w_xgb * max(0.0, float(np.expm1(model_xgb.predict(df_ens))[0])) +
                        w_cat * max(0.0, float(np.expm1(model_cat.predict(df_ens_cat))[0])))

            # --- Stacking zinciri (kendi lag'larından 3 model + meta) ---
            if meta_model is not None:
                f_stk = _update_features_from_chain(base, chains['stk'])
                df_stk = _make_feat_df(f_stk, feature_cols, dtypes, cat_cols)
                df_stk_cat = _make_feat_df(f_stk, feature_cols, dtypes, cat_cols, for_catboost=True)
                sp_lgb = max(0.0, float(np.expm1(model_lgb.predict(df_stk))[0]))
                sp_xgb = max(0.0, float(np.expm1(model_xgb.predict(df_stk))[0]))
                sp_cat = max(0.0, float(np.expm1(model_cat.predict(df_stk_cat))[0]))
                p_stk = max(0.0, float(meta_model.predict(
                    pd.DataFrame({'lgb': [sp_lgb], 'xgb': [sp_xgb], 'cat': [sp_cat]}))[0]))
            else:
                p_stk = p_ens

            # Her tahmin kendi zincirine eklenir
            chains['lgb'].append(p_lgb)
            chains['xgb'].append(p_xgb)
            chains['cat'].append(p_cat)
            chains['ens'].append(p_ens)
            chains['stk'].append(p_stk)

            preds['lgb'].append(p_lgb)
            preds['xgb'].append(p_xgb)
            preds['cat'].append(p_cat)
            preds['ens'].append(p_ens)
            preds['stk'].append(p_stk)

        test_sub['pred_lgb'] = np.round(preds['lgb'], 1)
        test_sub['pred_xgb'] = np.round(preds['xgb'], 1)
        test_sub['pred_cat'] = np.round(preds['cat'], 1)
        test_sub['Tahmin_Ensemble'] = np.round(preds['ens'], 1)
        test_sub['pred_stacking'] = np.round(preds['stk'], 1)
        # NOT (2026-09-17 bugfix): eskiden hangi model gosteriliyor olursa olsun
        # her zaman XGBoost'a karsi hesaplaniyordu (pred_xgb) -- Ensemble'a
        # karsi hesaplanacak sekilde duzeltildi (bkz. docs/degisiklik_gecmisi.md).
        test_sub['Hata'] = np.round(test_sub[target_col] - test_sub['Tahmin_Ensemble'], 1)

        test_results_list.append(test_sub)

    return pd.concat(test_results_list, axis=0)


def _metrics_table(validated_test_df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """5 model/yontemin (LGB/XGB/CAT/Ensemble/Stacking) ortak metrik karnesi."""
    y_true = validated_test_df[target_col].values
    models_preds = {
        'LightGBM': validated_test_df['pred_lgb'].values,
        'XGBoost': validated_test_df['pred_xgb'].values,
        'CatBoost': validated_test_df['pred_cat'].values,
        'Ensemble (Ağırlıklı)': validated_test_df['Tahmin_Ensemble'].values,
        '🧬 Stacking Ensemble': validated_test_df['pred_stacking'].values
    }

    metrics_list = []
    for m_name, p_val in models_preds.items():
        r2 = r2_score(y_true, p_val)
        mae = mean_absolute_error(y_true, p_val)
        rmse = np.sqrt(mean_squared_error(y_true, p_val))
        mape = mean_absolute_percentage_error(y_true, p_val) * 100
        wape = (np.sum(np.abs(y_true - p_val)) / (np.sum(y_true) + 1e-6)) * 100
        bias = ((np.sum(p_val) - np.sum(y_true)) / (np.sum(y_true) + 1e-6)) * 100

        metrics_list.append({
            'Model': m_name,
            'R² Skoru': round(r2, 3),
            'WAPE (%)': f"%{wape:.1f}",
            'MAPE (%)': f"%{mape:.1f}",
            'RMSE (Çift)': f"{int(rmse):,}",
            'MAE (Çift)': f"{int(mae):,}",
            'Sapma/Bias (%)': f"%{bias:+.1f}"
        })

    return pd.DataFrame(metrics_list)


def train_and_validate_ensemble(df: pd.DataFrame, test_weeks: int = 12, weights=None):
    """
    Modeli Train aralığında eğitir. Test setindeki 12 haftayı geriye dönük tam özyinelemeli
    (Recursive Multi-Step Out-of-Sample) olarak simüle ederek gerçekçi Backtest metriklerini hesaplar.
    """
    weights = load_ensemble_weights(weights)

    exclude_cols = ['Date', 'Year', 'SatisSayisi']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    target_col = 'SatisSayisi'
    cat_cols = [c for c in ['Gender', 'GroupDesc', 'StockGroupDesc'] if c in df.columns]

    df_proc = df.copy()
    for col in cat_cols:
        df_proc[col] = df_proc[col].astype('category')

    # Zaman bazlı split
    unique_dates = np.sort(df_proc['Date'].unique())
    train_dates = unique_dates[:-test_weeks]

    train_data = df_proc[df_proc['Date'].isin(train_dates)].copy()

    X_train = train_data[feature_cols]
    y_train_log = np.log1p(train_data[target_col])

    # Meta-Model (Stacking): OOF/VAL regresyonuyla kendi ağırlığını öğrenmek
    # yerine, zaten Optuna + 4-fold rolling-origin CV ile eşit ağırlığa karşı
    # doğrulanmış `weights`'i (yukarıda çözüldü) doğrudan kullanıyor.
    # NOT (2026-09-14): eskiden burada TimeSeriesSplit ile ayrı bir OOF
    # regresyonu (9 ekstra model eğitimi) yapılıp kendi ağırlığı öğreniliyordu.
    # 4-fold rolling-origin CV ile test edildi: bu regresyon küçük örneklemde
    # güvenilmez sonuçlar buluyordu (bir modele sıfır ağırlık verip bazen iyi
    # bazen kötü çıkan aşırı uç kararlar) — bkz. docs/degisiklik_gecmisi.md.
    # `weights` zaten daha güvenilir bir kaynaktan geliyor, tekrar öğrenmeye
    # gerek yok.
    meta_model = LinearRegression(positive=True, fit_intercept=False)
    meta_model.coef_ = np.array(weights, dtype=float)
    meta_model.intercept_ = 0.0
    meta_model.n_features_in_ = 3
    meta_model.feature_names_in_ = np.array(['lgb', 'xgb', 'cat'], dtype=object)

    # 1.5 Tüm Veri ile Nihai Base Modellerin Eğitimi
    model_lgb = lgb.LGBMRegressor(n_estimators=450, learning_rate=0.03, num_leaves=31, random_state=42, verbosity=-1)
    model_lgb.fit(X_train, y_train_log)

    model_xgb = xgb.XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=5, enable_categorical=True, random_state=42)
    model_xgb.fit(X_train, y_train_log)

    X_train_cat = X_train.copy()
    for col in cat_cols:
        X_train_cat[col] = X_train_cat[col].astype(str)

    model_cat = CatBoostRegressor(iterations=400, learning_rate=0.03, depth=5, cat_features=cat_cols, verbose=0, random_seed=42, allow_writing_files=False)
    model_cat.fit(X_train_cat, y_train_log)

    models = {
        'lgb': model_lgb,
        'xgb': model_xgb,
        'cat': model_cat,
        'meta_model': meta_model,
        'feature_names': feature_cols
    }

    # 2. ÖZYİNELEMELİ (RECURSIVE) GERÇEKÇİ BACKTEST SİMÜLASYONU
    validated_test_df = _run_recursive_chains(
        df_proc, feature_cols, cat_cols, target_col, test_weeks,
        model_lgb, model_xgb, model_cat, weights, dtypes=X_train.dtypes, meta_model=meta_model,
    )

    # 3. METRİK HESAPLAMA
    metrics_df = _metrics_table(validated_test_df, target_col)

    return models, validated_test_df, metrics_df


def run_recursive_backtest(models: dict, df: pd.DataFrame, test_weeks: int = 12,
                           weights=None):
    """
    ONCEDEN EGITILMIS modellerle (models/ klasorunden yuklenen) recursive backtest yapar.
    Model EGITMEZ; sadece son `test_weeks` haftayi ozyinelemeli simule eder.

    models  : {"lgb", "xgb", "cat", "feature_names"}  (tuning.load_bundle ciktisi)
    weights : (w_lgb, w_xgb, w_cat) ensemble agirliklari

    Donen: (validated_test_df, metrics_df)  -- train_and_validate_ensemble ile ayni sekil
    """
    weights = load_ensemble_weights(weights)

    model_lgb, model_xgb, model_cat = models['lgb'], models['xgb'], models['cat']
    feature_cols = models['feature_names']

    target_col = 'SatisSayisi'
    cat_cols = [c for c in ['Gender', 'GroupDesc', 'StockGroupDesc'] if c in feature_cols]

    df_proc = df.copy()
    for col in cat_cols:
        df_proc[col] = df_proc[col].astype('category')

    ref_dtypes = df_proc[feature_cols].dtypes

    validated_test_df = _run_recursive_chains(
        df_proc, feature_cols, cat_cols, target_col, test_weeks,
        model_lgb, model_xgb, model_cat, weights, dtypes=ref_dtypes,
        meta_model=models.get('meta_model'),
    )

    return validated_test_df, _metrics_table(validated_test_df, target_col)


def get_feature_importance_df(models_dict: dict) -> pd.DataFrame:
    feat_names = models_dict['feature_names']
    lgb_imp = models_dict['lgb'].feature_importances_
    xgb_imp = models_dict['xgb'].feature_importances_
    cat_imp = models_dict['cat'].get_feature_importance()

    def normalize(arr):
        total = arr.sum()
        return (arr / total) * 100 if total > 0 else np.zeros_like(arr)

    lgb_norm = normalize(lgb_imp)
    xgb_norm = normalize(xgb_imp)
    cat_norm = normalize(cat_imp)

    w_lgb, w_xgb, w_cat = load_ensemble_weights()

    ensemble_importance = (w_lgb * lgb_norm) + (w_xgb * xgb_norm) + (w_cat * cat_norm)

    df_imp = pd.DataFrame({
        'Feature': feat_names,
        'Importance': np.round(ensemble_importance, 2)
    }).sort_values('Importance', ascending=False).reset_index(drop=True)

    return df_imp