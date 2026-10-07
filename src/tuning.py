"""
Hiperparametre optimizasyonu ve ensemble oran optimizasyonu için yardımcı modul.

ADIM 1: Veriyi kronolojik olarak TRAIN / VALIDATION / TEST parçalarına ayırmak.
ADIM 2: Tahminleri 5 metrikle degerlendiren fonksiyon.
ADIM 3: Tek bir LightGBM modelini early stopping ile egitip degerlendirmek.
ADIM 4: Optuna ile LightGBM hiperparametrelerini VAL setinde optimize etmek.
ADIM 5: Ayni optimizasyonu XGBoost ve CatBoost icin de yapmak.
ADIM 6: 3 modelin harman (ensemble) oranlarini Optuna ile VAL'de optimize etmek.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import optuna
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

optuna.logging.set_verbosity(optuna.logging.WARNING)  # deneme deneme log basmasin

# Modele GIRMEYECEK sutunlar. Geri kalan her sey oznitelik olur.
EXCLUDE_COLS = ["Date", "Year", "SatisSayisi"]
TARGET_COL = "SatisSayisi"
CAT_COLS = ["Gender", "GroupDesc", "StockGroupDesc"]

# ──────────────────────────────────────────────────────────────────────────────
# AĞIRLIK SEÇİM YÖNTEMİ (run_cv_tuning() içinde uygulanır)
#
# Üretimde kullanılan nihai ensemble ağırlığı, `models/ensemble_config.json`
# içindeki "final_weights" alanında saklanır ve NASIL seçildiği
# "ensemble_weight_method" alanında yazılıdır.
#
# TARİHÇE — neden tek-VAL-penceresi optimizasyonu KULLANILMIYOR:
#   4 Eylül'de Optuna'nın TEK bir VAL penceresinde bulduğu ağırlıklar denendi
#   (VAL WAPE %2.70, en iyi) ama TEST'te %1.44 ile EN KÖTÜ çıktı — VAL setine
#   overfit etmişti (VAL sadece ~21 bağımsız hafta, çok küçük).
#
# ŞU ANKİ YÖNTEM (optimize_ensemble_weights_cv, 4-fold recursive rolling-origin):
#   Ağırlık araması tek pencere yerine 4 farklı dönemin ORTALAMA recursive WAPE'sine
#   göre yapılır, ve SONUÇ eşit ağırlıkla (1/3 her biri) karşılaştırılır:
#     - CV'de optimize edilen ağırlık eşit ağırlıktan gerçekten iyiyse -> o kullanılır
#     - Değilse (denk/kötüyse) -> eşit ağırlık kullanılır (daha az overfit riski)
#   Hangisinin seçildiği ve iki yöntemin CV WAPE'leri config'de saklanır.
#
# Ağırlık seçim mantığını değiştirmek isterseniz: run_cv_tuning() (bu dosyada)
# + app.py (Sekme 3'teki ağırlık açıklaması, ens_weights'ten dinamik okunur).
# ──────────────────────────────────────────────────────────────────────────────


def get_feature_cols(df: pd.DataFrame) -> list:
    """Hedef ve dislanan sutunlar haric her sey oznitelik."""
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def make_xy(df: pd.DataFrame, feature_cols: list, cat_as: str = "category"):
    """
    Bir DataFrame'i modele verilecek X (oznitelikler) ve y_log (log'lu hedef) olarak ayirir.

    Neden y_log: satislar cok carpik; log1p dagilimi sikistirir, egitim dengelenir.
    Tahminden sonra expm1 ile geri acacagiz.

    cat_as: kategorik sutunlar hangi tipte olsun?
      "category" -> LightGBM ve XGBoost (enable_categorical) boyle ister
      "str"      -> CatBoost boyle ister
    """
    X = df[feature_cols].copy()
    for c in CAT_COLS:
        if c in X.columns:
            X[c] = X[c].astype(str) if cat_as == "str" else X[c].astype("category")
    y_log = np.log1p(df[TARGET_COL].to_numpy(dtype=float))
    return X, y_log


def train_lightgbm(train_df, val_df, params: dict = None, stopping_rounds: int = 100,
                   objective: str = "regression"):
    """
    LightGBM'i TRAIN'de egitir, VAL hatasini izleyerek en iyi agac sayisinda durur.

    objective: "regression" (normal) veya "quantile" (kantil regresyon).
               Quantile icin params'a alpha=0.10 veya 0.90 ekleyin.

    Donen: (model, en_iyi_agac_sayisi)
    """
    if params is None:
        # Simdilik elle secilmis makul degerler (Adim 4'te Optuna bunlari arayacak).
        params = dict(learning_rate=0.03, num_leaves=31)

    feature_cols = get_feature_cols(train_df)
    X_tr, y_tr = make_xy(train_df, feature_cols)
    X_va, y_va = make_xy(val_df, feature_cols)

    model = lgb.LGBMRegressor(
        n_estimators=5000,      # bilerek cok yuksek; early stopping gercek sayiyi bulacak
        objective=objective,
        random_state=42,
        verbosity=-1,
        **params,
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],           # her agactan sonra bu sette hataya bak
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(stopping_rounds=stopping_rounds, verbose=False),
            lgb.log_evaluation(0),         # egitim ciktisi basma
        ],
    )

    return model, model.best_iteration_


def predict_sales(model, df, feature_cols, cat_as: str = "category"):
    """
    Model log uzayinda tahmin verir. expm1 ile gercek satis olcegine geri cevir,
    negatifleri sifirla (negatif satis olmaz).
    """
    X, _ = make_xy(df, feature_cols, cat_as=cat_as)
    pred_log = model.predict(X)            # early stopping varsa otomatik en iyi agac sayisini kullanir
    pred = np.expm1(pred_log)
    return np.clip(pred, 0, None)


def _suggest_params(trial, kind: str) -> dict:
    """Model turune gore Optuna'nin arayacagi hiperparametre aralklari (tek yerden)."""
    if kind == "lgb":
        return dict(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 200),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            subsample_freq=trial.suggest_int("subsample_freq", 1, 7),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        )
    if kind == "xgb":
        return dict(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            gamma=trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        )
    if kind == "cat":
        return dict(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            depth=trial.suggest_int("depth", 3, 10),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            random_strength=trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
        )
    raise ValueError(kind)


_TRAIN_FN = {}  # kind -> train fonksiyonu (dosya sonunda doldurulur)


def optimize_model_cv(df: pd.DataFrame, kind: str, n_trials: int = 25,
                      n_folds: int = 3, test_weeks: int = 12, val_weeks: int = 12, seed: int = 42):
    """
    Hiperparametreleri TEK val penceresi yerine `n_folds` rolling-origin fold'un
    ORTALAMA WAPE'sine gore optimize eder.

    Hiz icin skorlama DIRECT (tek adim) yapilir -- her fold'da tek bir .predict(),
    recursive 12-adim simulasyon YOK. (Final CV raporu recursive kalir.)
    Umitsiz denemeler MedianPruner ile 1. fold'dan sonra kesilir.

    Donen: optuna study (study.best_params).
    """
    train_fn = _TRAIN_FN[kind]
    cat_as = "str" if kind == "cat" else "category"
    dates = np.sort(df["Date"].unique())
    n = len(dates)

    # fold sinirlarini (train / val / test tarih kumeleri) onceden hazirla
    folds = []
    for i in range(n_folds):
        test_end = n - (n_folds - 1 - i) * test_weeks
        te_d = dates[test_end - test_weeks:test_end]
        va_d = dates[test_end - test_weeks - val_weeks:test_end - test_weeks]
        tr_d = dates[:test_end - test_weeks - val_weeks]
        folds.append((set(tr_d), set(va_d), set(te_d)))

    def objective(trial):
        params = _suggest_params(trial, kind)
        wapes = []
        for k, (tr_d, va_d, te_d) in enumerate(folds):
            tr = df[df["Date"].isin(tr_d)]
            va = df[df["Date"].isin(va_d)]
            te = df[df["Date"].isin(te_d)]
            model, _ = train_fn(tr, va, params=params)
            fcols = get_feature_cols(tr)
            pred = predict_sales(model, te, fcols, cat_as=cat_as)
            wapes.append(calculate_metrics(te[TARGET_COL], pred)["WAPE"])
            trial.report(float(np.mean(wapes)), step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(wapes))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=n_trials)
    return study


def optimize_lightgbm(train_df, val_df, n_trials: int = 40, seed: int = 42):
    """
    Optuna ile LightGBM hiperparametrelerini arar.
    Her deneme: ayarlari sec -> TRAIN'de egit -> VAL WAPE'sini olc.
    Optuna bu WAPE'yi kucultmeye calisir.

    Donen: optuna study nesnesi (study.best_params, study.best_value icinde)
    """
    fcols = get_feature_cols(train_df)
    y_val_true = val_df[TARGET_COL].to_numpy(dtype=float)

    def objective(trial):
        # Arama uzayi _suggest_params()'ta, optimize_model_cv ile TEK YERDEN paylasilir
        # (ikisi ayni araligi aramali; eskiden burada birebir kopyasi duruyordu).
        # NOT: subsample_freq bilerek suggest ediliyor -- yoksa best_params'a girmez,
        # yeniden egitimde subsample etkisiz kalir.
        params = _suggest_params(trial, "lgb")

        model, _ = train_lightgbm(train_df, val_df, params=params)
        val_pred = predict_sales(model, val_df, fcols)
        wape = calculate_metrics(y_val_true, val_pred)["WAPE"]
        return wape  # Optuna bunu minimize edecek

    # TPESampler: gecmis denemelere bakip umut vaat eden bolgeye yonelir (grid'den akilli).
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials)
    return study


# ======================================================================
# XGBoost
# ======================================================================

def train_xgboost(train_df, val_df, params: dict = None, stopping_rounds: int = 100,
                  objective: str = "reg:squarederror"):
    """XGBoost'u TRAIN'de egitir, VAL ile early stopping yapar.
    objective: "reg:squarederror" (normal) veya "reg:quantileerror" (kantil).
               Quantile icin params'a quantile_alpha=0.10 veya 0.90 ekleyin.
    Donen: (model, en_iyi_agac)."""
    if params is None:
        params = dict(learning_rate=0.03, max_depth=5)

    feature_cols = get_feature_cols(train_df)
    X_tr, y_tr = make_xy(train_df, feature_cols)          # kategorikler "category" tipinde
    X_va, y_va = make_xy(val_df, feature_cols)

    model = xgb.XGBRegressor(
        n_estimators=5000,                # tavan; early stopping gercek sayiyi bulur
        objective=objective,
        early_stopping_rounds=stopping_rounds,   # XGBoost'ta bu constructor'da
        enable_categorical=True,          # kategorik sutunlari kendi isler
        tree_method="hist",
        random_state=42,
        verbosity=0,
        **params,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model, model.best_iteration


def optimize_xgboost(train_df, val_df, n_trials: int = 40, seed: int = 42):
    """Optuna ile XGBoost ayarlarini VAL WAPE'ye gore arar."""
    fcols = get_feature_cols(train_df)
    y_val_true = val_df[TARGET_COL].to_numpy(dtype=float)

    def objective(trial):
        params = _suggest_params(trial, "xgb")   # arama uzayi tek yerden (bkz. _suggest_params)
        model, _ = train_xgboost(train_df, val_df, params=params)
        val_pred = predict_sales(model, val_df, fcols)
        return calculate_metrics(y_val_true, val_pred)["WAPE"]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials)
    return study


# ======================================================================
# CatBoost
# ======================================================================

def train_catboost(train_df, val_df, params: dict = None, stopping_rounds: int = 100,
                   loss_function: str = "RMSE"):
    """CatBoost'u TRAIN'de egitir, VAL ile early stopping yapar.
    loss_function: "RMSE" (normal) veya "Quantile:alpha=0.10" (kantil).
    Donen: (model, en_iyi_iterasyon)."""
    if params is None:
        params = dict(learning_rate=0.03, depth=5)

    feature_cols = get_feature_cols(train_df)
    cat_features = [c for c in CAT_COLS if c in feature_cols]

    # CatBoost kategorikleri STRING ister (category dtype degil).
    X_tr, y_tr = make_xy(train_df, feature_cols, cat_as="str")
    X_va, y_va = make_xy(val_df, feature_cols, cat_as="str")

    model = CatBoostRegressor(
        iterations=5000,                       # tavan
        loss_function=loss_function,
        early_stopping_rounds=stopping_rounds,
        cat_features=cat_features,
        random_seed=42,
        verbose=0,
        allow_writing_files=False,              # catboost_info/ klasoru birikmesin
        **params,
    )
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    return model, model.best_iteration_


def optimize_catboost(train_df, val_df, n_trials: int = 40, seed: int = 42):
    """Optuna ile CatBoost ayarlarini VAL WAPE'ye gore arar."""
    fcols = get_feature_cols(train_df)
    y_val_true = val_df[TARGET_COL].to_numpy(dtype=float)

    def objective(trial):
        params = _suggest_params(trial, "cat")   # arama uzayi tek yerden (bkz. _suggest_params)
        model, _ = train_catboost(train_df, val_df, params=params)
        val_pred = predict_sales(model, val_df, fcols, cat_as="str")
        return calculate_metrics(y_val_true, val_pred)["WAPE"]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials)
    return study


# optimize_model_cv'nin kullandigi train fonksiyonu haritasi (train_* tanimlandiktan sonra)
_TRAIN_FN.update({"lgb": train_lightgbm, "xgb": train_xgboost, "cat": train_catboost})


def run_cv_tuning(df: pd.DataFrame, n_trials: int = 25, n_folds: int = 3, out_dir=None,
                  min_ensemble_weight: float = 0.10):
    """
    3 modeli de CV skoruna gore optimize eder (optimize_model_cv), en iyi ayarlarla
    standart 70/15/15 split'te yeniden egitir.

    Ensemble agirligi SABIT DEGIL: optimize_ensemble_weights_cv ile 4-fold
    rolling-origin CV'de aranan agirlik, esit agirlikla (1/3 her biri)
    karsilastirilir ve yalnizca gercekten daha iyiyse secilir (bkz. dosya basindaki
    "AGIRLIK SECIM YONTEMI" notu). Secilen agirlik hem ensemble'da hem meta-modelde
    kullanilir.

    min_ensemble_weight: her modele garanti edilen taban agirlik (bkz.
        optimize_ensemble_weights_cv docstring'i). Varsayilan 0.10 --
        2026-09-16'da bulundu: bu projede %10 tabanla WAPE HIC bozulmuyor
        (bkz. docs/degisiklik_gecmisi.md "Ensemble agirligina manuel minimum
        taban" kaydi) ama serbest aramanin bir modele (LightGBM, Prophet/
        PatchTST'e asiri guvendigi icin) neredeyse sifir agirlik vermesini
        engelliyor. 0 verilirse eski (tabansiz) davranisa doner.

    models/ klasorunu gunceller: lightgbm/xgboost/catboost.joblib, bunlarin q10/q90
    kantil surumleri, meta_model.joblib ve ensemble_config.json.
    """
    tr, va, te = split_train_val_test(df)
    fcols = get_feature_cols(df)
    out_dir = Path(out_dir) if out_dir else Path(__file__).resolve().parent.parent / "models"
    out_dir.mkdir(exist_ok=True)

    best_params, tuned = {}, {}
    for kind, isim, cat_as in [("lgb", "LightGBM", "category"),
                               ("xgb", "XGBoost", "category"),
                               ("cat", "CatBoost", "str")]:
        print(f"\n{isim}: CV-tabanli optimizasyon ({n_trials} deneme, {n_folds} fold)...")
        study = optimize_model_cv(df, kind, n_trials=n_trials, n_folds=n_folds)
        print(f"  en iyi CV WAPE ortalamasi: %{study.best_value:.2f}")
        best_params[isim] = study.best_params
        model, _ = _TRAIN_FN[kind](tr, va, params=study.best_params)
        tuned[isim] = dict(model=model, cat_as=cat_as)
        joblib.dump(model, out_dir / f"{isim.lower()}.joblib")

    # Ensemble ağırlıklarını 4-fold rolling-origin (recursive) CV'ye göre optimize et.
    # NOT: Tek bir VAL penceresine göre aramak (eski `optimize_ensemble_weights`)
    # 4 Eylül'de VAL'e overfit ettiği için terk edilmişti (VAL WAPE %2.70 en iyi
    # ama TEST %1.44 en kötü çıkmıştı). Bu yüzden burada eşit ağırlıkla karşılaştırıp
    # CV'de gerçekten daha iyiyse optimize edilen ağırlığı, değilse eşit ağırlığı kullanıyoruz.
    print(f"\nEnsemble ağırlıkları rolling-origin CV (4 fold, recursive) ile optimize ediliyor "
          f"(min taban: %{min_ensemble_weight*100:.0f})...")
    _, cv_ozet = optimize_ensemble_weights_cv(df, n_folds=4, n_trials=300, best_params=best_params,
                                              min_weight=min_ensemble_weight)
    print(f"  Eşit ağırlık CV WAPE    : %{cv_ozet['esit_wape']}")
    print(f"  Optimize ağırlık CV WAPE: %{cv_ozet['optimize_wape']}  {cv_ozet['agirliklar']}")

    if cv_ozet["optimize_wape"] < cv_ozet["esit_wape"]:
        final_weights = cv_ozet["agirliklar"]
        weight_method = (f"CV-optimized (4 fold, min taban %{min_ensemble_weight*100:.0f}) — "
                          f"esit agirliktan iyi cikti "
                          f"(esit %{cv_ozet['esit_wape']} vs optimize %{cv_ozet['optimize_wape']})")
    else:
        final_weights = {"LightGBM": 1 / 3, "XGBoost": 1 / 3, "CatBoost": 1 / 3}
        weight_method = (f"Esit agirlik (1/3 her biri) — CV'de optimize edilenden iyi/denk cikti "
                          f"(esit %{cv_ozet['esit_wape']} vs optimize %{cv_ozet['optimize_wape']})")
    print(f"  Seçilen yöntem: {weight_method}")

    # TEST karnesi (direct)
    preds = {isim: predict_sales(v["model"], te, fcols, cat_as=v["cat_as"]) for isim, v in tuned.items()}
    ens = blend(preds, final_weights)
    test_metrics = calculate_metrics(te[TARGET_COL], ens, "FINAL Ensemble (CV-tuned, TEST direct)")

    # ── QUANTILE (KANTIL) MODELLER: Alt ve Üst Sınır ────────────────────────
    # Aynı best_params ile, sadece loss fonksiyonunu değiştirip 6 ek model eğitiyoruz.
    print("\nQuantile modeller eğitiliyor (alt=%10, üst=%90)...")
    for alpha in [0.10, 0.90]:
        tag = f"q{int(alpha * 100):02d}"  # "q10" veya "q90"

        # LightGBM quantile
        lgb_q_params = {**best_params["LightGBM"], "alpha": alpha}
        m_lgb_q, _ = train_lightgbm(tr, va, params=lgb_q_params, objective="quantile")
        joblib.dump(m_lgb_q, out_dir / f"lightgbm_{tag}.joblib")

        # XGBoost quantile
        xgb_q_params = {**best_params["XGBoost"], "quantile_alpha": alpha}
        m_xgb_q, _ = train_xgboost(tr, va, params=xgb_q_params, objective="reg:quantileerror")
        joblib.dump(m_xgb_q, out_dir / f"xgboost_{tag}.joblib")

        # CatBoost quantile
        m_cat_q, _ = train_catboost(tr, va, params=best_params["CatBoost"],
                                    loss_function=f"Quantile:alpha={alpha}")
        joblib.dump(m_cat_q, out_dir / f"catboost_{tag}.joblib")
        print(f"  ✅ {tag} modelleri kaydedildi.")

    # ── META-MODEL (Stacking) ─────────────────────────────────────────────
    # OOF/VAL regresyonuyla kendi ağırlığını öğrenmek yerine, zaten Optuna +
    # 4-fold rolling-origin CV ile eşit ağırlığa karşı doğrulanmış
    # `final_weights`'i (yukarıda çözüldü) doğrudan kullanıyor.
    # NOT (2026-09-14): eskiden burada VAL tahminleriyle ayrı bir
    # LinearRegression fit ediliyordu. 4-fold rolling-origin CV ile test
    # edildi: bu regresyon küçük VAL örnekleminde güvenilmez sonuçlar
    # buluyordu (bir modele sıfır ağırlık verip bazen iyi bazen kötü çıkan
    # aşırı uç kararlar) — bkz. docs/degisiklik_gecmisi.md. `final_weights`
    # zaten daha güvenilir bir kaynaktan geliyor, tekrar öğrenmeye gerek yok.
    print("\nStacking (Meta-Model) ayarlanıyor (final_weights ile)...")
    from sklearn.linear_model import LinearRegression
    meta_model = LinearRegression(positive=True, fit_intercept=False)
    meta_model.coef_ = np.array([final_weights["LightGBM"], final_weights["XGBoost"], final_weights["CatBoost"]])
    meta_model.intercept_ = 0.0
    meta_model.n_features_in_ = 3
    meta_model.feature_names_in_ = np.array(['lgb', 'xgb', 'cat'], dtype=object)

    joblib.dump(meta_model, out_dir / "meta_model.joblib")
    print("  ✅ Meta-model kaydedildi.")

    config = {
        "feature_cols": fcols,
        "cat_cols": [c for c in CAT_COLS if c in fcols],
        "final_weights": final_weights,
        "cat_as": {isim: v["cat_as"] for isim, v in tuned.items()},
        "best_params": best_params,
        "tuning": f"CV-based ({n_folds} folds, direct scoring)",
        "ensemble_weight_method": weight_method,
        "ensemble_weight_cv_comparison": cv_ozet,
        "test_metrics_final": test_metrics,
        "has_quantile_models": True,
    }
    with open(out_dir / "ensemble_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"\nTEST (direct): {test_metrics}")
    print(f"Kaydedildi -> {out_dir}/")
    return config


# ======================================================================
# Ensemble (harman) oran optimizasyonu
# ======================================================================

def stress_test(df: pd.DataFrame, test_weeks: int = 12, out_path=None):
    """
    Leakage / ezber kontrolu. Son `test_weeks` haftada 3 senaryoyu kiyaslar:
      1. Naive baseline  -> gecen yilin ayni haftasi (lag_sales_52) birebir kopyalanir
      2. Tam ensemble    -> kaydedilmis optimize modeller (lag_52 dahil)
      3. lag_52 cikarilmis -> ayni 3 model (varsayilan ayar) lag_sales_52 olmadan egitilir

    Model geriye donuk kopyalasaydi (1) iyi, (3) kotu olurdu.
    Sonuclari models/stress_test.json'a yazar.
    """
    from src.model import run_recursive_backtest

    dates = np.sort(df["Date"].unique())
    te = df[df["Date"].isin(dates[-test_weeks:])].copy()
    rows = []

    # 1) Naive: gecen yilin ayni haftasi
    rows.append(calculate_metrics(te[TARGET_COL], te["lag_sales_52"], "1. Naive (gecen yil kopyasi)"))

    # 2) Tam ensemble (kaydedilmis optimize modeller)
    models, config = load_bundle()
    w = config["final_weights"]
    weights = (w["LightGBM"], w["XGBoost"], w["CatBoost"])
    vdf_full, _ = run_recursive_backtest(models, df, test_weeks=test_weeks, weights=weights)
    rows.append(calculate_metrics(vdf_full[TARGET_COL], vdf_full["Tahmin_Ensemble"],
                                  "2. Tam ensemble (lag_52 dahil)"))

    # 3) lag_52 cikarilmis -> 3 model varsayilan ayarla yeniden egitilir
    cols_to_drop = [c for c in ["lag_sales_52", "lag_sales_52_avg3", "seasonal_index_ly", "seasonal_index_ly_smooth"] if c in df.columns]
    df_no52 = df.drop(columns=cols_to_drop)
    tr2, va2, _ = split_train_val_test(df_no52)
    m_lgb, _ = train_lightgbm(tr2, va2)
    m_xgb, _ = train_xgboost(tr2, va2)
    m_cat, _ = train_catboost(tr2, va2)
    models_no52 = {"lgb": m_lgb, "xgb": m_xgb, "cat": m_cat,
                   "feature_names": get_feature_cols(df_no52)}
    vdf_no52, _ = run_recursive_backtest(models_no52, df_no52, test_weeks=test_weeks,
                                         weights=weights)
    rows.append(calculate_metrics(vdf_no52[TARGET_COL], vdf_no52["Tahmin_Ensemble"],
                                  "3. lag_52 cikarilmis (varsayilan ayar)"))

    out_path = Path(out_path) if out_path else Path(__file__).resolve().parent.parent / "models" / "stress_test.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    return rows


def rolling_origin_cv(df: pd.DataFrame, n_folds: int = 4, test_weeks: int = 12,
                      val_weeks: int = 12, best_params=None, weights=None, out_path=None):
    """
    Rolling-origin cross-validation.

    Test penceresini zamanda ileri kaydirarak modeli n_folds kez sinar.
    Her fold:
      - train : o gune kadarki veri (val_weeks haftasi haric)
      - val   : train'in son val_weeks haftasi (early stopping icin)
      - test  : sonraki test_weeks hafta (model bunu hic gormedi)
    Tahmin, app.py ile ayni yontemle (recursive backtest) yapilir.

    NOT: Optuna her fold'da TEKRAR CALISMAZ. Adim 5'te bulunan hiperparametreler
    (ensemble_config.json) sabit kullanilir. CV'nin isi yeni ayar bulmak degil,
    mevcut ayarlarin farkli donemlerde ne kadar tutarli oldugunu olcmek.

    Donen: (fold_df, ozet)  ve sonucu models/cv_results.json'a yazar.
    """
    from src.model import run_recursive_backtest

    if best_params is None or weights is None:
        _, config = load_bundle()
        if best_params is None:
            best_params = config["best_params"]
        if weights is None:
            w = config["final_weights"]
            weights = (w["LightGBM"], w["XGBoost"], w["CatBoost"])

    dates = np.sort(df["Date"].unique())
    n = len(dates)
    fcols = get_feature_cols(df)

    fold_rows = []
    for i in range(n_folds):
        # En son fold, verinin sonuna dayanir; oncekiler test_weeks kadar geride
        test_end = n - (n_folds - 1 - i) * test_weeks
        test_start = test_end - test_weeks
        val_start = test_start - val_weeks

        tr = df[df["Date"].isin(dates[:val_start])].copy()
        va = df[df["Date"].isin(dates[val_start:test_start])].copy()

        # 3 modeli bu fold'un verisiyle, Adim 5'in ayarlariyla egit
        m_lgb, _ = train_lightgbm(tr, va, params=best_params["LightGBM"])
        m_xgb, _ = train_xgboost(tr, va, params=best_params["XGBoost"])
        m_cat, _ = train_catboost(tr, va, params=best_params["CatBoost"])

        models_fold = {"lgb": m_lgb, "xgb": m_xgb, "cat": m_cat, "feature_names": fcols}

        # Bu fold'un test penceresi = df_upto'nun son test_weeks haftasi
        df_upto = df[df["Date"].isin(dates[:test_end])].copy()
        vdf, _ = run_recursive_backtest(models_fold, df_upto, test_weeks=test_weeks,
                                        weights=weights)

        row = calculate_metrics(vdf[TARGET_COL], vdf["Tahmin_Ensemble"], f"Fold {i + 1}")
        row["test_baslangic"] = str(pd.Timestamp(dates[test_start]).date())
        row["test_bitis"] = str(pd.Timestamp(dates[test_end - 1]).date())
        fold_rows.append(row)
        print(f"  Fold {i + 1}: {row['test_baslangic']} -> {row['test_bitis']}  WAPE %{row['WAPE']}")

    fold_df = pd.DataFrame(fold_rows)
    ozet = {
        "n_folds": n_folds,
        "WAPE_ortalama": round(float(fold_df["WAPE"].mean()), 2),
        "WAPE_sapma": round(float(fold_df["WAPE"].std(ddof=0)), 2),
        "WAPE_min": round(float(fold_df["WAPE"].min()), 2),
        "WAPE_max": round(float(fold_df["WAPE"].max()), 2),
        "R2_ortalama": round(float(fold_df["R2"].mean()), 3),
        "folds": fold_rows,
    }

    out_path = Path(out_path) if out_path else Path(__file__).resolve().parent.parent / "models" / "cv_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ozet, f, indent=2, ensure_ascii=False)
    return fold_df, ozet


def error_breakdown(df: pd.DataFrame, test_weeks: int = 40):
    """
    Kaydedilmis modellerle genis bir pencerede recursive backtest yapar ve
    hatayi uc acidan kirar:
      1. Seri bazinda (Cinsiyet x Kategori) -> hangi seri kotu?
      2. Hafta bazinda -> hangi haftalar kotu? (bayram/sezon deseni var mi?)
      3. En kotu tekil (seri, hafta) satirlari -> tek tek ne olmus?

    Donen: (by_series, by_week, worst_rows) uc DataFrame.
    """
    from src.model import run_recursive_backtest

    models, config = load_bundle()
    w = config["final_weights"]
    weights = (w["LightGBM"], w["XGBoost"], w["CatBoost"])

    vdf, _ = run_recursive_backtest(models, df, test_weeks=test_weeks, weights=weights)
    vdf = vdf.copy()
    vdf["abs_err"] = (vdf["SatisSayisi"] - vdf["Tahmin_Ensemble"]).abs()
    vdf["seri"] = vdf["Gender"].astype(str) + " " + vdf["StockGroupDesc"].astype(str)

    def _grupla(key):
        g = vdf.groupby(key).agg(
            tot_err=("abs_err", "sum"),
            tot_gercek=("SatisSayisi", "sum"),
            tot_tahmin=("Tahmin_Ensemble", "sum"),
            ort_haftalik_satis=("SatisSayisi", "mean"),
            n=("SatisSayisi", "size"),
        ).reset_index()
        g["WAPE"] = (g["tot_err"] / (g["tot_gercek"] + 1e-9) * 100).round(2)
        g["Bias"] = ((g["tot_tahmin"] - g["tot_gercek"]) / (g["tot_gercek"] + 1e-9) * 100).round(2)
        g["ort_haftalik_satis"] = g["ort_haftalik_satis"].round(0)
        return g.drop(columns=["tot_err", "tot_gercek", "tot_tahmin"])

    # 1) Seri bazinda
    by_series = _grupla("seri").sort_values("WAPE", ascending=False)

    # 2) Hafta bazinda (Week = takvim haftasi; bayram desenini gormek icin)
    by_week = _grupla("Date").sort_values("WAPE", ascending=False)
    hafta_map = vdf.groupby("Date")["Week"].first()
    by_week["takvim_haftasi"] = by_week["Date"].map(hafta_map).astype(int)

    # 3) En kotu tekil satirlar
    vdf["yuzde_hata"] = (vdf["abs_err"] / (vdf["SatisSayisi"].abs() + 1e-9) * 100).round(1)
    worst_rows = (vdf.sort_values("abs_err", ascending=False)
                  .head(12)[["Date", "Week", "seri", "SatisSayisi", "Tahmin_Ensemble", "Hata", "yuzde_hata"]]
                  .reset_index(drop=True))

    return by_series, by_week, worst_rows


def load_bundle(models_dir=None):
    """
    Adim 6'da kaydedilen 3 optimize modeli ve konfigurasyonu diskten yukler.

    Donen:
      models : {"lgb": <model>, "xgb": <model>, "cat": <model>, "feature_names": [...]}
               (app.py ve supply_chain.py bu anahtarlari bekliyor)
      config : ensemble_config.json icerigi (final_weights, best_params, cat_as, ...)
    """
    models_dir = Path(models_dir) if models_dir else Path(__file__).resolve().parent.parent / "models"

    with open(models_dir / "ensemble_config.json", encoding="utf-8") as f:
        config = json.load(f)

    models = {
        "lgb": joblib.load(models_dir / "lightgbm.joblib"),
        "xgb": joblib.load(models_dir / "xgboost.joblib"),
        "cat": joblib.load(models_dir / "catboost.joblib"),
        "feature_names": config["feature_cols"],
    }
    
    meta_path = models_dir / "meta_model.joblib"
    if meta_path.exists():
        models["meta_model"] = joblib.load(meta_path)

    # Quantile (Kantil) modelleri yükle (varsa)
    for tag in ["q10", "q90"]:
        lgb_q = models_dir / f"lightgbm_{tag}.joblib"
        xgb_q = models_dir / f"xgboost_{tag}.joblib"
        cat_q = models_dir / f"catboost_{tag}.joblib"
        if lgb_q.exists() and xgb_q.exists() and cat_q.exists():
            models[f"lgb_{tag}"] = joblib.load(lgb_q)
            models[f"xgb_{tag}"] = joblib.load(xgb_q)
            models[f"cat_{tag}"] = joblib.load(cat_q)

    return models, config


def blend(preds: dict, weights: dict) -> np.ndarray:
    """
    preds   : {"LightGBM": dizi, "XGBoost": dizi, "CatBoost": dizi}  -- gercek olcek tahminler
    weights : {"LightGBM": 0.4, ...}  -- toplami 1 olan agirliklar
    Donen   : agirlikli toplam tahmin dizisi
    """
    return sum(weights[name] * preds[name] for name in preds)


def normalize_weights(raw: dict) -> dict:
    """Ham sayilari toplami 1 olacak sekilde olceklendirir."""
    toplam = sum(raw.values()) + 1e-12
    return {k: v / toplam for k, v in raw.items()}


def optimize_ensemble_weights_cv(df: pd.DataFrame, n_folds: int = 4, n_trials: int = 300,
                                 test_weeks: int = 12, val_weeks: int = 12, best_params=None,
                                 min_weight: float = 0.0, seed: int = 42):
    """
    Ensemble harman agirliklarini tek val penceresi yerine `n_folds` rolling-origin
    fold'un RECURSIVE ortalama WAPE'sine gore optimize eder.

    Verimlilik: her fold icin 3 model BIR KEZ egitilir, recursive backtest yapilir,
    3 modelin tahminleri saklanir. Optuna sadece bu saklanan tahminlerin lineer
    kombinasyonunu deniyor -> deneme basi milisaniye.

    min_weight: her modele garanti edilen TABAN agirlik (0 <= min_weight <= 1/3).
        SEBEP (2026-09-16, bkz. docs/degisiklik_gecmisi.md): 3 model birbirine
        yuksek korelasyonlu oldugu icin serbest optimizasyon koseye (bir modele
        ~0 agirlik) kacma egiliminde -- kucuk bir kalite farki bile asiri uc bir
        agirliga donusebiliyor (LightGBM'in Prophet/PatchTST'e asiri guvenip
        %0.1'e dusmesi gibi). Taban, WAPE'ye neredeyse hic zarar vermeden
        (bu projede %10 tabanda WAPE degismedi) cesitliligi/saglamligi koruyor.
        Varsayilan 0.0 -- eski (kisitsiz) davranisi bozmaz, cagiran bilerek
        secmeli.
    Donen: (study, ozet)  ozet['esit_wape'] ile kiyas.
    """
    from src.model import run_recursive_backtest

    if best_params is None:
        _, config = load_bundle()
        best_params = config["best_params"]

    dates = np.sort(df["Date"].unique())
    n = len(dates)

    # her fold icin 3 modelin recursive tahminlerini onceden hesapla
    fold_preds = []
    for i in range(n_folds):
        test_end = n - (n_folds - 1 - i) * test_weeks
        va_start = test_end - test_weeks - val_weeks
        tr = df[df["Date"].isin(dates[:va_start])]
        va = df[df["Date"].isin(dates[va_start:test_end - test_weeks])]
        m_lgb, _ = train_lightgbm(tr, va, params=best_params["LightGBM"])
        m_xgb, _ = train_xgboost(tr, va, params=best_params["XGBoost"])
        m_cat, _ = train_catboost(tr, va, params=best_params["CatBoost"])
        mm = {"lgb": m_lgb, "xgb": m_xgb, "cat": m_cat, "feature_names": get_feature_cols(tr)}
        df_upto = df[df["Date"].isin(dates[:test_end])]
        # Esit agirlik BILEREK: asagida sadece pred_lgb/pred_xgb/pred_cat (her modelin
        # kendi bagimsiz zinciri) kullaniliyor, bunlar agirliktan etkilenmiyor. Agirlik
        # vermezsek fonksiyon diskteki ensemble_config.json'u okur -- yani "agirlik
        # ariyoruz" derken eski agirliga bagimli oluruz (ve ilk egitimde dosya henuz yok).
        vdf, _ = run_recursive_backtest(mm, df_upto, test_weeks=test_weeks,
                                        weights=(1 / 3, 1 / 3, 1 / 3))
        fold_preds.append(dict(
            y=vdf[TARGET_COL].to_numpy(float),
            lgb=vdf["pred_lgb"].to_numpy(float),
            xgb=vdf["pred_xgb"].to_numpy(float),
            cat=vdf["pred_cat"].to_numpy(float),
        ))
        print(f"  fold {i + 1} tahminleri hazir")

    def _mean_wape(w):
        wapes = []
        for fp in fold_preds:
            blend_p = w[0] * fp["lgb"] + w[1] * fp["xgb"] + w[2] * fp["cat"]
            wapes.append(np.sum(np.abs(fp["y"] - blend_p)) / (np.sum(fp["y"]) + 1e-9) * 100)
        return float(np.mean(wapes))

    esit_wape = _mean_wape([1 / 3, 1 / 3, 1 / 3])

    assert 0.0 <= min_weight <= 1 / 3, "min_weight en fazla 1/3 olabilir (3 model, toplam=1)"
    kalan_pay = 1.0 - 3 * min_weight  # taban dusuldukten sonra Optuna'nin serbestce paylastiracagi pay

    def _tabanli_agirlik(raw):
        # raw (Optuna'nin 0-1 arasi 3 ham sayisi) once kendi icinde normalize edilir
        # (toplami 1 yapilir), sonra taban eklenip kalan pay orantili dagitilir.
        # Sonuc HER ZAMAN: toplam == 1 VE her agirlik >= min_weight (matematiksel garanti).
        raw_norm = raw / raw.sum()
        return min_weight + raw_norm * kalan_pay

    def objective(trial):
        raw = np.array([trial.suggest_float(k, 0.0, 1.0) for k in ("w_lgb", "w_xgb", "w_cat")])
        if raw.sum() < 1e-9:
            return 1e9
        return _mean_wape(_tabanli_agirlik(raw))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)

    raw = np.array([study.best_params[k] for k in ("w_lgb", "w_xgb", "w_cat")])
    w_opt = _tabanli_agirlik(raw)
    ozet = {
        "esit_wape": round(esit_wape, 2),
        "optimize_wape": round(study.best_value, 2),
        "min_weight": min_weight,
        "agirliklar": {"LightGBM": round(float(w_opt[0]), 3),
                       "XGBoost": round(float(w_opt[1]), 3),
                       "CatBoost": round(float(w_opt[2]), 3)},
    }
    return study, ozet


def optimize_ensemble_weights(val_preds: dict, y_val_true, n_trials: int = 80, seed: int = 42):
    """
    3 modelin VAL tahminlerini alir, harman agirliklarini Optuna ile arar.
    Amac: harmanin VAL WAPE'sini minimize etmek.

    val_preds : {"LightGBM": dizi, "XGBoost": dizi, "CatBoost": dizi}  -- gercek olcek
    Donen     : optuna study (study.best_params ham agirliklari icerir)
    """
    names = list(val_preds.keys())
    y_val_true = np.asarray(y_val_true, dtype=float)

    def objective(trial):
        # Her model icin 0-1 arasi ham bir sayi iste (parametre adi = model adi).
        raw = {n: trial.suggest_float(n, 0.0, 1.0) for n in names}
        if sum(raw.values()) < 1e-9:
            return 1e9   # hepsi ~0 ise gecersiz, agir ceza
        w = normalize_weights(raw)                 # toplami 1'e getir
        pred = blend(val_preds, w)
        return calculate_metrics(y_val_true, pred)["WAPE"]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials)
    return study


def split_train_val_test(df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15):
    """
    build_features() çıktısını zamana göre (kronolojik) üç parçaya ayırır.

    Parametreler
    ----------
    df : pd.DataFrame
        İçinde 'Date' sütunu olan, öznitelikleri hesaplanmış tablo.
    val_frac : float
        Validation parçasının oranı (varsayılan %15).
    test_frac : float
        Test parçasının oranı (varsayılan %15).
        Train oranı otomatik olarak: 1 - val_frac - test_frac  ( = %70 ) olur.

    Döndürür
    -------
    (train_df, val_df, test_df) : üç ayrı DataFrame
    """

    # 1) Verideki BENZERSIZ haftaları bul ve küçükten büyüğe sırala.
    #    df'de her hafta birden çok kez var (her kategori için bir satır),
    #    ama bölmeyi TARIH bazında yapacağız, satır bazında değil.
    #    Yoksa aynı haftanın bazı kategorileri train'e, bazıları test'e düşer -> leakage.
    unique_dates = np.sort(df["Date"].unique())
    n = len(unique_dates)  # kaç benzersiz hafta var

    # 2) Kesim noktalarını (index) hesapla.
    #    Örnek: n=190 hafta, val=0.15, test=0.15
    #      train_end = int(190 * 0.70) = 133   -> ilk 133 hafta train
    #      val_end   = int(190 * 0.85) = 161   -> 133..161 arası val (28 hafta)
    #                                             161..190 arası test (29 hafta)
    train_frac = 1.0 - val_frac - test_frac
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    # 3) Tarih listesini üç dilime böl.
    #    Dilimler ÜST ÜSTE BINMEZ: [0:train_end], [train_end:val_end], [val_end:]
    train_dates = unique_dates[:train_end]
    val_dates = unique_dates[train_end:val_end]
    test_dates = unique_dates[val_end:]

    # 4) Asıl tabloyu bu tarih kümelerine göre filtrele.
    #    .isin(...) -> "bu satırın Date'i şu listede mi?" diye satır satır bakar.
    #    .copy()   -> parçalar üzerinde sonradan sütun eklerken pandas uyarısı almamak için.
    train_df = df[df["Date"].isin(train_dates)].copy()
    val_df = df[df["Date"].isin(val_dates)].copy()
    test_df = df[df["Date"].isin(test_dates)].copy()

    return train_df, val_df, test_df


def calculate_metrics(y_true, y_pred, name: str = "") -> dict:
    """
    Bir modelin tahminlerini 5 metrikle ozetler.

    y_true : gercek satislar (dizi)
    y_pred : model tahminleri (dizi)
    name   : sonuc tablosunda gozukecek model adi
    """
    # Girdileri numpy dizisine cevir (liste, pandas Series, dizi... hepsi calissin).
    # float'a zorluyoruz ki tam sayi bolmesi kazasi olmasin.
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Payda sifir olursa (hic satis yoksa) bolme hatasi olmasin diye kucuk sabit.
    eps = 1e-9
    toplam_gercek = np.sum(y_true)

    # --- R2: varyansin ne kadari aciklandi (1 = mukemmel, 0 = ortalama kadar) ---
    r2 = r2_score(y_true, y_pred)

    # --- RMSE: ortalama hata buyuklugu, satis birimiyle ayni ---
    # mean_squared_error kareli hatayi verir; karekoku RMSE.
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # --- MAPE: ortalama yuzde hata (sifira yakin gerceklerde sisebilir) ---
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    # --- WAPE: tum mutlak hatalarin toplami / tum gercek satis ---
    # Once topla sonra bol -> buyuk haftalar dogal olarak daha agir basar.
    wape = np.sum(np.abs(y_true - y_pred)) / (toplam_gercek + eps) * 100

    # --- Bias: hatanin YONU. Pozitif = fazla tahmin, negatif = az tahmin ---
    bias = (np.sum(y_pred) - toplam_gercek) / (toplam_gercek + eps) * 100

    # float(...) ile sarmaliyoruz: numpy skalerini duz Python float'a cevirir,
    # yoksa tabloda 'np.float64(9.72)' gibi gozukur.
    return {
        "Model": name,
        "R2": round(float(r2), 3),
        "WAPE": round(float(wape), 2),
        "MAPE": round(float(mape), 2),
        "RMSE": round(float(rmse), 1),
        "Bias": round(float(bias), 2),
    }


if __name__ == "__main__":
    # Bu blok yalnızca dosyayı doğrudan çalıştırınca devreye girer:
    #   python -m src.tuning
    from src.data_loader import load_and_preprocess_data
    from src.features import build_features

    df = build_features(load_and_preprocess_data())

    tr, va, te = split_train_val_test(df)

    print(f"Toplam benzersiz hafta: {df['Date'].nunique()}")
    print(f"Toplam satır          : {len(df):,}\n")

    for name, d in [("TRAIN", tr), ("VAL", va), ("TEST", te)]:
        pay = len(d) / len(df) * 100
        print(
            f"{name:5s}: {len(d):>6,} satır (%{pay:4.1f}) | "
            f"{d['Date'].min().date()} -> {d['Date'].max().date()} | "
            f"{d['Date'].nunique()} hafta"
        )

    # Güvenlik kontrolü: parçaların tarih aralıkları çakışmamalı.
    assert tr["Date"].max() < va["Date"].min(), "TRAIN ile VAL çakışıyor!"
    assert va["Date"].max() < te["Date"].min(), "VAL ile TEST çakışıyor!"
    print("\nKontrol OK: parçalar zaman ekseninde çakışmıyor.")

    # --- ADIM 2 testi: yazidaki ornekle ayni sayilari uretiyor mu? ---
    gercek = np.array([8000, 12000, 6000, 10000])
    model_a = np.array([8800, 13000, 6700, 11000])   # hep fazla tahmin
    model_b = np.array([8800, 11000, 6700, 9000])    # bir fazla bir eksik
    print("\nMetrik testi:")
    print(" ", calculate_metrics(gercek, model_a, "Model A"))
    print(" ", calculate_metrics(gercek, model_b, "Model B"))

    # --- ADIM 5: 3 modeli de Optuna ile optimize et, TEST karnelerini karsilastir ---
    fcols = get_feature_cols(tr)
    N_TRIALS = 40

    # Her model icin: (isim, optimize fonksiyonu, train fonksiyonu, kategorik tip)
    setups = [
        ("LightGBM", optimize_lightgbm, train_lightgbm, "category"),
        ("XGBoost", optimize_xgboost, train_xgboost, "category"),
        ("CatBoost", optimize_catboost, train_catboost, "str"),
    ]

    tuned_models = {}   # ensemble adiminda kullanacagiz
    rows = []

    for isim, optimize_fn, train_fn, cat_as in setups:
        print("\n" + "=" * 60)
        print(f"ADIM 5: {isim} optimize ediliyor ({N_TRIALS} deneme)...")

        study = optimize_fn(tr, va, n_trials=N_TRIALS)
        print(f"  En iyi VAL WAPE: %{study.best_value:.2f}")
        print(f"  En iyi ayarlar : {study.best_params}")

        # En iyi ayarlarla yeniden egit
        model, best_iter = train_fn(tr, va, params=study.best_params)
        print(f"  En iyi agac/iterasyon sayisi: {best_iter}")

        val_pred = predict_sales(model, va, fcols, cat_as=cat_as)
        test_pred = predict_sales(model, te, fcols, cat_as=cat_as)

        tuned_models[isim] = dict(
            model=model, cat_as=cat_as, params=study.best_params,
            val_pred=val_pred, test_pred=test_pred,
        )
        rows.append(calculate_metrics(va[TARGET_COL], val_pred, f"{isim} (VAL)"))
        rows.append(calculate_metrics(te[TARGET_COL], test_pred, f"{isim} (TEST)"))

    print("\n" + "=" * 60)
    print("ADIM 5 SONUC: 3 modelin karnesi")
    print("=" * 60)
    print(pd.DataFrame(rows).to_string(index=False))

    # --- ADIM 6: ensemble oran optimizasyonu ---
    print("\n" + "=" * 60)
    print("ADIM 6: Ensemble harman oranlari araniyor (80 deneme)...")

    y_val_true = va[TARGET_COL].to_numpy(dtype=float)
    y_test_true = te[TARGET_COL].to_numpy(dtype=float)
    val_preds = {k: v["val_pred"] for k, v in tuned_models.items()}
    test_preds = {k: v["test_pred"] for k, v in tuned_models.items()}

    # ── AĞIRLIK TİPİ 2: OPTUNA AĞIRLIĞI ─────────────────────────────────────────
    # optimize_ensemble_weights() VAL seti üzerinde 80 farklı ağırlık kombinasyonu
    # dener ve WAPE'yi en çok düşüren oranı döndürür.
    # DİKKAT: Val mevsimi ≠ Test mevsimi ise burada bulunan ağırlık Test'te
    # kötüleşebilir (Val'e overfit). Aşağıdaki karşılaştırma bunu ortaya koyar.
    study_ens = optimize_ensemble_weights(val_preds, y_val_true, n_trials=80)
    w_opt = normalize_weights(study_ens.best_params)  # AĞIRLIK TİPİ: OPTUNA

    print("\nOptuna'nin buldugu agirliklar:")
    for k, v in w_opt.items():
        print(f"   {k:10s}: {v:.3f}")

    # ── AĞIRLIK TİPİ 3: ELLE AĞIRLIK ─────────────────────────────────────────────
    # Proje başında deneme yanılmayla belirlenmiş sabit oranlar (bkz. dosya başı
    # tarihçe notu). Eşit ağırlıktan daha kötü çıktığı için üretimde kullanılmadı;
    # burada sadece kıyas amaçlı duruyor. Farklı oranları test etmek isterseniz
    # bu sözlüğü değiştirin.
    w_elle = {"LightGBM": 0.40, "XGBoost": 0.35, "CatBoost": 0.25}  # AĞIRLIK TİPİ: ELLE

    # ── AĞIRLIK TİPİ 1: EŞİT AĞIRLIK ────────────────────────────────────────────
    # Her modele tam olarak 1/3 verilir. Ne Optuna ne elle ayar gerekir.
    # Küçük veri setlerinde genellikle en sağlam (robust) seçimdir.
    w_esit = {k: 1 / 3 for k in val_preds}              # AĞIRLIK TİPİ: EŞİT

    # ── 3 AĞIRLIK TİPİ KARŞILAŞTIRMASI ──────────────────────────────────────────
    # Her ağırlık tipi için hem VAL hem TEST metriği hesaplanır.
    # Gerçek performansı görmek için TEST sütununa bakın — Val'e değil!
    ens_rows = []
    for etiket, w in [
        ("ELLE   (w_elle)",  w_elle),   # ← elle belirlenen oranlar
        ("EŞİT   (1/3x3)",  w_esit),   # ← eşit ağırlık
        ("OPTUNA (w_opt)",   w_opt),    # ← Optuna'nın Val üzerinde bulduğu oranlar
    ]:
        ev = blend(val_preds, w)
        et = blend(test_preds, w)
        ens_rows.append(calculate_metrics(y_val_true, ev, f"{etiket} (VAL)"))
        ens_rows.append(calculate_metrics(y_test_true, et, f"{etiket} (TEST)"))

    print("\n" + "=" * 60)
    print("ADIM 6 SONUC: 3 agirlik tipinin kiyasi  [TEST sutununa bakin!]")
    print("=" * 60)
    print(pd.DataFrame(ens_rows).to_string(index=False))

    # ── FINAL MODEL ──────────────────────────────────────────────────────────────
    # NOT: TEST sonucuna bakarak agirlik SECMIYORUZ (bu, TEST'i model secimine
    # sizdirir ve TEST'in "hic gorulmemis veri" anlamini bozar). Karar VAL-tabanli
    # kaliyor: Optuna'nin TEK VAL penceresinde bulduguna guvenmek yerine (bu tam
    # olarak 4 Eylul'de VAL'e overfit ettigi icin terk edilen yontemdi), guvenli
    # varsayilan olan ESIT agirlik kullanilir. Rolling-origin CV ile dogrulanmis,
    # esit agirliga karsi gercekten daha iyi cikan bir agirlik istiyorsaniz
    # run_cv_tuning() kullanin (o fonksiyon 4-fold CV ile karsilastirip secer).
    final_weights = w_esit
    final_val  = blend(val_preds,  final_weights)
    final_test = blend(test_preds, final_weights)

    final_rows = [r for r in rows if "(TEST)" in r["Model"]]
    final_test_metrics = calculate_metrics(y_test_true, final_test, "FINAL Ensemble Esit (TEST)")
    final_rows.append(final_test_metrics)

    print("\n" + "=" * 60)
    print("FINAL: TEST seti karnesi (final model = ESIT agirlik; yukaridaki")
    print("       'TEST sutunu' sadece bilgi amacli, secime dahil edilmedi)")
    print("=" * 60)
    print(pd.DataFrame(final_rows).to_string(index=False))

    # --- META-MODEL (Stacking) ---
    # Ayri bir VAL regresyonu yerine (kucuk ornekte guvenilmez, bkz.
    # run_cv_tuning()'deki ayni not), yukarida zaten secilen final_weights
    # kullaniliyor -- tutarlilik icin.
    from sklearn.linear_model import LinearRegression
    meta_model = LinearRegression(positive=True, fit_intercept=False)
    meta_model.coef_ = np.array([final_weights["LightGBM"], final_weights["XGBoost"], final_weights["CatBoost"]])
    meta_model.intercept_ = 0.0
    meta_model.n_features_in_ = 3
    meta_model.feature_names_in_ = np.array(['lgb', 'xgb', 'cat'], dtype=object)

    # --- Modelleri ve konfigurasyonu diske kaydet ---
    out_dir = Path(__file__).resolve().parent.parent / "models"
    out_dir.mkdir(exist_ok=True)

    for isim, info in tuned_models.items():
        joblib.dump(info["model"], out_dir / f"{isim.lower()}.joblib")
    
    joblib.dump(meta_model, out_dir / "meta_model.joblib")

    config = {
        "feature_cols": fcols,
        "cat_cols": [c for c in CAT_COLS if c in fcols],
        "final_weights": final_weights,
        "ensemble_weight_method": "Esit agirlik (1/3 her biri) — __main__ bloğu TEST'i secime sizdirmamak icin varsayilan olarak esit agirlik kullanir.",
        "cat_as": {isim: info["cat_as"] for isim, info in tuned_models.items()},
        "best_params": {isim: info["params"] for isim, info in tuned_models.items()},
        "test_metrics_final": final_test_metrics,
        "test_metrics_per_model": {
            isim: calculate_metrics(y_test_true, info["test_pred"], isim)
            for isim, info in tuned_models.items()
        },
    }
    with open(out_dir / "ensemble_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\nKaydedildi -> {out_dir}/")
    print("  lightgbm.joblib, xgboost.joblib, catboost.joblib, ensemble_config.json")
