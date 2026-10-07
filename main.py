from src.data_loader import load_and_preprocess_data
from src.features import build_features
from src.model import run_recursive_backtest, train_and_validate_ensemble
from src.tuning import load_bundle

def main():
    print("1. [data_loader] Veriler okunuyor ve zaman ızgarası kuruluyor...")
    df_clean = load_and_preprocess_data()

    print("\n2. [features] Sızıntısız öznitelikler hesaplanıyor...")
    df_features = build_features(df_clean)

    try:
        print("\n3. [tuning] Üretimdeki CV-optimize modeller yükleniyor (models/)...")
        models, config = load_bundle()
        w = config["final_weights"]
        df_test_results, df_metrics = run_recursive_backtest(
            models, df_features, test_weeks=12,
            weights=(w["LightGBM"], w["XGBoost"], w["CatBoost"]),
        )
    except FileNotFoundError:
        print("\n3. [model] models/ klasöründe kayıtlı model yok -- CV'siz (sabit ayarlı,")
        print("   daha zayıf) ensemble sıfırdan eğitiliyor. Gerçek (CV-tuned, min ağırlık")
        print("   tabanlı) üretim modellerini yeniden üretmek için `python -m src.tuning`")
        print("   ÇALIŞTIRMAYIN -- o, terk edilmiş eski tek-VAL yöntemini kullanır ve")
        print("   models/ içeriğinin üzerine eşit ağırlıklı zayıf modeller yazar. Bunun")
        print("   yerine README.md'deki adımı izleyin:")
        print("     from src.tuning import run_cv_tuning")
        print("     run_cv_tuning(df_features, n_trials=40, n_folds=3)")
        models, df_test_results, df_metrics = train_and_validate_ensemble(df_features, test_weeks=12)

    # Örnek tahmin vs gerçekleşen satırları
    sample_cols = ['Date', 'Gender', 'StockGroupDesc', 'SatisSayisi', 'Tahmin_Ensemble', 'Hata']
    print("\nSon Test Haftalarından Örnek Sonuçlar:")
    print(df_test_results[sample_cols].head(10).to_string(index=False))

if __name__ == "__main__":
    main()