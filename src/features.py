from pathlib import Path

import numpy as np
import pandas as pd

from src.config import BTS_SEGMENTS

PROPHET_WALKFORWARD_PATH = Path(__file__).resolve().parent.parent / "models" / "prophet_walkforward_features.csv"
PATCHTST_WALKFORWARD_PATH = Path(__file__).resolve().parent.parent / "models" / "patchtst_walkforward_features.csv"


def _bayram_haftasi(year: int, ref_gun_of_year: int) -> int:
    """
    Hicri bayram (Eid) hafta numarasi. Hicri takvim her yil ~10.875 gun geriye kayar.
    `ref_gun_of_year` = 2023'teki bayramin yilin kacinci gunu oldugu.
    ONEMLI: tarihi HEDEF yilda yeniden kuruyoruz; yoksa ISO hafta numarasi
    yillar arasi 1 hafta kayabiliyor (eski kodun hatasi).
    """
    hedef_gun = round(ref_gun_of_year - (year - 2023) * 10.875)
    hedef_tarih = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=hedef_gun - 1)
    return int(hedef_tarih.isocalendar().week)


def get_dynamic_events(year: int):
    # 2023: Ramazan Bayrami 21 Nisan (yilin 111. gunu), Kurban 28 Haziran (179. gun)
    eid_ramazan_w = _bayram_haftasi(year, 111)
    eid_kurban_w = _bayram_haftasi(year, 179)

    may_sundays = [d for d in pd.date_range(f"{year}-05-01", f"{year}-05-31") if d.weekday() == 6]
    june_sundays = [d for d in pd.date_range(f"{year}-06-01", f"{year}-06-30") if d.weekday() == 6]

    mothers_week = may_sundays[1].isocalendar().week if len(may_sundays) > 1 else 19
    fathers_week = june_sundays[2].isocalendar().week if len(june_sundays) > 2 else 24

    # Ramazan alisverisi tum oruc ayina yayilir; pik, bayramdan onceki 2 haftadir.
    # Bayram haftasinin kendisinde satis genelde DUSER (magazalar kapali, seyahat).
    return {
        'ramazan_rampasi': [w for w in range(eid_ramazan_w - 4, eid_ramazan_w) if w >= 1],
        'ramazan_pik': [w for w in (eid_ramazan_w - 2, eid_ramazan_w - 1) if w >= 1],
        'ramazan_bayrami': [eid_ramazan_w],
        'kurban_rampasi': [w for w in (eid_kurban_w - 2, eid_kurban_w - 1) if w >= 1],
        'kurban_bayrami': [eid_kurban_w],
        'anneler': [mothers_week],
        'babalar': [fathers_week],
        'okul': [35, 36, 37, 38],
        'kasim': [46, 47, 48],
        'yilbasi': [51, 52],
        'yeni_yil_dususu': [1, 2, 3],  # hata analizi: Ocak ilk haftalari en kotu dönem
    }


def add_retail_events(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # olay adi -> get_dynamic_events sozlugundeki anahtar
    olaylar = {
        'ramazan_rampasi': 'ramazan_rampasi',
        'ramazan_pik': 'ramazan_pik',
        'ramazan_bayrami': 'ramazan_bayrami',
        'kurban_rampasi': 'kurban_rampasi',
        'kurban_bayrami': 'kurban_bayrami',
        'anneler_gunu': 'anneler',
        'babalar_gunu': 'babalar',
        'okula_donus': 'okul',
        'kasim_indirimleri': 'kasim',
        'yilbasi_etkisi': 'yilbasi',
        'yeni_yil_dususu': 'yeni_yil_dususu',
    }
    for col in olaylar:
        df[col] = 0

    for yr in df['Year'].unique():
        events = get_dynamic_events(int(yr))
        yil_maske = df['Year'] == yr
        for col, anahtar in olaylar.items():
            df.loc[yil_maske & (df['Week'].isin(events[anahtar])), col] = 1

    # Eski model uyumlulugu: 'ramazan_etkisi' / 'kurban_etkisi' hala uretiliyor
    # (rampa + pik birlesimi). Boylece kaydedilmis modeller de calismaya devam eder.
    df['ramazan_etkisi'] = df[['ramazan_rampasi', 'ramazan_pik']].max(axis=1)
    df['kurban_etkisi'] = df[['kurban_rampasi', 'kurban_bayrami']].max(axis=1)

    df['event_kadin_anneler'] = ((df['anneler_gunu'] == 1) & (df['Gender'] == 'Kadin')).astype(int)
    df['event_erkek_babalar'] = ((df['babalar_gunu'] == 1) & (df['Gender'] == 'Erkek')).astype(int)
    df['event_okul_sneaker'] = ((df['okula_donus'] == 1) & (df['StockGroupDesc'].isin(BTS_SEGMENTS))).astype(int)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    group_cols = ['Gender', 'GroupDesc', 'StockGroupDesc']

    # Veriyi zaman sırasına diz (Sızıntıyı önlemek için)
    df = df.sort_values(by=group_cols + ['Date'])

    # 1. Dinamik Takvim
    df = add_retail_events(df)

    # 2. Temel Fourier
    for k in [1, 2]:
        df[f'sin_fourier_{k}'] = np.sin(2 * np.pi * k * df['Week'] / 52.1775)
        df[f'cos_fourier_{k}'] = np.cos(2 * np.pi * k * df['Week'] / 52.1775)

    # 3. Dışsal Değişkenler (t-1)
    df['lag_stock_1'] = df.groupby(group_cols)['StokSayisi'].shift(1)
    df['lag_store_1'] = df.groupby(group_cols)['MagazaSayisi'].shift(1)
    # NOT (2026-09-14, CV ile dogrulandi): bu esik (mutlak VEYA rolling_mean_4'e
    # goreceli, ikisi de denendi) bu veri setinde HIC TETIKLENMIYOR -- lag_stock_1,
    # rolling_mean_4'un en az %58'i seviyesinde kaliyor, gercek stoksuz kalma hic
    # yasanmamis. Yani bu kolon su an surekli 0 (olu sinyal). Kayitli uretim
    # modelleri bu kolonu bekledigi icin (feature seti degismesin diye) kaldirilmadi,
    # ama gelecekte hacim buyuyup gercek stoksuzluk yasanirsa yeniden anlamli olabilir.
    df['is_stockout_risk_lag1'] = (
        (df['lag_stock_1'] <= 2) & (df.groupby(group_cols)['SatisSayisi'].shift(1) == 0)
    ).astype(int)

    # 4. Gecikmeli Satışlar (Strict Lags)
    #    NOT: lag_sales_1 (1 haftalik gecikme) BILEREK YOK. Model kararinin ~%40'ini
    #    ona verip "bu hafta ≈ gecen hafta" diyordu; recursive (cok adimli) tahminde
    #    hata katlaniyordu. Cikardik -> CV WAPE %34 -> %28, sapma ±%14 -> ±%3.
    df['lag_sales_2'] = df.groupby(group_cols)['SatisSayisi'].shift(2)
    df['lag_sales_4'] = df.groupby(group_cols)['SatisSayisi'].shift(4)
    df['lag_sales_52'] = df.groupby(group_cols)['SatisSayisi'].shift(52)

    # 5. Hareketli Ortalamalar (t-1 bazlı)
    for w in [4, 12, 52]:
        df[f'rolling_mean_{w}'] = df.groupby(group_cols)['SatisSayisi'].transform(
            lambda x: x.shift(1).rolling(window=w, min_periods=1).mean()
        )

    df['ema_sales_4'] = df.groupby(group_cols)['SatisSayisi'].transform(
        lambda x: x.shift(1).ewm(span=4, adjust=False).mean()
    )

    # 5b. Mevsimsel indeks (sizintisiz) -- "bu seri, bu hafta, kendi yillik
    #     seviyesinin kacta kaci?" yaz urunleri gibi mevsim-disi urunlerin kis
    #     tahminini duzeltmek icin. Sadece GECMIS veriden hesaplanir.
    #     lag_sales_52_avg3 = gecen yilin AYNI haftasi +/- 1 hafta ortalamasi
    #     (T-53, T-52, T-51), simetrik. NOT (2026-09-11): eskiden shift(51)
    #     kullaniliyordu ve asimetrik bir pencere (T-52,T-51,T-50) uretiyordu --
    #     bu ayni zamanda supply_chain.py'nin recursive tahmindeki manuel
    #     hesabiyla (simulated_sales[-53:-50], hep simetrikti) UYUSMUYORDU.
    #     shift(52)'ye duzeltilerek train/inference tutarliligi da saglandi.
    df['lag_sales_52_avg3'] = df.groupby(group_cols)['SatisSayisi'].transform(
        lambda x: x.shift(52).rolling(window=3, center=True, min_periods=1).mean()
    )
    df['seasonal_index_ly'] = df['lag_sales_52'] / (df['rolling_mean_52'] + 1e-6)
    df['seasonal_index_ly_smooth'] = df['lag_sales_52_avg3'] / (df['rolling_mean_52'] + 1e-6)

    # 6. Katma Değerli Oranlar
    df['sales_momentum'] = df['rolling_mean_4'] / (df['rolling_mean_12'] + 1e-6)
    df['stock_cover_pressure'] = df['lag_stock_1'] / (df['rolling_mean_4'] + 1e-6)

    # ---------------- YENİ EKLENEN ÖZELLİKLER ----------------
    # 6.a Karşı Cinsiyet Sinyali (Cross-Gender Halo Effect)
    # Geçen haftanın satışı
    df['Gecen_Hafta_Satis'] = df.groupby(group_cols)['SatisSayisi'].shift(1)
    
    def get_opposite_gender(g):
        if g == 'Kadin': return 'Erkek'
        if g == 'Erkek': return 'Kadin'
        return g
        
    df_opp = df[['Date', 'Gender', 'StockGroupDesc', 'Gecen_Hafta_Satis']].copy()
    df_opp['Gender'] = df_opp['Gender'].apply(get_opposite_gender)
    df_opp = df_opp.rename(columns={'Gecen_Hafta_Satis': 'cross_gender_lag_1'})
    
    # Mevcut df ile merge et (Sol birleşim)
    # GroupDesc Gender ile ayni oldugu icin on=[] kismindan cikarildi, yoksa eslesmez.
    df = pd.merge(df, df_opp, on=['Date', 'Gender', 'StockGroupDesc'], how='left')
    df['cross_gender_lag_1'] = df['cross_gender_lag_1'].fillna(0)
    df = df.drop(columns=['Gecen_Hafta_Satis'])

    # 6.c Kategori Depo Doluluk Oranı (Scale-Invariant Stock Segmentation)
    # Gelecekten kopya çekmemesi için Expanding Max (o güne kadar görülmüş en yüksek stok)
    # DENENDI (2026-09-14): magaza-basina-stoga normalize edilmis versiyon
    # (mevsimsel serilerin magaza sayisi degisimini hesaba katmak icin, mentor elestirisi)
    # rolling-origin CV'de test edildi: WAPE %21.86 -> %22.40 (kotulesti,
    # sapma biraz dustu ama net kazanc yok) -- teorik olarak daha dogru ama
    # pratikte iyilestirmedigi icin GERI ALINDI, eski (magaza-normalize
    # edilmemis) hali korundu.
    df['max_stock_seen'] = df.groupby(group_cols)['lag_stock_1'].transform(lambda x: x.expanding().max())
    df['stock_capacity_ratio'] = df['lag_stock_1'] / (df['max_stock_seen'] + 1)
    df = df.drop(columns=['max_stock_seen'])

    # 6.b Zaman İndeksi (Genel Büyüme Trendi)
    # ONEMLI (2026-09-11): eskiden "weeks_since_launch" (urun yasam dongusu) diye
    # adlandirilmisti ama bu veri setinde 8 serinin de gercek bir lansman tarihi
    # yok -- hepsi izgaranin ilk haftasindan beri var, yani deger tum serilerde
    # BIREBIR AYNIYDI. Isim yanlisti ("hangi urun ne kadar yeni" bilgisi vermiyor)
    # ama CV testinde cikarilinca performans kotulesti (WAPE %21->%23, en kotu
    # fold %25->%33) -- cunku bu, Fourier/Week gibi donguselerin aksine TEK
    # MONOTON ARTAN sinyaldi ve isin buyume trendini (bkz. docs/ "+%19 YoY")
    # takip etmenin tek yoluydu. Sinyali korumak icin isim/aciklama duzeltildi,
    # deger ayni: veri setinin ilk haftasindan (2023-01-02) bu yana gecen hafta.
    df['zaman_indeksi'] = ((df['Date'] - df['Date'].min()).dt.days // 7).astype(int)
    # -----------------------------------------------------------

    # 7. VERI SIZINTISI TEMIZLIGI
    #    Asagidaki sutunlar AYNI haftanin satisini iceriyor -> tahmin aninda bilinemez:
    #      StokSayisi      = o haftanin KAPANIS stogu (satista mekanik bagli)
    #      MagazaSayisi    = o haftanin magaza sayisi (contemporaneous; lag_store_1 kullanilir)
    #    Modele SADECE gecmis (lag'li) versiyonlar girer: lag_stock_1, lag_store_1.
    #    (Lag_Stok/GelenUrun/SellThroughRate data_loader.py'den kaldirildi -- hicbir
    #    yerde kullanilmiyorlardi, hesaplanip hemen atiliyorlardi.)
    sizinti = ['StokSayisi', 'MagazaSayisi']
    df = df.drop(columns=[c for c in sizinti if c in df.columns])

    for col in group_cols:
        df[col] = df[col].astype('category')

    # 8. DISSAL MODEL FEATURE'LARI (Prophet / PatchTST)
    #    notebooks/exp_prophet.ipynb (Asama 12) ve exp_patchtst.ipynb (Asama 11)'de
    #    sizintisiz walk-forward ile uretilen tahminler, mevcut 42 feature'a
    #    stacking girdisi olarak ekleniyor (bkz. src/aux_forecasters.py).
    #    ONEMLI: walk-forward MIN_TRAIN_WEEKS=104 sinirindan once (veri setinin
    #    ilk ~52 haftasi) bu iki sutun NaN kalir -- bu satirlar asagidaki
    #    dropna()'nin disinda tutuluyor (subset ile), yoksa 1104 satirin 416'si
    #    yanlislikla dusuyor olurdu.
    key_cols = ['Date', 'Gender', 'GroupDesc', 'StockGroupDesc']
    #    Bu dosyalar gercek veriyle uretildigi icin depoda yok; yoksa bu iki
    #    opsiyonel feature atlanir (ana model onlar olmadan da calisir).
    if PROPHET_WALKFORWARD_PATH.exists() and PATCHTST_WALKFORWARD_PATH.exists():
        prophet_wf = pd.read_csv(PROPHET_WALKFORWARD_PATH, parse_dates=['Date'])
        patchtst_wf = pd.read_csv(PATCHTST_WALKFORWARD_PATH, parse_dates=['Date'])
        df = df.merge(prophet_wf, on=key_cols, how='left').merge(patchtst_wf, on=key_cols, how='left')
        # merge() kategorik sutunlari object'e geri cevirir -- geri kategorik yap
        for col in group_cols:
            df[col] = df[col].astype('category')

    dropna_subset = [c for c in df.columns if c not in ('prophet_tahmin', 'patchtst_tahmin')]
    return df.dropna(subset=dropna_subset).reset_index(drop=True)