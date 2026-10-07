from pathlib import Path
import pandas as pd


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def read_csv_safe(file_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path, sep=None, engine='python', encoding='utf-8', thousands='.')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, sep=None, engine='python', encoding='iso-8859-9', thousands='.')
    df.columns = df.columns.str.strip()
    return df


def get_iso_monday(year: int, week: int) -> pd.Timestamp:
    """
    Herhangi bir Yıl ve Hafta için kesin Pazartesi tarihini üretir (ISO standardı).
    """
    # Yılın 4 Ocak günü kesinlikle 1. haftanın içindedir
    jan4 = pd.Timestamp(year, 1, 4)
    start_of_week1 = jan4 - pd.Timedelta(days=jan4.isoweekday() - 1)
    return start_of_week1 + pd.Timedelta(weeks=week - 1)


def load_and_preprocess_data() -> pd.DataFrame:
    # Gerçek veri (gizli, git'e girmez) varsa onu, yoksa depodaki uydurma örnek veriyi kullan.
    data_dir = get_project_root() / "data"
    if not (data_dir / "kadin_satis.csv").exists():
        data_dir = get_project_root() / "sample_data"
        print("UYARI: data/ bulunamadı -- sample_data/ altındaki UYDURMA örnek veri kullanılıyor.")

    # 1. Ham Verileri Oku
    df_ks = read_csv_safe(data_dir / "kadin_satis.csv").assign(Gender="Kadin")
    df_es = read_csv_safe(data_dir / "erkek_satis.csv").assign(Gender="Erkek")
    df_kst = read_csv_safe(data_dir / "kadin_stok.csv").assign(Gender="Kadin")
    df_est = read_csv_safe(data_dir / "erkek_stok.csv").assign(Gender="Erkek")

    df_sales_raw = pd.concat([df_ks, df_es], ignore_index=True)
    df_stock_raw = pd.concat([df_kst, df_est], ignore_index=True)

    # 2. String ve Sayısal Temizlik
    for df in [df_sales_raw, df_stock_raw]:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)
        df['Week'] = pd.to_numeric(df['Week'], errors='coerce').fillna(0).astype(int)
        df['StockGroupDesc'] = df['StockGroupDesc'].astype(str).str.strip().str.upper()
        # GroupDesc farklılıklarını (Kadın/Kadin) nötrle
        df['GroupDesc'] = df['Gender']

    df_sales_raw['SatisSayisi'] = pd.to_numeric(df_sales_raw['SatisSayisi'], errors='coerce').fillna(0).astype(float)
    df_stock_raw['StokSayisi'] = pd.to_numeric(df_stock_raw['StokSayisi'], errors='coerce').fillna(0).astype(float)
    df_stock_raw['MagazaSayisi'] = pd.to_numeric(df_stock_raw['MagazaSayisi'], errors='coerce').fillna(1).astype(float)

    # 3. Kırılım Seviyesinde Toplama
    key_cols = ['Year', 'Week', 'Gender', 'GroupDesc', 'StockGroupDesc']
    df_sales_agg = df_sales_raw.groupby(key_cols, as_index=False).agg({'SatisSayisi': 'sum'})
    df_stock_agg = df_stock_raw.groupby(key_cols, as_index=False).agg({
        'StokSayisi': 'sum',
        'MagazaSayisi': 'max'
    })

    # 4. Satış ve Stok Tablolarını Year-Week Üzerinden Birleştir (Tarih bağımsız kusursuz eşleşme)
    # Tüm kombinasyonlar (Gender x StockGroupDesc)
    unique_series = df_sales_agg[['Gender', 'GroupDesc', 'StockGroupDesc']].drop_duplicates()

    # Veri setinin ilk yılının 1. haftasından son yılın son haftasına kadar tüm haftalar.
    # Tamamlanmış yıllar 52 haftaya sabitlenir, SON (yarım) yıl veride görülen son
    # haftada biter. NOT (2026-09-14): burada eskiden yıl "2026" diye elle yazılıydı;
    # veri ilerleyince sessizce yanlışlaşacağı için son yıl dinamik okunuyor --
    # bugünkü veride davranış birebir aynı.
    #
    # BİLİNEN KISIT — 53. hafta: kaynak CSV'lerde 2023/2024/2025 için Week=53 satırları
    # var (toplam 33.019 çift, tüm satışın ~%0,35'i) ama ızgara 52 haftada bittiği için
    # bu satırlar left-merge'de eşleşmeyip SESSİZCE DÜŞÜYOR. Kasıtsız değil de kaçınılmaz:
    # get_iso_monday(2023, 53) == get_iso_monday(2024, 1) == 2024-01-01, yani 53. hafta
    # korunursa ertesi yılın 1. haftasıyla aynı tarihe çakışıyor (kaynak ISO değil,
    # perakende takvimi kullanıyor olmalı). Düzgün çözüm 53. haftayı 52'ye veya ertesi
    # yılın 1. haftasına toplamaktır; bu eğitim verisini değiştireceği için yapılmadı.
    son_yil = int(max(df_sales_agg['Year'].unique()))
    all_weeks = []
    for y in sorted(df_sales_agg['Year'].unique()):
        max_w = 52 if y != son_yil else df_sales_agg[df_sales_agg['Year'] == son_yil]['Week'].max()
        for w in range(1, max_w + 1):
            all_weeks.append({'Year': y, 'Week': w})

    df_weeks = pd.DataFrame(all_weeks)
    grid = unique_series.assign(key=1).merge(df_weeks.assign(key=1), on='key').drop('key', axis=1)

    # Satış ve Stoku Izgaraya Bağla
    final_df = grid.merge(df_sales_agg, on=key_cols, how='left')
    final_df['SatisSayisi'] = final_df['SatisSayisi'].fillna(0.0)

    final_df = final_df.merge(df_stock_agg, on=key_cols, how='left')
    final_df['StokSayisi'] = final_df.groupby(['Gender', 'StockGroupDesc'])['StokSayisi'].ffill().bfill().fillna(0.0)
    final_df['MagazaSayisi'] = final_df.groupby(['Gender', 'StockGroupDesc'])['MagazaSayisi'].ffill().bfill().fillna(
        1.0)

    # Tarihi En Son Adımda Kesin Olarak Üret
    final_df['Date'] = [get_iso_monday(y, w) for y, w in zip(final_df['Year'], final_df['Week'])]

    # Sırala
    final_df = final_df.sort_values(by=['Gender', 'StockGroupDesc', 'Date']).reset_index(drop=True)

    return final_df