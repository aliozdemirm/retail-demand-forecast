import json
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.data_loader import load_and_preprocess_data
from src.features import build_features
from src.model import train_and_validate_ensemble, run_recursive_backtest, get_feature_importance_df
from src.supply_chain import generate_recursive_forecast, calculate_order_decision
from src.tuning import load_bundle

# Sayfa Konfigürasyonu
st.set_page_config(
    page_title="Retail Demand & Inventory Decision Engine",
    page_icon="👟",
    layout="wide"
)

# Kurumsal Öznitelik Açıklama Sözlüğü
FEATURE_METADATA = {
    # 1. Gecikmeli Satış (Lag) ve Momentum Değişkenleri
    'lag_sales_2': {'name': 'T-2 Satış', 'desc': 'İki hafta önceki net satış adedi.'},
    'lag_sales_4': {'name': 'T-4 Satış (Geçen Ay)', 'desc': 'Dört hafta (yaklaşık 1 ay) önceki net satış adedi.'},
    'lag_sales_52': {'name': 'T-52 Satış (Geçen Yıl)', 'desc': 'Geçen yılın tam olarak aynı takvim haftasındaki satış adedi (Yıllık baz etkisi).'},
    'lag_sales_52_avg3': {'name': 'Geçen Yıl Aynı Dönem (3 Haftalık Ort.)', 'desc': 'Geçen yılın ilgili haftası ve etrafındaki (T-53, T-52, T-51) haftaların simetrik ortalaması. Takvim kaymalarına karşı yumuşatılmış yıllık mevsimsellik.'},
    'rolling_mean_4': {'name': '4 Haftalık Hareketli Ortalama', 'desc': 'Son 1 ayın ortalama haftalık satış hızı (Kısa vadeli trend).'},
    'rolling_mean_12': {'name': '12 Haftalık Hareketli Ortalama', 'desc': 'Son 1 çeyreğin (3 ay) ortalama haftalık satış hızı (Orta vadeli trend).'},
    'rolling_mean_52': {'name': '52 Haftalık Hareketli Ortalama', 'desc': 'Son 1 yılın ortalama haftalık satış hızı (Uzun vadeli trend ve baz hacim).'},
    'ema_sales_4': {'name': 'Üstel Hareketli Ortalama (EMA-4)', 'desc': 'Yakın geçmişe daha fazla ağırlık veren, son 1 aylık hassas trend göstergesi.'},
    'sales_momentum': {'name': 'Satış İvmesi (Momentum)', 'desc': 'Kısa vadeli trendin (4 hafta), orta vadeli trende (12 hafta) oranı. Ürünün ivme kazanıp kazanmadığını gösterir.'},
    'cross_gender_lag_1': {'name': 'Karşı Cinsiyet Etkisi (Halo Sinyali)', 'desc': 'Aynı kategorideki karşı cinsiyet ürünlerinin geçen haftaki satış hacmi. Aile ve çift alışverişlerindeki geçişkenliği yakalar.'},
    'stock_capacity_ratio': {'name': 'Kategori Depo Doluluk Oranı', 'desc': 'Kategorinin kendi tarihi zirvesine göre güncel stok doluluk yüzdesi. Büyük/Küçük kategori karmaşasını önler ve modelin ölçekten bağımsız karar almasını sağlar.'},
    'zaman_indeksi': {'name': 'Zaman İndeksi (Genel Büyüme Trendi)', 'desc': 'Veri setinin ilk haftasından (2023-01-02) bu yana geçen hafta sayısı — tüm seriler için aynı, tekil bir sayaç. Fourier/mevsimsellik özellikleri döngüsel olduğu için işin genel büyüme/küçülme trendini (yıldan yıla artış) yakalayan tek sinyal budur.'},

    # 2. Stok ve Operasyonel Kısıt Değişkenleri
    'lag_stock_1': {'name': 'T-1 Kapanış Stoku', 'desc': 'Bir önceki hafta devreden depo/sistem stok adedi. Talebin karşılanabilirliği için üst limittir.'},
    'lag_store_1': {'name': 'Aktif Mağaza Sayısı', 'desc': 'Bir önceki hafta ürünün fiilen satıldığı (bulunduğu) aktif lokasyon sayısı (Dağılım derinliği).'},
    'stock_cover_pressure': {'name': 'Stok Karşılama Oranı (WOS)', 'desc': 'Mevcut stoğun, son 4 haftalık ortalama satış hızına bölünmesiyle elde edilen "eldeki stok kaç hafta yeter" (Weeks of Supply) metriği.'},
    'is_stockout_risk_lag1': {'name': 'Yok Satma (Stock-out) Bayrağı', 'desc': 'Stok miktarının kritik seviyenin altına düşüp düşmediğini gösteren 1/0 ikili sinyali.'},

    # 3. Fourier & İndeks Tabanlı Mevsimsellik Değişkenleri
    'Week': {'name': 'Yılın Haftası (1-52)', 'desc': 'İlgili tarihin yıl içindeki hafta numarası.'},
    'seasonal_index_ly': {'name': 'Geçen Yıl Mevsimsellik İndeksi', 'desc': 'Geçen yılın aynı haftasındaki satışın, geçen yılın genel ortalamasına oranı.'},
    'seasonal_index_ly_smooth': {'name': 'Düzleştirilmiş Mevsimsellik İndeksi', 'desc': 'Takvimsel kaymaları (outlier) törpülemek için yumuşatılmış geçmiş yıl mevsimsellik katsayısı.'},
    'sin_fourier_1': {'name': 'Fourier Sinüs (Ana Döngü)', 'desc': '52 haftalık tam yıl döngüsünü modelleyen sinüs dalgası.'},
    'cos_fourier_1': {'name': 'Fourier Kosinüs (Ana Döngü)', 'desc': '52 haftalık tam yıl döngüsünü modelleyen kosinüs dalgası.'},
    'sin_fourier_2': {'name': 'Fourier Sinüs (Yarıyıl)', 'desc': '26 haftalık (İlkbahar/Yaz ve Sonbahar/Kış) sezon geçişlerini modelleyen frekans.'},
    'cos_fourier_2': {'name': 'Fourier Kosinüs (Yarıyıl)', 'desc': '26 haftalık sezon geçişlerini modelleyen frekans.'},

    # 4. Dinamik Tatil, Özel Gün ve Etkinlik Değişkenleri
    'ramazan_rampasi': {'name': 'Ramazan Alışveriş Rampası', 'desc': 'Bayramdan önceki 4 haftalık dönem — Ramazan boyunca kademeli artan alışveriş hareketliliğini işaretler (oruç ayına yayılan talep artışı).'},
    'ramazan_pik': {'name': 'Ramazan Bayramı Öncesi Pik', 'desc': 'Bayramdan hemen önceki 2 haftalık en yoğun alışveriş dönemini işaretler.'},
    'ramazan_bayrami': {'name': 'Ramazan Bayramı Haftası', 'desc': 'Ramazan bayramının denk geldiği takvim haftası (mağazalar kapalı/seyahat nedeniyle satış genelde düşer).'},
    'ramazan_etkisi': {'name': 'Ramazan Etkisi (Genel)', 'desc': 'Rampa + pik dönemlerini birlikte barındıran kapsayıcı değişken.'},

    'kurban_rampasi': {'name': 'Kurban Bayramı Öncesi Rampa', 'desc': 'Bayramdan önceki 2 haftalık alışveriş hareketlenme dönemini işaretler.'},
    'kurban_bayrami': {'name': 'Kurban Bayramı Haftası', 'desc': 'Kurban bayramının denk geldiği takvim haftası.'},
    'kurban_etkisi': {'name': 'Kurban Etkisi (Genel)', 'desc': 'Rampa + bayram haftasını birlikte barındıran kapsayıcı değişken.'},

    'event_kadin_anneler': {'name': 'Anneler Günü x Kadın Segmenti', 'desc': 'Anneler günü haftasının özellikle "Kadın" kategorisindeki satışlara etkileşimi.'},
    'anneler_gunu': {'name': 'Anneler Günü Etkisi', 'desc': 'Mayıs ayındaki hediyeleşme haftası etkisi.'},
    
    'event_erkek_babalar': {'name': 'Babalar Günü x Erkek Segmenti', 'desc': 'Babalar günü haftasının "Erkek" kategorisindeki etkileşimi.'},
    'babalar_gunu': {'name': 'Babalar Günü Etkisi', 'desc': 'Haziran ayındaki hediyeleşme haftası etkisi.'},
    
    'event_okul_sneaker': {'name': 'Okula Dönüş x Seçili Segmentler', 'desc': 'Ağustos-Eylül okula dönüş (BTS) döneminin belirli ürün kategorilerine spesifik etkisi.'},
    'okula_donus': {'name': 'Okula Dönüş (BTS)', 'desc': 'Okulların açılış dönemindeki genel talep artışı.'},
    
    'kasim_indirimleri': {'name': 'Kasım Kampanyaları (Black Friday)', 'desc': 'Kasım sonu agresif indirim haftaları etkisi.'},
    'yilbasi_etkisi': {'name': 'Yılbaşı Hediyeleşme Dönemi', 'desc': 'Aralık ayı sonu hediye alışverişi artışı.'},
    'yeni_yil_dususu': {'name': 'Ocak İlk 3 Hafta Düşüşü', 'desc': 'Yılbaşı sonrası (yılın 1., 2. ve 3. haftaları) yaşanan doğal talep geri çekilmesi.'},

    # 5. Hiyerarşik ve Kategorik Değişkenler
    'Gender': {'name': 'Cinsiyet Kırılımı', 'desc': 'Ürünün hitap ettiği cinsiyet grubu (Kadın/Erkek/Unisex).'},
    'GroupDesc': {'name': 'Ana Departman', 'desc': 'Ürünün ait olduğu üst departman hiyerarşisi.'},
    'StockGroupDesc': {'name': 'Ürün Kategorisi', 'desc': 'Ürünün alt kırılım kategorisi.'},

    # 6. Dış Model Tahminleri (Stacking)
    'prophet_tahmin': {'name': 'Prophet Tahmini', 'desc': 'Trend + mevsimsellik + tatil ayrıştırması yapan Prophet modelinin (sızıntısız walk-forward ile üretilmiş) haftalık satış tahmini — ana modele ek bir görüş olarak sunulur.'},
    'patchtst_tahmin': {'name': 'PatchTST Tahmini', 'desc': 'Transformer tabanlı, 8 seriyi tek havuzda (pooled) eğiten PatchTST modelinin (sızıntısız walk-forward ile üretilmiş) haftalık satış tahmini — ana modele ek bir görüş olarak sunulur.'},
}


def kisa_model_adi(model_etiketi: str) -> str:
    """Sidebar'daki emojili model etiketini metrik tablosunda aranacak kısa ada çevirir."""
    for anahtar in ("Stacking", "Ensemble", "LightGBM", "XGBoost"):
        if anahtar in model_etiketi:
            return anahtar
    return "CatBoost"

# Veri ve Model Yükleme (Önbellekli)
@st.cache_data
def load_all_data():
    df_clean = load_and_preprocess_data()
    df_feat = build_features(df_clean)
    return df_clean, df_feat

@st.cache_resource
def load_models(df_feat):
    """
    Once models/ klasorundeki optimize (Optuna) modelleri yuklemeyi dener.
    Bulamazsa eski yontemle (uygulama icinde egitim) devam eder.
    Donen: models, df_test_results, df_metrics_table, ensemble_weights, kaynak_etiketi
    """
    try:
        models, config = load_bundle()
        # Eger meta_model yoksa Stacking arayuzde calismaz ama Ensemble calismaya devam eder.
        w = config["final_weights"]                       # {"LightGBM": .., "XGBoost": .., "CatBoost": ..}
        weights = (w["LightGBM"], w["XGBoost"], w["CatBoost"])
        weight_note = config.get(
            "ensemble_weight_method",
            "Kaydedilmiş ağırlıklar (seçim yöntemi bu modelde belirtilmemiş)."
        )
        df_test, df_metrics = run_recursive_backtest(models, df_feat, test_weeks=12, weights=weights)
        return models, df_test, df_metrics, weights, "Optimize modeller (models/)", weight_note
    except (FileNotFoundError, KeyError):
        models, df_test, df_metrics = train_and_validate_ensemble(df_feat, test_weeks=12)
        return (models, df_test, df_metrics, (1/3, 1/3, 1/3), "Uygulama içi eğitim (Varsayılan Ağırlık)",
                "Eşit ağırlık (1/3 her biri) — kaydedilmiş model bulunamadığı için varsayılan.")

with st.spinner("Model ve veri boru hattı yükleniyor..."):
    df_clean, df_features = load_all_data()
    models, df_test_results, df_metrics_table, ens_weights, model_kaynak, ens_weight_note = load_models(df_features)

st.sidebar.caption(f"⚙️ Model kaynağı: {model_kaynak}")

# Yan Panel (Sidebar)
st.sidebar.title("Kırılım & Model Seçimi")

selected_model = st.sidebar.selectbox(
    "Tahmin Motoru / Model Mimarisi",
    ["🏆 Ensemble (Ağırlıklı)", "🧬 Stacking Ensemble (Meta-Model)", "⚡ LightGBM", "🌲 XGBoost", "🐱 CatBoost"],
    index=0
)

selected_gender = st.sidebar.selectbox("Cinsiyet", ["Tümü"] + list(df_clean['Gender'].unique()))

if selected_gender != "Tümü":
    available_cats = df_clean[df_clean['Gender'] == selected_gender]['StockGroupDesc'].unique()
else:
    available_cats = df_clean['StockGroupDesc'].unique()

selected_cat = st.sidebar.selectbox("Ürün Grubu", ["Tümü"] + list(available_cats))

st.sidebar.markdown("---")
st.sidebar.subheader("Tedarik Planlama Ayarları")
lead_time = st.sidebar.slider("Tedarik Süresi (Lead Time - Hafta)", min_value=1, max_value=12, value=4)
service_level = st.sidebar.selectbox(
    "Hizmet Seviyesi (Servis Oranı)",
    [("90% Güvenlik", 1.28), ("95% Güvenlik", 1.65), ("99% Güvenlik", 2.33)],
    index=1,
    format_func=lambda secim: secim[0]   # yoksa listede ham tuple ("90% Güvenlik", 1.28) görünür
)
horizon = st.sidebar.slider("Tahmin Ufku (Hafta)", min_value=max(4, lead_time + 1), max_value=max(16, lead_time + 1), value=max(8, lead_time + 1))

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Finansal Birim İktisat")
cost_per_pair = st.sidebar.number_input("Birim Alış Maliyeti (TL)", min_value=100.0, max_value=10000.0, value=850.0, step=50.0)
price_per_pair = st.sidebar.number_input("Birim Satış Fiyatı (TL)", min_value=100.0, max_value=20000.0, value=1650.0, step=50.0)

# Filtrelenmiş Veri Kümeleri
df_filtered_clean = df_clean.copy()
if selected_gender != "Tümü":
    df_filtered_clean = df_filtered_clean[df_filtered_clean['Gender'] == selected_gender]
if selected_cat != "Tümü":
    df_filtered_clean = df_filtered_clean[df_filtered_clean['StockGroupDesc'] == selected_cat]

df_test_filtered = df_test_results.copy()
if selected_gender != "Tümü":
    df_test_filtered = df_test_filtered[df_test_filtered['Gender'] == selected_gender]
if selected_cat != "Tümü":
    df_test_filtered = df_test_filtered[df_test_filtered['StockGroupDesc'] == selected_cat]

# 4 Sekmeli Ana Yapı
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Mevcut Durum & Geçmiş Analizi (2023-2026)",
    "🔮 Gelecek Talep & Sipariş Karar Motoru",
    "🧪 Model Doğrulama & Performans (Backtesting)",
    "🧠 Model Değişkenleri & Önem Tablosu (XAI)"
])

# ==========================================
# SEKME 1: MEVCUT DURUM ANALİZİ
# ==========================================
with tab1:
    st.header(f"Satış ve Envanter Analiz Paneli ({selected_gender} / {selected_cat})")

    df_weekly_agg = df_filtered_clean.groupby('Date', as_index=False).agg({
        'SatisSayisi': 'sum',
        'StokSayisi': 'sum',
        'MagazaSayisi': 'max'
    })

    df_weekly_agg['Satis_MA4'] = df_weekly_agg['SatisSayisi'].rolling(4, min_periods=1).mean()

    latest_week = df_weekly_agg.iloc[-1]
    total_sales = df_weekly_agg['SatisSayisi'].sum()
    current_stock = latest_week['StokSayisi']
    avg_weekly_sales = df_weekly_agg['SatisSayisi'].tail(12).mean()

    daily_sales_rate = avg_weekly_sales / 7.0
    coverage_days = current_stock / (daily_sales_rate + 1e-6)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Toplam Satış (2023-2026)", f"{int(total_sales):,} Çift")
    c2.metric("Son 12 Hafta Ort. Satış", f"{int(avg_weekly_sales):,} Çift/Hft")
    c3.metric("Merkez Depo Envanteri", f"{int(current_stock):,} Çift")

    if coverage_days < 1.0:
        c4.metric("Merkez Depo Tampon Süresi", f"{coverage_days * 24:.0f} Saat", help="Merkez Dağıtım Merkezi (DC) transit akış hızı.")
    else:
        c4.metric("Merkez Depo Tampon Süresi", f"{coverage_days:.1f} Gün", help="Merkez Dağıtım Merkezi (DC) transit akış hızı.")

    st.caption("ℹ️ **Not:** Envanter metriği yalnızca Merkez Dağıtım Merkezi (DC) tampon stoğunu yansıtır; mağaza içi raf stokları dahil değildir.")
    st.markdown("---")

    st.subheader("Haftalık Satış Hacmi & Trend Eğrisi")
    fig_sales = go.Figure()
    fig_sales.add_trace(go.Scatter(
        x=df_weekly_agg['Date'],
        y=df_weekly_agg['SatisSayisi'],
        mode='lines',
        name='Haftalık Net Satış',
        line=dict(color='#d32f2f', width=2)
    ))
    fig_sales.add_trace(go.Scatter(
        x=df_weekly_agg['Date'],
        y=df_weekly_agg['Satis_MA4'],
        mode='lines',
        name='4 Haftalık Trend (MA-4)',
        line=dict(color='#ff9800', width=2, dash='dot')
    ))
    fig_sales.update_layout(hovermode="x unified", height=350, xaxis_title="Tarih", yaxis_title="Haftalık Satış (Çift)")
    st.plotly_chart(fig_sales)

    st.subheader("Merkez Depo Envanter Seviyesi Takibi")
    fig_stock = go.Figure()
    fig_stock.add_trace(go.Scatter(
        x=df_weekly_agg['Date'],
        y=df_weekly_agg['StokSayisi'],
        mode='lines+markers',
        name='Kapanış Depo Stoku',
        line=dict(color='#2e7d32', width=2),
        fill='tozeroy',
        fillcolor='rgba(46, 125, 50, 0.1)'
    ))
    fig_stock.update_layout(hovermode="x unified", height=280, xaxis_title="Tarih", yaxis_title="Depo Stoku (Çift)")
    st.plotly_chart(fig_stock)

    st.markdown("---")

    col_kpi1, col_kpi2 = st.columns(2)
    with col_kpi1:
        st.subheader("Kategori Bazında Satış Payı Dağılımı")
        cat_share = df_filtered_clean.groupby('StockGroupDesc')['SatisSayisi'].sum().reset_index()
        fig_pie = px.pie(cat_share, values='SatisSayisi', names='StockGroupDesc', hole=0.4,
                         color_discrete_sequence=px.colors.qualitative.Safe)
        st.plotly_chart(fig_pie)

    with col_kpi2:
        st.subheader("Yıllık Satış & 2026 Yıl Sonu Beklentisi (FY Landing)")
        hist_years = df_filtered_clean[df_filtered_clean['Year'] < 2026].groupby('Year')['SatisSayisi'].sum().reset_index()
        actual_2026 = float(df_filtered_clean[df_filtered_clean['Year'] == 2026]['SatisSayisi'].sum())

        latest_week_2026 = int(df_filtered_clean[df_filtered_clean['Year'] == 2026]['Week'].max())
        remaining_weeks_2026 = max(0, 52 - latest_week_2026)

        if remaining_weeks_2026 > 0:
            pred_remaining_2026 = float(generate_recursive_forecast(
                models_dict=models,
                full_features_df=df_features,
                gender=selected_gender,
                category=selected_cat,
                forecast_horizon=remaining_weeks_2026,
                selected_model_name=selected_model,
                weights=ens_weights
            )[0].sum())
        else:
            pred_remaining_2026 = 0.0

        total_2026_expected = actual_2026 + pred_remaining_2026

        fig_landing = go.Figure()
        years_all = [str(y) for y in hist_years['Year']] + ['2026']
        actuals_all = list(hist_years['SatisSayisi']) + [actual_2026]

        fig_landing.add_trace(go.Bar(
            x=years_all,
            y=actuals_all,
            name='Gerçekleşen Fiili Satış',
            marker_color='#1976d2',
            text=[f"{v / 1e6:.2f}M" for v in actuals_all],
            textposition='auto'
        ))

        if pred_remaining_2026 > 0:
            fig_landing.add_trace(go.Bar(
                x=['2026'],
                y=[pred_remaining_2026],
                name=f'{selected_model} 2026 Tamamlama',
                marker=dict(color='rgba(255, 152, 0, 0.65)', line=dict(color='#e65100', width=1.5)),
                text=[f"+{pred_remaining_2026 / 1e6:.2f}M"],
                textposition='inside'
            ))

        fig_landing.update_layout(
            barmode='stack',
            height=380,
            xaxis_title="Yıl",
            yaxis_title="Toplam Satış Hacmi (Çift)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_landing)
        st.caption(f"📌 **2026 Kapanış Projeksiyonu ({selected_model}):** Gerçekleşen: **{actual_2026:,.0f}** | Yıl Sonu Beklenen: **{total_2026_expected:,.0f} Çift**")

# ==========================================
# SEKME 2: TAHMİN & TEDARİK KARAR MOTORU (OPERASYONEL & FİNANSAL ENTEGRASYON)
# ==========================================
with tab2:
    st.header(f"Gelecek Dönem Satın Alma & Envanter Karar Raporu ({selected_gender} / {selected_cat})")

    clean_m_name = kisa_model_adi(selected_model)
    active_metrics = df_metrics_table[df_metrics_table['Model'].str.contains(clean_m_name, na=False)].iloc[0]

    st.markdown(
        f"""
        <div style="background-color: #f0f4f8; padding: 12px 18px; border-radius: 8px; border-left: 5px solid #1976d2; margin-bottom: 15px;">
            <span style="font-size: 15px; font-weight: 600; color: #0d47a1;">⚙️ Aktif Karar Motoru: {selected_model}</span>
            <span style="margin-left: 20px; font-size: 14px; color: #333;">
                <b>Test Doğrulama Karnesi:</b> 
                R²: <b>{active_metrics['R² Skoru']}</b> | 
                WAPE: <b>{active_metrics['WAPE (%)']}</b> | 
                MAPE: <b>{active_metrics['MAPE (%)']}</b> | 
                RMSE: <b>{active_metrics['RMSE (Çift)']} Çift</b>
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    df_agg_for_stock = df_filtered_clean.groupby('Date', as_index=False).agg({'StokSayisi': 'sum', 'SatisSayisi': 'sum'})
    cur_stock = float(df_agg_for_stock['StokSayisi'].iloc[-1])

    future_preds, future_lower, future_upper = generate_recursive_forecast(
        models_dict=models,
        full_features_df=df_features,
        gender=selected_gender,
        category=selected_cat,
        forecast_horizon=horizon,
        selected_model_name=selected_model,
        weights=ens_weights
    )

    decision = calculate_order_decision(
        forecast_series=future_preds,
        current_stock=cur_stock,
        lead_time_weeks=lead_time,
        z_score=service_level[1],
        lower_series=future_lower,
        upper_series=future_upper
    )

    # 1. Operasyonel Metrikler (Fiziksel Çift)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Mevcut Depo Stoku", f"{int(decision['Mevcut_Stok']):,} Çift")
    k2.metric(f"Tedarik Süresi Talebi ({lead_time} Hft)", f"{int(decision['Gelecek_Talep_Toplami']):,} Çift")
    k3.metric("Güvenlik Stoku", f"{int(decision['Guvenlik_Stoku']):,} Çift")
    k4.metric("Önerilen Net Satın Alma", f"{int(decision['Onerilen_Siparis']):,} Çift", delta=decision['Finansal_Fren'])

    # 2. Finansal Metrikler (Birim İktisat & CFO Kartları)
    order_budget_tl = decision['Onerilen_Siparis'] * cost_per_pair
    expected_revenue_tl = decision['Gelecek_Talep_Toplami'] * price_per_pair
    expected_gross_profit_tl = decision['Gelecek_Talep_Toplami'] * (price_per_pair - cost_per_pair)
    margin_pct = ((price_per_pair - cost_per_pair) / price_per_pair) * 100.0

    st.markdown("##### 💼 Finansal Bütçe & Getiri Projeksiyonu")
    f_kpi1, f_kpi2, f_kpi3, f_kpi4 = st.columns(4)
    f_kpi1.metric("Tahsis Edilecek Satın Alma Bütçesi", f"₺{order_budget_tl:,.0f}", help="Önerilen siparişin toplam finansal maliyeti.")
    f_kpi2.metric("Beklenen Brüt Satış Hacmi", f"₺{expected_revenue_tl:,.0f}", help="Tedarik süresi boyunca oluşacak ciro.")
    f_kpi3.metric("Beklenen Brüt Kâr", f"₺{expected_gross_profit_tl:,.0f}", delta=f"%{margin_pct:.1f} Marj")

    unit_holding_cost_rate = 0.0075  # Haftalık %0.75 ortalama depo ve sermaye maliyeti
    holding_cost_tl = decision['Hedef_Stok'] * cost_per_pair * unit_holding_cost_rate * lead_time
    f_kpi4.metric("Dönemsel Stok Tutma Maliyeti", f"₺{holding_cost_tl:,.0f}", help="Depoda tutulan hedeflenen stoğun sermaye ve operasyonel maliyeti.")

    # 3. ŞEFFAF METODOLOJİ VE FORMÜL KARTI (MENTORÜN İSTEDİĞİ MATEMATİKSEL İSPAT)
    with st.expander("📐 Tedarik Zinciri & Finansal Hesaplama Metodolojisi (Formül Kartı)", expanded=False):
        st.markdown(r"""
        Bu panelde üretilen operasyonel sipariş adetleri ve sermaye tahsis bütçesi, uluslararası tedarik zinciri (APICS / SCOR) ve perakende birim iktisat standartlarına göre dinamik olarak hesaplanmaktadır:

        ---
        ##### 1. Operasyonel Tedarik ve Güvenlik Stoku Formülleri
        * **Tedarik Süresi Talebi ($D_{LT}$):**
          $$D_{LT} = \sum_{t=1}^{L} \hat{y}_t$$
          *Sipariş verildiği andan depoya teslim edilene kadar geçecek olan $L$ haftalık tedarik süresindeki model tahminlerinin toplamıdır.*

        * **Güvenlik Stoku ($SS$ - Safety Stock):**
          $$SS = Z \times \sigma_D \times \sqrt{L}$$
          *Burada $Z$: Seçilen hizmet seviyesi standart normal dağılım sapması (örn. %95 servis için $1.65$), $\sigma_D$: Tahmin dönemindeki talep oynaklığı (standart sapması), $L$: Tedarik süresidir (hafta).*

        * **Hedeflenen Güvenli Stok Seviyesi ($S$ - Order-Up-To Level):**
          $$S = D_{LT} + SS$$
          *Yok satma riskini engellemek için sistemin döngü sonunda depoda bulunmasını şart koştuğu toplam envanter seviyesidir.*

        * **Önerilen Net Satın Alma ($Q$):**
          $$Q = \max(0, S - I_{mevcut})$$
          *Depodaki mevcut hazır envanter ($I_{mevcut}$) hedeflenen stoktan düşülür. Eğer depodaki stok yeterliyse sipariş $0$ verilir (Finansal Fren).*

        ---
        ##### 2. Finansal Bütçe ve Birim İktisat Formülleri
        * **Tahsis Edilecek Satın Alma Bütçesi (TL):**
          $$\text{Bütçe} = Q \times \text{Birim Alış Maliyeti (COGS)}$$
          *CFO ve Satın Alma departmanının onayına gidecek olan net nakit çıkış gereksinimidir.*

        * **Beklenen Brüt Satış Hacmi (Ciro - TL):**
          $$\text{Ciro} = D_{LT} \times \text{Birim Satış Fiyatı (ASP)}$$

        * **Beklenen Brüt Kâr & Marj Oranı:**
          $$\text{Brüt Kâr} = D_{LT} \times (\text{ASP} - \text{COGS}) \quad \Big( \text{Brüt Marj \%} = \frac{\text{ASP} - \text{COGS}}{\text{ASP}} \times 100 \Big)$$

        * **Dönemsel Stok Tutma Maliyeti (Holding Cost - TL):**
          $$\text{Holding Cost} = S \times \text{COGS} \times h \times L$$
          *Depoda tutulan malın sigorta, depo kiralama, operasyonel işçilik ve alternatif sermaye getiri maliyetidir ($h \approx \%0.75/\text{hafta}$).*
        """)

    st.markdown("---")

    f_left, f_right = st.columns([2, 1])

    with f_left:
        st.subheader(f"Gelecek {horizon} Hafta Talep Projeksiyonu & Geçmiş Doğrulama")

        hist_tail = df_agg_for_stock.tail(24)
        last_date = hist_tail['Date'].iloc[-1]

        model_col_map = {
            "🏆 Ensemble (Ağırlıklı)": "Tahmin_Ensemble",
            "🧬 Stacking Ensemble (Meta-Model)": "pred_stacking",
            "⚡ LightGBM": "pred_lgb",
            "🌲 XGBoost": "pred_xgb",
            "🐱 CatBoost": "pred_cat"
        }
        chosen_test_col = model_col_map.get(selected_model, "Tahmin_Ensemble")

        df_test_agg = df_test_filtered.groupby('Date', as_index=False).agg({
            'SatisSayisi': 'sum',
            chosen_test_col: 'sum'
        })

        future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=horizon, freq='W-MON')

        fig_f = go.Figure()

        fig_f.add_trace(go.Scatter(
            x=hist_tail['Date'],
            y=hist_tail['SatisSayisi'],
            mode='lines+markers',
            name='Fiili Satış (Ground Truth)',
            line=dict(color='#0d47a1', width=2.5),
            marker=dict(size=5)
        ))

        fig_f.add_trace(go.Scatter(
            x=df_test_agg['Date'],
            y=df_test_agg[chosen_test_col],
            mode='lines+markers',
            name=f'{clean_m_name} Test Doğrulaması (Backtest)',
            line=dict(color='#2ca02c', width=2, dash='dot'),
            marker=dict(size=4)
        ))

        fig_f.add_trace(go.Scatter(
            x=[last_date] + list(future_dates),
            y=[hist_tail['SatisSayisi'].iloc[-1]] + list(future_preds),
            mode='lines+markers',
            name=f'{clean_m_name} Gelecek {horizon} Hafta Projeksiyonu',
            line=dict(color='#ff6f00', width=3, dash='dash'),
            marker=dict(size=6)
        ))

        # Güven Aralığı Bandı (Quantile modeller varsa)
        if future_lower is not None and future_upper is not None:
            fig_f.add_trace(go.Scatter(
                x=list(future_dates), y=list(future_upper),
                mode='lines', line=dict(width=0),
                showlegend=False, hoverinfo='skip'
            ))
            fig_f.add_trace(go.Scatter(
                x=list(future_dates), y=list(future_lower),
                mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor='rgba(255, 111, 0, 0.15)',
                name='%80 Güven Aralığı (P10-P90)'
            ))

        fig_f.add_vline(x=last_date, line_width=1.5, line_dash="dash", line_color="gray")

        fig_f.update_layout(
            hovermode="x unified",
            height=400,
            xaxis_title="Hafta / Tarih",
            yaxis_title="Haftalık Satış (Çift)",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_f)
        st.caption(f"📌 **Zaman Çizelgesi:** Mavi çizgi geçmiş fiili satışı, yeşil noktalı çizgi son 12 haftalık {selected_model} test performansını, turuncu kesikli çizgi ise geleceğe yönelik sipariş projeksiyonunu gösterir. Turuncu gölgeli alan ise modelin %80 güven aralığını (en az - en fazla beklenti) temsil eder.")

    with f_right:
        st.subheader("Tedarik Zinciri Karar Özeti")
        st.write(f"**Hedeflenen Güvenli Stok:** {int(decision['Hedef_Stok']):,} Çift")
        st.write(f"**Merkez Depo Tampon Süresi:** {decision['Stok_Karsilama_Haftasi_WOS']} Hafta")

        if decision['Finansal_Fren'].startswith("AKTIF"):
            st.error("🛑 FİNANSAL FREN AKTİF: Depodaki mal hedef seviyenin üzerinde. Sipariş açmayın.")
        else:
            st.success("📦 SİPARİŞ ONAYI: Güvenlik seviyesinin altına düşüş bekleniyor. Sipariş açılabilir.")

        # ERP İçin Sipariş Tablosu
        df_f_table = pd.DataFrame({
            'Tarih': [d.strftime('%Y-%m-%d') for d in future_dates],
            'Haftalık_Tahmin_Cift': [int(p) for p in future_preds],
            'Tahmini_Ciro_TL': [int(p * price_per_pair) for p in future_preds],
            'Maliyet_TL': [int(p * cost_per_pair) for p in future_preds]
        })
        st.dataframe(df_f_table, width='stretch', height=180)

        # ERP / Excel Dışa Aktarma Butonu
        csv_data = df_f_table.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Satın Alma Emrini İndir (ERP / Excel)",
            data=csv_data,
            file_name=f"Satin_Alma_Emri_{selected_gender}_{selected_cat}.csv",
            mime="text/csv",
            help="Bu tabloyu doğrudan SAP/Nebim sistemine yüklemek için CSV olarak indirebilirsiniz."
        )

# ==========================================
# SEKME 3: MODEL DOĞRULAMA & TEST PERFORMANSI (BACKTESTING)
# ==========================================
with tab3:
    st.header(f"🧪 Test Seti Doğrulama ve Model Kıyaslama Paneli ({selected_gender} / {selected_cat})")
    st.markdown("Modellerin eğitim sırasında **hiç görmediği son 12 haftalık test kümesindeki** performans karnesi. "
                "Bu **tek bir pencere**dir; dönemler arası gerçek performans için aşağıdaki cross-validation bölümüne bakın.")

    df_test_weekly = df_test_filtered.groupby('Date', as_index=False).agg({
        'SatisSayisi': 'sum',
        'pred_lgb': 'sum',
        'pred_xgb': 'sum',
        'pred_cat': 'sum',
        'Tahmin_Ensemble': 'sum',
        'pred_stacking': 'sum'
    })

    clean_m_name = kisa_model_adi(selected_model)
    selected_row = df_metrics_table[df_metrics_table['Model'].str.contains(clean_m_name, na=False)].iloc[0]

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric(f"{clean_m_name} R² Skoru", f"{selected_row['R² Skoru']}", help="1.0'a ne kadar yakınsa varyans açıklama başarısı o kadar yüksektir.")
    m_col2.metric(f"{clean_m_name} MAPE", f"{selected_row['MAPE (%)']}", help="Ortalama Mutlak Yüzde Hata")
    m_col3.metric(f"{clean_m_name} WAPE", f"{selected_row['WAPE (%)']}", help="Perakende standardı hacim ağırlıklı mutlak hata payı.")
    m_col4.metric(f"{clean_m_name} RMSE", f"{selected_row['RMSE (Çift)']}", help="Büyük hataları cezalandıran kök ortalama kare hata.")

    # NOT (2026-09-17 bugfix): buradaki direct-WAPE eskiden elle yazilmis
    # sabit bir sayiydi (%22,65) ve ensemble_config.json guncellendikce
    # sessizce bayatlamisti -- artik proje ilkesine uygun sekilde (bkz.
    # asagidaki stress-test/CV bolumleri) dosyadan canli okunuyor.
    _direct_wape_str = "bilinmiyor"
    _config_path = Path(__file__).resolve().parent / "models" / "ensemble_config.json"
    if _config_path.exists():
        with open(_config_path, encoding="utf-8") as f:
            _direct_metrics = json.load(f).get("test_metrics_final", {})
        if "WAPE" in _direct_metrics:
            _direct_wape_str = f"{_direct_metrics['WAPE']:.2f}".replace(".", ",")

    st.caption(
        "📐 **Bu karne nasıl ölçüldü:** son 12 hafta **özyinelemeli (recursive)** tahmin edilir — "
        "model kendi tahminini bir sonraki haftanın lag'i olarak kullanır, gerçek satışı görmez. "
        f"`models/ensemble_config.json` içindeki **%{_direct_wape_str}** ise farklı bir ölçümdür: "
        "%15'lik TEST diliminde **direct (tek adım)** tahmin, yani her hafta için gerçek geçmiş "
        "lag'ler verilir. Direct her zaman daha kolaydır; buradaki (daha yüksek) sayı gerçek "
        "kullanım senaryosuna daha yakındır. İkisini birbirinin yerine kullanmayın."
    )

    # --- Rolling-origin cross-validation: tek pencere yanilticidir ---
    cv_path = Path(__file__).resolve().parent / "models" / "cv_results.json"
    if cv_path.exists():
        with open(cv_path, encoding="utf-8") as f:
            _cv = json.load(f)
        st.markdown("---")
        st.subheader("🔁 Rolling-Origin Cross-Validation (dönemsel tutarlılık)")
        st.markdown(
            f"Yukarıdaki karne **tek bir** 12 haftalık pencereden gelir ve o pencere kolay bir "
            f"döneme denk gelebilir. Aşağıda model, test penceresi zamanda kaydırılarak "
            f"**{_cv['n_folds']} farklı dönemde** bağımsız olarak sınanmıştır."
        )
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Ortalama WAPE (tüm foldlar)", f"%{_cv['WAPE_ortalama']}",
                   help="Modelin dönemler arası GERÇEK ortalama hatası. Tek pencereden daha güvenilir.")
        cc2.metric("Oynaklık (± std)", f"±%{_cv['WAPE_sapma']}")
        cc3.metric("En kötü dönem", f"%{_cv['WAPE_max']}", help="Genelde yıl sonu / Q1 (düşük sezon).")

        _folds = pd.DataFrame(_cv["folds"])[["Model", "test_baslangic", "test_bitis", "WAPE", "MAPE", "RMSE", "Bias"]]
        _folds.columns = ["Fold", "Test başlangıç", "Test bitiş", "WAPE (%)", "MAPE (%)", "RMSE", "Bias (%)"]
        st.dataframe(_folds, width='stretch', hide_index=True)
        st.warning(
            "⚠️ **Değerlendirme notu:** Önceki sürümde 3 feature (`SellThroughRate`, `GelenUrun`, "
            "`StokSayisi`) aynı haftanın satışını içeriyordu (veri sızıntısı) ve WAPE'yi yapay olarak "
            "~%1–3'e düşürüyordu. Sızıntı temizlendikten sonra yukarıdaki değerler modelin **gerçek** "
            "performansıdır. En zayıf dönemler yıl sonu ve yaz (mevsimsel patlama gösteren seriler). "
            "İyileştirme çalışması bu dürüst taban üzerinden yürütülmektedir."
        )

    st.markdown("---")

    col_bench_left, col_bench_right = st.columns([1.1, 0.9])

    with col_bench_left:
        st.subheader("📋 Modellerin Test Seti Performans Karnesi")
        st.dataframe(df_metrics_table, width='stretch', hide_index=True)
        _w = " / ".join(f"%{v*100:.0f}" for v in ens_weights)
        st.caption(
            f"🏆 **Ensemble Ağırlığı:** {_w} (LGB / XGB / CAT). **Seçim yöntemi:** {ens_weight_note}"
        )
        st.caption(
            "ℹ️ **Not:** 2026-09-14'ten beri Stacking meta-modeli ayrı bir regresyon öğrenmiyor, "
            "doğrudan yukarıdaki ağırlığı kullanıyor. Bu yüzden **'Stacking Ensemble' ile "
            "'Ensemble (Ağırlıklı)' şu an matematiksel olarak aynı tahmini üretir** — tablodaki "
            "iki satırın birebir aynı çıkması hata değil, beklenen durumdur."
        )

    with col_bench_right:
        st.subheader("🎯 Tahmin Sapması (Forecast Bias) & Yönü")
        st.info(
            "💡 **Metriklerin İş Anlamı:**\n\n"
            "- **R² Skoru:** Modelin satış dalgalarını ve trend dönüşlerini yakalama yeteneğidir. "
            "8 seri tek havuzda toplandığı için şişmeye meyillidir — tek başına başlık yapılmamalı "
            "(bkz. `docs/model_dogrulama_bulgulari.md`, bölüm 2).\n"
            "- **WAPE:** Perakende standardı, hacim ağırlıklı gerçek hata payı. Bu tür haftalık "
            "kategori tahmininde iyi modeller %10–20 bandındadır; bu proje sızıntı temizlendikten "
            "sonra %20–30 bandında — yani hâlâ iyileştirme alanı var.\n"
            "- **Bias (%):** Modelin gereksiz stok (pozitif) veya yok satma (negatif) eğiliminde "
            "olup olmadığını gösterir."
        )

    st.markdown("---")

    with st.expander("🛡️ Model Dayanıklılık & Stres Testi Analizi (Leakage & Ezber Kontrolü)", expanded=False):
        st.markdown("""
        **Amaç:** Modelin geçmişi körü körüne kopyalayıp kopyalamadığını (Data Leakage / ezber) denetlemek.
        Son 12 haftalık test kümesinde üç senaryo karşılaştırılır:
        - **Naive:** gelecek hafta = geçen yılın aynı haftası (`lag_sales_52`) birebir kopyalanır
        - **Tam ensemble:** kaydedilmiş optimize modeller (`lag_52` dahil)
        - **`lag_52` çıkarılmış:** aynı 3 model, yıllık mevsimsel lag olmadan (varsayılan ayarla) yeniden eğitilir
        """)

        stress_path = Path(__file__).resolve().parent / "models" / "stress_test.json"
        if stress_path.exists():
            with open(stress_path, encoding="utf-8") as f:
                _stress = json.load(f)
            df_stress_test = pd.DataFrame(_stress).rename(columns={
                "Model": "Senaryo", "R2": "R² Skoru", "WAPE": "WAPE (%)",
                "MAPE": "MAPE (%)", "RMSE": "RMSE (Çift)", "Bias": "Bias (%)"
            })
            st.dataframe(df_stress_test, width='stretch', hide_index=True)

            _naive_wape = _stress[0]["WAPE"]
            _ens_wape = _stress[1]["WAPE"]
            _no52_wape = _stress[2]["WAPE"]
            st.success(f"""
            **📌 Yorum:** (sayılar yukarıdaki tablodan canlı okunuyor, elle yazılmadı)
            1. **Geçen yılı kopyalamıyor:** Naive senaryo WAPE %{_naive_wape} verirken model %{_ens_wape} — model gerçek örüntü öğreniyor, ama iş kolay değil.
            2. **Veri sızıntısı düzeltildi:** Önceki sürümde `SellThroughRate` / `GelenUrun` / `StokSayisi` feature'ları aynı haftanın satışını içeriyordu ve modeli yapay olarak %1–2 WAPE'ye indiriyordu. Temizlik sonrası gerçek performans yukarıdaki gibi.
            3. **`lag_52` çıkarınca:** WAPE %{_ens_wape} → %{_no52_wape} — hata değişiyor ama model çökmüyor, yıllık mevsimsellik faydalı ama tek başına belirleyici değil. (3. senaryo varsayılan ayarla, kıyas amaçlı.)
            """)
        else:
            st.info("Stres testi henüz çalıştırılmadı. `python -c \"from src.data_loader import load_and_preprocess_data; from src.features import build_features; from src.tuning import stress_test; stress_test(build_features(load_and_preprocess_data()))\"`")

    st.markdown("---")

    st.subheader("📈 Test Dönemi: Fiili Satış ile Model Tahminlerinin Kıyaslaması (Backtesting)")

    fig_backtest = go.Figure()

    fig_backtest.add_trace(go.Scatter(
        x=df_test_weekly['Date'],
        y=df_test_weekly['SatisSayisi'],
        mode='lines+markers',
        name='Gerçekleşen Fiili Satış (Ground Truth)',
        line=dict(color='#0d47a1', width=3.5),
        marker=dict(size=7)
    ))

    fig_backtest.add_trace(go.Scatter(
        x=df_test_weekly['Date'],
        y=df_test_weekly['Tahmin_Ensemble'],
        mode='lines+markers',
        name='Ağırlıklı Ensemble Tahmini',
        line=dict(color='#e65100', width=2.5, dash='dash'),
        marker=dict(size=5)
    ))

    fig_backtest.add_trace(go.Scatter(
        x=df_test_weekly['Date'],
        y=df_test_weekly['pred_stacking'],
        mode='lines+markers',
        name='Stacking Ensemble (Meta-Model)',
        line=dict(color='#ff9800', width=2.5, dash='dashdot'),
        marker=dict(size=5)
    ))

    fig_backtest.add_trace(go.Scatter(
        x=df_test_weekly['Date'],
        y=df_test_weekly['pred_lgb'],
        mode='lines',
        name='LightGBM Tahmini',
        line=dict(color='#2e7d32', width=1.5, dash='dot')
    ))

    fig_backtest.add_trace(go.Scatter(
        x=df_test_weekly['Date'],
        y=df_test_weekly['pred_xgb'],
        mode='lines',
        name='XGBoost Tahmini',
        line=dict(color='#c2185b', width=1.5, dash='dot')
    ))

    fig_backtest.add_trace(go.Scatter(
        x=df_test_weekly['Date'],
        y=df_test_weekly['pred_cat'],
        mode='lines',
        name='CatBoost Tahmini',
        line=dict(color='#7b1fa2', width=1.5, dash='dot')
    ))

    fig_backtest.update_layout(
        hovermode="x unified",
        height=420,
        xaxis_title="Hafta / Tarih",
        yaxis_title="Satış Adedi (Çift)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_backtest)

# ==========================================
# SEKME 4: MODEL DEĞİŞKENLERİ & ÖNEM TABLOSU
# ==========================================
with tab4:
    st.header("🧠 Açıklanabilir Yapay Zeka (XAI) & Öznitelik Kataloğu")
    st.markdown("Modelin talep tahminlerini üretirken hangi değişkenlere ne kadar ağırlık verdiğini ve bu değişkenlerin iş lügatindeki karşılıklarını inceleyebilirsiniz.")

    df_importance = get_feature_importance_df(models)
    df_importance['İş Kolu Değişken Adı'] = df_importance['Feature'].map(lambda x: FEATURE_METADATA.get(x, {}).get('name', x))
    df_importance['İş Mantığı ve Anlamı'] = df_importance['Feature'].map(lambda x: FEATURE_METADATA.get(x, {}).get('desc', 'Genel model değişkeni.'))
    df_importance['Model Etki Payı (%)'] = df_importance['Importance']

    col_imp_left, col_imp_right = st.columns([3, 2])

    with col_imp_left:
        st.subheader("En Etkili Öznitelikler (Feature Importance)")
        top_15 = df_importance.head(15).sort_values('Model Etki Payı (%)', ascending=True)

        fig_imp = px.bar(
            top_15,
            x='Model Etki Payı (%)',
            y='İş Kolu Değişken Adı',
            orientation='h',
            text='Model Etki Payı (%)',
            color='Model Etki Payı (%)',
            color_continuous_scale='Teal'
        )
        fig_imp.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_imp.update_layout(height=520, xaxis_title="Model Karar Ağırlığı (%)", yaxis_title="")
        st.plotly_chart(fig_imp)

    with col_imp_right:
        st.subheader("Model Karar Dağılımı Özeti")
        st.info(
            "💡 **Nasıl Yorumlanmalı?**\n\n"
            "- **Lag & Momentum:** Ürünün son 1-4 haftalık anlık ivmesi tahminin omurgasını oluşturur.\n"
            "- **Takvim & Kampanyalar:** Bayram ve özel gün bayrakları pik ve dip dalgalarını şekillendirir.\n"
            "- **Fourier Döngüleri:** Yaz/Kış sezon geçişlerindeki genel makro eğriyi belirler."
        )

    st.markdown("---")
    st.subheader("📋 Eksiksiz Öznitelik Kataloğu & Teknik Eşleme Tablosu")

    catalog_display = df_importance[['İş Kolu Değişken Adı', 'Feature', 'Model Etki Payı (%)', 'İş Mantığı ve Anlamı']].rename(
        columns={'Feature': 'Teknik Kod Adı'}
    )
    st.dataframe(catalog_display, width='stretch', height=400)