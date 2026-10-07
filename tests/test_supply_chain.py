"""Envanter/sipariş karar motorunun (src/supply_chain.py) sağlamlık kontrolleri.

Bu dosya 2026-09-14 ikinci denetiminde eklendi: `calculate_order_decision` projenin
asıl iş çıktısını (kaç çift sipariş verilecek) üretiyor ama hiç testi yoktu.
Model gerektirmeyen, saf fonksiyon olduğu için test etmesi ucuz.
"""
import numpy as np

from src.supply_chain import calculate_order_decision


def test_stok_yetersizse_siparis_onerilir():
    """Eldeki stok, tedarik süresi talebinin çok altındaysa sipariş > 0 ve fren PASİF olmalı."""
    tahmin = np.array([100.0] * 8)
    karar = calculate_order_decision(tahmin, current_stock=50.0, lead_time_weeks=4)
    assert karar["Onerilen_Siparis"] > 0
    assert karar["Finansal_Fren"].startswith("PASIF")


def test_stok_fazlaysa_finansal_fren_devreye_girer():
    """Eldeki stok hedefin üzerindeyse sipariş 0'a kırpılmalı ve fren AKTİF olmalı."""
    tahmin = np.array([100.0] * 8)
    karar = calculate_order_decision(tahmin, current_stock=100_000.0, lead_time_weeks=4)
    assert karar["Onerilen_Siparis"] == 0
    assert karar["Finansal_Fren"].startswith("AKTIF")


def test_hedef_stok_talep_arti_guvenlik_stogu():
    """Hedef stok = tedarik süresi talebi + güvenlik stoğu (formül kartındaki S = D_LT + SS)."""
    tahmin = np.array([100.0, 120.0, 80.0, 140.0, 90.0, 110.0])
    karar = calculate_order_decision(tahmin, current_stock=0.0, lead_time_weeks=4)
    beklenen = karar["Gelecek_Talep_Toplami"] + karar["Guvenlik_Stoku"]
    # ara değerler round(…, 0) ile yuvarlandığı için 1 çiftlik tolerans bırakıyoruz
    assert abs(karar["Hedef_Stok"] - beklenen) <= 1


def test_yuksek_hizmet_seviyesi_daha_cok_guvenlik_stogu():
    """z_score büyüdükçe (90% -> 99% servis) güvenlik stoğu artmalı."""
    tahmin = np.array([100.0, 200.0, 50.0, 300.0, 120.0])
    dusuk = calculate_order_decision(tahmin, current_stock=0.0, z_score=1.28)
    yuksek = calculate_order_decision(tahmin, current_stock=0.0, z_score=2.33)
    assert yuksek["Guvenlik_Stoku"] > dusuk["Guvenlik_Stoku"]
