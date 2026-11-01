# Assignment 1 — Google Colab Kurulum Talimatları

## 1. Colab'ı Aç
1. [colab.research.google.com](https://colab.research.google.com) → **New Notebook**
2. Sağ üst: **Runtime → Change Runtime Type → T4 GPU** seç (ücretsiz)

## 2. assignment1.py'yi Yükle
**Files** panelinden (sol ikon) `assignment1.py` dosyasını sürükle-bırak.

## 3. İlk Cell — Kütüphaneleri Kur
```python
!pip install -q transformers datasets accelerate torch
```

## 4. Kodu Çalıştır
```python
exec(open("assignment1.py").read())
```

**VEYA** dosyayı cell cell kopyala-yapıştır.

## 5. Sonuçlar
- Terminal çıktısında tüm tablolar görünür
- `assignment1_results.json` otomatik oluşur → rapor için kullan

## Tahmini Süre (T4 GPU, 0.5B model)
| Yöntem | ~Süre |
|--------|-------|
| Zero-Shot (30 soru) | 5-8 dk |
| Four-Shot (30 soru) | 6-10 dk |
| Self-Consistency 4x (30 soru) | 20-25 dk |
| Reflection/Revision (30 soru) | 10-15 dk |
| Tool Agent (30 soru) | 10-15 dk |
| **Toplam** | **~1-1.5 saat** |

## Önemli Notlar
- `Keep test reference solutions hidden from the model` ✅ — cevaplar sadece `answers_match()` içinde kullanılıyor
- `No paid APIs` ✅ — sadece HuggingFace ücretsiz model
- `Frozen weights` ✅ — `param.requires_grad = False` uygulandı
- `No eval/exec` ✅ — `safe_calc()` saf AST traversal kullanıyor
