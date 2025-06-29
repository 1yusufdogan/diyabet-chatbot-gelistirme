 📋 Proje Hakkında

Diyabet Danışmanı Chatbot Projesi, diyabet hastalarına yönelik geliştirilmiş bir yapay zekâ tabanlı sohbet robotudur. 
Bu chatbot; kan şekeri takibi, insülin dozları, beslenme tavsiyeleri ve genel diyabet yönetimi gibi konularda 
kullanıcılardan gelen soruları anlayarak uygun yanıtlar üretir.

Proje, Python dili ve Streamlit framework’ü kullanılarak geliştirilmiştir. 
Intent sınıflandırma ve doğal dil işleme için Hugging Face Transformers kütüphanesi kullanılarak, 
BERT ve T5 tabanlı iki farklı dil modeliyle test edilmiştir. 
Kullanıcılar, arayüz üzerinden BERT ve T5 modellerini seçerek chatbotun verdiği yanıtları deneyimleyebilir.

Başlıca özellikler:
- Türkçe veri seti ile geniş intent sınıflandırma (1000+ örnek cümle)
- BERT ve T5 modelleri ile çift LLM karşılaştırması
- Streamlit arayüzü üzerinden canlı sohbet
- Model performansını test etme ve confusion matrix görselleştirme


 📁 Proje Yapısı

chatbot_diyabet/
│
├── app/
│   └── streamlit_app.py          
│
├── models/
│   ├── huggingface_model_1.py     
│   ├── huggingface_model_2.py    
│
├── data/
│   ├── diabetes_chatbot_dataset_varied.csv  
│   ├── diabetes_chatbot_test.csv          
│   └── diabetes_logo.png          



🎯 Veri İçeriği


  - `Intent` → Kullanıcının niyeti (örneğin: selamlama, vedalaşma, kan şekeri sorgusu, insülin bilgisi).
  - `Example` → İlgili intent’e ait örnek kullanıcı cümlesi.
  - `Response` → Chatbot’un bu intent’e vereceği olası yanıt.

- Toplam Kayıt: 1000+ satır
- Intent Çeşidi: 10–15 farklı niyet türü (selamlama, reddetme, insülin tavsiyesi, diyet tavsiyesi, genel bilgi vb.)

 📌 Örnek Satır

| Intent      | Example                             | Response                                |
|-------------|-------------------------------------|-----------------------------------------|
| Greeting    | Merhaba!                            | Merhaba, size nasıl yardımcı olabilirim? |
| Goodbye     | Hoşça kal.                          | Görüşmek üzere, sağlıklı günler dilerim. |
| GlucoseInfo | Kan şekerim yüksekse ne yapmalıyım? | Kan şekeriniz yüksekse öncelikle doktorunuza danışın. |


 📈 İstatistik Dağılımı

| Intent Türü          | Örnek Sayısı |
|----------------------|---------------|
| Selamlama            | 100           |
| Vedalaşma            | 100           |
| Kan Şekeri Sorgusu   | 150           |
| İnsülin Bilgisi      | 150           |
| Diyet Tavsiyesi      | 150           |
| Reddetme/Onaylama    | 100           |
| Genel Sağlık Sorusu  | 100           |
| Diğer                | 150           |

Test Veri Seti

Model performansını ölçmek için, eğitim veri setinden ayrı `diabetes_chatbot_test.csv` dosyası hazırlanmıştır.
Bu test veri seti:
- Rastgele seçilmiş ~200 örnek içerir.
- Modelin Precision, Recall, F1 Score ve Confusion Matrix metriklerinin hesaplanmasında kullanılır.

 ⚙️ Üretim Süreci

Veri seti, n8n veya yapay zekâ destekli veri üretim teknikleri kullanılarak zenginleştirilmiştir.
Böylece chatbot’un gerçek senaryolarda daha doğru sınıflandırma yapması sağlanır.



🤖 Model Eğitimi ve Değerlendirme Süreci

 Model Seçimi
Bu projede intent sınıflandırması için iki farklı dil modeli kullanıldı:

BERT tabanlı sınıflandırıcı (huggingface_model_1.py)

T5 tabanlı sınıflandırıcı (huggingface_model_2.py)

Her iki model de Hugging Face Transformers kütüphanesinden alınan hazır ön-eğitimli (pre-trained) modellerdir.

Amaç: Aynı veri seti üzerinde farklı LLM mimarilerini denemek ve sonuçları karşılaştırmak.

 Test Verisi ve Ayrımı
Eğitim ve değerlendirme işlemleri için veri ayrımı:

diabetes_chatbot_dataset_varied.csv → Eğitim için kullanıldı.

diabetes_chatbot_test.csv → Test performansını objektif değerlendirmek için kullanıldı.

Eğitim ve test örneklerinin karışmaması için veri seti manuel olarak ayrıldı.

 Değerlendirme Metrikleri
Her iki model de aynı test veri setiyle değerlendirildi.

| Model | Precision | Recall | F1 Score |
| ----- | --------- | ------ | -------- |
| BERT  | 0.90      | 0.88   | 0.89     |
| T5    | 0.92      | 0.89   | 0.91     |
        
