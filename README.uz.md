# Reys kechikishini bashorat qilish va xavf tahlili

[English](README.md) · **Oʻzbekcha** · [Русский](README.ru.md)

[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![CatBoost 1.2.8](https://img.shields.io/badge/CatBoost-1.2.8-FFCC00?logo=catboost&logoColor=black)](https://catboost.ai/)
[![Streamlit 1.53](https://img.shields.io/badge/Streamlit-1.53-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](Dockerfile)
[![Model on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Hugging%20Face%20Hub-yellow)](https://huggingface.co/YOUR_HF_LOGIN/flight-delay-catboost)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

AQSH ichki reysining **15 daqiqadan koʻproq kechikib qoʻnish** ehtimolini (FAA boʻyicha kechikish
taʼrifi) baholaydigan toʻliq mashinaviy oʻrganish xizmati. Bashorat uchun faqat rejalashtirish
bosqichida maʼlum boʻlgan maʼlumotlar ishlatiladi: aviakompaniya, uchish va qoʻnish aeroportlari,
sana, uchish soati, masofa va rejadagi parvoz vaqti. Bashorat ustiga ilova *nima uchun* shunday
natija chiqqanini tushuntiradi (har bir reys uchun Shepli qiymatlari) va xavfni FAA xarajat
meʼyorlari asosida **kutilayotgan moliyaviy zararga** aylantiradi.

**🚀 Jonli demo:** https://huggingface.co/spaces/YOUR_HF_LOGIN/flight-delay-risk
*(bepul CPU Space: 48 soat tashrifsiz qolsa uxlab qoladi; birinchi ochilishda model yuklab olinadi va bir daqiqacha vaqt ketadi)*

![Bashorat ekrani](docs/img/02_prediction.png)

| Tushuntirish va iqtisodiyot | Model asoslari |
|:--:|:--:|
| ![SHAP va zarar tafsiloti](docs/img/03_shap_economics.png) | ![Model koʻrsatkichlari](docs/img/04_model_evidence.png) |

---

## Mundarija

1. [Xizmat nima qiladi](#xizmat-nima-qiladi)
2. [Maʼlumotlar](#maʼlumotlar)
3. [Yondashuv](#yondashuv)
4. [Natijalar](#natijalar)
5. [Model artefaktlari](#model-artefaktlari)
6. [Ishga tushirish](#ishga-tushirish)
7. [Loyiha tuzilmasi](#loyiha-tuzilmasi)
8. [Cheklovlar va rejalar](#cheklovlar-va-rejalar)
9. [Hujjatlar va muallif](#hujjatlar-va-muallif)

---

## Xizmat nima qiladi

* **Bashorat** — CatBoost tasniflagichi jiddiy kechikishning xom ehtimolini qaytaradi.
  Biznes qaror chegarasi — **0.37** (toʻliqlik, yaʼni recall uchun sozlangan, qarang:
  [Natijalar](#natijalar)): shu qiymat va undan yuqorida xulosa «kechikish ehtimoli yuqori» boʻladi.
* **Kontekst** — ehtimol maʼlumotlardagi oʻrtacha kechikish ulushi (18,9 %) bilan solishtiriladi,
  shuning uchun foydalanuvchi mavhum raqam oʻrniga «oʻrtachadan 1,8 barobar yuqori» koʻradi.
  *Xavf indeksi* shkalasi ehtimolni chegara atrofida faqat oʻqilishi oson boʻlishi uchun choʻzadi;
  bu kalibrlash emas.
* **Tushuntirish** — CatBoost oʻzi hisoblaydigan Shepli qiymatlari aynan shu reysni kechikish
  tomon yoki undan uzoqqa qaysi omillar surganini koʻrsatadi (aeroport, yoʻnalish, kun, soat,
  aviakompaniya va h.k.).
* **Iqtisodiyot** — kutilayotgan zarar = ehtimol × kechikish hodisasining umumiy qiymati. Hodisa
  qiymati aviakompaniyaning bevosita operatsion xarajatlari (kechikishning har daqiqasiga 110 USD),
  yoʻlovchilar vaqti (FAA Value of Travel Time Savings, yoʻlovchi-soatiga 62,50 USD) va tarmoq boʻylab
  zanjirli kechikishlar uchun 2,2× koeffitsiyentdan iborat.
* **Uchta interfeys tili** — ingliz, oʻzbek (lotin) va rus. Ilova brauzer tilida ochiladi
  (`uz*` → oʻzbekcha, `ru*` → ruscha, qolgan hollarda inglizcha), yon panelda til almashtirgich bor.

![Oʻzbekcha interfeys](docs/img/05_uzbek.png)

## Maʼlumotlar

* **Manba:** Kaggle — [2015 Flight Delays and Cancellations](https://www.kaggle.com/datasets/usdot/flight-delays)
  (AQSH Transport vazirligi, Bureau of Transportation Statistics). 2015-yildagi 5,8 million AQSH ichki
  reysi: 565 MB hajmli `flights.csv` hamda `airlines.csv` va `airports.csv` maʼlumotnomalari.
* **Repozitoriyda:** faqat ikkita kichik maʼlumotnoma va maʼlumotlardan olingan yoʻnalishlar
  jadvali (`models/lookup_data.joblib`). Xom `flights.csv` repozitoriyga **kiritilmagan**; u faqat
  modelni qayta oʻqitish uchun kerak. [Kaggle CLI](https://github.com/Kaggle/kaggle-api) orqali yuklab oling:

  ```bash
  kaggle datasets download -d usdot/flight-delays -p data --unzip
  ```

* **Oʻqitish toʻplami:** dastlabki 2,1 million qator (80 % oʻqitish / 20 % stratifikatsiyalangan test).
  Maqsad oʻzgaruvchi: `ARRIVAL_DELAY > 15` daqiqa; ijobiy sinf ulushi ≈ 18,9 %.
* **Maʼlumot sizib chiqishiga qarshi nazorat:** faqat uchishdan oldin maʼlum boʻlgan ustunlar
  ishlatiladi (`MONTH, DAY, DAY_OF_WEEK, AIRLINE, ORIGIN_AIRPORT, DESTINATION_AIRPORT,
  SCHEDULED_DEPARTURE, SCHEDULED_ARRIVAL, SCHEDULED_TIME, DISTANCE`). Haqiqiy uchish/qoʻnish vaqtlari,
  ruldan yurish vaqti, bort raqamlari va kechikish sabablari chiqarib tashlangan.

## Yondashuv

**Omillarni loyihalash (feature engineering)** — maxsus scikit-learn transformeri
`FlightFeatureEngineer` (`src/feature_engineering.py`) pipelinening birinchi bosqichi, shuning uchun
oʻqitishda ham, bashoratda ham bir xil mantiq qoʻllanadi:

| Omil | Gʻoya |
|---|---|
| `hour_sin`, `hour_cos` | uchish soati aylanadagi nuqta sifatida (23:00 va 00:00 bir-biriga yaqin) |
| `traffic_density` | oʻsha kuni oʻsha soatda uchish aeroportidan koʻtarilgan reyslar soni (uchish-qoʻnish yoʻlagi va terminal yuklamasi) |
| `AIRLINE_FLIGHT_COUNT` | aviakompaniyaning oʻsha kuni shu aeroportdan reyslari soni (ekipaj va samolyot almashinuvi bosimi, «domino effekti») |
| `ROUTE` | uchish–qoʻnish juftligi kategoriya sifatida (tirband yoʻnalishlar) |
| `plan_speed` | masofa / rejadagi vaqt: zich jadval havoda vaqtni qoplashga imkon qoldirmaydi |
| `IS_NIGHT`, `IS_WEEKEND` | 00:00–05:59 oraligʻidagi uchishlar; shanba/yakshanba |

Trafik statistikasi faqat oʻqitish toʻplamida hisoblanadi (`fit`) va bashoratda qidirib olinadi.

**Model** — `sklearn.pipeline.Pipeline` ichidagi `CatBoostClassifier`, GPU da oʻqitilgan
(`src/train_final.py`): 2500 iteratsiya, oʻrganish tezligi 0.03, chuqurlik 7, `l2_leaf_reg=12`,
81/19 sinf nomutanosibligini qoplash uchun `scale_pos_weight=4.2`, Bernoulli boʻyicha 0.8
tanlanma, AUC boʻyicha erta toʻxtatish (100 raund). Kategoriyali omillar CatBoostning tartiblangan
maqsadli statistikasi bilan qayta ishlanadi: `MONTH, DAY, DAY_OF_WEEK, AIRLINE, ORIGIN_AIRPORT,
DESTINATION_AIRPORT, ROUTE, IS_WEEKEND, IS_NIGHT`. Oʻqitish vaqti: RTX 4050 da taxminan 11 daqiqa.

**Chegara** — standart 0.5 chegara kechikishlarning atigi 66 % ini topadi. Aviakompaniya uchun
oʻtkazib yuborilgan kechikish (yolgʻon manfiy) yolgʻon signaldan qimmatroq boʻlgani sababli chegara
saralab chiqildi va **0.37** da qotirildi: toʻliqlik 0.82, aniqlik 0.31.

**Tushuntirish** — `src/analytics.py` CatBoostdan har bir qator uchun SHAP qiymatlarini soʻraydi
(`get_feature_importance(type="ShapValues")`) va eng kuchli yettita hissani koʻrsatadi.

## Natijalar

Test toʻplami: 408 907 reys (20 % ajratilgan qism). Manba: [`reports/metrics/`](reports/metrics/).

| Koʻrsatkich | Qiymat |
|---|---|
| ROC-AUC | **0.786** |
| Recall (toʻliqlik) @ 0.37 (tanlangan chegara) | **0.82** |
| Precision (aniqlik) @ 0.37 | 0.31 |
| Recall / Precision @ 0.50 | 0.66 / 0.40 |
| Accuracy @ 0.50 | 0.75 |

Chegaralarni saralash (`reports/metrics/recall085_results.txt`):

| Chegara | 0.20 | 0.25 | 0.30 | 0.35 | **0.37** | 0.40 | 0.45 | 0.50 |
|---|---|---|---|---|---|---|---|---|
| Recall | 0.967 | 0.933 | 0.891 | 0.840 | **0.818** | 0.783 | 0.723 | 0.660 |
| Precision | 0.220 | 0.240 | 0.265 | 0.293 | **0.306** | 0.325 | 0.359 | 0.396 |

| Precision–recall egri chizigʻi | Omillar ahamiyati |
|:--:|:--:|
| ![PR egri chizigʻi](reports/figures/pr_curve_optimal_threshold.png) | ![Omillar ahamiyati](reports/figures/CatBoost%20Feature%20Importance.jpg) |

Oldingi tajribalar (PCA, LightGBM bilan ansambllar, 30/45 daqiqalik maqsadlar) `notebooks/archive/`
va `reports/figures/` da saqlangan; yakuniy model ROC-AUC boʻyicha ularning barchasidan ustun chiqdi.

## Model artefaktlari

| Fayl | Hajmi | Qayerda | Ilova ishlatadi |
|---|---|---|---|
| `model_pipeline_main.pkl` — toʻliq pipeline (omillar transformeri + CatBoost), joblib `compress=3` | 433 MB | [Hugging Face Hub](https://huggingface.co/YOUR_HF_LOGIN/flight-delay-catboost) | **ha** |
| `catboost_model_main.cbm` — oʻsha CatBoost modeli, tabiiy formatda | 1,29 GB | faqat lokal | yoʻq |
| `lookup_data.joblib` — interfeys uchun yoʻnalishlar, aeroportlar, aviakompaniyalar | 0,1 MB | shu repozitoriy | ha |

Model katta, chunki CatBoost `ROUTE` (≈ 4 200 qiymat) va `DAY` kabi koʻp qiymatli kategoriyali
omillar uchun hisoblagich (CTR) jadvallarini saqlaydi.

**Yuklash mantiqi** (`src/model_loader.py`): agar `models/model_pipeline_main.pkl` lokal mavjud
boʻlsa, u ishlatiladi; aks holda fayl `huggingface_hub` orqali Hubdan bir marta yuklab olinadi va
keshlanadi. Manbani `HF_MODEL_REPO` va `HF_MODEL_FILE` muhit oʻzgaruvchilari bilan oʻzgartirish mumkin.

**Moslik:** pickle fayli Python 3.10, catboost 1.2.8, scikit-learn 1.7.2, numpy 2.2.6,
pandas 2.3.3 versiyalariga (barchasi `requirements.txt` da qotirilgan) va
`src/feature_engineering.py` dagi sinfga bogʻlangan. Bu faylni oʻzgartirish qayta oʻqitishni talab qiladi.

## Ishga tushirish

### Lokal (Python 3.10)

```bash
git clone https://github.com/YOUR_GITHUB_LOGIN/flight-delay-service.git
cd flight-delay-service
python -m venv venv
venv\Scripts\activate            # Windows   |   Linux/macOS da: source venv/bin/activate
pip install -r requirements.txt
streamlit run src/app.py
```

Windowsda `START_PROJECT.bat` faylini ikki marta bosish kifoya: u venv yaratadi, kutubxonalarni
oʻrnatadi va ilovani 8505 dan boshlab birinchi boʻsh portda ishga tushiradi.

Model birinchi ishga tushirishda Hugging Facedan yuklab olinadi (433 MB). Lokal nusxadan foydalanish
uchun uni `models/model_pipeline_main.pkl` manziliga joylashtiring.

### Docker

```bash
docker build -t flt-delay-app .
docker run -p 8501:8501 flt-delay-app
```

Soʻng http://localhost:8501 ni oching. Foydali variantlar:

```bash
# yuklab olish oʻrniga lokal model faylidan foydalanish
docker run -p 8501:8501 -v "$PWD/models:/app/models:ro" flt-delay-app

# agar sizning mintaqangizda standart PyPI CDN sekin boʻlsa — koʻzgu (mirror)
docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple -t flt-delay-app .
```

Obraz (python 3.10-slim, root boʻlmagan foydalanuvchi, `/_stcore/health` boʻyicha health check)
Hugging Face Space da ishlaydigan obrazning aynan oʻzi. Hajmi taxminan 1,6 GB, deyarli hammasi
CatBoost, SciPy va pyarrow.

### Qayta oʻqitish

```bash
kaggle datasets download -d usdot/flight-delays -p data --unzip
pip install -r requirements-dev.txt
python src/train_final.py          # sukut boʻyicha GPU; aks holda skriptda task_type="CPU" qoʻying
```

EDA va oʻqitish jarayoni `notebooks/01_EDA_and_Training.ipynb` da bayon qilingan.

## Loyiha tuzilmasi

```
flight_delay_service/
├── src/
│   ├── app.py                 Streamlit interfeysi
│   ├── analytics.py           xavf indeksi, nisbiy xavf, SHAP tushuntirishlari
│   ├── feature_engineering.py FlightFeatureEngineer (pickle qilingan pipelinening qismi)
│   ├── model_loader.py        lokal fayl yoki Hugging Face Hubdan yuklash
│   ├── train_final.py         oʻqitish skripti
│   └── i18n/                  en.py, uz.py, ru.py + tilni aniqlash
├── data/                      airlines.csv, airports.csv (flights.csv yuklab olinadi, repoda yoʻq)
├── models/                    lookup_data.joblib (+ yuklab olinadigan model_pipeline_main.pkl)
├── reports/figures/           PR egri chizigʻi, omillar ahamiyati, xatolar matritsalari
├── reports/metrics/           oʻqitishdan olingan xom koʻrsatkich hisobotlari
├── notebooks/                 01_EDA_and_Training.ipynb, archive/ — tajribalar
├── docs/                      Report.pdf, Tech_report.pdf (rus tilida), img/ — skrinshotlar
├── logos/                     interfeys uchun aviakompaniya logotiplari
├── Dockerfile · requirements.txt · requirements-dev.txt · START_PROJECT.bat
```

## Cheklovlar va rejalar

* **Model hajmi.** Siqilgan holda 433 MB / xotirada taxminan 1,3 GB. Ixcham qayta oʻqitish
  (`max_ctr_complexity=1`, `DAY` va `ROUTE` kategoriya sifatida emas) rejalashtirilgan.
* **Xavf indeksi kalibrlash emas.** Asosiy koʻrsatiladigan raqam — xom ehtimol; indeks shkalasi
  faqat vizual oʻzgartirish. Toʻlaqonli `CalibratedClassifierCV` bosqichi rejalarda.
* **Faqat 2015-yil AQSH maʼlumotlari.** Aeroportlar, aviakompaniyalar va trafik 2015-yilga tegishli;
  ilova metodni namoyish qiladi, operatsion vosita emas.
* **Trafik omillariga maʼlum sanalar kerak.** `traffic_density` va `AIRLINE_FLIGHT_COUNT` tanlangan
  aeroport/kun/soat uchun oʻqitish statistikasidan olinadi; uchramagan kombinatsiyalar uchun 1 qoʻyiladi.
* Rejada: transformer, yuklovchi va tarjimalar uchun pytest; GitHub Actions (lint + Docker build).

## Hujjatlar va muallif

* `docs/Report.pdf` — diplom ishi hisoboti, `docs/Tech_report.pdf` — texnik hisobot (ikkalasi rus tilida).
* Model kartasi: https://huggingface.co/YOUR_HF_LOGIN/flight-delay-catboost

**Askar Turanazarov** — Data Science boʻyicha diplom loyihasi, 2026. [MIT litsenziyasi](LICENSE) ostida tarqatiladi.
