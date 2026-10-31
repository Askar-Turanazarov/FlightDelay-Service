# Oʻzbekcha (lotin) interfeys matnlari.
# Imlo: oʻ / gʻ uchun ʻ (U+02BB), tutuq belgisi uchun ʼ (U+02BC).

STRINGS = {
    # --- sahifa va yon panel ---
    "app.title": "📊 Reys kechikishi xavfini tahlil qilish tizimi",
    "sidebar.header": "📝 Reys parametrlari",
    "sidebar.language": "Til",
    "sidebar.airline": "Aviakompaniya",
    "sidebar.origin": "Uchish aeroporti",
    "sidebar.destination": "Qoʻnish aeroporti",
    "sidebar.month": "Oy",
    "sidebar.day": "Kun",
    "sidebar.dep_hour": "Uchish soati",
    "sidebar.unknown": "Nomaʼlum",
    "footer.portfolio": "Portfolio",
    "model.loading": "Model yuklanmoqda (birinchi ishga tushirishda Hugging Face’dan yuklab olinadi)...",

    # --- boʻlimlar ---
    "tab.analysis": "🚀 Tahlil va prognoz",
    "tab.model": "📈 Model asoslari",

    # --- yuqori panel ---
    "route.header": "**📍 Yoʻnalish**",
    "flight.header": "**✈️ Reys tafsilotlari**",
    "flight.distance": "Masofa: **{km} km**",
    "flight.time": "Parvoz vaqti: **{duration}**",
    "airline.header": "**🛩 Aviakompaniya**",
    "airline.iata": "IATA kodi: **{code}**",
    "airline.unknown": "Nomaʼlum aviakompaniya",
    "duration.h_m": "{h} soat {m} daq.",
    "duration.m": "{m} daq.",

    # --- prognoz ---
    "button.analyze": "XAVF TAHLILINI OʻTKAZISH",
    "risk.level": "Xavf darajasi",
    "risk.low": "🔵 Past",
    "risk.moderate": "🟡 Oʻrtacha",
    "risk.high": "🟠 Yuqori",
    "risk.critical": "🔴 Kritik",
    "risk.index_bar": "Xavf indeksi (koʻrgazmali shkala): {idx}",
    "metric.prob": "Kechikish ehtimoli",
    "metric.prob_help": "Modelning xom (oʻzgartirilmagan) ehtimoli. Qaror chegarasi: {thr}.",
    "metric.rel": "Oʻrtachaga nisbatan ({base})",
    "rel.lower": "oʻrtacha kechikish ulushidan past",
    "rel.typical": "oʻrtacha kechikish ulushi darajasida",
    "rel.higher": "oʻrtacha kechikish ulushidan yuqori",
    "verdict.delay": "Model xulosasi: kechikish ehtimoli yuqori (ehtimol ≥ {thr}).",
    "verdict.ontime": "Model xulosasi: kechikish ehtimoli past (ehtimol < {thr}).",
    "econ.risk_value": "Kutilayotgan moliyaviy zarar (Risk Value): **${value}**",
    "insight": "💡 **Tahlil:** bugungi asosiy omil — **{feature}**, u kechikish xavfini {direction}.",
    "direction.up": "oshiradi",
    "direction.down": "kamaytiradi",

    # --- zarar tafsiloti ---
    "econ.expander": "Ehtimoliy zarar tafsiloti",
    "econ.total": "Kechikish yuz berganda umumiy zarar: **${value}**",
    "econ.formula": (
        "Kutilayotgan zarar = kechikish ehtimoli × kechikish yuz berganda umumiy zarar.\n\n"
        "Bu yerda: `{prob}` ehtimol va `{total} USD` umumiy zarar uchun kutilayotgan zarar `{expected} USD`."
    ),
    "econ.network": "Zanjirli (tarmoq) taʼsiri",
    "econ.direct": "Aviakompaniyaning bevosita xarajatlari",
    "econ.pax": "Yoʻlovchilar vaqtining yoʻqotilishi",
    "econ.method": "Metodika: FAA Value of Travel Time Savings (VTTS) va aviakompaniya operatsion xarajatlari.",

    # --- SHAP ---
    "shap.expander": "🔍 Nega aynan shunday natija? (SHAP omillar tahlili)",
    "shap.intro": "{origin} — {dest} reysi uchun har bir omilning yakuniy ehtimolga qoʻshgan hissasi (Shepli qiymatlari):",
    "shap.axis_feature": "Omil",
    "shap.axis_contrib": "Hissa",
    "shap.caption": "Musbat qiymatlar (qizil) kechikish xavfini oshiradi, manfiy qiymatlar (yashil) — kamaytiradi.",

    # --- omillar nomlari ---
    "feature.MONTH": "Oy",
    "feature.DAY": "Oyning kuni",
    "feature.DAY_OF_WEEK": "Hafta kuni",
    "feature.AIRLINE": "Aviakompaniya",
    "feature.ORIGIN_AIRPORT": "Uchish aeroporti",
    "feature.DESTINATION_AIRPORT": "Qoʻnish aeroporti",
    "feature.DEP_HOUR": "Uchish soati",
    "feature.ARR_HOUR": "Qoʻnish soati",
    "feature.SCHEDULED_TIME": "Rejadagi parvoz vaqti",
    "feature.DISTANCE": "Masofa",
    "feature.ROUTE": "Yoʻnalish",
    "feature.plan_speed": "Rejadagi tezlik",
    "feature.SCHEDULED_ARRIVAL": "Jadval boʻyicha qoʻnish",
    "feature.SCHEDULED_DEPARTURE": "Jadval boʻyicha uchish",
    "feature.hour_sin": "Soat (sin)",
    "feature.hour_cos": "Soat (cos)",
    "feature.traffic_density": "Aeroport yuklamasi (soatiga)",
    "feature.AIRLINE_FLIGHT_COUNT": "Aviakompaniya reyslari aeroportda (kuniga)",
    "feature.IS_NIGHT": "Tungi reys",
    "feature.IS_WEEKEND": "Dam olish kuni",

    # --- koʻrsatkichlar boʻlimi ---
    "metrics.header": "Model sifati (5,8 mln qatordan 2,1 mln qatorda oʻqitilgan)",
    "metrics.loaded.compact": "Yuklangan model: **ixcham** (max_ctr_complexity=3, {size}) — ommaviy demo aynan shu variantda ishlaydi.",
    "metrics.loaded.full": "Yuklangan model: **toʻliq** (diplom ishining asosiy modeli, {size}).",
    "metrics.auc": "Test toʻplamidagi ROC-AUC: **{auc}**",
    "metrics.recall": "0.37 chegarada Recall (toʻliqlik): **{recall}** (aniqlik {precision})",
    "metrics.variants_header": "Modelning ikki varianti",
    "metrics.variants_text": "Ikkala model ham bir xil maʼlumotlar, bir xil boʻlinish va bir xil parametrlarda oʻqitilgan; "
                             "farq faqat CatBoost CTR-jadvallarida kategoriyalar qanchalik chuqur birlashtirilishida "
                             "(`max_ctr_complexity`: 4 va 3). Ixcham variant xotirada 2,7 barobar kam joy oladi va bepul "
                             "hostingga sigʻadi, buning evaziga ROC-AUC 0,007 ga past. Toʻliq model diplom ishining asosiy natijasi boʻlib qoladi.",
    "metrics.col_metric": "Koʻrsatkich",
    "metrics.col_full": "Toʻliq model",
    "metrics.col_compact": "Ixcham model",
    "metrics.row_auc": "ROC-AUC",
    "metrics.row_recall": "Recall @ 0.37",
    "metrics.row_precision": "Precision @ 0.37",
    "metrics.row_size": "Pipeline hajmi (pkl)",
    "metrics.row_ram": "Yuklangandan keyingi RAM",
    "metrics.row_ctr": "CTR kombinatsiyalari (max_ctr_complexity)",
    "metrics.row_script": "Oʻqitish skripti",
    "metrics.figures_note": "Quyidagi grafiklar diplom ishi davomida toʻliq model uchun tuzilgan.",
    "metrics.pr_caption": "Tasniflash chegarasini tanlash (Precision–Recall)",
    "metrics.fi_caption": "Omillarning ahamiyati (CatBoost)",
}
