# Русские строки интерфейса.

STRINGS = {
    # --- страница и сайдбар ---
    "app.title": "📊 Аналитическая система прогнозирования риска задержки авиарейса",
    "sidebar.header": "📝 Параметры рейса",
    "sidebar.language": "Язык",
    "sidebar.airline": "Авиакомпания",
    "sidebar.origin": "Аэропорт вылета",
    "sidebar.destination": "Аэропорт прилёта",
    "sidebar.month": "Месяц",
    "sidebar.day": "День",
    "sidebar.dep_hour": "Час вылета",
    "sidebar.unknown": "Неизвестно",
    "model.loading": "Загрузка модели (при первом запуске скачивается с Hugging Face)...",

    # --- вкладки ---
    "tab.analysis": "🚀 Анализ и прогноз",
    "tab.model": "📈 Обоснование модели",

    # --- верхняя панель ---
    "route.header": "**📍 Маршрут**",
    "flight.header": "**✈️ Детали рейса**",
    "flight.distance": "Расстояние: **{km} км**",
    "flight.time": "Время полёта: **{duration}**",
    "airline.header": "**🛩 Авиакомпания**",
    "airline.iata": "Код IATA: **{code}**",
    "airline.unknown": "Неизвестная авиакомпания",
    "duration.h_m": "{h} ч {m} мин",
    "duration.m": "{m} мин",

    # --- прогноз ---
    "button.analyze": "ПРОВЕСТИ АНАЛИЗ РИСКОВ",
    "risk.level": "Уровень риска",
    "risk.low": "🔵 Низкий",
    "risk.moderate": "🟡 Умеренный",
    "risk.high": "🟠 Высокий",
    "risk.critical": "🔴 Критический",
    "risk.index_bar": "Индекс риска (визуальная шкала): {idx}",
    "metric.prob": "Вероятность задержки",
    "metric.prob_help": "Сырая вероятность модели. Порог решения: {thr}.",
    "metric.rel": "Относительно средней ({base})",
    "rel.lower": "ниже средней доли задержек",
    "rel.typical": "на уровне средней доли задержек",
    "rel.higher": "выше средней доли задержек",
    "verdict.delay": "Вердикт модели: задержка вероятна (вероятность ≥ {thr}).",
    "verdict.ontime": "Вердикт модели: задержка маловероятна (вероятность < {thr}).",
    "econ.risk_value": "Ожидаемый финансовый ущерб (Risk Value): **${value}**",
    "insight": "💡 **Анализ:** основной фактор сегодня — **{feature}**, он {direction} риск задержки.",
    "direction.up": "повышает",
    "direction.down": "снижает",

    # --- детализация ущерба ---
    "econ.expander": "Детализация потенциального ущерба",
    "econ.total": "Общий ущерб при наступлении задержки: **${value}**",
    "econ.formula": (
        "Ожидаемый ущерб = вероятность задержки × общий ущерб при наступлении задержки.\n\n"
        "Здесь: `{expected} USD` при вероятности `{prob}` общего ущерба `{total} USD`."
    ),
    "econ.network": "Сетевой эффект (каскадные сбои)",
    "econ.direct": "Прямые расходы авиакомпании",
    "econ.pax": "Потери времени пассажиров",
    "econ.method": "Методика: FAA Value of Travel Time Savings (VTTS) и операционные расходы авиакомпаний.",

    # --- SHAP ---
    "shap.expander": "🔍 Почему такой результат? (SHAP-анализ факторов)",
    "shap.intro": "Вклад каждого фактора в итоговую вероятность (значения Шепли) для рейса из {origin} в {dest}:",
    "shap.axis_feature": "Признак",
    "shap.axis_contrib": "Вклад",
    "shap.caption": "Положительные значения (красный) повышают риск задержки, отрицательные (зелёный) — снижают.",

    # --- названия признаков ---
    "feature.MONTH": "Месяц",
    "feature.DAY": "День месяца",
    "feature.DAY_OF_WEEK": "День недели",
    "feature.AIRLINE": "Авиакомпания",
    "feature.ORIGIN_AIRPORT": "Аэропорт вылета",
    "feature.DESTINATION_AIRPORT": "Аэропорт прилёта",
    "feature.DEP_HOUR": "Час вылета",
    "feature.ARR_HOUR": "Час прилёта",
    "feature.SCHEDULED_TIME": "Плановое время полёта",
    "feature.DISTANCE": "Расстояние",
    "feature.ROUTE": "Маршрут",
    "feature.plan_speed": "Плановая скорость",
    "feature.SCHEDULED_ARRIVAL": "Прилёт по расписанию",
    "feature.SCHEDULED_DEPARTURE": "Вылет по расписанию",
    "feature.hour_sin": "Час (sin)",
    "feature.hour_cos": "Час (cos)",
    "feature.traffic_density": "Загрузка аэропорта (в час)",
    "feature.AIRLINE_FLIGHT_COUNT": "Рейсов авиакомпании в аэропорту (в день)",
    "feature.IS_NIGHT": "Ночной рейс",
    "feature.IS_WEEKEND": "Выходной день",

    # --- вкладка метрик ---
    "metrics.header": "Качество модели (обучено на 2,1 млн из 5,8 млн строк)",
    "metrics.loaded.compact": "Загруженная модель: **компактная** (max_ctr_complexity=3, {size}) — именно она работает в публичной демке.",
    "metrics.loaded.full": "Загруженная модель: **полная** (эталонная модель диплома, {size}).",
    "metrics.auc": "ROC-AUC на тестовой выборке: **{auc}**",
    "metrics.recall": "Recall (полнота) при пороге 0.37: **{recall}** (точность {precision})",
    "metrics.variants_header": "Два варианта модели",
    "metrics.variants_text": "Обе модели обучены на одних данных, одном разбиении и с одними параметрами; разница только "
                             "в глубине комбинаций категорий в CTR-таблицах CatBoost (`max_ctr_complexity`: 4 и 3). "
                             "Компактная занимает в 2,7 раза меньше памяти и помещается на бесплатный хостинг ценой "
                             "0,007 ROC-AUC. Полная модель остаётся эталонным результатом диплома.",
    "metrics.col_metric": "Метрика",
    "metrics.col_full": "Полная модель",
    "metrics.col_compact": "Компактная модель",
    "metrics.row_auc": "ROC-AUC",
    "metrics.row_recall": "Recall @ 0.37",
    "metrics.row_precision": "Precision @ 0.37",
    "metrics.row_size": "Размер пайплайна (pkl)",
    "metrics.row_ram": "RAM после загрузки",
    "metrics.row_ctr": "Комбинации CTR (max_ctr_complexity)",
    "metrics.row_script": "Скрипт обучения",
    "metrics.figures_note": "Графики ниже построены для полной модели в ходе дипломной работы.",
    "metrics.pr_caption": "Выбор порога классификации (Precision–Recall)",
    "metrics.fi_caption": "Важность признаков (CatBoost)",
}
