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
    "metric.rel": "Относительно средней доли задержек ({base})",
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
    "metrics.auc": "ROC-AUC на тестовой выборке: **0.786**",
    "metrics.recall": "Recall (полнота) при пороге 0.37: **0.82** (точность 0.31)",
    "metrics.pr_caption": "Выбор порога классификации (Precision–Recall)",
    "metrics.fi_caption": "Важность признаков (CatBoost)",
}
