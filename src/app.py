# Askar Turanazarov / 1542 /

# src/app.py
# Основной файл приложения Streamlit для запуска и демонстрации работы модели CatBoost.
# Запуск программы осуществляется в терминале: streamlit run src/app.py


import streamlit as st
import pandas as pd
import joblib
import os
import sys
import plotly.graph_objects as go
import plotly.express as px

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.analytics import FlightAnalytics

# --- ПУТИ ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model_pipeline_main.pkl')
LOOKUP_PATH = os.path.join(BASE_DIR, 'models', 'lookup_data.joblib')

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

st.set_page_config(page_title="Flight Risk Analytics Pro", layout="wide")


# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---
def format_duration(minutes):
    h, m = divmod(int(minutes), 60)
    return f"{h} ч {m} мин" if h > 0 else f"{m} мин"


def calculate_economic_impact(prob, distance_miles):
    # Константы (базируются на данных FAA)
    COST_PER_MIN_AIRLINE = 110.0
    PASSENGER_TIME_VAL_HOUR = 62.50
    pax = 150 if distance_miles < 1000 else 280
    avg_delay = 52

    # 1. Прямой ущерб
    direct_flight_cost = avg_delay * COST_PER_MIN_AIRLINE
    pax_loss = (avg_delay / 60) * PASSENGER_TIME_VAL_HOUR * pax

    # 2. Сетевой эффект (Reactionary delay) — фиксированный коэффициент 2.2x от прямых потерь
    total_event_cost = (direct_flight_cost + pax_loss) * 2.2
    network_impact = total_event_cost - (direct_flight_cost + pax_loss)

    # 3. Математическое ожидание риска
    expected_risk_value = prob * total_event_cost

    return {
        "expected": expected_risk_value,
        "total_event": total_event_cost,
        "direct": direct_flight_cost,
        "pax": pax_loss,
        "network": network_impact
    }


def get_risk_level(prob):
    if prob < 0.2: return "🔵 Низкий", "green"
    if prob < 0.4: return "🟡 Умеренный", "orange"
    if prob < 0.6: return "🟠 Высокий", "red"
    return "🔴 Критический", "darkred"


# --- ЗАГРУЗКА РЕСУРСОВ ---
@st.cache_resource
def load_all():
    model = joblib.load(MODEL_PATH)
    data = joblib.load(LOOKUP_PATH)
    try:
        air_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'airports.csv'))
        # Собираем расширенный словарь данных об аэропортах
        air_info = air_df.set_index('IATA_CODE').to_dict('index')
        line_names = pd.read_csv(os.path.join(BASE_DIR, 'data', 'airlines.csv')).set_index('IATA_CODE')[
            'AIRLINE'].to_dict()
    except:
        air_info, line_names = {}, {}
    return model, data, air_info, line_names


model, data, air_info, line_names = load_all()

# Создаем объект для расчетов - постобработка результатов вероятностей
# Используем 0.44 как среднее, так как модель сейчас выдает примерно столько
analytics = FlightAnalytics(model, global_mean=0.44, global_std=0.12)

# --- ИНТЕРФЕЙС (SIDEBAR) ---
st.title("📊 Аналитическая система прогнозирования авиарисков - риск задержки авиарейса.")

with st.sidebar:
    st.header("📝 Параметры рейса")
    airline_code = st.selectbox("Авиакомпания", data['airlines'],
                                format_func=lambda x: f"{line_names.get(x, 'Unknown')} ({x})")
    origin_code = st.selectbox("Аэропорт вылета", data['airports'],
                               format_func=lambda x: f"{x} - {air_info.get(x, {}).get('AIRPORT', 'Unknown')}")

    dest_list = data['lookup'][data['lookup']['ORIGIN_AIRPORT'] == origin_code]['DESTINATION_AIRPORT'].unique()
    dest_code = st.selectbox("Аэропорт прилета", sorted(dest_list),
                             format_func=lambda x: f"{x} - {air_info.get(x, {}).get('AIRPORT', 'Unknown')}")

    st.divider()
    month = st.slider("Месяц", 1, 12, 3)
    day = st.slider("День", 1, 31, 15)
    dep_hour = st.slider("Час вылета", 0, 23, 12)

# --- ГЛАВНЫЙ ЭКРАН ---
tab1, tab2 = st.tabs(["🚀 Анализ и Прогноз", "📈 Обоснование модели"])

with tab1:
    route = data['lookup'][(data['lookup']['ORIGIN_AIRPORT'] == origin_code) &
                           (data['lookup']['DESTINATION_AIRPORT'] == dest_code)].iloc[0]

    # ВЕРХНЯЯ ПАНЕЛЬ С ДАННЫМИ
    c1, gap, c2, c3 = st.columns([1, 0.2, 0.6, 1.1])
    with c1:
        st.markdown("\t**📍 Маршрут**")

        # Получаем данные из air_info (названия и города)
        origin_full = air_info.get(origin_code, {})
        dest_full = air_info.get(dest_code, {})

        # Используем Flexbox для выравнивания
        st.markdown(f"""
                <div style="display: flex; align-items: flex-start; justify-content: space-between; margin-top: -20px;">
                    <div style="flex: 1;">
                        <h1 style="margin:0; font-family: serif; font-size: 2.3rem; line-height: 1;">{origin_code}</h1>
                        <div style="font-weight: bold; font-size: 0.85rem; margin-top: 2px; font-family: sans-serif;">{origin_full.get('AIRPORT', 'Unknown')}</div>
                        <div style="color: gray; font-size: 0.75rem; font-family: sans-serif;">{origin_full.get('CITY', '')}, {origin_full.get('STATE', '')}</div>
                    </div>
                    <div style="flex: 0.2; text-align: center; padding-top: 22px; font-size: 1.5rem; color: #4A90E2;">
                        ➔
                    </div>
                    <div style="flex: 1; text-align: right;">
                        <h1 style="margin:0; font-family: serif; font-size: 2.3rem; line-height: 1;">{dest_code}</h1>
                        <div style="font-weight: bold; font-size: 0.85rem; margin-top: 2px; font-family: sans-serif;">{dest_full.get('AIRPORT', 'Unknown')}</div>
                        <div style="color: gray; font-size: 0.75rem; font-family: sans-serif;">{dest_full.get('CITY', '')}, {dest_full.get('STATE', '')}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # выведем отдельные переменные городов для использования в блоке анализа
        origin_city = air_info.get(origin_code, {}).get('CITY', '')         # для SHAP-резюме
        destin_city = air_info.get(dest_code, {}).get('CITY', '')           # для SHAP-резюме
    with c2:
        st.write("**✈️ Детали полета (рейса)**")
        dist_km = int(route['DISTANCE'] * 1.609)
        st.write(f"Расстояние:    **{dist_km:,} км**".replace(',', ' '))
        st.write(f"Время полёта:  **{format_duration(route['SCHEDULED_TIME'])}**")

    with c3:
        st.write("**🛩 Авиакомпания**")
        # Создаём две под-колонки для размещения инфо и лого
        text_col, logo_col = st.columns([1, 1.2])

        # Подгружаем лого авиакомпаний:
        with logo_col:
            logo_path = f'logos/{airline_code}.png'
            try:
                st.image(logo_path)
            except:
                st.write('🛩')
        with text_col:
            st.write(line_names.get(airline_code, "Unknown Airline"))
            st.caption(f"Код ИАТА: **{airline_code}**")

    st.divider()

    col_map, col_res = st.columns([1.3, 1])

    with col_map:
        # Карта
        if origin_code in air_info and dest_code in air_info:
            fig = go.Figure(go.Scattergeo(
                locationmode='USA-states',
                lon=[air_info[origin_code]['LONGITUDE'], air_info[dest_code]['LONGITUDE']],
                lat=[air_info[origin_code]['LATITUDE'], air_info[dest_code]['LATITUDE']],
                mode='lines+markers',
                line=dict(width=5, color='#FF4B4B'),
                marker=dict(size=15, color='#1C83E1')
            ))
            fig.update_layout(height=400, margin={"r": 0, "t": 0, "l": 0, "b": 0}, geo_scope='usa')
            st.plotly_chart(fig, use_container_width=True)


    with col_res:
        if st.button("ПРОВЕСТИ АНАЛИЗ РИСКОВ", type="primary", use_container_width=True):
            # 1. Формируем входные данные
            input_df = pd.DataFrame([{
                'MONTH': month, 'DAY': day, 'DAY_OF_WEEK': 1, 'AIRLINE': airline_code,
                'ORIGIN_AIRPORT': origin_code, 'DESTINATION_AIRPORT': dest_code,
                'SCHEDULED_DEPARTURE': dep_hour * 100, 'DISTANCE': route['DISTANCE'],
                'SCHEDULED_TIME': route['SCHEDULED_TIME'], 'SCHEDULED_ARRIVAL': (dep_hour + 2) * 100,
                'DEP_HOUR': dep_hour, 'ARR_HOUR': (dep_hour + 2) % 24, 'IS_WEEKEND': 0
            }])

            # 2. Получаем сырую вероятность от CatBoost
            raw_prob = model.predict_proba(input_df)[0][1]

            # 3. Применяем калибровку, чтобы расширить диапазон
            prob = analytics.calibrate_probability(raw_prob)

            # 4. Получаем интерпретацию через Z-score (сравнение со средним)
            risk_text, risk_color, z_val = analytics.get_risk_interpretation(prob)

            # 5. Считаем экономику (уже на основе откалиброванной вероятности)
            impact = calculate_economic_impact(prob, route['DISTANCE'])

            # --- ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ---
            risk_label, color = get_risk_level(prob)
            st.markdown(f"### Уровень риска: <span style='color:{color}'>{risk_label}</span>", unsafe_allow_html=True)
            st.progress(prob)

            # Выводим вероятность и Z-интерпретацию
            col_p1, col_p2 = st.columns(2)
            col_p1.metric("Вероятность задержки", f"{prob:.1%}")
            col_p2.metric("Относительно нормы", f"{z_val:+.1f} σ", help=risk_text)

            st.warning(f"Финансовый риск (Risk Value): **${impact['expected']:,.0f}**")

            # --- ВАЖНОЕ - SHAP АНАЛИЗ ---
            # Текстовое краткое резюме для SHAP-анализа (интеллектуальный помощник предсказанию, интерпретация)
            shap_data = analytics.get_local_shap(input_df)

            # Словарь для перевода признаков на русский язык:
            names_map = {
                'MONTH': 'Месяц',
                'DAY': 'День',
                'DAY_OF_WEEK': 'День недели',
                'AIRLINE': 'Авиакомпания',
                'ORIGIN_AIRPORT': 'Аэропорт вылета',
                'DESTINATION_AIRPORT': 'Аэропорт прилёта',
                'DEP_HOUR': 'Час вылета',
                'SCHEDULED_TIME': 'Время в пути',
                'DISTANCE': 'Расстояние',
                'ROUTE': 'Маршрут',
                'plan_speed': 'Скорость (плановая)',
                'SCHEDULED_ARRIVAL': 'Прибытие (по расп.)',
                'SCHEDULED_DEPARTURE': 'Вылет (по расп.)'
            }

            # Очистка имен признаков и перевод
            # Убираем префиксы типа 'remainder__' или 'feature_eng__'
            shap_data['Признак'] = shap_data['Признак'].str.split('__').str[-1]

            # Маппим русские названия. Если слова в словаре нет — остаётся оригинал
            shap_data['Признак'] = shap_data['Признак'].map(names_map).fillna(shap_data['Признак'])

            # Далее работа блока краткого SHAP-анализа
            top_factor = shap_data.iloc[0]
            direction = "повышает" if top_factor['Вклад'] > 0 else "снижает"
            st.info(
                f"💡 **Анализ:** Основным фактором сегодня является **{top_factor['Признак']}**, который {direction} риск задержки рейса.")
    # st.markdown('---')

    # Блок подробной детализации предсказания модели:
    try:
        # Временная "переменная"-заглушка для правильной отрисовки
        _ = impact['total_event']

        # Выносим две отдельные колонки под детализированные отчёты предсказания (детализация ущерба + SHAP)
        details1, details2 = st.columns(2)

        # Первая колонка с детализацией ущерба
        with details1:
            with st.expander("Детализация потенциального ущерба"):
                st.write(f"Общий ущерб при наступлении задержки: **${impact['total_event']:,.0f}**")
                # st.markdown('---')
                st.write(f'''Финансовый риск: Вероятность задержки **Х** Общий ущерб при наступлении задержки рейса.
                \nПоэтому он равен `{impact['expected']:,.0f}USD` при вероятности `{prob:.1%}` общего ущерба 
                `{impact['total_event']:,.0f}USD`.''')
                st.markdown(f"""
                ---
                * **Сетевой эффект (каскадные сбои):** `${impact['network']:,.0f}`
                * **Прямые расходы авиакомпании:** `${impact['direct']:,.0f}`
                * **Потери времени пассажиров:** `${impact['pax']:,.0f}`
                """)
                st.caption("Методика расчета: FAA Value of Travel Time Savings (VTTS) & Airline Operational Costs.")

        # вторая колонка с SHAP-анализом конкретного рейса
        with details2:
            with st.expander("🔍 Почему такой результат? (SHAP-анализ факторов)"):
                st.write(
                    f"Вклад параметров в итоговую вероятность (значения Шепли) на рейсе из {origin_city} в {destin_city}:")
                # Получаем данные SHAP из нашего нового класса

                # Рисуем горизонтальный график
                fig_shap = px.bar(
                    shap_data,
                    x='Вклад',
                    y='Признак',
                    orientation='h',
                    color='Вклад',
                    color_continuous_scale='RdYlGn_r',  # Красный для +риска, зеленый для -риска
                    text_auto='.2f'
                )
                fig_shap.update_layout(height=300, showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_shap, use_container_width=True)

                st.caption(
                    "Положительные значения (красный) повышают риск задержки, отрицательные (зеленый) — снижают.")
    except NameError:
        pass

with tab2:
    st.header("Метрики качества (обучено на 2.1 млн строк)")

    m1, m2 = st.columns(2)
    with m1:
        st.info("Итоговый ROC-AUC модели на тестовой выборке: **0.786**")
        st.image("reports/figures/pr_curve_optimal_threshold.png",
                 caption="Выбор порога классификации (Precision-Recall)")
    with m2:
        st.info("Итоговый Recall (полнота) модели на тестовой выборке: **0.82**")
        st.image("reports/figures/CatBoost Feature Importance.jpg",
                 caption="Вклад признаков в прогноз (Feature Importance)")









