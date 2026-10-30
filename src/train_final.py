# Askar Turanazarov / 1542 /

# src/train_final.py
# Здесь мы загружаем данные, упаковываем и обучаем модель. Это важный файл проекта, который
# создаст и сохранит модель, используемая для предсказаний.

import pandas as pd
import os
import sys
import time
import numpy as np
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
import joblib
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.feature_engineering import FlightFeatureEngineer

print('\n# Здесь мы загружаем данные, упаковываем и обучаем модель. Это главный файл проекта, который')
print('# создаст и сохранит модель, используемая для предсказаний.\n')
print('-'*50)
print('Иницализация...')

def load_flight_data_final(n_rows=2100000):
    """Загружает данные из огромного датасета с одновременной оптимизации данных для экономии памяти
    ОЗУ (RAM) и видеопамяти (GPU) и ускорения обучения модели без потери качества данных. By Askar Turanazarov"""

    # Только то, что реально доступно ДО вылета
    cols_to_keep = [
        'MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE',
        'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
        'SCHEDULED_DEPARTURE', 'DISTANCE', 'SCHEDULED_TIME',
        'SCHEDULED_ARRIVAL', 'ARRIVAL_DELAY', 'TAIL_NUMBER'              # ARRIVAL_DELAY только для таргета!
    ]

    df = pd.read_csv('data/flights.csv', usecols=cols_to_keep, nrows=n_rows)

    # 1. Таргет - наша целевая, создадим. Задержка более 15 минут это стандарт в авации (и стандарт FAA для США)
    df = df.dropna(subset=['ARRIVAL_DELAY'])
    df['target'] = (df['ARRIVAL_DELAY'] > 15).astype(np.int8)

    # 2. Удаляем колонку-"утечку", а то сломает модель
    df = df.drop('ARRIVAL_DELAY', axis=1)

    # 3. Базовый Feature Engineering, попробуем добавить правильные признаки для модели
    df['DEP_HOUR'] = (df['SCHEDULED_DEPARTURE'] // 100).astype(np.int8)
    df['ARR_HOUR'] = (df['SCHEDULED_ARRIVAL'] // 100).astype(np.int8)
    df['IS_WEEKEND'] = df['DAY_OF_WEEK'].isin([6, 7]).astype(np.int8)

    # 4. Оптимизация типов для категорий
    cat_cols = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']
    for col in cat_cols:
        df[col] = df[col].astype(str).fillna('UNKNOWN')

    # 5. ВАЖНО: Оптимизация числовых типов (сжимаем int64) - так как модель стала "раздуваться"
    # Обоснование: MONTH (1-12) и DAY (1-31) легко влезают в int8 (-128...127)
    # DISTANCE и время в минутах влезают в int16 (-32768...32767)
    df['MONTH'] = df['MONTH'].astype(np.int8)
    df['DAY'] = df['DAY'].astype(np.int8)
    df['DAY_OF_WEEK'] = df['DAY_OF_WEEK'].astype(np.int8)

    # Время (HHMM) и Дистанция могут быть больше 127, берем int16
    df['SCHEDULED_DEPARTURE'] = df['SCHEDULED_DEPARTURE'].astype(np.int16)
    df['SCHEDULED_ARRIVAL'] = df['SCHEDULED_ARRIVAL'].astype(np.int16)
    df['DISTANCE'] = df['DISTANCE'].astype(np.int16)
    df['SCHEDULED_TIME'] = df['SCHEDULED_TIME'].fillna(0).astype(np.int16)

    return df


if __name__ == "__main__":
    print('\nНачало загрузки данных...')
    df = load_flight_data_final(2100000)
    print(f"Новый размер данных: {df.shape}")
    print(f'\n*Колонки:\n{df.columns}')
    print(f"\n*Объем данных в памяти: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

    # Базовый набор (сырье для трансформера)
    base_features = [
        'MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE',
        'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
        'SCHEDULED_DEPARTURE', 'DISTANCE', 'SCHEDULED_TIME',
        'SCHEDULED_ARRIVAL', 'DEP_HOUR', 'ARR_HOUR', 'IS_WEEKEND'
    ]

    X = df[base_features]
    y = df['target']

    # Разделяем на Train и Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Готово! Обучающая выборка: {X_train.shape}")

    # теперь сборка пайплайна и обучение модели Cat Boost (CatBoostClassifier)

    print('\n*Подготовка к работе - модель CatBoost...')
    print('*Инициализация модели и пайплайна...')
    print('-' * 50)

    # 1. Список категориальных признаков для CatBoost
    # Указываем признаки, которые являются текстовыми или логическими категориями.
    # CatBoost применит к ним встроенный алгоритм кодирования (Ordered TS).
    cat_features = [
        'MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE',
        'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'ROUTE',
        'IS_WEEKEND', 'IS_NIGHT'
    ]

    # 2. Сборка Pipeline (единый конвейер обработки)
    # Это гарантирует, что Feature Engineering применится и к Train, и к Test без ошибок.
    model_pipeline = Pipeline([
        # Первый шаг: Наш кастомный класс для создания 6 новых типов признаков
        ('feature_eng', FlightFeatureEngineer()),

        # Второй шаг: Классификатор с настройками под GPU и Recall
        ('clf', CatBoostClassifier(
            iterations=2500,  # Количество итераций (деревьев). На GPU можно ставить много.
            learning_rate=0.03,  # Скорость обучения. Чем меньше, тем точнее модель находит закономерности.
            depth=7,  # Глубина деревьев. 8 — "золотая середина" для табличных данных.
            l2_leaf_reg=12,  # Регуляризация. Предотвращает переобучение на шумах.

            # БАЛАНСИРОВКА: В данных 18.9% задержек. Вес 4.28 (81.0/18.9) заставляет модель
            # фокусироваться на задержках сильнее, чтобы поднять Recall. Попробуем уменьшить, чем отношение
            scale_pos_weight=4.2,

            # АППАРАТНОЕ УСКОРЕНИЕ: Использование видеокарты для сокращения времени обучения.
            task_type="GPU",
            devices='0',

            # Определяет стратегию отбора объектов для деревьев
            bootstrap_type='Bernoulli',
            subsample=0.8,
            # Политика роста деревьев - SymmetricTree — по умолчанию; Lossguide — для больших данных.
            grow_policy='SymmetricTree',

            # МОНИТОРИНГ: AUC — основная метрика. Recall и Precision выводятся для контроля
            # способности модели находить реальные опоздания в процессе обучения.
            eval_metric='AUC',
            custom_metric=['Recall', 'Precision'],

            random_seed=42,  # Фиксация случайности для воспроизводимости результатов.

            # Увеличивает разнообразие деревьев, снижает переобучение (меньше - увереннее, больше - модель сомневается)
            random_strength=2,
            early_stopping_rounds=100,  # Если в течение 100 итераций AUC не растет —
            # стопаем обучение. Это спасет от переобучения

            verbose=100,  # Печатать лог каждые 100 деревьев.
            cat_features=cat_features  # Указание модели на категориальные столбцы.
        ))
    ])

    # 3. ИСПОЛНЕНИЕ: ОБУЧЕНИЕ И ЗАМЕР ВРЕМЕНИ
    print("\n🛫 Запуск обучения модели на GPU с расширенным мониторингом...\n")

    start_time = time.time()  # Засекаем время старта (включает трансформацию данных)

    model_pipeline.fit(X_train, y_train)  # ОБУЧЕНИЕ МОДЕЛИ

    end_time = time.time()  # Фиксируем время завершения

    # Расчет затраченного времени
    training_duration = end_time - start_time
    minutes = int(training_duration // 60)
    seconds = int(training_duration % 60)

    # Сохранение пайплайна
    joblib.dump(model_pipeline, "models/model_pipeline_main.pkl", compress=3)

    # # Сохранение CatBoost отдельно (1): Обрезаем модель до лучшей итерации (удаляет мусорные деревья)
    # # Это часто высвобождает внутренние буферы
    # model_pipeline.named_steps['clf'].shrink(model_pipeline.named_steps['clf'].best_iteration_)

    # Сохранение CatBoost отдельно (2): Сохраняем модель через встроенный метод, отключая хранение метаданных
    # pool=None и использование только необходимых данных для предсказания
    model_pipeline.named_steps['clf'].save_model("models/catboost_model_main.cbm", format="cbm")

    print("-" * 50)
    print(f"🛬 Модель успешно обучена на 2 млн+ ({df.shape[0]:,.0f}) строк!")
    print(f"⏱ Суммарное время (FE + Обучение): {minutes} мин. {seconds} сек.")
    print("-" * 50)
    print('\n*Модель CatBoost сама отдельно сохранена в файле "../models/catboost_model_main.cbm".')
    print('*Pipeline с моделью весь сохранён в файле "../models/model_pipeline_main.pkl"')
    print("-" * 50)

