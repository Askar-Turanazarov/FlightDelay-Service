# Askar Turanazarov / 1542 /

# src/feature_engineering.py
# Здесь хранится класс FlightFeatureEngineer() - класс для проектирования признаков для улучшения метрик модели
# и более качественного обучения модели CatBoost

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# Используем Feature Engineering для создания новых колонок и выделения зависимостей для модели
class FlightFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Класс для продвинутого проектирования признаков (Feature Engineering).
    Превращает базовые данные о рейсах в предиктивные факторы возможных задержек авиарейсов.
    By Askar Turanazarov
    """

    def __init__(self):
        # Словари для хранения статистики, накопленной на этапе обучения (fit)
        self.traffic_map = {}  # загрузка (трафик) аэропорта
        self.airline_load_map = {}  # загрузка авиакомпании

    def fit(self, X, y=None):
        """
        Изучает закономерности трафика на тренировочных данных.
        Важно: fit делается только на Train, чтобы избежать утечки данных (Data Leakage).
        """
        # 1. Анализ пропускной способности аэропорта:
        # Считаем количество вылетов из каждого аэропорта в конкретный час каждого дня.
        # Обоснование: Перегрузка ВПП и терминалов — главная причина системных задержек.
        self.traffic_map = X.groupby(['ORIGIN_AIRPORT', 'MONTH', 'DAY', 'DEP_HOUR']).size().to_dict()

        # 2. Анализ операционной нагрузки авиакомпании:
        # Считаем рейсы конкретного перевозчика в конкретном аэропорту за день.
        # Обоснование: "Эффект домино" — при нехватке резервных бортов или экипажей
        # задержка одного рейса срывает всё расписание авиакомпании в этом хабе.
        self.airline_load_map = X.groupby(['ORIGIN_AIRPORT', 'AIRLINE', 'MONTH', 'DAY']).size().to_dict()

        return self

    def transform(self, X):
        """Применяет накопленные знания и создает новые признаки в DataFrame."""
        X_ = X.copy()

        # --- БЛОК 1: ЦИКЛИЧЕСКОЕ ВРЕМЯ ---
        # Обоснование: Обычный час (0-23) — линейный, но время циклично (23:59 ближе к 00:01).
        # Превращаем час в координаты на окружности через синус и косинус.
        X_['hour_sin'] = np.sin(2 * np.pi * X_['DEP_HOUR'] / 24)
        X_['hour_cos'] = np.cos(2 * np.pi * X_['DEP_HOUR'] / 24)

        # --- БЛОК 2: НАГРУЗКА И ТРАФИК ---
        # Извлекаем плотность трафика аэропорта (если данных нет, ставим 1 рейс).
        traffic_keys = pd.Series(list(zip(X_['ORIGIN_AIRPORT'], X_['MONTH'], X_['DAY'], X_['DEP_HOUR'])))
        X_['traffic_density'] = traffic_keys.map(self.traffic_map).fillna(1)

        # Извлекаем концентрацию рейсов авиакомпании (AIRLINE_FLIGHT_COUNT).
        airline_keys = pd.Series(list(zip(X_['ORIGIN_AIRPORT'], X_['AIRLINE'], X_['MONTH'], X_['DAY'])))
        X_['AIRLINE_FLIGHT_COUNT'] = airline_keys.map(self.airline_load_map).fillna(1)

        # --- БЛОК 3: ЛОГИСТИКА И ГЕОГРАФИЯ ---
        # Создаем уникальный идентификатор маршрута (Направление).
        # Обоснование: Определенные коридоры (например, в Нью-Йорк) перегружены чаще других.
        X_['ROUTE'] = X_['ORIGIN_AIRPORT'].astype(str) + "_" + X_['DESTINATION_AIRPORT'].astype(str)

        # Рассчитываем плановую скорость полета (мили в минуту).
        # Обоснование: Чем выше плановая скорость, тем меньше "запас" времени у пилота
        # на компенсацию предвылетной задержки в воздухе.
        X_['plan_speed'] = X_['DISTANCE'] / (X_['SCHEDULED_TIME'] + 1e-5)

        # Выделяем ночные рейсы (00:00 - 05:59).
        # Обоснование: Ночью трафик ниже, но задержки могут быть вызваны длительным тех. обслуживанием.
        X_['IS_NIGHT'] = ((X_['DEP_HOUR'] >= 0) & (X_['DEP_HOUR'] <= 5)).astype(int)

        return X_

