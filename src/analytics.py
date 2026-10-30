# Askar Turanazarov / 1542 /

# src/analytics.py
# Файл для глубокого анализа результатов предсказания модели CatBoost. Используется логика:
# Z-Scaling (Индекс риска), Калибровка (через Sigmoid Stretching) и Local SHAP (для конкретного рейса)

import numpy as np
import pandas as pd
import catboost as cb
import plotly.express as px


class FlightAnalytics:
    def __init__(self, model, global_mean=0.189, global_std=0.12):
        """
        Класс FlightAnalytics используется для глубокого анализа результатов
        предсказания модели CatBoost
        """
        # global_mean: средняя вероятность задержки по датасету (18.9%)
        # global_std: примерное стандартное отклонение (можно взять 0.12)
        self.model = model
        self.avg = global_mean
        self.std = global_std

    def calibrate_probability(self, prob):
        """
        Растягивает вероятность для лучшей визуализации (Sigmoid Stretching).
        Помогает уйти от узкого диапазона 40-60%.
        """
        # Превращаем вероятность в логит (score)
        # Добавляем epsilon чтобы избежать деления на ноль
        eps = 1e-5
        logit = np.log((prob + eps) / (1 - prob + eps))

        # Сдвигаем центр и увеличиваем контрастность (коэффициент 1.5-2.0)
        # Это настройка чувствительности интерфейса
        calibrated = 1 / (1 + np.exp(-1.8 * (logit - 0.5)))
        return calibrated

    def get_risk_interpretation(self, prob):
        """
        Рассчитывает Индекс риска (Z-score) и текстовое пояснение.
        """
        z_score = (prob - self.avg) / self.std

        if z_score <= -0.5:
            text = "Намного надежнее среднего"
            color = "green"
        elif z_score <= 0.5:
            text = "В рамках нормы для данного типа рейсов"
            color = "gray"
        else:
            diff_pct = (prob / self.avg - 1) * 100
            text = f"Риск выше среднего на {diff_pct:.0f}%"
            color = "red"

        return text, color, z_score

    def get_local_shap(self, input_df):
        """
        Считает вклад каждой фичи (признака) в конкретный прогноз.
        Использует быстрый встроенный метод CatBoost - .get_feature_importance()
        """
        # Получаем SHAP-значения для одной строки
        # Проверяем, является ли модель пайплайном
        if hasattr(self.model, 'named_steps'):
            actual_model = self.model.named_steps['clf']  # Имя модели из пайплайна
            transformer = self.model.named_steps['feature_eng']  # Твой трансформер

            # Прогоняем через feature_eng (обработка 2.1 млн строк и генерация 7 новых фичей)
            processed_data = transformer.transform(input_df)

            # Если после трансформации это numpy array, превращаем обратно в DataFrame для удобства
            if isinstance(processed_data, np.ndarray):
                processed_data = pd.DataFrame(processed_data, columns=transformer.get_feature_names_out())
        else:
            actual_model = self.model
            processed_data = input_df

            # 2. Самый важный момент: получаем индексы или имена категориальных признаков
            # CatBoost хранит их в самой модели
        cat_features = actual_model.get_cat_feature_indices()

        # 3. Создаем Pool, явно указывая категориальные признаки
        # Это устраняет ошибку "Cannot convert 'AS' to float"
        pool_data = cb.Pool(data=processed_data, cat_features=cat_features)

        # 4. Вычисляем SHAP-значения (вклады фичей)
        shap_values = actual_model.get_feature_importance(
            data=pool_data,
            type='ShapValues'
        )[0]

        feature_contributions = shap_values[:-1]
        feature_names = processed_data.columns

        # 5. Формируем результат для визуализации
        shap_df = pd.DataFrame({
            'Признак': feature_names,
            'Вклад': feature_contributions
        })

        # Маппинг имен для красоты (сокращаем технические префиксы после трансформера)
        shap_df['Признак'] = shap_df['Признак'].apply(lambda x: x.split('__')[-1])

        shap_df['Abs_Вклад'] = shap_df['Вклад'].abs()
        shap_df = shap_df.sort_values('Abs_Вклад', ascending=False).head(7)

        return shap_df





