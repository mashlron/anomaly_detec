# Импорт необходимых библиотек для работы с данными
import pandas as pd
import numpy as np

# Импорт библиотек для визуализации
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Импорт утилит для работы со временем и вывода таблиц
from datetime import datetime
from tabulate import tabulate

# Импорт методов машинного обучения из sklearn
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

# Импорт функций для комбинаторики и прогресс-бара
from itertools import combinations
from tqdm import tqdm

# Импорт компонентов для нейросетевой модели
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping

# Импорт утилит для параллельных вычислений и системных функций
import warnings
from joblib import Parallel, delayed
import multiprocessing
import os

# Настройка matplotlib для работы без GUI
matplotlib.use('Agg')

# Отключение предупреждений для чистоты вывода
warnings.filterwarnings('ignore')

# Настройка уровня логирования TensorFlow (2 = только ошибки)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Отключение оптимизаций oneDNN для совместимости
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Дополнительное отключение UserWarning
warnings.filterwarnings('ignore', category=UserWarning)


def load_and_preprocess():
    """Загрузка и предварительная обработка данных из CSV-файла"""
    try:
        # Загрузка данных из файла
        data = pd.read_csv('real_1_v2.csv')

        # Сортировка данных по timestamp
        data = data.sort_values('timestamp').reset_index(drop=True)

        # Преобразование timestamp в datetime с шагом +1 день начиная с 01.01.2020
        base_date = pd.to_datetime('2020-01-01 00:00:00')
        data['datetime'] = base_date + pd.to_timedelta(data.index, unit='d')
        data = data.drop(['timestamp'], axis=1)
        # Вывод информации о загруженных данных
        print("\n" + "=" * 50)
        print(data.head(10))
        print(f"Файл успешно загружен (строк: {len(data)})")
        print(f"Распределение аномалий:\n{data['is_anomaly'].value_counts().to_string()}")
        print("=" * 50)

        # Проверка и обработка пропущенных значений
        initial_rows = len(data)
        data = data.dropna()

        # Удаление строк с нулевыми значениями только в столбце 'value'
        zero_mask = (data['value'] == 0)
        data = data[~zero_mask]

        removed_rows = initial_rows - len(data)
        if removed_rows > 0:
            print(f"\nУдалено строк (пропуски и нулевые значения): {removed_rows}")
            print(f"Осталось строк: {len(data)}")
        else:
            print("\nПропущенных и нулевых значений не обнаружено")

        return data

    except FileNotFoundError:
        # Обработка ошибки отсутствия файла
        print("\nОшибка: файл не найден в текущей директории")
        exit()


def add_time_features(data, time_col):
    """Извлечение временных признаков из столбца с датой/временем"""
    # Преобразование строки в datetime, если еще не преобразовано
    if not pd.api.types.is_datetime64_any_dtype(data[time_col]):
        data[time_col] = pd.to_datetime(data[time_col])

    # Извлечение различных временных характеристик
    # data['hour'] = data[time_col].dt.hour  # Час дня (0-23)
    data['day'] = data[time_col].dt.day  # День месяца
    data['month'] = data[time_col].dt.month  # Месяц (1-12)
    data['day_of_week'] = data[time_col].dt.dayofweek + 1  # День недели (1-7)
    data['quarter'] = data[time_col].dt.quarter  # Квартал (1-4)

    # Вывод информации о добавленных признаках
    print("\n" + "=" * 50)
    print("ДОБАВЛЕНЫ ВРЕМЕННЫЕ ПРИЗНАКИ")
    print("=" * 50)
    print(tabulate(data[['day', 'month', 'day_of_week', 'quarter']].head(5),
                   headers='keys', tablefmt='pretty', showindex=False))
    return data


def add_lags_and_moving_avg(data, features, time_col, max_lag=10):
    """Генерация лагов и скользящих средних для выбранных признаков"""
    # Сортировка данных по времени для корректного расчета сдвигов
    data = data.sort_values(by=time_col).reset_index(drop=True)

    # Для каждого выбранного признака создаем лаги и скользящие средние
    for col in features:
        # Создание лагов от 1 до max_lag
        for lag in range(1, max_lag + 1):
            data[f'lag_{lag}_{col}'] = data[col].shift(lag)

        # Создание скользящих средних с окном от 2 до max_lag
        for window in range(2, max_lag + 1):
            data[f'ma_{window}_{col}'] = data[col].rolling(window).mean().shift(1)

    # Получение списков созданных лагов и скользящих средних
    lag_cols = [c for c in data.columns if 'lag_' in c]
    ma_cols = [c for c in data.columns if 'ma_' in c]

    # Удаление строк с NaN (появились из-за сдвигов)
    initial_rows = len(data)
    data = data.dropna()
    print(f"\nУдалено строк с NaN: {initial_rows - len(data)}")

    # Вывод примеров созданных лагов
    print("\nЛаги:")
    print(tabulate(data[lag_cols].head(5), headers='keys', tablefmt='pretty', showindex=False))

    # Вывод примеров скользящих средних
    print("\nСкользящие средние:")
    print(tabulate(data[ma_cols].head(5), headers='keys', tablefmt='pretty', showindex=False))

    return data


def select_variables(data):
    """Интерактивный выбор переменных для анализа"""
    # Вывод списка доступных столбцов с их типами
    print("\n" + "=" * 50)
    print("ДОСТУПНЫЕ СТОЛБЦЫ")
    print("=" * 50)
    print(tabulate(
        pd.DataFrame({
            "№": range(1, len(data.columns) + 1),
            "Столбец": data.columns,
            "Тип": data.dtypes
        }),
        headers='keys',
        tablefmt='pretty',
        showindex=False
    ))

    # Запрос выбора временного столбца
    time_col = data.columns[int(input("\nВведите № временного столбца: ")) - 1]

    # Запрос выбора числовых столбцов (можно несколько через запятую)
    feature_nums = input("Введите № числового столбца: ").split(',')
    features = [data.columns[int(num.strip()) - 1] for num in feature_nums]

    return data, time_col, features


class AnomalyDetector:
    """Класс для обнаружения аномалий тремя методами"""

    def __init__(self):
        # Инициализация стандартизатора для автоэнкодера
        self.scaler = StandardScaler()

    def train_isolation_forest(self, X, contamination):
        """Обучение модели Isolation Forest"""
        model = IsolationForest(n_estimators=500,
                                max_samples='auto',
                                contamination=contamination,
                                random_state=42)
        model.fit(X)
        return model

    def train_one_class_svm(self, X, contamination):
        """Обучение One-Class SVM"""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = OneClassSVM(
            kernel='rbf',
            nu=contamination,
            gamma='auto'
        )
        model.fit(X_scaled)
        return model

    def train_autoencoder(self, X, epochs=50):
        """Обучение автоэнкодера"""
        # Стандартизация данных
        X_scaled = self.scaler.fit_transform(X)

        # Создание модели автоэнкодера
        model = Sequential([
            InputLayer(shape=(X.shape[1],)),
            Dense(128, activation='relu', kernel_regularizer=l1_l2(0.01)),
            Dropout(0.2),
            Dense(64, activation='relu', kernel_regularizer=l1_l2(0.01)),
            Dense(32, activation='relu'),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(128, activation='relu'),
            Dense(X.shape[1], activation='linear')
        ])

        model.compile(optimizer=Adam(0.001), loss='mse')
        history = model.fit(
            X_scaled, X_scaled,
            epochs=epochs,
            batch_size=32,
            validation_split=0.1,
            verbose=0
        )
        return model

    def evaluate_model(self, model, X, y, model_type):
        """Оценка качества модели"""
        if model_type == "IsolationForest":
            # Для Isolation Forest: -1 = аномалия, 1 = норма
            preds = np.where(model.predict(X) == -1, 1, 0)
        elif model_type == "OneClassSVM":
            # Для One-Class SVM: -1 = аномалия, 1 = норма
            preds = np.where(model.predict(X) == -1, 1, 0)
        else:
            # Для автоэнкодера: вычисляем MSE реконструкции
            X_scaled = self.scaler.transform(X)
            reconstructions = model.predict(X_scaled, verbose=0)
            mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
            # Аномалии = объекты с MSE выше 95% квантиля
            preds = np.where(mse > np.quantile(mse, 0.95), 1, 0)

        # Возвращаем метрики качества
        return {
            'f1': f1_score(y, preds),  # F1-мера
            'precision': precision_score(y, preds),  # Точность
            'recall': recall_score(y, preds),  # Полнота
            'cm': confusion_matrix(y, preds)  # Матрица ошибок
        }


def generate_combinations(base_features, lag_features, ma_features, n_lags, n_mas):
    """Генерация всех комбинаций признаков"""
    # Создание комбинаций лагов и скользящих средних
    lag_combinations = list(combinations(lag_features, n_lags)) if n_lags <= len(lag_features) else []
    ma_combinations = list(combinations(ma_features, n_mas)) if n_mas <= len(ma_features) else []

    all_combinations = []
    # Комбинируем каждую комбинацию лагов с каждой комбинацией скользящих средних
    for lags in lag_combinations:
        for mas in ma_combinations:
            # Формируем полный набор признаков
            features = list(base_features) + list(lags) + list(mas)
            all_combinations.append({
                'features': features,  # Все признаки для модели
                'lags': lags,  # Использованные лаги
                'mas': mas  # Использованные скользящие средние
            })

    print(f"\nСгенерировано комбинаций: {len(all_combinations)}")
    return all_combinations


def process_combination(combination, X_scaled, y, contamination):
    """Обработка одной комбинации признаков"""
    detector = AnomalyDetector()
    results = []

    # 1. Обучение и оценка Isolation Forest
    model_if = detector.train_isolation_forest(X_scaled[combination['features']], contamination)
    metrics_if = detector.evaluate_model(model_if, X_scaled[combination['features']], y, "IsolationForest")
    results.append({
        'method': 'IsolationForest',
        'features': combination['features'],
        'lags': combination['lags'],
        'mas': combination['mas'],
        **metrics_if
    })

    # 2. Обучение и оценка One-Class SVM
    model_svm = OneClassSVM(kernel='rbf', nu=contamination, gamma='auto', cache_size=1000)
    model_svm.fit(X_scaled[combination['features']])
    metrics_svm = detector.evaluate_model(model_svm, X_scaled[combination['features']], y, "OneClassSVM")
    results.append({
        'method': 'OneClassSVM',
        'features': combination['features'],
        'lags': combination['lags'],
        'mas': combination['mas'],
        **metrics_svm
    })

    # 3. Обучение и оценка Autoencoder
    model_ae = detector.train_autoencoder(X_scaled[combination['features']])
    metrics_ae = detector.evaluate_model(model_ae, X_scaled[combination['features']], y, "Autoencoder")
    results.append({
        'method': 'Autoencoder',
        'features': combination['features'],
        'lags': combination['lags'],
        'mas': combination['mas'],
        **metrics_ae
    })

    return results


def analyze_data(data, target_col, n_lags, n_mas, n_jobs=-1):
    """Основная функция анализа данных"""
    # Базовые временные признаки
    base_features = ['day', 'month', 'day_of_week', 'quarter']

    # Автоматическое определение лагов и скользящих средних
    lag_features = [col for col in data.columns if 'lag_' in col]
    ma_features = [col for col in data.columns if 'ma_' in col]

    # Проверка доступности запрошенного количества признаков
    if n_lags > len(lag_features):
        print(
            f"\nВнимание: запрошено {n_lags} лагов, но доступно только {len(lag_features)}. Используется максимум доступных.")
        n_lags = len(lag_features)

    if n_mas > len(ma_features):
        print(
            f"\nВнимание: запрошено {n_mas} скользящих средних, но доступно только {len(ma_features)}. Используется максимум доступных.")
        n_mas = len(ma_features)

    # Расчет доли аномалий для моделей
    contamination = float(data[target_col].mean())

    # Разделение на признаки и целевую переменную
    X = data.drop(columns=[target_col])
    y = data[target_col]
    # Генерация всех комбинаций признаков
    combinations_list = generate_combinations(base_features, lag_features, ma_features, n_lags, n_mas)

    if not combinations_list:
        print("\nОшибка: не удалось сгенерировать комбинации признаков. Проверьте параметры n_lags и n_mas.")
        return pd.DataFrame()

    print(f"\nВсего комбинаций для анализа: {len(combinations_list)}")
    print(f"Используется ядер CPU: {multiprocessing.cpu_count() if n_jobs == -1 else n_jobs}")

    # Параллельный запуск обработки комбинаций
    try:
        all_results = Parallel(n_jobs=n_jobs)(
            delayed(process_combination)(comb, X, y, contamination)
            for comb in tqdm(combinations_list, desc="Анализ комбинаций")
        )
    except Exception as e:
        print(f"\nОшибка при параллельном выполнении: {e}")
        return pd.DataFrame()

    # Сбор всех результатов в один DataFrame
    flat_results = [item for sublist in all_results for item in sublist if item]
    if not flat_results:
        print("\nНе удалось получить результаты. Возможно, все комбинации вызвали ошибки.")
        return pd.DataFrame()

    results_df = pd.DataFrame(flat_results)
    return results_df


def save_plot_as_image(cm, metrics, method, comb_num, output_dir='results'):
    """Сохранение графиков с результатами в файл"""
    # Создание директории, если не существует
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Создание фигуры с двумя субплогами
    plt.figure(figsize=(12, 5))

    # 1. Матрица ошибок
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.title(f"Confusion Matrix ({method})")

    # 2. График метрик
    plt.subplot(1, 2, 2)
    plt.bar(['Precision', 'Recall', 'F1'],
            [metrics['precision'], metrics['recall'], metrics['f1']])
    plt.ylim(0, 1)
    plt.title("Метрики качества")

    # Сохранение в файл
    plt.tight_layout()
    filename = f"{output_dir}/{method}_comb_{comb_num}.png"
    plt.savefig(filename)
    plt.close()
    return filename


def print_top_results(results_df, top_k=3):
    """Вывод и сохранение лучших результатов"""
    if results_df.empty:
        print("\nНет результатов для отображения.")
        return

    # Создание директории для результатов
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Анализ для каждого метода
    for method in ['IsolationForest', 'OneClassSVM', 'Autoencoder']:
        # Выбор топ-K результатов для текущего метода
        method_results = results_df[results_df['method'] == method] \
            .sort_values('f1', ascending=False) \
            .head(top_k)

        if method_results.empty:
            print(f"\nНет результатов для метода {method}")
            continue

        # Вывод заголовка
        print(f"\n{'=' * 50}")
        print(f"{method} - Top {top_k}")
        print("=" * 50)

        # Вывод информации по каждой топовой комбинации
        for idx, (_, row) in enumerate(method_results.iterrows(), 1):
            print(f"\nКомбинация #{idx}")
            print(f"F1: {row['f1']:.3f} | Precision: {row['precision']:.3f} | Recall: {row['recall']:.3f}")
            print("\nПризнаки:")
            print(f"Базовые: hour, day, month, day_of_week, quarter")
            print(f"Лаги: {', '.join(row['lags'])}")
            print(f"Скользящие средние: {', '.join(row['mas'])}")

            # Сохранение графиков
            img_path = save_plot_as_image(row['cm'], row, method, idx)
            print(f"\nГрафики сохранены в: {img_path}")


def main():
    """Главная функция выполнения программы"""
    # 1. Загрузка и предобработка данных
    data = load_and_preprocess()

    # 2. Выбор переменных для анализа
    data, time_col, features = select_variables(data)

    # 3. Добавление временных признаков
    data = add_time_features(data, time_col)

    # 4. Генерация лагов и скользящих средних
    data = add_lags_and_moving_avg(data, features, time_col, max_lag=10)

    # 5. Настройка параметров анализа
    print("\n" + "=" * 50)
    print("НАСТРОЙКА ПАРАМЕТРОВ АНАЛИЗА")
    print("=" * 50)
    n_lags = int(input("Введите количество лагов в комбинации: "))
    n_mas = int(input("Введите количество скользящих средних в комбинации: "))
    top_k = int(input("Введите количество топовых результатов для вывода: "))

    # 6. Запуск анализа данных
    results_df = analyze_data(data, 'is_anomaly', n_lags=n_lags, n_mas=n_mas)

    # 7. Сохранение и вывод результатов
    if not results_df.empty:
        # Сохранение всех результатов в CSV
        results_df.to_csv('anomaly_results.csv', index=False)
        print("\nРезультаты сохранены в anomaly_results.csv")

        # Вывод лучших результатов
        print_top_results(results_df, top_k=top_k)

        # 8. Вывод абсолютно лучшего результата
        best_result = results_df.sort_values('f1', ascending=False).iloc[0]
        print("\n" + "=" * 50)
        print("ЛУЧШИЙ РЕЗУЛЬТАТ")
        print("=" * 50)
        print(f"Метод: {best_result['method']}")
        print(f"F1-score: {best_result['f1']:.3f}")
        print(f"Precision: {best_result['precision']:.3f}")
        print(f"Recall: {best_result['recall']:.3f}")
        print("\nИспользованные признаки:")
        print(f"Базовые: hour, day, month, day_of_week, quarter")
        print(f"Лаги: {', '.join(best_result['lags'])}")
        print(f"Скользящие средние: {', '.join(best_result['mas'])}")


if __name__ == "__main__":
    # Точка входа в программу
    main()