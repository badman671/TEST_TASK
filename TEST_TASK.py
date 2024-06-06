import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.stats.proportion as smp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import sqlite3

# Подключение к базе данных
db_file_path = 'path_to_your_database/testcase.db'
conn = sqlite3.connect(db_file_path)

# Задание 1: Успешность прототипа

# Классический подход
successful_prototypes_our = 0
total_prototypes_our = 200

# Вероятность успеха следующего прототипа
# В классическом подходе вероятность рассчитывается как отношение успешных прототипов к общему числу прототипов
prob_success_classic = successful_prototypes_our / total_prototypes_our if total_prototypes_our > 0 else 0

# Баесовский подход
# Параметры априорного распределения Beta(1, 1) - это равномерное распределение
alpha_prior = 1
beta_prior = 1

# Параметры апостериорного распределения Beta обновляются с учетом наблюдаемых данных
alpha_posterior = alpha_prior + successful_prototypes_our
beta_posterior = beta_prior + total_prototypes_our - successful_prototypes_our

# Апостериорное распределение
posterior_distribution = stats.beta(alpha_posterior, beta_posterior)

# Вероятность успеха 201-го прототипа - среднее апостериорного распределения
probability_success_201 = posterior_distribution.mean()

print("Задание 1: Успешность прототипа")
print(f"Классический подход: {prob_success_classic}")
print(f"Баесовский подход: {probability_success_201}")

# Задание 2: Сравнение групп платящих игроков

# 1. Дизайн эксперимента:
# Оптимальный дизайн эксперимента включает случайное разделение новых игроков на контрольную и экспериментальную группы.
# Контрольная группа не получает обновление, экспериментальная группа получает.
# Сбор данных о проценте платящих игроков в обеих группах.
# Анализ разницы в процентах платящих игроков между группами с использованием статистических методов.

# 2. Расчет длительности эксперимента
# Параметры
daily_new_players = 100
control_conversion_rate = 0.10
expected_conversion_rate = 0.11
alpha = 0.05  # уровень значимости
power = 0.80  # мощность теста

# Размер эффекта рассчитывается на основе разницы между ожидаемым и текущим уровнем конверсии
effect_size = sms.proportion_effectsize(control_conversion_rate, expected_conversion_rate)

# Расчет размера выборки для каждой группы
sample_size = sms.NormalIndPower().solve_power(effect_size, power=power, alpha=alpha, ratio=1)
sample_size = int(np.ceil(sample_size))

# Количество дней для эксперимента рассчитывается исходя из ежедневного притока новых игроков
days_needed = sample_size / daily_new_players
print("\nЗадание 2: Сравнение групп платящих игроков")
print(f"Размер выборки для каждой группы: {sample_size}")
print(f"Количество дней для эксперимента: {days_needed}")

# 3. Генерация датасета с 10% плательщиков (контроль)
np.random.seed(42)  # Устанавливаем seed для воспроизводимости результатов

# Размеры групп
n_control = sample_size
n_experiment = sample_size

# Генерация контрольной группы (10% платящих)
control_group = np.random.binomial(1, control_conversion_rate, n_control)

# Генерация экспериментальных групп с разными процентами платящих
experiment_group_9 = np.random.binomial(1, 0.09, n_experiment)
experiment_group_10 = np.random.binomial(1, 0.10, n_experiment)
experiment_group_11 = np.random.binomial(1, 0.11, n_experiment)
experiment_group_12 = np.random.binomial(1, 0.12, n_experiment)

# Создание DataFrame для удобства
data = pd.DataFrame({
    'group': ['control'] * n_control + ['experiment_9'] * n_experiment + ['experiment_10'] * n_experiment + ['experiment_11'] * n_experiment + ['experiment_12'] * n_experiment,
    'payment': np.concatenate([control_group, experiment_group_9, experiment_group_10, experiment_group_11, experiment_group_12])
})

# 4. Расчет доверительных интервалов для контрольной и экспериментальных групп

# Функция для расчета доверительных интервалов
def calculate_confidence_interval(data, group_name, alpha=0.05):
    group_data = data[data['group'] == group_name]['payment']
    successes = group_data.sum()  # Количество успешных (платящих) игроков
    nobs = len(group_data)  # Общее количество игроков в группе
    ci_lower, ci_upper = smp.proportion_confint(successes, nobs, alpha=alpha, method='normal')
    return (ci_lower, ci_upper)

# Доверительные интервалы для контрольной группы
ci_control = calculate_confidence_interval(data, 'control')

# Доверительные интервалы для экспериментальных групп
ci_experiment_9 = calculate_confidence_interval(data, 'experiment_9')
ci_experiment_10 = calculate_confidence_interval(data, 'experiment_10')
ci_experiment_11 = calculate_confidence_interval(data, 'experiment_11')
ci_experiment_12 = calculate_confidence_interval(data, 'experiment_12')

print(f"Доверительный интервал для контрольной группы (10% плательщиков): {ci_control}")
print(f"Доверительный интервал для экспериментальной группы 9%: {ci_experiment_9}")
print(f"Доверительный интервал для экспериментальной группы 10%: {ci_experiment_10}")
print(f"Доверительный интервал для экспериментальной группы 11%: {ci_experiment_11}")
print(f"Доверительный интервал для экспериментальной группы 12%: {ci_experiment_12}")

# 5. Баесовский подход: расчет HDI для контрольной и экспериментальных групп

# Функция для расчета HDI (Highest Density Interval - Интервал наибольшей плотности)
def calculate_hdi(trace, hdi_prob=0.95):
    return np.percentile(trace, [(1 - hdi_prob) / 2 * 100, (1 + hdi_prob) / 2 * 100])

# Функция для баесовского анализа
def bayesian_analysis(data, group_name, alpha_prior=1, beta_prior=1, hdi_prob=0.95):
    group_data = data[data['group'] == group_name]['payment']
    successes = group_data.sum()  # Количество успешных (платящих) игроков
    nobs = len(group_data)  # Общее количество игроков в группе
    
    # Апостериорное распределение
    alpha_posterior = alpha_prior + successes
    beta_posterior = beta_prior + nobs - successes
    posterior_distribution = stats.beta(alpha_posterior, beta_posterior)
    
    # Генерация выборок из апостериорного распределения
    samples = posterior_distribution.rvs(10000)
    
    # Расчет HDI
    hdi_interval = calculate_hdi(samples, hdi_prob=hdi_prob)
    
    return hdi_interval

# Расчет HDI для каждой группы
hdi_control = bayesian_analysis(data, 'control')
hdi_experiment_9 = bayesian_analysis(data, 'experiment_9')
hdi_experiment_10 = bayesian_analysis(data, 'experiment_10')
hdi_experiment_11 = bayesian_analysis(data, 'experiment_11')
hdi_experiment_12 = bayesian_analysis(data, 'experiment_12')

print(f"HDI для контрольной группы (10% плательщиков): {hdi_control}")
print(f"HDI для экспериментальной группы 9%: {hdi_experiment_9}")
print(f"HDI для экспериментальной группы 10%: {hdi_experiment_10}")
print(f"HDI для экспериментальной группы 11%: {hdi_experiment_11}")
print(f"HDI для экспериментальной группы 12%: {hdi_experiment_12}")

# Задание 4: Эффективность рекламных кампаний

# 1. Подтверждение или опровержение гипотезы

# Загрузка данных из таблицы costs
query_costs = "SELECT * FROM costs"
costs_data = pd.read_sql_query(query_costs, conn)

# Загрузка данных из таблицы revenue
query_revenue = "SELECT * FROM revenue"
revenue_data = pd.read_sql_query(query_revenue, conn)

# Объединение таблиц costs и revenue по campaign_id и Install_Dates
merged_data = pd.merge(costs_data, revenue_data, on=['campaign_id', 'Install_Dates', 'Country'])

# Расчет ROAS на 60-й день
merged_data['ROAS_60d'] = merged_data['60d_LTV'] / merged_data['spends']

# Визуализация зависимости ROAS от затрат
plt.figure(figsize=(10, 6))
sns.scatterplot(data=merged_data, x='spends', y='ROAS_60d')
plt.xlabel('Затраты (COST)')
plt.ylabel('ROAS на 60-й день')
plt.title('Зависимость ROAS на 60-й день от затрат')
plt.show()

# Корреляционный анализ
correlation = merged_data[['spends', 'ROAS_60d']].corr()
print("Корреляционный анализ между затратами и ROAS на 60-й день:")
print(correlation)

# 2. Оптимизация бюджета
# Расчет маркетинговой прибыли для каждой кампании
merged_data['marketing_profit'] = merged_data['60d_LTV'] - merged_data['spends']

# Группировка данных по campaign_id и расчет среднего значения прибыли
profit_by_campaign = merged_data.groupby('campaign_id')['marketing_profit'].mean().reset_index()

# Поиск максимальной прибыли и соответствующего бюджета
max_profit_campaign = profit_by_campaign.loc[profit_by_campaign['marketing_profit'].idxmax()]
print(f"Кампания с максимальной средней маркетинговой прибылью: {max_profit_campaign}")

# 3. Рекомендации по бюджетам
# Добавление среднего значения прибыли в исходные данные
merged_data = merged_data.merge(profit_by_campaign, on='campaign_id', suffixes=('', '_mean'))

# Рекомендации по бюджетам
merged_data['recommendation'] = merged_data['marketing_profit_mean'].apply(
    lambda x: 'Increase budget' if x > 0 else 'Decrease budget or stop'
)

# Вывод рекомендаций для первых нескольких кампаний
recommendations = merged_data[['campaign_id', 'marketing_profit_mean', 'recommendation']].drop_duplicates()
print("Рекомендации по бюджетам для первых нескольких кампаний:")
print(recommendations.head())

# Задание 5: Связь рекламного и органического трафика

# Загрузка данных из таблицы source_comparison
query_source_comparison = "SELECT * FROM source_comparison"
source_comparison_data = pd.read_sql_query(query_source_comparison, conn)

# Преобразование даты в формат datetime
source_comparison_data['Install_Dates'] = pd.to_datetime(source_comparison_data['Install_Dates'])

# Группировка данных по дате и источнику
daily_installs = source_comparison_data.groupby(['Install_Dates', 'source_type'])['installs'].sum().unstack().fillna(0)

# Визуализация ежедневных установок
plt.figure(figsize=(14, 7))
daily_installs.plot()
plt.xlabel('Дата')
plt.ylabel('Количество установок')
plt.title('Ежедневные установки: рекламный и органический источники')
plt.show()

# Модель линейной регрессии для проверки зависимости
X = daily_installs['Paid']
y = daily_installs['Organic']
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
summary = model.summary()
print("Результаты регрессионного анализа:")
print(summary)

# Тест на независимость Пирсона
correlation, p_value = pearsonr(daily_installs['Paid'], daily_installs['Organic'])
print(f"Коэффициент корреляции Пирсона: {correlation}")
print(f"p-value: {p_value}")

# Закрытие соединения с базой данных
conn.close()

