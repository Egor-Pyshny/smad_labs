import numpy as np
from numpy.random import poisson
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import kstest


def count_metrics(data):
    mean_value = np.mean(data)
    median_value = np.median(data)
    mode_value = stats.mode(data).mode
    variance_value = np.var(data)
    coefficient_of_variation = (np.std(data) / mean_value) * 100
    print(f"Среднее: {mean_value}")
    print(f"Медиана: {median_value}")
    print(f"Мода: {mode_value}")
    print(f"Дисперсия: {variance_value}")
    print(f"Коэффициент вариации: {coefficient_of_variation} %\n")


def clean_data(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_numbers = [num for num in data if lower_bound <= num <= upper_bound]
    lst3 = [i for i in data if i not in filtered_numbers]
    print(f'Выбросы {lst3}\n')
    return filtered_numbers


def div_for_intervals(data, intervals_number):
    counts, bin_edges = np.histogram(data, bins=intervals_number)
    intervals = []
    group_indices = np.digitize(data, bin_edges)
    for i in range(1, intervals_number + 1):
        intervals.append([data[j] for j in range(0, len(data)) if group_indices[j] == i])
    count = len(list(filter(lambda x: x == max(data), data)))
    count2 = len(list(filter(lambda x: x == max(data), intervals[len(intervals) - 1])))
    dif = abs(count2 - count)
    for i in range(dif):
        intervals[len(intervals) - 1].append(max(data))
    return intervals, counts, bin_edges


def count_group_variances(intervals):
    variances = [sum([(x - np.mean(interval)) ** 2 for x in interval]) / len(interval) for interval in intervals]
    print(f'Групповая дисперсия = {variances}')
    return variances


def count_intragroup_variance(intervals):
    denominator = sum([sum([(x - np.mean(interval)) ** 2 for x in interval]) for interval in intervals])
    return denominator / sum([len(interval) for interval in intervals])


def count_intergroup_variance(intervals, data):
    mean = np.mean(data)
    denominator = sum([sum([((np.mean(interval) - mean) ** 2) * len(interval) for interval in intervals])])
    return denominator / len(data)


def count_variances(intervals, data):
    group_vars = count_group_variances(intervals)
    intragroup_var = count_intragroup_variance(intervals)
    intergroup_variance = count_intergroup_variance(intervals, data)
    print(f'Внутри дисперсия = {intragroup_var}')
    print(f'Межгр дисперсия = {intergroup_variance}')
    check_variances_rule(intragroup_var, intergroup_variance, data)
    return group_vars, intragroup_var, intergroup_variance


def check_variances_rule(intragroup_var, intergroup_variance, data):
    assert np.isclose(np.var(data), intragroup_var + intergroup_variance, atol=0.001), \
        f"{intragroup_var + intergroup_variance}"
    print(f'{np.var(data)} == {intragroup_var + intergroup_variance}')
    print(f'Доля межгр в общей = {intergroup_variance / np.var(data) * 100} %\n')


def draw_graphics(data, counts, bin_edges):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bin_edges, edgecolor='black', alpha=0.7)
    plt.title('Гистограмма')
    plt.xlabel('Интервалы')
    plt.ylabel('Количество')
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    plt.plot(bin_centers, counts, marker='o', linestyle='-', color='red', label='Полигон частот')
    plt.legend()
    plt.grid(True)
    plt.show()


def check_Kolmogorov(data, type):
    def cdf_theoretical(x):
        return stats.norm.cdf(x, loc=0, scale=1)
    res = kstest(np.array(data), cdf_theoretical)
    data2 = poisson(5, 100)
    res1 = kstest(data2, cdf_theoretical)

    if res < 0.05:
        print(f'Закон распределения не {type}')
        return False
    print(f'Закон распределения подходит')
    return True


def main():
    data = [
        24, 24, 24, 25, 19, 23, 20, 19, 18, 24, 22, 20, 19, 15, 20, 23, 22, 25, 27,
        20, 22, 23, 22, 21, 23, 26, 29, 30, 26, 28, 28, 20, 21, 20, 19, 22, 27, 22,
        25, 27, 30, 28, 33, 30, 26, 29, 29, 24, 23, 23, 24, 24, 26, 26, 25, 27, 23,
        21, 16, 19
    ]
    count_metrics(data)

    cleaned_data = clean_data(data)

    s = int(1 + np.log2(len(cleaned_data)))

    intervals, counts, bin_edges = div_for_intervals(cleaned_data, s)
    draw_graphics(cleaned_data, counts, bin_edges)
    count_metrics(cleaned_data)
    count_variances(intervals, cleaned_data)

    intervals, counts, bin_edges = div_for_intervals(cleaned_data, s + 1)
    count_variances(intervals, cleaned_data)

    intervals, counts, bin_edges = div_for_intervals(cleaned_data, s - 1)
    count_variances(intervals, cleaned_data)

    check_Kolmogorov(cleaned_data, 'norm')

if __name__ == '__main__':
    main()