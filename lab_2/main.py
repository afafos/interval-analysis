import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

import intvalpy as ip

def is_tolerance_set_empty(A, b):
    """
    Определяет, является ли допусковое множество для заданных интервальной матрицы A и
    интервального вектора b пустым.

    Функция вычисляет максимум распознающего функционала Tol для системы A x = b
    с помощью метода maximize из библиотеки intvalpy.
    Если найденное максимальное значение Tol меньше 0, считается, что допусковое
    множество пусто.

    Параметры
    ----------
    A : ip.Interval
        Интервальная матрица (коэффициенты системы).
    b : ip.Interval
        Интервальный вектор (правая часть системы).

    Возвращает
    -------
    tuple
        Кортеж (is_empty, argmax_point), где:
        - is_empty (bool): True, если допусковое множество пусто, иначе False.
        - argmax_point (list): Точка (вектор x), в которой Tol достигает максимума.
    """
    max_tol = ip.linear.Tol.maximize(A, b)
    val = float(max_tol[1])
    return val < 0, [float(v) for v in max_tol[0]]


def b_correction(b, k):
    """
    Выполняет b-коррекцию интервального вектора b путём расширения каждого интервала
    на k в обе стороны.

    Функция создаёт интервальный вектор e, в котором каждый интервал равен [-k, k],
    и прибавляет его к исходному вектору b. Таким образом расширяются границы b,
    что может помочь достичь ненулевого допускового множества.

    Параметры
    ----------
    b : ip.Interval
        Исходный интервальный вектор (правая часть системы).
    k : float
        Величина расширения каждого интервала вектора b.

    Возвращает
    -------
    ip.Interval
        Новый интервальный вектор b, расширенный на k в обе стороны.
    """
    e = ip.Interval([[-k, k] for _ in range(len(b))])
    return b + e


def find_b_correction_min_K(A, b, eps=1e-2, max_iterations=500):
    """
    Подбирает минимальный коэффициент k (>= 0) для b-коррекции так, чтобы
    допусковое множество стало непустым.

    Алгоритм:
      1) Начинается экспоненциальный рост k, пока система остаётся несовместной
         (Tol < 0).
      2) Когда найдётся верхняя граница, при которой допусковое множество перестанет
         быть пустым, запускается двоичный поиск между предыдущим значением k и
         найденной верхней границей.
      3) Итерации продолжаются, пока не будет достигнута заданная точность eps
         или превышен лимит max_iterations.

    Параметры
    ----------
    A : ip.Interval
        Интервальная матрица (коэффициенты системы).
    b : ip.Interval
        Интервальный вектор (правая часть системы).
    eps : float, optional
        Точность поиска по двоичному алгоритму (по умолчанию 1e-2).
    max_iterations : int, optional
        Максимальное число итераций при двоичном поиске (по умолчанию 500).

    Возвращает
    -------
    tuple
        (corrected_b, found_k, iterations_used), где:
        - corrected_b (ip.Interval): скорректированный вектор b, в котором
          каждый интервал расширен до найденного k;
        - found_k (float): минимальное найденное расширение;
        - iterations_used (int): число итераций, затраченных на поиск.

    Исключения
    ----------
    Exception
        Если даже при больших значениях k не удалось получить ненулевое
        допусковое множество, вызывается исключение.
    """
    prev_k = 0
    cur_k = 0
    iteration = 0
    corrected_b = b
    is_empty, _ = is_tolerance_set_empty(A, corrected_b)

    # Шаг 1: Экспоненциальный рост k
    while is_empty and iteration <= max_iterations:
        prev_k = cur_k
        cur_k = math.exp(iteration)  # экспоненциальное возрастание k
        corrected_b = b_correction(b, cur_k)
        is_empty, _ = is_tolerance_set_empty(A, corrected_b)
        iteration += 1

    # Если и после max_iterations не нашли ненулевое множество, выходим
    if is_empty:
        raise Exception('Could not find K for b-correction')

    # Шаг 2: Двоичный поиск между prev_k и cur_k
    iteration = 0
    while abs(prev_k - cur_k) > eps and iteration <= max_iterations:
        mid_k = (prev_k + cur_k) / 2
        corrected_b = b_correction(b, mid_k)
        is_empty, _ = is_tolerance_set_empty(A, corrected_b)

        if is_empty:
            # Если всё ещё пусто, значит нужно увеличить k
            prev_k = mid_k
        else:
            # Если ненулевое множество получено, уменьшаем k
            cur_k = mid_k

        iteration += 1

    corrected_b = b_correction(b, cur_k)

    return corrected_b, cur_k, iteration


def A_correction(A, b):
    """
    Выполняет A-коррекцию интервальной матрицы A за счёт сужения интервалов.

    Алгоритм:
      1) Находит точку максимума Tol и его значение (max_val).
      2) Исходя из соотношения |T| / (|x1| + ... + |xn|), приближённо определяет
         нижнюю границу сужения.
      3) Верхняя граница берётся как минимум из радиусов интервалов A.
      4) Выбирается среднее (или другое) значение e внутри диапазона для
         окончательного сужения.

    Параметры
    ----------
    A : ip.Interval
        Исходная интервальная матрица.
    b : ip.Interval
        Интервальный вектор (правая часть системы) — необходим для вычисления
        значения Tol и поисковой точки.

    Возвращает
    -------
    ip.Interval
        Новая интервальная матрица, в которой интервалы для каждого элемента
        сужены на величину e.
    """
    max_tol = ip.linear.Tol.maximize(A, b)
    # Точка, в которой достигается максимум Tol
    max_point = [float(v) for v in max_tol[0]]
    # Максимальное значение Tol
    max_val = float(max_tol[1])

    # Приблизительная оценка нижней границы сужения
    # (учитывается сумма модулей координат точки и модуль Tol)
    lower_bound = abs(max_val) / sum(abs(mp) for mp in max_point if mp != 0)

    # Радиусы интервалов матрицы A
    rad_A = ip.rad(A)
    rad_A = np.array(rad_A, dtype=float)

    # Верхняя граница — минимальный радиус среди всех элементов
    upper_bound = float(rad_A.min())

    # В качестве "опорной" величины возьмём среднее
    e = (float(lower_bound) + float(upper_bound)) / 2

    corrected_A = []
    for i in range(len(A)):
        A_i = []
        for j in range(len(A[0])):
            a_low = float(A[i][j]._a)
            a_up = float(A[i][j]._b)
            # Сужаем интервалы за счёт e
            A_i.append([a_low + e, a_up - e])
        corrected_A.append(A_i)

    return ip.Interval(corrected_A)


def plot_tol(axis, A, b):
    """
    Строит поверхностный 3D-график значений функционала Tol(x1, x2) для
    интервалов A, b.

    На осях X и Y откладываются координаты (x1, x2), на оси Z — значение
    Tol(x1, x2). Точка глобального максимума Tol отмечается красным маркером.

    Параметры
    ----------
    axis : matplotlib.axes._subplots.Axes3DSubplot
        3D-ось для рисования (объект matplotlib).
    A : ip.Interval
        Интервальная матрица (коэффициенты системы).
    b : ip.Interval
        Интервальный вектор (правая часть системы).
    """
    max_tol = ip.linear.Tol.maximize(A, b)
    sol = [float(v) for v in max_tol[0]]
    max_val = float(max_tol[1])

    grid_min, grid_max = sol[0] - 2, sol[0] + 2
    x_1_, x_2_ = np.mgrid[grid_min:grid_max:70j, grid_min:grid_max:70j]
    list_x_1 = np.linspace(grid_min, grid_max, 70)
    list_x_2 = np.linspace(grid_min, grid_max, 70)

    list_tol = np.zeros((70, 70))

    # Для каждого (x1, x2) вычисляем Tol как минимум по строкам системы
    for idx_x1, x1 in enumerate(list_x_1):
        for idx_x2, x2 in enumerate(list_x_2):
            x = [x1, x2]
            tol_values = []
            for i in range(len(b)):
                a_values = [float(ip.mid(A[i][j])) for j in range(len(x))]
                sum_ = sum(a_values[j] * x[j] for j in range(len(x)))

                rad_b = float(ip.rad(b[i]))
                mid_b = float(ip.mid(b[i]))
                mag_val = float(abs(mid_b - sum_))
                tol = rad_b - mag_val
                tol_values.append(tol)
            list_tol[idx_x1, idx_x2] = min(tol_values)

    axis.view_init(elev=30, azim=45)
    axis.plot_surface(x_1_, x_2_, list_tol, cmap='plasma')
    # Отмечаем на графике точку (sol[0], sol[1], max_val)
    axis.scatter(sol[0], sol[1], max_val, color='red', s=50)


def plot_tol_functional(axis, A, b):
    """
    Строит проекцию допускового множества в 2D для системы A x = b с использованием
    значений Tol.

    На плоскости (x1, x2) точкам, где Tol >= 0, присваивается одно значение
    (например, 1 — область решения), а где Tol < 0, присваивается другое значение
    (0 — вне решения). Таким образом, визуализируется область допустимых (x1, x2).

    Параметры
    ----------
    axis : matplotlib.axes.Axes
        Ось для рисования (объект matplotlib).
    A : ip.Interval
        Интервальная матрица (коэффициенты системы).
    b : ip.Interval
        Интервальный вектор (правая часть системы).
    """
    max_tol = ip.linear.Tol.maximize(A, b)
    solution = [float(v) for v in max_tol[0]]

    x = np.linspace(solution[0] - 2, solution[0] + 2, 101)
    y = np.linspace(solution[1] - 2, solution[1] + 2, 101)
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros_like(xx)

    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            val = float(ip.linear.Tol.value(A, b, [xx[i, j], yy[i, j]]))
            zz[i, j] = 1 if val >= 0 else 0

    axis.contourf(xx, yy, zz, levels=1, colors=['lightcoral', 'lightgreen'])
    axis.scatter(solution[0], solution[1], color='red', marker='x', s=50)
    axis.set_xlabel('x₁')
    axis.set_ylabel('x₂')


A_1 = ip.Interval([
    [[0.65, 1.25], [0.70, 1.3]],
    [[0.75, 1.35], [0.70, 1.3]]
])
b_1 = ip.Interval([
    [2.75, 3.15],
    [2.85, 3.25],
])

A_2 = ip.Interval([
    [[0.65, 1.25], [0.70, 1.3]],
    [[0.75, 1.35], [0.70, 1.3]],
    [[0.8, 1.4], [0.70, 1.3]],
])
b_2 = ip.Interval([
    [2.75, 3.15],
    [2.85, 3.25],
    [2.90, 3.3],
])

A_3 = ip.Interval([
    [[0.65, 1.25], [0.70, 1.3]],
    [[0.75, 1.35], [0.70, 1.3]],
    [[0.8, 1.4], [0.70, 1.3]],
    [[-0.3, 0.3], [0.70, 1.3]],
])
b_3 = ip.Interval([
    [2.75, 3.15],
    [2.85, 3.25],
    [2.90, 3.3],
    [1.8, 2.2],
])

systems = (
    (A_1, b_1),
    (A_2, b_2),
    (A_3, b_3),
)

fig_raw = plt.figure(figsize=(15, 6))
fig_b_corrected = plt.figure(figsize=(15, 6))
fig_b_corrected_2d = plt.figure(figsize=(15, 5))
fig_a_corrected = plt.figure(figsize=(15, 6))
fig_a_corrected_2d = plt.figure(figsize=(15, 5))
fig_ab_corrected = plt.figure(figsize=(15, 6))
fig_ab_corrected_2d = plt.figure(figsize=(15, 5))

for index, (A, b) in enumerate(systems):
    # Исходная система
    axis_raw = fig_raw.add_subplot(131 + index, projection='3d')
    axis_raw.set_title(f'Формулировка {index + 1}')
    plot_tol(axis_raw, A, b)

    # b-коррекция
    b_corrected = b_correction(b, 1)
    axis_b_corrected = fig_b_corrected.add_subplot(131 + index, projection='3d')
    axis_b_corrected.set_title(f'Формулировка {index + 1}')
    plot_tol(axis_b_corrected, A, b_corrected)
    axis_b_corrected_2d = fig_b_corrected_2d.add_subplot(131 + index)
    axis_b_corrected_2d.set_title(f'Формулировка {index + 1}')
    plot_tol_functional(axis_b_corrected_2d, A, b_corrected)

    # A-коррекция
    A_corrected = A_correction(A, b)
    axis_a_corrected = fig_a_corrected.add_subplot(131 + index, projection='3d')
    axis_a_corrected.set_title(f'Формулировка {index + 1}')
    plot_tol(axis_a_corrected, A_corrected, b)
    axis_a_corrected_2d = fig_a_corrected_2d.add_subplot(131 + index)
    axis_a_corrected_2d.set_title(f'Формулировка {index + 1}')
    plot_tol_functional(axis_a_corrected_2d, A_corrected, b)

    # Ab-коррекция
    axis_ab_corrected = fig_ab_corrected.add_subplot(131 + index, projection='3d')
    axis_ab_corrected.set_title(f'Формулировка {index + 1}')
    plot_tol(axis_ab_corrected, A_corrected, b_corrected)
    axis_ab_corrected_2d = fig_ab_corrected_2d.add_subplot(131 + index)
    axis_ab_corrected_2d.set_title(f'Формулировка {index + 1}')
    plot_tol_functional(axis_ab_corrected_2d, A_corrected, b_corrected)

fig_raw.tight_layout()
fig_b_corrected.tight_layout()
fig_b_corrected_2d.tight_layout()
fig_a_corrected.tight_layout()
fig_a_corrected_2d.tight_layout()
fig_ab_corrected.tight_layout()
fig_ab_corrected_2d.tight_layout()

# Сохранение фигур
os.makedirs('plots', exist_ok=True)
fig_raw.savefig('plots/fig_raw.png', dpi=300)
fig_b_corrected.savefig('plots/fig_b_corrected.png', dpi=300)
fig_b_corrected_2d.savefig('plots/fig_b_corrected_2d.png', dpi=300)
fig_a_corrected.savefig('plots/fig_a_corrected.png', dpi=300)
fig_a_corrected_2d.savefig('plots/fig_a_corrected_2d.png', dpi=300)
fig_ab_corrected.savefig('plots/fig_ab_corrected.png', dpi=300)
fig_ab_corrected_2d.savefig('plots/fig_ab_corrected_2d.png', dpi=300)

# plt.show()
