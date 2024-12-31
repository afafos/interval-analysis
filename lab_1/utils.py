import numpy as np
from interval_module import Interval

def get_interval_matrix(eps: float):
    """
    Возвращает матрицу интервалов 2x2 в зависимости от значения eps.

    [1.05 ± eps,  0.95 ± eps]
    [1.0 ± eps ,  1.0 ± eps ]
    """
    return np.array([
        [Interval(1.05 - eps, 1.05 + eps), Interval(0.95 - eps, 0.95 + eps)],
        [Interval(1 - eps,   1 + eps),    Interval(1 - eps,   1 + eps)]
    ])

def print_matrix_for_latex(matrix, index=0):
    print("\\begin{equation}\n\\text A_%d = \\begin{pmatrix}" % index)
    for items in matrix:
        print("&".join([item.__repr__() for item in items]) + "\\\\\n")
    print("\\end{pmatrix}\n\\end{equation}")

def find_max_middle(matrix):
    """
    Находит максимальное значение среди середины (mid) всех элементов матрицы.
    """
    max_mid = -float("inf")
    for row in matrix:
        for interval in row:
            max_mid = max(max_mid, interval.mid())
    return max_mid

def determ(i, j, matrix):
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

def optimize(left, right, delta) -> (float, float):
    """
    Функция для поиска значения eps, при котором 0 попадает в определитель,
    путём модифицированного метода половинного деления.
    На каждом шаге вычисляется определитель интервальной матрицы и проверяется,
    попадает ли 0 в полученный интервал.

    Параметры:
    -----------
    left : float
        Левая граница отрезка для eps.
    right : float
        Правая граница отрезка для eps.
    delta : float
        Требуемая точность: процесс итераций останавливается, когда (right - left) < delta.

    Возвращает:
    -----------
    tuple
        (right, left, counter)
        где right и left — финальные границы интервала eps после поиска,
        counter — количество итераций.
    """
    counter = 0
    while right - left > delta:
        c = (right + left) / 2
        counter += 1

        # Формируем интервальную матрицу с параметром c
        matrix_tmp = get_interval_matrix(c)
        # Вычисляем её определитель
        interval = determ(0, 0, matrix_tmp)

        if counter < 5 or counter == 34:
            print("-" * 20)
            print(f"$\\delta = {c}$")
            print(f"Number: {counter}:\n{print_matrix_for_latex(matrix_tmp, counter)}"
                  f"\nИтоговый интервал из определителя {interval}")

        # Проверяем, принадлежит ли 0 нашему определителю
        if not 0 in interval:
            left = c
            print("-" * 20 + "\n" + "-" * 20)
            print(f"$\\delta = {c}$")
            print(f"Number: {counter}:\n{print_matrix_for_latex(matrix_tmp, counter)}"
                  f"\nИтоговый интервал из определителя {interval}")
        else:
            right = c

        print(f"Текущие границы [{left}, {right}]")

    return right, left, counter

def determinant_optimization_new(matrix=None, delta=1e-5):
    """
    Функция-обёртка для вычисления оптимального значения eps (минимального delta),
    при котором 0 попадает в интервал определителя матрицы.
    Использует функцию optimize для поиска решения методом бисекции.

    Параметры:
    -----------
    matrix : np.ndarray, optional
        Матрица интервалов. Если не задана, по умолчанию вызывается get_interval_matrix(0).
    delta : float, optional
        Точность, с которой нужно определить значение eps. По умолчанию 1e-5.

    Возвращает:
    -----------
    float
        Найденное значение eps, при котором 0 входит в определитель A.
    """
    if matrix is None:
        matrix = get_interval_matrix(0)

    # Ищем максимальную середину в начальной матрице для установки правой границы поиска
    mid = find_max_middle(matrix)

    eps_curr = mid        # Правая граница
    eps_left_bound = 0    # Левая граница
    counter = 1

    # Запускаем поиск
    eps_curr, eps_left_bound, amount = optimize(eps_left_bound, eps_curr, delta)

    print(f"Кол-во вызовов функции: {counter + 1}")
    return eps_curr


determinant_optimization_new(delta=1e-10)
