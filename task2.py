# =====================================================
# Завдання 2.1. Явне рівняння ламаної
# Дані: вершини ламаної: (-4, -4), (-3, -5), (0, 3), (2, -4)
# За формулою Бернштейна для ламаної (див. приклад у завданні 2.1) маємо:
#
#   y(x) = 1/2 * [ y0 + slope0*(x - x0) + y_{n-1} + slope_{n-1}*(x - x_{n-1}) ]
#          + 1/2 * Σ ( (slope_{k} - slope_{k-1}) * |x - x_k| ),  k = 1,..., n-1
#
# де slope_i = (y_{i+1} - y_i)/(x_{i+1} - x_i) і n = 3 (оскільки є 4 точки).
# =====================================================
import scipy as sp

print("Завдання 2.1. Явне рівняння ламаної")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

# задаємо символьну змінну
x = sp.symbols('x', real=True)

# Варіант 3: координати вершин ламаної
X = [-4, -3, 0, 2]
Y = [-4, -5, 3, -4]
n = len(X) - 1  # n = 3

# Обчислюємо нахили на кожному сегменті
slope = []
for i in range(n):
    slope.append((Y[i + 1] - Y[i]) / (X[i + 1] - X[i]))

# Обчислюємо суму різниць нахилів
sum_term = 0
for k in range(1, n):
    diff = slope[k] - slope[k - 1]
    sum_term += diff * sp.Abs(x - X[k])

expr = 0.5 * (Y[0] + slope[0] * (x - X[0]) + Y[n - 1] + slope[n - 1] * (x - X[n - 1])) + 0.5 * sum_term
expr = sp.simplify(expr)
print("Явне рівняння ламаної (Формула Бернштейна):")
sp.pprint(expr)

# Побудова графіка
f_expr = sp.lambdify(x, expr, "numpy")
x_vals = np.linspace(-6, 4, 400)
y_vals = f_expr(x_vals)

plt.figure(figsize=(6, 4))
plt.plot(x_vals, y_vals, 'b-', label='Ламана (явне рівняння)')
plt.plot(X, Y, 'ro', markersize=8, label='Вузли')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Завдання 2.1. Явне рівняння ламаної")
plt.legend()
plt.grid(True)
plt.show()

# =====================================================
# Завдання 2.2. Формульне представлення неперервної кускової функції
# =====================================================
print("Завдання 2.2. Формульне представлення неперервної кускової функції")
# Визначаємо кускову функцію
f_piece = sp.Piecewise(
    (-1 - x, x <= -1),
    (x * (x + 1), (x > -1) & (x <= 0)),
    (x * (1 - x), (x > 0) & (x <= 1)),
    (1 - x, x > 1)
)
print("Неперервна кускова функція:")
sp.pprint(f_piece)

# Задано точки розбиття (для варіанту 3)
x0 = -1
x1 = 0
x2 = 1

# Задані функції f0, f1, f2, f3:
# При x ≤ x0: f0(x) = -1 - x
# На проміжку (x0, x1]: f1(x) = x*(x + 1)
# На проміжку (x1, x2]: f2(x) = x*(x - 1)
# При x ≥ x2: f3(x) = 1 - x
f0 = lambda t: -1 - t
f1 = lambda t: t * (t + 1)
f2 = lambda t: t * (1 - t)
f3 = lambda t: 1 - t

# Допоміжні функції:
# Qₗ(x, a) = (x - a - |x - a|)/2
Ql = lambda xx, a: (xx - a - sp.Abs(xx - a)) / 2
# Q(x, a) = (x - a + |x - a|)/2
Q = lambda xx, a: (xx - a + sp.Abs(xx - a)) / 2
# Π(x, a, w) = (w + |x - a| - |(x - a) - w|)/2, де w > 0
Pi = lambda xx, a, w: (w + sp.Abs(xx - a) - sp.Abs(xx - a - w)) / 2

# Побудова формульного виразу f(x) за схемою:
# f(x) = f0(x0 + Ql(x, x0))
#       + [f1(x0 + Π(x, x0, x1 - x0)) - f1(x0)]
#       + [f2(x1 + Π(x, x1, x2 - x1)) - f2(x1)]
#       + [f3(x2 + Q(x, x2)) - f3(x2)]
expr = f0(x0 + Ql(x, x0)) \
       + (f1(x0 + Pi(x, x0, x1 - x0)) - f1(x0)) \
       + (f2(x1 + Pi(x, x1, x2 - x1)) - f2(x1)) \
       + (f3(x2 + Q(x, x2)) - f3(x2))

# Спрощення виразу (за бажанням)
expr_simpl = sp.simplify(expr)

# -------------------------------
# 3. Формула Гевісайда (Heaviside)
# -------------------------------
# Позначимо f1(x) = -1 - x, f2(x) = x(x+1), f3(x) = x(x-1), f4(x) = 1 - x.
# Границі (точки переходу) x1 = -1, x2 = 0, x3 = 1.
# Тоді:
#   f_H(x) = f1(x)
#          + [f2(x) - f1(x)]*H(x - x1)
#          + [f3(x) - f2(x)]*H(x - x2)
#          + [f4(x) - f3(x)]*H(x - x3),
# де H(...) – функція Гевісайда.

Heaviside = sp.Heaviside  # Символьна функція Гевісайда в Sympy


def f1_local(xx):  # для x <= -1
    return -1 - xx


def f2_local(xx):  # для -1 < x <= 0
    return xx * (xx + 1)


def f3_local(xx):  # для 0 < x <= 1
    return xx * (1 - xx)


def f4_local(xx):  # для x > 1
    return 1 - xx


x_b1, x_b2, x_b3 = -1, 0, 1

f_heaviside = (
        f1_local(x)
        + (f2_local(x) - f1_local(x)) * Heaviside(x - x_b1)
        + (f3_local(x) - f2_local(x)) * Heaviside(x - x_b2)
        + (f4_local(x) - f3_local(x)) * Heaviside(x - x_b3)
)

f_heaviside_simpl = sp.simplify(f_heaviside)

# -------------------------------
# Перетворення символьних виразів у функції (для обчислень NumPy)
# -------------------------------
f_piece_func = sp.lambdify(x, f_piece, "numpy")
expr_func = sp.lambdify(x, expr_simpl, "numpy")
f_heaviside_func = sp.lambdify(x, f_heaviside_simpl, "numpy")

# -------------------------------
# Побудова графіків
# -------------------------------
x_vals = np.linspace(-2, 2, 500)
y_piece = f_piece_func(x_vals)
y_expr = expr_func(x_vals)
y_heaviside = f_heaviside_func(x_vals)

plt.figure(figsize=(7, 5))
# 3) Формула Гевісайда (зелений)
plt.plot(x_vals, y_heaviside, color='green', label='Heaviside', linewidth=6)
# 2) Формульне представлення (синій)
plt.plot(x_vals, y_expr, color='blue', label='Формула Q, Ql, Π', linewidth=4)
# 1) Piecewise (фіолетовий)
plt.plot(x_vals, y_piece, 'm-', label='Piecewise', linewidth=2)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("2.2 Кускова функція: 3 різні способи задання")
plt.grid(True)
plt.legend()
plt.show()

# Друк спрощених виразів (за бажанням):
print("=== Piecewise (спрощення не застосовуємо, це сам Piecewise) ===")
sp.pprint(f_piece)
print("\n=== Формульне представлення (Q, Ql, П) ===")
sp.pprint(expr_simpl)
print("\n=== Формула Гевісайда ===")
sp.pprint(f_heaviside_simpl)

# =====================================================
# Завдання 2.3. Неперервні кусково-поліноміальні функції загального вигляду
# =====================================================
# Визначення поліномів на кожному відрізку:
p1 = 1 - (x + 1) ** 2  # для x <= 0
p2 = (x - 1) ** 2 - 1  # для x > 0

# Piecewise представлення:
p_piece = sp.Piecewise((p1, x <= 0), (p2, x > 0))

print("Неперервна кусково-поліноміальна функція (piecewise):")
sp.pprint(p_piece)
print("\n")

# Побудова глобального представлення за формулою (2)
# Формула (3): P(x) = 1/2*(p1(x)+p2(x))
P = sp.simplify((p1 + p2) / 2)
# Обчислення P1(x) для x != 0: (p2(x)-P(x))/x
P1_expr = sp.simplify((p2 - P) / x)
# Глобальне представлення: P(x) + P1(x)*|x|
P_global = sp.simplify(P + P1_expr * sp.Abs(x))

print("Поліном P(x) (за формулою (3)):")
sp.pprint(P)
print("\nПоліном P1(x) (знаходиться як (p2-P)/x):")
sp.pprint(P1_expr)
print("\nГлобальне представлення кускового полінома (формула (2)):")
sp.pprint(P_global)

# Перетворення виразів у числові функції для побудови графіків
p_piece_func = sp.lambdify(x, p_piece, "numpy")
P_global_func = sp.lambdify(x, P_global, "numpy")

# Генерація значень x для графіка
x_vals = np.linspace(-3, 3, 400)
y_piece = p_piece_func(x_vals)
y_global = P_global_func(x_vals)

# Побудова графіків обох представлень
plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_piece, 'b-', label='Piecewise представлення')
plt.plot(x_vals, y_global, 'r--', label='Глобальна формула (2)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('2.3 Порівняння представлень кускового полінома')
plt.legend()
plt.grid(True)
plt.show()

# =====================================================
# Завдання 2.4. Рівняння кубічного сплайну (Вар. 3)
# Дані:
#   X = [-4, -3, 0, 2], Y = [-4, -5, 3, -4]
# Побудуємо два сплайни:
#   (a) кубічний сплайн із затисненням (clamped): f'(x0)=f'(xn)=1,
#   (b) натуральний сплайн: f''(x0)=f''(xn)=0.
# Використовуємо scipy.interpolate.CubicSpline [&#8203;:contentReference[oaicite:3]{index=3}].
# TODO єдиний формульний вираз
# =====================================================
print("Завдання 2.4. Кубічні сплайни")

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


# =====================================================
# Функція обчислення коефіцієнтів кубічного сплайну
# =====================================================
def compute_cubic_spline(x, y, bc_type='natural'):
    n = len(x) - 1
    h = np.diff(x)
    A = np.zeros((n + 1, n + 1))
    b_vec = np.zeros(n + 1)

    # Система для внутрішніх вузлів (i = 1 .. n-1)
    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        b_vec[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    # Задаємо крайові умови
    if bc_type == 'natural':
        A[0, 0] = 1.0
        A[n, n] = 1.0
        b_vec[0] = 0.0
        b_vec[n] = 0.0
    elif bc_type == 'clamped':
        # Для clamped сплайну f'(x₀)=f'(xₙ)=1
        A[0, 0] = 2 * h[0]
        A[0, 1] = h[0]
        A[n, n - 1] = h[n - 1]
        A[n, n] = 2 * h[n - 1]
        b_vec[0] = 3 * ((y[1] - y[0]) / h[0] - 1)
        b_vec[n] = 3 * (1 - (y[n] - y[n - 1]) / h[n - 1])
    else:
        raise ValueError("bc_type має бути 'natural' або 'clamped'")

    # Розв'язуємо систему для c (коефіцієнти, пов'язані з похідними)
    c = np.linalg.solve(A, b_vec)

    # Обчислюємо коефіцієнти a, b та d для кожного інтервалу
    a_coef = y[:-1]  # значення функції в початкових точках інтервалів
    b_coef = np.zeros(n)
    d_coef = np.zeros(n)
    for i in range(n):
        b_coef[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 3
        d_coef[i] = (c[i + 1] - c[i]) / (3 * h[i])

    return a_coef, b_coef, c[:-1], d_coef, x


# =====================================================
# Функція обчислення значень сплайну для побудови графіку
# =====================================================
def evaluate_spline(x_eval, coeffs):
    a_coef, b_coef, c_coef, d_coef, knots = coeffs
    y_eval = np.zeros_like(x_eval)
    n = len(a_coef)
    for j, x_val in enumerate(x_eval):
        # Знаходимо відповідний інтервал
        i = np.searchsorted(knots, x_val) - 1
        if i < 0:
            i = 0
        elif i >= n:
            i = n - 1
        dx = x_val - knots[i]
        y_eval[j] = a_coef[i] + b_coef[i] * dx + c_coef[i] * dx ** 2 + d_coef[i] * dx ** 3
    return y_eval


# =====================================================
# Основна частина програми
# =====================================================

# Задана множина опорних точок
X_arr = np.array([-4, -3, 0, 2], dtype=float)
Y_arr = np.array([-4, -5, 3, -4], dtype=float)

# Обчислення коефіцієнтів для сплайнів з двома крайовими умовами
coeffs_natural = compute_cubic_spline(X_arr, Y_arr, bc_type='natural')
coeffs_clamped = compute_cubic_spline(X_arr, Y_arr, bc_type='clamped')

# Виведення символьних формул
print("Символьний єдиний формульний вираз для кубічного сплайну (natural):")
sp.pprint(sym_expr_natural)
print("\nСимвольний єдиний формульний вираз для кубічного сплайну (clamped):")
sp.pprint(sym_expr_clamped)

# Побудова графіка з обома сплайнами
x_vals = np.linspace(X_arr[0] - 1, X_arr[-1] + 1, 400)
y_natural = evaluate_spline(x_vals, coeffs_natural)
y_clamped = evaluate_spline(x_vals, coeffs_clamped)

plt.figure(figsize=(8, 5))
plt.plot(x_vals, y_natural, 'g--', label="Natural Spline")
plt.plot(x_vals, y_clamped, 'b-', label="Clamped Spline")
plt.plot(X_arr, Y_arr, 'ro', markersize=8, label="Опорні точки")
plt.xlabel("x")
plt.ylabel("S(x)")
plt.title("Кубічні сплайни: natural та clamped")
plt.legend()
plt.grid(True)
plt.show()


# =====================================================
# Завдання 2.5. Кусково-кубічний поліном Ерміта
# Дані: ті ж вузли: X = [-1, 0, 1, 2], Y = [2, 1, 2, 5]
#  похідні f'(xi) = [-1,-1,3,3].
# =====================================================

# Функція для обчислення кубічного сегменту Ерміта на [x_{i-1}, x_i]
def hermite_segment(x_sym, x0, x1, y0, y1, g0, g1):
    h = x1 - x0
    t = (x_sym - x0) / h
    # Стандартна формула кубічного Ерміта:
    return (2 * t ** 3 - 3 * t ** 2 + 1) * y0 + (t ** 3 - 2 * t ** 2 + t) * h * g0 + (-2 * t ** 3 + 3 * t ** 2) * y1 + (t ** 3 - t ** 2) * h * g1


# Функція для генерації уніфікованого символьного виразу для кусково-кубічного полінома Ерміта
# згідно з формулою (7)–(8)
def generate_symbolic_unified_hermite_formula(x_sym, xp, yp, dyp):
    # Кількість сегментів = len(xp) - 1
    n = len(xp) - 1

    # Обчислюємо p1(x) – першу поліноміальну ланку на [x0, x1]
    p1_expr = hermite_segment(x_sym, xp[0], xp[1], yp[0], yp[1], dyp[0], dyp[1])

    # Обчислюємо pn(x) – останню поліноміальну ланку на [x_{n-1}, x_n]
    pn_expr = hermite_segment(x_sym, xp[n - 1], xp[n], yp[n - 1], yp[n], dyp[n - 1], dyp[n])

    # Обчислюємо суму для внутрішніх вузлів (i = 1,..., n-1)
    interior_sum = 0
    # Для i від 1 до n-1:
    # (зауважимо, що якщо вузлів 4, то n = 3, а i пробігає 1, 2 – внутрішні вузли)
    for i in range(1, n):
        h_left = xp[i] - xp[i - 1]
        h_right = xp[i + 1] - xp[i]
        # Формуємо окремі частини для формули (8)
        term1 = (yp[i + 1] - yp[i]) / (h_right ** 2)
        term2 = (yp[i] - yp[i - 1]) / (h_left ** 2)
        term3 = (dyp[i + 1] + 2 * dyp[i]) / (h_right)
        term4 = (dyp[i - 1] + 2 * dyp[i]) / (h_left)
        term5 = (dyp[i] + dyp[i + 1]) / (h_right ** 2)
        term6 = 2 * (yp[i + 1] - yp[i]) / (h_right ** 3)
        term7 = 2 * (yp[i] - yp[i - 1]) / (h_left ** 3)
        term8 = (dyp[i - 1] + dyp[i]) / (h_left ** 2)
        # Формула (8)
        P1_i = sp.Rational(1, 2) * (3 * (term1 + term2) - (term3 + term4) + (term5 - term6 + term7 - term8) * (x_sym - xp[i]))
        # Додаємо внесок для вузла xi: P1_i(x)*(x - xi)*|x - xi|
        interior_sum += P1_i * (x_sym - xp[i]) * sp.Abs(x_sym - xp[i])

    # Остаточний уніфікований вираз згідно з формулою (7)
    unified_expr = sp.Rational(1, 2) * (p1_expr + pn_expr) + interior_sum
    unified_expr = sp.simplify(unified_expr)
    return unified_expr


# ===============================
# Основна частина
# ===============================
print("Завдання 2.5. Кусково-кубічний поліном Ерміта (уніфікований запис)")

# Вхідні дані
xp = [-1, 0, 1, 2]  # вузли
yp = [2, 1, 2, 5]  # значення функції
dyp = [-1, -1, 3, 3]  # значення похідних


# Отримуємо Piecewise-функцію з лінійним продовженням (опціонально, як базовий приклад)
def universal_hermite_linear_extension(x_sym, xp, yp, dyp):
    def hermite_seg(x_sym, x0, x1, y0, y1, dy0, dy1):
        h = x1 - x0
        t = (x_sym - x0) / h
        h00 = 2 * t ** 3 - 3 * t ** 2 + 1
        h10 = t ** 3 - 2 * t ** 2 + t
        h01 = -2 * t ** 3 + 3 * t ** 2
        h11 = t ** 3 - t ** 2
        return h00 * y0 + h10 * h * dy0 + h01 * y1 + h11 * h * dy1

    n = len(xp)
    segments = []
    # Будуємо сегменти для кожного інтервалу
    for i in range(n - 1):
        seg_expr = hermite_seg(x_sym, xp[i], xp[i + 1], yp[i], yp[i + 1], dyp[i], dyp[i + 1])
        cond = (x_sym >= xp[i]) & (x_sym <= xp[i + 1])
        segments.append((sp.simplify(seg_expr), cond))
    # Лінійне продовження вліво
    left_line = yp[0] + dyp[0] * (x_sym - xp[0])
    segments.insert(0, (left_line, x_sym < xp[0]))
    # Лінійне продовження вправо
    right_line = yp[-1] + dyp[-1] * (x_sym - xp[-1])
    segments.append((right_line, x_sym > xp[-1]))

    return sp.simplify(sp.Piecewise(*segments))


H_piecewise = universal_hermite_linear_extension(x, xp, yp, dyp)
print("\nPiecewise‑функція з лінійним продовженням:")
sp.pprint(H_piecewise)

# Генеруємо єдиний символьний формульний вираз за формулами (7)–(8)
unified_H_expr = generate_symbolic_unified_hermite_formula(x, xp, yp, dyp)
print("\nЄдиний символьний формульний вираз для полінома Ерміта:")
sp.pretty_print(unified_H_expr)

# Перетворюємо символьний вираз на функцію для побудови графіку
H_func_unified = sp.lambdify(x, unified_H_expr, 'numpy')

# Точки для графіка
xx = np.linspace(xp[0] - 1, xp[-1] + 1, 400)
yy_unified = H_func_unified(xx)

plt.figure(figsize=(8, 6))
plt.plot(xx, yy_unified, 'b', label="Unified Hermite polynomial")
plt.scatter(xp, yp, color='r', zorder=5, label="Вузли")
plt.xlabel("x")
plt.ylabel("H(x)")
plt.title("Кусково-кубічний поліном Ерміта (уніфікований запис)")
plt.grid(True)
plt.legend()
plt.show()
# =====================================================
# Завдання 2.6.1. Формульне представлення параметричних рівнянь ламаної
# Дані: вершини в площині: [(-1, 0), (-3, 0), (0, 2), (3, 0), (1, 0), (1, -2), (-1, -2), (-1, 0)]
# Побудувати символьні параметричні рівняння кожного сегменту (лінійна інтерполяція)
# та зобразити графік ламаної.
# =====================================================
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

print("Завдання 2.6.1. Параметрична ламана")
# Визначення символьної змінної параметра t
t = sp.symbols('t', real=True)


# Допоміжна функція P(t, a, w)
def P_func(t_sym, a, w):
    return (1 / (2 * w)) * (w + sp.Abs(t_sym - a) - sp.Abs(t_sym - a - w))


# Задано вершини ламаної (радіус-вектори) у площині
# (x, y)
points = [(-1, 0), (-3, 0), (0, 2), (3, 0), (1, 0), (1, -2), (-1, -2), (-1, 0)]

# Призначаємо вузлам монотонно зростаючі значення параметра:
# t0, t1, ..., t7 (тут беремо t_i = i)
t_vals = list(range(len(points)))  # [0, 1, 2, ..., 7]

# Побудова єдиних символьних формул для x(t) та y(t)
# x(t) = x0 + ∑(x_i - x_{i-1}) * P(t, t_{i-1}, t_i-t_{i-1})
# y(t) = y0 + ∑(y_i - y_{i-1}) * P(t, t_{i-1}, t_i-t_{i-1})

# Для x(t)
x0 = points[0][0]
unified_x_expr = sp.Float(x0)
for i in range(1, len(points)):
    delta_x = points[i][0] - points[i - 1][0]
    a = t_vals[i - 1]
    w_val = t_vals[i] - t_vals[i - 1]  # Тут w_val = 1, але формула загальна
    unified_x_expr += delta_x * P_func(t, a, w_val)

# Для y(t)
y0 = points[0][1]
unified_y_expr = sp.Float(y0)
for i in range(1, len(points)):
    delta_y = points[i][1] - points[i - 1][1]
    a = t_vals[i - 1]
    w_val = t_vals[i] - t_vals[i - 1]
    unified_y_expr += delta_y * P_func(t, a, w_val)

# Спрощуємо вирази
unified_x_expr = sp.simplify(unified_x_expr)
unified_y_expr = sp.simplify(unified_y_expr)

print("Єдина символьна формула для x(t):")
sp.pretty_print(unified_x_expr)
print("\nЄдина символьна формула для y(t):")
sp.pretty_print(unified_y_expr)

# Перетворюємо символьні вирази у функції для чисельного обчислення
x_func = sp.lambdify(t, unified_x_expr, 'numpy')
y_func = sp.lambdify(t, unified_y_expr, 'numpy')

# Обираємо сітку значень параметра t
t_min = t_vals[0]
t_max = t_vals[-1]
t_array = np.linspace(t_min, t_max, 400)

# Обчислюємо координати x та y
x_vals = x_func(t_array)
y_vals = y_func(t_array)

# Побудова графіка ламаної
plt.figure(figsize=(6, 4))
plt.plot(x_vals, y_vals, 'b-', label='Уніфікована ламана')
# Позначаємо задані вершини
vertices_x = [pt[0] for pt in points]
vertices_y = [pt[1] for pt in points]
plt.plot(vertices_x, vertices_y, 'ro', markersize=8, label='Вузли')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Параметрична ламана (уніфікований запис)")
plt.legend()
plt.grid(True)
plt.show()

# =====================================================
# Завдання 2.6.2. Параметричне рівняння просторової ламаної, що формує контур тетраедра
# Дані: координати вершин тетраедра: A, B, C, D.
# Ламана проходить послідовно через вершини: A -> B -> C -> A -> D -> C -> B -> D
# (відрізок BC використовується двічі в різних напрямках).
# =====================================================
print("Завдання 2.6.2. Параметричне рівняння просторової ламаної")
# Визначення символьної змінної параметра t
t = sp.symbols('t', real=True)


# Допоміжна функція P(t, a, w)
def P_func(t_sym, a, w):
    return (1 / (2 * w)) * (w + sp.Abs(t_sym - a) - sp.Abs(t_sym - a - w))


# Задано вершини (радіус-вектори) просторової ламаної
# Приклад: вершини тетраедра, задані у певній послідовності
A = (7, 2, 4)
B = (7, -1, -2)
C = (3, 3, 1)
D = (-4, 2, 1)
# Послідовність вершин (можна задати довільну послідовність)
points = [A, B, C, A, D, C, B, D]
vertex_labels = ['A', 'B', 'C', 'A', 'D', 'C', 'B', 'D']
# Призначаємо вузлам параметричні значення t0, t1, ..., t7 (монотонно зростають, наприклад, 0, 1, 2, ...)
t_vals = list(range(len(points)))  # [0, 1, 2, ..., 7]

# Обчислення єдиного символьного виразу для кожної координати:
# x(t) = x0 + Σ (x_i - x_{i-1})*P(t, t_{i-1}, t_i-t_{i-1})
# аналогічно для y(t) та z(t)
x0 = points[0][0]
y0 = points[0][1]
z0 = points[0][2]

unified_x_expr = sp.Float(x0)
unified_y_expr = sp.Float(y0)
unified_z_expr = sp.Float(z0)

for i in range(1, len(points)):
    a_val = t_vals[i - 1]
    w_val = t_vals[i] - t_vals[i - 1]  # тут за замовчуванням w_val = 1, але формула універсальна
    delta_x = points[i][0] - points[i - 1][0]
    delta_y = points[i][1] - points[i - 1][1]
    delta_z = points[i][2] - points[i - 1][2]

    unified_x_expr += delta_x * P_func(t, a_val, w_val)
    unified_y_expr += delta_y * P_func(t, a_val, w_val)
    unified_z_expr += delta_z * P_func(t, a_val, w_val)

# Спрощення символьних виразів
unified_x_expr = sp.simplify(unified_x_expr)
unified_y_expr = sp.simplify(unified_y_expr)
unified_z_expr = sp.simplify(unified_z_expr)

print("Єдина символьна формула для x(t):")
sp.pretty_print(unified_x_expr)
print("\nЄдина символьна формула для y(t):")
sp.pretty_print(unified_y_expr)
print("\nЄдина символьна формула для z(t):")
sp.pretty_print(unified_z_expr)

# Перетворення символьних виразів у функції для чисельного обчислення
x_func = sp.lambdify(t, unified_x_expr, 'numpy')
y_func = sp.lambdify(t, unified_y_expr, 'numpy')
z_func = sp.lambdify(t, unified_z_expr, 'numpy')

# Побудова сітки значень параметра t
t_min = t_vals[0]
t_max = t_vals[-1]
t_array = np.linspace(t_min, t_max, 400)

# Обчислення координат для ламаної
x_vals = x_func(t_array)
y_vals = y_func(t_array)
z_vals = z_func(t_array)

# Побудова 3D графіка ламаної
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_vals, y_vals, z_vals, 'b-', label='Уніфікована просторовa ламана')

# Позначення заданих вершин
vertices_x = [pt[0] for pt in points]
vertices_y = [pt[1] for pt in points]
vertices_z = [pt[2] for pt in points]
ax.scatter(vertices_x, vertices_y, vertices_z, c='r', s=50, label='Вузли')
# Додаємо підписи до кожної вершини
for (x_v, y_v, z_v), label in zip(points, vertex_labels):
    ax.text(x_v, y_v, z_v, f' {label}', color='black', fontsize=12)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Завдання 2.6.2. Параметричне рівняння просторової ламаної")
ax.legend()
plt.show()
# =====================================================
# Завдання 2.7. Параметричні кубічні сплайни (Вар. 3)
# Дані: для варіанту 3 обираємо точки:
#   [(-2, 2), (4, 3), (1, -3), (-4, -5)]
# Для замкнутої кривої повторимо першу точку.
# =====================================================
print("Завдання 2.7. Параметричні кубічні сплайни (замкнута крива, Вар. 3)")


# ===============================
# 1. Обчислення коефіцієнтів кубічного сплайну для одновимірних даних
# (модифіковано для повернення вектора других похідних c усіх вузлів)
# ===============================
def compute_cubic_spline(x, y, bc_type='natural'):
    """
    Обчислює коефіцієнти кубічного сплайну для заданих вузлів x та значень y.
    bc_type може бути:
      - 'natural' (натуральний сплайн),
      - 'clamped' (затиснутий),
      - 'closed' (замкнений).
    Повертає кортеж:
      (a, b, c, d, x)
    де:
      a[i] = y[i] (для i = 0, …, n-1),
      b, c, d – коефіцієнти для кожного сегмента,
      c – вектор других похідних у всіх вузлах (довжини n+1).
    """
    n = len(x) - 1
    h = np.diff(x)
    A = np.zeros((n + 1, n + 1))
    b_vec = np.zeros(n + 1)

    # Заповнюємо систему для внутрішніх вузлів (i = 1 .. n-1)
    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        b_vec[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    # Задаємо граничні умови
    if bc_type == 'natural':
        A[0, 0] = 1.0
        A[n, n] = 1.0
        b_vec[0] = 0.0
        b_vec[n] = 0.0
    elif bc_type == 'clamped':
        A[0, 0] = 2 * h[0]
        A[0, 1] = h[0]
        A[n, n - 1] = h[n - 1]
        A[n, n] = 2 * h[n - 1]
        b_vec[0] = 3 * ((y[1] - y[0]) / h[0] - 1)
        b_vec[n] = 3 * (1 - (y[n] - y[n - 1]) / h[n - 1])
    elif bc_type == 'closed':
        # Замкнений сплайн: періодичні умови: s0 = s_n, f(x0)=f(xn)
        A[0, 0] = 1
        A[0, n] = -1
        b_vec[0] = 0

        # Останній рядок: зв'язок між першою та останньою ділянками
        A[-1, 0] = -2 * h[1]
        A[-1, 1] = -h[1]
        A[-1, -1 - 1] = -h[-1]
        A[-1, -1] = -2 * h[-1]
        b_vec[-1] = 3 * (((y[-1] - y[-1 - 1]) / h[-1]) - ((y[1] - y[0]) / h[1]))
    else:
        raise ValueError("bc_type має бути 'natural', 'clamped' або 'closed'")

    # Розв'язуємо систему для c (другі похідні)
    c = np.linalg.solve(A, b_vec)

    # Обчислюємо коефіцієнти a, b, d для кожного інтервалу
    a_coeff = y[:-1]  # f(x_i)
    b_coeff = np.zeros(n)
    d_coeff = np.zeros(n)
    for i in range(n):
        b_coeff[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 3.0
        d_coeff[i] = (c[i + 1] - c[i]) / (3.0 * h[i])
    return a_coeff, b_coeff, c, d_coeff, x


# ===============================
# 2. Уніфікований символьний запис для параметричних кубічних сплайнів
# згідно з формулою:
#
# S(t) = 1/2 ( p1(t) + pn(t) ) +
#        (1/12) * Σ{i=1}^{n-1} [ (r''_{i+1}-r''_i)/(t_{i+1}-t_i) - (r''_i-r''_{i-1})/(t_i-t_{i-1}) ] * |t-t_i|^3
#
# де для кожної координати (x або y):
#
# p(t) = (t_{i+1}-t)^3/(6h) * s_i + (t-t_i)^3/(6h) * s_{i+1}
#        + ((y_i)/h - h*s_i/6)*(t_{i+1}-t) + ((y_{i+1})/h - h*s_{i+1}/6)*(t-t_i)
#
# ===============================
def unified_symbolic_spline_expr(t_sym, t_arr, y_arr, s):
    """
    Будує уніфікований символьний вираз для сплайну y(t)
    згідно з формулою параметричних кубічних сплайнів.

    Вхідні дані:
      t_sym - символьна змінна (sp.symbols('t'))
      t_arr - масив параметрів (вузлових значень t)
      y_arr - масив значень координати (x або y) у вузлах
      s     - масив других похідних (s_i) у вузлах, обчислених сплайном
    """
    n = len(t_arr) - 1  # кількість сегментів
    # p1(t): поліном для першого сегмента (i = 0, використовуючи вузли 0 та 1)
    t0 = t_arr[0]
    t1 = t_arr[1]
    h0 = t1 - t0
    y0 = y_arr[0]
    y1 = y_arr[1]
    p1 = ((t1 - t_sym) ** 3 / (6 * h0)) * s[0] + ((t_sym - t0) ** 3 / (6 * h0)) * s[1] \
         + ((y0) / h0 - h0 * s[0] / 6) * (t1 - t_sym) + ((y1) / h0 - h0 * s[1] / 6) * (t_sym - t0)

    # pn(t): поліном для останнього сегмента (i = n-1, використовуючи вузли n-1 та n)
    tn_minus1 = t_arr[n - 1]
    tn = t_arr[n]
    h_last = tn - tn_minus1
    y_n_minus1 = y_arr[n - 1]
    y_n = y_arr[n]
    pn = ((tn - t_sym) ** 3 / (6 * h_last)) * s[n - 1] + ((t_sym - tn_minus1) ** 3 / (6 * h_last)) * s[n] \
         + ((y_n_minus1) / h_last - h_last * s[n - 1] / 6) * (tn - t_sym) + ((y_n) / h_last - h_last * s[n] / 6) * (t_sym - tn_minus1)

    sum_term = 0
    # Сума по внутрішніх вузлах: i = 1, 2, ..., n-1
    for i in range(1, n):
        # Різниця других похідних у сусідніх вузлах
        term = ((s[i + 1] - s[i]) / (t_arr[i + 1] - t_arr[i]) - (s[i] - s[i - 1]) / (t_arr[i] - t_arr[i - 1]))
        sum_term += term * sp.Abs(t_sym - t_arr[i]) ** 3
    unified_expr = sp.Rational(1, 2) * (p1 + pn) + sp.Rational(1, 12) * sum_term
    return sp.simplify(unified_expr)


# ===============================
# 3. Основна частина: побудова параметричних сплайнів
# ===============================

# Приклад: Задані точки (радіус-вектори) у площині
# Для прикладу беремо замкнену криву: задано 4 точки, остання повторюється
points = [(-2, 2), (4, 3), (1, -3), (-4, -5)]
points_closed = points + [points[0]]

# Призначаємо параметр t для кожного вузла (монотонно зростаючі значення)
# Наприклад, рівномірно: t = 0, 1, 2, 3, 4
t_arr = np.linspace(0, len(points_closed) - 1, len(points_closed))
# Масиви координат
x_points = np.array([p[0] for p in points_closed], dtype=float)
y_points = np.array([p[1] for p in points_closed], dtype=float)

# Обчислюємо коефіцієнти сплайнів для x та y з використанням замкнених граничних умов
coeffs_x = compute_cubic_spline(t_arr, x_points, bc_type='closed')
coeffs_y = compute_cubic_spline(t_arr, y_points, bc_type='closed')
# coeffs_x, coeffs_y мають вигляд (a, b, c, d, t_arr)
# c – вектор других похідних для відповідної координати

# Отримуємо масиви других похідних для x та y
s_x = coeffs_x[2]  # c_x
s_y = coeffs_y[2]  # c_y

# Визначаємо символьну змінну параметра t
t_sym = sp.symbols('t', real=True)

# Генеруємо уніфіковані символьні формули для x(t) та y(t)
unified_x_expr = unified_symbolic_spline_expr(t_sym, t_arr, x_points, s_x)
unified_y_expr = unified_symbolic_spline_expr(t_sym, t_arr, y_points, s_y)

print("Уніфікована символьна формула для x(t):")
sp.pretty_print(unified_x_expr)
print("\nУніфікована символьна формула для y(t):")
sp.pretty_print(unified_y_expr)

# Перетворюємо символьні вирази у числові функції за допомогою lambdify
x_func = sp.lambdify(t_sym, unified_x_expr, 'numpy')
y_func = sp.lambdify(t_sym, unified_y_expr, 'numpy')

# Створюємо щільну сітку значень параметра t для побудови графіку
t_dense = np.linspace(t_arr[0], t_arr[-1], 400)
x_dense = x_func(t_dense)
y_dense = y_func(t_dense)

# Побудова графіку параметричної кривої
plt.figure(figsize=(8, 8))
plt.plot(x_dense, y_dense, 'b-', linewidth=2, label='Параметричний сплайн')
plt.plot(x_points, y_points, 'ro', markersize=8, label='Вузли')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Параметричні кубічні сплайни (уніфікований запис)")
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()
