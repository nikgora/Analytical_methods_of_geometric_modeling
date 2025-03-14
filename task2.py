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
# 1) Piecewise (фіолетовий)
plt.plot(x_vals, y_piece, 'm-', label='Piecewise', linewidth=2)
# 2) Формульне представлення (синій)
plt.plot(x_vals, y_expr, color='blue', label='Формула Q, Ql, Π', linewidth=4)
# 3) Формула Гевісайда (зелений)
plt.plot(x_vals, y_heaviside, color='green', label='Heaviside', linewidth=6)

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
# =====================================================
print("Завдання 2.4. Кубічні сплайни")


def compute_cubic_spline(x, y, bc_type='natural'):
    """
    Обчислює коефіцієнти кубічного сплайну.

    Параметри:
      x, y    – масиви опорних точок.
      bc_type – тип крайових умов:
                 'natural' – натуральний сплайн (f''(x₀)=f''(xₙ)=0),
                 'clamped' – сплайн із затисненням (f'(x₀)=f'(xₙ)=1).

    Повертає кортеж:
      (a_coef, b_coef, c_coef, d_coef, knots)
    де для i-го інтервалу [xᵢ, xᵢ₊₁]:
      Sᵢ(x) = a_coef[i] + b_coef[i]*(x - xᵢ) + c_coef[i]*(x - xᵢ)² + d_coef[i]*(x - xᵢ)³.
    """
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
        # Натуральний сплайн: f''(x₀) = f''(xₙ) = 0
        A[0, 0] = 1.0
        A[n, n] = 1.0
        b_vec[0] = 0.0
        b_vec[n] = 0.0
    elif bc_type == 'clamped':
        # Сплайн із затисненням: f'(x₀) = f'(xₙ) = 1
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

    # Повертаємо повний список опорних точок (knots)
    return a_coef, b_coef, c[:-1], d_coef, x


# ===============================
# Функція для обчислення значень сплайну (за отриманими коефіцієнтами)
# ===============================
def evaluate_spline(x_eval, coeffs):
    """
    Обчислює значення кубічного сплайну в точках x_eval.

    coeffs – кортеж (a_coef, b_coef, c_coef, d_coef, knots).
    """
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


# ===============================
# Функція для генерації єдиного формульного виразу (piecewise) сплайну
# ===============================
def generate_spline_formula(coeffs):
    """
    Генерує текстове представлення сплайну для кожного інтервалу.

    Для кожного інтервалу [xᵢ, xᵢ₊₁] виводиться вираз:
      S(x) = aᵢ + bᵢ*(x - xᵢ) + cᵢ*(x - xᵢ)² + dᵢ*(x - xᵢ)³.
    """
    a_coef, b_coef, c_coef, d_coef, knots = coeffs
    n = len(a_coef)
    formulas = []
    for i in range(n):
        formulas.append(
            f"For x in [{knots[i]}, {knots[i + 1]}]:\n"
            f"  S(x) = {a_coef[i]:.6g} + {b_coef[i]:.6g}*(x - {knots[i]}) + {c_coef[i]:.6g}*(x - {knots[i]})^2 + {d_coef[i]:.6g}*(x - {knots[i]})^3"
        )
    return "\n\n".join(formulas)


# ===============================
# Основна частина: Обчислення та виведення результатів
# ===============================

# Задане множину опорних точок
X_arr = np.array([-4, -3, 0, 2], dtype=float)
Y_arr = np.array([-4, -5, 3, -4], dtype=float)

# 1. Кубічний сплайн із затисненням (clamped): f'(x₀)=f'(xₙ)=1
coeffs_clamped = compute_cubic_spline(X_arr, Y_arr, bc_type='clamped')
formula_clamped = generate_spline_formula(coeffs_clamped)
print("=== Кубічний сплайн з одиничними першими похідними (clamped) ===")
print(formula_clamped)
print("\n----------------------------------------------------\n")

# 2. Натуральний кубічний сплайн (natural): f''(x₀)=f''(xₙ)=0
coeffs_natural = compute_cubic_spline(X_arr, Y_arr, bc_type='natural')
formula_natural = generate_spline_formula(coeffs_natural)
print("=== Кубічний сплайн з нульовими другими похідними (natural) ===")
print(formula_natural)
print("\n----------------------------------------------------\n")

# Для наочності побудуємо графіки обох сплайнів
x_vals = np.linspace(X_arr[0] - 1, X_arr[-1] + 1, 400)
y_clamped = evaluate_spline(x_vals, coeffs_clamped)
y_natural = evaluate_spline(x_vals, coeffs_natural)

plt.figure(figsize=(8, 5))
plt.plot(x_vals, y_clamped, 'b-', label="Clamped (f'(x₀)=f'(xₙ)=1)")
plt.plot(x_vals, y_natural, 'g--', label="Natural (f''(x₀)=f''(xₙ)=0)")
plt.plot(X_arr, Y_arr, 'ro', markersize=8, label="Опорні точки")
plt.xlabel("x")
plt.ylabel("S(x)")
plt.title("Кубічні сплайни: clamped та natural")
plt.legend()
plt.grid(True)
plt.show()

# =====================================================
# Завдання 2.5. Кусково-кубічний поліном Ерміта
# Дані: ті ж вузли: X = [-1, 0, 1, 2], Y = [2, 1, 2, 5]
#  похідні f'(xi) = [-1,-1,3,3].
# Будуємо кожну кубічну ланку за заданою формулою [&#8203;:contentReference[oaicite:4]{index=4}].
# =====================================================
print("Завдання 2.5. Кусково-кубічний поліном Ерміта")


def universal_hermite_linear_extension(x_sym, xp, yp, dyp):
    """
    Повертає символьну Piecewise-функцію, що представляє
    кусково-кубічний поліном Ерміта на [xp[0], xp[-1]]
    з лінійним продовженням поза цим інтервалом.

    Аргументи:
      x_sym - символьна змінна (наприклад, sp.Symbol('x'))
      xp    - список вузлів (x-координат)
      yp    - список значень функції в цих вузлах
      dyp   - список значень похідних у цих вузлах
    """

    # Допоміжна функція, що повертає кубічний поліном Ерміта на [x0, x1].
    def hermite_segment(x_sym, x0, x1, y0, y1, dy0, dy1):
        h = x1 - x0
        t = (x_sym - x0) / h
        h00 = 2 * t ** 3 - 3 * t ** 2 + 1
        h10 = t ** 3 - 2 * t ** 2 + t
        h01 = -2 * t ** 3 + 3 * t ** 2
        h11 = t ** 3 - t ** 2
        return (h00 * y0 +
                h10 * h * dy0 +
                h01 * y1 +
                h11 * h * dy1)

    n = len(xp)
    segments = []
    # Генеруємо кубічні поліноми Ерміта на кожному інтервалі
    for i in range(n - 1):
        seg_expr = hermite_segment(x_sym, xp[i], xp[i + 1],
                                   yp[i], yp[i + 1],
                                   dyp[i], dyp[i + 1])
        cond = (x_sym >= xp[i]) & (x_sym <= xp[i + 1])
        segments.append((sp.simplify(seg_expr), cond))

    # ЛІНІЙНЕ продовження вліво:
    # y = y_p[0] + dyp[0]*(x - x_p[0])
    left_line = yp[0] + dyp[0] * (x_sym - xp[0])
    segments.insert(0, (left_line, x_sym < xp[0]))

    # ЛІНІЙНЕ продовження вправо:
    # y = y_p[-1] + dyp[-1]*(x - x_p[-1])
    right_line = yp[-1] + dyp[-1] * (x_sym - xp[-1])
    segments.append((right_line, x_sym > xp[-1]))

    # Формуємо Piecewise-функцію
    piecewise_func = sp.Piecewise(*segments)
    return sp.simplify(piecewise_func)


# Приклад використання

# Вхідні дані: списки вузлів, значень та похідних (цей приклад можна змінити)
xp = [-1, 0, 1, 2]
yp = [2, 1, 2, 5]
dyp = [-1, -1, 3, 3]

H_lin_ext = universal_hermite_linear_extension(x, xp, yp, dyp)

print("Piecewise-функція з лінійним продовженням:")
sp.pprint(H_lin_ext)

# Перетворюємо на звичайну Python-функцію
H_func = sp.lambdify(x, H_lin_ext, 'numpy')

# Точки для графіка
xx = np.linspace(xp[0] - 1, xp[-1] + 1, 400)
yy = H_func(xx)

plt.figure(figsize=(8, 6))
plt.plot(xx, yy, 'b', label="Hermite spline з лінійним продовженням")
plt.scatter(xp, yp, color='r', zorder=5, label="Вузли")
plt.xlabel("x")
plt.ylabel("H(x)")
plt.title("Кусково-кубічний поліном Ерміта з лінійним продовженням")
plt.grid(True)
plt.legend()
plt.show()
# =====================================================
# Завдання 2.6.1. Формульне представлення параметричних рівнянь ламаної
# Дані: вершини в площині: [(0, -1), (0, -3), (2, 0), (0, 3), (0, 1), (1, -2), (-1, -2), (-1, 0)]
# Побудувати символьні параметричні рівняння кожного сегменту (лінійна інтерполяція)
# та зобразити графік ламаної.
# =====================================================
print("Завдання 2.6.1. Параметричні рівняння ламаної")
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Символьна змінна параметру
t = sp.symbols('t', real=True)

# Задано вершини ламаної
points_2_6_1 = [(-1, 0), (-3, 0), (0, 2), (3, 0), (1, 0), (1, -2), (-1, -2), (-1, 0)]
# Призначаємо параметричні значення для кожної точки: t = 0, 1, 2, ..., 6
t_vals = list(range(len(points_2_6_1)))

# Обчислюємо параметричні рівняння для кожного сегменту
segments_2_6_1 = []
for i in range(len(points_2_6_1) - 1):
    t1, t2 = t_vals[i], t_vals[i + 1]
    x1, y1 = points_2_6_1[i]
    x2, y2 = points_2_6_1[i + 1]
    # Лінійна інтерполяція: x(t) та y(t)
    x_seg = sp.simplify(x1 + (x2 - x1) / (t2 - t1) * (t - t1))
    y_seg = sp.simplify(y1 + (y2 - y1) / (t2 - t1) * (t - t1))
    segments_2_6_1.append(((t >= t1) & (t < t2), (x_seg, y_seg)))

print("Параметричні рівняння ламаної (2.6.1):")
for idx, (cond, (x_eq, y_eq)) in enumerate(segments_2_6_1, start=1):
    print(f"\nСегмент {idx}:")
    print("  x(t) =", sp.pretty(x_eq))
    print("  y(t) =", sp.pretty(y_eq))
    print("  для t, де", sp.pretty(cond))

# =====================================================
# Побудова графіка для завдання 2.6.1
# Обчислюємо дискретні значення для кожного сегменту
x_points = []
y_points = []

for i in range(len(points_2_6_1) - 1):
    t1, t2 = t_vals[i], t_vals[i + 1]
    # Створюємо масив значень параметру для поточного сегмента
    t_seg = np.linspace(t1, t2, 100)
    x1, y1 = points_2_6_1[i]
    x2, y2 = points_2_6_1[i + 1]
    # Лінійна інтерполяція: обчислення x та y для даного t_seg
    x_seg = x1 + (x2 - x1) / (t2 - t1) * (t_seg - t1)
    y_seg = y1 + (y2 - y1) / (t2 - t1) * (t_seg - t1)
    x_points.extend(x_seg)
    y_points.extend(y_seg)

plt.figure(figsize=(6, 4))
plt.plot(x_points, y_points, 'b-', label='Параметрична ламана')
# Позначимо задані вершини
xp, yp = zip(*points_2_6_1)
plt.plot(xp, yp, 'ro', markersize=8, label='Вузли')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Завдання 2.6.1. Графік параметричної ламаної")
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
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Задаємо координати вершин тетраедра
A = (7, 2, 4)
B = (7, -1, -2)
C = (3, 3, 1)
D = (-4, 2, 1)

# Формуємо послідовність вершин згідно з умовою: A, B, C, A, D, C, B, D
points = [A, B, C, A, D, C, B, D]

# Призначаємо параметричні значення для кожної точки: t = 0, 1, ..., 7
t_vals = list(range(len(points)))

# Побудова символічного представлення кожного сегменту ламаної
segments_symbolic = []  # список символічних виразів для кожного сегменту
for i in range(len(points) - 1):
    t1, t2 = t_vals[i], t_vals[i + 1]
    x1, y1, z1 = points[i]
    x2, y2, z2 = points[i + 1]
    # Лінійна інтерполяція для сегменту (символьне представлення)
    x_seg_sym = sp.simplify(x1 + (x2 - x1) / (t2 - t1) * (t - t1))
    y_seg_sym = sp.simplify(y1 + (y2 - y1) / (t2 - t1) * (t - t1))
    z_seg_sym = sp.simplify(z1 + (z2 - z1) / (t2 - t1) * (t - t1))
    segments_symbolic.append((x_seg_sym, y_seg_sym, z_seg_sym))

print("Символічні параметричні рівняння для кожного сегменту ламаної:")
for idx, (x_eq, y_eq, z_eq) in enumerate(segments_symbolic, start=1):
    print(f"\nСегмент {idx}:")
    print("  x(t) =", sp.pretty(x_eq))
    print("  y(t) =", sp.pretty(y_eq))
    print("  z(t) =", sp.pretty(z_eq))

# Побудова дискретних даних для графіка
x_points = []
y_points = []
z_points = []
for i in range(len(points) - 1):
    t1, t2 = t_vals[i], t_vals[i + 1]
    x1, y1, z1 = points[i]
    x2, y2, z2 = points[i + 1]
    # Створюємо масив значень параметра для поточного сегменту
    t_seg = np.linspace(t1, t2, 100)
    x_seg = x1 + (x2 - x1) / (t2 - t1) * (t_seg - t1)
    y_seg = y1 + (y2 - y1) / (t2 - t1) * (t_seg - t1)
    z_seg = z1 + (z2 - z1) / (t2 - t1) * (t_seg - t1)
    x_points.extend(x_seg)
    y_points.extend(y_seg)
    z_points.extend(z_seg)

# Побудова 3D графіка ламаної
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_points, y_points, z_points, 'b-', label='Параметрична ламана')
# Позначаємо вершини
xp, yp, zp = zip(*points)
ax.scatter(xp, yp, zp, c='r', s=50, label='Вершини')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Контур тетраедра: A → B → C → A → D → C → B → D")
ax.legend()
plt.show()

# =====================================================
# Завдання 2.7. Параметричні кубічні сплайни (Вар. 3)
# Дані: для варіанту 3 обираємо точки:
#   [(-2, 2), (4, 3), (1, -3), (-4, -5)]
# Для замкнутої кривої повторимо першу точку.
# =====================================================
print("Завдання 2.7. Параметричні кубічні сплайни (замкнута крива, Вар. 3)")


def compute_cubic_spline(x, y, bc_type='natural'):
    n = len(x) - 1
    h = np.diff(x)
    A = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)
    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        b[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    if bc_type == 'natural':
        A[0, 0] = 1.0
        A[n, n] = 1.0
        b[0] = 0.0
        b[n] = 0.0
    elif bc_type == 'clamped':
        A[0, 0] = 2 * h[0]
        A[0, 1] = h[0]
        A[n, n - 1] = h[n - 1]
        A[n, n] = 2 * h[n - 1]
        b[0] = 3 * ((y[1] - y[0]) / h[0] - 1)
        b[n] = 3 * (1 - (y[n] - y[n - 1]) / h[n - 1])
    elif bc_type == 'closed':
        # (c) Замкнений сплайн - періодичні граничні умови
        # Для періодичного сплайну: s0 = sn, s'0 = s'n, s''0 = s''n

        # Перший рядок: s0 = sn
        A[0, 0] = 1
        A[0, n] = -1
        b[0] = 0

        # Останній рядок: зв'язок між першою та останньою ділянками
        A[-1, 0] = -2 * h[1]
        A[-1, 1] = -h[1]
        A[-1, -1 - 1] = -h[-1]
        A[-1, -1] = -2 * h[-1]
        b[-1] = 3 * (((y[-1] - y[-1 - 1]) / h[-1]) - ((y[1] - y[0]) / h[1]))
    c = np.linalg.solve(A, b)
    a_coeff = y[:-1]
    b_coeff = np.zeros(n)
    d_coeff = np.zeros(n)
    for i in range(n):
        b_coeff[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 3.0
        d_coeff[i] = (c[i + 1] - c[i]) / (3.0 * h[i])
    return a_coeff, b_coeff, c[:-1], d_coeff, x[:-1]


def evaluate_spline(x_eval, x_knots, coeffs):
    a, b, c, d, xi = coeffs
    y_eval = np.zeros_like(x_eval)
    for i in range(len(x_eval)):
        idx = np.searchsorted(x_knots, x_eval[i]) - 1
        if idx < 0:
            idx = 0
        if idx >= len(xi):
            idx = len(xi) - 1
        dx = x_eval[i] - xi[idx]
        y_eval[i] = a[idx] + b[idx] * dx + c[idx] * dx ** 2 + d[idx] * dx ** 3
    return y_eval


def evaluate_polygon(points, num_points_per_edge=50):
    poly_x = []
    poly_y = []
    for i in range(len(points) - 1):
        p0 = np.array(points[i])
        p1 = np.array(points[i + 1])
        t = np.linspace(0, 1, num_points_per_edge)
        segment = np.outer(1 - t, p0) + np.outer(t, p1)
        poly_x.extend(segment[:, 0])
        poly_y.extend(segment[:, 1])
    return np.array(poly_x), np.array(poly_y)


def print_polygon_equations(points):
    print("Рівняння сегментів полігона:")
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        # Якщо відрізок не вертикальний
        if x2 != x1:
            m = (y2 - y1) / float(x2 - x1)
            b = y1 - m * x1
            print("Сегмент %d: y = %.3f*x + %.3f" % (i + 1, m, b))
        else:
            print("Сегмент %d: x = %.3f" % (i + 1, x1))


# Задані точки (вузли) для замкнутої кривої
points = [(-2, 2), (4, 3), (1, -3), (-4, -5)]
points_closed = points + [points[0]]

# Призначаємо параметри t: 0, 1, 2, 3, 4
t_arr = np.linspace(0, len(points_closed) - 1, len(points_closed))
x_points = np.array([p[0] for p in points_closed])
y_points = np.array([p[1] for p in points_closed])

# Обчислення коефіцієнтів сплайнів для x та y із замкненими граничними умовами
coeffs_x = compute_cubic_spline(t_arr, x_points, bc_type='closed')
coeffs_y = compute_cubic_spline(t_arr, y_points, bc_type='closed')

# Обчислення точок сплайну
t_dense = np.linspace(t_arr[0], t_arr[-1], 400)
x_dense = evaluate_spline(t_dense, t_arr, coeffs_x)
y_dense = evaluate_spline(t_dense, t_arr, coeffs_y)

# Генеруємо точки полігона (лінійна інтерполяція між вершинами)
polygon_points = points + [points[0]]
poly_x, poly_y = evaluate_polygon(polygon_points, num_points_per_edge=50)

# Виведення рівнянь для кожного сегмента полігона
print_polygon_equations(polygon_points)

# Побудова графіків
plt.figure(figsize=(8, 8))
plt.plot(x_dense, y_dense, 'b-', linewidth=2, label='Параметричний сплайн')
plt.plot(poly_x, poly_y, 'g--', linewidth=2, label='Полігон')
plt.plot(x_points, y_points, 'ro', markersize=8, label='Вузли')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Параметричні кубічні сплайни та полігон")
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()
