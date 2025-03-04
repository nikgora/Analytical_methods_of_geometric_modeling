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

# Визначаємо кускову функцію
f_piece = sp.Piecewise(
    (-1 - x, x <= -1),
    (x * (x + 1), (x > -1) & (x <= 0)),
    (x * (x - 1), (x > 0) & (x <= 1)),
    (1 - x, x > 1)
)
print("Неперервна кускова функція:")
sp.pprint(f_piece)

# Побудова графіка
f_piece_func = sp.lambdify(x, f_piece, "numpy")
x_vals2 = np.linspace(-1.5, 1.5, 400)
y_vals2 = f_piece_func(x_vals2)

plt.figure(figsize=(6, 4))
plt.plot(x_vals2, y_vals2, 'm-', label='Кускова функція')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Завдання 2.2. Кускова функція")
plt.legend()
plt.grid(True)
plt.show()

# =====================================================
# Завдання 2.3. Неперервні кусково-поліноміальні функції загального вигляду
# =====================================================

p_piece = sp.Piecewise(
    (1 - (x + 1) ** 2, x <= 0),
    ((x - 1) ** 2 - 1, (x > 0))
)
print("Неперервна кусково-поліноміальна функція:")
sp.pprint(p_piece)

# Побудова графіка
p_piece_func = sp.lambdify(x, p_piece, "numpy")
x_vals3 = np.linspace(-3, 5, 400)
y_vals3 = p_piece_func(x_vals3)

plt.figure(figsize=(6, 4))
plt.plot(x_vals3, y_vals3, 'c-', label='Кусково-поліноміальна функція')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Завдання 2.3. Кусково-поліноміальна функція")
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

from scipy.interpolate import CubicSpline

X_arr = np.array(X, dtype=float)
Y_arr = np.array(Y, dtype=float)

# (a) Clamped сплайн (перші похідні рівні 1)
cs_clamped = CubicSpline(X_arr, Y_arr, bc_type=((1, 1.0), (1, 1.0)))
# (b) Натуральний сплайн (другі похідні рівні 0)
cs_natural = CubicSpline(X_arr, Y_arr, bc_type='natural')

x_vals4 = np.linspace(min(X_arr) - 1, max(X_arr) + 1, 400)
y_clamped = cs_clamped(x_vals4)
y_natural = cs_natural(x_vals4)

plt.figure(figsize=(6, 4))
plt.plot(x_vals4, y_clamped, 'b-', label='Кубічний сплайн (clamped)')
plt.plot(x_vals4, y_natural, 'g--', label='Натуральний сплайн')
plt.plot(X_arr, Y_arr, 'ro', markersize=8, label='Вузли')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Завдання 2.4. Кубічні сплайни")
plt.grid(True)
plt.show()


# =====================================================
# Завдання 2.5. Кусково-кубічний поліном Ерміта
# Дані: ті ж вузли: X = [-1, 0, 1, 2], Y = [2, 1, 2, 5]
#  похідні f'(xi) = [-1,-1,3,3].
# Будуємо кожну кубічну ланку за заданою формулою [&#8203;:contentReference[oaicite:4]{index=4}].
# =====================================================

def hermite_segment(x_val, p1, p2):
    x1, y1, g1 = p1
    x2, y2, g2 = p2
    h = x2 - x1
    term1 = y1 * (h + 2 * (x_val - x1)) * (x_val - x2) ** 2 / h ** 3
    term2 = g1 * (x_val - x1) * (x_val - x2) ** 2 / h ** 2
    term3 = g2 * (x_val - x2) * (x_val - x1) ** 2 / h ** 2
    term4 = y2 * (h - 2 * (x_val - x2)) * (x_val - x1) ** 2 / h ** 3
    return sp.simplify(term1 + term2 + term3 + term4)


g_val = [-1, -1, 3, 3]
X = [-1, 0, 1, 2]
Y = [2, 1, 2, 5]
nodes = list(zip(X, Y, g_val))
segments = []
for i in range(1, len(nodes)):
    seg = hermite_segment(x, nodes[i - 1], nodes[i])
    segments.append(seg)
    print(f"Сегмент між x = {nodes[i - 1][0]} та x = {nodes[i][0]}:")
    sp.pprint(seg)

# Створюємо кускову функцію (Piecewise)
hermite_piecewise = sp.Piecewise(
    (segments[0], (x >= nodes[0][0]) & (x < nodes[1][0])),
    (segments[1], (x >= nodes[1][0]) & (x < nodes[2][0])),
    (segments[2], (x >= nodes[2][0]) & (x <= nodes[3][0]))
)
print("\nКусковий поліном Ерміта:")
sp.pprint(hermite_piecewise)

# Побудова графіка
x_vals5 = np.linspace(min(X) - 1, max(X) + 1, 400)
hermite_func = sp.lambdify(x, hermite_piecewise, "numpy")
y_vals5 = hermite_func(x_vals5)

plt.figure(figsize=(6, 4))
plt.plot(x_vals5, y_vals5, 'm-', label='Поліном Ерміта')
plt.plot(X_arr, Y_arr, 'ko', markersize=8, label='Вузли')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Завдання 2.5. Поліном Ерміта")
plt.grid(True)
plt.show()

# =====================================================
# Завдання 2.6.1. Формульне представлення параметричних рівнянь ламаної
# Дані: вершини в площині: [(0, -1), (0, -3), (2, 0), (0, 3), (0, 1), (1, -2), (-1, -2), (-1, 0)]
# Побудувати символьні параметричні рівняння кожного сегменту (лінійна інтерполяція)
# та зобразити графік ламаної.
# =====================================================

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
# Використаємо CubicSpline з умовою periodic [&#8203;:contentReference[oaicite:5]{index=5}].
# =====================================================

points_2_7 = [(-2, 2), (4, 3), (1, -3), (-4, -5)]
points_closed = points_2_7 + [points_2_7[0]]
n_points = len(points_closed)
# Призначимо параметр t як 0, 1, ..., n_points-1
t_arr = np.linspace(0, n_points - 1, n_points)
x_points = np.array([p[0] for p in points_closed], dtype=float)
y_points = np.array([p[1] for p in points_closed], dtype=float)

# Побудова параметричних сплайнів з періодичними умовами
csx = CubicSpline(t_arr, x_points, bc_type='periodic')
csy = CubicSpline(t_arr, y_points, bc_type='periodic')

t_dense = np.linspace(t_arr[0], t_arr[-1], 400)
x_dense = csx(t_dense)
y_dense = csy(t_dense)

plt.figure(figsize=(6, 6))
plt.plot(x_dense, y_dense, 'b-', label='Параметричний сплайн')
plt.plot(x_points, y_points, 'ro--', markersize=8, label='Вузли')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Завдання 2.7. Параметричні кубічні сплайни (замкнута крива, Вар. 3)")
plt.axis('equal')
plt.legend()
plt.grid(True)
plt.show()
