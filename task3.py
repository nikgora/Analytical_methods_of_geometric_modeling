# =====================================
# Завдання 3.1, варіант 3 (обертання навколо OZ)
# =====================================
print("Завдання 3.1. Обертання фігури навколо осі ")
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def revolve_line_around_x(x1, y1, x2, y2, n_t=50, n_phi=50):
    """
    Повертає X, Y, Z – 2D-сітки координат для лінії від (x1, y1) до (x2, y2),
    оберненої навколо осі OX.
    n_t, n_phi – кількість точок для параметрів t та phi.
    """
    # Параметр t для відрізка [0,1]
    t = np.linspace(0, 1, n_t)
    # Параметр phi для обертання [0, 2*pi]
    phi = np.linspace(0, 2 * np.pi, n_phi)

    # Створюємо 2D-сітки для t і phi
    T, PHI = np.meshgrid(t, phi)

    # Параметризація відрізка в площині XY:
    # x змінюється лінійно, y – відстань від осі обертання (ось OX)
    X_line = x1 + (x2 - x1) * T
    Y_line = y1 + (y2 - y1) * T

    # Обертання навколо осі OX: координата X залишається, а Y та Z визначають коло
    X = X_line
    Y = Y_line * np.cos(PHI)
    Z = Y_line * np.sin(PHI)

    return X, Y, Z


# Задаємо вершини фігури
A = (0, 0)
B = (4, 0)  # нижня основа, вісь обертання
F = (4, 1)
D = (2.8, 2)
C = (1.2, 2)
E = (0, 1)

# Побудова початкової фігури як замкненого полігону (z=0)
polygon = np.array([
    [A[0], A[1], 0],
    [B[0], B[1], 0],
    [F[0], F[1], 0],
    [D[0], D[1], 0],
    [C[0], C[1], 0],
    [E[0], E[1], 0],
    [A[0], A[1], 0]  # повертаємось до A
])

# Обчислюємо поверхні обертання для кожного відрізка, окрім A-B (віссю обертання)
X_BF, Y_BF, Z_BF = revolve_line_around_x(B[0], B[1], F[0], F[1])
X_FD, Y_FD, Z_FD = revolve_line_around_x(F[0], F[1], D[0], D[1])
X_DC, Y_DC, Z_DC = revolve_line_around_x(D[0], D[1], C[0], C[1])
X_CE, Y_CE, Z_CE = revolve_line_around_x(C[0], C[1], E[0], E[1])
X_EA, Y_EA, Z_EA = revolve_line_around_x(E[0], E[1], A[0], A[1])

# Створюємо 3D-сцену
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Малюємо початковий полігон (фігуру) у площині z=0
ax.plot(polygon[:, 0], polygon[:, 1], polygon[:, 2], color='k', linewidth=2, marker='o', label='Початкова фігура')

# Налаштовуємо параметри для відображення поверхонь
surf_kwargs = dict(rstride=1, cstride=1, cmap='viridis', alpha=0.7, edgecolor='none')

# Малюємо поверхні обертання для кожного відрізка
ax.plot_surface(X_BF, Y_BF, Z_BF, **surf_kwargs)
ax.plot_surface(X_FD, Y_FD, Z_FD, **surf_kwargs)
ax.plot_surface(X_DC, Y_DC, Z_DC, **surf_kwargs)
ax.plot_surface(X_CE, Y_CE, Z_CE, **surf_kwargs)
ax.plot_surface(X_EA, Y_EA, Z_EA, **surf_kwargs)

# Малюємо вісь обертання (від A до B) червоним кольором
ax.plot([A[0], B[0]], [A[1], B[1]], [0, 0], color='red', linewidth=4, label='Вісь обертання')

# Налаштовуємо підписи осей та масштабування
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_box_aspect((1, 1, 1))
ax.view_init(elev=30, azim=120)  # elev — кут підйому, azim — азимутальний кут
plt.title("Обертання фігури навколо нижньої основи (червона лінія)")
ax.legend()
plt.show()

# =====================================================
# Завдання 3.2.
# =====================================================
print("Завдання 3.2. Поверхня перенесення ламаної кривої")


def resample_polyline(points, num_points=300):
    """
    Розбиває ламану, задану списком точок (x, y), на більше число точок
    за допомогою лінійної інтерполяції.
    """
    points = np.array(points)
    # Обчислюємо відстані між послідовними точками
    distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    cumulative = np.concatenate(([0], np.cumsum(distances)))
    total_length = cumulative[-1]
    u = cumulative / total_length  # нормований параметр від 0 до 1
    # Рівномірні значення параметра
    u_uniform = np.linspace(0, 1, num_points)
    # Інтерполяція для кожної координати
    x_interp = np.interp(u_uniform, u, points[:, 0])
    y_interp = np.interp(u_uniform, u, points[:, 1])
    return x_interp, y_interp


# Дані завдання 2.6.1: вершини ламаної
points_polyline = [(-1, 0), (-3, 0), (0, 2), (3, 0), (1, 0), (1, -2), (-1, -2), (-1, 0)]

# Отримаємо більше точок, що описують ламану
x_curve, y_curve = resample_polyline(points_polyline, num_points=300)

# Вибираємо довжину перенесення (екструзії) – напр., L = 3
L = 10
s = np.linspace(0, L, 5)

# Створюємо сітку для параметрів:
# t – параметр уздовж ламаної (взятий за індекс точок), s – параметр перенесення
X_surface = np.tile(x_curve, (len(s), 1))
Y_surface = np.tile(y_curve, (len(s), 1))
Z_surface = np.tile(s[:, np.newaxis], (1, len(x_curve)))

# Візуалізація:
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Малюємо отриману поверхню перенесення
surf = ax.plot_surface(X_surface, Y_surface, Z_surface, cmap='plasma', alpha=0.8, edgecolor='none')

# Малюємо початкову ламану криву (в площині z=0)
ax.plot(x_curve, y_curve, np.zeros_like(x_curve), 'k-', linewidth=1, label='Ламана крива')

# Налаштовуємо підписи осей та заголовок
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Поверхня перенесення ламаної кривої (екструзія вздовж осі Z)')
ax.legend()
plt.show()

# Друкуємо параметричні рівняння
print("Параметричні рівняння поверхні перенесення:")
print("x = x(t), де x(t) задається інтерполяцією точки ламаної:")
print(points_polyline)
print("y = y(t)")
print("z = s, де s ∈ [0, {:.1f}]".format(L))

# =====================================================
# Завдання 3.3.1 Обмежена лінійчата поверхня
# Побудова параметричних рівнянь зони трикутника A, B, C
# та візуалізація зафарбованої області і ламаної контуру.
# =====================================================
print("Завдання 3.3.1 Обмежена лінійчата поверхня")

# Задаємо координати вершин трикутника (A, B, C)
A = np.array([3, 3, -1])
B = np.array([5, 5, -2])
C = np.array([4, 1, 1])

# -----------------------------------------------------------------
# Параметричне рівняння зони трикутника:
# P(u, v) = (1 - u - v)*A + u*B + v*C, де 0 ≤ u ≤ 1, 0 ≤ v ≤ 1 та u + v ≤ 1.
#
# Покоординатно:
#   x(u, v) = A_x*(1 - u - v) + B_x*u + C_x*v,
#   y(u, v) = A_y*(1 - u - v) + B_y*u + C_y*v,
#   z(u, v) = A_z*(1 - u - v) + B_z*u + C_z*v.
# -----------------------------------------------------------------
print("Параметричне рівняння зони трикутника:")
print("x(u,v) = {}*(1 - u - v) + {}*u + {}*v".format(A[0], B[0], C[0]))
print("y(u,v) = {}*(1 - u - v) + {}*u + {}*v".format(A[1], B[1], C[1]))
print("z(u,v) = {}*(1 - u - v) + {}*u + {}*v".format(A[2], B[2], C[2]))
print("де 0 ≤ u, v та u + v ≤ 1")

# -----------------------------------------------------------------
# Параметричні рівняння ламаної (контур трикутника)
# Розглянемо три сегменти:
# 1) Сегмент AB (від A до B):
#       x(t) = A_x + (B_x - A_x)*t,
#       y(t) = A_y + (B_y - A_y)*t,
#       z(t) = A_z + (B_z - A_z)*t,  де 0 ≤ t ≤ 1.
#
# 2) Сегмент BC (від B до C):
#       x(t) = B_x + (C_x - B_x)*t,
#       y(t) = B_y + (C_y - B_y)*t,
#       z(t) = B_z + (C_z - B_z)*t,  де 0 ≤ t ≤ 1.
#
# 3) Сегмент CA (від C до A):
#       x(t) = C_x + (A_x - C_x)*t,
#       y(t) = C_y + (A_y - C_y)*t,
#       z(t) = C_z + (A_z - C_z)*t,  де 0 ≤ t ≤ 1.
# -----------------------------------------------------------------
print("\nПараметричні рівняння ламаної (контур трикутника):")

print("\nСегмент AB (від A до B):")
print("x(t) = {} + {}*t".format(A[0], B[0] - A[0]))
print("y(t) = {} + {}*t".format(A[1], B[1] - A[1]))
print("z(t) = {} + {}*t".format(A[2], B[2] - A[2]))
print("де 0 ≤ t ≤ 1")

print("\nСегмент BC (від B до C):")
print("x(t) = {} + {}*t".format(B[0], C[0] - B[0]))
print("y(t) = {} + {}*t".format(B[1], C[1] - B[1]))
print("z(t) = {} + {}*t".format(B[2], C[2] - B[2]))
print("де 0 ≤ t ≤ 1")

print("\nСегмент CA (від C до A):")
print("x(t) = {} + {}*t".format(C[0], A[0] - C[0]))
print("y(t) = {} + {}*t".format(C[1], A[1] - C[1]))
print("z(t) = {} + {}*t".format(C[2], A[2] - C[2]))
print("де 0 ≤ t ≤ 1")

# -----------------------------------------------------------------
# Побудова області трикутника за допомогою сітки параметрів
# -----------------------------------------------------------------
n = 50  # Кількість точок для сітки
u = np.linspace(0, 1, n)
v = np.linspace(0, 1, n)
U, V = np.meshgrid(u, v)

# Використовуємо маску, щоб залишити точки, де u + v ≤ 1
mask = (U + V) <= 1
U_valid = U[mask]
V_valid = V[mask]

# Обчислення координат точок області трикутника
X = (1 - U_valid - V_valid) * A[0] + U_valid * B[0] + V_valid * C[0]
Y = (1 - U_valid - V_valid) * A[1] + U_valid * B[1] + V_valid * C[1]
Z = (1 - U_valid - V_valid) * A[2] + U_valid * B[2] + V_valid * C[2]

# -----------------------------------------------------------------
# Побудова 3D-графіка
# -----------------------------------------------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Побудова зафарбованої області трикутника
ax.plot_trisurf(X, Y, Z, color='cyan', alpha=0.5, edgecolor='gray')

# Побудова ламаної (контур трикутника) за порядком вершин A -> B -> C -> A
triangle_x = [A[0], B[0], C[0], A[0]]
triangle_y = [A[1], B[1], C[1], A[1]]
triangle_z = [A[2], B[2], C[2], A[2]]
ax.plot(triangle_x, triangle_y, triangle_z, color='black', linewidth=2)

# Налаштування підписів осей та заголовку графіка
ax.set_title("Трикутна область та її контур")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()
# =====================================================
# Завдання 3.3.2. Обмежена лінійчата поверхня для паралелограма
# Знаходження четвертої вершини D (протилежної A) для паралелограма з вершинами A, B, C
# та побудова параметричних рівнянь області і контуру паралелограма.
# =====================================================
print("Завдання 3.3.2. Обмежена лінійчата поверхня для паралелограма")
# Обчислення координат четвертої вершини D (протилежної A)
# Формула: D = B + C - A
D = B + C - A
print("Координати вершини D:", D)
# Обчислення координат четвертої вершини D (протилежної A) за формулою: D = B + C - A
D = B + C - A
print("Координати вершини D:", D)

# -----------------------------------------------------------------
# Параметричне рівняння області паралелограма:
# P(u, v) = A + u*(B - A) + v*(C - A), де 0 ≤ u ≤ 1 та 0 ≤ v ≤ 1.
#
# Покоординатно:
#   x(u, v) = A_x + (B_x - A_x)*u + (C_x - A_x)*v,
#   y(u, v) = A_y + (B_y - A_y)*u + (C_y - A_y)*v,
#   z(u, v) = A_z + (B_z - A_z)*u + (C_z - A_z)*v.
# -----------------------------------------------------------------
print("\nПараметричне рівняння області паралелограма:")
print("x(u, v) = {} + {}*u + {}*v".format(A[0], B[0] - A[0], C[0] - A[0]))
print("y(u, v) = {} + {}*u + {}*v".format(A[1], B[1] - A[1], C[1] - A[1]))
print("z(u, v) = {} + {}*u + {}*v".format(A[2], B[2] - A[2], C[2] - A[2]))
print("де 0 ≤ u ≤ 1 та 0 ≤ v ≤ 1")

# -----------------------------------------------------------------
# Параметричне рівняння ламаної (контур паралелограма)
# Сегменти ламаної:
# 1) Сегмент AB: від A до B:
#       x(t) = A_x + (B_x - A_x)*t,
#       y(t) = A_y + (B_y - A_y)*t,
#       z(t) = A_z + (B_z - A_z)*t,   де 0 ≤ t ≤ 1.
#
# 2) Сегмент BD: від B до D:
#       x(t) = B_x + (D_x - B_x)*t,
#       y(t) = B_y + (D_y - B_y)*t,
#       z(t) = B_z + (D_z - B_z)*t,   де 0 ≤ t ≤ 1.
#
# 3) Сегмент DC: від D до C:
#       x(t) = D_x + (C_x - D_x)*t,
#       y(t) = D_y + (C_y - D_y)*t,
#       z(t) = D_z + (C_z - D_z)*t,   де 0 ≤ t ≤ 1.
#
# 4) Сегмент CA: від C до A:
#       x(t) = C_x + (A_x - C_x)*t,
#       y(t) = C_y + (A_y - C_y)*t,
#       z(t) = C_z + (A_z - C_z)*t,   де 0 ≤ t ≤ 1.
# -----------------------------------------------------------------
print("\nПараметричні рівняння ламаної (контур паралелограма):")

print("\nСегмент AB (від A до B):")
print("x(t) = {} + {}*t".format(A[0], B[0] - A[0]))
print("y(t) = {} + {}*t".format(A[1], B[1] - A[1]))
print("z(t) = {} + {}*t".format(A[2], B[2] - A[2]))
print("де 0 ≤ t ≤ 1")

print("\nСегмент BD (від B до D):")
print("x(t) = {} + {}*t".format(B[0], D[0] - B[0]))
print("y(t) = {} + {}*t".format(B[1], D[1] - B[1]))
print("z(t) = {} + {}*t".format(B[2], D[2] - B[2]))
print("де 0 ≤ t ≤ 1")

print("\nСегмент DC (від D до C):")
print("x(t) = {} + {}*t".format(D[0], C[0] - D[0]))
print("y(t) = {} + {}*t".format(D[1], C[1] - D[1]))
print("z(t) = {} + {}*t".format(D[2], C[2] - D[2]))
print("де 0 ≤ t ≤ 1")

print("\nСегмент CA (від C до A):")
print("x(t) = {} + {}*t".format(C[0], A[0] - C[0]))
print("y(t) = {} + {}*t".format(C[1], A[1] - C[1]))
print("z(t) = {} + {}*t".format(C[2], A[2] - C[2]))
print("де 0 ≤ t ≤ 1")

# -----------------------------------------------------------------
# Побудова 3D-графіка паралелограма та його контуру
# -----------------------------------------------------------------
# Створення сітки параметрів u та v для області паралелограма
n = 50  # Кількість точок для сітки
u = np.linspace(0, 1, n)
v = np.linspace(0, 1, n)
U, V = np.meshgrid(u, v)

# Обчислення координат точок області паралелограма
X = A[0] + U * (B[0] - A[0]) + V * (C[0] - A[0])
Y = A[1] + U * (B[1] - A[1]) + V * (C[1] - A[1])
Z = A[2] + U * (B[2] - A[2]) + V * (C[2] - A[2])

# Створення 3D-графіка
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Побудова зафарбованої області паралелограма
ax.plot_surface(X, Y, Z, color='lightblue', alpha=0.5, edgecolor='gray')

# Побудова ламаної (контур паралелограма)
# Порядок вершин: A -> B -> D -> C -> A
contour_x = [A[0], B[0], D[0], C[0], A[0]]
contour_y = [A[1], B[1], D[1], C[1], A[1]]
contour_z = [A[2], B[2], D[2], C[2], A[2]]
ax.plot(contour_x, contour_y, contour_z, color='black', linewidth=2)

# Налаштування підписів осей та заголовку графіка
ax.set_title("Паралелограм та його контур")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()
# =====================================================
# Завдання 3.3.3. Параметричні рівняння області гіперболоїда
#         z = x^2 - y^2, розташованої над зоною площини XY,
# утвореною кривими: y = 2x^2 - 7x - 3 та y = 5x^2 + 2x - 3,
# де точки їх перетинання обчислено за допомогою Python.
# =====================================================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Для 3D-візуалізації
import sympy as sp

print("Завдання 3.3.3. Параметричні рівняння області гіперболоїда")
# ---------------------------
# 1. Обчислення точок перетину кривих за допомогою sympy
# ---------------------------
x = sp.symbols('x')
# Задаємо вирази для кривих:
y1_expr = 2 * x ** 2 - 7 * x - 3  # крива 1: y = 2x^2 - 7x - 3
y2_expr = 5 * x ** 2 + 2 * x - 3  # крива 2: y = 5x^2 + 2x - 3

# Розв'язуємо рівняння перетину: y1_expr - y2_expr = 0
solutions = sp.solve(y1_expr - y2_expr, x)
solutions = sorted([float(sol) for sol in solutions])
print("Розрахунок точок перетину:")
print("Знайдені x: ", solutions)

# Обчислюємо відповідні y (використаємо y1_expr)
intersection_points = [(sol, float(y1_expr.subs(x, sol))) for sol in solutions]
print("Точки перетину (x, y):", intersection_points)

# Визначаємо область параметра u як [u_min, u_max]
u_min, u_max = solutions[0], solutions[-1]
print("Область параметра u: [{:.2f}, {:.2f}]".format(u_min, u_max))
print("--------------------------------------------------\n")

# ---------------------------
# 2. Символічне формулювання параметричних рівнянь
# ---------------------------
# Введемо символи u та v
u, v = sp.symbols('u v')

# Нехай x = u
x_sym = u

# Нижня крива (y_down): y = 5*u^2 + 2*u - 3
y_down_sym = 5 * u ** 2 + 2 * u - 3

# Верхня крива (y_up): y = 2*u^2 - 7*u - 3
y_up_sym = 2 * u ** 2 - 7 * u - 3

# Інтерполяція для y: y = y_down*(1 - v) + y_up*v
y_sym = sp.simplify(y_down_sym * (1 - v) + y_up_sym * v)

# Поверхня гіперболоїда: z = x^2 - y^2
z_sym = sp.simplify(x_sym ** 2 - y_sym ** 2)

# Друкуємо символічні параметричні рівняння:
print("Символічні параметричні рівняння:")
print("x(u,v) =", sp.pretty(x_sym))
print("y(u,v) =", sp.pretty(y_sym))
print("z(u,v) =", sp.pretty(z_sym))
print("де:")
print("   u ∈ [{:.2f}, {:.2f}]".format(u_min, u_max))
print("   v ∈ [0, 1]")
print("--------------------------------------------------\n")

# ---------------------------
# 3. Перетворення символічних виразів у числові функції для побудови графіка
# ---------------------------
f_x = sp.lambdify((u, v), x_sym, 'numpy')
f_y = sp.lambdify((u, v), y_sym, 'numpy')
f_z = sp.lambdify((u, v), z_sym, 'numpy')

# ---------------------------
# 4. Побудова графіка поверхні
# ---------------------------
n = 50  # Кількість точок сітки
u_vals = np.linspace(u_min, u_max, n)
v_vals = np.linspace(0, 1, n)
U, V = np.meshgrid(u_vals, v_vals)

# Обчислення координат поверхні
X = f_x(U, V)
Y = f_y(U, V)
Z = f_z(U, V)

# Створення 3D-графіка
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Побудова поверхні з використанням кольорової мапи
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')

ax.set_title("Поверхня гіперболоїда z = x^2 - y^2 над зоною в XY")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
print("Завдання 3.4 Параметричні рівняння поверхні подібних поперечних перерізів")
# =====================================================
# Завдання 3.4.
# Побудова поверхні подібних поперечних перерізів.
# -----------------------------------------------------
# БАЗА (у площині XY, z=0): трикутник (-1,0), (0,1), (1,0).
# ПРОФІЛЬ: ламана  (0,1), (0.75,0.75), (0.75,0.25),(0,0).
# =====================================================
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# -----------------------------------------------------
# Функції для опису координат базового трикутника (piecewise)
# Параметр v змінюється від 0 до 3 для замкнення контуру
# -----------------------------------------------------
def x_b(v):
    return np.piecewise(
        v,
        [(0 <= v) & (v < 1),
         (1 <= v) & (v < 2),
         (2 <= v) & (v <= 3)],
        [
            lambda v: -1 + v,  # від (-1,0) до (0,1)
            lambda v: (v - 1),  # від (0,1) до (1,0)
            lambda v: 1 - 2 * (v - 2)  # від (1,0) назад до (-1,0)
        ]
    )


def y_b(v):
    return np.piecewise(
        v,
        [(0 <= v) & (v < 1),
         (1 <= v) & (v < 2),
         (2 <= v) & (v <= 3)],
        [
            lambda v: v,  # від (-1,0) до (0,1)
            lambda v: 1 - (v - 1),  # від (0,1) до (1,0)
            lambda v: 0 * np.ones_like(v)  # від (1,0) назад до (-1,0)
        ]
    )


# Створюємо масив параметрів v для побудови контуру
v_vals = np.linspace(0, 3, 100)
x_vals = x_b(v_vals)
y_vals = y_b(v_vals)
z_vals = np.zeros_like(v_vals)  # z=0 для всіх точок

# -----------------------------------------------------
# Побудова 3D-графіка
# -----------------------------------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Малюємо контур трикутника
ax.plot(x_vals, y_vals, z_vals, label="Контур трикутника", lw=2)

# Малюємо вершини червоними крапками
vertices_x = np.array([-1, 0, 1])
vertices_y = np.array([0, 1, 0])
vertices_z = np.array([0, 0, 0])
ax.scatter(vertices_x, vertices_y, vertices_z, color='red', s=50, label="Вершини")

# Налаштування осей та заголовку
ax.set_title("Базовий трикутник в 3D (z = 0)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()

plt.show()
