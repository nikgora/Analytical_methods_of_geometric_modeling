# =====================================
# Завдання 3.1, варіант 3 (обертання навколо OX)
# =====================================
print("Завдання 3.1. Обертання фігури навколо осі ")
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

# Оголошення символьних змінних
t, u, v = sp.symbols('t u v', real=True)


# Функція P(t, a, w) для інтерполяції ламаної:
# P(t, a, w) = 1/(2w) * (w + |t - a| - |t - a - w|)
def P(a, w):
    return (1 / (2 * w)) * (w + sp.Abs(t - a) - sp.Abs(t - a - w))


# Функція для побудови символьних виразів ламаної для довільної кількості вершин
def build_polyline_expr(vertices, t_values=None):
    n = len(vertices)
    # Якщо значення параметра не задані – рівномірний розподіл: 0, 1, 2, ..., n-1
    if t_values is None:
        t_values = list(range(n))
    if len(t_values) != n:
        raise ValueError("Кількість значень параметра має дорівнювати кількості вершин.")

    # Початкове положення – перша вершина
    x_expr = sp.sympify(vertices[0][0])
    y_expr = sp.sympify(vertices[0][1])
    # Додаємо внески для кожного відрізка ламаної
    for i in range(1, n):
        dt = t_values[i] - t_values[i - 1]
        dx = vertices[i][0] - vertices[i - 1][0]
        dy = vertices[i][1] - vertices[i - 1][1]
        x_expr += dx * P(t_values[i - 1], dt)
        y_expr += dy * P(t_values[i - 1], dt)
    return sp.simplify(x_expr), sp.simplify(y_expr), t_values


# Задаємо список вершин ламаної.
# Наприклад, остання точка (0,0) закриває ламану.
vertices = [(0, 0), (2, 0), (2, 1), (1.5, 2), (0.5, 2), (0, 1), (0, 0)]

# Отримуємо символьні вирази для x(t) та y(t)
x_expr, y_expr, t_vals_sym = build_polyline_expr(vertices)

# Будуємо параметричне рівняння поверхні обертання навколо осі OX:
# X(u, v) = x(u), Y(u, v) = y(u)*cos(v), Z(u, v) = y(u)*sin(v)
X_expr = x_expr
Y_expr = y_expr * sp.cos(v)
Z_expr = y_expr * sp.sin(v)

# Виведення символьних рівнянь
print("Параметричне рівняння ламаної:")
print("x(u) =")
sp.pprint(x_expr)
print("\ny(u) =")
sp.pprint(y_expr)

print("\nПараметричне рівняння поверхні обертання (навколо осі OX):")
print("X(u, v) =")
sp.pprint(X_expr)
print("\nY(u, v) =")
sp.pprint(Y_expr)
print("\nZ(u, v) =")
sp.pprint(Z_expr)

# Створюємо числові функції для візуалізації
f_x = sp.lambdify(t, x_expr, 'numpy')
f_y = sp.lambdify(t, y_expr, 'numpy')
f_X = sp.lambdify((t, v), X_expr, 'numpy')
f_Y = sp.lambdify((t, v), Y_expr, 'numpy')
f_Z = sp.lambdify((t, v), Z_expr, 'numpy')

# Побудова 2D-графіку ламаної за допомогою Matplotlib
t_num = np.linspace(t_vals_sym[0], t_vals_sym[-1], 300)
x_num = f_x(t_num)
y_num = f_y(t_num)

plt.figure(figsize=(6, 4))
plt.plot(x_num, y_num, 'b-', label='Ламана')
# Позначення вершин червоними точками
vertices_x = [pt[0] for pt in vertices]
vertices_y = [pt[1] for pt in vertices]
plt.scatter(vertices_x, vertices_y, color='red', zorder=5)
plt.title("Ламана в площині XY")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

# Створення сітки параметрів для побудови поверхні
t_vals = np.linspace(t_vals_sym[0], t_vals_sym[-1], 100)
v_vals = np.linspace(0, 2 * np.pi, 100)
T, V = np.meshgrid(t_vals, v_vals)
X_vals = f_X(T, V)
Y_vals = f_Y(T, V)
Z_vals = f_Z(T, V)

# Для PyVista створимо StructuredGrid.
# Зауважте, що np.meshgrid повертає масиви форми (n_v, n_t).
# Ми встановлюємо dims = (n_t, n_v, 1) і трансформуємо точки у Fortran-порядку.
nt = len(t_vals)
nv = len(v_vals)
points = np.column_stack([
    X_vals.T.ravel(order='F'),
    Y_vals.T.ravel(order='F'),
    Z_vals.T.ravel(order='F')
])
grid = pv.StructuredGrid()
grid.points = points
grid.dimensions = (nt, nv, 1)

# Генеруюча ламана: значення при v = 0 (тобто Z = 0)
t_line = np.linspace(t_vals_sym[0], t_vals_sym[-1], 300)
x_line = f_x(t_line)
y_line = f_y(t_line)
z_line = np.zeros_like(t_line)
curve_points = np.column_stack([x_line, y_line, z_line])
n_line = len(t_line)
# Формуємо масив з'єднувальних індексів для ламаної:
lines = np.hstack(([n_line], np.arange(n_line)))
polyline = pv.PolyData()
polyline.points = curve_points
polyline.lines = lines

# Створення першої PyVista-сцени з першим кутом огляду (наприклад, azimuth=45, elevation=30)
p1 = pv.Plotter(window_size=(800, 600))
p1.add_mesh(grid, opacity=0.8, cmap='viridis', show_scalar_bar=False)
p1.add_mesh(polyline, color='red', line_width=5)
p1.add_text("Поверхня обертання (вигляд 1)", position='upper_edge', font_size=14, shadow=True)
# Налаштування камери вручну
p1.camera_position = [(np.max(X_vals), np.max(Y_vals), np.max(Z_vals)),
                      (np.mean(X_vals), np.mean(Y_vals), np.mean(Z_vals)),
                      (0, 0, 1)]
p1.camera.azimuth = 45
p1.camera.elevation = 30
p1.show()

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
ax.view_init(elev=10, azim=-30, roll=0)

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
import sympy as sp


def bernstein_polyline_symbolic(X, Y):
    """
    Обчислює символічний вираз p(x) за формулою Бернштейна для ламаної,
    заданої контрольними точками (X[i], Y[i]), i = 0,1,...,n.

    Якщо задано лише 2 точки (n=1), повертається звичайна лінійна інтерполяція.

    Формула:

      p(x) = 1/2 * [
           y0 + (y1-y0)/(x1-x0) * (x - x0)
         + y_{n-1} + (y_n-y_{n-1})/(x_n-x_{n-1}) * (x - x_{n-1})
         + Σ_{k=1}^{n-1} { ( (y_{k+1}-y_k)/(x_{k+1}-x_k) - (y_k-y_{k-1})/(x_k-x_{k-1}) ) * |x - x_k| }
      ]

    Аргументи:
      X : список або масив контрольних x-координат (довжина n+1)
      Y : список або масив контрольних y-координат (довжина n+1)

    Повертає:
      Символьний вираз p(x) (Sympy expression).
    """
    # Символьна змінна x
    x = sp.symbols('x')

    n = len(X) - 1  # якщо n=1 => 2 точки
    if n == 1:
        expr = Y[0] + (Y[1] - Y[0]) / (X[1] - X[0]) * (x - X[0])
        return sp.simplify(expr)

    # Обчислюємо внесок крайових (лівий та правий)
    term_left = Y[0] + (Y[1] - Y[0]) / (X[1] - X[0]) * (x - X[0])
    term_right = Y[n - 1] + (Y[n] - Y[n - 1]) / (X[n] - X[n - 1]) * (x - X[n - 1])

    # Обчислюємо суму по внутрішнім вузлам k = 1,2,..., n-1
    sum_internal = 0
    for k in range(1, n):
        slope_plus = (Y[k + 1] - Y[k]) / (X[k + 1] - X[k])
        slope_minus = (Y[k] - Y[k - 1]) / (X[k] - X[k - 1])
        sum_internal += (slope_plus - slope_minus) * sp.Abs(x - X[k])

    expr = sp.Rational(1, 2) * (term_left + term_right + sum_internal)
    expr = sp.simplify(expr)
    return expr


# =============================================================================
# Приклад використання.
# Контрольні точки можуть бути будь-якої кількості, наприклад:
# X = [1,2,3,4,5] та Y = [1,1,62,123,1]
# =============================================================================
# Задаємо контрольні точки
X = [-1, 0, 1]
Y = [0, 1, 0]

# Отримуємо символічний вираз p(x)
p_expr = bernstein_polyline_symbolic(X, Y)

# Виводимо отриману формулу
print("Символічна формула p(x) для заданих контрольних точок:")
sp.pretty_print(p_expr)

# Перетворюємо символічний вираз у чисельну функцію для обчислення
p_func = sp.lambdify(sp.symbols('x'), p_expr, 'numpy')

# Створюємо щільну сітку значень x (тобто безперервну функцію)
x_vals = np.linspace(X[0], X[-1], 500)
y_vals = p_func(x_vals)

# Побудова графіка (у 3D; оскільки крива лежить у площині XY, використаємо z=0)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_vals, y_vals, np.zeros_like(x_vals), label="Bernstein Polyline", lw=2, color='blue')
ax.scatter(X, Y, np.zeros_like(X), color='red', s=50, label="Контрольні точки")
ax.set_title("Bernstein Polyline Interpolation")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.legend()

# Масштабування осей
max_range = max(np.ptp(x_vals), np.ptp(y_vals)) / 2
mid_x = 0.5 * (x_vals.min() + x_vals.max())
mid_y = 0.5 * (y_vals.min() + y_vals.max())
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(-0.5, 0.5)

plt.show()
