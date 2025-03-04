# =====================================================
# Завдання 3.1. Поверхня обертання (варіант 3)
# Профільна крива в площині XZ задана точками:
# u:    [0, 1, 2, 3, 4]
# φ(u): [0, 2, 1, 2, 0]   (радіус обертання)
# ψ(u): [0, 0, 1, 2, 2]   (координата Z)
#
# Параметричні рівняння поверхні обертання:
# X = φ(u)*cos(v),  Y = φ(u)*sin(v),  Z = ψ(u),
# де u ∈ [0,4],  v ∈ [0, 2π].#TODO
# =====================================================

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from mpl_toolkits.mplot3d import Axes3D  # для 3D-графіки

# Задаємо символьну змінну та параметр u
u, v = sp.symbols('u v')

# Дані профільної кривої:
u_vals = [0, 1, 2, 3, 4]
phi_vals = [0, 2, 1, 2, 0]  # x-координати профілю
psi_vals = [0, 0, 1, 2, 2]  # z-координати профілю


# Будуємо інтерполяційні поліноми Лагранжа для φ(u) та ψ(u)
def lagrange_poly_param(u_sym, U, F):
    n = len(U)
    L = 0
    for i in range(n):
        li = 1
        for j in range(n):
            if j != i:
                li *= (u_sym - U[j]) / (U[i] - U[j])
        L += F[i] * li
    return sp.simplify(L)


phi_poly = lagrange_poly_param(u, u_vals, phi_vals)
psi_poly = lagrange_poly_param(u, u_vals, psi_vals)

print("Інтерполяційний поліном для φ(u):")
sp.pprint(phi_poly)
print("\nІнтерполяційний поліном для ψ(u):")
sp.pprint(psi_poly)

# Параметричні рівняння поверхні обертання:
X_expr = sp.simplify(phi_poly * sp.cos(v))
Y_expr = sp.simplify(phi_poly * sp.sin(v))
Z_expr = psi_poly

print("\nПараметричні рівняння поверхні обертання:")
print("X(u,v) = ")
sp.pprint(X_expr)
print("Y(u,v) = ")
sp.pprint(Y_expr)
print("Z(u,v) = ")
sp.pprint(Z_expr)

# Перетворимо у чисельні функції для побудови графіка
X_func = sp.lambdify((u, v), X_expr, "numpy")
Y_func = sp.lambdify((u, v), Y_expr, "numpy")
Z_func = sp.lambdify((u, v), Z_expr, "numpy")

# Створюємо сітку параметрів
u_vals_dense = np.linspace(0, 4, 50)
v_vals_dense = np.linspace(0, 2 * np.pi, 50)
U_grid, V_grid = np.meshgrid(u_vals_dense, v_vals_dense)

X_grid = X_func(U_grid, V_grid)
Y_grid = Y_func(U_grid, V_grid)
Z_grid = Z_func(U_grid, V_grid)

# Побудова 3D-графіка
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis', alpha=0.8, edgecolor='none')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Завдання 3.1. Поверхня обертання")
plt.show()
# =====================================================
# Завдання 3.2. Поверхня перенесення
# Використаємо ту ж профільну криву з 3.1.
# Параметричні рівняння:
# X(u,v) = φ(u),   Y(u,v) = v,   Z(u,v) = ψ(u),
# де u ∈ [0,4],  v ∈ [0, 3] (інтервал перенесення обрано самостійно).
# =====================================================

# Використовуємо ті ж phi_poly та psi_poly з попереднього завдання.

X2_expr = phi_poly
Y2_expr = v
Z2_expr = psi_poly

print("Параметричні рівняння поверхні перенесення:")
print("X(u,v) = ")
sp.pprint(X2_expr)
print("Y(u,v) = v")
print("Z(u,v) = ")
sp.pprint(Z2_expr)

# Перетворюємо у чисельні функції
X2_func = sp.lambdify((u, v), X2_expr, "numpy")
Y2_func = sp.lambdify((u, v), Y2_expr, "numpy")
Z2_func = sp.lambdify((u, v), Z2_expr, "numpy")

u_vals_dense = np.linspace(0, 4, 50)
v_vals_dense = np.linspace(0, 3, 50)
U2, V2 = np.meshgrid(u_vals_dense, v_vals_dense)

X2_grid = X2_func(U2, V2)
Y2_grid = Y2_func(U2, V2)
Z2_grid = Z2_func(U2, V2)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X2_grid, Y2_grid, Z2_grid, cmap='plasma', alpha=0.8, edgecolor='none')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Завдання 3.2. Поверхня перенесення (вар. 3)")
plt.show()
# =====================================================
# Завдання 3.3. Обмежена лінійчата поверхня
# Нехай:
# r0(u) = (u, sin(u), 0) та r1(u) = (u, sin(u)+1, 3), для u ∈ [0, 2π].
# Рівняння поверхні: r(u,v) = (1-v)*r0(u) + v*r1(u),  v ∈ [0,1].
# =====================================================

# Задаємо параметр u
u_vals = np.linspace(0, 2 * np.pi, 200)


def r0(u_val):
    return np.array([u_val, np.sin(u_val), 0])


def r1(u_val):
    return np.array([u_val, np.sin(u_val) + 1, 3])


# Створимо функції для кожної компоненти поверхні
def ruled_surface(u_val, v_val):
    r0_val = np.array(r0(u_val))
    r1_val = np.array(r1(u_val))
    return (1 - v_val) * r0_val + v_val * r1_val


# Побудова сітки
u_dense = np.linspace(0, 2 * np.pi, 50)
v_dense = np.linspace(0, 1, 20)
U_ruled, V_ruled = np.meshgrid(u_dense, v_dense)

X_ruled = np.zeros_like(U_ruled)
Y_ruled = np.zeros_like(U_ruled)
Z_ruled = np.zeros_like(U_ruled)

for i in range(U_ruled.shape[0]):
    for j in range(U_ruled.shape[1]):
        point = ruled_surface(U_ruled[i, j], V_ruled[i, j])
        X_ruled[i, j] = point[0]
        Y_ruled[i, j] = point[1]
        Z_ruled[i, j] = point[2]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X_ruled, Y_ruled, Z_ruled, cmap='coolwarm', alpha=0.8, edgecolor='none')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Завдання 3.3. Обмежена лінійчата поверхня (вар. 3)")
plt.show()
# =====================================================
# Завдання 3.4. Поверхні подібних поперечних перерізів (вар. 3)
# Нехай направляюча крива (профіль) в площині XZ задається:
# x_p(u) = 1 + 0.5*u,   z_p(u) = u,    u ∈ [0,4]
# Базовий переріз у площині XY – окружність:
# x_b(v)=cos(v),  y_b(v)=sin(v),  v ∈ [0, 2π]
# Припустимо, що точка перетину профілю з базовим перерізом має x0 = x_p(0) = 1.
#
# Тоді параметричне рівняння поверхні має вигляд:
# X(u,v)= x_p(u)*cos(v),  Y(u,v)= x_p(u)*sin(v),  Z(u,v)= z_p(u) = u.
# =====================================================

# Задаємо параметр u та v
u, v = sp.symbols('u v')

# Направляюча крива:
xp_expr = 1 + 0.5 * u
zp_expr = u
# Базовий переріз (окружність)
xb_expr = sp.cos(v)
yb_expr = sp.sin(v)

# Рівняння поверхні:
X_expr_34 = sp.simplify(xp_expr * xb_expr)
Y_expr_34 = sp.simplify(xp_expr * yb_expr)
Z_expr_34 = sp.simplify(zp_expr)

print("Параметричні рівняння поверхні подібних поперечних перерізів:")
print("X(u,v) =")
sp.pprint(X_expr_34)
print("Y(u,v) =")
sp.pprint(Y_expr_34)
print("Z(u,v) =")
sp.pprint(Z_expr_34)

# Перетворення у чисельні функції
X34_func = sp.lambdify((u, v), X_expr_34, "numpy")
Y34_func = sp.lambdify((u, v), Y_expr_34, "numpy")
Z34_func = sp.lambdify((u, v), Z_expr_34, "numpy")

u_vals_dense = np.linspace(0, 4, 50)
v_vals_dense = np.linspace(0, 2 * np.pi, 50)
U34, V34 = np.meshgrid(u_vals_dense, v_vals_dense)

X34_grid = X34_func(U34, V34)
Y34_grid = Y34_func(U34, V34)
Z34_grid = Z34_func(U34, V34)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X34_grid, Y34_grid, Z34_grid, cmap='copper', alpha=0.8, edgecolor='none')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Завдання 3.4. Поверхня подібних поперечних перерізів (вар. 3)")
plt.show()
