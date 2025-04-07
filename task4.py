import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import pyvista as pv

print("Завдання 4.1.1\n")

# Оголошення символьних змінних
x, y, z = sp.symbols('x y z', real=True)
R, a = sp.symbols('R a', positive=True)


# Функція перетину/різниці: ir(u,v) = (u + v - |u - v|)/2
def ir(u, v):
    return (u + v - sp.Abs(u - v)) / 2


# Ідентифікаційна функція для кола з вирізаним квадратом:
# 1) Коло: ω_circle(x,y) = R^2 - (x^2+y^2)
w_circle = R ** 2 - (x ** 2 + y ** 2)

# 2) Квадрат: ω_square(x,y) = ir(a - |x|, a - |y|)
w_square = ir(a - sp.Abs(x), a - sp.Abs(y))

# 3) Різниця (коло без внутрішньої частини квадрата):
#    ω(x,y) = ir(ω_circle(x,y), - ω_square(x,y))
omega = ir(w_circle, -w_square)

# Підставляємо конкретні значення: R = 1, a = 0.5
omega_specific = sp.simplify(omega.subs({R: 1, a: 0.5}))

# Виведення ідентифікаційної функції ω(x,y) у символьному вигляді
print("Ідентифікаційна функція ω(x,y):")
sp.pprint(omega_specific)
print("\nОбласть визначається нерівністю:")
print("ω(x,y) > 0")

# Для побудови графіку перетворимо символьний вираз у числову функцію
f = sp.lambdify((x, y), omega_specific, 'numpy')

# Створення сітки для побудови графіку
x_vals = np.linspace(-1.5, 1.5, 400)
y_vals = np.linspace(-1.5, 1.5, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f(X, Y)

# Побудова графіку: зафарбовуємо область, де ω(x,y) > 0
plt.figure(figsize=(6, 6))
plt.contourf(X, Y, Z, levels=[0, np.max(Z)], colors=['lightblue'])
plt.contour(X, Y, Z, levels=[0], colors='blue', linewidths=2)
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.title("Коло з вирізаним квадратом")
plt.show()

print("Завдання 4.1.2")

# Ідентифікаційні функції для кожного обмеження:
# 1) x >= 0
omega1 = x
# 2) 1/4 + x - y >= 0
omega2 = sp.Rational(1, 4) + x - y
# 3) x^2 + 2y^2 <= 1   ->   1 - (x^2 + 2y^2) >= 0
omega3 = 1 - (x ** 2 + 2 * y ** 2)

# Побудова загальної ідентифікаційної функції як перетину (мінімум значень)
omega12 = ir(omega1, omega2)
omega = ir(omega12, omega3)
omega_simpl = sp.simplify(omega)

# Виведення неявного рівняння контура фігури:
print("Неявне рівняння контура фігури:")
print("ω(x,y) = 0, де ω(x,y) = ")
sp.pprint(omega_simpl)
print("\nОбласть фігури: ω(x,y) > 0")

# Для побудови графіку перетворимо символьний вираз у числову функцію
f = sp.lambdify((x, y), omega_simpl, 'numpy')

# Створення сітки для побудови графіку
x_vals = np.linspace(-1.5, 1.5, 1000)
y_vals = np.linspace(-1.5, 1.5, 1000)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f(X, Y)

# Побудова графіку: малюємо контур, де ω(x,y) = 0
plt.figure(figsize=(6, 6))
plt.contourf(X, Y, Z, levels=[0, np.max(Z)], colors=['lightblue'])
plt.contour(X, Y, Z, levels=[0], colors='blue', linewidths=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Неявний контур фігури: ω(x,y) = 0')
plt.grid(True)
plt.show()
print("Завдання 4.2")

# Ідентифікаційні функції для обмежень в площині (xy)
omega1 = x  # x >= 0
omega2 = sp.Rational(1, 4) + x - y  # 1/4 + x - y >= 0
omega3 = 1 - (x ** 2 + 2 * y ** 2)  # 1 - (x^2 + 2y^2) >= 0 (тобто x^2+2y^2<=1)

# Побудова ідентифікаційної функції для (x,y)-області
omega_xy = ir(ir(omega1, omega2), omega3)
omega_xy_simpl = sp.simplify(omega_xy)

# Ідентифікаційні функції для обмежень по z:
omega_z1 = z  # z >= 0
omega_z2 = 3 - z  # z <= 3  <=> 3 - z >= 0

# Побудова повної 3D ідентифікаційної функції як перетин області в площині та проміжку по z
omega_xyz = ir(ir(omega_xy_simpl, omega_z1), omega_z2)
omega_xyz_simpl = sp.simplify(omega_xyz)

print("Ідентифікаційна функція ω(x,y,z):")
sp.pprint(omega_xyz_simpl)

# Отримання символьного рівняння контуру поверхні тіла (ω(x,y,z)=0)
print("\nНеявне рівняння контуру поверхні тіла:")
sp.pprint(omega_xyz_simpl)
print(" = 0")

# Перетворення символьного виразу у числову функцію
f = sp.lambdify((x, y, z), omega_xyz_simpl, 'numpy')

# Задаємо сітку точок для (x,y,z)
x_vals = np.linspace(-0.5, 2.5, 100)
y_vals = np.linspace(-0.5, 2.0, 100)
z_vals = np.linspace(-0.5, 3.5, 100)
# Використовуємо параметр indexing='ij', щоб координати відповідали порядку (x,y,z)
X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
F = f(X, Y, Z)

# Отримання розмірів сітки
nx, ny, nz = len(x_vals), len(y_vals), len(z_vals)

# Створення StructuredGrid:
# Формуємо масив точок розмірності (N,3)
points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
grid = pv.StructuredGrid()
grid.points = points
grid.dimensions = (nx, ny, nz)

# Додаємо дані функції до сітки (дані мають бути у порядку Fortran)
grid["values"] = F.flatten(order="F")

# Отримання ізоповерхні для рівня 0 (тобто ω(x,y,z)=0)
contours = grid.contour(isosurfaces=[0], scalars="values")

# Візуалізація ізоповерхні за допомогою PyVista
plotter = pv.Plotter()
plotter.add_mesh(contours, color="blue", opacity=0.5, label="ω(x,y,z)=0")
plotter.add_axes(xlabel='x', ylabel='y', zlabel='z')
plotter.add_legend()
plotter.add_title("Поверхня тіла, задана системою обмежень")
plotter.show()
