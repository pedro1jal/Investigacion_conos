import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls
import time

def proj_cone(G, u):
    """
    Proyecta el vector u sobre el cono convexo P = cone(G)
    mediante mínimos cuadrados no negativos (equivalente a lsqnonneg en MATLAB).
    """
    x, _ = nnls(G, u)
    return G @ x
def sphere_param(theta, phi):
    """
    Parametrización de la esfera unitaria en R^3 usando coordenadas esféricas.
    theta: ángulo polar [0, pi]
    phi: ángulo azimutal [0, 2*pi]
    """
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])

def funcion(theta, phi, G):
    """
    Función objetivo f(u) = ||Proj_Q(u)||_2 donde Q = R^3_+.
    """

    u = sphere_param(theta, phi)
    if np.any(u < 0):
        return max(u) # Si algún componente es negativo, la proyección es cero
    else:
        proj_u = proj_cone(G, u)
    return np.linalg.norm(proj_u)

G = np.eye(3)
f = lambda theta, phi: funcion(theta, phi, G)


# Parámetros angulares
theta = np.linspace(0, np.pi, 200)
phi = np.linspace(0, 2*np.pi, 200)

A, B = np.meshgrid(theta, phi)

# Evaluar la función en cada par de ángulos
F = np.zeros_like(A)

for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        F[i, j] = funcion(A[i, j], B[i, j], G)

# Gráfica 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(A, B, F, cmap='viridis', edgecolor='none')

fig.colorbar(surf, ax=ax, label=r'$F(\theta,\phi)$')

ax.set_xlabel(r'$\theta$ (rad)')
ax.set_ylabel(r'$\phi$ (rad)')
ax.set_zlabel(r'$F(\theta,\phi)$')
# ax.set_title(r'$F(\theta,\phi)=\|\operatorname{Proj}_{\operatorname{cone}(G)}(u)\|$')

plt.tight_layout()
plt.show()