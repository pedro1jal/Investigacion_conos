import numpy as np
import time
from scipy.optimize import minimize, nnls
def problema_polar(G,H):
    # G,H son matrices de generadores de los conos P y Q respectivamente.
    n = G.shape[0]
    iterados = []
    def callback(z):
        iterados.append(z.copy())
    def constraint_GH(z , G, H):
        # desigualdad
        t = z[0]
        x = z[1:]
        return -((H.T@G)@x - t)
    def constraint_x(z):
        # desigualdad
        x = z[1:]
        return x
    def constraint_norm(z,G):
        # desgualdad
        return -(np.linalg.norm(G@z[1:])**2 - 1)

    def f_objetivo(z):
        return z[0]  # minimizar t
    z0 = np.ones(n+1)  # punto inicial (t, x)

    restricciones = [
        {'type': 'ineq', 'fun': constraint_GH, 'args':(G,H)},
        {'type': 'ineq', 'fun': constraint_x},
        {'type': 'ineq', 'fun': constraint_norm, 'args':(G,)},
    ]
    result = minimize(f_objetivo, z0, method='SLSQP', constraints=restricciones, callback=callback)
    t_opt = result.x[0]
    x_opt = result.x[1:]
    distancias = [
        np.linalg.norm(iterados[k+1] - iterados[k])
        for k in range(len(iterados)-1)
    ]
    return t_opt, x_opt, distancias

def proj_cone(G, u):
    """
    Proyecta el vector u sobre el cono convexo P = cone(G)
    mediante mínimos cuadrados no negativos (equivalente a lsqnonneg en MATLAB).
    """
    x, _ = nnls(G, u)
    return G @ x
def es_polar(u,H, tol = 1e-8):
    return np.all(H.T @ u <= tol)
def f_obj(u, H):
    return np.linalg.norm(proj_cone(H, u))
def gradiente(u, H):
    projuQ = proj_cone(H, u)
    norma_projuQ = np.linalg.norm(projuQ)
    if norma_projuQ < 1e-12:
        return np.zeros_like(u)
    else:
        return projuQ / norma_projuQ
def linesearch(u, g, H, t0=1.0, alpha=0.5, beta=0.5):
    t = t0
    while f_obj(u - t * g, H) > f_obj(u, H) - alpha * t * np.dot(g, g):
        t *= beta
    return t
def exceso_gradiente_proyectado_optimizado(G, H, x0 = None, t = 0.1, max_iter=1000, eps=1e-5):
    G = G / np.linalg.norm(G, axis=0, keepdims=True)
    H = H / np.linalg.norm(H, axis=0, keepdims=True)
    n = G.shape[0]
    p = G.shape[1]
    q = H.shape[1]
    if x0 is None:
        u0 = G@np.ones(p)
    else: 
        u0 = G@x0
    u0 = u0 / np.linalg.norm(u0)
    u_curr = u0.copy()
    valores_normas = []
    # bucle 
    converged = False
    tiempo = time.perf_counter()
    t_act = t
    for k in range(1, max_iter + 1):
        # verificar si es polar o no 
        if es_polar(u_curr,H):
            t_opt, u_opt, distancias = problema_polar(G,H)
            valores_normas += distancias
            converged = True
            break
            
        # de no ser polar, inicia el descenso proyectado
        projuQ = proj_cone(H, u_curr)
        norma_projuQ = np.linalg.norm(projuQ)
        # gradiente
        if norma_projuQ < 1e-12:
            g = np.zeros(n)
        else:
            g = gradiente(u_curr, H)
        # Paso euclideo directo en R^n
        t_k = max(t_act , 1e-6)  # Evitar pasos demasiado pequeños
        t_act = linesearch(u_curr, g, H, t0 = t_k)
        us = u_curr - t_act * g
        # Proy. sobre el cono P y retracción (normalización)
        projuP = proj_cone(G, us)
        norm_projuP = np.linalg.norm(projuP)
        
        if norm_projuP < 1e-12:
            u_update = u_curr.copy()
        else:
            u_update = projuP / norm_projuP
            
        if np.linalg.norm(u_update - u_curr) < eps:
            converged = True
            u_opt = u_update
            break
        valores_normas.append(np.linalg.norm(u_update - u_curr))
        u_curr = u_update
    v_opt = proj_cone(H, u_curr)/np.linalg.norm(proj_cone(H, u_curr))
    tiempo_final = time.perf_counter() - tiempo
    resultados = {"u_opt": u_opt, "v_opt": v_opt,"coseno" :np.dot(v_opt, u_opt) , 
            "angulo": np.arccos(np.clip(np.dot(v_opt, u_opt), -1, 1)), 
            "tiempo_final": tiempo_final, "k": k, "converged": converged,
              "valores_normas": valores_normas}
    return resultados