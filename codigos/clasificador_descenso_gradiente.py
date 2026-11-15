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

# en caso de no estar en el polar
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



def descenso_gradiente_proyectado(G, H, u_curr, t_act, eps = 1e-5):
    n = G.shape[0]

    # Proyección sobre el cono Q
    projuQ = proj_cone(H, u_curr)
    norma_projuQ = np.linalg.norm(projuQ)

    # Cálculo del gradiente
    if norma_projuQ < 1e-12:
        g = np.zeros(n)
    else:
        g = gradiente(u_curr, H)

    # Búsqueda lineal
    t_k = max(t_act, 1e-6)
    t_act = linesearch(u_curr, g, H, t0=t_k)

    # Paso de descenso
    us = u_curr - t_act * g

    # Proyección sobre P y normalización
    projuP = proj_cone(G, us)
    norm_projuP = np.linalg.norm(projuP)

    if norm_projuP < eps:
        u_update = u_curr.copy()
    else:
        u_update = projuP / norm_projuP

    return u_update, t_act
def sampleo_ortante_positivo(n_muestras, p, puntos_excluidos, radio):
    
    validos = np.empty((0, p))
    
    while len(validos) < n_muestras:
        # 1. Generamos un lote de puntos candidatos en [0, 1]^p
        faltantes = n_muestras - len(validos)
        candidatos = np.random.uniform(0, 1, size=(faltantes, p))
        
        # 2. Calculamos la distancia euclidiana de cada candidato a TODOS los puntos excluidos
        # Resultado tiene forma (faltantes, n_excluidos)
        distancias = np.linalg.norm(candidatos[:, None, :] - puntos_excluidos[None, :, :], axis=-1)
        
        # 3. Creamos máscara: True solo si el candidato está fuera del radio de TODOS los excluidos
        mascara_validos = np.all(distancias > radio, axis=1)
        
        # 4. Concatenamos los nuevos válidos con los que ya teníamos
        validos = np.vstack((validos, candidatos[mascara_validos]))
        
    return validos

def exceso_grad_proy(
    G, H, x0=None, t=0.1, max_iter=50, eps=1e-5):
    """
    calcula el exceso entre dos conos convexos poliedrales P = cone(G) y Q = cone(H) resolviendo el problema 
    min_u \in P max_v \in Q <u,v> por medio del descenso del gradiente proyectado

    El problema se reformula de min_u \in P max_v \in Q <u,v> a min_u \in P f(u), donde f(u) = max_v \in Q <u,v>.

    esta función retorna los vectores solución
    u = u* y v = v*, el coseno del ángulo exceso entre los dos conos y el ángulo mismo

    los que se debe de ingresar son las matrices generadoras de P y Q los cuales son G y H, respectivamente.
    El x0 es un vector de pesos, el cual inicia a u0 = Gx0, por lo que este debe ser no negativo (x0>=0).
    """
    G = G / np.linalg.norm(G, axis=0, keepdims=True)
    H = H / np.linalg.norm(H, axis=0, keepdims=True)
    p = G.shape[1]
    f = lambda x : f_obj(x,H)
    if x0 is None:
        vect_rand = np.random.uniform(0,1, size = (1,p))
        u0 = G @ vect_rand.ravel()
    else:
        u0 = G @ x0
    u_curr = u0 / np.linalg.norm(u0)
    X_sam_total = np.empty((0, p))
    ini_time = time.perf_counter()
    t_act = t
    for _ in np.arange(1,3):
        if es_polar(u_curr,H):
            t_opt, u_opt, distancias = problema_polar(G,H)
            break
        u_update, t_act = descenso_gradiente_proyectado(G, H, u_curr, t_act, eps)
        y_star = f(u_update)

        # for _1 in np.arange(2):
        X_sample = sampleo_ortante_positivo(1000, p, X_sam_total, radio = 0.01)
        X_sam_total = np.concatenate((X_sam_total, X_sample), axis = 0)
        U_sample = X_sample @ G.T
        U_sample /= np.linalg.norm(U_sample, axis=1, keepdims=True)
        lista = [(u , f(u)) for u in U_sample]
        min_lista = min(lista , key=lambda item: item[1])
        if y_star <= min_lista[1]:
            print("no hay mejor mínimo")
        else: 
            u_curr = min_lista[0]
    u_opt = u_curr/np.linalg.norm(u_curr)
    v_opt = proj_cone(H,u_opt)/np.linalg.norm(proj_cone(H,u_opt))
    coseno = u_opt@v_opt
    tiempo = time.perf_counter()- ini_time
    return {"u": u_opt,
            "v": v_opt,
            "coseno": coseno,
            "angulo": np.arccos(coseno),
            "tiempo": tiempo, 
            "largo_sample": len(X_sam_total)}


########### clasificador

from concurrent.futures import ThreadPoolExecutor, as_completed
# from optimizador import *
def f_dist(a1, a2):
    a1 = np.clip(a1, -1, 1)
    a2 = np.clip(a2, -1, 1)
    t1 = np.arccos(a1)
    t2 = np.arccos(a2)
    return 2 * np.sqrt((np.sin(t1/2))**2 + (np.sin(t2/2))**2)

def calcular_centroide_y_varianza(cono):
    """
    Calcula el centroide direccional y la varianza angular de un cono.
    Asume que las columnas del cono representan los rayos generadores.
    """
    cono = np.asarray(cono)
    
    # Caso degenerado: Si es un único rayo (vector o matriz de una columna)
    if cono.ndim == 1 or cono.shape[1] == 1:
        rayo = cono.ravel()
        norma = np.linalg.norm(rayo)
        centroide = rayo / norma if norma > 0 else rayo
        return centroide, 0.0  # Varianza angular es cero
        
    # 1. Normalizar cada rayo por seguridad (asegurar que sean unitarios)
    normas_rayos = np.linalg.norm(cono, axis=0, keepdims=True)
    normas_rayos[normas_rayos == 0] = 1.0  # Evitar división por cero
    rayos_unitarios = cono / normas_rayos
    
    # 2. Centroide Direccional (Ecuación 1)
    suma_rayos = np.sum(rayos_unitarios, axis=1)
    norma_suma = np.linalg.norm(suma_rayos)
    centroide = suma_rayos / norma_suma if norma_suma > 0 else suma_rayos
    
    # 3. Varianza Angular (Ecuación 2)
    # Producto punto de cada rayo con el centroide
    productos_punto = np.dot(rayos_unitarios.T, centroide)
    productos_punto = np.clip(productos_punto, -1.0, 1.0) # Control de precisión numérica
    
    thetas = np.arccos(productos_punto)
    varianza_angular = np.mean(thetas ** 2)
    
    return centroide, varianza_angular

def evaluar_clase(clase, cono_train, cono_prueba):
    if cono_prueba.shape[1] == 1:
        a = cono_prueba.ravel()
        a1 = cono_prueba.ravel()
        b = proj_cone(cono_train, a).ravel()
        b1 = proj_cone(cono_train, -a).ravel()
        distancia_cos = np.dot(a, b)
        distancia_sim = f_dist(np.dot(a, b), np.dot(a1, b1))
        distancia_min = distancia_cos
    else:
        # Sim1
        res1 = exceso_grad_proy(cono_prueba,cono_train)   #(cono_prueba, cono_train)
        # a, b, c = res1['u_opt']
        cos1, t1  = res1['coseno'] , res1['tiempo_final']
    #     {
    #     "u_opt": u_opt,
    #     "v_opt": v_opt,
    #     "coseno": np.dot(v_opt, u_opt),
    #     "angulo": np.arccos(
    #         np.clip(np.dot(v_opt, u_opt), -1, 1)
    #     ),
    #     "tiempo_final": tiempo_final,
    #     "k": k,
    #     "converged": converged,
    #     "valores_normas": valores_normas
    # }

        # Sim2
        res2 = exceso_grad_proy(cono_train, cono_prueba)

        cos2, t2  = res2['coseno'] , res2['tiempo_final']
        # u_min, v_min, coseno_max = min_angle(cono_train, cono_prueba)
        # Normalización
        # a = a / np.linalg.norm(a)
        # b = b / np.linalg.norm(b)
        # a1 = a1 / np.linalg.norm(a1)
        # b1 = b1 / np.linalg.norm(b1)
        # producto2 = np.dot(a1, b)
        
        # if abs(coseno_max - producto2) <= 0.01:
        #     print("el calculo del algoritmo de proyecciones es aproximado al de calcular <u2 , v1>")
        # else:
        #     print("el calculo del algoritmo de proyecciones NO es aproximado al de calcular <u2 , v1>")
        # if coseno_max > producto2:
        #     print("el algoritmo SÍ es mejor que <u2 , v1>")
        # else:
        #     print("el algoritmo NO es mejor que <u2 , v1>")

        # Cálculos
        # distancia_cos = np.dot(a, b)
        distancia_sim = f_dist(cos1, cos2)
        # distancia_min = coseno_max 
        
    # --- NUEVO: Calculamos la varianza y centroide del cono_train en el hilo ---
    centroide_train, var_train = calcular_centroide_y_varianza(cono_train)
        
    return clase, distancia_cos, distancia_sim, distancia_min, centroide_train, var_train
# Función principal
def clasificador(cono_prueba, conos_entrenamiento, tolerancia=1e-7):

    # INICIO DE MEDICIÓN: Comenzamos a medir el tiempo para este cono_prueba
    tiempo_inicio = time.perf_counter()
    
    # 1. Calcular centroide y varianza del cono consulta (P_q) una sola vez
    centroide_prueba, var_prueba = calcular_centroide_y_varianza(cono_prueba)
    
    resultados = {}

    # 2. Ejecución multihilo
    with ThreadPoolExecutor(max_workers=4) as executor:
        futuros = {
            executor.submit(evaluar_clase, clase, cono, cono_prueba): clase
            for clase, cono in conos_entrenamiento.items()
        }

        for futuro in as_completed(futuros):
            try:
                clase_eval, dist_cos, dist_sim, dist_min, c_train, v_train = futuro.result()
                resultados[clase_eval] = {
                    'dist_cos': dist_cos,
                    'dist_sim': dist_sim,
                    # 'dist_min': dist_min,
                    'centroide': c_train,
                    'varianza': v_train
                }
            except Exception as exc:
                print(f"La clase {futuros[futuro]} generó una excepción: {exc}")

    if not resultados:
        # Si algo falla de raíz, calculamos el tiempo transcurrido de todas formas
        tiempo_total = time.perf_counter() - tiempo_inicio
        return None, None, None, tiempo_total

    # --- LÓGICA DE DESEMPATE MATEMÁTICO ---
    def resolver_empate(clases_empatadas, nombre_metrica):
        if len(clases_empatadas) == 1:
            return clases_empatadas[0]
            
        print(f"!!! [EMPATE] Encontrado en '{nombre_metrica}' entre las clases: {clases_empatadas}")
        
        if np.isclose(var_prueba, 0.0, atol=tolerancia):
            print("   -> Aplicando Criterio Secundario (Similitud Media) debido a var(P_q) = 0")
            mejor_clase = None
            mejor_sim_media = -np.inf
            for clase in clases_empatadas:
                c_train = resultados[clase]['centroide']
                sim_media = np.dot(centroide_prueba, c_train)
                if sim_media > mejor_sim_media:
                    mejor_sim_media = sim_media
                    mejor_clase = clase
            return mejor_clase

        mejor_clase = None
        min_d_sigma = np.inf
        clases_degeneradas_train = []
        
        for clase in clases_empatadas:
            v_train = resultados[clase]['varianza']
            if np.isclose(v_train, 0.0, atol=tolerancia):
                clases_degeneradas_train.append(clase)
                continue
                
            d_sigma = abs((var_prueba / v_train) - 1.0)
            if d_sigma < min_d_sigma:
                min_d_sigma = d_sigma
                mejor_clase = clase
                
        if mejor_clase is None and clases_degeneradas_train:
            print("   -> Aplicando Criterio Secundario debido a varianza cero en clases de entrenamiento")
            mejor_sim_media = -np.inf
            for clase in clases_degeneradas_train:
                c_train = resultados[clase]['centroide']
                sim_media = np.dot(centroide_prueba, c_train)
                if sim_media > mejor_sim_media:
                    mejor_sim_media = sim_media
                    mejor_clase = clase
                    
        print(f"   -> Resuelto con éxito. Ganador: {mejor_clase}")
        return mejor_clase

    # 3. Procesar las tres clasificaciones principales buscando empates
    todas_las_clases = list(resultados.keys())
    
    # Criterio Coseno (Maximizar)
    max_cos = max(resultados[c]['dist_cos'] for c in todas_las_clases)
    empatados_cos = [c for c in todas_las_clases if np.isclose(resultados[c]['dist_cos'], max_cos, atol=tolerancia)]
    mejor_clase_coseno = resolver_empate(empatados_cos, "Score Coseno")

    # Criterio Ángulo Mínimo (Maximizar)
    # max_min = max(resultados[c]['dist_min'] for c in todas_las_clases)
    # empatados_min = [c for c in todas_las_clases if np.isclose(resultados[c]['dist_min'], max_min, atol=tolerancia)]
    # mejor_clase_min = resolver_empate(empatados_min, "Score Mínimo")

    # Criterio Distancia / Similitud (Minimizar)
    min_dist = min(resultados[c]['dist_sim'] for c in todas_las_clases)
    empatados_dist = [c for c in todas_las_clases if np.isclose(resultados[c]['dist_sim'], min_dist, atol=tolerancia)]
    mejor_clase_distancia = resolver_empate(empatados_dist, "Score Distancia")

    # FIN DE MEDICIÓN: Calculamos los segundos totales transcurridos
    tiempo_total = time.perf_counter() - tiempo_inicio

    # Ahora la función retorna también el tiempo_total al final
    return mejor_clase_coseno, mejor_clase_distancia, tiempo_total


# def clasificador_conos(cono_prueba, conos_entrenamiento, tolerancia=1e-7, n_clases = 4):
    # 1. Calcular centroide y varianza del cono consulta (P_q) una sola vez
    centroide_prueba, var_prueba = calcular_centroide_y_varianza(cono_prueba)
    
    resultados = {}

    # 2. Ejecución multihilo
    with ThreadPoolExecutor(max_workers=n_clases) as executor:
        futuros = {
            executor.submit(evaluar_clase, clase, cono, cono_prueba): clase
            for clase, cono in conos_entrenamiento.items()
        }

        for futuro in as_completed(futuros):
            try:
                clase_eval, dist_cos, dist_sim, dist_min, c_train, v_train = futuro.result()
                resultados[clase_eval] = {
                    'dist_cos': dist_cos,
                    'dist_sim': dist_sim,
                    'dist_min': dist_min,
                    'centroide': c_train,
                    'varianza': v_train
                }
            except Exception as exc:
                print(f"La clase {futuros[futuro]} generó una excepción: {exc}")

    if not resultados:
        return None, None, None

    # --- LÓGICA DE DESEMPATE MATEMÁTICO ---
    def resolver_empate(clases_empatadas, nombre_metrica):
        # Si no hay empate real, retornar la única clase
        if len(clases_empatadas) == 1:
            return clases_empatadas[0]
            
        print(f"Empate en '{nombre_metrica}' entre las clases: {clases_empatadas}")
        
        # CASO SECUNDARIO: Cono consulta es un único rayo (varianza == 0)
        if np.isclose(var_prueba, 0.0, atol=tolerancia):
            print("   -> Aplicando Criterio Secundario (Similitud Media) ")
            mejor_clase = None
            mejor_sim_media = -np.inf
            for clase in clases_empatadas:
                c_train = resultados[clase]['centroide']
                sim_media = np.dot(centroide_prueba, c_train) # <u , q_c_barra>
                if sim_media > mejor_sim_media:
                    mejor_sim_media = sim_media
                    mejor_clase = clase
            return mejor_clase

        # CRITERIO PRINCIPAL: Distancia de varianzas d_sigma (Ecuación 3)
        mejor_clase = None
        min_d_sigma = np.inf
        clases_degeneradas_train = [] # Resguardo por si un Q_c del dataset tiene un solo rayo
        
        for clase in clases_empatadas:
            v_train = resultados[clase]['varianza']
            
            # Protección por si la clase de entrenamiento tiene varianza 0
            if np.isclose(v_train, 0.0, atol=tolerancia):
                clases_degeneradas_train.append(clase)
                continue
                
            # d_sigma = | var(P_q) / var(Q_c) - 1 |
            d_sigma = abs((var_prueba / v_train) - 1.0)
            if d_sigma < min_d_sigma:
                min_d_sigma = d_sigma
                mejor_clase = clase
                
        # Si todas las clases empatadas del dataset tenían varianza 0 (caso raro pero posible)
        if mejor_clase is None and clases_degeneradas_train:
            print("Aplicando Criterio Secundario debido a varianza cero en clases de entrenamiento")
            mejor_sim_media = -np.inf
            for clase in clases_degeneradas_train:
                c_train = resultados[clase]['centroide']
                sim_media = np.dot(centroide_prueba, c_train)
                if sim_media > mejor_sim_media:
                    mejor_sim_media = sim_media
                    mejor_clase = clase
                    
        print(f"   -> Resuelto con éxito. Ganador: {mejor_clase}")
        return mejor_clase

    # 3. Procesar las tres clasificaciones principales buscando empates
    todas_las_clases = list(resultados.keys())
    
    # Criterio Coseno (Maximizar)
    max_cos = max(resultados[c]['dist_cos'] for c in todas_las_clases)
    empatados_cos = [c for c in todas_las_clases if np.isclose(resultados[c]['dist_cos'], max_cos, atol=tolerancia)]
    mejor_clase_coseno = resolver_empate(empatados_cos, "Score Coseno")

    # Criterio Ángulo Mínimo (Maximizar)
    max_min = max(resultados[c]['dist_min'] for c in todas_las_clases)
    empatados_min = [c for c in todas_las_clases if np.isclose(resultados[c]['dist_min'], max_min, atol=tolerancia)]
    mejor_clase_min = resolver_empate(empatados_min, "Score Mínimo")

    # Criterio Distancia / Similitud (Minimizar)
    min_dist = min(resultados[c]['dist_sim'] for c in todas_las_clases)
    empatados_dist = [c for c in todas_las_clases if np.isclose(resultados[c]['dist_sim'], min_dist, atol=tolerancia)]
    mejor_clase_distancia = resolver_empate(empatados_dist, "Score Distancia")

    return mejor_clase_coseno, mejor_clase_distancia, mejor_clase_min