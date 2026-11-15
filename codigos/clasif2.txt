import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from optimizador import *
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
        b = ProjConeUnitSphere(a, cono_train).ravel()
        b1 = ProjConeUnitSphere(-a, cono_train).ravel()
        distancia_cos = np.dot(a, b)
        distancia_sim = f_dist(np.dot(a, b), np.dot(a1, b1))
        distancia_min = distancia_cos
    else:
        # Sim1
        a, b, c = SimCPfmincon(cono_prueba, cono_train)
        # Sim2
        a1, b1, c1 = SimCPfmincon(cono_train, cono_prueba)
        u_min, v_min, coseno_max = min_angle(cono_train, cono_prueba)
        # Normalización
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        a1 = a1 / np.linalg.norm(a1)
        b1 = b1 / np.linalg.norm(b1)
        producto2 = np.dot(a1, b)
        
        if abs(coseno_max - producto2) <= 0.01:
            print("el calculo del algoritmo de proyecciones es aproximado al de calcular <u2 , v1>")
        else:
            print("el calculo del algoritmo de proyecciones NO es aproximado al de calcular <u2 , v1>")
        if coseno_max > producto2:
            print("el algoritmo SÍ es mejor que <u2 , v1>")
        else:
            print("el algoritmo NO es mejor que <u2 , v1>")

        # Cálculos
        distancia_cos = np.dot(a, b)
        distancia_sim = f_dist(np.dot(a, b), np.dot(a1, b1))
        distancia_min = coseno_max
        
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
                    'dist_min': dist_min,
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
    max_min = max(resultados[c]['dist_min'] for c in todas_las_clases)
    empatados_min = [c for c in todas_las_clases if np.isclose(resultados[c]['dist_min'], max_min, atol=tolerancia)]
    mejor_clase_min = resolver_empate(empatados_min, "Score Mínimo")

    # Criterio Distancia / Similitud (Minimizar)
    min_dist = min(resultados[c]['dist_sim'] for c in todas_las_clases)
    empatados_dist = [c for c in todas_las_clases if np.isclose(resultados[c]['dist_sim'], min_dist, atol=tolerancia)]
    mejor_clase_distancia = resolver_empate(empatados_dist, "Score Distancia")

    # FIN DE MEDICIÓN: Calculamos los segundos totales transcurridos
    tiempo_total = time.perf_counter() - tiempo_inicio

    # Ahora la función retorna también el tiempo_total al final
    return mejor_clase_coseno, mejor_clase_distancia, mejor_clase_min, tiempo_total


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