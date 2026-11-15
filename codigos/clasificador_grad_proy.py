
import numpy as np
import time
from scipy.optimize import nnls
from concurrent.futures import ThreadPoolExecutor, as_completed

_TOL = 1e-12


# ============================================================
# 1. FUNCIONES AUXILIARES
# ============================================================

def _preparar_cono(G):
    """
    Normaliza los generadores del cono.

    Las columnas de G son los rayos generadores.
    Elimina las columnas nulas.
    """
    G = np.asarray(G, dtype=float)

    if G.ndim == 1:
        G = G[:, None]

    if G.ndim != 2 or min(G.shape) == 0:
        raise ValueError("El cono debe contener generadores.")

    if not np.isfinite(G).all():
        raise ValueError("Los generadores deben ser finitos.")

    normas = np.linalg.norm(G, axis=0)
    G = G[:, normas > _TOL]

    if G.shape[1] == 0:
        raise ValueError("Todos los generadores son nulos.")

    return G / np.linalg.norm(G, axis=0, keepdims=True)


def proj_cone(G, u):
    """
    Proyección euclidiana sobre P = cone(G)
    mediante mínimos cuadrados no negativos.
    """
    u = np.asarray(u, dtype=float).reshape(-1)

    if G.shape[0] != u.size:
        raise ValueError("Dimensiones incompatibles.")

    coef, _ = nnls(G, u)

    return G @ coef


def es_polar(u, H, tol=1e-9):
    """
    Determina si u pertenece al cono polar de cone(H).
    """
    return bool(np.all(H.T @ u <= tol))


def f_obj(u, H):
    """
    Función objetivo:

        F(u) = 1/2 ||Proj_Q(u)||²
    """
    v = proj_cone(H, u)
    return 0.5 * float(v @ v)


def gradiente(u, H):
    """
    Gradiente de F:

        grad F(u) = Proj_Q(u)
    """
    return proj_cone(H, u)


def _retraer(G, x):
    """
    Proyecta x sobre cone(G) y normaliza
    el resultado a norma unitaria.
    """
    u = proj_cone(G, x)
    norma = np.linalg.norm(u)

    if norma <= _TOL:
        return None

    return u / norma


# ============================================================
# 2. DESCENSO DE GRADIENTE PROYECTADO
# ============================================================

def descenso_gradiente_proyectado(
    G,
    H,
    u_curr,
    t_act=0.1,
    beta=0.5,
    max_backtracks=35
):
    """
    Realiza una iteración de descenso proyectado.

    1. Calcula el gradiente.
    2. Ejecuta búsqueda lineal.
    3. Proyecta sobre cone(G).
    4. Normaliza.
    5. Comprueba disminución del objetivo.
    """

    g = gradiente(u_curr, H)
    anterior = 0.5 * float(g @ g)

    if anterior < _TOL**2:
        return u_curr.copy(), 0.0

    paso = float(t_act)

    for _ in range(max_backtracks):

        # Paso de gradiente
        us = u_curr - paso * g

        # Proyección y normalización
        nuevo = _retraer(G, us)

        if nuevo is not None:

            d = nuevo - u_curr

            # Condición de descenso suficiente
            condicion = (
                f_obj(nuevo, H)
                <= anterior
                - 1e-4 * float(d @ d) / max(paso, _TOL)
                + 1e-14
            )

            if condicion:
                return nuevo, paso

        paso *= beta

    return u_curr.copy(), 0.0


# ============================================================
# 3. EXPLORACIÓN TIPO MONTE CARLO
# ============================================================

def _muestra_montecarlo(
    G,
    H,
    rng,
    n_muestras,
    n_lotes,
    mejor_valor
):
    """
    Explora puntos aleatorios factibles de cone(G).

    Cada punto se genera mediante:

        u = G @ alpha / ||G @ alpha||

    donde alpha contiene coeficientes positivos.

    Si encuentra un punto con menor función objetivo,
    lo devuelve como nuevo candidato.
    """

    mejor_u = None
    evaluadas = 0

    for _ in range(n_lotes):

        # Coeficientes aleatorios positivos
        coeficientes = rng.uniform(
            0.0,
            1.0,
            size=(n_muestras, G.shape[1])
        )

        # Generación de puntos del cono
        puntos = coeficientes @ G.T

        normas = np.linalg.norm(puntos, axis=1)

        for u, norma in zip(puntos, normas):

            if norma <= _TOL:
                continue

            u = u / norma
            evaluadas += 1

            valor = f_obj(u, H)

            # Actualizar solo si mejora
            if valor < mejor_valor - 1e-14:

                mejor_valor = valor
                mejor_u = u.copy()

    return mejor_u, mejor_valor, evaluadas


# ============================================================
# 4. PGD + MONTE CARLO
# ============================================================

def exceso_gradiente_proyectado(
    G,
    H,
    x0=None,
    t=0.1,
    max_iter=100,
    eps=1e-6,
    mc_muestras=100,
    mc_lotes=2,
    mc_cada=5,
    random_state=None
):
    """
    Optimización mediante descenso de gradiente proyectado
    y exploración aleatoria tipo Monte Carlo.

    Monte Carlo se ejecuta cada mc_cada iteraciones
    o cuando PGD se estanca.

    Si encuentra un punto mejor, PGD continúa desde él.
    """

    tiempo_inicio = time.perf_counter()

    G = _preparar_cono(G)
    H = _preparar_cono(H)

    if G.shape[0] != H.shape[0]:
        raise ValueError("Los conos deben tener igual dimensión.")

    if (
        t <= 0 or max_iter < 1 or eps <= 0
        or mc_muestras < 0 or mc_lotes < 0
        or mc_cada < 1
    ):
        raise ValueError("Parámetros inválidos.")

    p = G.shape[1]

    # --------------------------------------------------------
    # Punto inicial
    # --------------------------------------------------------

    if x0 is None:
        pesos = np.ones(p)
    else:
        pesos = np.asarray(x0, dtype=float).reshape(-1)

        if (
            pesos.size != p
            or not np.isfinite(pesos).all()
            or np.any(pesos < 0)
        ):
            raise ValueError(
                "x0 debe contener coeficientes no negativos."
            )

    u = G @ pesos

    if np.linalg.norm(u) <= _TOL:
        u = G[:, 0].copy()
    else:
        u = u / np.linalg.norm(u)

    # --------------------------------------------------------
    # Variables de seguimiento
    # --------------------------------------------------------

    rng = np.random.default_rng(random_state)

    paso = float(t)

    historial_objetivo = [f_obj(u, H)]
    valores_normas = []

    evaluaciones_mc = 0
    escapes_mc = 0

    converged = False
    motivo = "max_iter"
    k = 0

    # --------------------------------------------------------
    # Bucle principal
    # --------------------------------------------------------

    for k in range(1, max_iter + 1):

        # Condición polar
        if es_polar(u, H):

            converged = True
            motivo = "polar"
            break

        anterior = f_obj(u, H)

        # ----------------------------------------------------
        # Paso PGD
        # ----------------------------------------------------

        nuevo, paso_aceptado = descenso_gradiente_proyectado(
            G,
            H,
            u,
            paso
        )

        nuevo_valor = f_obj(nuevo, H)

        desplazamiento = float(
            np.linalg.norm(nuevo - u)
        )

        sin_progreso = (
            paso_aceptado == 0
            or desplazamiento < eps
        )

        hubo_escape = False

        # ----------------------------------------------------
        # Exploración Monte Carlo
        # ----------------------------------------------------

        ejecutar_mc = (
            mc_muestras > 0
            and mc_lotes > 0
            and (
                k % mc_cada == 0
                or sin_progreso
            )
        )

        if ejecutar_mc:

            muestra, valor_mc, n = _muestra_montecarlo(
                G,
                H,
                rng,
                mc_muestras,
                mc_lotes,
                min(anterior, nuevo_valor)
            )

            evaluaciones_mc += n

            if muestra is not None:

                # Conservamos la mejor muestra encontrada
                nuevo = muestra
                nuevo_valor = valor_mc

                desplazamiento = float(
                    np.linalg.norm(nuevo - u)
                )

                hubo_escape = True
                escapes_mc += 1

                # Reiniciar paso después del escape
                paso = float(t)

        # ----------------------------------------------------
        # Protección contra incrementos del objetivo
        # ----------------------------------------------------

        if nuevo_valor > anterior + 1e-12:

            nuevo = u.copy()
            nuevo_valor = anterior
            desplazamiento = 0.0

        if paso_aceptado > 0 and not hubo_escape:

            paso = min(
                max(float(t), 1.0),
                1.5 * paso_aceptado
            )

        # ----------------------------------------------------
        # Actualización definitiva
        # ----------------------------------------------------

        u = nuevo

        valores_normas.append(desplazamiento)
        historial_objetivo.append(nuevo_valor)

        # ----------------------------------------------------
        # Convergencia
        # ----------------------------------------------------

        if sin_progreso and not hubo_escape:

            converged = paso_aceptado > 0

            motivo = (
                "desplazamiento_pequeno"
                if converged
                else "sin_paso_aceptable"
            )

            break

    # --------------------------------------------------------
    # Resultado
    # --------------------------------------------------------

    proy = proj_cone(H, u)
    norma = float(np.linalg.norm(proy))

    if norma > _TOL:
        v = proy / norma
    else:
        v = None

    tiempo_final = time.perf_counter() - tiempo_inicio

    return {
        "u_opt": u,
        "v_opt": v,
        "coseno": float(np.clip(norma, 0.0, 1.0)),
        "objetivo": f_obj(u, H),
        "tiempo_final": tiempo_final,
        "k": k,
        "converged": converged,
        "motivo_parada": motivo,
        "valores_normas": valores_normas,
        "historial_objetivo": historial_objetivo,
        "evaluaciones_mc": evaluaciones_mc,
        "escapes_mc": escapes_mc
    }


# ============================================================
# 5. DISTANCIA ENTRE CONOS
# ============================================================

def f_dist(a1, a2):
    """
    Distancia bidireccional equivalente a:

    2 * sqrt(
        sin(arccos(a1)/2)^2 +
        sin(arccos(a2)/2)^2
    )

    Se evita calcular arccos directamente.
    """

    a1, a2 = np.clip([a1, a2], 0.0, 1.0)

    return float(
        np.sqrt(
            2 * (1 - a1)
            + 2 * (1 - a2)
        )
    )


# ============================================================
# 6. CENTROIDES Y VARIANZAS ANGULARES
# ============================================================

def calcular_centroide_y_varianza(cono):
    """
    Calcula el centroide direccional y la varianza
    angular de los rayos generadores.

    Se utilizan únicamente para resolver empates.
    """

    G = _preparar_cono(cono)

    centro = G.sum(axis=1)
    norma = np.linalg.norm(centro)

    if norma <= _TOL:
        return np.zeros(G.shape[0]), np.nan

    centro = centro / norma

    productos = G.T @ centro

    theta = np.arccos(
        np.clip(productos, -1.0, 1.0)
    )

    varianza = float(np.mean(theta**2))

    return centro, varianza


# ============================================================
# 7. EVALUACIÓN DE UNA CLASE
# ============================================================

def evaluar_clase(
    clase,
    cono_train,
    cono_prueba,
    **pgd_kwargs
):
    """
    Evalúa la similitud entre el cono de prueba
    y el cono de entrenamiento.

    Ejecuta PGD en ambas direcciones.
    """

    # Dirección prueba -> entrenamiento
    res1 = exceso_gradiente_proyectado(
        cono_prueba,
        cono_train,
        **pgd_kwargs
    )

    # Dirección entrenamiento -> prueba
    res2 = exceso_gradiente_proyectado(
        cono_train,
        cono_prueba,
        **pgd_kwargs
    )

    centro, varianza = calcular_centroide_y_varianza(
        cono_train
    )

    resultados = {
        "dist_cos": res1["coseno"],

        "dist_sim": f_dist(
            res1["coseno"],
            res2["coseno"]
        ),

        "centroide": centro,
        "varianza": varianza,

        "coseno_prueba_train": res1["coseno"],
        "coseno_train_prueba": res2["coseno"],

        "tiempo_pgd": (
            res1["tiempo_final"]
            + res2["tiempo_final"]
        ),

        "iteraciones": (
            res1["k"],
            res2["k"]
        ),

        "converged": (
            res1["converged"],
            res2["converged"]
        ),

        "evaluaciones_mc": (
            res1["evaluaciones_mc"]
            + res2["evaluaciones_mc"]
        ),

        "escapes_mc": (
            res1["escapes_mc"]
            + res2["escapes_mc"]
        )
    }

    return clase, resultados


# ============================================================
# 8. CRITERIOS DE DESEMPATE
# ============================================================

def _resolver_empate(
    empatadas,
    resultados,
    centro_prueba,
    var_prueba,
    tolerancia
):
    """
    Conserva los criterios originales:

    1. Si la varianza de prueba no es cero,
       comparar las varianzas relativas.

    2. Si la varianza es cero o persiste el empate,
       utilizar similitud entre centroides.
    """

    if len(empatadas) == 1:
        return empatadas[0]

    # --------------------------------------------------------
    # Criterio por varianza angular
    # --------------------------------------------------------

    if (
        np.isfinite(var_prueba)
        and not np.isclose(
            var_prueba,
            0.0,
            atol=tolerancia
        )
    ):

        distancias = {}

        for clase in empatadas:

            var_train = resultados[clase]["varianza"]

            if (
                np.isfinite(var_train)
                and not np.isclose(
                    var_train,
                    0.0,
                    atol=tolerancia
                )
            ):

                distancias[clase] = abs(
                    var_prueba / var_train - 1.0
                )

        if distancias:

            minimo = min(distancias.values())

            empatadas = [
                c for c in empatadas
                if c in distancias
                and np.isclose(
                    distancias[c],
                    minimo,
                    atol=tolerancia,
                    rtol=0
                )
            ]

            if len(empatadas) == 1:
                return empatadas[0]

    # --------------------------------------------------------
    # Criterio por similitud de centroides
    # --------------------------------------------------------

    similitudes = {
        c: float(
            centro_prueba @ resultados[c]["centroide"]
        )
        for c in empatadas
    }

    mejor = max(similitudes.values())

    empatadas = [
        c for c in empatadas
        if np.isclose(
            similitudes[c],
            mejor,
            atol=tolerancia,
            rtol=0
        )
    ]

    # Si el empate es total, se conserva el orden de entrada
    return empatadas[0]


# ============================================================
# 9. CLASIFICADOR PRINCIPAL
# ============================================================

def clasificador(
    cono_prueba,
    conos_entrenamiento,
    tolerancia=1e-7,
    max_workers=1,
    devolver_detalles=False,
    **pgd_kwargs
):
    """
    Clasifica un cono de prueba respecto de los conos
    de entrenamiento.

    Criterios:

    1. Máximo score coseno.
    2. Mínima distancia bidireccional.

    Ambos scores se obtienen únicamente mediante PGD
    con exploración Monte Carlo.

    Retorno:
        mejor_clase_coseno,
        mejor_clase_distancia,
        tiempo_total

    Si devolver_detalles=True, también retorna
    un diccionario con los resultados por clase.
    """

    if not conos_entrenamiento:
        raise ValueError(
            "Se requiere al menos un cono de entrenamiento."
        )

    if tolerancia < 0 or max_workers < 1:
        raise ValueError("Parámetros inválidos.")

    tiempo_inicio = time.perf_counter()

    prueba = _preparar_cono(cono_prueba)

    centro_prueba, var_prueba = (
        calcular_centroide_y_varianza(prueba)
    )

    clases = list(conos_entrenamiento)
    resultados = {}

    # --------------------------------------------------------
    # Evaluación de cada clase
    # --------------------------------------------------------

    if max_workers == 1:

        for clase in clases:

            _, detalle = evaluar_clase(
                clase,
                conos_entrenamiento[clase],
                prueba,
                **pgd_kwargs
            )

            resultados[clase] = detalle

    else:

        with ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:

            futuros = {
                executor.submit(
                    evaluar_clase,
                    clase,
                    conos_entrenamiento[clase],
                    prueba,
                    **pgd_kwargs
                ): clase
                for clase in clases
            }

            for futuro in as_completed(futuros):

                clase, detalle = futuro.result()
                resultados[clase] = detalle

    # --------------------------------------------------------
    # Clasificación por score coseno
    # --------------------------------------------------------

    max_cos = max(
        resultados[c]["dist_cos"]
        for c in clases
    )

    empatadas_cos = [
        c for c in clases
        if np.isclose(
            resultados[c]["dist_cos"],
            max_cos,
            atol=tolerancia,
            rtol=0
        )
    ]

    mejor_clase_coseno = _resolver_empate(
        empatadas_cos,
        resultados,
        centro_prueba,
        var_prueba,
        tolerancia
    )

    # --------------------------------------------------------
    # Clasificación por distancia bidireccional
    # --------------------------------------------------------

    min_dist = min(
        resultados[c]["dist_sim"]
        for c in clases
    )

    empatadas_dist = [
        c for c in clases
        if np.isclose(
            resultados[c]["dist_sim"],
            min_dist,
            atol=tolerancia,
            rtol=0
        )
    ]

    mejor_clase_distancia = _resolver_empate(
        empatadas_dist,
        resultados,
        centro_prueba,
        var_prueba,
        tolerancia
    )

    tiempo_total = (
        time.perf_counter() - tiempo_inicio
    )

    if devolver_detalles:

        return (
            mejor_clase_coseno,
            mejor_clase_distancia,
            tiempo_total,
            resultados
        )

    return (
        mejor_clase_coseno,
        mejor_clase_distancia,
        tiempo_total
    )


# ============================================================
# 10. EJEMPLO DE USO
# ============================================================

if __name__ == "__main__":

    e1, e2 = np.eye(2)

    conos_entrenamiento = {
        "clase_x": np.column_stack([
            e1,
            [1.0, 0.2]
        ]),

        "clase_y": np.column_stack([
            e2,
            [0.2, 1.0]
        ])
    }

    cono_prueba = np.column_stack([
        [1.0, 0.02],
        [1.0, 0.08]
    ])

    clase_cos, clase_dist, tiempo, detalles = clasificador(
        cono_prueba,
        conos_entrenamiento,
        t=0.1,
        max_iter=500,
        eps=1e-6,
        mc_muestras=1000,
        mc_lotes=5,
        mc_cada=1,
        random_state=42,
        devolver_detalles=True
    )

    print("Clase por coseno:", clase_cos)
    print("Clase por distancia:", clase_dist)
    print("Tiempo:", tiempo)

    for clase, resultado in detalles.items():

        print(f"\nClase: {clase}")
        print("Score coseno:", resultado["dist_cos"])
        print("Distancia:", resultado["dist_sim"])
        print("Iteraciones:", resultado["iteraciones"])
        print("Evaluaciones MC:", resultado["evaluaciones_mc"])
        print("Escapes MC:", resultado["escapes_mc"])
