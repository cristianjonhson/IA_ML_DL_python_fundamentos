"""
Guía práctica en Python:
Fundamentos de IA, Machine Learning y Deep Learning

Objetivo:
- Entender la jerarquía IA → ML → DL
- Ubicar regresión, clasificación y clustering
- Diferenciar datos estructurados y no estructurados
- Comprender dato real, aumentado y sintético
- Ver ejemplos base en Python
"""

# ============================================================
# 1. MAPA CONCEPTUAL: IA, ML Y DL
# ============================================================

conceptos = {
    "IA": {
        "nombre": "Inteligencia Artificial",
        "descripcion": "Campo general que busca crear sistemas capaces de imitar capacidades humanas.",
        "ejemplos": [
            "Chatbots",
            "Sistemas expertos",
            "Asistentes virtuales",
            "Reconocimiento de imágenes",
            "Automatización inteligente"
        ]
    },
    "ML": {
        "nombre": "Machine Learning",
        "descripcion": "Rama de la IA donde los modelos aprenden patrones desde datos.",
        "ejemplos": [
            "Regresión",
            "Clasificación",
            "Clustering",
            "Árboles de decisión",
            "Random Forest"
        ]
    },
    "DL": {
        "nombre": "Deep Learning",
        "descripcion": "Rama del ML basada en redes neuronales profundas.",
        "ejemplos": [
            "CNN",
            "RNN",
            "Transformers",
            "Modelos de lenguaje",
            "Visión por computadora"
        ]
    }
}


def mostrar_mapa_conceptual():
    print("\n=== MAPA CONCEPTUAL IA → ML → DL ===\n")

    for clave, valor in conceptos.items():
        print(f"{clave}: {valor['nombre']}")
        print(f"Descripción: {valor['descripcion']}")
        print("Ejemplos:")
        for ejemplo in valor["ejemplos"]:
            print(f" - {ejemplo}")
        print()


mostrar_mapa_conceptual()


# ============================================================
# 2. TIPOS DE DATOS
# ============================================================

tipos_datos = {
    "estructurados": {
        "descripcion": "Datos organizados en filas y columnas.",
        "ejemplos": [
            "Excel",
            "Bases de datos SQL",
            "Ventas",
            "Clientes",
            "Precios",
            "Inventarios"
        ],
        "modelos_recomendados": [
            "Regresión",
            "Clasificación",
            "Árboles de decisión",
            "Random Forest",
            "XGBoost"
        ]
    },
    "no_estructurados": {
        "descripcion": "Datos que no tienen forma tabular simple.",
        "ejemplos": [
            "Texto",
            "Imágenes",
            "Audio",
            "Video",
            "PDF",
            "Correos electrónicos"
        ],
        "modelos_recomendados": [
            "Deep Learning",
            "Transformers",
            "CNN",
            "Modelos de lenguaje",
            "Modelos multimodales"
        ]
    }
}


def mostrar_tipos_datos():
    print("\n=== TIPOS DE DATOS ===\n")

    for tipo, info in tipos_datos.items():
        print(f"Tipo: {tipo}")
        print(f"Descripción: {info['descripcion']}")
        print("Ejemplos:", ", ".join(info["ejemplos"]))
        print("Modelos recomendados:", ", ".join(info["modelos_recomendados"]))
        print()


mostrar_tipos_datos()


# ============================================================
# 3. DATOS ORGÁNICOS, AUMENTADOS Y SINTÉTICOS
# ============================================================

datos = {
    "dato_real": {
        "nombre": "Dato real, original u observado",
        "descripcion": "Dato capturado desde el mundo real.",
        "ejemplo": "Una venta real registrada en una base de datos."
    },
    "dato_aumentado": {
        "nombre": "Dato aumentado",
        "descripcion": "Dato creado al transformar un dato real.",
        "ejemplo": "Una imagen rotada, recortada o con brillo modificado."
    },
    "dato_sintetico": {
        "nombre": "Dato sintético",
        "descripcion": "Dato generado artificialmente.",
        "ejemplo": "Clientes ficticios creados para probar un modelo."
    }
}


def mostrar_tipos_datos_origen():
    print("\n=== DATO REAL, AUMENTADO Y SINTÉTICO ===\n")

    for clave, valor in datos.items():
        print(f"{valor['nombre']}")
        print(f"Descripción: {valor['descripcion']}")
        print(f"Ejemplo: {valor['ejemplo']}")
        print()


mostrar_tipos_datos_origen()


# ============================================================
# 4. MACHINE LEARNING: REGRESIÓN
# ============================================================

"""
La regresión pertenece a Machine Learning supervisado.

Sirve para predecir valores numéricos continuos.

Ejemplos:
- Precio de una casa
- Temperatura
- Ventas futuras
- Tiempo estimado de entrega
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


def ejemplo_regresion():
    print("\n=== EJEMPLO DE REGRESIÓN ===\n")

    # Dataset simple: metros cuadrados y precio de casas
    data = {
        "metros_cuadrados": [40, 50, 60, 70, 80, 90, 100, 120, 140, 160],
        "habitaciones": [1, 2, 2, 3, 3, 3, 4, 4, 5, 5],
        "precio": [50000, 65000, 70000, 85000, 95000, 105000, 120000, 140000, 165000, 190000]
    }

    df = pd.DataFrame(data)

    print("Datos de ejemplo:")
    print(df)

    X = df[["metros_cuadrados", "habitaciones"]]
    y = df["precio"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    predicciones = modelo.predict(X_test)

    print("\nPredicciones:")
    print(predicciones)

    mae = mean_absolute_error(y_test, predicciones)
    rmse = np.sqrt(mean_squared_error(y_test, predicciones))

    print(f"\nMAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    # Aquí el modelo usa lo que aprendió durante el entrenamiento:
    nueva_casa = pd.DataFrame({
        "metros_cuadrados": [110],
        "habitaciones": [4]
    })

    precio_estimado = modelo.predict(nueva_casa)

    print(f"\nPrecio estimado para una casa de 110 m2 y 4 habitaciones: {precio_estimado[0]:.2f}")

    nuevas_casas = pd.DataFrame({
    "metros_cuadrados": [110, 130, 150],
    "habitaciones": [4, 4, 5]
            })

    precios_estimados = modelo.predict(nuevas_casas)

    print(f"Precios estimados para nuevas casas: {precios_estimados[0]:.2f}, {precios_estimados[1]:.2f}, {precios_estimados[2]:.2f}")


ejemplo_regresion()


# ============================================================
# 5. MACHINE LEARNING: CLASIFICACIÓN
# ============================================================

"""
La clasificación pertenece a Machine Learning supervisado.

Sirve para predecir categorías.

Ejemplos:
- Spam o no spam
- Cliente compra o no compra
- Riesgo bajo, medio o alto
- Enfermedad positiva o negativa
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


def ejemplo_clasificacion():
    print("\n=== EJEMPLO DE CLASIFICACIÓN ===\n")

    # Dataset simple: comportamiento de clientes
    data = {
        "edad": [18, 22, 25, 30, 35, 40, 45, 50, 55, 60],
        "visitas_web": [1, 3, 4, 5, 7, 8, 6, 4, 3, 2],
        "compro": [0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
    }

    df = pd.DataFrame(data)

    print("Datos de ejemplo:")
    print(df)

    X = df[["edad", "visitas_web"]]
    y = df["compro"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    modelo = DecisionTreeClassifier(random_state=42)
    modelo.fit(X_train, y_train)

    predicciones = modelo.predict(X_test)

    accuracy = accuracy_score(y_test, predicciones)

    print(f"\nAccuracy: {accuracy:.2f}")
    print("\nReporte de clasificación:")
    print(classification_report(y_test, predicciones))

    nuevo_cliente = pd.DataFrame({
        "edad": [28],
        "visitas_web": [6]
    })

    resultado = modelo.predict(nuevo_cliente)

    if resultado[0] == 1:
        print("\nPredicción: El cliente probablemente comprará.")
    else:
        print("\nPredicción: El cliente probablemente no comprará.")


ejemplo_clasificacion()


# ============================================================
# 6. MACHINE LEARNING: CLUSTERING
# ============================================================

"""
El clustering pertenece a Machine Learning no supervisado.

Sirve para agrupar datos similares sin tener etiquetas previas.

Ejemplos:
- Segmentar clientes
- Agrupar estudiantes por comportamiento
- Detectar patrones de consumo
"""

from sklearn.cluster import KMeans


def ejemplo_clustering():
    print("\n=== EJEMPLO DE CLUSTERING ===\n")

    data = {
        "ingresos": [300, 400, 500, 1500, 1600, 1700, 3000, 3200, 3500],
        "gasto_mensual": [100, 150, 200, 700, 800, 850, 2000, 2100, 2300]
    }

    df = pd.DataFrame(data)

    print("Datos de clientes:")
    print(df)

    modelo = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["grupo"] = modelo.fit_predict(df[["ingresos", "gasto_mensual"]])

    print("\nClientes agrupados:")
    print(df)


ejemplo_clustering()


# ============================================================
# 7. DATOS NO ESTRUCTURADOS: TEXTO
# ============================================================

"""
El texto es un dato no estructurado.

Para que un modelo de ML pueda trabajar con texto, primero debemos convertirlo
a números. Una técnica clásica es TF-IDF.

En Deep Learning moderno se suelen usar embeddings y Transformers.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def ejemplo_texto_no_estructurado():
    print("\n=== EJEMPLO CON TEXTO NO ESTRUCTURADO ===\n")

    textos = [
        "Me encantó el curso de inteligencia artificial",
        "La clase fue muy mala",
        "Excelente explicación del profesor",
        "No entendí nada y fue aburrido",
        "Muy buen contenido y buenos ejemplos",
        "El material estaba desordenado"
    ]

    etiquetas = [1, 0, 1, 0, 1, 0]

    vectorizador = TfidfVectorizer()
    X = vectorizador.fit_transform(textos)

    modelo = LogisticRegression()
    modelo.fit(X, etiquetas)

    nuevo_texto = ["La explicación fue clara y útil"]
    nuevo_vector = vectorizador.transform(nuevo_texto)

    prediccion = modelo.predict(nuevo_vector)

    if prediccion[0] == 1:
        print("Sentimiento positivo")
    else:
        print("Sentimiento negativo")


ejemplo_texto_no_estructurado()


# ============================================================
# 8. AUMENTO DE DATOS EN TEXTO
# ============================================================

"""
El aumento de datos permite crear nuevas muestras a partir de datos reales.

En texto se puede hacer con:
- Sinónimos
- Parafraseo
- Traducción inversa
- Reordenamiento
- Enmascaramiento de palabras

Aquí se muestra un ejemplo simple y manual.
"""


def aumentar_texto_simple(texto):
    reemplazos = {
        "bueno": "excelente",
        "malo": "deficiente",
        "curso": "programa",
        "profesor": "docente",
        "clase": "sesión"
    }

    texto_aumentado = texto

    for palabra, reemplazo in reemplazos.items():
        texto_aumentado = texto_aumentado.replace(palabra, reemplazo)

    return texto_aumentado


def ejemplo_aumento_texto():
    print("\n=== EJEMPLO DE AUMENTO DE DATOS EN TEXTO ===\n")

    texto_original = "El curso fue bueno y el profesor explicó bien"
    texto_aumentado = aumentar_texto_simple(texto_original)

    print("Texto original:")
    print(texto_original)

    print("\nTexto aumentado:")
    print(texto_aumentado)


ejemplo_aumento_texto()


# ============================================================
# 9. AUMENTO DE DATOS EN IMÁGENES
# ============================================================

"""
En imágenes, el aumento de datos es muy común en Deep Learning.

Transformaciones típicas:
- Rotación
- Recorte
- Cambio de brillo
- Cambio de contraste
- Flip horizontal
- Zoom
- Ruido

Este ejemplo requiere una imagen llamada 'imagen.jpg' en la misma carpeta.
"""

from PIL import Image, ImageEnhance
import os


def ejemplo_aumento_imagen():
    print("\n=== EJEMPLO DE AUMENTO DE DATOS EN IMAGEN ===\n")

    ruta_imagen = "imagen.jpg"

    if not os.path.exists(ruta_imagen):
        print("No se encontró 'imagen.jpg'.")
        print("Para probar este ejemplo, coloca una imagen con ese nombre en la carpeta del script.")
        return

    imagen = Image.open(ruta_imagen)

    imagen_rotada = imagen.rotate(15)
    imagen_flip = imagen.transpose(Image.FLIP_LEFT_RIGHT)

    enhancer = ImageEnhance.Brightness(imagen)
    imagen_brillo = enhancer.enhance(1.5)

    imagen_rotada.save("imagen_rotada.jpg")
    imagen_flip.save("imagen_flip.jpg")
    imagen_brillo.save("imagen_brillo.jpg")

    print("Imágenes aumentadas guardadas:")
    print("- imagen_rotada.jpg")
    print("- imagen_flip.jpg")
    print("- imagen_brillo.jpg")


ejemplo_aumento_imagen()


# ============================================================
# 10. DATO SINTÉTICO
# ============================================================

"""
Un dato sintético es generado artificialmente.

Puede servir para:
- Pruebas
- Simulaciones
- Entrenamiento cuando hay pocos datos reales
- Balanceo de clases

Pero siempre se debe validar que represente correctamente la realidad.
"""


def generar_datos_sinteticos_clientes(cantidad=10):
    np.random.seed(42)

    datos_sinteticos = pd.DataFrame({
        "edad": np.random.randint(18, 65, cantidad),
        "ingresos": np.random.randint(300000, 3000000, cantidad),
        "visitas_web": np.random.randint(1, 20, cantidad),
        "compro": np.random.randint(0, 2, cantidad)
    })

    return datos_sinteticos


def ejemplo_datos_sinteticos():
    print("\n=== EJEMPLO DE DATOS SINTÉTICOS ===\n")

    df_sintetico = generar_datos_sinteticos_clientes(10)

    print(df_sintetico)


ejemplo_datos_sinteticos()


# ============================================================
# 11. TABLA FINAL DE UBICACIÓN DE CONCEPTOS
# ============================================================

def tabla_resumen():
    print("\n=== TABLA RESUMEN DE CONCEPTOS ===\n")

    resumen = pd.DataFrame([
        {
            "concepto": "IA",
            "nivel": "General",
            "descripcion": "Campo amplio de sistemas inteligentes"
        },
        {
            "concepto": "ML",
            "nivel": "Intermedio",
            "descripcion": "Modelos que aprenden desde datos"
        },
        {
            "concepto": "DL",
            "nivel": "Avanzado",
            "descripcion": "Redes neuronales profundas"
        },
        {
            "concepto": "Regresión",
            "nivel": "ML supervisado",
            "descripcion": "Predice valores numéricos"
        },
        {
            "concepto": "Clasificación",
            "nivel": "ML supervisado",
            "descripcion": "Predice categorías"
        },
        {
            "concepto": "Clustering",
            "nivel": "ML no supervisado",
            "descripcion": "Agrupa datos similares"
        },
        {
            "concepto": "Datos estructurados",
            "nivel": "Datos",
            "descripcion": "Filas y columnas"
        },
        {
            "concepto": "Datos no estructurados",
            "nivel": "Datos",
            "descripcion": "Texto, imagen, audio o video"
        },
        {
            "concepto": "Dato real",
            "nivel": "Origen de datos",
            "descripcion": "Dato capturado desde el mundo real"
        },
        {
            "concepto": "Dato aumentado",
            "nivel": "Estrategia de datos",
            "descripcion": "Variación de un dato real"
        },
        {
            "concepto": "Dato sintético",
            "nivel": "Estrategia de datos",
            "descripcion": "Dato generado artificialmente"
        }
    ])

    print(resumen)


tabla_resumen()


# ============================================================
# 12. CONCLUSIÓN
# ============================================================

def conclusion():
    print("\n=== CONCLUSIÓN ===\n")

    print("""
La Inteligencia Artificial es el campo más amplio.

Machine Learning es una rama de la IA que permite aprender desde datos.

Deep Learning es una rama del Machine Learning que usa redes neuronales profundas.

La regresión, clasificación y clustering pertenecen a Machine Learning.

Los datos estructurados suelen trabajarse muy bien con Machine Learning clásico.

Los datos no estructurados, como texto, imagen, audio y video, suelen beneficiarse más de Deep Learning.

El dato real es la base del entrenamiento.

El dato aumentado mejora la diversidad del conjunto de datos.

El dato sintético puede complementar, pero debe validarse cuidadosamente.
""")


conclusion()
