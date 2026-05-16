# Guía práctica de IA, Machine Learning y Deep Learning con Python

Este proyecto contiene una guía práctica en Python para comprender los fundamentos de:

* Inteligencia Artificial
* Machine Learning
* Deep Learning
* Regresión
* Clasificación
* Clustering
* Datos estructurados
* Datos no estructurados
* Aumento de datos
* Datos reales, aumentados y sintéticos

El objetivo es que los estudiantes puedan aprender los conceptos base de IA no solo de forma teórica, sino también ejecutando ejemplos simples en Python.

---

## 1. Objetivo del proyecto

Este repositorio busca explicar de forma progresiva la relación entre:

```text
Inteligencia Artificial → Machine Learning → Deep Learning
```

Además, muestra ejemplos prácticos para entender dónde se ubican técnicas como:

* Regresión lineal
* Clasificación
* Clustering
* Procesamiento de texto
* Aumento de datos
* Generación de datos sintéticos

---

## 2. Estructura conceptual

La guía se basa en la siguiente jerarquía:

```text
Inteligencia Artificial
│
├── Machine Learning
│   │
│   ├── Aprendizaje supervisado
│   │   ├── Regresión
│   │   └── Clasificación
│   │
│   ├── Aprendizaje no supervisado
│   │   └── Clustering
│   │
│   └── Deep Learning
│       ├── Redes neuronales profundas
│       ├── CNN
│       ├── RNN
│       └── Transformers
│
└── IA simbólica, reglas y sistemas expertos
```

---

## 3. Requisitos

Para ejecutar este proyecto necesitas tener instalado:

* Python 3.10 o superior
* pip
* Entorno virtual recomendado

---

## 4. Crear entorno virtual

### En Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### En macOS o Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 5. Instalar dependencias

Ejecuta el siguiente comando:

```bash
pip install numpy pandas scikit-learn matplotlib pillow
```

También puedes crear un archivo `requirements.txt` con este contenido:

```txt
numpy
pandas
scikit-learn
matplotlib
pillow
```

Y luego instalarlo con:

```bash
pip install -r requirements.txt
```

---

## 6. Ejecutar el proyecto

El archivo principal es:

```text
guia_ia_ml_dl.py
```

Para ejecutarlo:

```bash
python guia_ia_ml_dl.py
```

En macOS o Linux también puedes usar:

```bash
python3 guia_ia_ml_dl.py
```

---

## 7. Contenido de la guía

El script incluye los siguientes módulos prácticos:

| Sección | Tema                  | Descripción                                       |
| ------- | --------------------- | ------------------------------------------------- |
| 1       | IA, ML y DL           | Explica la jerarquía conceptual                   |
| 2       | Tipos de datos        | Diferencia datos estructurados y no estructurados |
| 3       | Origen de datos       | Explica dato real, aumentado y sintético          |
| 4       | Regresión             | Predicción de valores numéricos                   |
| 5       | Clasificación         | Predicción de categorías                          |
| 6       | Clustering            | Agrupamiento de datos sin etiquetas               |
| 7       | Texto no estructurado | Procesamiento básico de texto con TF-IDF          |
| 8       | Aumento de texto      | Ejemplo simple de reemplazo por sinónimos         |
| 9       | Aumento de imagen     | Rotación, flip y cambio de brillo                 |
| 10      | Datos sintéticos      | Generación artificial de datos de clientes        |
| 11      | Tabla resumen         | Comparación final de conceptos                    |
| 12      | Conclusión            | Resumen general del aprendizaje                   |

---

## 8. Conceptos principales

### Inteligencia Artificial

La Inteligencia Artificial es el campo general que busca crear sistemas capaces de realizar tareas que normalmente requieren inteligencia humana.

Ejemplos:

* Chatbots
* Sistemas expertos
* Asistentes virtuales
* Reconocimiento de imágenes
* Automatización inteligente

---

### Machine Learning

Machine Learning es una rama de la IA donde los modelos aprenden patrones a partir de datos.

Ejemplos:

* Regresión
* Clasificación
* Clustering
* Árboles de decisión
* Random Forest

---

### Deep Learning

Deep Learning es una rama del Machine Learning basada en redes neuronales profundas.

Ejemplos:

* CNN
* RNN
* Transformers
* Modelos de lenguaje
* Visión por computadora

---

## 9. Ejemplos incluidos

### Regresión

La regresión se utiliza para predecir valores numéricos.

Ejemplo del proyecto:

```text
Predecir el precio de una casa según sus metros cuadrados y número de habitaciones.
```

---

### Clasificación

La clasificación se utiliza para predecir categorías.

Ejemplo del proyecto:

```text
Predecir si un cliente comprará o no comprará.
```

---

### Clustering

El clustering permite agrupar datos similares sin etiquetas previas.

Ejemplo del proyecto:

```text
Agrupar clientes según ingresos y gasto mensual.
```

---

### Texto no estructurado

El texto es un dato no estructurado. Para usarlo en Machine Learning, primero debe convertirse en números.

Ejemplo del proyecto:

```text
Clasificar comentarios como positivos o negativos usando TF-IDF.
```

---

### Aumento de datos

El aumento de datos permite crear variaciones a partir de datos existentes.

Ejemplos:

* Cambiar palabras por sinónimos
* Rotar imágenes
* Cambiar brillo
* Agregar ruido
* Crear variaciones de muestras reales

---

### Datos sintéticos

Los datos sintéticos son datos generados artificialmente.

Ejemplo del proyecto:

```text
Crear clientes ficticios con edad, ingresos, visitas web y compra.
```

---

## 10. Imagen para aumento de datos

El script incluye un ejemplo de aumento de imágenes.

Para probarlo, coloca una imagen llamada:

```text
imagen.jpg
```

en la misma carpeta del archivo `guia_ia_ml_dl.py`.

Luego ejecuta el script y se generarán archivos como:

```text
imagen_rotada.jpg
imagen_flip.jpg
imagen_brillo.jpg
```

---

## 11. Posible estructura del repositorio

```text
guia-ia-ml-dl-python/
│
├── guia_ia_ml_dl.py
├── requirements.txt
├── README.md
├── imagen.jpg
│
└── outputs/
    ├── imagen_rotada.jpg
    ├── imagen_flip.jpg
    └── imagen_brillo.jpg
```

---

## 12. Crear archivo requirements.txt

Puedes crear el archivo manualmente:

```txt
numpy
pandas
scikit-learn
matplotlib
pillow
```

O generarlo desde el entorno virtual:

```bash
pip freeze > requirements.txt
```

---

## 13. Recomendación para estudiantes

Se recomienda estudiar el proyecto en este orden:

1. Leer la explicación de IA, ML y DL.
2. Ejecutar el script completo.
3. Revisar la salida en consola.
4. Modificar los datos de ejemplo.
5. Cambiar los modelos.
6. Probar nuevas predicciones.
7. Agregar nuevos ejemplos.
8. Crear gráficos para visualizar resultados.

---

## 14. Actividades sugeridas

### Actividad 1

Modificar el ejemplo de regresión para predecir el valor de un automóvil usando:

* Año
* Kilometraje
* Marca
* Precio

---

### Actividad 2

Modificar el ejemplo de clasificación para predecir si un estudiante aprueba o reprueba usando:

* Horas de estudio
* Asistencia
* Nota anterior

---

### Actividad 3

Modificar el ejemplo de clustering para agrupar estudiantes según:

* Tiempo de estudio
* Promedio de notas
* Participación en clases

---

### Actividad 4

Agregar más frases al ejemplo de análisis de texto y probar si el modelo mejora.

---

### Actividad 5

Probar el aumento de datos con diferentes imágenes.

---

## 15. Buenas prácticas

* Usar entorno virtual.
* Separar datos, código y resultados.
* No subir archivos pesados innecesarios.
* Documentar cada ejemplo.
* Explicar cada concepto antes del código.
* Validar los datos sintéticos antes de usarlos.
* No asumir que Deep Learning siempre es la mejor solución.
* Priorizar modelos simples cuando el problema lo permita.

---

## 16. Ideas para mejorar el proyecto

Algunas mejoras posibles son:

* Agregar gráficos con Matplotlib.
* Separar cada ejemplo en archivos distintos.
* Crear notebooks de Jupyter.
* Agregar ejemplos con redes neuronales.
* Agregar un ejemplo con TensorFlow o PyTorch.
* Crear una carpeta de datasets.
* Crear ejercicios con soluciones.
* Agregar preguntas tipo cuestionario.

---

## 18. Autor

Proyecto creado por Cristian Jonhson como guía educativa para estudiar los fundamentos de Inteligencia Artificial, Machine Learning y Deep Learning usando Python.
