# 🤖 TelecomX - Parte 2: Predicción de Cancelación (Churn)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)

## 📋 Descripción del Proyecto

Este proyecto es la **Parte 2** del desafío TelecomX, enfocado en **desarrollar modelos predictivos de Machine Learning** capaces de prever qué clientes tienen mayor probabilidad de cancelar sus servicios (churn).

Utilizamos los datos ya tratados de la **Parte 1** (ETL) para construir un pipeline robusto de modelado predictivo que ayudará a TelecomX a anticiparse al problema de cancelación de clientes.

---

## 🎯 Objetivos del Desafío

✅ **Preparar los datos** para el modelado (tratamiento, codificación, normalización)

✅ **Realizar análisis de correlación** y selección de variables

✅ **Entrenar 2+ modelos** de clasificación

✅ **Evaluar el rendimiento** con métricas completas (Accuracy, Precision, Recall, F1-Score, AUC-ROC)

✅ **Interpretar los resultados** incluyendo importancia de variables

✅ **Crear conclusión estratégica** señalando los principales factores que influyen en la cancelación

---

## 🗂️ Estructura del Proyecto

```
telecomAluraParte02/
│
├── TelecomX_ML_Prediction.ipynb    # Notebook principal con todo el análisis ML
├── preparar_datos.py                # Script para generar datos tratados
├── datos_tratados.csv               # Datos procesados (generado)
├── telecom_data_processed.csv       # Copia de datos (generado)
│
├── best_churn_model.pkl            # Mejor modelo entrenado (generado)
├── scaler.pkl                      # Escalador guardado (generado)
├── modelos_comparacion.csv         # Resultados de todos los modelos (generado)
├── feature_importance.csv          # Importancia de variables (generado)
│
├── README.md                       # Este archivo
├── requirements.txt                # Dependencias del proyecto
└── .gitignore                      # Archivos a ignorar en git
```

---

## 🛠️ Instalación

### Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Jupyter Notebook o JupyterLab
- Datos de la Parte 1 (TelecomX_Data.json)

### Paso 1: Clonar el repositorio

```bash
git clone https://github.com/sakunext/telecom-alura-g9.git
cd telecom-alura-g9/telecomAluraParte02
```

### Paso 2: Crear entorno virtual (recomendado)

```bash
python -m venv venv
```

**Activar el entorno:**

- Windows:
  ```bash
  venv\Scripts\activate
  ```

- Linux/Mac:
  ```bash
  source venv/bin/activate
  ```

### Paso 3: Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## 🚀 Uso

### Opción 1: Ejecutar desde Jupyter Notebook (Recomendado)

1. **Generar datos tratados:**

   ```bash
   python preparar_datos.py
   ```

   Esto creará los archivos `datos_tratados.csv` y `telecom_data_processed.csv`

2. **Abrir Jupyter Notebook:**

   ```bash
   jupyter notebook TelecomX_ML_Prediction.ipynb
   ```

3. **Ejecutar todas las celdas:**
   
   - En Jupyter: `Kernel` → `Restart & Run All`
   - O ejecutar celda por celda con `Shift + Enter`

### Opción 2: Ejecutar paso a paso

**Paso 1: Preparar datos**
```bash
python preparar_datos.py
```

**Paso 2: Abrir el notebook y ejecutar cada fase:**
- Fase 1: Importación de bibliotecas
- Fase 2: Análisis exploratorio
- Fase 3: Preparación de datos
- Fase 4: Análisis de correlación
- Fase 5: División y normalización
- Fase 6: Entrenamiento de modelos
- Fase 7: Evaluación
- Fase 8: Interpretación
- Fase 9: Conclusiones estratégicas
- Fase 10: Guardar resultados

---

## 📊 Metodología

### 1. Preparación de Datos

- ✅ Carga de datos tratados desde la Parte 1
- ✅ Verificación de valores nulos
- ✅ Codificación de variable objetivo (Churn: Yes/No → 1/0)
- ✅ Codificación de variables categóricas (Label Encoding)

### 2. Análisis de Correlación

- ✅ Matriz de correlación de Pearson
- ✅ Identificación de variables más correlacionadas con Churn
- ✅ Mapa de calor de correlaciones
- ✅ Selección de features relevantes

### 3. División y Normalización

- ✅ Split Train/Test: 80% / 20%
- ✅ Estratificación por Churn (mantener proporción)
- ✅ Normalización con StandardScaler (media=0, std=1)

### 4. Modelos Entrenados

Se entrenan y evalúan **5 modelos de clasificación**:

| # | Modelo                    | Tipo                |
|---|---------------------------|---------------------|
| 1 | Logistic Regression       | Lineal              |
| 2 | Decision Tree             | Árbol de decisión   |
| 3 | Random Forest             | Ensemble (Bagging)  |
| 4 | Gradient Boosting         | Ensemble (Boosting) |
| 5 | Support Vector Machine    | Kernel (RBF)        |

### 5. Métricas de Evaluación

Para cada modelo se calculan:

- **Accuracy**: Porcentaje de predicciones correctas
- **Precision**: De los predichos como Churn, cuántos realmente cancelan
- **Recall**: De los que cancelan, cuántos son detectados
- **F1-Score**: Balance entre Precision y Recall
- **AUC-ROC**: Capacidad de distinguir entre clases

### 6. Interpretación

- ✅ Importancia de variables (Feature Importance)
- ✅ Coeficientes de regresión logística
- ✅ Curvas ROC comparativas
- ✅ Matriz de confusión del mejor modelo

---

## 📈 Resultados

### Mejor Modelo

El modelo con mejor desempeño será seleccionado automáticamente según **AUC-ROC**.

**Ejemplo de resultados esperados:**

| Modelo              | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---------------------|----------|-----------|--------|----------|---------|
| Random Forest       | 0.8145   | 0.6792    | 0.5234 | 0.5919   | 0.8456  |
| Gradient Boosting   | 0.8102   | 0.6623    | 0.5156 | 0.5792   | 0.8398  |
| Logistic Regression | 0.8045   | 0.6512    | 0.4892 | 0.5589   | 0.8234  |
| SVM                 | 0.7989   | 0.6345    | 0.4723 | 0.5412   | 0.8156  |
| Decision Tree       | 0.7823   | 0.5989    | 0.4567 | 0.5178   | 0.7891  |

> **Nota:** Los valores exactos dependerán de tus datos reales.

### Factores Clave de Churn

El análisis de importancia de variables revelará los **top factores** que influyen en la cancelación:

1. **Tenure** (Antigüedad del cliente)
2. **Contract** (Tipo de contrato)
3. **MonthlyCharges** (Cargo mensual)
4. **InternetService** (Tipo de servicio de internet)
5. **PaymentMethod** (Método de pago)
6. ... (otros factores)

---

## 💡 Conclusiones Estratégicas

### Recomendaciones para TelecomX

#### 1. 🎯 Estrategia de Retención Predictiva

- Implementar el mejor modelo en producción
- Scoring mensual de todos los clientes activos
- Crear programa de retención para clientes de alto riesgo

#### 2. 📞 Acciones Preventivas

- Contactar proactivamente clientes con probabilidad > 70% de churn
- Ofrecer beneficios personalizados según factores de riesgo
- Monitorear satisfacción de clientes en riesgo

#### 3. 📊 Monitoreo Continuo

- Reentrenar el modelo trimestralmente
- Medir impacto de estrategias de retención
- Dashboard de seguimiento de predicciones vs realidad

#### 4. 🔍 Enfoque en Factores Clave

- Diseñar intervenciones específicas para las variables más importantes
- Mejorar experiencia en contratos mes a mes
- Optimizar precios según segmentos de riesgo

#### 5. 💰 Impacto Económico Esperado

- **Reducción estimada de churn:** 15-25%
- **Aumento en retención:** Mayor Customer Lifetime Value (CLV)
- **ROI esperado:** Positivo en 6-12 meses

---

## 📁 Archivos Generados

Después de ejecutar el notebook, se generarán:

| Archivo                      | Descripción                                    |
|------------------------------|------------------------------------------------|
| `best_churn_model.pkl`       | Mejor modelo entrenado (serializado)          |
| `scaler.pkl`                 | Escalador StandardScaler (para predicciones)  |
| `modelos_comparacion.csv`    | Tabla comparativa de métricas                 |
| `feature_importance.csv`     | Importancia de variables del mejor modelo     |
| `datos_tratados.csv`         | Datos procesados y listos para ML             |

---

## 🔧 Tecnologías Utilizadas

### Lenguaje y Entorno

- **Python 3.8+**: Lenguaje de programación
- **Jupyter Notebook**: Entorno interactivo de análisis

### Bibliotecas de Datos

- **pandas 1.3.0+**: Manipulación de datos
- **numpy 1.21.0+**: Operaciones numéricas

### Visualización

- **matplotlib 3.4.0+**: Gráficos básicos
- **seaborn 0.11.0+**: Visualizaciones estadísticas avanzadas

### Machine Learning

- **scikit-learn 0.24.0+**: 
  - Modelos: LogisticRegression, DecisionTree, RandomForest, GradientBoosting, SVM
  - Preprocessing: StandardScaler, LabelEncoder
  - Métricas: accuracy, precision, recall, f1, AUC-ROC
  - Validación: train_test_split, cross_val_score

---

## 👤 Autor

**Analista Junior de Machine Learning**  
Equipo de Data Science - TelecomX  
Fecha: Marzo 2026

---

## 📄 Licencia

Este proyecto es parte del desafío educativo de Alura LATAM.

---

## 🔗 Enlaces Relacionados

- [Parte 1: Análisis ETL de TelecomX](../telecomAluraParte01/)
- [Documentación de scikit-learn](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)

---

## 🆘 Soporte

Si encuentras algún problema:

1. Verifica que hayas completado la **Parte 1** del desafío
2. Asegúrate de tener el archivo `TelecomX_Data.json` en `../telecomAluraParte01/`
3. Ejecuta `python preparar_datos.py` antes de abrir el notebook
4. Verifica que todas las dependencias estén instaladas: `pip list`

---

## ✅ Checklist de Completitud

- [x] ✅ Preparación de datos para modelado
- [x] ✅ Análisis de correlación y selección de variables
- [x] ✅ Entrenamiento de 5 modelos de clasificación
- [x] ✅ Evaluación con múltiples métricas
- [x] ✅ Interpretación de importancia de variables
- [x] ✅ Conclusiones estratégicas y recomendaciones
- [x] ✅ Documentación completa del proyecto
- [x] ✅ Modelo guardado para producción

---

## 🎉 Felicidades

¡Has completado exitosamente el **Desafío TelecomX - Parte 2: Predicción de Churn**!

Ahora cuentas con:
- ✅ Un modelo predictivo robusto
- ✅ Insights accionables sobre factores de churn
- ✅ Recomendaciones estratégicas para retención
- ✅ Pipeline reproducible para futuras actualizaciones

**¡Estás listo para implementar soluciones de Machine Learning en producción! 🚀**
#
