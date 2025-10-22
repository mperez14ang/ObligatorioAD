# 🏠 Proyecto de Predicción de Precios de Alquiler - Versión Mejorada

## 📌 Descripción
Sistema completo de Machine Learning para predecir precios de alquiler de apartamentos utilizando técnicas avanzadas de análisis de datos, clustering geográfico y múltiples algoritmos de predicción.

## ✨ Mejoras Implementadas

### 1. **Análisis Exploratorio Avanzado**
- Informe automático de calidad de datos
- Detección de outliers con múltiples métodos (IQR, Z-Score, Isolation Forest)
- Análisis de distribuciones con visualizaciones profesionales
- Matriz de correlación interactiva

### 2. **Clustering Geográfico Inteligente**
- Creación automática de "barrios" usando K-Means
- Optimización del número de clusters con Silhouette Score
- Detección de zonas de alta densidad con DBSCAN
- Perfiles detallados de cada barrio (precio promedio, amenidades, clasificación)

### 3. **Feature Engineering Avanzado**
- **Ratios y métricas derivadas:**
  - Precio por pie cuadrado
  - Precio por habitación/baño
  - Ratio precio vs mediana del barrio
  - Desviación del precio respecto al barrio
  
- **Extracción de características de texto:**
  - Detección de palabras clave (renovado, lujo, vista, seguridad)
  - Análisis de longitud y complejidad del texto
  - Extracción automática de amenidades desde descripción

- **Categorización inteligente:**
  - Tamaño (micro, pequeño, mediano, grande)
  - Precio (económico, medio, premium)
  - Barrios (económico, medio-bajo, medio-alto, premium)

### 4. **Imputación Inteligente**
- Imputación contextual basada en texto para amenidades
- KNN Imputer para valores numéricos
- Detección automática de mascotas en descripciones

### 5. **Modelado con 8+ Algoritmos**
- **Modelos lineales:** Linear Regression, Ridge, Lasso, ElasticNet
- **Modelos de árboles:** Decision Tree, Random Forest, Gradient Boosting
- **Modelos avanzados:** XGBoost, LightGBM, CatBoost
- Comparación automática y selección del mejor modelo

### 6. **Sistema de Análisis de Barrios**
- Perfiles detallados de cada barrio
- Sistema de recomendación basado en preferencias
- Detección de anomalías en precios
- Mapa interactivo con clusters y heat map

### 7. **Visualizaciones Profesionales**
- Mapas interactivos con Plotly
- Dashboard de análisis de predicciones
- Análisis de importancia de características
- Visualización de reducción dimensional (PCA, t-SNE)

## 📊 Resultados Esperados

- **R² Score:** 0.70-0.85 (dependiendo de la calidad de los datos)
- **RMSE:** ~15-20% del precio promedio
- **Mejores modelos:** XGBoost, Random Forest, LightGBM

## 🚀 Cómo Ejecutar

### Requisitos
```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost
pip install matplotlib seaborn plotly folium
pip install scipy
```

### Ejecución
```bash
cd /Users/manuelperez/Facultad/AnalisisDeDatos/ObligatorioAD
python proyecto_mejorado.py
```

## 📁 Estructura del Proyecto

```
ObligatorioAD/
├── Resources/
│   ├── Tema_9.csv                    # Dataset principal
│   └── apartment+for+rent+classified/ # Dataset UCI adicional
├── proyecto_mejorado.py              # Script principal mejorado
├── analisis_barrios_avanzado.py      # Módulo de análisis avanzado
├── EDA+PreparamientodeDatos.ipynb    # Notebook original
└── README.md                          # Este archivo
```

## 🎯 Características Clave del Dataset

- **9,950 registros** después de limpieza inicial
- **15 columnas** originales
- **30+ features** después de feature engineering
- **Clustering geográfico** identifica barrios automáticamente
- **Imputación inteligente** reduce valores faltantes a ~0%

## 📈 Pipeline de Procesamiento

1. **Carga de Datos** → Análisis de calidad
2. **Limpieza** → Eliminación de columnas irrelevantes
3. **Feature Engineering** → 30+ nuevas características
4. **Clustering Geográfico** → Identificación de barrios
5. **Imputación** → Valores faltantes < 1%
6. **Outliers** → Manejo conservador (1% extremos)
7. **Selección de Features** → Top 20 por mutual information
8. **Modelado** → 8+ algoritmos con validación cruzada
9. **Evaluación** → Métricas completas (R², RMSE, MAE, MAPE)
10. **Exportación** → Modelos y resultados guardados

## 💡 Insights Clave

1. **La ubicación es crucial:** Los clusters geográficos son una de las características más importantes
2. **Amenidades importantes:** Parking, gimnasio y piscina tienen alto impacto en el precio
3. **Tamaño vs Precio:** Relación no lineal, apartamentos muy grandes pueden tener precio/sqft menor
4. **Detección de gangas:** El sistema identifica propiedades subvaloradas automáticamente

## 🔮 Próximos Pasos Sugeridos

1. **Incorporar datos externos:**
   - Criminalidad por zona
   - Proximidad a transporte público
   - Calidad de escuelas cercanas

2. **Mejorar el modelo:**
   - Ensemble de los mejores modelos
   - Optimización bayesiana de hiperparámetros
   - Validación temporal si hay datos históricos

3. **Productización:**
   - API REST con FastAPI
   - Dashboard interactivo con Streamlit
   - Pipeline automatizado con MLflow

## 📝 Notas Importantes

- El proyecto detecta automáticamente el número óptimo de barrios
- La imputación de amenidades usa análisis de texto en las descripciones
- Los outliers extremos (>99.5 percentil) se manejan conservadoramente
- El sistema de recomendación considera múltiples factores ponderados

## 🎓 Créditos

**Autor:** Manuel Pérez  
**Universidad:** Facultad de Análisis de Datos  
**Fecha:** 2025  
**Basado en:** Mejores prácticas de ML y análisis geoespacial

## 📚 Referencias

- UCI Machine Learning Repository - Apartment Dataset
- Scikit-learn Documentation
- XGBoost, LightGBM, CatBoost Documentation
- Técnicas de clustering geográfico (K-Means, DBSCAN)
- Feature Engineering for Machine Learning

---

**⚡ Nota:** Este proyecto representa una mejora significativa sobre el análisis básico, incorporando técnicas profesionales de ciencia de datos y machine learning.
