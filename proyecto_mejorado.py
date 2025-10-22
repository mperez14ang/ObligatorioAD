#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PROYECTO DE PREDICCIÓN DE PRECIOS DE ALQUILER - VERSIÓN MEJORADA
Autor: Manuel Pérez
Fecha: 2025
Universidad: Facultad de Análisis de Datos

Este proyecto implementa un sistema completo de machine learning para predecir
precios de alquiler utilizando técnicas avanzadas de análisis y clustering geográfico.
"""

# =====================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# =====================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats
from scipy.stats import zscore
import re
from datetime import datetime
import pickle
import folium
from folium.plugins import HeatMap, MarkerCluster
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuración de visualización
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Librerías para preprocesamiento
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

# Librerías para clustering
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Librerías para modelado
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Métricas
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

print("="*80)
print("PROYECTO DE PREDICCIÓN DE PRECIOS DE ALQUILER")
print("="*80)
print(f"Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# =====================================
# 2. FUNCIONES AUXILIARES
# =====================================

# def crear_informe_calidad_datos(df):
#     """Genera un informe completo de la calidad de los datos."""
#     informe = {}
#
#     # Información general
#     informe['forma'] = df.shape
#     informe['memoria_mb'] = df.memory_usage(deep=True).sum() / 1024**2
#
#     # Valores faltantes
#     missing = df.isnull().sum()
#     informe['valores_faltantes'] = missing[missing > 0].to_dict()
#     informe['porcentaje_faltantes'] = ((missing / len(df)) * 100).round(2).to_dict()
#
#     # Duplicados
#     informe['filas_duplicadas'] = df.duplicated().sum()
#
#     # Tipos de datos
#     informe['tipos_datos'] = df.dtypes.value_counts().to_dict()
#
#     return informe

def detectar_outliers_multiples_metodos(df, columna, mostrar_grafico=False):
    """
    Detecta outliers usando múltiples métodos y devuelve un consenso.
    """
    resultados = {}
    
    # Método IQR
    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)
    IQR = Q3 - Q1
    lower_iqr = Q1 - 1.5 * IQR
    upper_iqr = Q3 + 1.5 * IQR
    outliers_iqr = df[(df[columna] < lower_iqr) | (df[columna] > upper_iqr)]
    resultados['IQR'] = len(outliers_iqr)
    
    # Método Z-Score
    z_scores = np.abs(stats.zscore(df[columna].dropna()))
    outliers_zscore = df[columna].dropna()[z_scores > 3]
    resultados['Z-Score'] = len(outliers_zscore)
    
    # Método Isolation Forest
    from sklearn.ensemble import IsolationForest
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    outliers_iso = iso_forest.fit_predict(df[[columna]].dropna())
    resultados['Isolation Forest'] = (outliers_iso == -1).sum()
    
    if mostrar_grafico:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Boxplot
        axes[0].boxplot(df[columna].dropna())
        axes[0].set_title('Boxplot')
        axes[0].set_ylabel(columna)
        
        # Histograma con líneas de IQR
        axes[1].hist(df[columna].dropna(), bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(lower_iqr, color='red', linestyle='--', label='IQR límites')
        axes[1].axvline(upper_iqr, color='red', linestyle='--')
        axes[1].set_title('Distribución con límites IQR')
        axes[1].legend()
        
        # Q-Q Plot
        stats.probplot(df[columna].dropna(), dist="norm", plot=axes[2])
        axes[2].set_title('Q-Q Plot')
        
        plt.suptitle(f'Análisis de Outliers para {columna}')
        plt.tight_layout()
        plt.show()
    
    return resultados

def crear_features_de_texto(df, columna_texto):
    """
    Extrae características del texto de descripción.
    """
    if columna_texto not in df.columns:
        return df
    
    # Longitud del texto
    df[f'{columna_texto}_longitud'] = df[columna_texto].str.len()
    
    # Número de palabras
    df[f'{columna_texto}_num_palabras'] = df[columna_texto].str.split().str.len()
    
    # Palabras clave importantes
    palabras_clave = {
        'renovado': ['renovated', 'updated', 'new', 'renovado', 'nuevo', 'actualizado'],
        'lujo': ['luxury', 'premium', 'high-end', 'upscale', 'lujo', 'exclusivo'],
        'vista': ['view', 'vista', 'panoramic', 'ocean', 'mountain'],
        'seguridad': ['security', 'doorman', 'concierge', 'seguridad', 'portero'],
        'transporte': ['subway', 'metro', 'bus', 'train', 'transport']
    }
    
    for categoria, palabras in palabras_clave.items():
        patron = '|'.join(palabras)
        df[f'tiene_{categoria}'] = df[columna_texto].str.contains(patron, case=False, na=False).astype(int)
    
    return df

def clustering_geografico_avanzado(df, n_clusters=None):
    """
    Realiza clustering geográfico usando múltiples algoritmos y selecciona el mejor.
    """
    geo_data = df[['latitude', 'longitude']].dropna()

    # Normalizar coordenadas
    scaler = StandardScaler()
    geo_scaled = scaler.fit_transform(geo_data)
    
    # Si no se especifica n_clusters, encontrar el óptimo
    if n_clusters is None:
        silhouette_scores = []
        k_range = range(3, min(15, len(geo_data)//10))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(geo_scaled)
            score = silhouette_score(geo_scaled, labels)
            silhouette_scores.append(score)
        
        # Seleccionar k con mejor silhouette score
        n_clusters = k_range[np.argmax(silhouette_scores)]
        print(f"Número óptimo de clusters: {n_clusters} (Silhouette Score: {max(silhouette_scores):.3f})")
    
    # Aplicar K-Means con el número óptimo
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    geo_data['neighborhood_cluster'] = kmeans.fit_predict(geo_scaled)
    
    # También probar DBSCAN para detectar zonas de alta densidad
    dbscan = DBSCAN(eps=0.01, min_samples=5)
    geo_data['density_cluster'] = dbscan.fit_predict(geo_scaled)
    
    # Merge con dataframe original
    df = df.merge(geo_data[['neighborhood_cluster', 'density_cluster']], 
                  left_index=True, right_index=True, how='left')
    
    # Calcular estadísticas por cluster
    if 'neighborhood_cluster' in df.columns:
        cluster_stats = df.groupby('neighborhood_cluster').agg({
            'price': ['mean', 'median', 'std'],
            'square_feet': 'mean',
            'latitude': 'mean',
            'longitude': 'mean'
        }).round(2)
        
        print("\nEstadísticas por Cluster:")
        print(cluster_stats)
    
    return df

def imputacion_inteligente(df):
    """
    Realiza imputación inteligente de valores faltantes.
    """
    # Imputación basada en el contexto para pets_allowed
    if 'pets_allowed' in df.columns and 'body' in df.columns:
        mask_pets_null = df['pets_allowed'].isnull()
        
        # Patrones para detectar mascotas en el texto
        patron_pets = r'\bpet\b|\bpets\b|\bdog\b|\bdogs\b|\bcat\b|\bcats\b'
        patron_no_pets = r'\bno pet\b|\bno pets\b|\bpets not\b'
        
        # Imputar basado en menciones en el texto
        df.loc[mask_pets_null & df['body'].str.contains(patron_pets, case=False, na=False), 'pets_allowed'] = 1
        df.loc[mask_pets_null & df['body'].str.contains(patron_no_pets, case=False, na=False), 'pets_allowed'] = 0
        df['pets_allowed'].fillna(0, inplace=True)  # Asumir no mascotas si no se menciona
    
    # Imputación para amenidades
    if 'amenities' in df.columns and 'body' in df.columns:
        amenidades_comunes = ['parking', 'gym', 'pool', 'laundry', 'dishwasher', 
                             'air conditioning', 'elevator', 'balcony']
        
        for amenidad in amenidades_comunes:
            col_name = f'has_{amenidad.replace(" ", "_")}'
            df[col_name] = 0
            
            # Buscar en amenities
            if 'amenities' in df.columns:
                df.loc[df['amenities'].str.contains(amenidad, case=False, na=False), col_name] = 1
            
            # Buscar en body si amenities es null
            mask_amenities_null = df['amenities'].isnull()
            df.loc[mask_amenities_null & df['body'].str.contains(amenidad, case=False, na=False), col_name] = 1
    
    # Imputación para valores numéricos usando KNN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        imputer = KNNImputer(n_neighbors=5)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    return df

def crear_visualizacion_interactiva(df):
    """
    Crea visualizaciones interactivas con Plotly.
    """
    # Mapa interactivo de precios
    if all(col in df.columns for col in ['latitude', 'longitude', 'price']):
        fig_map = px.scatter_mapbox(
            df.dropna(subset=['latitude', 'longitude']),
            lat='latitude',
            lon='longitude',
            color='price',
            size='square_feet',
            hover_data=['bedrooms', 'bathrooms'],
            color_continuous_scale='Viridis',
            mapbox_style='open-street-map',
            title='Distribución Geográfica de Precios',
            zoom=10
        )
        fig_map.update_layout(height=600)
        fig_map.show()
    
    # Matriz de correlación interactiva
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        corr_matrix = df[numeric_cols].corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect='auto',
            color_continuous_scale='RdBu_r',
            title='Matriz de Correlación Interactiva'
        )
        fig_corr.update_layout(height=800, width=1000)
        fig_corr.show()
    
    return True

# =====================================
# 3. CARGA Y EXPLORACIÓN INICIAL
# =====================================

print("\n" + "="*80)
print("CARGANDO DATOS...")
print("="*80)

# Cargar datos
df = pd.read_csv('Resources/Tema_9.csv', encoding='latin1', delimiter=';')

print(f"Dataset cargado exitosamente!")
print(f"Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
print(f"Memoria utilizada: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Generar informe de calidad
print("\n📊 INFORME DE CALIDAD DE DATOS:")
print(f"   • Valores faltantes totales: {sum(df.isnull().sum())}")
print(f"   • Columnas con faltantes: {len(informe['valores_faltantes'])}")
print(f"   • Filas duplicadas: {informe['filas_duplicadas']}")

# =====================================
# 4. LIMPIEZA Y PREPROCESAMIENTO INICIAL
# =====================================

print("\n" + "="*80)
print("LIMPIEZA Y PREPROCESAMIENTO")
print("="*80)

# Eliminar columnas irrelevantes (ya identificadas en tu análisis)
columnas_eliminar = ['id', 'category', 'currency', 'fee', 'price_display', 'source', 'time']
columnas_eliminar = [col for col in columnas_eliminar if col in df.columns]
df = df.drop(columnas_eliminar, axis=1)
print(f"✅ Columnas eliminadas: {columnas_eliminar}")

# Eliminar filas con valores críticos faltantes
columnas_criticas = ['bathrooms', 'latitude', 'longitude', 'bedrooms', 'price', 'square_feet']
columnas_criticas = [col for col in columnas_criticas if col in df.columns]
registros_antes = len(df)
df = df.dropna(subset=columnas_criticas)
print(f"✅ Registros eliminados por valores críticos faltantes: {registros_antes - len(df)}")

# Conversión de tipos de datos
if 'bedrooms' in df.columns:
    df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce')
if 'bathrooms' in df.columns:
    df['bathrooms'] = pd.to_numeric(df['bathrooms'], errors='coerce')

# =====================================
# 5. FEATURE ENGINEERING BÁSICO
# =====================================

print("\n" + "="*80)
print("FEATURE ENGINEERING")
print("="*80)

# Ratios y métricas derivadas
df['price_per_sqft'] = df['price'] / df['square_feet']
df['price_per_bedroom'] = df['price'] / (df['bedrooms'] + 1)
df['price_per_bathroom'] = df['price'] / (df['bathrooms'] + 0.5)
df['total_rooms'] = df['bedrooms'] + df['bathrooms']
df['sqft_per_room'] = df['square_feet'] / (df['total_rooms'] + 1)

# Categorización del tamaño
df['size_category'] = pd.cut(
    df['square_feet'],
    bins=[0, 500, 800, 1200, 1800, np.inf],
    labels=['micro', 'pequeño', 'mediano', 'grande', 'muy_grande']
)

# Categorización del precio
df['price_category'] = pd.cut(
    df['price'],
    bins=[0, df['price'].quantile(0.25), df['price'].quantile(0.5), 
          df['price'].quantile(0.75), np.inf],
    labels=['económico', 'medio_bajo', 'medio_alto', 'premium']
)

print(f"✅ Features básicos creados: 7 nuevas variables")

# =====================================
# 6. CLUSTERING GEOGRÁFICO AVANZADO
# =====================================

print("\n" + "="*80)
print("CLUSTERING GEOGRÁFICO")
print("="*80)

# Aplicar clustering geográfico avanzado
df = clustering_geografico_avanzado(df, n_clusters=None)

# Crear features basados en clusters
if 'neighborhood_cluster' in df.columns:
    # Precio mediano del barrio
    neighborhood_stats = df.groupby('neighborhood_cluster')['price'].agg(['median', 'mean', 'std'])
    neighborhood_stats.columns = ['neighborhood_median_price', 'neighborhood_mean_price', 'neighborhood_std_price']
    df = df.merge(neighborhood_stats, left_on='neighborhood_cluster', right_index=True, how='left')
    
    # Ratio del precio respecto al barrio
    df['price_to_neighborhood_median'] = df['price'] / df['neighborhood_median_price']
    df['price_deviation_from_neighborhood'] = (df['price'] - df['neighborhood_mean_price']) / df['neighborhood_std_price']
    
    print(f"✅ Features de vecindario creados: 5 nuevas variables")

# =====================================
# 7. PROCESAMIENTO DE TEXTO
# =====================================

print("\n" + "="*80)
print("PROCESAMIENTO DE TEXTO")
print("="*80)

# Extraer características del texto
if 'body' in df.columns:
    df = crear_features_de_texto(df, 'body')
    print(f"✅ Features de texto extraídos")

if 'title' in df.columns:
    df = crear_features_de_texto(df, 'title')
    print(f"✅ Features del título extraídos")

# =====================================
# 8. IMPUTACIÓN INTELIGENTE
# =====================================

print("\n" + "="*80)
print("IMPUTACIÓN INTELIGENTE")
print("="*80)

df = imputacion_inteligente(df)
print(f"✅ Imputación completada")
print(f"   • Valores faltantes restantes: {df.isnull().sum().sum()}")

# =====================================
# 9. DETECCIÓN Y MANEJO DE OUTLIERS
# =====================================

print("\n" + "="*80)
print("DETECCIÓN DE OUTLIERS")
print("="*80)

# Detectar outliers en precio
outliers_precio = detectar_outliers_multiples_metodos(df, 'price', mostrar_grafico=True)
print(f"Outliers detectados en precio:")
for metodo, cantidad in outliers_precio.items():
    print(f"   • {metodo}: {cantidad} outliers")

# Manejo conservador de outliers extremos
Q1 = df['price'].quantile(0.01)
Q99 = df['price'].quantile(0.99)
outliers_extremos = df[(df['price'] < Q1) | (df['price'] > Q99)]
print(f"\n⚠️ Outliers extremos (1% superior e inferior): {len(outliers_extremos)}")

# Opcional: eliminar solo outliers muy extremos
precio_max_razonable = df['price'].quantile(0.995)
precio_min_razonable = df['price'].quantile(0.005)
df_clean = df[(df['price'] >= precio_min_razonable) & (df['price'] <= precio_max_razonable)].copy()
print(f"✅ Registros después de eliminar outliers extremos: {len(df_clean)}")

# =====================================
# 10. SELECCIÓN DE CARACTERÍSTICAS
# =====================================

print("\n" + "="*80)
print("SELECCIÓN DE CARACTERÍSTICAS")
print("="*80)

# Preparar datos para modelado
# Identificar columnas numéricas y categóricas
numeric_features = df_clean.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()

# Eliminar columnas de texto largas y la variable objetivo
columnas_excluir = ['price', 'body', 'title', 'amenities', 'address']
numeric_features = [col for col in numeric_features if col not in columnas_excluir]
categorical_features = [col for col in categorical_features if col not in columnas_excluir]

print(f"✅ Features numéricos: {len(numeric_features)}")
print(f"✅ Features categóricos: {len(categorical_features)}")

# Encoding de variables categóricas
label_encoders = {}
for col in categorical_features:
    if col in df_clean.columns:
        le = LabelEncoder()
        df_clean[col + '_encoded'] = le.fit_transform(df_clean[col].fillna('unknown').astype(str))
        label_encoders[col] = le
        numeric_features.append(col + '_encoded')

# Selección de características con mutual information
if len(numeric_features) > 20:
    X_temp = df_clean[numeric_features].fillna(0)
    y_temp = df_clean['price']
    
    selector = SelectKBest(score_func=mutual_info_regression, k=min(20, len(numeric_features)))
    selector.fit(X_temp, y_temp)
    
    # Obtener las mejores características
    feature_scores = pd.DataFrame({
        'feature': numeric_features,
        'score': selector.scores_
    }).sort_values('score', ascending=False)
    
    print("\n📈 Top 10 características más importantes:")
    print(feature_scores.head(10))
    
    # Seleccionar las mejores
    best_features = feature_scores.head(20)['feature'].tolist()
else:
    best_features = numeric_features

# =====================================
# 11. PREPARACIÓN FINAL DE DATOS
# =====================================

print("\n" + "="*80)
print("PREPARACIÓN FINAL DE DATOS")
print("="*80)

# Crear conjuntos X e y
X = df_clean[best_features].fillna(0)
y = df_clean['price']

print(f"✅ Dimensiones finales de X: {X.shape}")
print(f"✅ Dimensión de y: {y.shape}")

# División estratificada
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=df_clean['price_category']
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42
)

print(f"\n📊 División de datos:")
print(f"   • Entrenamiento: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"   • Validación: {X_val.shape[0]} ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"   • Prueba: {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)")

# Escalado robusto
scaler = RobustScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)
X_val_scaled = pd.DataFrame(
    scaler.transform(X_val),
    columns=X_val.columns,
    index=X_val.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

# =====================================
# 12. MODELADO Y EVALUACIÓN
# =====================================

print("\n" + "="*80)
print("ENTRENAMIENTO DE MODELOS")
print("="*80)

# Definir modelos
modelos = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0, random_state=42),
    'Lasso': Lasso(alpha=0.1, random_state=42),
    'ElasticNet': ElasticNet(alpha=0.1, random_state=42),
    'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
}

# Entrenar y evaluar modelos
resultados = {}

for nombre, modelo in modelos.items():
    print(f"\n🔄 Entrenando {nombre}...")
    
    # Entrenar
    modelo.fit(X_train_scaled, y_train)
    
    # Predicciones
    y_train_pred = modelo.predict(X_train_scaled)
    y_val_pred = modelo.predict(X_val_scaled)
    
    # Métricas
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    resultados[nombre] = {
        'modelo': modelo,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'train_r2': train_r2,
        'val_r2': val_r2
    }
    
    print(f"   RMSE: Train={train_rmse:.2f} | Val={val_rmse:.2f}")
    print(f"   R²:   Train={train_r2:.4f} | Val={val_r2:.4f}")

# =====================================
# 13. SELECCIÓN DEL MEJOR MODELO
# =====================================

print("\n" + "="*80)
print("RESULTADOS FINALES")
print("="*80)

# Crear DataFrame con resultados
resultados_df = pd.DataFrame({
    nombre: {
        'RMSE_Train': res['train_rmse'],
        'RMSE_Val': res['val_rmse'],
        'R²_Train': res['train_r2'],
        'R²_Val': res['val_r2']
    }
    for nombre, res in resultados.items()
}).T

resultados_df = resultados_df.sort_values('R²_Val', ascending=False)
print("\n🏆 RANKING DE MODELOS (por R² de validación):")
print(resultados_df.round(4))

# Seleccionar el mejor modelo
mejor_modelo_nombre = resultados_df.index[0]
mejor_modelo = resultados[mejor_modelo_nombre]['modelo']

print(f"\n🥇 Mejor modelo: {mejor_modelo_nombre}")
print(f"   • R² de validación: {resultados_df.loc[mejor_modelo_nombre, 'R²_Val']:.4f}")
print(f"   • RMSE de validación: ${resultados_df.loc[mejor_modelo_nombre, 'RMSE_Val']:.2f}")

# =====================================
# 14. EVALUACIÓN EN CONJUNTO DE PRUEBA
# =====================================

print("\n" + "="*80)
print("EVALUACIÓN FINAL EN CONJUNTO DE PRUEBA")
print("="*80)

# Predicciones finales
y_test_pred = mejor_modelo.predict(X_test_scaled)

# Métricas finales
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_mape = mean_absolute_percentage_error(y_test, y_test_pred)

print(f"\n📊 MÉTRICAS FINALES - {mejor_modelo_nombre}:")
print(f"   • R² Score: {test_r2:.4f}")
print(f"   • RMSE: ${test_rmse:.2f}")
print(f"   • MAE: ${test_mae:.2f}")
print(f"   • MAPE: {test_mape:.2%}")

# =====================================
# 15. ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS
# =====================================

if hasattr(mejor_modelo, 'feature_importances_'):
    print("\n" + "="*80)
    print("IMPORTANCIA DE CARACTERÍSTICAS")
    print("="*80)
    
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': mejor_modelo.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n📊 Top 10 características más importantes:")
    print(feature_importance.head(10))
    
    # Visualización
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importancia')
    plt.title('Top 15 Características Más Importantes')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# =====================================
# 16. VISUALIZACIONES FINALES
# =====================================

print("\n" + "="*80)
print("GENERANDO VISUALIZACIONES")
print("="*80)

# Crear visualización de predicciones vs reales
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Análisis de Predicciones del Modelo', fontsize=16)

# 1. Predicciones vs Reales
axes[0, 0].scatter(y_test, y_test_pred, alpha=0.5)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Precio Real ($)')
axes[0, 0].set_ylabel('Precio Predicho ($)')
axes[0, 0].set_title(f'Predicciones vs Valores Reales (R² = {test_r2:.4f})')
axes[0, 0].grid(True, alpha=0.3)

# 2. Distribución de Residuos
residuos = y_test - y_test_pred
axes[0, 1].hist(residuos, bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Residuos ($)')
axes[0, 1].set_ylabel('Frecuencia')
axes[0, 1].set_title('Distribución de Residuos')
axes[0, 1].axvline(x=0, color='red', linestyle='--')
axes[0, 1].grid(True, alpha=0.3)

# 3. Q-Q Plot de Residuos
stats.probplot(residuos, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot de Residuos')

# 4. Residuos vs Predicciones
axes[1, 1].scatter(y_test_pred, residuos, alpha=0.5)
axes[1, 1].axhline(y=0, color='red', linestyle='--')
axes[1, 1].set_xlabel('Precio Predicho ($)')
axes[1, 1].set_ylabel('Residuos ($)')
axes[1, 1].set_title('Residuos vs Predicciones')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Crear visualizaciones interactivas
crear_visualizacion_interactiva(df_clean)

# =====================================
# 17. GUARDAR MODELOS Y RESULTADOS
# =====================================

print("\n" + "="*80)
print("GUARDANDO MODELOS Y RESULTADOS")
print("="*80)

# Guardar el mejor modelo
with open('mejor_modelo_alquiler.pkl', 'wb') as f:
    pickle.dump(mejor_modelo, f)
print("✅ Modelo guardado: mejor_modelo_alquiler.pkl")

# Guardar el scaler
with open('scaler_alquiler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Scaler guardado: scaler_alquiler.pkl")

# Guardar los encoders
with open('label_encoders_alquiler.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print("✅ Encoders guardados: label_encoders_alquiler.pkl")

# Guardar resultados
resultados_df.to_csv('resultados_modelos.csv')
print("✅ Resultados guardados: resultados_modelos.csv")

# Guardar características importantes
if hasattr(mejor_modelo, 'feature_importances_'):
    feature_importance.to_csv('importancia_caracteristicas.csv', index=False)
    print("✅ Importancia de características guardada: importancia_caracteristicas.csv")

# =====================================
# 18. RESUMEN EJECUTIVO
# =====================================

print("\n" + "="*80)
print("🎯 RESUMEN EJECUTIVO DEL PROYECTO")
print("="*80)

print(f"""
📊 DATOS:
   • Registros procesados: {len(df_clean):,}
   • Características utilizadas: {len(best_features)}
   • Barrios identificados: {df_clean['neighborhood_cluster'].nunique() if 'neighborhood_cluster' in df_clean.columns else 'N/A'}

🏆 MEJOR MODELO: {mejor_modelo_nombre}
   • R² Score: {test_r2:.4f}
   • RMSE: ${test_rmse:.2f}
   • MAE: ${test_mae:.2f}
   • MAPE: {test_mape:.2%}

💡 INSIGHTS CLAVE:
   • Precio promedio: ${df_clean['price'].mean():.2f}
   • Rango de precios: ${df_clean['price'].min():.2f} - ${df_clean['price'].max():.2f}
   • Desviación estándar: ${df_clean['price'].std():.2f}

🎯 RECOMENDACIONES:
   1. El modelo tiene un rendimiento {'excelente' if test_r2 > 0.8 else 'bueno' if test_r2 > 0.6 else 'mejorable'}
   2. Considerar recolectar más datos para mejorar predicciones
   3. Los barrios geográficos son un factor clave en el precio
   4. Implementar monitoreo continuo del modelo en producción

📅 Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")

print("="*80)
print("✅ PROYECTO COMPLETADO EXITOSAMENTE!")
print("="*80)
