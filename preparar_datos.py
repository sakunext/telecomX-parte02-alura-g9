"""
Script para preparar datos tratados desde la Parte 1
Genera el archivo 'datos_tratados.csv' necesario para el modelado ML
"""

import json
import pandas as pd
import numpy as np

print("="*70)
print("📊 PREPARACIÓN DE DATOS - TelecomX Parte 2")
print("="*70)

# Cargar datos desde el JSON de la Parte 1
json_path = "../telecomAluraParte01/TelecomX_Data.json"
print(f"\n📥 Cargando datos desde: {json_path}")

try:
    with open(json_path, 'r', encoding='utf-8') as file:
        data_json = json.load(file)
    print(f"✅ Datos cargados: {len(data_json)} registros")
except FileNotFoundError:
    print(f"❌ Error: No se encontró el archivo {json_path}")
    print("💡 Asegúrate de haber completado la Parte 1 del desafío")
    exit(1)

# Función para aplanar la estructura JSON
def flatten_customer_data(customer_list):
    """
    Convierte la estructura JSON anidada en un DataFrame plano
    """
    flattened_data = []
    
    for customer in customer_list:
        flat_record = {}
        
        # Información del cliente
        if 'customer' in customer:
            flat_record['CustomerID'] = customer['customer'].get('customerID', '')
            flat_record['Gender'] = customer['customer'].get('gender', '')
            flat_record['SeniorCitizen'] = customer['customer'].get('seniorCitizen', 0)
            flat_record['Partner'] = customer['customer'].get('partner', '')
            flat_record['Dependents'] = customer['customer'].get('dependents', '')
        
        # Información telefónica
        if 'phone' in customer:
            flat_record['PhoneService'] = customer['phone'].get('phoneService', '')
            flat_record['MultipleLines'] = customer['phone'].get('multipleLines', '')
        
        # Información de internet
        if 'internet' in customer:
            flat_record['InternetService'] = customer['internet'].get('internetService', '')
            flat_record['OnlineSecurity'] = customer['internet'].get('onlineSecurity', '')
            flat_record['OnlineBackup'] = customer['internet'].get('onlineBackup', '')
            flat_record['DeviceProtection'] = customer['internet'].get('deviceProtection', '')
            flat_record['TechSupport'] = customer['internet'].get('techSupport', '')
            flat_record['StreamingTV'] = customer['internet'].get('streamingTV', '')
            flat_record['StreamingMovies'] = customer['internet'].get('streamingMovies', '')
        
        # Información de cuenta
        if 'account' in customer:
            flat_record['Tenure'] = customer['account'].get('tenure', 0)
            flat_record['Contract'] = customer['account'].get('contract', '')
            flat_record['PaperlessBilling'] = customer['account'].get('paperlessBilling', '')
            flat_record['PaymentMethod'] = customer['account'].get('paymentMethod', '')
            flat_record['MonthlyCharges'] = customer['account'].get('monthlyCharges', 0.0)
            flat_record['TotalCharges'] = customer['account'].get('totalCharges', 0.0)
            flat_record['Churn'] = customer['account'].get('churn', '')
        
        flattened_data.append(flat_record)
    
    return pd.DataFrame(flattened_data)

print("\n🔧 Transformando datos...")
df = flatten_customer_data(data_json)
print(f"✅ DataFrame creado: {df.shape[0]} filas, {df.shape[1]} columnas")

# Limpieza de datos
print("\n🧹 Limpiando datos...")

# Convertir TotalCharges a numérico (puede haber espacios vacíos)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Imputar valores nulos en TotalCharges con 0 (clientes nuevos)
if df['TotalCharges'].isnull().sum() > 0:
    print(f"   - Imputando {df['TotalCharges'].isnull().sum()} valores nulos en TotalCharges")
    df['TotalCharges'].fillna(0, inplace=True)

# Verificar tipos de datos
print("\n📊 Tipos de datos:")
print(df.dtypes.value_counts())

# Feature Engineering
print("\n⚙️ Creando features adicionales...")

# 1. Categoría de antigüedad
def categorize_tenure(months):
    if months <= 12:
        return 'Nuevo'
    elif months <= 36:
        return 'Medio'
    else:
        return 'Antiguo'

df['TenureCategory'] = df['Tenure'].apply(categorize_tenure)
print("   ✅ TenureCategory creada")

# 2. Categoría de cargo mensual
def categorize_charges(charge):
    if charge < 30:
        return 'Bajo'
    elif charge < 70:
        return 'Medio'
    else:
        return 'Alto'

df['MonthlyChargesCategory'] = df['MonthlyCharges'].apply(categorize_charges)
print("   ✅ MonthlyChargesCategory creada")

# 3. Indicador de servicios de seguridad
df['HasSecurityServices'] = ((df['OnlineSecurity'] == 'Yes') | 
                               (df['OnlineBackup'] == 'Yes') | 
                               (df['DeviceProtection'] == 'Yes') | 
                               (df['TechSupport'] == 'Yes')).astype(int)
print("   ✅ HasSecurityServices creada")

# 4. Indicador de servicios de streaming
df['HasStreamingServices'] = ((df['StreamingTV'] == 'Yes') | 
                                (df['StreamingMovies'] == 'Yes')).astype(int)
print("   ✅ HasStreamingServices creada")

# 5. Promedio de cargo mensual por tenure
df['AvgMonthlyChargePerTenure'] = df['MonthlyCharges'] / (df['Tenure'] + 1)
print("   ✅ AvgMonthlyChargePerTenure creada")

# Resumen final
print(f"\n📈 Dataset final: {df.shape[0]} filas, {df.shape[1]} columnas")
print(f"\n📋 Columnas del dataset:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2d}. {col}")

# Estadísticas de Churn
print(f"\n🎯 Distribución de Churn:")
print(df['Churn'].value_counts())
print(f"\nProporción de Churn:")
print(df['Churn'].value_counts(normalize=True) * 100)

# Guardar datos tratados
output_file = "datos_tratados.csv"
df.to_csv(output_file, index=False)
print(f"\n💾 Datos guardados en: {output_file}")

# También crear una copia para compatibilidad con el notebook de la Parte 1
output_file_2 = "telecom_data_processed.csv"
df.to_csv(output_file_2, index=False)
print(f"💾 Copia guardada en: {output_file_2}")

print("\n" + "="*70)
print("✅ PREPARACIÓN COMPLETADA CON ÉXITO")
print("="*70)
print("\n💡 Ahora puedes ejecutar el notebook TelecomX_ML_Prediction.ipynb")
