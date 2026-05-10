import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

print("Carregando dados limpos do arquivo local...")

try:
    # Lê direto o arquivo que você já limpou e salvou na etapa anterior
    df = pd.read_csv('../data/orders_silver.csv')
    print(f"✔️ Dados carregados com sucesso! Total de linhas: {df.shape[0]} | Colunas: {df.shape[1]}")
except Exception as e:
    print(f"❌ Erro ao abrir o arquivo 'orders_silver.csv': {e}")
    print("Verifique se o arquivo está na mesma pasta onde você está executando o terminal.")
    exit()

# ==========================================
# TRATAMENTO 1: Variáveis de Data (Enriquecimento)
# ==========================================
print("\nProcessando variáveis de data...")
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])

# Extraindo atributos temporais importantes
df['compra_ano'] = df['order_purchase_timestamp'].dt.year
df['compra_mes'] = df['order_purchase_timestamp'].dt.month
df['compra_dia_semana'] = df['order_purchase_timestamp'].dt.dayofweek  # 0=Segunda, 6=Domingo

# Criando variável categórica/booleana: "É fim de semana?" (Sábado=5, Domingo=6)
df['eh_fim_de_semana'] = df['compra_dia_semana'].apply(lambda x: 1 if x >= 5 else 0)


# ==========================================
# TRATAMENTO 2: Variáveis Categóricas (One-Hot Encoding)
# ==========================================
print("Tratando variáveis categóricas (One-Hot Encoding)...")
encoder = OneHotEncoder(sparse_output=False, drop='first') 
status_encoded = encoder.fit_transform(df[['order_status']])

# Criando um DataFrame com as novas colunas geradas pelo Encoder
status_columns = [f"status_{cat}" for cat in encoder.categories_[0][1:]]
df_status_encoded = pd.DataFrame(status_encoded, columns=status_columns, index=df.index)

# Juntando de volta ao DataFrame principal
df = pd.concat([df, df_status_encoded], axis=1)


# ==========================================
# TRATAMENTO 3: Variáveis Numéricas (Scaling / Normalização)
# ==========================================
print("🔢 Normalizando variáveis numéricas...")
df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
df['tempo_entrega_dias'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days

# Preenchendo valores ausentes do tempo de entrega com a MEDIANA (conforme regra de aceitação)
mediana_tempo = df['tempo_entrega_dias'].median()
df['tempo_entrega_dias'] = df['tempo_entrega_dias'].fillna(mediana_tempo)

# Aplicando MinMaxScaler para colocar o tempo de entrega na escala de 0 a 1
scaler = MinMaxScaler()
df['tempo_entrega_normalizado'] = scaler.fit_transform(df[['tempo_entrega_dias']])


# ==========================================
# SALVANDO O RESULTADO (Pronto para Machine Learning!)
# ==========================================
# Mantendo apenas as colunas transformadas para o modelo
colunas_modelo = [
    'compra_ano', 'compra_mes', 'compra_dia_semana', 'eh_fim_de_semana',
    'tempo_entrega_normalizado'
] + status_columns

df_gold = df[colunas_modelo]

# Salva o arquivo final localmente
df_gold.to_csv('../data/orders_gold.csv', index=False)
print("\n🎯 Engenharia de Recursos finalizada com sucesso!")
print(f"📂 Arquivo gerado: 'orders_preparado_gold.csv' com {df_gold.shape[1]} features prontas para o modelo.")