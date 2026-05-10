import pandas as pd
import numpy as np

# 1. Carregar o conjunto de dados
df = pd.read_csv('olist_orders_dataset.csv')

print("Dados originais carregados. Total de linhas:", len(df))

# 2. Remover valores ausentes (missing values) nas colunas críticas
# Removemos linhas onde a chave principal ou o ID do cliente estejam faltando.
df = df.dropna(subset=['order_id', 'customer_id'])

# 3. Tratamento de duplicatas
# Removemos duplicatas com base no 'order_id'
antes_duplicadas = len(df)
df = df.drop_duplicates(subset=['order_id'], keep='first')
print(f"Duplicatas removidas: {antes_duplicadas - len(df)}")

# 4. Padronização de textos e conversão para caixa baixa/alta conforme o contrato
if 'order_status' in df.columns:
    df['order_status'] = df['order_status'].str.lower().str.strip()

# 5. Padronização de datas (Remoção de timezone e conversão para o formato yyyy-mm-dd)
data_columns = [
    'order_purchase_timestamp', 'order_approved_at', 
    'order_delivered_carrier_date', 'order_delivered_customer_date', 
    'order_estimated_delivery_date'
]

for col in data_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce').dt.date

# 6. Salvar o arquivo limpo e padronizado
df.to_csv('olist_orders_dataset_silver.csv', index=False)
print("Tratamento concluído! Arquivo salvo como 'olist_orders_dataset_silver.csv'.")