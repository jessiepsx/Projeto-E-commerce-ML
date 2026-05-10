import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurando o estilo dos gráficos
sns.set_theme(style="whitegrid")

print("⚡ Iniciando análise de Outliers e Inconsistências...")

# 1. Carregar os dados da Camada Silver
caminho_silver = '../data/orders_silver.csv'

# Fallback caso você execute direto da raiz ou de dentro de análises
if not os.path.exists(caminho_silver):
    caminho_silver = 'data/orders_silver.csv'

try:
    df = pd.read_csv(caminho_silver)
    print(f"✔️ Base Silver carregada com sucesso! Total de linhas: {df.shape[0]}")
except Exception as e:
    print(f"❌ Erro ao abrir o arquivo: {e}")
    print("Verifique se o arquivo está na pasta 'data/silver/orders_silver.csv'.")
    exit()

# Criando coluna temporária de tempo de entrega para a análise
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
df['tempo_entrega_dias'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days

# =====================================================================
# 1. VALIDAÇÃO DE VALORES INVÁLIDOS (Erros de Domínio/Negócio)
# =====================================================================
print("\n🔍 1. Validando valores inválidos...")

# Tempo de entrega não pode ser negativo (data de entrega anterior à de compra)
invalidos = df[df['tempo_entrega_dias'] < 0]
qtd_invalidos = invalidos.shape[0]

if qtd_invalidos > 0:
    print(f"⚠️ Alerta: Encontrados {qtd_invalidos} registros com tempo de entrega negativo!")
    # Salva os inválidos para auditoria
    os.makedirs('data/auditoria', exist_ok=True)
    invalidos.to_csv('data/auditoria/valores_invalidos_entrega.csv', index=False)
    print("📂 Registros inválidos salvos em: 'data/auditoria/valores_invalidos_entrega.csv'")
else:
    print("🟢 Nenhum valor negativo ou inválido encontrado no tempo de entrega!")


# =====================================================================
# 2. IDENTIFICAÇÃO DE OUTLIERS (Método Estatístico IQR)
# =====================================================================
print("\n📊 2. Calculando limites estatísticos (IQR)...")

# Removemos valores nulos apenas para o cálculo do IQR (pedidos não entregues ainda)
dados_validos = df['tempo_entrega_dias'].dropna()

Q1 = dados_validos.quantile(0.25)
Q3 = dados_validos.quantile(0.75)
IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

print(f"   🔹 Primeiro Quartil (Q1 - 25%): {Q1:.2f} dias")
print(f"   🔹 Terceiro Quartil (Q3 - 75%): {Q3:.2f} dias")
print(f"   🔹 Amplitude Interquartil (IQR): {IQR:.2f} dias")
print(f"   🔹 Limite Inferior de Outliers: {limite_inferior:.2f} dias")
print(f"   🔹 Limite Superior de Outliers (Corte): {limite_superior:.2f} dias")

# Classificando quem é outlier
# (Como o limite inferior costuma dar negativo, focamos em valores que estouram o limite superior)
outliers = df[df['tempo_entrega_dias'] > limite_superior]
qtd_outliers = outliers.shape[0]
percentual_outliers = (qtd_outliers / df.shape[0]) * 100

print(f"🚨 Outliers detectados: {qtd_outliers} registros ({percentual_outliers:.2f}% da base)")


# =====================================================================
# 3. VISUALIZAÇÃO GRÁFICA (Boxplot)
# =====================================================================
print("\n📈 3. Gerando gráfico Boxplot...")

os.makedirs('analises/graficos', exist_ok=True)

plt.figure(figsize=(10, 6))
sns.boxplot(x=df['tempo_entrega_dias'], color='#4C6EF5', flierprops={"marker": "x", "markerfacecolor": "red"})

plt.title('Distribuição do Tempo de Entrega (Identificação de Outliers)', fontsize=14, pad=15)
plt.xlabel('Tempo de Entrega (Dias)', fontsize=12)
plt.axvline(x=limite_superior, color='red', linestyle='--', label=f'Limite Superior Outlier ({limite_superior:.1f} dias)')
plt.legend()

# Salva o gráfico na pasta de relatórios/análises
caminho_grafico = 'analises/graficos/boxplot_tempo_entrega.png'
plt.savefig(caminho_grafico, dpi=300, bbox_inches='tight')
plt.close()
print(f"📸 Gráfico salvo com sucesso em: '{caminho_grafico}'")


# =====================================================================
# 4. PERSISTÊNCIA DOS OUTLIERS (Rastreabilidade na Camada Silver)
# =====================================================================
print("\n💾 4. Salvando registros de outliers para rastreabilidade...")

# Garante que a pasta silver existe
os.makedirs('data', exist_ok=True)

# Salvando a tabela de outliers na pasta silver para documentação futura do grupo
caminho_outliers = 'data/orders_anomalias_silver.csv'
outliers.to_csv(caminho_outliers, index=False)
print(f"💾 Tabela de outliers salva em: '{caminho_outliers}'")

print("\n🏁 Processo concluído com sucesso!")
print(f"Resumo: {qtd_outliers} outliers catalogados | {qtd_invalidos} registros inválidos mapeados.")