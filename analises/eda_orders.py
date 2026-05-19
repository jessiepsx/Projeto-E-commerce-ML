import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração visual dos gráficos
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]

# 1. CARREGAR OS DADOS
# (Substitua pelo caminho do seu CSV ou pela sua conexão do Supabase)
df = pd.read_csv('../data/orders_silver.csv') 

# Converter as colunas de data de texto para o formato de data real
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])

# Criar coluna de Ano-Mês para o gráfico de evolução
df['ano_mes'] = df['order_purchase_timestamp'].dt.to_period('M').astype(str)

# Criar coluna de Tempo de Entrega Real (em dias)
df['tempo_entrega_dias'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days


# ====================================================================
# GRÁFICO 1: EVOLUÇÃO MENSAL DE VENDAS
# ====================================================================
plt.figure()
vendas_mensais = df.groupby('ano_mes').size().reset_index(name='qtd_pedidos')
# Ordenar os meses para o gráfico fazer sentido cronologicamente
vendas_mensais = vendas_mensais.sort_values('ano_mes')

sns.lineplot(data=vendas_mensais, x='ano_mes', y='qtd_pedidos', marker='o', color='royalblue', linewidth=2.5)
plt.title('Evolução Mensal do Volume de Pedidos (Visão de Negócio)', fontsize=14, fontweight='bold')
plt.xlabel('Meses', fontsize=12)
plt.ylabel('Quantidade de Pedidos', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('01_evolucao_vendas.png') # Salva o gráfico como imagem
plt.show()


# ====================================================================
# GRÁFICO 2: DISTRIBUIÇÃO DO STATUS DOS PEDIDOS
# ====================================================================
plt.figure()
status_orders = df['order_status'].value_counts().reset_index()
status_orders.columns = ['status', 'total']

# Criando gráfico de barras horizontais para facilitar a leitura dos status
sns.barplot(data=status_orders, x='total', y='status', palette='viridis')
plt.title('Saúde Operacional: Distribuição do Status dos Pedidos', fontsize=14, fontweight='bold')
plt.xlabel('Total de Pedidos', fontsize=12)
plt.ylabel('Status do Pedido', fontsize=12)
plt.tight_layout()
plt.savefig('02_status_pedidos.png')
plt.show()


# ====================================================================
# GRÁFICO 3: DISTRIBUIÇÃO DO TEMPO DE ENTREGA (HISTOGRAMA)
# ====================================================================
plt.figure()
# Filtrando valores nulos e limitando a 30 dias para não distorcer o gráfico com os outliers extremos
df_entrega_limpa = df[df['tempo_entrega_dias'].notnull() & (df['tempo_entrega_dias'] >= 0) & (df['tempo_entrega_dias'] <= 30)]

sns.histplot(data=df_entrega_limpa, x='tempo_entrega_dias', bins=30, kde=True, color='darkorange')
plt.axvline(df_entrega_limpa['tempo_entrega_dias'].median(), color='red', linestyle='--', label=f"Mediana: {df_entrega_limpa['tempo_entrega_dias'].median()} dias")
plt.title('Performance Logística: Como se distribuem os prazos de entrega real?', fontsize=14, fontweight='bold')
plt.xlabel('Dias para Entrega', fontsize=12)
plt.ylabel('Frequência (Quantidade de Pedidos)', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('03_tempo_entrega.png')
plt.show()