import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# 1. Carregamento dos dados
df = pd.read_csv('dados/E Commerce Dataset - E Comm.csv')

print("="*50)
print("FASE 1: ANÁLISE EXPLORATÓRIA DE DADOS (EDA)")
print("="*50)

print(f"\n1. Tamanho da base: {df.shape[0]} linhas e {df.shape[1]} colunas.")
print("\n2. Tipos de dados e contagem de nulos:")
print(df.info())
print("\n3. Sumário Estatístico Descritivo:")
print(df.describe())

# Configurando a tela para receber 3 gráficos lado a lado
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.set_style("white") 

# Gráfico 1: Desbalanceamento do Churn

sns.countplot(data=df, x='Churn', ax=axes[0], palette=['#d3d3d3', '#1f77b4'])
axes[0].set_title('1. Desbalanceamento do Churn (Alvo)', fontweight='bold')
axes[0].spines[['top', 'right']].set_visible(False)

# Gráfico 2: Distribuição do Tempo de Permanência (Tenure)
sns.histplot(data=df, x='Tenure', kde=True, ax=axes[1], color='#333333')
axes[1].set_title('2. Distribuição do Tempo de Permanência (Tenure)', fontweight='bold')
axes[1].spines[['top', 'right']].set_visible(False)

# Gráfico 3: Mapa de Calor de Correlação
matriz_corr = df.corr(method='pearson', numeric_only=True)
sns.heatmap(matriz_corr, annot=False, cmap='coolwarm', ax=axes[2]) 
axes[2].set_title('3. Mapa de Calor (Correlação de Pearson)', fontweight='bold')

plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("FASE 2: TRATAMENTO E LIMPEZA (DATA PREP)")
print("="*50)

# 1. Tratamento de Duplicados
duplicados = df.duplicated().sum()
df = df.drop_duplicates()
print(f"1. Duplicados: Foram encontradas e removidas {duplicados} linhas duplicadas.")

# 2. Tratamento de Nulos
colunas_com_nulos = df.columns[df.isnull().any()].tolist()
for col in colunas_com_nulos:
    if df[col].dtype in ['float64', 'int64']:
        mediana_coluna = df[col].median()
        df[col] = df[col].fillna(mediana_coluna)
        
print(f"2. Nulos: Valores ausentes nas colunas {colunas_com_nulos} foram preenchidos com a Mediana.")

# 3. Tratamento de Outliers
linhas_antes = df.shape[0]
df = df[df['WarehouseToHome'] <= 100]
outliers_removidos = linhas_antes - df.shape[0]
print(f"3. Outliers: Foram removidos {outliers_removidos} registos extremos (outliers).")


print("\n" + "="*50)
print("FASE 3: FEATURE ENGINEERING (COLUNA CALCULADA)")
print("="*50)

# Criar proporção de cashback por pedido
df['cashback_por_pedido'] = df['CashbackAmount'] / df['OrderCount']
print("Coluna 'cashback_por_pedido' criada com sucesso.")


print("\n" + "="*50)
print("FASE 4: SEPARAÇÃO, BALANCEAMENTO E ESCALONAMENTO")
print("="*50)

# 1. Convertendo textos em números para a máquina entender
df_encoded = pd.get_dummies(df, drop_first=True)

# 2. Separação de Dados
# O ID do cliente não ajuda a prever nada, então jogamos fora. O alvo (y) é o Churn.
X = df_encoded.drop(columns=['Churn', 'CustomerID']) 
y = df_encoded['Churn']

# Separando em Treino (80%) e Teste (20%) mantendo a proporção original
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
print(f"Split Seguro: {X_train.shape[0]} linhas para treino e {X_test.shape[0]} para teste.")

# 3. Balanceamento de Classes no Treino
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print(f"Antes do SMOTE (Treino): \n{y_train.value_counts().to_dict()}")
print(f"Depois do SMOTE (Treino - Equilibrado): \n{y_train_bal.value_counts().to_dict()}")

# 4. Escalonamento
# O modelo KNN precisa dos dados na mesma proporção de escala.
scaler = StandardScaler()
X_train_knn = scaler.fit_transform(X_train_bal)
X_test_knn = scaler.transform(X_test)
print("StandardScaler aplicado com sucesso.")

print("\n" + "="*50)
print("FASE 5: MODELAGEM E DIAGNÓSTICO DE OVERFITTING")
print("="*50)

# --- 1. OTIMIZAÇÃO DO KNN ---
print("\n--- Testando KNN (K Vizinhos) ---")
valores_k = [3, 5, 7, 9]
for k in valores_k:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_knn, y_train_bal)
    
    acc_treino = accuracy_score(y_train_bal, knn.predict(X_train_knn))
    acc_teste = accuracy_score(y_test, knn.predict(X_test_knn))
    print(f"KNN (K={k}): Acurácia Treino = {acc_treino:.4f} | Acurácia Teste = {acc_teste:.4f}")

# --- 2. OTIMIZAÇÃO DA ÁRVORE DE DECISÃO ---
print("\n--- Testando Árvore de Decisão (Profundidade) ---")
valores_depth = [3, 5, 7, None]
for depth in valores_depth:
    arvore = DecisionTreeClassifier(max_depth=depth, random_state=42)
    arvore.fit(X_train_bal, y_train_bal)
    
    acc_treino = accuracy_score(y_train_bal, arvore.predict(X_train_bal))
    acc_teste = accuracy_score(y_test, arvore.predict(X_test))
    
    depth_str = depth if depth is not None else "Ilimitada"
    print(f"Árvore (Profundidade={depth_str}): Acurácia Treino = {acc_treino:.4f} | Acurácia Teste = {acc_teste:.4f}")

print("\n" + "="*50)
print("FASE 6: AVALIAÇÃO E VEREDITO DE NEGÓCIOS")
print("="*50)

# 1. Treinando os Campeões
melhor_knn = KNeighborsClassifier(n_neighbors=3)
melhor_knn.fit(X_train_knn, y_train_bal)
y_pred_knn = melhor_knn.predict(X_test_knn)

melhor_arvore = DecisionTreeClassifier(max_depth=7, random_state=42)
melhor_arvore.fit(X_train_bal, y_train_bal)
y_pred_arvore = melhor_arvore.predict(X_test)

# 2. Relatórios de Classificação
print("\n--- Relatório KNN (K=3) ---")
print(classification_report(y_test, y_pred_knn))

print("\n--- Relatório Árvore (Profundidade=7) ---")
print(classification_report(y_test, y_pred_arvore))

# 3. Matrizes de Confusão (Visual)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_knn, ax=axes[0], cmap='Blues', colorbar=False)
axes[0].set_title('Matriz de Confusão - KNN (K=3)', fontweight='bold')

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_arvore, ax=axes[1], cmap='Oranges', colorbar=False)
axes[1].set_title('Matriz de Confusão - Árvore (Prof=7)', fontweight='bold')

plt.tight_layout()
plt.show()