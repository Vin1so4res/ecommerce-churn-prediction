Predição de Churn em E-commerce (Machine Learning Pipeline)

> O Problema de Negócio
Um aplicativo de vendas online precisa prever quais clientes estão prestes a abandonar a plataforma (Churn = 1). O objetivo é identificar esses usuários com antecedência para oferecer cupons preventivos de retenção. O maior desafio estratégico é minimizar o impacto financeiro: o custo de perder um cliente em definitivo (Falso Negativo) é drasticamente maior do que o custo de oferecer um desconto para um cliente que já iria comprar (Falso Positivo).

> Resumo Executivo: Insights da Análise Exploratória (EDA)
Durante a fase de exploração e preparação dos dados, três descobertas guiaram a modelagem:
1. **Desbalanceamento Severo:** A variável alvo (`Churn`) apresentou uma proporção majoritária de clientes ativos. Foi necessário aplicar a técnica **SMOTE** exclusivamente nos dados de treino para evitar que a IA ficasse enviesada.
2. **Dados Ausentes e Outliers:** Variáveis como `Tenure` e `WarehouseToHome` possuíam dados ausentes e valores discrepantes. Optou-se pela imputação via **Mediana** para garantir robustez estatística e remoção de outliers extremos para não prejudicar o cálculo de distâncias euclidianas do algoritmo KNN.
3. **Engenharia de Features:** Criou-se a variável `cashback_por_pedido` para agregar valor preditivo ao comportamento de compra do usuário.

> Veredito do Modelo
Foram testadas múltiplas configurações para os algoritmos **KNN** e **Árvore de Decisão**, monitorando as métricas de Treino e Teste para evitar *overfitting*. 

**O modelo escolhido para produção foi o KNN (K=3).** 

**Justificativa:** Analisando a Matriz de Confusão, o KNN (K=3) apresentou apenas **34 Falsos Negativos**, contra 67 do melhor modelo de Árvore de Decisão. No contexto do mercado digital e de VSLs, o Custo de Aquisição de Cliente (CAC) é o maior gargalo. Perder 33 clientes a mais (que a Árvore deixaria passar) significa uma perda irrecuperável de Lifetime Value (LTV). O KNN mostrou-se muito mais eficiente em sinalizar os clientes em risco, justificando sua implementação na operação da empresa.
