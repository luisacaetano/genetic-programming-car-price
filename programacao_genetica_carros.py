"""
Objetivo: Estimar o preço de carros usando Programação Genética
"""

import pandas as pd
import numpy as np
import random
import copy
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# =============================================================================
# 1. LEITURA E PREPARAÇÃO DOS DADOS
# =============================================================================

def carregar_dados(caminho_csv):
    """Carrega e prepara a base de dados."""
    df = pd.read_csv(caminho_csv)

    print("=" * 60)
    print("INSPEÇÃO INICIAL DOS DADOS")
    print("=" * 60)
    print(f"\nDimensões: {df.shape[0]} linhas x {df.shape[1]} colunas")
    print(f"\nColunas: {list(df.columns)}")
    print(f"\nTipos de dados:\n{df.dtypes}")
    print(f"\nEstatísticas descritivas:\n{df.describe()}")
    print(f"\nValores nulos:\n{df.isnull().sum()}")

    return df


def preparar_dados(df):
    """Codifica variáveis categóricas e separa features do target."""
    df_encoded = df.copy()

    # Colunas categóricas que precisam ser codificadas
    colunas_categoricas = ['Brand', 'Model', 'Fuel_Type', 'Transmission']

    # Label Encoding para cada coluna categórica
    encoders = {}
    for col in colunas_categoricas:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        encoders[col] = le

    # Separar features (X) e target (y)
    X = df_encoded.drop('Price', axis=1)
    y = df_encoded['Price']

    # Converter para arrays numpy
    X = X.values.astype(float)
    y = y.values.astype(float)

    # Nomes das features para referência
    nomes_features = list(df_encoded.drop('Price', axis=1).columns)

    print(f"\nFeatures utilizadas: {nomes_features}")
    print(f"Shape X: {X.shape}, Shape y: {y.shape}")

    return X, y, nomes_features


# =============================================================================
# 2. REPRESENTAÇÃO DOS INDIVÍDUOS (ÁRVORE DE EXPRESSÃO)
# =============================================================================

class No:
    """Representa um nó na árvore de expressão."""

    def __init__(self, valor, esquerda=None, direita=None):
        self.valor = valor          # operador, variável (índice) ou constante
        self.esquerda = esquerda    # subárvore esquerda
        self.direita = direita      # subárvore direita

    def __str__(self):
        return self._to_string()

    def _to_string(self):
        """Converte a árvore em string legível."""
        if self.esquerda is None and self.direita is None:
            # Nó folha (terminal)
            if isinstance(self.valor, int):
                return f"X{self.valor}"
            else:
                return str(round(self.valor, 2))
        elif self.direita is None:
            # Operador unário
            return f"{self.valor}({self.esquerda._to_string()})"
        else:
            # Operador binário
            return f"({self.esquerda._to_string()} {self.valor} {self.direita._to_string()})"

    def copiar(self):
        """Cria uma cópia profunda do nó."""
        return copy.deepcopy(self)


# Operadores disponíveis conforme enunciado
OPERADORES_BINARIOS = ['+', '-', '*', '/', '**']  # soma, subtração, multiplicação, divisão, potenciação
OPERADORES_UNARIOS = ['sqrt', 'round']             # raiz quadrada, arredondamento


def criar_no_aleatorio(num_features, profundidade_atual, profundidade_max):
    """Cria um nó aleatório na árvore."""

    # Se atingiu profundidade máxima, criar terminal
    if profundidade_atual >= profundidade_max:
        return criar_terminal(num_features)

    # Escolher entre terminal, operador unário ou binário
    escolha = random.random()

    if escolha < 0.3:  # 30% chance de terminal
        return criar_terminal(num_features)
    elif escolha < 0.5:  # 20% chance de operador unário
        op = random.choice(OPERADORES_UNARIOS)
        filho = criar_no_aleatorio(num_features, profundidade_atual + 1, profundidade_max)
        return No(op, filho, None)
    else:  # 50% chance de operador binário
        op = random.choice(OPERADORES_BINARIOS)
        esq = criar_no_aleatorio(num_features, profundidade_atual + 1, profundidade_max)
        dir = criar_no_aleatorio(num_features, profundidade_atual + 1, profundidade_max)
        return No(op, esq, dir)


def criar_terminal(num_features):
    """Cria um nó terminal (variável ou constante)."""
    if random.random() < 0.7:  # 70% chance de variável
        return No(random.randint(0, num_features - 1))
    else:  # 30% chance de constante
        return No(random.uniform(-10, 10))


def criar_individuo(num_features, profundidade_max=4):
    """Cria um indivíduo (árvore de expressão) aleatório."""
    return criar_no_aleatorio(num_features, 0, profundidade_max)


# =============================================================================
# 3. AVALIAÇÃO DAS EXPRESSÕES
# =============================================================================

def avaliar_expressao(no, X_linha):
    """
    Avalia uma expressão (árvore) para uma linha de dados.

    Parâmetros:
    - no: nó raiz da árvore
    - X_linha: array com valores das features para uma observação

    Retorna: valor numérico da expressão
    """
    try:
        if no.esquerda is None and no.direita is None:
            # Nó terminal
            if isinstance(no.valor, int):
                # É um índice de variável
                return X_linha[no.valor]
            else:
                # É uma constante
                return no.valor

        elif no.direita is None:
            # Operador unário
            val_esq = avaliar_expressao(no.esquerda, X_linha)

            if no.valor == 'sqrt':
                return np.sqrt(abs(val_esq))  # sqrt de valor absoluto para evitar erro
            elif no.valor == 'round':
                return round(val_esq)

        else:
            # Operador binário
            val_esq = avaliar_expressao(no.esquerda, X_linha)
            val_dir = avaliar_expressao(no.direita, X_linha)

            if no.valor == '+':
                return val_esq + val_dir
            elif no.valor == '-':
                return val_esq - val_dir
            elif no.valor == '*':
                return val_esq * val_dir
            elif no.valor == '/':
                # Divisão protegida para evitar divisão por zero
                if abs(val_dir) < 1e-10:
                    return val_esq
                return val_esq / val_dir
            elif no.valor == '**':
                # Potenciação protegida
                if abs(val_esq) < 1e-10:
                    return 0
                # Limitar expoente para evitar overflow
                exp = max(-5, min(5, val_dir))
                resultado = abs(val_esq) ** exp
                if np.isnan(resultado) or np.isinf(resultado):
                    return val_esq
                return resultado

    except Exception:
        return 0

    return 0


def prever(individuo, X):
    """Faz previsões para todas as linhas de X."""
    previsoes = []
    for i in range(len(X)):
        pred = avaliar_expressao(individuo, X[i])
        # Limitar valores extremos
        if np.isnan(pred) or np.isinf(pred):
            pred = 0
        pred = max(-1e10, min(1e10, pred))
        previsoes.append(pred)
    return np.array(previsoes)


# =============================================================================
# 4. FUNÇÃO DE FITNESS
# =============================================================================

def calcular_fitness(individuo, X, y):
    """
    Calcula o fitness do indivíduo usando o Erro Absoluto Médio (MAE).

    Justificativa da métrica MAE:
    - É intuitiva: representa o erro médio em unidades de preço
    - É robusta a outliers comparada ao MSE
    - Facilita interpretação: "em média, o modelo erra X reais"

    Retorna: MAE (quanto menor, melhor)
    """
    previsoes = prever(individuo, X)
    mae = np.mean(np.abs(y - previsoes))

    # Penalizar árvores muito grandes (controle de bloat)
    tamanho = contar_nos(individuo)
    penalidade = tamanho * 0.1  # pequena penalidade por complexidade

    return mae + penalidade


def contar_nos(no):
    """Conta o número de nós na árvore."""
    if no is None:
        return 0
    return 1 + contar_nos(no.esquerda) + contar_nos(no.direita)


# =============================================================================
# 5. POPULAÇÃO INICIAL
# =============================================================================

def criar_populacao(tamanho_pop, num_features, profundidade_max=4):
    """Cria a população inicial de indivíduos."""
    populacao = []
    for _ in range(tamanho_pop):
        individuo = criar_individuo(num_features, profundidade_max)
        populacao.append(individuo)
    return populacao


# =============================================================================
# 6. SELEÇÃO POR TORNEIO BINÁRIO
# =============================================================================

def torneio_binario(populacao, fitness_lista):
    """
    Realiza seleção por torneio binário.

    Funcionamento:
    - Seleciona 2 indivíduos aleatoriamente
    - Retorna o que tiver MENOR fitness (menor erro)
    """
    idx1, idx2 = random.sample(range(len(populacao)), 2)

    if fitness_lista[idx1] < fitness_lista[idx2]:
        return populacao[idx1].copiar()
    else:
        return populacao[idx2].copiar()


# =============================================================================
# 7. CRUZAMENTO (CROSSOVER)
# =============================================================================

def obter_todos_nos(no, lista_nos=None, caminho=None):
    """Obtém todos os nós da árvore com seus caminhos."""
    if lista_nos is None:
        lista_nos = []
    if caminho is None:
        caminho = []

    lista_nos.append((no, caminho.copy()))

    if no.esquerda is not None:
        obter_todos_nos(no.esquerda, lista_nos, caminho + ['esquerda'])
    if no.direita is not None:
        obter_todos_nos(no.direita, lista_nos, caminho + ['direita'])

    return lista_nos


def substituir_no(raiz, caminho, novo_no):
    """Substitui um nó no caminho especificado."""
    if len(caminho) == 0:
        return novo_no

    atual = raiz
    for i, direcao in enumerate(caminho[:-1]):
        if direcao == 'esquerda':
            atual = atual.esquerda
        else:
            atual = atual.direita

    if caminho[-1] == 'esquerda':
        atual.esquerda = novo_no
    else:
        atual.direita = novo_no

    return raiz


def cruzamento(pai1, pai2, prob_crossover=0.9):
    """
    Realiza cruzamento entre dois pais.

    Implementação:
    - Seleciona um ponto de corte aleatório em cada pai
    - Troca as subárvores entre os pais
    - Retorna dois filhos
    """
    if random.random() > prob_crossover:
        return pai1.copiar(), pai2.copiar()

    filho1 = pai1.copiar()
    filho2 = pai2.copiar()

    # Obter todos os nós de cada pai
    nos_f1 = obter_todos_nos(filho1)
    nos_f2 = obter_todos_nos(filho2)

    if len(nos_f1) < 2 or len(nos_f2) < 2:
        return filho1, filho2

    # Selecionar pontos de cruzamento (evitar raiz para manter estrutura)
    _, caminho1 = random.choice(nos_f1[1:]) if len(nos_f1) > 1 else nos_f1[0]
    no2, caminho2 = random.choice(nos_f2[1:]) if len(nos_f2) > 1 else nos_f2[0]

    # Obter subárvore do filho1 para trocar
    atual = filho1
    for direcao in caminho1:
        if direcao == 'esquerda':
            atual = atual.esquerda
        else:
            atual = atual.direita
    subarvore1 = atual.copiar()

    # Trocar subárvores
    if caminho1:
        substituir_no(filho1, caminho1, no2.copiar())
    if caminho2:
        substituir_no(filho2, caminho2, subarvore1)

    return filho1, filho2


# =============================================================================
# 8. MUTAÇÃO
# =============================================================================

def mutacao(individuo, num_features, prob_mutacao=0.2):
    """
    Aplica mutação em um indivíduo.

    Tipos de mutação implementados:
    1. Mutação de ponto: troca valor de um nó terminal
    2. Mutação de operador: troca um operador por outro
    3. Mutação de subárvore: substitui uma subárvore por uma nova aleatória
    """
    if random.random() > prob_mutacao:
        return individuo

    mutante = individuo.copiar()
    nos = obter_todos_nos(mutante)

    if len(nos) == 0:
        return mutante

    # Escolher um nó aleatório para mutar
    no_escolhido, caminho = random.choice(nos)

    tipo_mutacao = random.choice(['ponto', 'operador', 'subarvore'])

    if tipo_mutacao == 'ponto' and no_escolhido.esquerda is None:
        # Mutação de ponto: trocar terminal
        if isinstance(no_escolhido.valor, int):
            no_escolhido.valor = random.randint(0, num_features - 1)
        else:
            no_escolhido.valor = random.uniform(-10, 10)

    elif tipo_mutacao == 'operador' and no_escolhido.esquerda is not None:
        # Mutação de operador
        if no_escolhido.direita is None:  # operador unário
            no_escolhido.valor = random.choice(OPERADORES_UNARIOS)
        else:  # operador binário
            no_escolhido.valor = random.choice(OPERADORES_BINARIOS)

    else:
        # Mutação de subárvore: criar nova subárvore
        nova_subarvore = criar_individuo(num_features, profundidade_max=2)
        if caminho:
            substituir_no(mutante, caminho, nova_subarvore)
        else:
            mutante = nova_subarvore

    return mutante


# =============================================================================
# 9. ALGORITMO PRINCIPAL DE PROGRAMAÇÃO GENÉTICA
# =============================================================================

def programacao_genetica(X_treino, y_treino, X_teste, y_teste,
                         tamanho_pop=100,
                         num_geracoes=50,
                         prob_crossover=0.9,
                         prob_mutacao=0.2,
                         tamanho_elite=2,
                         profundidade_max=4):
    """
    Algoritmo principal de Programação Genética.

    Critério de parada:
    - Número máximo de gerações (padrão: 50)
    - Convergência: se o melhor fitness não melhorar por 10 gerações consecutivas
    """

    num_features = X_treino.shape[1]

    print("\n" + "=" * 60)
    print("INICIANDO PROGRAMAÇÃO GENÉTICA")
    print("=" * 60)
    print(f"Tamanho da população: {tamanho_pop}")
    print(f"Número máximo de gerações: {num_geracoes}")
    print(f"Probabilidade de crossover: {prob_crossover}")
    print(f"Probabilidade de mutação: {prob_mutacao}")
    print(f"Tamanho do elitismo: {tamanho_elite}")
    print(f"Profundidade máxima da árvore: {profundidade_max}")

    # Criar população inicial
    print("\nGerando população inicial...")
    populacao = criar_populacao(tamanho_pop, num_features, profundidade_max)

    # Histórico para análise de convergência
    historico_melhor_fitness = []
    historico_media_fitness = []
    geracoes_sem_melhoria = 0
    melhor_fitness_global = float('inf')
    melhor_individuo_global = None

    # Loop evolutivo principal
    for geracao in range(num_geracoes):

        # Calcular fitness de todos os indivíduos
        fitness_lista = [calcular_fitness(ind, X_treino, y_treino) for ind in populacao]

        # Encontrar melhor indivíduo da geração
        idx_melhor = np.argmin(fitness_lista)
        melhor_fitness = fitness_lista[idx_melhor]
        media_fitness = np.mean(fitness_lista)

        # Atualizar melhor global
        if melhor_fitness < melhor_fitness_global:
            melhor_fitness_global = melhor_fitness
            melhor_individuo_global = populacao[idx_melhor].copiar()
            geracoes_sem_melhoria = 0
        else:
            geracoes_sem_melhoria += 1

        # Salvar histórico
        historico_melhor_fitness.append(melhor_fitness)
        historico_media_fitness.append(media_fitness)

        # Mostrar progresso
        if geracao % 5 == 0 or geracao == num_geracoes - 1:
            print(f"Geração {geracao:3d} | Melhor Fitness: {melhor_fitness:10.2f} | "
                  f"Média: {media_fitness:10.2f} | Melhor Global: {melhor_fitness_global:10.2f}")

        # Critério de parada: convergência
        if geracoes_sem_melhoria >= 10:
            print(f"\n>>> Convergência detectada na geração {geracao}! "
                  f"(10 gerações sem melhoria)")
            break

        # --- ELITISMO ---
        # Ordenar população por fitness e manter os melhores
        indices_ordenados = np.argsort(fitness_lista)
        elite = [populacao[i].copiar() for i in indices_ordenados[:tamanho_elite]]

        # --- CRIAR NOVA POPULAÇÃO ---
        nova_populacao = elite.copy()

        while len(nova_populacao) < tamanho_pop:
            # Seleção por torneio binário
            pai1 = torneio_binario(populacao, fitness_lista)
            pai2 = torneio_binario(populacao, fitness_lista)

            # Cruzamento
            filho1, filho2 = cruzamento(pai1, pai2, prob_crossover)

            # Mutação
            filho1 = mutacao(filho1, num_features, prob_mutacao)
            filho2 = mutacao(filho2, num_features, prob_mutacao)

            nova_populacao.append(filho1)
            if len(nova_populacao) < tamanho_pop:
                nova_populacao.append(filho2)

        populacao = nova_populacao

    # Avaliação final
    print("\n" + "=" * 60)
    print("RESULTADOS FINAIS")
    print("=" * 60)

    # Calcular métricas no conjunto de TREINO
    previsoes_treino = prever(melhor_individuo_global, X_treino)
    mae_treino = np.mean(np.abs(y_treino - previsoes_treino))
    rmse_treino = np.sqrt(np.mean((y_treino - previsoes_treino) ** 2))

    # Calcular métricas no conjunto de TESTE
    previsoes_teste = prever(melhor_individuo_global, X_teste)
    mae_teste = np.mean(np.abs(y_teste - previsoes_teste))
    rmse_teste = np.sqrt(np.mean((y_teste - previsoes_teste) ** 2))

    print(f"\n--- Melhor Indivíduo (Expressão Matemática) ---")
    print(f"{melhor_individuo_global}")

    print(f"\n--- Desempenho no Treinamento ---")
    print(f"MAE (Erro Absoluto Médio): {mae_treino:.2f}")
    print(f"RMSE (Raiz do Erro Quadrático Médio): {rmse_treino:.2f}")

    print(f"\n--- Desempenho no Teste ---")
    print(f"MAE (Erro Absoluto Médio): {mae_teste:.2f}")
    print(f"RMSE (Raiz do Erro Quadrático Médio): {rmse_teste:.2f}")

    # Comparação de valores reais vs previstos (amostra)
    print(f"\n--- Comparação: Valores Reais vs Previstos (10 amostras do teste) ---")
    print(f"{'Real':>12} | {'Previsto':>12} | {'Erro':>12}")
    print("-" * 42)
    indices_amostra = random.sample(range(len(y_teste)), min(10, len(y_teste)))
    for i in indices_amostra:
        erro = abs(y_teste[i] - previsoes_teste[i])
        print(f"{y_teste[i]:12.2f} | {previsoes_teste[i]:12.2f} | {erro:12.2f}")

    return (melhor_individuo_global, historico_melhor_fitness, historico_media_fitness,
            mae_treino, mae_teste)


def plotar_convergencia(historico_melhor, historico_media, salvar=True):
    """Plota gráfico da evolução do fitness ao longo das gerações."""
    plt.figure(figsize=(10, 6))
    plt.plot(historico_melhor, label='Melhor Fitness', linewidth=2, color='blue')
    plt.plot(historico_media, label='Média do Fitness', linewidth=2, color='orange', alpha=0.7)
    plt.xlabel('Geração', fontsize=12)
    plt.ylabel('Fitness (MAE)', fontsize=12)
    plt.title('Evolução do Fitness ao Longo das Gerações', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if salvar:
        plt.savefig('/Users/luisacaetano/Downloads/Knime/convergencia_fitness.png', dpi=150)
        print("\nGráfico salvo em: convergencia_fitness.png")

    plt.show()


# =============================================================================
# 10. EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":

    # Definir seed para reprodutibilidade
    random.seed(42)
    np.random.seed(42)

    # Caminho do arquivo CSV
    CAMINHO_CSV = "/Users/luisacaetano/Downloads/Knime/car_price_dataset.csv"

    # 1. Carregar dados
    df = carregar_dados(CAMINHO_CSV)

    # 2. Preparar dados
    X, y, nomes_features = preparar_dados(df)

    # 3. Separar em treino (70%) e teste (30%)
    # Nota: Para regressão, usamos shuffle ao invés de stratify
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"\n--- Divisão dos Dados ---")
    print(f"Treino: {len(X_treino)} amostras (70%)")
    print(f"Teste: {len(X_teste)} amostras (30%)")

    # 4. Executar Programação Genética
    resultado = programacao_genetica(
        X_treino, y_treino,
        X_teste, y_teste,
        tamanho_pop=100,        # tamanho da população
        num_geracoes=50,        # número máximo de gerações
        prob_crossover=0.9,     # probabilidade de cruzamento
        prob_mutacao=0.2,       # probabilidade de mutação
        tamanho_elite=2,        # número de indivíduos elite
        profundidade_max=5      # profundidade máxima das árvores
    )

    melhor_ind, hist_melhor, hist_media, mae_treino, mae_teste = resultado

    # 5. Plotar convergência
    plotar_convergencia(hist_melhor, hist_media)

    print("\n" + "=" * 60)
    print("PROGRAMA FINALIZADO COM SUCESSO!")
    print("=" * 60)
