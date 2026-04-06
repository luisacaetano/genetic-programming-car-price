"""
Trabalho de Programação Genética - Estimativa de Preço de Carros
Disciplina: Inteligência Artificial
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
# LEITURA E PREPARAÇÃO DOS DADOS
# =============================================================================

def carregar_dados(caminho_csv):
    """Carrega o CSV e mostra informações básicas da base."""
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
    """Transforma colunas categóricas em números e separa X e y."""
    df_encoded = df.copy()

    # colunas de texto que precisam virar números
    colunas_categoricas = ['Brand', 'Model', 'Fuel_Type', 'Transmission']

    encoders = {}
    for col in colunas_categoricas:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        encoders[col] = le

    # X = features, y = preço (o que queremos prever)
    X = df_encoded.drop('Price', axis=1)
    y = df_encoded['Price']

    X = X.values.astype(float)
    y = y.values.astype(float)

    nomes_features = list(df_encoded.drop('Price', axis=1).columns)

    print(f"\nFeatures utilizadas: {nomes_features}")
    print(f"Shape X: {X.shape}, Shape y: {y.shape}")

    return X, y, nomes_features


# =============================================================================
# REPRESENTAÇÃO DOS INDIVÍDUOS (ÁRVORE DE EXPRESSÃO)
# =============================================================================

class No:
    """Cada nó pode ser um operador (+, -, etc) ou um valor (variável/constante)."""

    def __init__(self, valor, esquerda=None, direita=None):
        self.valor = valor
        self.esquerda = esquerda
        self.direita = direita

    def __str__(self):
        return self._to_string()

    def _to_string(self):
        """Monta a expressão em formato legível."""
        if self.esquerda is None and self.direita is None:
            # é uma folha (variável ou constante)
            if isinstance(self.valor, int):
                return f"X{self.valor}"
            else:
                return str(round(self.valor, 2))
        elif self.direita is None:
            # operador com 1 argumento (sqrt, round)
            return f"{self.valor}({self.esquerda._to_string()})"
        else:
            # operador com 2 argumentos (+, -, *, /, **)
            return f"({self.esquerda._to_string()} {self.valor} {self.direita._to_string()})"

    def copiar(self):
        return copy.deepcopy(self)


# operadores que o enunciado pede
OPERADORES_BINARIOS = ['+', '-', '*', '/', '**']
OPERADORES_UNARIOS = ['sqrt', 'round']


def criar_no_aleatorio(num_features, profundidade_atual, profundidade_max):
    """Cria um nó aleatório, respeitando a profundidade máxima."""

    if profundidade_atual >= profundidade_max:
        return criar_terminal(num_features)

    escolha = random.random()

    if escolha < 0.3:
        return criar_terminal(num_features)
    elif escolha < 0.5:
        op = random.choice(OPERADORES_UNARIOS)
        filho = criar_no_aleatorio(num_features, profundidade_atual + 1, profundidade_max)
        return No(op, filho, None)
    else:
        op = random.choice(OPERADORES_BINARIOS)
        esq = criar_no_aleatorio(num_features, profundidade_atual + 1, profundidade_max)
        dir = criar_no_aleatorio(num_features, profundidade_atual + 1, profundidade_max)
        return No(op, esq, dir)


def criar_terminal(num_features):
    """Cria uma folha: ou uma variável (X0, X1, ...) ou uma constante."""
    if random.random() < 0.7:
        return No(random.randint(0, num_features - 1))
    else:
        return No(random.uniform(-10, 10))


def criar_individuo(num_features, profundidade_max=4):
    """Gera uma árvore de expressão aleatória."""
    return criar_no_aleatorio(num_features, 0, profundidade_max)


# =============================================================================
# AVALIAÇÃO DAS EXPRESSÕES
# =============================================================================

def avaliar_expressao(no, X_linha):
    """Calcula o valor da expressão para uma linha de dados."""
    try:
        if no.esquerda is None and no.direita is None:
            if isinstance(no.valor, int):
                return X_linha[no.valor]
            else:
                return no.valor

        elif no.direita is None:
            val_esq = avaliar_expressao(no.esquerda, X_linha)

            if no.valor == 'sqrt':
                return np.sqrt(abs(val_esq))
            elif no.valor == 'round':
                return round(val_esq)

        else:
            val_esq = avaliar_expressao(no.esquerda, X_linha)
            val_dir = avaliar_expressao(no.direita, X_linha)

            if no.valor == '+':
                return val_esq + val_dir
            elif no.valor == '-':
                return val_esq - val_dir
            elif no.valor == '*':
                return val_esq * val_dir
            elif no.valor == '/':
                # evita divisão por zero
                if abs(val_dir) < 1e-10:
                    return val_esq
                return val_esq / val_dir
            elif no.valor == '**':
                if abs(val_esq) < 1e-10:
                    return 0
                # limita o expoente pra não explodir
                exp = max(-5, min(5, val_dir))
                resultado = abs(val_esq) ** exp
                if np.isnan(resultado) or np.isinf(resultado):
                    return val_esq
                return resultado

    except Exception:
        return 0

    return 0


def prever(individuo, X):
    """Aplica a expressão em todas as linhas e retorna as previsões."""
    previsoes = []
    for i in range(len(X)):
        pred = avaliar_expressao(individuo, X[i])
        pred = float(pred) if pred is not None else 0.0
        if np.isnan(pred) or np.isinf(pred):
            pred = 0.0
        pred = max(-1e10, min(1e10, pred))
        previsoes.append(pred)
    return np.array(previsoes)


# =============================================================================
# FUNÇÃO DE FITNESS
# =============================================================================

def calcular_fitness(individuo, X, y):
    """
    Usa o MAE (erro absoluto médio) como fitness.
    Escolhi MAE porque é fácil de interpretar: representa o erro médio em reais.
    Também é mais robusto a outliers que o MSE.
    Quanto menor o MAE, melhor o indivíduo.
    """
    previsoes = prever(individuo, X)
    mae = np.mean(np.abs(y - previsoes))

    # penaliza árvores muito grandes pra evitar expressões gigantes
    tamanho = contar_nos(individuo)
    penalidade = tamanho * 0.1

    return mae + penalidade


def contar_nos(no):
    """Conta quantos nós tem na árvore."""
    if no is None:
        return 0
    return 1 + contar_nos(no.esquerda) + contar_nos(no.direita)


# =============================================================================
# POPULAÇÃO INICIAL
# =============================================================================

def criar_populacao(tamanho_pop, num_features, profundidade_max=4):
    """Gera a primeira geração de indivíduos aleatórios."""
    populacao = []
    for _ in range(tamanho_pop):
        individuo = criar_individuo(num_features, profundidade_max)
        populacao.append(individuo)
    return populacao


# =============================================================================
# SELEÇÃO POR TORNEIO BINÁRIO
# =============================================================================

def torneio_binario(populacao, fitness_lista):
    """
    Pega 2 indivíduos aleatórios e retorna o melhor (menor fitness).
    É o método de seleção pedido no enunciado.
    """
    idx1, idx2 = random.sample(range(len(populacao)), 2)

    if fitness_lista[idx1] < fitness_lista[idx2]:
        return populacao[idx1].copiar()
    else:
        return populacao[idx2].copiar()


# =============================================================================
# CRUZAMENTO (CROSSOVER)
# =============================================================================

def obter_todos_nos(no, lista_nos=None, caminho=None):
    """Percorre a árvore e guarda todos os nós com o caminho até eles."""
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
    """Troca um nó específico por outro, seguindo o caminho."""
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
    Troca subárvores entre dois pais pra gerar dois filhos.
    Escolhe um ponto aleatório em cada pai e faz a troca.
    """
    if random.random() > prob_crossover:
        return pai1.copiar(), pai2.copiar()

    filho1 = pai1.copiar()
    filho2 = pai2.copiar()

    nos_f1 = obter_todos_nos(filho1)
    nos_f2 = obter_todos_nos(filho2)

    if len(nos_f1) < 2 or len(nos_f2) < 2:
        return filho1, filho2

    _, caminho1 = random.choice(nos_f1[1:]) if len(nos_f1) > 1 else nos_f1[0]
    no2, caminho2 = random.choice(nos_f2[1:]) if len(nos_f2) > 1 else nos_f2[0]

    atual = filho1
    for direcao in caminho1:
        if direcao == 'esquerda':
            atual = atual.esquerda
        else:
            atual = atual.direita
    subarvore1 = atual.copiar()

    if caminho1:
        substituir_no(filho1, caminho1, no2.copiar())
    if caminho2:
        substituir_no(filho2, caminho2, subarvore1)

    return filho1, filho2


# =============================================================================
# MUTAÇÃO
# =============================================================================

def mutacao(individuo, num_features, prob_mutacao=0.2):
    """
    Faz pequenas alterações no indivíduo. Pode ser:
    - trocar o valor de uma folha (variável ou constante)
    - trocar um operador por outro
    - substituir uma subárvore inteira por uma nova
    """
    if random.random() > prob_mutacao:
        return individuo

    mutante = individuo.copiar()
    nos = obter_todos_nos(mutante)

    if len(nos) == 0:
        return mutante

    no_escolhido, caminho = random.choice(nos)

    tipo_mutacao = random.choice(['ponto', 'operador', 'subarvore'])

    if tipo_mutacao == 'ponto' and no_escolhido.esquerda is None:
        if isinstance(no_escolhido.valor, int):
            no_escolhido.valor = random.randint(0, num_features - 1)
        else:
            no_escolhido.valor = random.uniform(-10, 10)

    elif tipo_mutacao == 'operador' and no_escolhido.esquerda is not None:
        if no_escolhido.direita is None:
            no_escolhido.valor = random.choice(OPERADORES_UNARIOS)
        else:
            no_escolhido.valor = random.choice(OPERADORES_BINARIOS)

    else:
        nova_subarvore = criar_individuo(num_features, profundidade_max=2)
        if caminho:
            substituir_no(mutante, caminho, nova_subarvore)
        else:
            mutante = nova_subarvore

    return mutante


# =============================================================================
# ALGORITMO PRINCIPAL
# =============================================================================

def programacao_genetica(X_treino, y_treino, X_teste, y_teste,
                         tamanho_pop=100,
                         num_geracoes=50,
                         prob_crossover=0.9,
                         prob_mutacao=0.2,
                         tamanho_elite=2,
                         profundidade_max=4):
    """
    Loop principal da programação genética.
    Para quando atinge o máximo de gerações ou quando fica 15 gerações sem melhorar.
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

    print("\nGerando população inicial...")
    populacao = criar_populacao(tamanho_pop, num_features, profundidade_max)

    historico_melhor_fitness = []
    historico_media_fitness = []
    geracoes_sem_melhoria = 0
    melhor_fitness_global = float('inf')
    melhor_individuo_global = None

    for geracao in range(num_geracoes):

        fitness_lista = [calcular_fitness(ind, X_treino, y_treino) for ind in populacao]

        idx_melhor = np.argmin(fitness_lista)
        melhor_fitness = fitness_lista[idx_melhor]
        media_fitness = np.mean(fitness_lista)

        if melhor_fitness < melhor_fitness_global:
            melhor_fitness_global = melhor_fitness
            melhor_individuo_global = populacao[idx_melhor].copiar()
            geracoes_sem_melhoria = 0
        else:
            geracoes_sem_melhoria += 1

        historico_melhor_fitness.append(melhor_fitness)
        historico_media_fitness.append(media_fitness)

        if geracao % 5 == 0 or geracao == num_geracoes - 1:
            print(f"Geração {geracao:3d} | Melhor Fitness: {melhor_fitness:10.2f} | "
                  f"Média: {media_fitness:10.2f} | Melhor Global: {melhor_fitness_global:10.2f}")

        # para se ficou estagnado
        if geracoes_sem_melhoria >= 15:
            print(f"\n>>> Convergiu na geração {geracao} (15 gerações sem melhoria)")
            break

        # elitismo: guarda os melhores
        indices_ordenados = np.argsort(fitness_lista)
        elite = [populacao[i].copiar() for i in indices_ordenados[:tamanho_elite]]

        # monta a nova geração
        nova_populacao = elite.copy()

        while len(nova_populacao) < tamanho_pop:
            pai1 = torneio_binario(populacao, fitness_lista)
            pai2 = torneio_binario(populacao, fitness_lista)

            filho1, filho2 = cruzamento(pai1, pai2, prob_crossover)

            filho1 = mutacao(filho1, num_features, prob_mutacao)
            filho2 = mutacao(filho2, num_features, prob_mutacao)

            nova_populacao.append(filho1)
            if len(nova_populacao) < tamanho_pop:
                nova_populacao.append(filho2)

        populacao = nova_populacao

    # resultados finais
    print("\n" + "=" * 60)
    print("RESULTADOS FINAIS")
    print("=" * 60)

    previsoes_treino = prever(melhor_individuo_global, X_treino)
    mae_treino = np.mean(np.abs(y_treino - previsoes_treino))
    rmse_treino = np.sqrt(np.mean((y_treino - previsoes_treino) ** 2))

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
    """Mostra o gráfico de como o fitness evoluiu."""
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
        plt.savefig('/Users/luisacaetano/genetic-programming-car-price/convergencia_fitness.png', dpi=150)
        print("\nGráfico salvo em: convergencia_fitness.png")

    plt.show()


# =============================================================================
# EXECUÇÃO
# =============================================================================

if __name__ == "__main__":

    random.seed(42)
    np.random.seed(42)

    CAMINHO_CSV = "/Users/luisacaetano/genetic-programming-car-price/car_price_dataset.csv"

    df = carregar_dados(CAMINHO_CSV)

    X, y, nomes_features = preparar_dados(df)

    # divide em treino/teste de forma estratificada (mantém distribuição de preços parecida)
    n_bins = 10
    y_binned = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y_binned
    )

    print(f"\n--- Verificação da Estratificação ---")
    print(f"Média dos preços no treino: {np.mean(y_treino):.2f}")
    print(f"Média dos preços no teste: {np.mean(y_teste):.2f}")
    print(f"Desvio padrão no treino: {np.std(y_treino):.2f}")
    print(f"Desvio padrão no teste: {np.std(y_teste):.2f}")

    print(f"\n--- Divisão dos Dados ---")
    print(f"Treino: {len(X_treino)} amostras (70%)")
    print(f"Teste: {len(X_teste)} amostras (30%)")

    resultado = programacao_genetica(
        X_treino, y_treino,
        X_teste, y_teste,
        tamanho_pop=150,
        num_geracoes=70,
        prob_crossover=0.85,
        prob_mutacao=0.35,
        tamanho_elite=3,
        profundidade_max=5
    )

    melhor_ind, hist_melhor, hist_media, mae_treino, mae_teste = resultado

    plotar_convergencia(hist_melhor, hist_media)

    print("\n" + "=" * 60)
    print("PROGRAMA FINALIZADO!")
    print("=" * 60)
