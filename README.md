# Programação Genética para Predição de Preços de Carros

Trabalho prático da disciplina de **Inteligência Artificial** do curso de Bacharelado em Ciência da Computação.

## Objetivo

Desenvolver um sistema em Python que utiliza **Programação Genética** para encontrar uma expressão matemática capaz de estimar o preço de carros com base em seus atributos.

## O que é Programação Genética?

Programação Genética (PG) é uma técnica de computação evolutiva inspirada na teoria da evolução de Darwin. Ao invés de usar fórmulas fixas como na regressão linear tradicional, a PG "evolui" expressões matemáticas através de:

- **Seleção**: os melhores indivíduos (expressões) têm mais chance de se reproduzir
- **Cruzamento**: duas expressões trocam partes entre si, gerando filhos
- **Mutação**: pequenas alterações aleatórias introduzem variação
- **Elitismo**: os melhores sempre sobrevivem para a próxima geração

## Base de Dados

Utilizamos o dataset [Car Price Dataset](https://www.kaggle.com/datasets/mos3santos/conjunto-de-dados-de-precos-de-carros) do Kaggle, contendo informações sobre carros:

| Atributo | Descrição |
|----------|-----------|
| Brand | Marca do carro |
| Model | Modelo |
| Year | Ano de fabricação |
| Engine_Size | Tamanho do motor |
| Fuel_Type | Tipo de combustível |
| Transmission | Tipo de câmbio |
| Mileage | Quilometragem |
| Doors | Número de portas |
| Owner_Count | Quantidade de donos anteriores |
| **Price** | Preço (variável alvo) |

## Estrutura do Projeto

```
programacao-genetica-carros/
├── programacao_genetica_carros.py   # Código principal
├── car_price_dataset.csv            # Base de dados
├── convergencia_fitness.png         # Gráfico de convergência (gerado)
└── README.md                        # Este arquivo
```

## Requisitos

```bash
pip install pandas numpy scikit-learn matplotlib
```

## Como Executar

```bash
python programacao_genetica_carros.py
```

## Operadores Implementados

Conforme especificado no enunciado:

| Operador | Tipo | Descrição |
|----------|------|-----------|
| `+` | Binário | Soma |
| `-` | Binário | Subtração |
| `*` | Binário | Multiplicação |
| `/` | Binário | Divisão (protegida) |
| `**` | Binário | Potenciação |
| `sqrt` | Unário | Raiz quadrada |
| `round` | Unário | Arredondamento |

## Parâmetros do Algoritmo

| Parâmetro | Valor |
|-----------|-------|
| Tamanho da população | 100 |
| Número máximo de gerações | 50 |
| Probabilidade de crossover | 90% |
| Probabilidade de mutação | 20% |
| Tamanho do elitismo | 2 |
| Profundidade máxima da árvore | 5 |

## Função de Fitness

Utilizamos o **MAE (Mean Absolute Error)** como função de fitness:

```
MAE = (1/n) * Σ|y_real - y_previsto|
```

**Justificativa:**
- É intuitiva: representa o erro médio em unidades de preço
- É robusta a outliers comparada ao MSE
- Facilita interpretação: "em média, o modelo erra X reais"

## Critério de Parada

O algoritmo para quando:
1. Atinge o número máximo de gerações (50), **OU**
2. Detecta convergência: 10 gerações consecutivas sem melhoria no melhor fitness

## Saídas do Programa

- Melhor expressão matemática encontrada
- MAE e RMSE no conjunto de treino e teste
- Comparação entre valores reais e previstos
- Gráfico de evolução do fitness ao longo das gerações

## Autora

Luisa Caetano

## Licença

Este projeto é de uso acadêmico.
