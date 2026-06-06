import numpy as np
import matplotlib.pyplot as plt

POP_SIZE = 40
GENERATIONS = 100
X_MIN, X_MAX = -1.0, 2.0
CROSSOVER_PROB = 0.9
MUTATION_PROB = 0.2
MUTATION_STD = 0.1


def fitness_function(x):
    return x * np.sin(10 * np.pi * x) + 1

def tournament_selection(pop, fitness, k=3):
    selected = []
    for _ in range(len(pop)):
        idx = np.random.choice(len(pop), k, replace=False)
        selected.append(pop[idx[np.argmax(fitness[idx])]])
    return np.array(selected)

def arithmetic_crossover(p1, p2):
    alpha = np.random.rand()
    return alpha * p1 + (1 - alpha) * p2, alpha * p2 + (1 - alpha) * p1

def mutate(x):
    if np.random.rand() < MUTATION_PROB:
        x += np.random.normal(0, MUTATION_STD)
    return np.clip(x, X_MIN, X_MAX)

best_history = []
mean_history = []

def execute(restrictions, initial, ff):
    np.random.seed(42)
    global population
    population = np.random.uniform(X_MIN, X_MAX, POP_SIZE)
    global restriction
    global fitness_function
    restriction = restrictions
    initial_solution = initial
    fitness_function = ff
    solutions = []
    #for _ in range(30):

    for _ in range(GENERATIONS):
        fitness = fitness_function(population)
        best_history.append(np.max(fitness))
        mean_history.append(np.mean(fitness))
        parents = tournament_selection(population, fitness)
        offspring = []
        np.random.shuffle(parents)
        for i in range(0, POP_SIZE, 2):
            if np.random.rand() < CROSSOVER_PROB:
                c1, c2 = arithmetic_crossover(parents[i], parents[i + 1])
            else:
                c1, c2 = parents[i], parents[i + 1]
            offspring.extend([mutate(c1), mutate(c2)])
        population = np.array(offspring)
    #solutions.append(solution)

    print("Melhor solucao (Variaveis de decisao): \n - x1 = {} \n - x2 = {}".format(min(solutions, key=fitness_function)[0], min(solutions, key=fitness_function)[1]))
    print("Solucao Minima: {}".format(fitness_function(min(solutions, key=fitness_function))))
    print("Solucao Maxima: {}".format(fitness_function(max(solutions, key=fitness_function))))
    print("Solucao Media: {}".format(sum(fitness_function(sol) for sol in solutions) / len(solutions)))
    print("Desvio Padrao: {}".format(math.sqrt(sum((fitness_function(sol) - (sum(fitness_function(sol) for sol in solutions) / len(solutions))) ** 2 for sol in solutions) / len(solutions))))

    objetive_values = [fitness_function(sol) for sol in solutions]

    plt.figure(figsize=(8, 6))
    plt.boxplot(objetive_values, vert=True, patch_artist=True)
    
    plt.title('Buca Tabu - Distribuição dos Valores da Função Objetivo')
    plt.ylabel('Valor da Função Objetivo')
    plt.xticks([1], ['Busca Tabu'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.show()