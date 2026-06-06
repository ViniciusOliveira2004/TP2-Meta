import numpy as np
import matplotlib.pyplot as plt
import math

POP_SIZE = 40
GENERATIONS = 100
CROSSOVER_PROB = 0.9
MUTATION_PROB = 0.2
MUTATION_STD = 0.1
PENALTY_FACTOR = 1000000  # Constante alta para a Penalidade Estática

# 1. Ajuste da Seleção para MINIMIZAÇÃO
def tournament_selection(pop, fitness, k=3):
    selected = []
    for _ in range(len(pop)):
        idx = np.random.choice(len(pop), k, replace=False)
        selected.append(pop[idx[np.argmin(fitness[idx])]])
    return np.array(selected)

# 2. Ajuste do Cruzamento Aritmético para Vetores Multidimensionais
def arithmetic_crossover(p1, p2):
    alpha = np.random.rand()
    c1 = alpha * p1 + (1 - alpha) * p2
    c2 = alpha * p2 + (1 - alpha) * p1
    return c1, c2

# 3. Ajuste da Mutação com Limites Vetoriais específicos por problema
def mutate(x, bounds_min, bounds_max):
    if np.random.rand() < MUTATION_PROB:
        x += np.random.normal(0, MUTATION_STD, size=len(x))
    return np.clip(x, bounds_min, bounds_max)

# 4. Função de Execução Principal (Com as 30 rodadas independentes)
def execute(ef, nv, bmin, bmax, problem):
    global num_vars
    global bounds_min
    global bounds_max
    global eval_func
    num_vars = nv
    bounds_min = bmin
    bounds_max = bmax
    eval_func = ef

    best_solutions = []  # Vai guardar o valor real do objetivo de cada uma das 30 execuções

    # Loop exigido: 30 execuções independentes
    for run in range(30): #[cite: 54]
        population = np.random.uniform(bounds_min, bounds_max, (POP_SIZE, num_vars))
        
        best_run_obj = float('inf')
        
        for _ in range(GENERATIONS):
            # Calcula Função Objetivo e Violação para cada indivíduo
            obj_values = np.zeros(POP_SIZE)
            violations = np.zeros(POP_SIZE)
            for i in range(POP_SIZE): 
                obj_values[i], violations[i] = eval_func(population[i])
            
            # --- FÓRMULA DA PENALIDADE ESTÁTICA ---
            # Fitness Penalizado = f(x) + C * soma_das_violacoes
            fitness_penalizado = obj_values + (PENALTY_FACTOR * violations)
            
            # Rastreia a melhor solução factível da geração
            for i in range(POP_SIZE):
                if violations[i] == 0 and obj_values[i] < best_run_obj:
                    best_run_obj = obj_values[i]
            
            # Evolução: Seleção, Cruzamento e Mutação
            parents = tournament_selection(population, fitness_penalizado)
            offspring = []
            np.random.shuffle(parents)
            for i in range(0, POP_SIZE, 2):
                if np.random.rand() < CROSSOVER_PROB:
                    c1, c2 = arithmetic_crossover(parents[i], parents[i+1])
                else:
                    c1, c2 = parents[i], parents[i+1]
                offspring.extend([mutate(c1, bounds_min, bounds_max), 
                                  mutate(c2, bounds_min, bounds_max)])
            population = np.array(offspring)
            
        # Caso não tenha achado nenhuma 100% factível, pega a de menor fitness penalizado
        if best_run_obj == float('inf'):
            best_run_obj = obj_values[np.argmin(fitness_penalizado)]
            
        best_solutions.append(best_run_obj)

    # --- CÁLCULO DAS ESTATÍSTICAS FINAIS (30 Rodadas) ---
    min_val = np.min(best_solutions) #[cite: 55]
    max_val = np.max(best_solutions) #[cite: 55]
    mean_val = np.mean(best_solutions) #[cite: 55]
    std_val = np.std(best_solutions) #[cite: 55]

    print("\n--- RESULTADOS FINAIS (30 EXECUÇÕES) ---")
    print(f"Mínimo (Melhor): {min_val:.4f}") #[cite: 55]
    print(f"Máximo: {max_val:.4f}") #[cite: 55]
    print(f"Média: {mean_val:.4f}") #[cite: 55]
    print(f"Desvio Padrão: {std_val:.4f}") #[cite: 55]

    # --- GERAR BOXPLOT EXIGIDO ---
    plt.figure(figsize=(6, 5))
    plt.boxplot(best_solutions, vert=True, patch_artist=True)
    plt.title(f'Algoritmo Genético - Problema {problem}\n(Penalidade Estática)')
    plt.ylabel('Valor da Função Objetivo')
    plt.xticks([1], ['Configuração A'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()