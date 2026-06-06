import numpy as np
import matplotlib.pyplot as plt
import math

POP_SIZE = 40
GENERATIONS = 100
CROSSOVER_PROB = 0.9
MUTATION_PROB = 0.2
MUTATION_STD = 0.1

def tournament_selection(pop, obj_values, violations, epsilon, k=3):
    selected = []
    for _ in range(len(pop)):
        idx = np.random.choice(len(pop), k, replace=False)
        best_idx = idx[0]
        
        # Implementação das Regras de Comparação Lexicográfica baseadas em Epsilon
        for cand_idx in idx[1:]:
            v_best = violations[best_idx]
            v_cand = violations[cand_idx]
            f_best = obj_values[best_idx]
            f_cand = obj_values[cand_idx]
            
            # Regra 1: Ambos são soluções epsilon-factíveis -> Escolhe o de menor função objetivo
            if v_cand <= epsilon and v_best <= epsilon:
                if f_cand < f_best:
                    best_idx = cand_idx
            # Regra 2: O candidato é epsilon-factível e o atual melhor não é -> Candidato ganha
            elif v_cand <= epsilon and v_best > epsilon:
                best_idx = cand_idx
            # Regra 3: Ambos violam além de epsilon -> Escolhe o que possui a menor violação total
            elif v_cand > epsilon and v_best > epsilon:
                if v_cand < v_best:
                    best_idx = cand_idx
                    
        selected.append(pop[best_idx])
    return np.array(selected)

def arithmetic_crossover(p1, p2):
    alpha = np.random.rand()
    c1 = alpha * p1 + (1 - alpha) * p2
    c2 = alpha * p2 + (1 - alpha) * p1
    return c1, c2

def mutate(x, bounds_min, bounds_max):
    if np.random.rand() < MUTATION_PROB:
        x += np.random.normal(0, MUTATION_STD, size=len(x))
    return np.clip(x, bounds_min, bounds_max)

def execute(ef, nv, bmin, bmax, problem):
    global num_vars
    global bounds_min
    global bounds_max
    global eval_func
    num_vars = nv
    bounds_min = bmin
    bounds_max = bmax
    eval_func = ef

    best_solutions = []  

    for run in range(30): 
        population = np.random.uniform(bounds_min, bounds_max, (POP_SIZE, num_vars))
        best_run_obj = float('inf')
        
        for gen in range(GENERATIONS):
            obj_values = np.zeros(POP_SIZE)
            violations = np.zeros(POP_SIZE)
            for i in range(POP_SIZE): 
                obj_values[i], violations[i] = eval_func(population[i])
            
            # Dinâmica do Epsilon: Cálculo do limite inicial baseado na geração zero
            if gen == 0:
                epsilon_0 = np.max(violations) if np.max(violations) > 0 else 1.0
            
            # Controle de decaimento não-linear do nível de tolerância (Epsilon) até 80% das gerações
            T_c = int(0.8 * GENERATIONS)
            if gen < T_c:
                epsilon = epsilon_0 * ((1 - gen / T_c) ** 3)
            else:
                epsilon = 0.0
            
            for i in range(POP_SIZE):
                if violations[i] == 0 and obj_values[i] < best_run_obj:
                    best_run_obj = obj_values[i]
            
            # Substituição da penalidade pelo envio direto das violações e do epsilon atual para a seleção
            parents = tournament_selection(population, obj_values, violations, epsilon)
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
            
        if best_run_obj == float('inf'):
            best_run_obj = obj_values[np.argmin(violations)]
            
        best_solutions.append(best_run_obj)

    min_val = np.min(best_solutions) 
    max_val = np.max(best_solutions) 
    mean_val = np.mean(best_solutions) 
    std_val = np.std(best_solutions) 

    print("\n--- RESULTADOS FINAIS (30 EXECUÇÕES) ---")
    print(f"Mínimo (Melhor): {min_val:.4f}") 
    print(f"Máximo: {max_val:.4f}") 
    print(f"Média: {mean_val:.4f}") 
    print(f"Desvio Padrão: {std_val:.4f}") 

    plt.figure(figsize=(6, 5))
    plt.boxplot(best_solutions, vert=True, patch_artist=True)
    plt.title(f'Algoritmo Genético - Problema {problem}\n(ε-constrained)')
    plt.ylabel('Valor da Função Objetivo')
    plt.xticks([1], ['Configuração B'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()