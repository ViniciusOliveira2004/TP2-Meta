import math
import numpy as np

# Problema 1 - Avaliação Completa
def avaliar_problema_1(individuo):
    x1, x2, u1, u2 = individuo
    
    # Função Objetivo (Minimização)
    f_obj = (x1**0.6) + (x2**0.6) - 6*x1 - 4*u1 + 3*u2 [cite: 19]
    
    # Cálculo das Violações
    violacoes = []
    
    # Igualdade h1: x2 - 3x1 - 3u1 = 0 -> |h(x)| - 0.0001 <= 0
    h1 = abs(x2 - 3*x1 - 3*u1) - 0.0001 [cite: 16, 20]
    violacoes.append(max(0.0, h1))
    
    # Inequação g1: x1 + 2u1 <= 4
    g1 = x1 + 2*u1 - 4 [cite: 21]
    violacoes.append(max(0.0, g1))
    
    # Inequação g2: x2 + 2u2 <= 4
    g2 = x2 + 2*u2 - 4 [cite: 22]
    violacoes.append(max(0.0, g2))
    
    return f_obj, sum(violacoes)

# Problema 2 - Avaliação Completa (7 variáveis)
def avaliar_problema_2(individuo):
    x1, x2, x3, x4, x5, x6, x7 = individuo [cite: 26]
    
    # Função Objetivo (Minimização)
    f_obj = (0.7854 * x1 * (x2**2) * (3.3333*(x3**2) + 14.9334*x3 - 43.0934) - 
             1.508 * x1 * (x6**2 + x7**2) + 7.477 * (x6**3 + x7**3) + 
             0.7854 * (x4*(x6**2) + x5*(x7**2))) [cite: 27, 28, 29]
    
    violacoes = []
    
    # Restrições de Desigualdade (convertidas para <= 0)
    g1 = 27 - (x1 * (x2**2) * x3) [cite: 32]
    g2 = 397.5 - (x1 * (x2**2) * (x3**2)) [cite: 32]
    g3 = 1.93 - (x2 * (x6**4) * x3 * (x4**-3)) [cite: 33]
    g4 = 1.93 - (x2 * (x7**4) * x3 * (x5**-3)) [cite: 33]
    
    # Auxiliares para g5 e g6
    A1 = ((745.0 * x4 / (x2 * x3))**2 + 16.9e6)**0.5 [cite: 35]
    B1 = 0.1 * (x6**3) [cite: 35]
    g5 = (A1 * B1) - 1 - 1100 [cite: 34]
    
    A2 = ((745.0 * x5 / (x2 * x3))**2 + 157.5e6)**0.5 [cite: 37]
    B2 = 0.1 * (x7**3) [cite: 38]
    g6 = (A2 / B2) - 850 [cite: 36]
    
    g7 = (x2 * x3) - 40 [cite: 39]
    g8 = 5 - (x1 / x2) [cite: 40]
    g9 = (x1 / x2) - 12 [cite: 41]
    g10 = (1.5 * x6) - x4 + 1.9 [cite: 42]
    g11 = (1.5 * x7) - x5 + 1.9 [cite: 43]
    
    for g in [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11]:
        violacoes.append(max(0.0, g))
        
    return f_obj, sum(violacoes)


print("\nSelecione o problema:")
print(" 1 - Problema 1 - Penalidade Estatica")
print(" 2 - Problema 1 - ε-constrained")
print(" 3 - Problema 2 - Penalidade Estatica")
print(" 4 - Problema 2 - ε-constrained")
choice = input("Escolha (1-4): ")
restriction_function = None
initial_solution = None


print("\nSolucao inicial: {}".format(initial_solution))
print("Executando BUSCA TABU...")

print("\nExecutando SIMULATED ANNEALING...")
