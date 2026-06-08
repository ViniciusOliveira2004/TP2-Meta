import math
import numpy as np
import algoritmos.AG_penalidade_estatica as AG_penalidade_estatica
import algoritmos.AG_constrained as AG_constrained

def avaluate_problem_1(individual):
    x1, x2, u1, u2 = individual
    f_obj = (x1**0.6) + (x2**0.6) - 6*x1 - 4*u1 + 3*u2 
    violations = []
    
    h1 = abs(x2 - 3*x1 - 3*u1) - 0.0001
    violations.append(max(0.0, h1))
    g1 = x1 + 2*u1 - 4 
    violations.append(max(0.0, g1))
    g2 = x2 + 2*u2 - 4 
    violations.append(max(0.0, g2))
    
    return f_obj, sum(violations)


def avaluate_problem_2(individual):
    x1, x2, x3, x4, x5, x6, x7 = individual
    f_obj = (0.7854 * x1 * (x2**2) * (3.3333*(x3**2) + 14.9334*x3 - 43.0934) - 
             1.508 * x1 * (x6**2 + x7**2) + 7.477 * (x6**3 + x7**3) + 
             0.7854 * (x4*(x6**2) + x5*(x7**2)))
    
    violations = []
    g1 = 27 - (x1 * (x2**2) * x3)
    g2 = 397.5 - (x1 * (x2**2) * (x3**2))
    g3 = 1.93 - (x2 * (x6**4) * x3 * (x4**-3))
    g4 = 1.93 - (x2 * (x7**4) * x3 * (x5**-3))
    
    A1 = ((745.0 * x4 / (x2 * x3))**2 + 16.9e6)**0.5
    B1 = 0.1 * (x6**3)
    g5 = (A1 * B1) - 1 - 1100
    
    A2 = ((745.0 * x5 / (x2 * x3))**2 + 157.5e6)**0.5
    B2 = 0.1 * (x7**3) 
    g6 = (A2 / B2) - 850 
    
    g7 = (x2 * x3) - 40
    g8 = 5 - (x1 / x2)
    g9 = (x1 / x2) - 12
    g10 = (1.5 * x6) - x4 + 1.9 
    g11 = (1.5 * x7) - x5 + 1.9 
    
    for g in [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11]:
        violations.append(max(0.0, g))
        
    return f_obj, sum(violations)

print("\nSelecione o problema:")
print(" 1 - Problema 1 - Penalidade Estatica")
print(" 2 - Problema 1 - ε-constrained")
print(" 3 - Problema 2 - Penalidade Estatica")
print(" 4 - Problema 2 - ε-constrained")
choice = input("Escolha (1-4): ")

if choice == "1":
    AG_penalidade_estatica.execute(
        ef=avaluate_problem_1, 
        nv=4,
        bmin=np.array([0.0, 0.0, 0.0, 0.0]),
        bmax=np.array([3.0, 10.0, 10.0, 1.0]),
        problem=1
    )
elif choice == "2":
    AG_constrained.execute(
        ef=avaluate_problem_1, 
        nv=4,
        bmin=np.array([0.0, 0.0, 0.0, 0.0]), 
        bmax=np.array([3.0, 10.0, 10.0, 1.0]),
        problem=1
    )
elif choice == "3":
    AG_penalidade_estatica.execute(
        ef=avaluate_problem_2, 
        nv=7,
        bmin=np.array([2.6, 0.7, 17.0, 7.3, 7.3, 2.9, 5.0]),
        bmax=np.array([3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5]),
        problem=2
    )
elif choice == "4":
    AG_constrained.execute(
        ef=avaluate_problem_2, 
        nv=7,
        bmin=np.array([2.6, 0.7, 17.0, 7.3, 7.3, 2.9, 5.0]),
        bmax=np.array([3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5]),
        problem=2
    )