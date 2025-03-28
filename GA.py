import numpy as np
import random
import json  # 新增，用于写入结果文件
from test import get_function_details

def ga_optimize(func, lb, ub, dim, npop, cr, mr, max_evals):
    # 初始化种群
    population = np.random.uniform(lb, ub, (npop, dim))
    fitness = np.array([func(ind) for ind in population])
    eval_count = npop
    best_idx = np.argmin(fitness)
    best = population[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    while eval_count < max_evals:
        new_population = []
        new_fitness = []
        for i in range(npop):
            # 锦标赛选择获得两个父代并缓存父代适应度
            idxs = random.sample(range(npop), 2)
            if fitness[idxs[0]] <= fitness[idxs[1]]:
                parent1 = population[idxs[0]].copy()
                p1_fitness = fitness[idxs[0]]
            else:
                parent1 = population[idxs[1]].copy()
                p1_fitness = fitness[idxs[1]]
            idxs = random.sample(range(npop), 2)
            if fitness[idxs[0]] <= fitness[idxs[1]]:
                parent2 = population[idxs[0]].copy()
            else:
                parent2 = population[idxs[1]].copy()
            
            # 均匀交叉（采用向量化操作）
            mask = np.random.rand(dim) < cr
            child = np.where(mask, parent1, parent2)
            
            # 向量化变异：每个基因以 mr 概率重新随机生成
            mutation_mask = np.random.rand(dim) < mr
            child[mutation_mask] = np.random.uniform(lb, ub, np.sum(mutation_mask))
            
            child_fitness = func(child)
            eval_count += 1
            
            # 选择较优个体进入新种群
            if child_fitness < p1_fitness:
                new_population.append(child)
                new_fitness.append(child_fitness)
                if child_fitness < best_fitness:
                    best_fitness = child_fitness
                    best = child.copy()
            else:
                new_population.append(parent1)
                new_fitness.append(p1_fitness)
        population = np.array(new_population)
        fitness = np.array(new_fitness)
    return best, best_fitness, eval_count

def run_benchmark():
    functions = ['F' + str(i) for i in range(1, 14)]
    results = {}
    log_lines = []
    # 遍历所有参数组合：NP、CR、MR
    for npop in range(20, 101, 20):
        for cr_multiplier in range(1, 10):
            cr = cr_multiplier * 0.1
            for mr_multiplier in range(1, 10):
                mr = mr_multiplier * 0.01
                param_key = f"NP={npop}, CR={cr:.1f}, MR={mr:.2f}"
                results[param_key] = {}
                for f_name in functions:
                    lb, ub, dim, func = get_function_details(f_name)
                    max_evals = 10000 * dim
                    best, best_fitness, evals = ga_optimize(func, lb, ub, dim, npop, cr, mr, max_evals)
                    results[param_key][f_name] = best_fitness
                    line = f"{param_key} on Function {f_name}, Best Fitness: {best_fitness:.4e}"
                    print(line)
                    log_lines.append(line)
    # 将所有结果写入结果文件
    with open("d:\\pycharm_code\\results.txt", "w") as f:
        for line in log_lines:
            f.write(line + "\n")
    return results

if __name__ == "__main__":
    run_benchmark()
