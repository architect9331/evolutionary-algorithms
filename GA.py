# 新增 GPU 支持，如果可用则使用 cupy
try:
    import cupy as cp
    use_gpu_default = True
except ImportError:
    cp = None
    use_gpu_default = False
import numpy as np
import random
from test import get_function_details  # ...existing code importing F1～F24 as needed...

def tournament_select(population, fitness, tournament_size=2, xp=np):
    indices = xp.random.choice(len(population), tournament_size, replace=False)
    best = int(indices[0])
    for idx in indices:
        idx_int = int(idx)
        if fitness[idx_int] < fitness[best]:
            best = idx_int
    return population[best]

def crossover(parent1, parent2, xp=np):
    child1, child2 = xp.copy(parent1), xp.copy(parent2)
    for i in range(len(parent1)):
        if xp.random.rand() < 0.5:
            child1[i], child2[i] = parent2[i], parent1[i]
    return child1, child2

def mutation(child, lb, ub, mutation_rate=0.1, xp=np):
    for i in range(len(child)):
        if xp.random.rand() < mutation_rate:
            child[i] += xp.random.randn()
            child[i] = xp.clip(child[i], lb, ub)
    return child

def check_gpu():
    # 判断 GPU 是否可用：如果 cupy 已加载且能获取到设备数，则 GPU 可用
    if cp is not None:
        try:
            count = cp.cuda.runtime.getDeviceCount()
            print(f"GPU devices available: {count}")
            return count > 0
        except Exception as e:
            print(f"GPU 不可用: {e}")
            return False
    else:
        print("cupy 模块未安装，GPU 不可用")
        return False

def run_ga(fobj, lb, ub, dim, pop_size=50, generations=100, mutation_rate=0.1, cp_rate=0.5, use_gpu=use_gpu_default):
    xp = cp if (use_gpu and cp is not None) else np
    population = xp.random.uniform(lb, ub, (pop_size, dim))
    best, best_fitness = None, float('inf')
    for gen in range(generations):
        # 将个体转换为 numpy 数组供 fobj 计算
        fitness = [fobj(ind.get() if use_gpu and cp is not None else np.array(ind)) for ind in population]
        # 更新全局最优解
        for i in range(pop_size):
            if fitness[i] < best_fitness:
                best_fitness = fitness[i]
                best = population[i]
        new_population = []
        for _ in range(pop_size // 2):
            parent1 = tournament_select(population, fitness, xp=xp)
            parent2 = tournament_select(population, fitness, xp=xp)
            if xp.random.rand() < cp_rate:
                child1, child2 = crossover(parent1, parent2, xp=xp)
            else:
                child1, child2 = xp.copy(parent1), xp.copy(parent2)
            child1 = mutation(child1, lb, ub, mutation_rate, xp=xp)
            child2 = mutation(child2, lb, ub, mutation_rate, xp=xp)
            new_population.extend([child1, child2])
        population = xp.array(new_population)
    # 若使用 GPU，则将最佳解转换为 numpy 数组后返回
    if use_gpu and cp is not None:
        best = best.get()  # 修改：显式使用 .get() 进行转换
    return best, best_fitness

if __name__ == '__main__':
    # 检查 GPU 可用性
    gpu_status = check_gpu()
    status_line = "GPU 可用\n" if gpu_status else "GPU 不可用\n"
    print(status_line, end="")
    # 将输出结果写入文档，同时打印到控制台
    output_path = "d:\\pycharm_code\\output_results.txt"
    with open(output_path, "w", encoding="utf-8") as fout:
        fout.write(status_line)
        functions = ['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13']
        for NP in range(20, 101, 20):
            for cp_val in [round(x, 2) for x in np.arange(0.1, 1.0, 0.1)]:
                for mp in [round(x, 2) for x in np.arange(0.01, 0.1, 0.01)]:
                    line = f"Parameters: NP={NP}, Crossover Probability={cp_val}, Mutation Probability={mp}\n"
                    print(line, end="")
                    fout.write(line)
                    for func_name in functions:
                        lb, ub, dim, fobj = get_function_details(func_name)
                        max_evals = 10000 * dim
                        generations = max_evals // NP  # 计算代数
                        best_sol, best_fit = run_ga(fobj, lb, ub, dim, pop_size=NP, generations=generations, mutation_rate=mp, cp_rate=cp_val, use_gpu=True)
                        line = f'{func_name}: Best Fitness: {best_fit}\n'
                        print(line, end="")
                        fout.write(line)
                    line = "---------------------------------------------------\n"
                    print(line, end="")
                    fout.write(line)
