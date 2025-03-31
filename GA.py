import numpy as np
import random
import os
from test import get_function_details


def ga_optimize(func, lb, ub, dim, npop, cr, mr, max_evals):
    population = np.random.uniform(lb, ub, (npop, dim))
    fitness = np.array([func(ind) for ind in population])
    eval_count = npop
    best_idx = np.argmin(fitness)
    best = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    convergence_curve = [best_fitness]  # 记录初始适应度
    interval = max_evals // 30
    checkpoints = list(range(interval, max_evals + 1, interval))[:30]
    checkpoint_idx = 0

    while eval_count < max_evals:
        new_population = []
        new_fitness = []
        for i in range(npop):
            # 锦标赛选择
            idxs = random.sample(range(npop), 2)
            parent1 = population[idxs[0]] if fitness[idxs[0]] < fitness[idxs[1]] else population[idxs[1]]
            p1_fitness = fitness[idxs[0]] if fitness[idxs[0]] < fitness[idxs[1]] else fitness[idxs[1]]

            idxs = random.sample(range(npop), 2)
            parent2 = population[idxs[0]] if fitness[idxs[0]] < fitness[idxs[1]] else population[idxs[1]]

            # 交叉和变异
            mask = np.random.rand(dim) < cr
            child = np.where(mask, parent1, parent2)
            mutation_mask = np.random.rand(dim) < mr
            child[mutation_mask] = np.random.uniform(lb, ub, np.sum(mutation_mask))

            child_fitness = func(child)
            eval_count += 1

            # 记录收敛过程
            while checkpoint_idx < len(checkpoints) and eval_count >= checkpoints[checkpoint_idx]:
                convergence_curve.append(best_fitness)
                checkpoint_idx += 1

            # 更新种群
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

    # 填充剩余检查点
    while len(convergence_curve) < 30:
        convergence_curve.append(best_fitness)
    return best, best_fitness, eval_count, convergence_curve[:30]


def run_benchmark(repeat_times=5):
    functions = ['F' + str(i) for i in range(1, 14)]
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    total_tasks = len(functions) * (len(range(20, 101, 20)) * len(range(1, 10)) * len(range(1, 10))) * repeat_times
    completed_tasks = 0

    for f_name in functions:
        # 获取函数详情
        lb, ub, dim, func = get_function_details(f_name)
        max_evals = 10000 * dim

        # 遍历所有参数组合
        for npop in range(20, 101, 20):
            for cr_i in range(1, 10):
                cr = cr_i * 0.1
                for mr_i in range(1, 10):
                    mr = mr_i * 0.01
                    param_key = f"NP={npop}_CR={cr:.1f}_MR={mr:.2f}"
                    results_file = os.path.join(results_dir, f"{f_name}_{param_key}.txt")

                    # 运行多次实验并写入结果
                    with open(results_file, 'w') as f:
                        for run in range(repeat_times):
                            _, _, _, curve = ga_optimize(func, lb, ub, dim, npop, cr, mr, max_evals)
                            line = ' '.join(f"{val:.4e}" for val in curve)
                            f.write(line + '\n')

                            # 更新进度
                            completed_tasks += 1
                            progress = (completed_tasks / total_tasks) * 100
                            print(f"Progress: {progress:.2f}% | Function: {f_name} | Params: {param_key} | Run: {run + 1}/{repeat_times}")

    print("Benchmark completed. Results saved in the 'results' folder.")


if __name__ == "__main__":
    run_benchmark(repeat_times=5)