#define _CRT_SECURE_NO_WARNINGS
#include "Self_Define_Functions.h"

#include <cstdio>

#include <algorithm>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <iomanip>
#include <string>
#include <boost/random.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/cauchy_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real.hpp>

using namespace std;
double* OShift, * M, * y, * z, * x_bound;
int ini_flag = 0, n_flag, func_flag, * SS;

int main(int argc, char* argv[])
{
    //for (double crossover_probability = 1; crossover_probability <= 1.01; crossover_probability += 0.05) {
        //for (double mutation_probability = 0.002; mutation_probability <= 0.051; mutation_probability += 0.002) {
    for (double Population_Size = 10; Population_Size <= 101; Population_Size += 10) {
        cout << "当前种群：" << Population_Size << endl;
        cout << "当前交叉概率：" << crossover_probability << endl;
        cout << "当前变异概率：" << mutation_probability << endl;

        int Tournament_Size = Population_Size * 0.2;

        // 定义需要运行的函数集合
        //int funToRun[] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30 };  //function set
        //int funToRun[] = { 6,7,8,9,10 };
        //int funToRun[] = {11,12,13,14,15};
        //int funToRun[] = {16,17,18,19,20};
        //int funToRun[] = {6,7,8,9,10,11,12,13,14,15};
        //int funToRun[] = { 8,9 };
        //int funToRun[] = {6,7,10,11,12,13,14,15};
        int funToRun[] = { 10, };
        //int funToRun[] = {6,7,8,9,11,12,13,14,15};
        int funNum = 1; // 总函数数量
        int function_index; // 函数索引
        int i, j;
        int run_index; // 运行次数索引
        int fitness_counter = 0; // 当前适应度评估次数
        int Max_Iter = MAX_FV / Population_Size; // 最大评估次数
        int gl_best = 0; // 全局最优位置索引
        double* final_global_best = new double[dim]; // 存储全局最优解
        double final_val; // 最终值
        double** population = new double* [Population_Size]; // 存储种群基因组
        double** selected_parents = new double* [Population_Size]; // 存储选定的父代
        for (i = 0; i < Population_Size; ++i)
        {
            population[i] = new double[dim];
            selected_parents[i] = new double[dim];
        }
        double current_best_result; // 当前最佳结果
        double r1, r2; // 随机数
        double* results = new double[Population_Size]; // 存储整个种群的适应度结果
        double* selection_probability = new double[Population_Size]; // 存储每个染色体的选择概率
        double MAX = 100; // 基因最大值
        double MIN = -100; // 基因最小值
        for (function_index = 0; function_index < funNum; function_index++)
        {
            cout << "开始运行函数 " << funToRun[function_index] << "!" << endl;
            char fun[10];
            snprintf(fun, 10, "%d", funToRun[function_index]);
            string filename_fitness = "D:/Genetic_Algorithm/GA/GA/Results/Fitness_result_for_function_" + string(fun) + "_" + to_string(Population_Size) + ".txt";
            //string filename_fitness = "D:/Genetic_Algorithm/GA/GA/Results/Fitness_result_for_function_" + string(fun) + "_" + to_string(crossover_probability) + "_" + to_string(mutation_probability) + ".txt";
            ofstream out_fitness(filename_fitness.c_str());
            if (!out_fitness)
            {
                cerr << "无法打开文件 " << filename_fitness << endl;
                exit(1);
            }

            for (run_index = 0; run_index < timesOfRun; run_index++)
            {

                int evaluations_per_output = MAX_FV / 30;
                int current_output_evaluation = evaluations_per_output;

                cout << "正在运行第 " << run_index << " 次！" << endl;
                boost::mt19937 generator(time(0) * rand());
                boost::uniform_real<> uniform_real_generate_r(0, 1);
                boost::variate_generator
                    < boost::mt19937&, boost::uniform_real<> > random_real_num_r(generator, uniform_real_generate_r); // 生成 [0,1] 范围内的随机数
                fitness_counter = 0;
                // 初始化种群
                for (i = 0; i < dim; ++i)
                {
                    boost::uniform_real<> uniform_real_generate_x(MIN, MAX);
                    boost::variate_generator
                        < boost::mt19937&, boost::uniform_real<> > random_real_num_x(generator, uniform_real_generate_x);
                    for (j = 0; j < Population_Size; ++j)
                    {
                        population[j][i] = random_real_num_x();
                    }
                }
                for (int i = 0; i < Population_Size; i++) // 计算种群的适应度
                {
                    cec14_test_func(population[i], &results[i], dim, 1, funToRun[function_index]);
                    results[i] = results[i] - funToRun[function_index] * 100;

                }
                fitness_counter += Population_Size;
                // 寻找全局最优位置
                final_val = results[0];
                gl_best = 0;
                for (i = 1; i < Population_Size; ++i)
                {
                    if (final_val > results[i])
                    {
                        final_val = results[i];
                        gl_best = i;
                    }
                }
                memcpy(final_global_best, population[gl_best], sizeof(double) * dim); // 更新全局最优位置
                // 主要迭代
                while (fitness_counter < MAX_FV)
                {
                    // 计算选择概率
                    //Selection_Probability(results, selection_probability, Population_Size);
                    // 选择操作
                    //Selection(selected_parents, population, selection_probability, Population_Size, dim);
                    Selection2(selected_parents, population, results, Population_Size, dim, Tournament_Size);
                    // 交叉
                    i = 0;
                    while (i < Population_Size)
                    {
                        if (random_real_num_r() <= crossover_probability)
                        {
                            j = i + 1;
                            while (j < Population_Size)
                            {
                                if (random_real_num_r() <= crossover_probability)
                                {
                                    Crossover(selected_parents[i], selected_parents[j], dim);
                                    break;
                                }
                                else
                                    ++j;

                            }
                            i = j + 1;
                        }
                        else
                            ++i;

                    }
                    // 变异
                    Mutation(selected_parents, MIN, MAX, mutation_probability, Population_Size, dim);

                    // 计算种群的适应度
                    for (int i = 0; i < Population_Size; i++) // 计算适应度
                    {
                        cec14_test_func(selected_parents[i], &results[i], dim, 1, funToRun[function_index]);
                        results[i] = results[i] - funToRun[function_index] * 100;

                    }
                    fitness_counter += Population_Size;
                    // 更新种群
                    for (i = 0; i < Population_Size; ++i)
                    {
                        memcpy(population[i], selected_parents[i], sizeof(double) * dim);
                    }
                    // 寻找当前最佳位置
                    current_best_result = results[0];
                    gl_best = 0;
                    for (i = 1; i < Population_Size; ++i)
                    {
                        if (current_best_result > results[i])
                        {
                            current_best_result = results[i];
                            gl_best = i;
                        }
                    }
                    // 更新全局最优
                    if (results[gl_best] < final_val)
                    {
                        final_val = results[gl_best];
                        memcpy(final_global_best, population[gl_best], sizeof(double) * dim);
                    }

                    if (fitness_counter >= current_output_evaluation)
                    {
                        out_fitness << final_val << " "; // 输出当前结果到文件
                        current_output_evaluation += evaluations_per_output; // 更新下一个输出的评估次数
                    }


                    // printf("%f\n", final_val);
                } // 结束 while
                //out_fitness << final_val << endl; // 输出最终结果到文件
                out_fitness << endl;
            } // 结束 for 运行次数
            out_fitness.close();
        } // 结束 for 函数集合

        // 释放资源
        for (i = 0; i < Population_Size; ++i)
        {
            delete[]population[i];
            delete[]selected_parents[i];
        }
        delete[]population;
        delete[]selected_parents;

        delete[]final_global_best;
        delete[]results;
        delete[]selection_probability;
    }
    
    return 0;
} // 结束 main
