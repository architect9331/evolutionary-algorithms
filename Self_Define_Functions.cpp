
#include "Self_Define_Functions.h"
#include <math.h>
#include <iostream>
#include <fstream>
const double PI = 3.1415926535897932384626433832795;
// the following function is the example in the book
void Selection_Probability(double *results, double *probability, int population_size)
{
	int i;

	double sum = 0;
	for (i = 0; i < population_size; ++i)
	{
		sum += 1.0/results[i];
	}

	for (i = 0; i < population_size; ++i)
	{
		probability[i] = (1.0/results[i]) / sum;
	}

	for (i = 1; i < population_size; ++i)
	{
		probability[i] = probability[i - 1] + probability[i];
	}
}
int Roulette_Selection(double *probability, int population_size)
{
	int i, selected_index;

	boost::mt19937 generator(time(0)*rand());
	boost::uniform_real<> uniform_real_generate_r(0, 1);
	boost::variate_generator< boost::mt19937&, boost::uniform_real<> > random_real_num_r(generator, uniform_real_generate_r);//to generate a random number within [0,1]

	double random_pr = random_real_num_r();

	selected_index = -1;

	for (i = 0; i < population_size; ++i)
	{
		if (random_pr <= probability[i])
		{
			selected_index = i;
			break;
		}
	}

	return selected_index;

}

void Selection(double **selected_parent, double **population, double *probability, int population_size, int dim)
{
	int i;

	int selected_index;

	for (i = 0; i < population_size; ++i)
	{
		selected_index = Roulette_Selection(probability, population_size);
		memcpy(selected_parent[i], population[selected_index], sizeof(double)*dim);
	}
}

int Tournament_Selection(double* fitness_values,int population_size,int tournament_size)
{
	int i, rand_index;
	double current_best_fitness = DBL_MAX;
	int best_index = -1;

	// 设置随机数生成器
	boost::mt19937 generator(time(0) * rand());

	// 定义 0 到 population_size 的整数均匀分布
	boost::uniform_int<> uniform_int_generate_r(0, population_size);

	// 定义变量生成器
	boost::variate_generator<boost::mt19937&, boost::uniform_int<>> random_int_num_r(generator, uniform_int_generate_r);

	for (i = 0; i < tournament_size; ++i)
	{
		rand_index = generator() % population_size;
		if (current_best_fitness > fitness_values[rand_index])
		{
			current_best_fitness = fitness_values[rand_index];
			best_index = rand_index;
		}
	}

	return best_index;
}

//int Tournament_Selection(double* fitness_values, int population_size, int tournament_size)
//{
//	int i, rand_index;
//	double current_best_fitness = DBL_MAX;
//	int best_index = -1;
//
//	// 设置随机数生成器
//	boost::mt19937 generator(time(0) * rand());
//
//	// 创建索引列表
//	std::vector<int> indices(population_size);
//	for (int i = 0; i < population_size; ++i)
//	{
//		indices[i] = i;
//	}
//
//	// 打乱索引列表
//	std::shuffle(indices.begin(), indices.end(), generator);
//
//	// 从打乱的索引列表中选择前tournament_size个
//	for (i = 0; i < tournament_size; ++i)
//	{
//		rand_index = indices[i];
//		if (current_best_fitness > fitness_values[rand_index])
//		{
//			current_best_fitness = fitness_values[rand_index];
//			best_index = rand_index;
//		}
//	}
//
//	return best_index;
//}


void Selection2(double** selected_parent, double** population, double* results, int population_size, int dim, int tournament_size)
{
	int i, selected_index;

	for (i = 0; i < population_size; ++i)
	{
		selected_index = Tournament_Selection(results, population_size, tournament_size); // 使用锦标赛选择函数选取一个个体索引
		memcpy(selected_parent[i], population[selected_index], sizeof(double) * dim); // 复制选定的个体到 selected_parent 数组中
	}
}

void Crossover(double *parent1, double *parent2, int dim)
{
	int i;
	int rand_dim;

	boost::mt19937 generator(time(0)*rand());
	boost::uniform_int<> uniform_rand_dim(0, dim - 1);
	boost::variate_generator< boost::mt19937&, boost::uniform_int<> > random_dim(generator, uniform_rand_dim);//to generate a random dim within [0,dim-1]

	rand_dim = random_dim();

	for (i = 0; i < rand_dim; ++i)
	{
		std::swap(parent1[i], parent2[i]);
	}

}

void Mutation(double **population, double MIN, double MAX, double mutation_probability, int population_size, int dim)
{
	int i, j;
	boost::mt19937 generator(time(0)*rand());
	boost::uniform_real<> uniform_real_generate_r(0, 1);
	boost::variate_generator< boost::mt19937&, boost::uniform_real<> > random_real_num_r(generator, uniform_real_generate_r);//to generate a random number within [0,1]

	for (i = 0; i < dim; ++i)
	{
		boost::uniform_real<> uniform_real_generate_x(MIN, MAX);
		boost::variate_generator< boost::mt19937&, boost::uniform_real<> > random_real_num_x(generator, uniform_real_generate_x);

		for (j = 0; j < population_size; ++j)
		{
			if (random_real_num_r() <= mutation_probability)
			{
				population[j][i] = random_real_num_x();
			}
		}
	}
}





