#ifndef SELF_DEFINE_FUNCTIONS_H_INCLUDED
#define SELF_DEFINE_FUNCTIONS_H_INCLUDED
#include <cstdio>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <iomanip>
#include <math.h>
#include <string.h>
// The following is the library of random number generators in boost
#include <boost/random.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/cauchy_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real.hpp>
//the settings of the program parameters
const int timesOfRun = 10;//the number of independent runs
const int dim = 30;//the dimension size of the problem to be optimized
const int MAX_FV = 10000 * dim;//the maximum number of fitness evaluationsi
//the settings of parameters in GA
const int Population_Size = 100;//Population Size
const double mutation_probability = 0.02;//Mutation probability
const double crossover_probability = 0.9;//Crossover probability
void Selection_Probability(double *results, double *probability, int population_size);
int Roulette_Selection(double *probability, int population_size);
int Tournament_Selection(double* fitness_values, int population_size, int tournament_size);
void Selection2(double** selected_parent, double** population, double* probability, int population_size, int dim, int tournament_size);
void Selection(double **selected_parent, double **population, double *probability, int population_size, int dim);
void Crossover(double *parent1, double *parent2, int dim);
void Mutation(double **population, double MIN, double MAX, double mutation_probability, int population_size, int dim);
void cec14_test_func(double *x, double *f, int nx, int mx, int func_num);
#endif // SELF_DEFINE_FUNCTIONS_H_INCLUDED
