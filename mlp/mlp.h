

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <GL/glut.h>

#define CLASS_SIZE 26

int hidden_layers,hidden,inputs,neurons,outputs;

int inputs_neuron,max_epochs;

double learning_rate = 0.001;
double eta = 0.01;
int num_epochs;
double loss;
 
double testResults[4];

int stopVal;

typedef double (*mlp_actfun)(double a);

typedef double* (*mlp_actout)(double const *a,int x);


typedef struct letter
{
    /* How many inputs, outputs, and hidden neurons. */
    int inputs, hidden_layers, hidden, outputs;
    
    /* All errors (total_weights long). */
    double error[42];

    /* Which activation function to use for hidden neurons. Default: mlp_act_sigmoid*/
    mlp_actfun activation_hidden;

    /* Which activation function to use for output. Default: mlp_act_sigmoid*/
    mlp_actfun activation_output;

    /* Total number of weights, and size of weights buffer. */
    int total_weights;

    /* Total number of neurons + inputs and size of output buffer. */
    int total_neurons;

    /* All weights (total_weights long). */
    double *weight;
    
    

    /* Stores input array and output of each neuron (total_neurons long). */
    double *output;

    /* Stores delta of each hidden and output neuron (total_neurons - inputs long). */
    double *delta;

}mlp;
