


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


double **input, **class;

int samples;

const char *class_names[CLASS_SIZE] = {"A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"}; 



void mlp_randomize(mlp *t) 
{
    int i;
    double r;
    double min = -2.0;
    double max = 2.0;
    double range = (max - min); 
    double div = RAND_MAX / range;
  //  srand(time(NULL));
    for (i = 0; i < t->total_weights; ++i)
    {           
      r = min + (rand() / div);
      t->weight[i] = r;       
    }
}

double mlp_act_sigmoid(double a) 
{
  return (1/(1+exp(-a)));   
}

double mlp_sigmoid_derivative(double a) 
{
  return (mlp_act_sigmoid(a) * (1-mlp_act_sigmoid(a)));   
}


double mlp_act_threshold(double a) 
{
    return a > 0;
}


double mlp_act_linear(double a) 
{
    return a;
}



double cross_entropy_loss(double const a, double const y)
{
   return -(a*log(y)); //For multi class 
}




double *mlp_act_softmax(double *input, int input_len) 
{
  assert(input);
  double *x = input;
  
  double m = -INFINITY;
  for (int i = 0; i < input_len; i++)
  {
    if (x[i] > m)
     {
      m = x[i];
    }
  }
 
  double sum = 0.0;
  for (int i = 0; i < input_len; i++)
   {
    sum += exp(x[i]-m);
  }

  const double offset = m + log(sum);
  for (int i = 0; i < input_len; i++) 
  {
    x[i] = exp(x[i] - offset);
  }
  return x;
}





mlp* init_mlp(char **arglist,int n)
{
    mlp *ret;
    int i,j,k,p;
    char *split = NULL;
    double out;
    FILE *in = fopen(arglist[1], "r");
    if (!in)
    {
        printf("Could not open training data file: %s\n", arglist[1]);
        exit(1);
    }

    /* Loop through the data to get a count. */
    char line[1024];
    samples=0;
    while (!feof(in) && fgets(line, 1024, in)) 
    {
        ++samples;
    }
   // printf("train data csv read and number of samples is %d\n",samples);
  
    
    inputs_neuron=16;
    hidden_layers=1;
    inputs=n;
    hidden=n;
    outputs=CLASS_SIZE;
    
   
    
    const int hidden_weights = hidden_layers ? (inputs+1) * hidden + (hidden_layers-1) * (hidden+1) * hidden : 0;
    const int output_weights = (hidden_layers ? (hidden+1) : (inputs+1)) * outputs;
    const int total_weights = (hidden_weights + output_weights);

    const int total_neurons = (inputs + hidden * hidden_layers + outputs);
    
    /* Allocate extra size for weights, outputs, and deltas. */
    const int size = sizeof(mlp) + sizeof(double) * (total_weights + total_neurons + (total_neurons - inputs));
    ret = malloc(size);
    if (!ret) return 0;
    
    //set neurons data for input,hiddent and putput layers
    ret->inputs = inputs;
    ret->hidden_layers = hidden_layers;
    ret->hidden = hidden;
    ret->outputs = outputs;
    
   // printf("Number of neurons in input,hidden and output layers is %d %d and %d\n",inputs,hidden,outputs);

    ret->total_weights = total_weights;
    ret->total_neurons = total_neurons;
    
    for(int k=0;k<total_neurons;k++)
     ret->error[k]=0.0;

    /* Set pointers. */
    ret->weight = (double*)((char*)ret + sizeof(mlp));
    ret->output = ret->weight + ret->total_weights;
    ret->delta = ret->output + ret->total_neurons;
   

    mlp_randomize(ret);

    ret->activation_hidden = mlp_act_sigmoid;
    ret->activation_output = mlp_act_sigmoid;
    
     //Load data into input array
    input = (double **)malloc(samples * sizeof(double *));
    class = (double **)malloc(samples * sizeof(double *));
    
    fseek(in, 0, SEEK_SET);
    if(input!=NULL)
    {
     
      for (i = 0,k=0; i < samples; ++i) 
      {
        input[i] = (double *)malloc(16 * sizeof(double));
        class[i] = (double *)malloc(26 * sizeof(double));
                
        if (fgets(line, 1024, in) == NULL) 
        {
            perror("fgets");
            exit(1);
        }
      //  printf("Line read %s\n",line);
        for(p=0;p<26;p++)
        class[i][p]=0.0;
        
        
        split = strtok(line, ",");
        out = atof(split);
        for (j = 0; j < 16; ++j) 
        {
            split = strtok(0, ",");          
            input[i][j] = atof(split);
        }
        if(1<= (int)out <=26)
        {
          int x = (int)out;
          class[i][x]=1.0;
        //  printf("character output is %s\n",class_names[x]);          
        }
        else
          printf("class not found/valid\n");
      }
    }
    
    fclose(in);
    return ret;   
   
}


double const *mlp_run(mlp const *t, double const *inputs)
 {
    double const *w = t->weight;
    double *o = t->output + t->inputs;
    double const *i = t->output;
    double softResult[CLASS_SIZE];

    /* Copy the inputs to the scratch area, where we also store each neuron's output, for consistency. This way the first layer isn't a special case. */
    memcpy(t->output, inputs, sizeof(double) * t->inputs);

    int h, j, k;

    const mlp_actfun act = t->activation_hidden;
    const mlp_actfun acto = t->activation_output;

    
    /* Figure hidden layers, if any. */
    for (h = 0; h < t->hidden_layers; ++h)
    {
        for (j = 0; j < t->hidden; ++j)
         {
            double sum = *w++ * -1.0;
            for (k = 0; k < (h == 0 ? t->inputs : t->hidden); ++k)
            {
                sum += *w++ * i[k];
            }
            *o++ = act(sum);
        }
        i += (h == 0 ? t->inputs : t->hidden);
    }

    double const *ret = o;
   
     /* Figure output layer. */  
   
       
      for (j = 0; j < t->outputs; j++) 
      {
         double sum = *w++ * -1.0;      
         for (k = 0; k < (t->hidden_layers ? t->hidden : t->inputs); ++k)
         {
            sum += *w++ * i[k];           
          
         }
        // *o++ = act(sum);   
        softResult[j]=sum;                
       }
       
       //Now apply softmax;
       double *res = mlp_act_softmax(softResult,CLASS_SIZE);
       for(j=0;j<t->outputs;++j)
       {
           //printf("output for j is %.2f\n",*res);
           *o++ = *res++;           
       }
    
    /* Sanity check that we used all weights and wrote all outputs. */
    assert(w - t->weight == t->total_weights);
    assert(o - t->output == t->total_neurons);
    
    return ret;
}




double mlp_train(mlp *m, double **inputs, double **desired_outputs,int lossfunction,int stop) 
{
    /* To begin with, we must run the network forward. */
    double const *ret;
    int s;
    int flag=1;
   
    num_epochs=0;
    
    double euclidieandelta;  
    int incorrectSets,incorrectResult;
    double trainingSetAccuracy;
    double totalSamples;
     
    loss = 0.0;  
    totalSamples=0.0; 
    incorrectSets = 0;  
    while(flag)
    {  
       loss=0.0;      
       trainingSetAccuracy=0.0;
       totalSamples +=samples;  
        if(stop==1)
        { incorrectSets=0;  }
       for(s=0;s<samples;s++)
       {
           int i,j,k,h,x;
           
           euclidieandelta = 0.0;              
           incorrectResult = 0;
                              
           mlp_run(m, inputs[s]);         
        
           /* First set the output layer Errors and deltas. */
           {   
              
              double  *o = m->output + m->inputs + m->hidden * m->hidden_layers; /* First output. */
             
              double *d = m->delta + m->hidden * m->hidden_layers; /* First delta. */
              double const *t = desired_outputs[0]; /* First desired output. */
              int eindex =  m->inputs + m->hidden * m->hidden_layers;
        
             /* Set output layer deltas. */
       
             if (lossfunction == 1)  //Use Sum of squares deviation method to calculate error
              {               
                for (j = 0; j < m->outputs; ++j) 
                {
                 
                  *d++ = (*t - *o) * *o * (1.0 - *o);  
                  m->error[eindex] = 0.5 * pow((*t - *o),2);
                  loss += m->error[eindex++];
                  if((*t - *o)!=0)
                   incorrectResult++;
                   ++o; ++t; 
                }       
             }
             else
             {                 //Use cross entropy error function
              for (j = 0; j < m->outputs; ++j) 
              {
                  m->error[eindex] = cross_entropy_loss(*t,*o);                
                  loss += m->error[eindex++];  //Cumulative Error in global variable
                  *d++ = (*t - *o); //with cross entropy*/                
                  if((*t - *o)!=0)
                   incorrectResult++;
                  ++o; ++t;
               }         
             }                              
           }  
           if(incorrectResult > 0)
           incorrectSets++;
           
          // printf("value of incorrectsets is %d in epoch %d\n",incorrectSets,num_epochs);
            /* Train the outputs. adjust weights */
          {  
              /* Find first output delta. */
             double const *d = m->delta + m->hidden * m->hidden_layers; /* First output delta. */
        
             // int eindex =  m->inputs + m->hidden * m->hidden_layers;
                 
             /* Find first weight to first output delta. */
             double *w = m->weight + (m->hidden_layers
                ? ((m->inputs+1) * m->hidden + (m->hidden+1) * m->hidden * (m->hidden_layers-1))
                : (0));

             /* Find first output in previous layer. */
             double const * const i = m->output + (m->hidden_layers
                ? (m->inputs + (m->hidden) * (m->hidden_layers-1))
                : 0);

              
               double deltaWt=0.0;
               for (j = 0; j < m->outputs; ++j)
               {
                    for (k = 0; k < (m->hidden_layers ? m->hidden : m->inputs) + 1; ++k) 
                    {
                       if(k==0)
                        {
                             deltaWt = *d * learning_rate * (-1.0);
                             *w++ = (*w)+deltaWt;
                         }
                       else
                       {
                            //printf("In for loop k and *w are %d - %e\n",k,*w);
                          deltaWt = *d * learning_rate * i[k-1];
                          *w++ = (*w)+deltaWt;
                          //printf("In outer for loop, updated weight and delta are %e - %e with  k  and j %d -%d\n",*w, deltaWt,k,j); 
                        }
                     }
                    ++d;  
                 }         
                 assert(w - m->weight == m->total_weights);
             }
             /* Set hidden layer deltas, start on last layer and work backwards. */
             /* Note that loop is skipped in the case of hidden_layers == 0. */
            
            for (h = m->hidden_layers - 1; h >= 0; --h)
            {

                  /* Find first output and delta in this layer. */
               double const *o = m->output + m->inputs + (h * m->hidden);
               double *d = m->delta + (h * m->hidden);
                 /* Find first delta in following layer (which may be hidden or output). */
               double const * const dd = m->delta + ((h+1) * m->hidden);
               /* Find first weight in following layer (which may be hidden or output). */
               double const * const ww = m->weight + ((m->inputs+1) * m->hidden) + ((m->hidden+1) * m->hidden * (h));
         
               for (j = 0; j < m->hidden; ++j)
               {
                    double delta = 0.0;
                    for (k = 0; k < (h == m->hidden_layers-1 ? m->outputs : m->hidden); ++k)
                    {
                        const double forward_delta = dd[k];
                        const int windex = k * (m->hidden + 1) + (j + 1);
                        const double forward_weight = ww[windex];
                        delta += forward_delta * forward_weight;                
                     }
                    *d = *o * (1.0-*o) * delta;
                    euclidieandelta += (delta) * (delta);
                    ++d; ++o;
               } 
             }  
             euclidieandelta = sqrt(euclidieandelta);
             loss += sqrt(loss);
           
             if(stop==1)  //Use stopping criteria as  ||ΔW|| < ε
             { 
                if(euclidieandelta<eta)
                {
                // printf("Met stopping criteria in %d epoch for sample %d \n",num_epochs,s);
                 flag=0;  
                 break;          
                }               
              }
              else
              {                 
                 if(loss<1.0)
                     break;
              }      
             
             /* Train the hidden layers & adjust weights */
              for (h = m->hidden_layers - 1; h >= 0; --h)
               {

                /* Find first delta in this layer. */
                double const *d = m->delta + (h * m->hidden);
        
               //int eindex =  m->hidden;
         
               /* Find first input to this layer. */
                double const *i = m->output + (h ? (m->inputs + m->hidden * (h-1)) : 0);

                /* Find first weight to this layer. */
                double *w = m->weight + (h ? ((m->inputs+1) * m->hidden + (m->hidden+1) * (m->hidden) * (h-1)) : 0);

                for (j = 0; j < m->hidden; ++j) 
                {
                     for (k = 0; k < (h == 0 ? m->inputs : m->hidden) + 1; ++k) 
                     {
                        if (k == 0)
                        {
                           *w++ += (*d) * learning_rate * -1.0;
                        } 
                        else 
                        {
                           *w++ += (*d) * learning_rate * i[k-1];
                        }
                     }
                   ++d;
                }
              }                                     
        }          
        num_epochs++;
        //printf("stop value and num_epochs is %d %d\n",stop,num_epochs);
        if(!flag)
        {
           //("For stop %d, incorrectSets and samples are %d - %d\n",stop,incorrectSets,samples);
           trainingSetAccuracy = 100.0-((double)incorrectSets/samples * 100.0);
           break;          
        }
        if(stop==2)
        {
           if(num_epochs==100)
           {
           // printf("Number of incorrect sets in epoch is %d - %d and sample size is %d\n",incorrectSets,num_epochs,samples);
            trainingSetAccuracy = 100.0-((double) incorrectSets/totalSamples * 100.0);
            break; 
            }                        
         }           
     }  
     return trainingSetAccuracy;
}


 double func(double x)
 {
    return x;
 }        
 void draw(double x1, double x2, double y1, double y2, int N)
  {
        	double x, dx = 1;
        	int i;
         
        	glPushMatrix(); /* GL_MODELVIEW is default */         
        	
        	if(stopVal==1)
        	glScalef(1.0/ (x2 - x1) , 1.0 / 100.0, 1.0);
        	else
        	glScalef(1.0/ (x2 - x1) , 1.0 / 500.0, 1.0);
        
        	glTranslatef(-x1, -y1, 0.0);
        	glColor3f(1.0, 1.0, 1.0);
         
        	glBegin(GL_LINE_STRIP);
         
        	for(x = x1, i=0; x < x2 && i<4; x += dx,i++)
        	{
        		glVertex2f(x, testResults[i]);
        	}
         
        	glEnd();
         
        	glPopMatrix();
 }
         
 /* Redrawing func */
 void redraw()
 {
      double m,n,o,p;
      glClearColor(0, 0, 0, 0);
      glClear(GL_COLOR_BUFFER_BIT);
        	glMatrixMode(GL_MODELVIEW);
        	glLoadIdentity();
             m=testResults[0];
             n=testResults[1];
             o=testResults[2];
             p=testResults[3];
             
        	draw(5,8,0,100,1);      
        	glutSwapBuffers();
 }
         
        /* Idle proc. Redisplays, if called. */
 void idle(void)
  {
        	glutPostRedisplay();
  }
         
        /* Key press processing */
  void key(unsigned char c, int x, int y)
   {
      	if(c == 27) exit(0);
   }
         
        /* Window reashape */
        void reshape(int w, int h)
        {
        	glViewport(0, 0, w, h);
        	glMatrixMode(GL_PROJECTION);
        	glLoadIdentity();
        	glOrtho(0, 1, 0, 1, -1, 1);
        	glMatrixMode(GL_MODELVIEW);
        }
         
     void drawGraph(int x,char **y)
     {
           glutInit(&x,y);
        	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
        	glutCreateWindow("MLP train ");
         
        	/* Register GLUT callbacks. */
        	glutDisplayFunc(redraw);
        	glutKeyboardFunc(key);
        	glutReshapeFunc(reshape);
        	glutIdleFunc(idle);
         
        	/* Init the GL state */
        	glLineWidth(1.0);         
        	/* Main loop */
        	glutMainLoop();        
     }
     





int main(int argc,char *argv[])
{
    
    mlp *m;
    int i,j,x,y;
   
    
    if(argc!=4)
    {
      printf("Usage: <mlp exe> <train data file> <loss function> <stopping criteria>\n");
      exit(1);
    }
    
    y = atoi(argv[2]);
    if(y==1 || y==2)
     {
         if(y==1)
         printf("Using sum of squares deviation loss function as per choice\n");
         else
         printf("Using cross entropy loss function as per choice\n");
         x = atoi(argv[3]);
         stopVal=x;
         if(x==1 || x==2)
         {
            if(x==1)
            printf("Using stopping criteria as ||ΔW|| < ε \n");
            else
            printf("Using stopping criteria of epoch size 100 as per choice given\n");
         }
         else
         {
            printf("Enter 1 to use stopping criteria as  ||ΔW|| < ε \n");
            printf("Enter 2 to use stopping criteria of epoch size 100\n");
            exit(1);
         }
     }
     else
     {
       printf("Enter 1 to use sum of squares deviation loss function\n");
       printf("Enter 2 to use cross entropy loss function\n");
       exit(1);
     }  
      for(i=0,j=5;j<9;i++,j++)
    {
    
      printf("Train MLP with %d neuron inputs\n",j);
      
      m=init_mlp(argv,j);
      
      testResults[i] = mlp_train(m, input, class,y,x);
          if(m)
          free(m);
    }
    
    //Used in for loop to print and test data inputs
    /*
    printf("Number of inputs : %d \n",m->inputs);
    printf("Number of hidden layers : %d \n",m->hidden_layers);
    printf("Number of neurons in hidden layer : %d\n",m->hidden);
    printf("Number of neurons in ouput layer: %d\n",m->outputs);
    printf("Total number of neurons : %d \n",m->total_neurons);
    
    */
     /* Train the network with backpropagation. */      
    printf("Completed training \n");
    for(i=5,j=0;i<9 && j<4;i++,j++)
    {
       printf(" MLP with %d input neurons has classification accuracy %.2f\n",i,testResults[j]);
    } 
   
    drawGraph(argc,argv); 
}
