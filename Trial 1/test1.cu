//General Include
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

//CUDA Include(s)
// includes, project
#include <cuda_runtime.h>
#include <cufft.h>
#include <helper\helper_functions.h>
#include <helper\helper_cuda.h>
//#include "exception.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
//#include <helper_functions.h>
//#include <helper_cuda.h>

typedef float2 cplx; //Complex Version CUDA - cuFFT
#define N 32 // Try Out
#define M_PI 3.14159265358979323846

//Private Function Prototype
void dft(cplx in[], cplx out[], int n);
__global__ void dft_CUDA(cplx *in, cplx *out, int n);
__device__ double phaseDFT(unsigned int idx, unsigned int i, int n);
__device__ void takeOut(cplx *in, cplx *out, int n);

int main(int argc, char **argv) // int argc, char **argv
{
	
	int signalSize = 0, p = 1;
    if (argc > 1)
    {
    	signalSize = atoi(argv[1]);
    }
    else 
    	signalSize = N;

    //To Fill Array for power of two number array
    while (signalSize>p)
    {
    	p *= 2;
    }

    if (signalSize < p) 
    	{
    		printf("Signal Size = Not Power of Two\n");
    		signalSize = p; //Add to meet the requirement of power of Two
    		printf("Add --SignalSize-- to meet the requirement of power of Two\n");
    		//return 0;
    	}

    printf("Number Input Array : %d\n", signalSize);
    
    cplx *x = (cplx *)malloc(sizeof(cplx)*signalSize);
    cplx *y_h = (cplx *)malloc(sizeof(cplx)*signalSize); 
    memset(y_h, 1, sizeof(cplx)*signalSize);
    cplx *y_d = (cplx *)malloc(sizeof(cplx)*signalSize);
    memset(y_d, 1, sizeof(cplx)*signalSize);

    for (int i = 0; i<(signalSize/2); i++)
    {
    	x[i].x = 1;
    	x[i+(signalSize/2)].x = 0;
    	x[i].y = 0;
    	x[i+(signalSize/2)].y = 0;
    }

    	printf("Before Forward FFT\n");
    for (int j = 0; j<signalSize; j++)
    {
        y_d[j].x = 0;
        y_d[j].y = 0;
    	printf("Starting values: X = %.2f %+.2fi\tY  = %.2f %+.2fi\n", x[j].x, x[j].y, y_h[j].x, y_h[j].y);
    }

    //Initializing Number Thread and Block
    int maxThreads = 32;
    int n_threads_per_blocks = maxThreads;
    int numBlock = signalSize / n_threads_per_blocks; //Initial Number Block
    //int pKali = 1;

    printf("Number Threads / Block = %d\n", n_threads_per_blocks);
    printf("Number Block = %d\n", numBlock);
    //Calculate using DFT - Non CUDA
    dft(x, y_h, signalSize); //Out-Place ; X Input, Y Output, n Number Array Input

    printf("After Forward DFT\n");
    for (int j = 0;j<signalSize;j++)
    {
    	printf("Result values: Initial Signals X  = %.2f %+.2fi \t Y from DFT = %.2f %+.2fi\n", x[j].x, x[j].y, y_h[j].x, y_h[j].y);	
    }

    //CUDA Malloc
    cplx *d_x;
    checkCudaErrors(cudaMalloc((void **)&d_x, sizeof(cplx)*signalSize));
    checkCudaErrors(cudaMemset(d_x, 1, sizeof(cplx)*signalSize));
    cplx *d_y;
    checkCudaErrors(cudaMalloc((void **)&d_y, sizeof(cplx)*signalSize*signalSize));
    checkCudaErrors(cudaMemset(d_y, 1, sizeof(cplx)*signalSize*signalSize));
    // cplx *d_temp;
    // checkCudaErrors(cudaMalloc((void **)&d_temp, sizeof(cplx)*signalSize*signalSize));
    // checkCudaErrors(cudaMemset(d_temp, 0, sizeof(cplx)*signalSize*signalSize));

    //CUDA Memcpy
    checkCudaErrors(cudaMemcpy(d_x, x, sizeof(cplx)*signalSize, cudaMemcpyHostToDevice));

    //Launch Kernel CUDA ----- In-Place ; X Input, Y Output, n Number Array Input
    dft_CUDA<<<numBlock, n_threads_per_blocks>>>(d_x, d_y, signalSize);
    //cudaDeviceSynchronize();
    // cudaThreadSynchronize();

    //CUDA Memcpy
    checkCudaErrors(cudaMemcpy(y_d, d_x, sizeof(cplx)*signalSize, cudaMemcpyDeviceToHost));

    
    printf("After Forward DFT - CUDA\n");

    printf("\tFROM DFT \t\t\t\t\t\t\t FROM DFT CUDA\n");

    cplx diff;
    for (int j = 0;j<signalSize;j++)
    {
        diff.x = y_h[j].x - y_d[j].x;
        diff.y = y_h[j].y - y_d[j].y;
    	printf("Result values: Initial Signals Y Host  = %.5f %+.5fi \t Y from DFT CUDA = %.5f %+.5fi >>> Difference = %.5f %+.5fi\n", y_h[j].x, y_h[j].y, y_d[j].x, y_d[j].y, diff.x, diff.y);	
    }

    //Free Memory - CPU & CUDA Arch.
    free(x);
    free(y_h);
    free(y_d);
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));

    getchar();
    return 0;
}

//Out-Place Algorithm and Arithmetic Complexity O(n^2)   
void dft(cplx in[], cplx out[], int n)
{

	double sumr = 0, sumi = 0;
	double phase = 0.0;
	for (int s =0; s<n; s++)  //Loop Output
	{
		sumr = 0.0;
		sumi = 0.0;

		for (int t = 0; t < n ; t++) //Loop for Operating Input with DFT Equation
		{
			phase = 2 * M_PI * t * s/n; //M_PI = PHI
			sumr += in[t].x * cos(phase)  + in[t].y * sin(phase);
			sumi += (-1) * in[t].x * sin(phase) + in[t].y * cos(phase);
		}
		out[s].x = sumr;
		out[s].y = sumi;
	}
}

__global__ void dft_CUDA( cplx *in,  cplx *out, int n)
{
    //unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ cplx temp[32];
    unsigned int idx = threadIdx.x;
    double sumr = 0.0,
            sumi = 0.0;
    //unsigned int incrementId = blockDim.x * gridDim.x;
    //double phase;
   // int toIncrement = 0;
    
    for (int i = 0; i < n ; i++)
    {
        out[idx+i*n].x = in[i].x *cos(phaseDFT(idx,i,n)) + in[i].y * sin(phaseDFT(idx, i, n));
        out[idx+i*n].y = (-1) * in[i].x * sin(phaseDFT(idx, i, n)) + in[i].y * cos(phaseDFT(idx, i, n));
        printf("out[%u+%d*%d = %d] = %.2f +%.2fi\n",idx, i, n, (idx+i*n), out[idx+i*n].x, out[idx+i*n].y);
        printf("Phase = %.2f", phaseDFT(idx,i,n));
        __syncthreads();//+ in[i].y * sin(2*M_PI*idx*i);

        // in[idx+n].x = out[idx*n+i].x;
        // in[idx+n].y = out[idx*n+i].y;
        // printf("out Result[%u*%d+%d = %d] = %.2f +%.2fi\n",idx, n, i, (i+idx*n), out[i+idx*n].x, out[i+idx*n].y);
      

    }

      for (int j = 0; j < n; j++)
            {
                temp[idx].x = out[idx*n+j].x;
                temp[idx].y = out[idx*n+j].y;
                printf("out Result[%u*%d+%d = %d] = %.2f +%.2fi\n",idx, n, j, (j+idx*n), out[j+idx*n].x, out[j+idx*n].y);
                __syncthreads();

                sumr = 0.0;
                sumi = 0.0;
                for (int k = 0; k < n; k++)
                {
                    sumr = sumr + temp[k].x;
                    sumi = sumi + temp[k].y;
                    printf("in[%d] Result = %.2f +%.2fi\n",j, temp[k].x, temp[k].y);
                }

                in[j].x = sumr;
                in[j].y = sumi;
            }


    //takeOut(out, in, n);

    // for (int k = 0; k < n ; i++)
    // {
    //     for (int j = 0; j < n; j++)
    //     {
    //         sumr = sumr + out[i+j*n].x;
    //         printf("SUmr = %.2f + out[%u + %d * %d].x = Total Idx = %d \n", sumr, i, j, n, j*n+i);
    //         sumi = sumi + out[i+j*n].y;
    //         printf("out Result[%u+%d*%d = %d] = %.2f +%.2fi\n",i, j, n, (i+j*n), out[i+j*n].x, out[i+j*n].y);

           
    //         printf("in[%d] Result[%u+%d*%d = %d] = %.2f +%.2fi\n",i, i, j, n, (i+j*n), in[i+j*n].x, in[i+j*n].y);
    //     }
    //     in[i].x = sumr;
    //     in[i].y = sumi;
    // }

    // for (int i = 0; i < n; i++)
    // {   
    //     in[idx].x = in[idx].x + out[i+idx*n].x;
    //     printf("SUmr = %.2f + out[%u + %d * %d].x = Total Idx = %d \n", sumr, idx, i, n, idx*n+i);
    //     in[idx].y = in[idx].y + out[i+idx*n].y;
    //     printf("out Result[%u+%d*%d = %d] = %.2f +%.2fi\n",idx, i, n, (i+idx*n), out[i+idx*n].x, out[i+idx*n].y);

    //     // in[i].x = sumr;
    //     // in[i].y = sumi;
    //     printf("in[%d] Result[%u+%d*%d = %d] = %.2f +%.2fi\n",i, idx, i, n, (i+idx*n), in[i+idx*n].x, in[i+idx*n].y);
    // }
   // shiftIn(out,n,(-1*toIncrement));
}

__device__ double phaseDFT(unsigned int idx, unsigned int i, int n)
{
    double out;
    out = 2 * M_PI * idx * i / n;
    return out;
}

__device__ void takeOut(cplx *in, cplx *out, int n)
{
    double sumr = 0.0, sumi = 0.0;
    for (int i = 0; i < n ; i++)
    {
        for (int j = 0; j < n; j++)
        {
            sumr = sumr + in[i+j*n].x;
            printf("SUmr = %.2f + out[%u + %d * %d].x = Total Idx = %d \n", sumr, i, j, n, j*n+i);
            sumi = sumi + in[i+j*n].y;
            printf("out Result[%u+%d*%d = %d] = %.2f +%.2fi\n",i, j, n, (i+j*n), out[i+j*n].x, out[i+j*n].y);

            printf("in[%d] Result[%u+%d*%d = %d] = %.2f +%.2fi\n",i, i, j, n, (i+j*n), in[i+j*n].x, in[i+j*n].y);
        }
        out[i].x = sumr;
        out[i].y = sumi;
    }
}