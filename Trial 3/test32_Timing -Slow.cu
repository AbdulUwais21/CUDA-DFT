//General Include
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <sys/time.h>

//CUDA Include(s)
// includes, project
#include <cuda_runtime.h>
#include <cufft.h>
#include <helper/helper_functions.h>
#include <helper/helper_cuda.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

typedef float2 cplx; //Complex Version CUDA - cuFFT
#define N 16 // Try Out
#define M_PI 3.14159265358979323846
#define NStreams 16

//Private Function Prototype
void dft(cplx in[], cplx out[], int n);
void fft_it(cplx In[], int u);
unsigned int bitrev( unsigned int n,  unsigned int bits) ;
// __global__ void dft_CUDA(cplx *in, cplx *out, int n);
__global__ void dft_CUDA( cplx *in,  cplx *out, int n, int n_block_in_stream, int n_stream, int startIdx, int outIdx);
__global__ void sumDFT_CUDA(cplx *in, cplx *out, int n_stream);
__device__ double phaseDFT(unsigned int idx, unsigned int i, int n);

int main(int argc, char **argv) // int argc, char **argv
{
	//Initialising
    cudaError_t result;
    size_t freeMemGet1, totalMemGet1,
            freeMemGet2, totalMemGet2;
	int signalSize = 0, p = 1;
    clock_t startmal, stopmal,
            startDFTCPU, stopDFTCPU,
            startFFTCPU, stopFFTCPU;
    float cpuTimer1, cpuTimer2, cpuTimer3;

    //For GPU


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

    size_t nBytes = signalSize * sizeof(cplx);
    
    startmal = clock();
    ////////////////////////////////////////////////////////
    cplx *x = (cplx *)malloc(sizeof(cplx)*signalSize);
    cplx *y_h = (cplx *)malloc(sizeof(cplx)*signalSize); 
    memset(y_h, 0, sizeof(cplx)*signalSize);
    cplx *y_d = (cplx *)malloc(sizeof(cplx)*signalSize);
    memset(y_d, 0, sizeof(cplx)*signalSize);

    // cplx *x;  
    // checkCudaErrors(cudaHostAlloc((void **)&x, nBytes, cudaHostAllocDefault));
    // cplx *y_h;
    // checkCudaErrors(cudaHostAlloc((void **)&y_h, nBytes, cudaHostAllocDefault));
    // memset(y_h, 0, sizeof(cplx)*signalSize);
    // cplx *y_d;
    // checkCudaErrors(cudaHostAlloc((void **)&y_d, nBytes, cudaHostAllocDefault));
    // memset(y_d, 0, sizeof(cplx)*signalSize);

    for (int i = 0; i<(signalSize/2); i++)
    {
    	x[i].x = 1;
    	x[i+(signalSize/2)].x = 0;
    	x[i].y = 0;
    	x[i+(signalSize/2)].y = 0;
    }

    ////////////////////////////////////////////////////////
    stopmal = clock();
    cpuTimer1 = (float)(stopmal - startmal) / (CLOCKS_PER_SEC);

    	printf("Before Forward FFT\n");
    // for (int j = 0; j<signalSize; j++)
    // {
    //     // y_d[j].x = 0;
    //     // y_d[j].y = 0;
    // 	printf("Starting values: X = %.2f %+.2fi\tY  = %.2f %+.2fi\n", x[j].x, x[j].y, y_h[j].x, y_h[j].y);
    // }

    //Initializing Number Thread and Block
    int WarpSize = 32;
    int n_threads_per_blocks = WarpSize;
    //Initial Number Block

    // int numBlock = signalSize / n_threads_per_blocks;
    if (signalSize <= WarpSize)
    {
        n_threads_per_blocks = signalSize;
        // numBlock = 1;
    }

    dim3 block (n_threads_per_blocks);
    dim3 grid ((signalSize + block.x - 1)/ block.x);
    printf("<<< Grid (%d, %d), Block (%d, %d)>>>\n", grid.x, grid.y, block.x, block.y);
    // int n_threads_per_blocks2 = signalSize / n_threads_per_blocks;
    //int pKali = 1;

    /////For Testing 2D Block 1D Grid
    // dim3 block (n_threads_per_blocks, n_threads_per_blocks2);

    // printf("Number Threads / Block = %d\n", n_threads_per_blocks);
    // printf("Number Block = %d\n", numBlock);

    
    /////////////////////////////////////////////////////////////////////////////
    startDFTCPU = clock();
    //Calculate using DFT - Non CUDA
    dft(x, y_h, signalSize); //Out-Place ; X Input, Y Output, n Number Array Input
    stopDFTCPU = clock();
    cpuTimer2 = (float)(stopDFTCPU - startDFTCPU) / (CLOCKS_PER_SEC);
    /////////////////////////////////////////////////////////////////////////////

    // printf("After Forward DFT\n");
    // for (int j = 0;j<signalSize;j++)
    // {
    // 	printf("Result values: Initial Signals X  = %.2f %+.2fi \t Y from DFT = %.2f %+.2fi\n", x[j].x, x[j].y, y_h[j].x, y_h[j].y);	
    // }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////CUDA///////////////////////////////////////////////////////////
    //Cuda Event
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // int NStreams = 0;
    // if (signalSize < NumStreams)
    // {
    //     NStreams = signalSize;
    // }
    // else
    // {
    //     NStreams = NumStreams;
    // }
    //Determine Number Grid Parallel Operation
    int numElStreams = signalSize / NStreams;
    // size_t elStreamByte = numElStreams * sizeof(cplx);
    // grid.x = (numElStreams + block.x -1)/block.x;

    //CUDA Stream Initializing
    cudaStream_t streams[NStreams];

    for (int i = 0; i < NStreams; i++)
    {
        checkCudaErrors(cudaStreamCreate(&streams[i]));
    }
    //CUDA Malloc
    cplx *d_x;
    checkCudaErrors(cudaMalloc((void **)&d_x, nBytes));
    cplx *d_y;
    checkCudaErrors(cudaMalloc((void **)&d_y, nBytes*NStreams));

    //CUDA Memcpy
    checkCudaErrors(cudaEventRecord(start, 0));
    checkCudaErrors(cudaMemcpy(d_x, x, nBytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    float cuda_memcpyH2D_time;
    checkCudaErrors(cudaEventElapsedTime(&cuda_memcpyH2D_time, start, stop));
    
    //Launch Kernel CUDA ----- Out-Place ; X Input, Y Output, n Number Array Input
    checkCudaErrors(cudaEventRecord(start, 0));
    int j =0;
    for (int i = 0; i < NStreams ; i++)
    {
        int startIdx = i * numElStreams;
        dft_CUDA<<<grid, block, 0, streams[i]>>>(d_x, d_y, signalSize, numElStreams, NStreams, startIdx, j);    
        j++;
    }

    sumDFT_CUDA<<<grid, block>>>(d_y, d_x, NStreams);
    // dft_CUDA<<<grid, block>>>(d_x, d_y, signalSize);    
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    float kernel_time;
    checkCudaErrors(cudaEventElapsedTime(&kernel_time, start, stop));

    // cudaThreadSynchronize();
    // checkCudaErrors(cudaDeviceSynchronize());

    result = cudaMemGetInfo(&freeMemGet1, &totalMemGet1);
    if (result == cudaSuccess)
    {
    printf("\nAfter Kernel Execution :Available Memory : %d MB, Total Memory : %d MB\n", freeMemGet1/(1024*1024), totalMemGet1/(1024*1024));
    }

    //CUDA Memcpy
    checkCudaErrors(cudaEventRecord(start, 0));
    checkCudaErrors(cudaMemcpy(y_d, d_x, nBytes, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    float cuda_memcpyD2H_time;
    checkCudaErrors(cudaEventElapsedTime(&cuda_memcpyD2H_time, start, stop));

    float GPU_total_time = cuda_memcpyH2D_time + kernel_time + cuda_memcpyD2H_time;

    /////////////////////////////////////////////////////////////////////////////////////////////////////
    //Arithmetic Complexity 0(n log2 n)
    //In-Place Algorithm
    startFFTCPU = clock();
    fft_it(x, signalSize);
    stopFFTCPU = clock();
    cpuTimer3 = (float)(stopFFTCPU - startFFTCPU) / CLOCKS_PER_SEC;
    /////////////////////////////////////////////////////////////////////////////////////////////////////


    printf("After Forward DFT - CUDA\n");

    printf("\tFROM DFT CPU \t\t\t\t\t\t\t FROM DFT CUDA\n");

    
    /////////////////////Result Check//////////////////////////////////
    if ( signalSize < 512)
    {
        cplx diff;
        for (int j = 0;j<signalSize;j++)
        {
            diff.x = y_h[j].x - y_d[j].x;
            diff.y = y_h[j].y - y_d[j].y;
        	printf("Result values: Signals FFT CPU = %.5f %+.5fi \t Signals Y from DFT CPU Host  = %.5f %+.5fi \t Y from DFT CUDA = %.5f %+.5fi \t\t>>> Difference = %.5f %+.5fi\n", x[j].x, x[j].y, y_h[j].x, y_h[j].y, y_d[j].x, y_d[j].y, diff.x, diff.y);	
        }
    }
    //////////////////CALCULATING - Check Memory GPU////////////////////////////////
    result = cudaMemGetInfo(&freeMemGet2, &totalMemGet2);
    if (result == cudaSuccess)
    {
    printf("\nAvailable Memory : %d MB, Total Memory : %d MB\n", freeMemGet2/(1024*1024), totalMemGet2/(1024*1024));
    }

    /////////////////CPU TIMER////////////////////////
    printf("\nCPU Malloc, Memset, and Initializing = %f ms\n", cpuTimer1*1000);
    printf("CPU DFT Function = %f ms\n", cpuTimer2*1000);
    printf("CPU FFT Iterative Function = %f ms\n", cpuTimer3*1000);

     printf("\n");
    printf("Measured Timing and Throughput :\n");
    printf("Memcpy H2D : \t %f ms (%f GB/s)\n", cuda_memcpyH2D_time, nBytes/cuda_memcpyH2D_time);
    printf("Kernel  : \t %f ms (%f GB/s)\n", kernel_time, nBytes/kernel_time);
    printf("Memcpy D2H : \t %f ms (%f GB/s)\n", cuda_memcpyD2H_time, nBytes/cuda_memcpyH2D_time);
    printf("GPU Total Time: \t %f ms (%f GB/s)\n", GPU_total_time, nBytes/GPU_total_time);


    //Free Memory - CPU & CUDA Arch.
    free(x);
    free(y_h);
    free(y_d);
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));

    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    for (int i = 0; i < NStreams; i++)
    {
        checkCudaErrors(cudaStreamDestroy(streams[i]));
    }
    checkCudaErrors(cudaDeviceReset());

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

//Arithmetic Complexity 0(n log2 n)
//In-Place Algorithm
void fft_it(cplx In[], int u)
{
     int n= 1,h_idx= 1, j= 1, k= 1,s= 1, jumlah_bits = 0;
     double *costri = (double *) malloc(sizeof(double)*u/2);
     double *sintri = (double *) malloc(sizeof(double)*u/2);
     cplx *out = (cplx *)malloc(sizeof(cplx)*u);

    n = u;    // Jumlah Data
    int in_bit= n- 1; // Jika data dihitung dari 0
    while (in_bit>=h_idx)
    {
        if (n < 3)
        {
            jumlah_bits = 1;
        }
        else
        {
            h_idx *= 2;
            jumlah_bits += 1;
        }
    }


    for (int i = 0; i<n; i++)
    {
        out[i] = In[i];
    }
    for (int f=0;f<n;f++)
    {
        unsigned int bitr = bitrev(f, jumlah_bits);
        In[f] = out[bitr];
    }

    for (int  m = 0; m < n/2 ; m++)
    {
        costri[m] = cos(2 * M_PI * m / n);
        sintri[m] = sin(2 * M_PI * m / n);
    }

    for (s=2; s<=n ; s*=2)
    {
        int hs = s / 2;
        int ts = n / s;
        
        for (int loop1 = 0; loop1 < n; loop1+=s)
        {
            j = 0;
            k = 0;
            for (j = loop1, k = 0; j < loop1+hs; j++, k += ts)
            {
                double temptre = In[j+hs].x * costri[k] + In[j+hs].y * sintri[k];
                double temptim = (-1) * In[j+hs].x * sintri[k] + In[j+hs].y * costri[k];
                In[j+hs].x = In[j].x - temptre;
                In[j+hs].y = In[j].y - temptim;
                In[j].x += temptre;
                In[j].y += temptim;
            }
        }
    }
}

 unsigned int bitrev( unsigned int n,  unsigned int bits) 
{                           // n = 
     int i, nrev, M;   // nrev will store the bit-reversed pattern
    M = 1<<bits;            // find N: shift left 1 by the number of bits
    nrev = n;   
    for(i=1; i<bits; i++)
    {
        n >>= 1;
        nrev <<= 1;
        nrev |= n & 1;   // give LSB of n to nrev
    }
    nrev &= M-1;         // clear all bits more significant than N-1
    
    return nrev;
}

// __global__ void dft_CUDA( cplx *in,  cplx *out, int n)
// {
//     unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

//     if (idx < n)
//     {
//         for (int i = 0; i < n ; i++ )
//         {
//             out[idx+n].x = (float)((in[i].x *cos(phaseDFT(idx,i,n)) + in[i].y * sin(phaseDFT(idx, i, n))));
//             out[idx+n].y = (float)(((-1) * in[i].x * sin(phaseDFT(idx, i, n)) + in[i].y * cos(phaseDFT(idx, i, n))));
//             out[idx].x = out[idx].x + out[idx+n].x;
//             out[idx].y = out[idx].y + out[idx+n].y;
//         }    
//     }
// }

__global__ void dft_CUDA( cplx *in,  cplx *out, int n, int n_block_in_stream, int n_stream, int startIdx, int outIdx)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int bdimx = blockDim.x;
    float sumr = 0.0, sumi = 0.0;
    // if (idx < n)
    // {
        for (int i = startIdx; i < startIdx+n_block_in_stream ; i++ )
        {
            sumr += (float)((in[i].x *cos(phaseDFT(idx,i,n)) + in[i].y * sin(phaseDFT(idx, i, n))));
            sumi += (float)(((-1) * in[i].x * sin(phaseDFT(idx, i, n)) + in[i].y * cos(phaseDFT(idx, i, n))));
            printf("Idx = %d\n", idx);
        } 
            out[idx+outIdx*bdimx].x = sumr;
            out[idx+outIdx*bdimx].y = sumi;
            printf("Out[%d + %d * %d = %d] = %f + %+fi\n", idx, outIdx, bdimx, idx+(outIdx*bdimx),
                out[idx+outIdx*bdimx].x, out[idx+outIdx*bdimx].y);
    // }

    // if (outIdx == n_block_in_stream)
    // {

    // }
}

__global__ void sumDFT_CUDA(cplx *in, cplx *out, int n_stream)
{
     unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
     unsigned int bdimx = blockDim.x;
     float sumr = 0.0, sumi = 0.0;
    // if (idx < n_stream)
    // {
        for (int i = 0; i < n_stream; i++)
        {
            sumr += in[idx+i*bdimx].x;
            sumi += in[idx+i*bdimx].y;
            printf("In[%d + %d * %d = %d] = %f + %+fi\n", idx, i, bdimx, idx+(i*bdimx),
                in[idx+i*bdimx].x, in[idx+i*bdimx].y);
        }

        out[idx].x = sumr;
        out[idx].y = sumi;
    // }
}


__device__ double phaseDFT(unsigned int idx, unsigned int i, int n)
{
    double out;
    out = 2 * M_PI * idx * i / n;
    return out;
}