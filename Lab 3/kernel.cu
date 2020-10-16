#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_atomic_functions.h"

#include <time.h>
#include <windows.h>

#define N 2048

#define M 1024
#define T 100
int a[N], b[N], c[N];

using namespace std;

__global__ void add(int* a, int* b, int* c) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}

void FillArrays() {

    for (int i = 0; i < N; i++) {
        srand(i);
        a[i] = rand() % 1000000;
        b[i] = rand() & 1000000;
    }
    return;
}

float add_serial() {

    float time;
    LARGE_INTEGER t1, t2, tc;
    QueryPerformanceFrequency(&tc);
    QueryPerformanceCounter(&t1);

    FillArrays();

    QueryPerformanceCounter(&t2);
    time = (float)(t2.QuadPart - t1.QuadPart) / (double)tc.QuadPart * 1000;
  
    for (int i = 0; i < N; i++)
        c[i] = a[i] + b[i];

    return time;
}

float add_parallel_explicit() {

    int* d_a, * d_b, * d_c;

    cudaEvent_t startT, stopT;
    float time;
    cudaEventCreate(&startT);
    cudaEventCreate(&stopT);
    cudaEventRecord(startT, 0);

    cudaMalloc((void**)&d_a, sizeof(int) * N);
    cudaMalloc((void**)&d_b, sizeof(int) * N);
    cudaMalloc((void**)&d_c, sizeof(int) * N);

    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    add << <(N + M - 1) / M, M >> > (d_a, d_b, d_c);

    cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(stopT, 0);
    cudaEventSynchronize(stopT);
    cudaEventElapsedTime(&time, startT, stopT);
    cudaEventDestroy(startT);
    cudaEventDestroy(stopT);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);


    
    return time;
}

float add_parallel_unified() {

    int* aa, * bb, * cc;

    cudaEvent_t startT, stopT;
    float time, time1;
    

    LARGE_INTEGER t1, t2, tc;
    QueryPerformanceFrequency(&tc);
    QueryPerformanceCounter(&t1);

    cudaMallocManaged(&aa, N * sizeof(int));
    cudaMallocManaged(&bb, N * sizeof(int));
    cudaMallocManaged(&cc, N * sizeof(int));

    QueryPerformanceCounter(&t2);
    time1 = (float)(t2.QuadPart - t1.QuadPart) / (double)tc.QuadPart * 1000;
    
    for (int i = 0; i < N; i++) {
        srand(i);
        aa[i] = rand() % 1000000;
        bb[i] = rand() & 1000000;
    }

    cudaEventCreate(&startT);
    cudaEventCreate(&stopT);
    cudaEventRecord(startT);

    add << <(N + M - 1) / M, M >> > (aa, bb, cc);

    cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "error! : %s\n", cudaGetErrorString(cudaErr));
    }

    cudaEventRecord(stopT);
    cudaEventSynchronize(stopT);
    cudaEventElapsedTime(&time, startT, stopT);
    cudaEventDestroy(startT);
    cudaEventDestroy(stopT);

    cudaFree(aa);
    cudaFree(bb);
    cudaFree(cc);

    return time + time1;
}



int main() {

    float t1 = 0, t2 = 0, t3 = 0;

    for (int i = 0; i < T; i++) {
        //t1 += add_serial() / T;
        if (i == 1) {
            t2 = 0;
            t3 = 0;
        }
        t2 += add_parallel_explicit() / (T-1);
        t3 += add_parallel_unified() / (T-1);
        cout << "iteration: " << i << endl;
    }



    //cout << "serial time:\t" << t1 << endl;
    cout << "parallel explicit time:\t" << t2 << endl;
    cout << "parallel unified time:\t" << t3 << endl;

    return 0;
}