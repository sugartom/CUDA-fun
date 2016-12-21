////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

//
// Matrix multiplication: C = A * B.
// Host code.
//
// This sample implements matrix multiplication as described in Chapter 3
// of the programming guide and uses the CUBLAS library to demonstrate
// the best performance.

// SOME PRECAUTIONS:
// IF WE WANT TO CALCULATE ROW-MAJOR MATRIX MULTIPLY C = A * B,
// WE JUST NEED CALL CUBLAS API IN A REVERSE ORDER: cublasSegemm(B, A)!
// The reason is explained as follows:

// CUBLAS library uses column-major storage, but C/C++ use row-major storage.
// When passing the matrix pointer to CUBLAS, the memory layout alters from
// row-major to column-major, which is equivalent to an implicit transpose.

// In the case of row-major C/C++ matrix A, B, and a simple matrix multiplication
// C = A * B, we can't use the input order like cublasSgemm(A, B)  because of
// implicit transpose. The actual result of cublasSegemm(A, B) is A(T) * B(T).
// If col(A(T)) != row(B(T)), equal to row(A) != col(B), A(T) and B(T) are not
// multipliable. Moreover, even if A(T) and B(T) are multipliable, the result C
// is a column-based cublas matrix, which means C(T) in C/C++, we need extra
// transpose code to convert it to a row-based C/C++ matrix.

// To solve the problem, let's consider our desired result C, a row-major matrix.
// In cublas format, it is C(T) actually (because of the implicit transpose).
// C = A * B, so C(T) = (A * B) (T) = B(T) * A(T). Cublas matrice B(T) and A(T)
// happen to be C/C++ matrice B and A (still because of the implicit transpose)!
// We don't need extra transpose code, we only need alter the input order!
//
// CUBLAS provides high-performance matrix multiplication.
// See also:
// V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
// in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
// Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
//

// Utilities and system includes
#include <assert.h>
#include <helper_string.h>  // helper for shared functions common to CUDA Samples

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>

#ifndef min
#define min(a,b) ((a < b) ? a : b)
#endif
#ifndef max
#define max(a,b) ((a > b) ? a : b)
#endif

// This is a kernel that does no real work but runs at least for a specified number of clocks
__global__ void clock_block(clock_t *d_o, clock_t clock_count)
{
    unsigned int start_clock = (unsigned int) clock();

    clock_t clock_offset = 0;

    while (clock_offset < clock_count)
    {
        unsigned int end_clock = (unsigned int) clock();

        // The code below should work like
        // this (thanks to modular arithmetics):
        //
        // clock_offset = (clock_t) (end_clock > start_clock ?
        //                           end_clock - start_clock :
        //                           end_clock + (0xffffffffu - start_clock));
        //
        // Indeed, let m = 2^32 then
        // end - start = end + m - start (mod m).

        clock_offset = (clock_t)(end_clock - start_clock);
    }

    d_o[0] = clock_offset;
}

typedef struct _matrixSize      // Optional Command-line multiplier for matrix sizes
{
    unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
} sMatrixSize;

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set matrix multiply on CPU
//! C = A * B
//! @param C          reference data, computed but preallocated
//! @param A          matrix A as provided to device
//! @param B          matrix B as provided to device
//! @param hA         height of matrix A
//! @param wB         width of matrix B
////////////////////////////////////////////////////////////////////////////////

// Allocates a matrix with random float entries.
void randomInit(float *data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

void initializeCUDA(int argc, char **argv, int &devID, int &iSizeMultiple, sMatrixSize &matrix_size)
{
    // By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
    cudaError_t error;
    devID = 0;

    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        devID = getCmdLineArgumentInt(argc, (const char **)argv, "device");
        error = cudaSetDevice(devID);

        if (error != cudaSuccess)
        {
            printf("cudaSetDevice returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }
    }

    // get number of SMs on this GPU
    error = cudaGetDevice(&devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }


    if (checkCmdLineFlag(argc, (const char **)argv, "sizemult"))
    {
        iSizeMultiple = getCmdLineArgumentInt(argc, (const char **)argv, "sizemult");
    }

    // iSizeMultiple = min(iSizeMultiple, 10);
    // iSizeMultiple = max(iSizeMultiple, 1);

    cudaDeviceProp deviceProp;

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);

    // use a larger block size for Fermi and above
    int block_size = (deviceProp.major < 2) ? 16 : 32;

    matrix_size.uiWA = 1 * block_size * iSizeMultiple;
    matrix_size.uiHA = 1 * block_size * iSizeMultiple;
    matrix_size.uiWB = 1 * block_size * iSizeMultiple;
    matrix_size.uiHB = 1 * block_size * iSizeMultiple;
    matrix_size.uiWC = 1 * block_size * iSizeMultiple;
    matrix_size.uiHC = 1 * block_size * iSizeMultiple;

    printf("MatrixA(%u,%u), MatrixB(%u,%u), MatrixC(%u,%u)\n",
           matrix_size.uiHA, matrix_size.uiWA,
           matrix_size.uiHB, matrix_size.uiWB,
           matrix_size.uiHC, matrix_size.uiWC);

    if( matrix_size.uiWA != matrix_size.uiHB ||
        matrix_size.uiHA != matrix_size.uiHC ||
        matrix_size.uiWB != matrix_size.uiWC)
    {
       printf("ERROR: Matrix sizes do not match!\n");
       exit(-1);
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test matrix multiply using CUBLAS
////////////////////////////////////////////////////////////////////////////////
int matrixMultiply(int argc, char **argv, int devID, sMatrixSize &matrix_size)
{
    cudaDeviceProp deviceProp;

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

    // use a larger block size for Fermi and above
    // int block_size = (deviceProp.major < 2) ? 16 : 32;

    // set seed for rand()
    srand(2006);

    // allocate host memory for matrices A and B
    unsigned int size_A = matrix_size.uiWA * matrix_size.uiHA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = (float *)malloc(mem_size_A);
	float *h_A2 = (float *)malloc(mem_size_A);
    float *h_A3 = (float *)malloc(mem_size_A);
    float *h_A4 = (float *)malloc(mem_size_A);
	unsigned int size_B = matrix_size.uiWB * matrix_size.uiHB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = (float *)malloc(mem_size_B);
	float *h_B2 = (float *)malloc(mem_size_B);
    float *h_B3 = (float *)malloc(mem_size_B);
    float *h_B4 = (float *)malloc(mem_size_B);

    // set seed for rand()
    srand(2006);

    // initialize host memory
    randomInit(h_A, size_A);
	randomInit(h_A2, size_A);
    randomInit(h_A3, size_A);
    randomInit(h_A4, size_A);
    randomInit(h_B, size_B);
	randomInit(h_B2, size_B);
    randomInit(h_B3, size_B);
    randomInit(h_B4, size_B);

    // allocate device memory
    float *d_A, *d_B, *d_C;
	float *d_A2, *d_B2, *d_C2;
    float *d_A3, *d_B3, *d_C3;
    float *d_A4, *d_B4, *d_C4;
    unsigned int size_C = matrix_size.uiWC * matrix_size.uiHC;
    unsigned int mem_size_C = sizeof(float) * size_C;

    // allocate host memory for the result
    float *h_CUBLAS = (float *) malloc(mem_size_C);
	float *h_CUBLAS2 = (float *) malloc(mem_size_C);
    float *h_CUBLAS3 = (float *) malloc(mem_size_C);
    float *h_CUBLAS4 = (float *) malloc(mem_size_C);

    checkCudaErrors(cudaMalloc((void **) &d_A, mem_size_A));
	checkCudaErrors(cudaMalloc((void **) &d_A2, mem_size_A));
    checkCudaErrors(cudaMalloc((void **) &d_A3, mem_size_A));
    checkCudaErrors(cudaMalloc((void **) &d_A4, mem_size_A));
    checkCudaErrors(cudaMalloc((void **) &d_B, mem_size_B));
	checkCudaErrors(cudaMalloc((void **) &d_B2, mem_size_B));
    checkCudaErrors(cudaMalloc((void **) &d_B3, mem_size_B));
    checkCudaErrors(cudaMalloc((void **) &d_B4, mem_size_B));
    checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_A2, h_A2, mem_size_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_A3, h_A3, mem_size_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_A4, h_A4, mem_size_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_B2, h_B2, mem_size_B, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B3, h_B3, mem_size_B, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B4, h_B4, mem_size_B, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &d_C, mem_size_C));
	checkCudaErrors(cudaMalloc((void **) &d_C2, mem_size_C));
    checkCudaErrors(cudaMalloc((void **) &d_C3, mem_size_C));
    checkCudaErrors(cudaMalloc((void **) &d_C4, mem_size_C));

    // setup execution parameters
    // dim3 threads(block_size, block_size);
    // dim3 grid(matrix_size.uiWC / threads.x, matrix_size.uiHC / threads.y);

	cudaStream_t stream1, stream2;
	checkCudaErrors(cudaStreamCreate(&stream1));
	checkCudaErrors(cudaStreamCreate(&stream2));

    cudaStream_t stream3, stream4;
    checkCudaErrors(cudaStreamCreate(&stream3));
    checkCudaErrors(cudaStreamCreate(&stream4));

    // create and start timer
    printf("Computing result using CUBLAS...");

    // execute the kernel
    int nIter = 10;

	// tom
    int event_num = 2;
    cudaEvent_t *tomEvent = (cudaEvent_t *) malloc(event_num * sizeof(cudaEvent_t));
    for (int i = 0; i < event_num; i++) {
        checkCudaErrors(cudaEventCreate(&tomEvent[i]));
    }
    float kernel_time = 1; // time the kernel should run in ms
    clock_t time_clocks = (clock_t)(kernel_time * deviceProp.clockRate);

    int wait_num = 1;
    int nbytes = wait_num * sizeof(clock_t);   // number of data bytes
    clock_t *a = 0;                     // pointer to the array data in host memory
    checkCudaErrors(cudaMallocHost((void **)&a, nbytes));
    clock_t *d_a = 0;             // pointers to data and init value in the device memory
    checkCudaErrors(cudaMalloc((void **)&d_a, nbytes));
    // tom

    // CUBLAS version 2.0
    {
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        cublasHandle_t handle;
        cudaEvent_t start, stop;

        checkCudaErrors(cublasCreate(&handle));

        //Perform warmup operation with cublas
        checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B, matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWB));

        // Allocate CUDA events that we'll use for timing
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));

        // Record the start event
        checkCudaErrors(cudaEventRecord(start, NULL));

		// tom
        clock_block<<<1,1,0,stream2>>>(&d_a[0], time_clocks);
        checkCudaErrors(cudaEventRecord(tomEvent[0], stream2));
        // tom

        for (int j = 0; j < nIter; j++)
        {
            //note cublas is column primary!
            //need to transpose the order
			checkCudaErrors(cublasSetStream(handle, stream3));
			if (j == 0) {
				checkCudaErrors(cudaStreamWaitEvent(stream3, tomEvent[0],0));
			}
            checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B3, matrix_size.uiWB, d_A3, matrix_size.uiWA, &beta, d_C3, matrix_size.uiWB));
			checkCudaErrors(cublasSetStream(handle, stream4));
			if (j == 0) {
				checkCudaErrors(cudaStreamWaitEvent(stream4, tomEvent[0],0));
			}
			checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B4, matrix_size.uiWB, d_A4, matrix_size.uiWA, &beta, d_C4, matrix_size.uiWB));
        }

        checkCudaErrors(cudaEventRecord(tomEvent[1], stream3));


        // unsigned long int counter=0;
        // while (cudaEventQuery(tomEvent[0]) == cudaErrorNotReady)
        // {
        //     counter++;
        // }

        for (int j = 0; j < nIter; j++)
        {
            //note cublas is column primary!
            //need to transpose the order
            checkCudaErrors(cublasSetStream(handle, stream1));
            if (j == 0) {
                checkCudaErrors(cudaStreamWaitEvent(stream1, tomEvent[1],0));
            }
            checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B, matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWB));
            checkCudaErrors(cublasSetStream(handle, stream2));
            if (j == 0) {
                checkCudaErrors(cudaStreamWaitEvent(stream2, tomEvent[1],0));
            }
            checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B2, matrix_size.uiWB, d_A2, matrix_size.uiWA, &beta, d_C2, matrix_size.uiWB));
        }




        printf("done.\n");

        // Record the stop event
        checkCudaErrors(cudaEventRecord(stop, NULL));

        // Wait for the stop event to complete
        checkCudaErrors(cudaEventSynchronize(stop));

        float msecTotal = 0.0f;
        checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

        // Compute and print the performance
        float msecPerMatrixMul = msecTotal / nIter;
		printf("Runnignt time = %.3f msec\n", msecPerMatrixMul);

		checkCudaErrors(cudaMemcpy(h_CUBLAS, d_C, mem_size_C, cudaMemcpyDeviceToHost));

        // Destroy the handle
        checkCudaErrors(cublasDestroy(handle));
    }

    // clean up memory
    free(h_A);
    free(h_B);
	free(h_CUBLAS);
	free(h_A2);
	free(h_B2);
	free(h_CUBLAS2);
    free(h_A3);
    free(h_B3);
    free(h_CUBLAS3);
    free(h_A4);
    free(h_B4);
    free(h_CUBLAS4);
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
	checkCudaErrors(cudaFree(d_A2));
	checkCudaErrors(cudaFree(d_B2));
	checkCudaErrors(cudaFree(d_C2));
    checkCudaErrors(cudaFree(d_A3));
    checkCudaErrors(cudaFree(d_B3));
    checkCudaErrors(cudaFree(d_C3));
    checkCudaErrors(cudaFree(d_A4));
    checkCudaErrors(cudaFree(d_B4));
    checkCudaErrors(cudaFree(d_C4));

    if (true)
    {
        return EXIT_SUCCESS;    // return value = 1
    }
    else
    {
        return EXIT_FAILURE;     // return value = 0
    }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    if (argc != 2) {
        printf("ERROR: wrong input format! It should be: ./matrixMulCUBLAS [number of 32s]\n");
        printf("For example: ./matrixMulCUBLAS 8\n");
        return -1;
    } else {
        printf("[Matrix Multiply CUBLAS] - Starting...\n");

        int devID = 0, sizeMult = atoi(argv[1]);
        sMatrixSize matrix_size;

        initializeCUDA(argc, argv, devID, sizeMult, matrix_size);

        int matrix_result = matrixMultiply(argc, argv, devID, matrix_size);

        return matrix_result;
    }
}
