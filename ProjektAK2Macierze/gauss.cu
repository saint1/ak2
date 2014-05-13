#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<conio.h>
#include<stdlib.h>
#include<stdio.h> 
#include<math.h> 
#include<cuda.h> 
#include<iostream>
#include <time.h>
#define MAXBLOCKSIZE 512

clock_t start, koniec;

using namespace std;

int Size;
float *a, *b, *finalVec;
float *m;
#define N 20
#define Matrix_Block_Size	32
#define Matrix_Stripe_Size	8

#define THRESHOLD_PIVOT		1e-20
/*-------------------------------------------------------
 ** Fan1() -- Calculate multiplier matrix
 ** Pay attention to the index.  Index i give the range
 ** which starts from 0 to range-1.  The real values of
 ** the index should be adjust and related with the value
 ** of t which is defined on the ForwardSub().
 **-------------------------------------------------------
 */
__global__ void Fan1(float *m_cuda, float *a_cuda, int Size, int t)
{   
	//if(threadIdx.x + blockIdx.x * blockDim.x >= Size-1-t) printf(".");
	//printf("blockIDx.x:%d,threadIdx.x:%d,Size:%d,t:%d,Size-1-t:%d\n",blockIdx.x,threadIdx.x,Size,t,Size-1-t);

	if(threadIdx.x + blockIdx.x * blockDim.x >= Size-1-t) return;
	*(m_cuda+Size*(blockDim.x*blockIdx.x+threadIdx.x+t+1)+t) = *(a_cuda+Size*(blockDim.x*blockIdx.x+threadIdx.x+t+1)+t) / *(a_cuda+Size*t+t);
}

/*-------------------------------------------------------
 ** Fan2() -- Modify the matrix A into LUD
 **-------------------------------------------------------
 */ 

__global__ void Fan2(float *m_cuda, float *a_cuda, float *b_cuda,int Size, int j1, int t)
{
	if(threadIdx.x + blockIdx.x * blockDim.x >= Size-1-t) return;
	if(threadIdx.y + blockIdx.y * blockDim.y >= Size-t) return;

	int xidx = blockIdx.x * blockDim.x + threadIdx.x;
	int yidx = blockIdx.y * blockDim.y + threadIdx.y;
	//printf("blockIdx.x:%d,threadIdx.x:%d,blockIdx.y:%d,threadIdx.y:%d,blockDim.x:%d,blockDim.y:%d\n",blockIdx.x,threadIdx.x,blockIdx.y,threadIdx.y,blockDim.x,blockDim.y);

	a_cuda[Size*(xidx+1+t)+(yidx+t)] -= m_cuda[Size*(xidx+1+t)+t] * a_cuda[Size*t+(yidx+t)];
	//a_cuda[xidx+1+t][yidx+t] -= m_cuda[xidx+1+t][t] * a_cuda[t][yidx+t];
	if(yidx == 0){
		//printf("blockIdx.x:%d,threadIdx.x:%d,blockIdx.y:%d,threadIdx.y:%d,blockDim.x:%d,blockDim.y:%d\n",blockIdx.x,threadIdx.x,blockIdx.y,threadIdx.y,blockDim.x,blockDim.y);
		//printf("xidx:%d,yidx:%d\n",xidx,yidx);
		b_cuda[xidx+1+t] -= m_cuda[Size*(xidx+1+t)+(yidx+t)] * b_cuda[t];
	}
}

/*------------------------------------------------------
 ** ForwardSub() -- Forward substitution of Gaussian
 ** elimination.
 **------------------------------------------------------
 */
void ForwardSub()
{
	int t;
    float *m_cuda,*a_cuda,*b_cuda;

	// allocate memory on GPU
	cudaMalloc((void **) &m_cuda, Size * Size * sizeof(float));

	cudaMalloc((void **) &a_cuda, Size * Size * sizeof(float));

	cudaMalloc((void **) &b_cuda, Size * sizeof(float));	

	// copy memory to GPU
	cudaMemcpy(m_cuda, m, Size * Size * sizeof(float),cudaMemcpyHostToDevice );
	cudaMemcpy(a_cuda, a, Size * Size * sizeof(float),cudaMemcpyHostToDevice );
	cudaMemcpy(b_cuda, b, Size * sizeof(float),cudaMemcpyHostToDevice );

	int block_size,grid_size;

	block_size = MAXBLOCKSIZE;
	grid_size = (Size/block_size) + (!(Size%block_size)? 0:1);
	//printf("1d grid size: %d\n",grid_size);


	dim3 dimBlock(block_size);
	dim3 dimGrid(grid_size);
	//dim3 dimGrid( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1) );

	int blockSize2d, gridSize2d;
	blockSize2d = 4;
	gridSize2d = (Size/blockSize2d) + (!(Size%blockSize2d?0:1)); 

	dim3 dimBlockXY(blockSize2d,blockSize2d);
	dim3 dimGridXY(gridSize2d,gridSize2d);

    // begin timing kernels

	for (t=0; t<(Size-1); t++) {
		Fan1<<<dimGrid,dimBlock>>>(m_cuda,a_cuda,Size,t);
		cudaThreadSynchronize();
		Fan2<<<dimGridXY,dimBlockXY>>>(m_cuda,a_cuda,b_cuda,Size,Size-t,t);
		cudaThreadSynchronize();

	}


	// copy memory back to CPU
	cudaMemcpy(m, m_cuda, Size * Size * sizeof(float),cudaMemcpyDeviceToHost );
	cudaMemcpy(a, a_cuda, Size * Size * sizeof(float),cudaMemcpyDeviceToHost );
	cudaMemcpy(b, b_cuda, Size * sizeof(float),cudaMemcpyDeviceToHost );
	cudaFree(m_cuda);
	cudaFree(a_cuda);
	cudaFree(b_cuda);
}

/*------------------------------------------------------
 ** BackSub() -- Backward substitution
 **------------------------------------------------------
 */

void BackSub()
{
	// create a new vector to hold the final answer
	finalVec = (float *) malloc(Size * sizeof(float));
	// solve "bottom up"
	int i,j;
	for(i=0;i<Size;i++){
		finalVec[Size-i-1]=b[Size-i-1];
		for(j=0;j<i;j++)
		{
			finalVec[Size-i-1]-=*(a+Size*(Size-i-1)+(Size-j-1)) * finalVec[Size-j-1];
		}
		finalVec[Size-i-1]=finalVec[Size-i-1]/ *(a+Size*(Size-i-1)+(Size-i-1));
	}
}
/******************************************/
/* ELIMINATION KERNEL jB - kB - NO SHARED */
/******************************************/
__global__ void kernel_elim_block_jB_kB_no_shared(double* A, double* B, double* R, int Num_Blocks, int iB, int i0, int i1) {

	int jB = (iB + 1) + blockIdx.x; 
	int j0 = jB * N / Num_Blocks; 
	int j1 = (jB + 1) * N / Num_Blocks; 	
	int j = j0 + threadIdx.x;
	
	int kB = (iB + 1) + blockIdx.y; 
	int k0 = kB * N / Num_Blocks;
	int k1 = (kB + 1) * N / Num_Blocks; 
	int k = k0 + threadIdx.y;

	if (j < j1 && k < k1)
		for (int i = i0; i < i1; i++) {
			A[j*N+k] -= R[j*Matrix_Block_Size+(i-i0)] * A[i*N+k]; }
}
/******************************************/
/* ELIMINATION KERNEL iB - iB - NO SHARED */
/******************************************/
__global__ void kernel_elim_block_iB_iB(int* pokpiv, double* A, double* B, double* R, int i0, int i1) {

	for (int i = i0; i < i1; i++) {

		if (fabs(A[i*N+i]) < THRESHOLD_PIVOT) *pokpiv = 0;
		else {
			*pokpiv = 1;
			for (int j = i+1; j < i1; j++) {
				R[j*Matrix_Block_Size+(i-i0)] = A[j*N+i] / A[i*N+i];
				//printf("Matrix kernel1 %i %i %f\n",j,i-i0,R[j*Matrix_Block_Size+(i-i0)]);
				B[j] -= R[j*Matrix_Block_Size+(i-i0)] * B[i];
				for (int k = i; k < i1; k++) {
					A[j*N+k] -= R[j*Matrix_Block_Size+(i-i0)] * A[i*N+k]; }
				//	printf("kernel_elim_block_iB_iB %i %i %f\n",j,k,A[j*N+k]);
				//}
			}
		}
	}
}

/******************************************/
/* ELIMINATION KERNEL iB - kB - NO SHARED */
/******************************************/
__global__ void kernel_elim_block_iB_kB(double* A, double* B, double* R, int Num_Blocks, int iB, int i0, int i1) {

	int kB = (iB + 1) + blockIdx.x; 
	
	int k0 = kB * N / Num_Blocks;
	int k1 = (kB + 1) * N / Num_Blocks; 
	int k = k0 + threadIdx.x;

	if (k < k1) {
		for (int i = i0; i < i1 ; i++)
			for (int j = i + 1; j < i1 ; j++) {
				A[j*N+k] -= R[j*Matrix_Block_Size+(i-i0)] * A[i*N+k]; }
				//printf("kernel_elim_block_iB_kB %i %i %f\n",j,k,A[j*N+k]);
				////printf("Matrix kernel2 %i %i %f\n",j,i-i0,R[j*Matrix_Block_Size+(i-i0)]); }
	}
}

/******************************************/
/* ELIMINATION KERNEL jB - iB - NO SHARED */
/******************************************/
__global__ void kernel_elim_block_jB_iB(double* A, double* B, double* R, int Num_Blocks, int iB, int i0, int i1) {

	int jB = (iB + 1) + blockIdx.x; 
	
	int j0 = jB * N / Num_Blocks;
	int j1 = (jB + 1) * N / Num_Blocks; 
	int j = j0 + threadIdx.x;

	if (j < j1)
		for (int i = i0 ; i < i1 ; i++) {
			R[j*Matrix_Block_Size+(i-i0)] = A[j*N+i] / A[i*N+i];
			B[j] -= R[j*Matrix_Block_Size+(i-i0)] * B[i];
			for (int k = i ; k < i1 ; k++) {
				A[j*N+k] -= R[j*Matrix_Block_Size+(i-i0)] * A[i*N+k]; }
				//printf("kernel_elim_block_jB_iB %i %i %f\n",j,k,A[j*N+k]); } 
		}
}
int forward_elimination_GPU_tiling(double* A, double* B) {

	int Num_Blocks = N / Matrix_Block_Size + (N % Matrix_Block_Size ? 1 : 0);

	// --- GPU memory allocations
	double *d_A; size_t sA = sizeof(double) * N * N;	
	cudaMalloc((void**)&d_A,sA);
	double *d_B; size_t sB = sizeof(double) * N;						
	cudaMalloc((void**)&d_B,sB);
	double *d_R; size_t sR = sizeof(double) * N * Matrix_Block_Size;
	cudaMalloc((void**)&d_R,sR);
	int ok_pivoting, *d_ok_pivoting;									
	cudaMalloc((void**)&d_ok_pivoting,sizeof(int));

	// --- CPU->GPU matrix copies
	cudaMemcpy(d_A,A,sA,cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,B,sB,cudaMemcpyHostToDevice);

	for (int iB=0; iB<Num_Blocks; iB++) {

		int i0 = iB * N / Num_Blocks; 
		int i1 = (iB+1) * N / Num_Blocks;

		kernel_elim_block_iB_iB<<<1,1>>>(d_ok_pivoting,d_A,d_B,d_R,i0,i1);
		cudaThreadSynchronize();

		cudaMemcpy(&ok_pivoting, d_ok_pivoting, sizeof(int), cudaMemcpyDeviceToHost);
		if (!ok_pivoting) return 0;

		kernel_elim_block_iB_kB<<<Num_Blocks-(iB+1),Matrix_Block_Size>>>(d_A,d_B,d_R,Num_Blocks,iB,i0,i1);
		cudaThreadSynchronize();

		if (iB < Num_Blocks-1) {
	
			kernel_elim_block_jB_iB<<<Num_Blocks-(iB+1),Matrix_Block_Size>>>(d_A,d_B,d_R,Num_Blocks,iB,i0,i1);
			cudaThreadSynchronize();

			dim3 blocks(Num_Blocks-(iB+1),Num_Blocks-(iB+1));
			dim3 threads(Matrix_Block_Size,Matrix_Block_Size);
	
			 kernel_elim_block_jB_kB_no_shared <<<blocks,threads>>>(d_A,d_B,d_R,Num_Blocks,iB,i0,i1);
			
			cudaThreadSynchronize();

		}
	}

	cudaMemcpy(A,d_A,sA,cudaMemcpyDeviceToHost);
	cudaMemcpy(B,d_B,sB,cudaMemcpyDeviceToHost);

	cudaFree(d_A); cudaFree(d_B);
	cudaFree(d_R); cudaFree(d_ok_pivoting);

	return 1;

}

int solution_of_a_triangular_system_CPU(double* A, double* B, double* x) {
	
	if (fabs(A[(N-1)*N+(N-1)]) < THRESHOLD_PIVOT) return 0;

	x[N-1] = B[N-1] / A[(N-1)*N+(N-1)];
	
	for (int i=N-2; i>=0; i--) {
		double s = 0;

		for (int j=i+1; j<N; j++)
			s += A[i*N+j] * x[j];

		x[i] = (B[i] - s) / A[i*N+i];
	}

	return 1;
}

int main(int argc , char **argv) 
{ 
	


    Size = 0; 

	cin>>Size;
	//N=Size;

	a = (float*)malloc(sizeof(float)*Size*(Size)); 
	m = (float *) malloc(Size * Size * sizeof(float));
	double* A = (double*)malloc(N*N*sizeof(double));
	double* B = (double*)malloc(N*sizeof(double));
	double* x = (double*)malloc(N*sizeof(double));
	for (int i=0; i<Size*Size; i++)
			*(m+i) = 0.0;
	for(int i =0 ; i< Size ;i++) 
    { 
		B[i]= (double)(rand()%50+1);
        for(int j =0 ; j< Size; j++) 
        { 		
			A[i*Size+j] = (double)(rand()%50+1);
		}
	}
	//b = (float *) malloc(Size * sizeof(float)); 

	cout<<"GPU"<<endl;
	start = clock();
	for(int i =0 ; i< Size ;i++)         
		printf("%f ",A[i]);
	cout<<endl;
	//ForwardSub();
	forward_elimination_GPU_tiling(A,B);
	//solution_of_a_triangular_system_CPU(A,B,x);
	koniec = clock(); 
	long delta=(long)(koniec - start);
	//BackSub();
	for(int i =0 ; i< Size ;i++)         
		printf("%f ",A[i]);
	//for(int i =0 ; i< Size ;i++)         
	//	printf("x%d = %f ",i,x[i]);
	cout<<endl;
	cout<<"czas dzialania: "<<delta<<" [ms]"<<endl;

	double **AB, *X;
	int      n,i,j,k;
	  
  double m_,s;
  AB = new double * [Size];
	X  = new double [Size];
	  cout<< endl<<endl<<"CPU\n";
	for(i = 0; i < Size; i++) AB[i] = new double[Size + 1];
  // eliminacja wspó³czynników
	start = clock();
  for(i = 0; i < Size; i++){
		for(j = 0; j < Size; j++) AB[i][j] = A[i*Size+j];
		AB[i][Size] = B[i];
  }
  for(i = 0; i < Size - 1; i++)
  {
    for(j = i + 1; j < Size; j++)
    {
      
	  m_ = -AB[j][i] / AB[i][i];	
      
	  for(k = i + 1; k <= Size; k++)
        
		  AB[j][k] += m_ * AB[i][k];	
		  

    }
  }
    
  // wyliczanie niewiadomych

  for(i = Size - 1; i >= 0; i--)
  {
    s = AB[i][Size];
    for(j = Size - 1; j >= i + 1; j--)
      s -= AB[i][j] * X[j];   
    X[i] = s / AB[i][i];
  }

    for(i = 0; i < Size; i++)
	printf("x%d = %f ",i,X[i]);
  	koniec = clock(); 
	delta=(long)(koniec - start);
  	cout<<endl;
	cout<<"czas dzialania: "<<delta<<" [ms]"<<endl;
    free(m);
    free(a);
    free(b);
	getchar();
    _getch(); 
    return 0; 
}
