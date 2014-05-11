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


int main(int argc , char **argv) 
{ 

    Size = 0; 

	cin>>Size;
	a = (float*)malloc(sizeof(float)*Size*(Size)); 
	m = (float *) malloc(Size * Size * sizeof(float));
	
	for (int i=0; i<Size*Size; i++)
			*(m+i) = 0.0;
	for(int i =0 ; i< Size ;i++) 
    { 
        for(int j =0 ; j< Size; j++) 
        { 		
			a[i*Size+j] = (float)(rand()%50+1);
		}
	}
	b = (float *) malloc(Size * sizeof(float)); 
	for (int i=0; i<Size; i++) {
		b[i]= (float)(rand()%50+1);
	}
	cout<<"GPU"<<endl;
	start = clock();
	ForwardSub();
	
	koniec = clock(); 
	long delta=(long)(koniec - start);
	BackSub();
	//for(int i =0 ; i< Size ;i++)         
	//	printf("x%d = %f ",i,finalVec[i]);
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
		for(j = 0; j < Size; j++) AB[i][j] = a[i*Size+j];
		AB[i][Size] = b[i];
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
    	koniec = clock(); 
	delta=(long)(koniec - start);
  // wyliczanie niewiadomych

  for(i = Size - 1; i >= 0; i--)
  {
    s = AB[i][Size];
    for(j = Size - 1; j >= i + 1; j--)
      s -= AB[i][j] * X[j];   
    X[i] = s / AB[i][i];
  }

    //for(i = 0; i < Size; i++)
	//printf("x%d = %f ",i,X[i]);

  	cout<<endl;
	cout<<"czas dzialania: "<<delta<<" [ms]"<<endl;
    free(m);
    free(a);
    free(b);
	getchar();
    _getch(); 
    return 0; 
}
