#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <iomanip>
#include <cmath>

#include <stdio.h>
using namespace std;

const double eps = 1e-12; // sta³a przybli¿enia zera
//test ms:)
__global__ void addAndMulGauss(double *bj, const double *ai, const  double m)
{
    int i = threadIdx.x;
    bj[i] += m * ai[i];
}


// Funkcja realizuje algorytm eliminacji Gaussa
//---------------------------------------------
bool gaussWithCuda(int n, double ** AB, double * X,unsigned int size)
{

	int i,j,k;
	double m,s;
   
	int *dev_ai = 0;
	int *dev_bj = 0;
	double *dev_m = 0;

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_m, sizeof(double));

	cudaStatus = cudaMalloc((void**)&dev_ai, size * sizeof(double));

	cudaStatus = cudaMalloc((void**)&dev_bj, size * sizeof(double));
	// eliminacja wspó³czynników

	for(i = 0; i < n - 1; i++)
	{
		for(j = i + 1; j < n; j++)
		{
			if(fabs(AB[i][i]) < eps) return false;
      
			m = -AB[j][i] / AB[i][i];	
      
			cudaStatus = cudaMemcpy(dev_ai, AB[i], size * sizeof(double), cudaMemcpyHostToDevice);
			cudaStatus = cudaMemcpy(dev_bj, AB[j], size * sizeof(double), cudaMemcpyHostToDevice);
			cudaStatus = cudaMemcpy(dev_m, &m, sizeof(double), cudaMemcpyHostToDevice);
			
		//	for(k = i + 1; k <= n; k++)
        
				//AB[j][k] += m * AB[i][k];	//zrownoleglenie		  
			//addAndMulGauss<<<1, size>>>(dev_bj, dev_ai, dev_m);
			cudaStatus = cudaDeviceSynchronize();

			// Copy output vector from GPU buffer to host memory.
			cudaStatus = cudaMemcpy(AB[j], dev_bj, size * sizeof(double), cudaMemcpyDeviceToHost);

		}
	}
	// wyliczanie niewiadomych

	for(i = n - 1; i >= 0; i--)
	{
		s = AB[i][n];
		for(j = n - 1; j >= i + 1; j--)
			s -= AB[i][j] * X[j];
		if(fabs(AB[i][i]) < eps) return false;
		X[i] = s / AB[i][i]; //zrownoleglenia
	}
	return true;
}
// Funkcja realizuje algorytm eliminacji Gaussa
//---------------------------------------------
bool gauss(int n, double ** AB, double * X)
{

  int i,j,k;
  double m,s;

  // eliminacja wspó³czynników

  for(i = 0; i < n - 1; i++)
  {
    for(j = i + 1; j < n; j++)
    {
      if(fabs(AB[i][i]) < eps) return false;
      
	  m = -AB[j][i] / AB[i][i];	
      
	  for(k = i + 1; k <= n; k++)
        
		  AB[j][k] += m * AB[i][k];	//zrownoleglenie
		  

    }
  }

  // wyliczanie niewiadomych

  for(i = n - 1; i >= 0; i--)
  {
    s = AB[i][n];
    for(j = n - 1; j >= i + 1; j--)
      s -= AB[i][j] * X[j];
    if(fabs(AB[i][i]) < eps) return false;
    X[i] = s / AB[i][i];
  }
  return true;
}

// Program g³ówny
//---------------

int main1)
{
	clock_t start, koniec;
	double **AB, *X;
	int      n,i,j;

	cout << setprecision(4) << fixed;
  
	// odczytujemy liczbê niewiadomych

	cin >> n;

	// tworzymy macierze AB i X

	AB = new double * [n];
	X  = new double [n];

	for(i = 0; i < n; i++) AB[i] = new double[n + 1];

	// odczytujemy dane dla macierzy AB

	for(i = 0; i < n; i++)
		for(j = 0; j <= n; j++) AB[i][j] = rand()%50+1;//cin >> AB[i][j];
	
	start = clock(); // bie¿¹cy czas systemowy w ms		

	if(gauss(n,AB,X))
	{
		koniec = clock(); // bie¿¹cy czas systemowy w ms
		long delta=(long)(koniec - start);//czas dzia³añ w ms
		cout <<"czas wykonania: "<< delta << " ms\n";
		for(i = 0; i < n; i++)
		cout << "x" << i + 1 << " = " << setw(9) << X[i]
		<< endl;	
	}
	else
	cout << "DZIELNIK ZERO\n";

	
	start = clock(); // bie¿¹cy czas systemowy w ms		
	if(gaussWithCuda(n,AB,X,n))
	{
		koniec = clock(); // bie¿¹cy czas systemowy w ms
		long delta=(long)(koniec - start);//czas dzia³añ w ms
		cout <<"czas wykonania: "<< delta << " ms\n";
		for(i = 0; i < n; i++)
		cout << "x" << i + 1 << " = " << setw(9) << X[i]
		<< endl;	
	}
	else
	cout << "DZIELNIK ZERO\n";


	// usuwamy macierze z pamiêci

	for(i = 0; i < n; i++) delete [] AB[i];
		delete [] AB;	
		delete [] X;
		getchar();
		getchar();
		return 0;
} 