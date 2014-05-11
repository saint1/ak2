
#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <string>
#include <stdio.h>
#include <time.h> 

using namespace std;

void printMatrix(int n, double **M)
{
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++)
			cout<<M[i][j]<<"\t";
		cout<<endl;
	}
	cout<<endl;
}

void generateMatrix(int n)
{
	srand (time(NULL));
	double **A, **B;
	A = new double * [n];
	B = new double * [n];
	for(int i = 0; i < n; i++) {
		A[i] = new double[n];
		B[i] = new double[n];
	}
	
	//Generujemy macierz trójkątną górną
	for(int i = 0; i < n; i++)
	{
		for(int j = 0; j < n; j++)
		{
			if(i > j){
				A[i][j] = 0;
			}else{
				A[i][j] = rand()%10+1;
			}
		}
	}

	//Generujemy macierz trójkątną dolną
	for(int i = 0; i < n; i++)
	{
		for(int j = 0; j < n; j++)
		{
			if(i < j){
				B[i][j] = 0;
			}else{
				B[i][j] = rand()%10+1;
			}
		}
	}

	ofstream file;

	file.open ("example.txt");
	file<<n<<" "<<n<<endl;
	
	//Mnożymy AxB i zapisujemy do pliku
	for(int i = 0; i < n; i++)//w
	{
		for(int j = 0; j < n; j++)//k
		{
			//Rij = A ity wiers x B ita columna
			double result = 0;
			for(int k = 0; k < n; k++)
				result += A[i][k] * B[k][j]; 
			file<<result<<" ";
		}
		file<<endl;
	}
	file.close();
}

double** loadMatrix(string fileName)
{
	double **M;
	ifstream file ((char*)fileName.c_str());
	if (file.is_open())
	{
		int n,m;
		file>>n>>m;
		
		M = new double * [n];
		for(int i = 0; i < n; i++) M[i] = new double[m];

		for(int i = 0; i < n; i++)
		{
			for(int j = 0; j < m; j++)
			{
				file>>M[i][j];
			}
		}
	}
	file.close();
	return M;
}