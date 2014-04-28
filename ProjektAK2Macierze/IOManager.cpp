#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <string>

#include <stdio.h>
using namespace std;

//Tworzy macierz N x M
void generateMatrix(int n, int m)
{
	double **M;
	M = new double * [n];
	for(int i = 0; i < n; i++) {
		M[i] = new double[m];
	}
	ofstream file;

	file.open ("example.txt");
	file<<n<<" "<<m<<endl;

	for(int i = 0; i < n; i++)
	{
		for(int j = 0; j <= m; j++)
		{
			M[i][j] = rand()%50+1;
			file<<M[i][j]<<" ";
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
			for(int j = 0; j <= m; j++)
			{
				file>>M[i][j];
			}
		}
	}
	file.close();
	return M;
}
