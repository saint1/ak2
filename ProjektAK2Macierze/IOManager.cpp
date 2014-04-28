#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <string>
#include <stdio.h>

using namespace std;

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
		for(int j = 0; j < m; j++)
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
			for(int j = 0; j < m; j++)
			{
				file>>M[i][j];
			}
		}
	}
	file.close();
	return M;
}

// Rekurencyjna funkcja obliczająca rozwinięcie Laplace'a
//Funkcja rekurencyjna det(n,w,WK,A):
//Parametrami funkcji są:
//n – stopień podmacierzy – przekazywane przez wartość
//w – bieżący wiersz macierzy głównej, w którym rozpoczyna się podmacierz – przekazywane przez wartość
//WK – wektor kolumn o n elementach – przekazanie przez referencję
//A – macierz podstawowa – przekazanie przez referencję
//-------------------------------------------------------
double det(int n, int w, int * WK, double ** A)
{
  int    i,j,k,m, * KK;
  double s;

  if(n == 1)                                    // sprawdzamy warunek zakończenia rekurencji

    return A[w][WK[0]];                         // macierz 1 x 1, wyznacznik równy elementowi

  else
  {

    KK = new int[n - 1];                        // tworzymy dynamiczny wektor kolumn

    s = 0;                                      // zerujemy wartość rozwinięcia
    m = 1;                                      // początkowy mnożnik

    for(i = 0; i < n; i++)                      // pętla obliczająca rozwinięcie
    {

      k = 0;                                    // tworzymy wektor kolumn dla rekurencji

      for(j = 0; j < n - 1; j++)                // ma on o 1 kolumnę mniej niż WK
      {
        if(k == i) k++;                         // pomijamy bieżącą kolumnę
        KK[j] = WK[k++];                        // pozostałe kolumny przenosimy do KK
      }

      s += m * A[w][WK[i]] * det(n - 1,w  + 1, KK, A);

      m = -m;                                   // kolejny mnożnik

    }

    delete [] KK;                               // usuwamy zbędną już tablicę dynamiczną

    return s;                                   // ustalamy wartość funkcji

  }
}

//Funkcja obliczajaca rzad macierzy
//----------------
int rankOfMatrix(int row, int col, double **M)
{
	int n; //maksymalny rozmiar macierzy dla której mozna policzyć wyznacznik
	int * WK;   //wiersz kolumn
	if( row > col ) n = col;
	else n = row;

	//Step 1: Liczymy wyznacznik dla najwiekszej macierzy
	WK = new int[n];									// tworzymy wiersz kolumn
	for(int i = 0; i < n; i++)							// wypełniamy go numerami kolumn
		WK[i] = i;

	double detM = det(n,0,WK,M);
	
	//Step 2: Liczymy pozostale wyznaczniki jesli detM!=0
	int r = 0; //wiersz poczatku macierzy
	int k = 0;
	while(detM == 0 && n > 1)
	{
		if( r+n > row )									//sprawdzenie czy mozemy policzyc wyznacznik
		{
			n--;
			r=0;
			k=0;
		}
		
		WK = new int[n];								// tworzymy wiersz kolumn
		for(int i = 0; i < n; i++){                      // wypełniamy go numerami kolumn
			if( i < k )	WK[i] = i;
			else WK[i] = i+1;
		}
		detM = det(n,r,WK,M);
		delete []WK;

		k++;											//nastepna kolumna do wyciecia
		if(k >= col) r++;								//jesli sprawdzono wszystkie mozliwosci dla kolum przej do nastepnego wiersza
	}
	
	return n;
}