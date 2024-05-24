#include <omp.h>
#include <iostream>
#include <windows.h>
#include <immintrin.h>  // 包含AVX支持
using namespace std;

const int  n = 1000;
float arr[n][n];
float A[n][n];
 int NUM_THREADS = 7; //工作线程数量
 int cycle = 5;

void init()
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			arr[i][j] = 0;
		}
		arr[i][i] = 1.0;
		for (int j = i + 1; j < n; j++)
			arr[i][j] = rand() % 100;
	}

	for (int i = 0; i < n; i++)
	{
		int k1 = rand() % n;
		int k2 = rand() % n;
		for (int j = 0; j < n; j++)
		{
			arr[i][j] += arr[0][j];
			arr[k1][j] += arr[k2][j];
		}
	}
}


void ReStart()
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
			A[i][j] = arr[i][j];
	}
}


void f_ordinary()
{
	for (int k = 0; k < n; k++)
	{
		for (int j = k + 1; j < n; j++)
		{
			A[k][j] = A[k][j] * 1.0 / A[k][k];
		}
		A[k][k] = 1.0;

		for (int i = k + 1; i < n; i++)
		{
			for (int j = k + 1; j < n; j++)
			{
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			}
			A[i][k] = 0;
		}
	}
}

void f_omp_static()
{
#pragma omp parallel num_threads(NUM_THREADS)

	for (int k = 0; k < n; k++)
	{
		//串行部分
#pragma omp single
		{
			float tmp = A[k][k];
			for (int j = k + 1; j < n; j++)
			{
				A[k][j] = A[k][j] / tmp;
			}
			A[k][k] = 1.0;
		}

		//并行部分
#pragma omp for schedule(static)
		for (int i = k + 1; i < n; i++)
		{
			float tmp = A[i][k];
			for (int j = k + 1; j < n; j++) 
				A[i][j] = A[i][j] - tmp * A[k][j];
			
			A[i][k] = 0;
		}
		
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}

void f_omp_static_avx()
{
#pragma omp parallel num_threads(NUM_THREADS)
	{
		for (int k = 0; k < n; k++)
		{
			// 串行部分
#pragma omp single
			{
				float tmp = A[k][k];
				__m256 tmp_vec = _mm256_set1_ps(tmp);  // 将tmp广播到8个浮点数

				for (int j = k + 1; j < n; j += 8)
				{
					__m256 a_vec = _mm256_loadu_ps(&A[k][j]); // 加载A[k][j]到j+7的数据
					a_vec = _mm256_div_ps(a_vec, tmp_vec);   // 并行除法
					_mm256_storeu_ps(&A[k][j], a_vec);       // 存储结果回A[k][j]
				}
				A[k][k] = 1.0;
			}

			// 并行部分
#pragma omp for schedule(static)
			for (int i = k + 1; i < n; i++)
			{
				float tmp = A[i][k];
				__m256 tmp_vec = _mm256_set1_ps(tmp);

				for (int j = k + 1; j < n; j += 8)
				{
					__m256 a_vec = _mm256_loadu_ps(&A[i][j]);
					__m256 k_vec = _mm256_loadu_ps(&A[k][j]);
					k_vec = _mm256_mul_ps(tmp_vec, k_vec); // 并行乘法
					a_vec = _mm256_sub_ps(a_vec, k_vec);   // 并行减法
					_mm256_storeu_ps(&A[i][j], a_vec);     // 存储结果回A[i][j]
				}
				A[i][k] = 0;
			}
		}
	}
}

void f_omp_static_avx_barrier()
{
#pragma omp parallel num_threads(NUM_THREADS)
//#pragma omp parallel if(parallel), num_threads(NUM_THREADS),private(i, j, k, tmp)
	{
		for (int k = 0; k < n; k++)
		{
			// 串行部分
#pragma omp master
			{
				float tmp = A[k][k];
				__m256 tmp_vec = _mm256_set1_ps(tmp);  // 将tmp广播到8个浮点数

				for (int j = k + 1; j < n; j += 8)
				{
					__m256 a_vec = _mm256_loadu_ps(&A[k][j]); // 加载A[k][j]到j+7的数据
					a_vec = _mm256_div_ps(a_vec, tmp_vec);   // 并行除法
					_mm256_storeu_ps(&A[k][j], a_vec);       // 存储结果回A[k][j]
				}
				A[k][k] = 1.0;
			}

			// 并行部分
#pragma omp barrier
#pragma omp for schedule(static)
			for (int i = k + 1; i < n; i++)
			{
				float tmp = A[i][k];
				__m256 tmp_vec = _mm256_set1_ps(tmp);

				for (int j = k + 1; j < n; j += 8)
				{
					__m256 a_vec = _mm256_loadu_ps(&A[i][j]);
					__m256 k_vec = _mm256_loadu_ps(&A[k][j]);
					k_vec = _mm256_mul_ps(tmp_vec, k_vec); // 并行乘法
					a_vec = _mm256_sub_ps(a_vec, k_vec);   // 并行减法
					_mm256_storeu_ps(&A[i][j], a_vec);     // 存储结果回A[i][j]
				}
				A[i][k] = 0;
			}
		}
	}
}


void f_omp_static_avx_division()
{
#pragma omp parallel num_threads(NUM_THREADS)
	{
		for (int k = 0; k < n; k++)
		{
			// 串行部分

			
				float tmp = A[k][k];
				__m256 tmp_vec = _mm256_set1_ps(tmp);  // 将tmp广播到8个浮点数
#pragma omp for schedule(static)
				for (int j = k + 1; j < n; j += 8)
				{
					__m256 a_vec = _mm256_loadu_ps(&A[k][j]); // 加载A[k][j]到j+7的数据
					a_vec = _mm256_div_ps(a_vec, tmp_vec);   // 并行除法
					_mm256_storeu_ps(&A[k][j], a_vec);       // 存储结果回A[k][j]
				}
				A[k][k] = 1.0;
			

			// 并行部分
#pragma omp for schedule(static)
			for (int i = k + 1; i < n; i++)
			{
				float tmp = A[i][k];
				__m256 tmp_vec = _mm256_set1_ps(tmp);

				for (int j = k + 1; j < n; j += 8)
				{
					__m256 a_vec = _mm256_loadu_ps(&A[i][j]);
					__m256 k_vec = _mm256_loadu_ps(&A[k][j]);
					k_vec = _mm256_mul_ps(tmp_vec, k_vec); // 并行乘法
					a_vec = _mm256_sub_ps(a_vec, k_vec);   // 并行减法
					_mm256_storeu_ps(&A[i][j], a_vec);     // 存储结果回A[i][j]
				}
				A[i][k] = 0;
			}
		}
	}
}
 
void f_omp_static_simd()
{
#pragma omp parallel num_threads(NUM_THREADS)
	{
		for (int k = 0; k < n; k++)
		{
			// 串行部分
#pragma omp single
			{
				float tmp = A[k][k];
				// 向量化内部循环
#pragma omp simd
				for (int j = k + 1; j < n; j++)
				{
					A[k][j] = A[k][j] / tmp;
				}
				A[k][k] = 1.0;
			}

			// 并行部分
#pragma omp for schedule(static)
			for (int i = k + 1; i < n; i++)
			{
				float tmp = A[i][k];
				// 向量化内部循环
#pragma omp simd
				for (int j = k + 1; j < n; j++)
				{
					A[i][j] = A[i][j] - tmp * A[k][j];
				}
				A[i][k] = 0;
			}

			// 离开for循环时，各个线程默认同步，进入下一行的处理
		}
	}
}

 
void f_omp_dynamic()
{
#pragma omp parallel num_threads(NUM_THREADS)

	for (int k = 0; k < n; k++)
	{
		//串行部分
#pragma omp single
		{
			float tmp = A[k][k];
			for (int j = k + 1; j < n; j++)
			{
				A[k][j] = A[k][j] / tmp;
			}
			A[k][k] = 1.0;
		}

		//并行部分
#pragma omp for schedule(dynamic, 80)
		for (int i = k + 1; i < n; i++)
		{
			float tmp = A[i][k];
			for (int j = k + 1; j < n; j++)
				A[i][j] = A[i][j] - tmp * A[k][j];
			A[i][k] = 0;
		}
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}

void f_omp_dynamic_avx()
{
#pragma omp parallel num_threads(NUM_THREADS)
	{
		for (int k = 0; k < n; k++)
		{
			// 串行部分
#pragma omp single
			{
				float tmp = A[k][k];
				__m256 tmp_vec = _mm256_set1_ps(tmp);  // 将tmp广播到8个浮点数

				for (int j = k + 1; j < n; j += 8)
				{
					__m256 a_vec = _mm256_loadu_ps(&A[k][j]); // 加载A[k][j]到j+7的数据
					a_vec = _mm256_div_ps(a_vec, tmp_vec);   // 并行除法
					_mm256_storeu_ps(&A[k][j], a_vec);       // 存储结果回A[k][j]
				}
				A[k][k] = 1.0;
			}

			// 并行部分
#pragma omp for schedule(dynamic, 200)
			for (int i = k + 1; i < n; i++)
			{
				float tmp = A[i][k];
				__m256 tmp_vec = _mm256_set1_ps(tmp);

				for (int j = k + 1; j < n; j += 8)
				{
					__m256 a_vec = _mm256_loadu_ps(&A[i][j]);
					__m256 k_vec = _mm256_loadu_ps(&A[k][j]);
					k_vec = _mm256_mul_ps(tmp_vec, k_vec); // 并行乘法
					a_vec = _mm256_sub_ps(a_vec, k_vec);   // 并行减法
					_mm256_storeu_ps(&A[i][j], a_vec);     // 存储结果回A[i][j]
				}
				A[i][k] = 0;
			}
		}
	}
}

void f_omp_guided()
{
#pragma omp parallel num_threads(NUM_THREADS)

	for (int k = 0; k < n; k++)
	{
		//串行部分
#pragma omp single
		{
			float tmp = A[k][k];
			for (int j = k + 1; j < n; j++)
			{
				A[k][j] = A[k][j] / tmp;
			}
			A[k][k] = 1.0;
		}

		//并行部分
#pragma omp for schedule(guided, 80)
		for (int i = k + 1; i < n; i++)
		{
			float tmp = A[i][k];
			for (int j = k + 1; j < n; j++)
				A[i][j] = A[i][j] - tmp * A[k][j];
			A[i][k] = 0;
		}
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}

void f_omp_guided_avx()
{
#pragma omp parallel num_threads(NUM_THREADS)
	{
		for (int k = 0; k < n; k++)
		{
			// 串行部分
#pragma omp single
			{
				float tmp = A[k][k];
				__m256 tmp_vec = _mm256_set1_ps(tmp);  // 将tmp广播到8个浮点数

				for (int j = k + 1; j < n; j += 8)
				{
					__m256 a_vec = _mm256_loadu_ps(&A[k][j]); // 加载A[k][j]到j+7的数据
					a_vec = _mm256_div_ps(a_vec, tmp_vec);   // 并行除法
					_mm256_storeu_ps(&A[k][j], a_vec);       // 存储结果回A[k][j]
				}
				A[k][k] = 1.0;
			}

			// 并行部分
#pragma omp for schedule(guided, 80)
			for (int i = k + 1; i < n; i++)
			{
				float tmp = A[i][k];
				__m256 tmp_vec = _mm256_set1_ps(tmp);

				for (int j = k + 1; j < n; j += 8)
				{
					__m256 a_vec = _mm256_loadu_ps(&A[i][j]);
					__m256 k_vec = _mm256_loadu_ps(&A[k][j]);
					k_vec = _mm256_mul_ps(tmp_vec, k_vec); // 并行乘法
					a_vec = _mm256_sub_ps(a_vec, k_vec);   // 并行减法
					_mm256_storeu_ps(&A[i][j], a_vec);     // 存储结果回A[i][j]
				}
				A[i][k] = 0;
			}
		}
	}
}







 



int main()
{
	init();
	double seconds = 0;
	long long head, tail, freq, noww;
	 
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

	 
	 for (int i=0;i<cycle;i++){
		ReStart();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);//开始计时
		f_ordinary();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);//结束计时
		 
	seconds += (tail-head) * 1000.0 / freq  ;//单位 ms
	 }
	cout << "f_ordinary: " << seconds/cycle << " ms" << endl;

	seconds = 0;
	for (int i = 0; i < cycle; i++) {
		ReStart();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);//开始计时
		f_omp_static();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);//结束计时
		seconds += (tail - head) * 1000.0 / freq;//单位 ms
	}
	cout << "f_omp_static: " << seconds / cycle << " ms" << endl;

	seconds = 0;
	for (int i = 0; i < cycle; i++) {
		ReStart();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);//开始计时
		f_omp_dynamic();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);//结束计时
		seconds += (tail - head) * 1000.0 / freq;//单位 ms
	}
	cout << "f_omp_dynamic: " << seconds/cycle << " ms" << endl;

	 
		 
	 
	seconds = 0;
	for (int i = 0; i < cycle; i++) {
		ReStart();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);//开始计时
		f_omp_guided();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);//结束计时
		seconds += (tail - head) * 1000.0 / freq;//单位 ms
	}
	cout << "f_omp_guided: " << seconds / cycle << " ms" << endl;
	 
	seconds = 0;
	for (int i = 0; i < cycle; i++) {
		ReStart();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);//开始计时
		f_omp_static_avx();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);//结束计时
		seconds += (tail - head) * 1000.0 / freq;//单位 ms
	}
		cout << "f_omp_static_avx: " << seconds / cycle << " ms" << endl;
		 
	
	seconds = 0;
	for (int i=0;i<cycle;i++){
	ReStart();
	QueryPerformanceCounter((LARGE_INTEGER*)&head);//开始计时
	f_omp_dynamic_avx();
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);//结束计时
	seconds += (tail - head) * 1000.0 / freq;//单位 ms
	
	}
	cout << "f_omp_dynamic_avx: " << seconds/cycle << " ms" << endl;

	
	seconds = 0;
	for (int i = 0; i < cycle; i++) {
		ReStart();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);//开始计时
		f_omp_guided_avx();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);//结束计时
		seconds += (tail - head) * 1000.0 / freq;//单位 ms
	}
	cout << "f_omp_guided_avx: " << seconds / cycle << " ms" << endl;
	
	
	seconds = 0;
	for (int i = 0; i < cycle; i++) {
		ReStart();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);//开始计时
		f_omp_static_avx_barrier();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);//结束计时
		seconds += (tail - head) * 1000.0 / freq;//单位 ms
	}
	cout << "f_omp_static_avx_barrier: " << seconds / cycle << " ms" << endl;
	
	seconds = 0;
	for (int i = 0; i < cycle; i++) {
		ReStart();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);//开始计时
		f_omp_static_avx_division();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);//结束计时
		seconds += (tail - head) * 1000.0 / freq;//单位 ms
	}
	cout << "f_omp_static_avx_division: " << seconds / cycle << " ms" << endl;


	seconds = 0;
	for (int i = 0; i < cycle; i++) {
		ReStart();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);//开始计时
		f_omp_static_simd();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);//结束计时
		seconds += (tail - head) * 1000.0 / freq;//单位 ms
	}
	cout << "f_omp_static_simd: " << seconds / cycle << " ms" << endl;

	 

}
