#include <pmmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>
#include <nmmintrin.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <immintrin.h> //AVX、AVX2
//#include <windows.h>
#include <chrono>
using namespace std;



const int n = 3000;
float A[n][n];


void init()
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			A[i][j] = 0;
		}
		A[i][i] = 1.0;
		for (int j = i + 1; j < n; j++)
			A[i][j] = rand();
	}

	for (int k = 0; k < n; k++)
	{
		for (int i = k + 1; i < n; i++)
		{
			for (int j = 0; j < n; j++)
			{
				A[i][j] += A[k][j];
			}
		}
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


__m128 va, vt, vx, vaij, vaik, vakj;
void f_sse()
{
	for (int k = 0; k < n; k++)
	{
		// 将对角线元素加载到 SIMD 寄存器中
		vt = _mm_set1_ps(A[k][k]);

		// 使用 SIMD 处理每个块中的 4 个元素
		int j;
		for (j = k + 1; j + 4 <= n; j += 4)
		{
			// 将 4 个连续元素加载到 SIMD 寄存器中
			va = _mm_loadu_ps(&(A[k][j]));

			// 使用 SIMD 将每个元素除以对角线元素
			va = _mm_div_ps(va, vt);

			// 将结果存回内存
			_mm_storeu_ps(&(A[k][j]), va);
		}

		// 分别处理剩余的元素
		for (; j < n; j++)
		{
			A[k][j] /= A[k][k];
		}

		// 将对角线元素设置为 1
		A[k][k] = 1.0;

		// 对后续行进行高斯消元
		for (int i = k + 1; i < n; i++)
		{
			// 将乘数加载到 SIMD 寄存器中
			vaik = _mm_set1_ps(A[i][k]);

			// 使用 SIMD 处理每个块中的 4 个元素
			for (j = k + 1; j + 4 <= n; j += 4)
			{
				// 从两行加载元素到 SIMD 寄存器中
				vakj = _mm_loadu_ps(&(A[k][j]));
				vaij = _mm_loadu_ps(&(A[i][j]));

				// 使用 SIMD 进行乘法和减法
				vx = _mm_mul_ps(vakj, vaik);
				vaij = _mm_sub_ps(vaij, vx);

				// 将结果存回内存
				_mm_storeu_ps(&A[i][j], vaij);
			}

			// 分别处理剩余的元素
			for (; j < n; j++)
			{
				A[i][j] -= A[i][k] * A[k][j];
			}

			// 将乘数元素设置为 0
			A[i][k] = 0;
		}
	}
}




__m256 va2, vt2, vx2, vaij2, vaik2, vakj2;

void f_avx256()
{
	for (int k = 0; k < n; k++)
	{
		// 计算对角线元素的逆数
		float diagonal_inverse = 1.0f / A[k][k];

		// 对角线元素设置为 1
		A[k][k] = 1.0f;

		// 处理当前行除对角线元素外的元素
		for (int j = k + 1; j < n; j++)
		{
			A[k][j] *= diagonal_inverse;
		}

		// 处理后续行
		for (int i = k + 1; i < n; i++)
		{
			// 计算乘数
			float multiplier = A[i][k];

			// 设置乘数向量
			__m256 multiplier_vector = _mm256_set1_ps(multiplier);

			// 处理 j 向量化部分
			int j;
			for (j = k + 1; j + 8 <= n; j += 8)
			{
				// 加载 A[k][j] 和 A[i][j] 向量
				__m256 A_kj_vector = _mm256_loadu_ps(&(A[k][j]));
				__m256 A_ij_vector = _mm256_loadu_ps(&(A[i][j]));

				// 计算乘积
				__m256 product_vector = _mm256_mul_ps(A_kj_vector, multiplier_vector);

				// 计算差值
				__m256 result_vector = _mm256_sub_ps(A_ij_vector, product_vector);

				// 存储结果
				_mm256_storeu_ps(&(A[i][j]), result_vector);
			}

			// 处理剩余部分
			for (; j < n; j++)
			{
				A[i][j] -= multiplier * A[k][j];
			}

			// 将乘数元素设置为 0
			A[i][k] = 0.0f;
		}
	}
}



int main()
{

	 


	 
 


	init();

	chrono::high_resolution_clock::time_point startTime = chrono::high_resolution_clock::now();

	// 执行函数
	f_ordinary();

	// 计时器结束
	chrono::high_resolution_clock::time_point endTime = chrono::high_resolution_clock::now();
	auto timeDuration = chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();

	// 输出执行时间
	cout << "ordinary time: " << timeDuration << " microseconds" << endl;


	chrono::high_resolution_clock::time_point startTime1 = chrono::high_resolution_clock::now();

	// 执行函数
	f_sse();

	// 计时器结束
	chrono::high_resolution_clock::time_point endTime1 = chrono::high_resolution_clock::now();
	auto timeDuration1 = chrono::duration_cast<std::chrono::microseconds>(endTime1 - startTime1).count();

	// 输出执行时间
	cout << "f_sse time: " << timeDuration1 << " microseconds" << endl;


	chrono::high_resolution_clock::time_point startTime2 = chrono::high_resolution_clock::now();

	// 执行函数
	f_avx256();

	// 计时器结束
	chrono::high_resolution_clock::time_point endTime2 = chrono::high_resolution_clock::now();
	auto timeDuration2 = chrono::duration_cast<std::chrono::microseconds>(endTime2 - startTime2).count();

	// 输出执行时间
	cout << "f_avx256 time: " << timeDuration2 << " microseconds" << endl;


}
