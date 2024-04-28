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

/*
unsigned int Act[2076][522] = { 0 };
unsigned int Pas[2076][522] = { 0 };

const int Num = 521;
const int pasNum = 14325;
const int lieNum = 2076;
*/

/*


unsigned int Act[23045][722] = { 0 };
unsigned int Pas[23045][722] = { 0 };

const int Num = 721;
const int pasNum = 14325;
const int lieNum = 23045;

*/
/*
unsigned int Act[64950][1188] = { 0 };
unsigned int Pas[64950][1188] = { 0 };

const int Num = 1187;
const int pasNum = 14921;
const int lieNum = 64950;
*/


unsigned int Act[113045][1622] = { 0 };
unsigned int Pas[113045][1622] = { 0 };

const int Num = 1621;
const int pasNum = 47381;
const int lieNum = 113045;



//消元子初始化
void init_A()
{
    unsigned int a;
    std::ifstream infile("act2.txt");

    // 从文件中逐行读取数据
    std::string line;
    int index;
    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        iss >> index;

        // 逐个读取数字并处理
        while (iss >> a)
        {
            int k = a % 32;
            int j = a / 32;
            int temp = 1 << k;

            // 直接修改数组值，避免重复访问
            Act[index][Num - 1 - j] += temp;
            Act[index][Num] = 1;
        }
    }
}

//被消元行初始化
void init_P()
{
    unsigned int a;
    std::ifstream infile("pas2.txt");

    // 从文件中逐行读取数据
    std::string line;
    int index = 0;
    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        iss >> Pas[index][Num]; // 将每行第一个数字存入 Pas[index][Num]

        // 逐个读取数字并处理
        while (iss >> a)
        {
            int k = a % 32;
            int j = a / 32;
            int temp = 1 << k;

            // 直接修改数组值，避免重复访问
            Pas[index][Num - 1 - j] += temp;
        }
        index++;
    }
}



void f_ordinary()
{
   for (int i = lieNum - 1; i >= -1; i -= 8)
    {
        for (int j = 0; j < pasNum; j++)
        {
            int num = Pas[j][Num];
            if (num <= i && num >= i - 7)
            {
                if (Act[num][Num] == 1)
                {
                    for (int k = 0; k < Num; k++)
                        Pas[j][k] ^= Act[num][k];

                    int S_num = 0;
                    for (int num = 0; num < Num; num++)
                    {
                        if (Pas[j][num] != 0)
                        {
                            unsigned int temp = Pas[j][num];
                            while (temp != 0)
                            {
                                temp = temp >> 1;
                                S_num++;
                            }
                            S_num += num * 32;
                            break;
                        }
                    }
                    Pas[j][Num] = S_num - 1;
                }
                else
                {
                    for (int k = 0; k < Num; k++)
                        Act[num][k] = Pas[j][k];

                    Act[num][Num] = 1;
                    break;
                }

            }
        }
    }

  for (int i = lieNum + 7; i >= 0; i--)
{
    for (int j = 0; j < pasNum; j++)
    {
        if (Pas[j][Num] == i)
        {
            if (Act[i][Num] == 1)
            {
                for (int k = 0; k < Num; k++)
                {
                    Pas[j][k] ^= Act[i][k];
                }

                int S_num = 0;
                for (int num = 0; num < Num; num++)
                {
                    if (Pas[j][num] != 0)
                    {
                        unsigned int temp = Pas[j][num];
                        while (temp != 0)
                        {
                            temp = temp >> 1;
                            S_num++;
                        }
                        S_num += num * 32;
                        break;
                    }
                }
                Pas[j][Num] = S_num - 1;
            }
            else
            {
                for (int k = 0; k < Num; k++)
                {
                    Act[i][k] = Pas[j][k];
                }

                Act[i][Num] = 1;
                break;
            }
        }
    }
}



__m128 va_Pas;
__m128 va_Act;


void f_sse()
{
    int i;
    for (i = lieNum - 1; i - 8 >= -1; i -= 8)
    {
        for (int j = 0; j < pasNum; j++)
        {
            while (Pas[j][Num] <= i && Pas[j][Num] >= i - 7)
            {
                int index = Pas[j][Num];
                if (Act[index][Num] == 1)
                {

                    //*******************并行优化部分***********************
                    //********
                    int k;
                    for (k = 0; k + 4 <= Num; k += 4)
                    {
                        //Pas[j][k] = Pas[j][k] ^ Act[index][k];
                        va_Pas = _mm_loadu_ps((float*)&(Pas[j][k]));
                        va_Act = _mm_loadu_ps((float*)&(Act[index][k]));

                        va_Pas = _mm_xor_ps(va_Pas, va_Act);
                        _mm_store_ss((float*)&(Pas[j][k]), va_Pas);
                    }

                    for (; k < Num; k++)
                    {
                        Pas[j][k] = Pas[j][k] ^ Act[index][k];
                    }
                    //*******
                    //********************并行优化部分***********************


                    int num = 0, S_num = 0;
                    for (num = 0; num < Num; num++)
                    {
                        if (Pas[j][num] != 0)
                        {
                            unsigned int temp = Pas[j][num];
                            while (temp != 0)
                            {
                                temp = temp >> 1;
                                S_num++;
                            }
                            S_num += num * 32;
                            break;
                        }
                    }
                    Pas[j][Num] = S_num - 1;

                }
                else
                {
                    for (int k = 0; k < Num; k++)
                        Act[index][k] = Pas[j][k];

                    Act[index][Num] = 1;
                    break;
                }
            }
        }
    }


    for (i = i + 8; i >= 0; i--)
    {
        for (int j = 0; j < pasNum; j++)
        {
            while (Pas[j][Num] == i)
            {
                if (Act[i][Num] == 1)
                {

                    //*******************并行优化部分***********************
                    //********
                    int k;
                    for (k = 0; k + 4 <= Num; k += 4)
                    {
                        //Pas[j][k] = Pas[j][k] ^ Act[i][k];
                        va_Pas = _mm_loadu_ps((float*)&(Pas[j][k]));
                        va_Act = _mm_loadu_ps((float*)&(Act[i][k]));
                        va_Pas = _mm_xor_ps(va_Pas, va_Act);
                        _mm_store_ss((float*)&(Pas[j][k]), va_Pas);
                    }

                    for (; k < Num; k++)
                    {
                        Pas[j][k] = Pas[j][k] ^ Act[i][k];
                    }
                    //*******
                    //********************并行优化部分***********************



                    int num = 0, S_num = 0;
                    for (num = 0; num < Num; num++)
                    {
                        if (Pas[j][num] != 0)
                        {
                            unsigned int temp = Pas[j][num];
                            while (temp != 0)
                            {
                                temp = temp >> 1;
                                S_num++;
                            }
                            S_num += num * 32;
                            break;
                        }
                    }
                    Pas[j][Num] = S_num - 1;

                }
                else
                {
                    for (int k = 0; k < Num; k++)
                        Act[i][k] = Pas[j][k];

                    Act[i][Num] = 1;
                    break;
                }
            }
        }
    }

}


__m256 va_Pas2;
__m256 va_Act2;

void f_avx256()
{
    int i;
    for (i = lieNum - 1; i - 8 >= -1; i -= 8)
    {
        for (int j = 0; j < pasNum; j++)
        {
            while (Pas[j][Num] <= i && Pas[j][Num] >= i - 7)
            {
                int index = Pas[j][Num];
                if (Act[index][Num] == 1)
                {

                    //*******************并行优化部分***********************
                    //********
                    int k;
                    for (k = 0; k + 8 <= Num; k += 8)
                    {
                        //Pas[j][k] = Pas[j][k] ^ Act[index][k];
                        va_Pas2 = _mm256_loadu_ps((float*)&(Pas[j][k]));
                        va_Act2 = _mm256_loadu_ps((float*)&(Act[index][k]));

                        va_Pas2 = _mm256_xor_ps(va_Pas2, va_Act2);
                        _mm256_storeu_ps((float*)&(Pas[j][k]), va_Pas2);
                    }

                    for (; k < Num; k++)
                    {
                        Pas[j][k] = Pas[j][k] ^ Act[index][k];
                    }
                    //*******
                    //********************并行优化部分***********************


                    int num = 0, S_num = 0;
                    for (num = 0; num < Num; num++)
                    {
                        if (Pas[j][num] != 0)
                        {
                            unsigned int temp = Pas[j][num];
                            while (temp != 0)
                            {
                                temp = temp >> 1;
                                S_num++;
                            }
                            S_num += num * 32;
                            break;
                        }
                    }
                    Pas[j][Num] = S_num - 1;

                }
                else
                {
                    for (int k = 0; k < Num; k++)
                        Act[index][k] = Pas[j][k];

                    Act[index][Num] = 1;
                    break;
                }
            }
        }
    }


    for (i = i + 8; i >= 0; i--)
    {
        for (int j = 0; j < pasNum; j++)
        {
            while (Pas[j][Num] == i)
            {
                if (Act[i][Num] == 1)
                {

                    //*******************并行优化部分***********************
                    //********
                    int k;
                    for (k = 0; k + 8 <= Num; k += 8)
                    {
                        //Pas[j][k] = Pas[j][k] ^ Act[i][k];
                        va_Pas2 = _mm256_loadu_ps((float*)&(Pas[j][k]));
                        va_Act2 = _mm256_loadu_ps((float*)&(Act[i][k]));
                        va_Pas2 = _mm256_xor_ps(va_Pas2, va_Act2);
                        _mm256_storeu_ps((float*)&(Pas[j][k]), va_Pas2);
                    }

                    for (; k < Num; k++)
                    {
                        Pas[j][k] = Pas[j][k] ^ Act[i][k];
                    }
                    //*******
                    //********************并行优化部分***********************



                    int num = 0, S_num = 0;
                    for (num = 0; num < Num; num++)
                    {
                        if (Pas[j][num] != 0)
                        {
                            unsigned int temp = Pas[j][num];
                            while (temp != 0)
                            {
                                temp = temp >> 1;
                                S_num++;
                            }
                            S_num += num * 32;
                            break;
                        }
                    }
                    Pas[j][Num] = S_num - 1;

                }
                else
                {
                    for (int k = 0; k < Num; k++)
                        Act[i][k] = Pas[j][k];

                    Act[i][Num] = 1;
                    break;
                }
            }
        }
    }
}


__m512 va_Pas3;
__m512 va_Act3;

void f_avx512()
{
    int i;
    for (i = lieNum - 1; i - 8 >= -1; i -= 8)
    {
        for (int j = 0; j < pasNum; j++)
        {
            while (Pas[j][Num] <= i && Pas[j][Num] >= i - 7)
            {
                int index = Pas[j][Num];
                if (Act[index][Num] == 1)
                {

                    //*******************并行优化部分***********************
                    //********
                    int k;
                    for (k = 0; k + 16 <= Num; k += 16)
                    {
                        //Pas[j][k] = Pas[j][k] ^ Act[index][k];
                        va_Pas3 = _mm512_loadu_ps((float*)&(Pas[j][k]));
                        va_Act3 = _mm512_loadu_ps((float*)&(Act[index][k]));

                        va_Pas3 = _mm512_xor_ps(va_Pas3, va_Act3);
                        _mm512_storeu_ps((float*)&(Pas[j][k]), va_Pas3);
                    }

                    for (; k < Num; k++)
                    {
                        Pas[j][k] = Pas[j][k] ^ Act[index][k];
                    }
                    //*******
                    //********************并行优化部分***********************


                    int num = 0, S_num = 0;
                    for (num = 0; num < Num; num++)
                    {
                        if (Pas[j][num] != 0)
                        {
                            unsigned int temp = Pas[j][num];
                            while (temp != 0)
                            {
                                temp = temp >> 1;
                                S_num++;
                            }
                            S_num += num * 32;
                            break;
                        }
                    }
                    Pas[j][Num] = S_num - 1;

                }
                else
                {
                    for (int k = 0; k < Num; k++)
                        Act[index][k] = Pas[j][k];

                    Act[index][Num] = 1;
                    break;
                }
            }
        }
    }


    for (i = i + 8; i >= 0; i--)
    {
        for (int j = 0; j < pasNum; j++)
        {
            while (Pas[j][Num] == i)
            {
                if (Act[i][Num] == 1)
                {

                    //*******************并行优化部分***********************
                    //********
                    int k;
                    for (k = 0; k + 16 <= Num; k += 16)
                    {
                        //Pas[j][k] = Pas[j][k] ^ Act[i][k];
                        va_Pas3 = _mm512_loadu_ps((float*)&(Pas[j][k]));
                        va_Act3 = _mm512_loadu_ps((float*)&(Act[i][k]));
                        va_Pas3 = _mm512_xor_ps(va_Pas3, va_Act3);
                        _mm512_storeu_ps((float*)&(Pas[j][k]), va_Pas3);
                    }

                    for (; k < Num; k++)
                    {
                        Pas[j][k] = Pas[j][k] ^ Act[i][k];
                    }
                    //*******
                    //********************并行优化部分***********************



                    int num = 0, S_num = 0;
                    for (num = 0; num < Num; num++)
                    {
                        if (Pas[j][num] != 0)
                        {
                            unsigned int temp = Pas[j][num];
                            while (temp != 0)
                            {
                                temp = temp >> 1;
                                S_num++;
                            }
                            S_num += num * 32;
                            break;
                        }
                    }
                    Pas[j][Num] = S_num - 1;

                }
                else
                {
                    for (int k = 0; k < Num; k++)
                        Act[i][k] = Pas[j][k];

                    Act[i][Num] = 1;
                    break;
                }
            }
        }
    }
}






int main()
{
    
 
    init_A();
    init_P();


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

 
    chrono::high_resolution_clock::time_point startTime3 = chrono::high_resolution_clock::now();

    // 执行函数
    f_avx512();

    // 计时器结束
    chrono::high_resolution_clock::time_point endTime3 = chrono::high_resolution_clock::now();
    auto timeDuration3 = chrono::duration_cast<std::chrono::microseconds>(endTime3 - startTime3).count();

    // 输出执行时间
    cout << "f_avx512 time: " << timeDuration3 << " microseconds" << endl;
    
}
