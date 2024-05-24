#include <omp.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <windows.h>
#include <intrin.h>
#include <immintrin.h>
using namespace std;


 


unsigned int Act[43577][1363] = { 0 }; //消元子
unsigned int Pas[54274][1363] = { 0 };

const int Num = 1362;  //记录消元子/行是否为空
const int pasNum = 54274;
const int lieNum = 43577;

int cycle = 5;

//线程数定义
const int NUM_THREADS = 7; //线程数量


//全局变量定义，用于判断接下来是否进入下一轮
bool sign;

struct threadParam_t
{
    int t_id; // 线程 id
};




//消元子初始化
void init_A()
{
    //每个消元子第一个为1位所在的位置，就是它所在二维数组的行号
    //例如：消元子（561，...）由Act[561][]存放
    unsigned int a;
    ifstream infile("act3.txt");
    char fin[10000] = { 0 };
    int index;
    //从文件中提取行
    while (infile.getline(fin, sizeof(fin)))
    {
        std::stringstream line(fin);
        int biaoji = 0;

        //从行中提取单个的数字
        while (line >> a)
        {
            if (biaoji == 0)
            {
                //取每行第一个数字为行标
                index = a;
                biaoji = 1;
            }
            int k = a % 32;
            int j = a / 32;

            int temp = 1 << k;
            Act[index][Num - 1 - j] += temp;
            Act[index][Num] = 1;//设置该位置记录消元子该行是否为空，为空则是0，否则为1
        }
    }
}

//被消元行初始化
void init_P()
{
    //直接按照磁盘文件的顺序存，在磁盘文件是第几行，在数组就是第几行
    unsigned int a;
    ifstream infile("pas3.txt");
    char fin[10000] = { 0 };
    int index = 0;
    //从文件中提取行
    while (infile.getline(fin, sizeof(fin)))
    {
        std::stringstream line(fin);
        int biaoji = 0;

        //从行中提取单个的数字
        while (line >> a)
        {
            if (biaoji == 0)
            {
                //用Pas[ ][263]存放被消元行每行第一个数字，用于之后的消元操作
                Pas[index][Num] = a;
                biaoji = 1;
            }

            int k = a % 32;
            int j = a / 32;

            int temp = 1 << k;
            Pas[index][Num - 1 - j] += temp;
        }
        index++;
    }
}



 


void f_ordinary()
{
    int i;
    for (i = lieNum - 1; i - 8 >= -1; i -= 8)
    {
        //每轮处理8个消元子，范围：首项在 i-7 到 i

        for (int j = 0; j < pasNum; j++)
        {
            //看4535个被消元行有没有首项在此范围内的
            while (Pas[j][Num] <= i && Pas[j][Num] >= i - 7)
            {
                int index = Pas[j][Num];
                if (Act[index][Num] == 1)//消元子不为空
                {
                    //Pas[j][]和Act[（Pas[j][18]）][]做异或
                    for (int k = 0; k < Num; k++)
                    {
                        Pas[j][k] = Pas[j][k] ^ Act[index][k];
                    }

                    //更新Pas[j][18]存的首项值
                    //做完异或之后继续找这个数的首项，存到Pas[j][18]，若还在范围里会继续while循环
                    //找异或之后Pas[j][ ]的首项
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
                else//消元子为空
                {
                    //Pas[j][]来补齐消元子
                    for (int k = 0; k < Num; k++)
                        Act[index][k] = Pas[j][k];

                    Act[index][Num] = 1;//设置消元子非空
                    break;
                }

            }
        }
    }


    for (i = i + 8; i >= 0; i--)
    {
        //每轮处理1个消元子，范围：首项等于i

        for (int j = 0; j < pasNum; j++)
        {
            //看53个被消元行有没有首项等于i的
            while (Pas[j][Num] == i)
            {
                if (Act[i][Num] == 1)//消元子不为空
                {
                    //Pas[j][]和Act[i][]做异或
                    for (int k = 0; k < Num; k++)
                    {
                        Pas[j][k] = Pas[j][k] ^ Act[i][k];
                    }

                    //更新Pas[j][18]存的首项值
                    //做完异或之后继续找这个数的首项，存到Pas[j][18]，若还在范围里会继续while循环
                    //找异或之后Pas[j][ ]的首项
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
                else//消元子为空
                {
                    //Pas[j][]来补齐消元子
                    for (int k = 0; k < Num; k++)
                        Act[i][k] = Pas[j][k];

                    Act[i][Num] = 1;//设置消元子非空
                    break;
                }
            }
        }
    }
}



__m256 va_Pas2;
__m256 va_Act2;

void f_avx()
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

 
void f_omp_avx()
{
    int i;
#pragma omp parallel num_threads(NUM_THREADS)

    
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
                    int numIterations = (Num + 7) / 8; // 计算迭代次数，每次迭代处理8个元素

#pragma omp for schedule(dynamic, 80)
                    for (int iter = 0; iter < numIterations; iter++)
                    {
                         k = iter * 8; // 计算当前迭代的起始索引

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
 
                    int numIterations = (Num + 7) / 8; // 计算迭代次数，每次迭代处理8个元素

#pragma omp for schedule(dynamic, 80)
                    for (int iter = 0; iter < numIterations; iter++)
                    {
                        k = iter * 8; // 计算当前迭代的起始索引
                        
                       
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

 



int main()
{
    init_A();
    init_P();

    double seconds = 0;
    long long head, tail, freq, noww;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);



    for (int i = 0; i < cycle; i++) {
        
        QueryPerformanceCounter((LARGE_INTEGER*)&head);//开始计时
        f_ordinary();
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);//结束计时
        seconds += (tail - head) * 1000.0 / freq;//单位 ms
    }
    cout << "f_ordinary(): " << seconds / cycle << " ms" << endl;

    for (int i = 0; i < cycle; i++) {

        QueryPerformanceCounter((LARGE_INTEGER*)&head);//开始计时
        f_avx();
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);//结束计时
        seconds += (tail - head) * 1000.0 / freq;//单位 ms
    }
    cout << "f_avx: " << seconds / cycle << " ms" << endl;

    for (int i = 0; i < cycle; i++) {

        QueryPerformanceCounter((LARGE_INTEGER*)&head);//开始计时
        f_omp_avx();
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);//结束计时
        seconds += (tail - head) * 1000.0 / freq;//单位 ms
    }
    cout << "f_opm_avx: " << seconds / cycle << " ms" << endl;

}
