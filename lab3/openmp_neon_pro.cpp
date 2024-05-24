#include <omp.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <semaphore.h>
#include <sys/time.h>
# include <arm_neon.h> // use Neon
using namespace std;

 


unsigned int Act[43577][1363] = { 0 };
unsigned int Pas[54274][1363] = { 0 };

const int Num = 1362;
const int pasNum = 54274;
const int lieNum = 43577;



 
const int NUM_THREADS = 7; //线程数量












bool sign;//全局变量定义，用于判断接下来是否进入下一轮




struct threadParam_t
{
    int t_id; // 线程 id
};

 
void init_A()
{
     
    unsigned int a;
    ifstream infile("act3.txt");
    char fin[10000] = { 0 };
    int index;
    
    while (infile.getline(fin, sizeof(fin)))
    {
        std::stringstream line(fin);
        int biaoji = 0;

         
        while (line >> a)
        {
            if (biaoji == 0)
            {
                
                index = a;
                biaoji = 1;
            }
            int k = a % 32;
            int j = a / 32;

            int temp = 1 << k;
            Act[index][Num - 1 - j] += temp;
            Act[index][Num] = 1; 
        }
    }
}

 
void init_P()
{
     
    unsigned int a;
    ifstream infile("pas3.txt");
    char fin[10000] = { 0 };
    int index = 0;
   
    while (infile.getline(fin, sizeof(fin)))
    {
        std::stringstream line(fin);
        int biaoji = 0;

      
        while (line >> a)
        {
            if (biaoji == 0)
            {
                 
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
    for (i = lieNum-1; i - 8 >= -1; i -= 8)
    {
      

        for (int j = 0; j < pasNum; j++)
        {
          
            while (Pas[j][Num] <= i && Pas[j][Num] >= i - 7)
            {
                int index = Pas[j][Num];
                if (Act[index][Num] == 1) 
                {
                  
                    for (int k = 0; k < Num; k++)
                    {
                        Pas[j][k] = Pas[j][k] ^ Act[index][k];
                    }

                  
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
                 
                    for (int k = 0; k < Num; k++)
                    {
                        Pas[j][k] = Pas[j][k] ^ Act[i][k];
                    }

                   
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




void f_omp1()
{
    uint32x4_t va_Pas =  vmovq_n_u32(0);
    uint32x4_t va_Act =  vmovq_n_u32(0);
    bool sign;
    #pragma omp parallel num_threads(NUM_THREADS), private(va_Pas, va_Act)
    do
    {
        

        for (int i = lieNum - 1; i - 8 >= -1; i -= 8)
        {
           
            #pragma omp for schedule(static)
            for (int j = 0; j < pasNum; j++)
            {
              
                while (Pas[j][Num] <= i && Pas[j][Num] >= i - 7)
                {
                    int index = Pas[j][Num];

                    if (Act[index][Num] == 1) 
                    {
                        
                        int k;
                        for (k = 0; k+4 <= Num; k+=4)
                        {
                           
                            va_Pas =  vld1q_u32(& (Pas[j][k]));
                            va_Act =  vld1q_u32(& (Act[index][k]));

                            va_Pas = veorq_u32(va_Pas,va_Act);
                            vst1q_u32( &(Pas[j][k]) , va_Pas );
                        }

                        for( ; k<Num; k++ )
                        {
                            Pas[j][k] = Pas[j][k] ^ Act[index][k];
                        }
                       
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
                        break;
                    }
                }
            }
        }


        for (int i = lieNum%8-1; i >= 0; i--)
        {
          
            #pragma omp for schedule(static)
            for (int j = 0; j < pasNum; j++)
            {
              
                while (Pas[j][Num] == i)
                {
                    if (Act[i][Num] == 1) 
                    {
                       
                        int k;
                        for (k = 0; k+4 <= Num; k+=4)
                        {
                           
                            va_Pas =  vld1q_u32(& (Pas[j][k]));
                            va_Act =  vld1q_u32(& (Act[i][k]));

                            va_Pas = veorq_u32(va_Pas,va_Act);
                            vst1q_u32( &(Pas[j][k]) , va_Pas );
                        }

                        for( ; k<Num; k++ )
                        {
                            Pas[j][k] = Pas[j][k] ^ Act[i][k];
                        }
                       
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
                        break;
                    }
                }
            }
        }

        
    #pragma omp single
    {
        sign = false;
        for (int i = 0; i < pasNum; i++)
        {
           
            int temp = Pas[i][Num];
            if (temp == -1)
            {
           
                continue;
            }

           
            if (Act[temp][Num] == 0)
            {
               
                for (int k = 0; k < Num; k++)
                    Act[temp][k] = Pas[i][k];
               
                Pas[i][Num] = -1;
             
                sign = true;
            }
        }
    }

    }while (sign == true);

}










void f_omp2()
{
   
    #pragma omp parallel num_threads(NUM_THREADS)
 
    for (int i = lieNum-1; i - 8 >= -1; i -= 8)
    {
        for (int j = 0; j < pasNum; j++)
        {
            while (Pas[j][Num] <= i && Pas[j][Num] >= i - 7)
            {
                int index = Pas[j][Num];
                if (Act[index][Num] == 1)
                {

                    
                    #pragma omp for schedule(static)
                    for (int k = 0; k < Num; k++)
                    {
                        Pas[j][k] = Pas[j][k] ^ Act[i][k];
                    }

                 

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


    for (int i = lieNum%8-1; i >= 0; i--)
    {
        for (int j = 0; j < pasNum; j++)
        {
            while (Pas[j][Num] == i)
            {
                if (Act[i][Num] == 1)
                {

                   
                    #pragma omp for schedule(static)
                    for (int k = 0; k < Num; k++)
                    {
                        Pas[j][k] = Pas[j][k] ^ Act[i][k];
                    }

 


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

  
    struct timeval head,tail;
    double seconds;

    gettimeofday(&head, NULL); 
    f_ordinary();
    gettimeofday(&tail, NULL); 
    seconds = ((tail.tv_sec - head.tv_sec)*1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0; 
    cout<<"ordinary_time: "<<seconds<<" ms"<<endl;

     gettimeofday(&head, NULL); 
    f_omp1();
    gettimeofday(&tail, NULL); 
    seconds = ((tail.tv_sec - head.tv_sec)*1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0; 
    cout<<"omp1_time: "<<seconds<<" ms"<<endl;

     gettimeofday(&head, NULL); 
    f_omp2();
    gettimeofday(&tail, NULL); 
    seconds = ((tail.tv_sec - head.tv_sec)*1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0; 
    cout<<"omp2_time: "<<seconds<<" ms"<<endl;




}
