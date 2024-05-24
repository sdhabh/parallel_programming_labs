 #include <iostream>
#include <sys/time.h>
#include <random>
using namespace std;

#define ull unsigned long long int

const ull N = 1000000;
ull a[N];
int LOOP = 10;

void init()
{
random_device rd;
mt19937 gen(rd()); // 使用随机设备作为种子
uniform_int_distribution<int> dist(1, 100); // 生成1到100之间的随机整数
    for (ull i = 0; i < N; i++)
        a[i] = dist(gen);
}

void ordinary()
{
    struct timeval start;
    struct timeval end;
    gettimeofday(&start,NULL);
    for(int l=0;l<LOOP;l++)
    {
        // init();
        ull sum = 0;
        for (int i = 0; i < N - 1; i+=2)
            sum += a[i], sum += a[i+1]; 
    }
    gettimeofday(&end,NULL);
    cout<<"ordinary:"<<((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec))*1.0/1000/LOOP <<endl;
}

void optimize()
{
    struct timeval start;
    struct timeval end;
    gettimeofday(&start,NULL);
    for(int l=0;l<LOOP;l++)
    {
        ull sum1 = 0, sum2 = 0;
        for(int i=0;i<N-1; i+=2)
            sum1+=a[i],sum2+= a[i+1];
        ull sum = sum1 + sum2;
    }
    gettimeofday(&end,NULL);
    cout<<"optimize:"<<((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec))*1.0/1000/LOOP <<endl;
}


void recursion0(int s)//递归
  {
      if(s==1)
      {
          return;
      }
      else
      {
          for(int i=0;i<s/2;i++)
          {
              a[i]+=a[s-i-1];
              s=s/2;
              recursion0(s);
          }
      }
  }

void recursion()//递归
  {
    struct timeval start;
    struct timeval end;
    gettimeofday(&start,NULL);
    for(int l=0;l<LOOP;l++)
    {
     recursion0(N);
    }
    gettimeofday(&end,NULL);
    cout<<"recursion:"<<((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec))*1.0/1000/LOOP<<endl; 
  }

void doubleloop()//双重循环
  {
    struct timeval start;
    struct timeval end;
    gettimeofday(&start,NULL);
for(int l=0;l<LOOP;l++)
    {
      for(int i=N;i>=1;i=i/2)
      {
          for(int j=0;j<i/2;j++)
          {
              a[j]+=a[i-j-1];
          }
      }
    }
      gettimeofday(&end,NULL);
    cout<<"doubleloop:"<<((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec))*1.0/1000/LOOP<<endl;       
  }
  
int main()
{
    init();
    ordinary();
    optimize();
    recursion();
    doubleloop();
}
