# include <arm_neon.h> // use Neon
# include <sys/time.h>

# include <iostream>
using namespace std;

const int n=1000;
float A[n][n];
float B[n][n];

float32x4_t va = vmovq_n_f32(0);
float32x4_t vx = vmovq_n_f32(0);
float32x4_t vaij = vmovq_n_f32(0);
float32x4_t vaik = vmovq_n_f32(0);
float32x4_t vakj = vmovq_n_f32(0);



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


void f_ordinary_cache()
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < i; j++)
        {
            B[j][i] = A[i][j];
            A[i][j] = 0; // 相当于原来的 A[i][k] = 0;
        }
    }

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
                A[i][j] = A[i][j] - B[k][i] * A[k][j];
            }
            //A[i][k] = 0;
        }
    }
}




void f_pro_optimized()
{
    for (int k = 0; k < n; k++)
    {
        float32x4_t vt = vmovq_n_f32(A[k][k]);

        for (int j = k + 1; j + 3 < n; j += 4)
        {
            float32x4_t va = vld1q_f32(&(A[k][j]));
            va = vdivq_f32(va, vt);
            vst1q_f32(&(A[k][j]), va);
        }

        for (int j = (n & ~3); j < n; j++)
        {
            A[k][j] *= 1.0 / A[k][k];
        }
        A[k][k] = 1.0;

        for (int i = k + 1; i < n; i++)
        {
            float32x4_t vaik = vmovq_n_f32(A[i][k]);

            for (int j = k + 1; j + 3 < n; j += 4)
            {
                float32x4_t vakj = vld1q_f32(&(A[k][j]));
                float32x4_t vaij = vld1q_f32(&(A[i][j]));
                float32x4_t vx = vmulq_f32(vakj, vaik);
                vaij = vsubq_f32(vaij, vx);
                vst1q_f32(&(A[i][j]), vaij);
            }

            for (int j = (n & ~3); j < n; j++)
            {
                A[i][j] -= A[i][k] * A[k][j];
            }

            A[i][k] = 0;
        }
    }
}


void f_pro_cache_optimized()
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < i; j++)
        {
            B[j][i] = A[i][j];
            A[i][j] = 0; // 相当于原来的 A[i][k] = 0;
        }
    }

    for (int k = 0; k < n; k++)
    {
        float32x4_t vt = vmovq_n_f32(A[k][k]);

        for (int j = k + 1; j + 3 < n; j += 4)
        {
            float32x4_t va = vld1q_f32(&(A[k][j]));
            va = vdivq_f32(va, vt);
            vst1q_f32(&(A[k][j]), va);
        }

        for (int j = (n & ~3); j < n; j++)
        {
            A[k][j] *= 1.0 / A[k][k];
        }
        A[k][k] = 1.0;

        for (int i = k + 1; i < n; i++)
        {
            float32x4_t vaik = vmovq_n_f32(B[k][i]);

            for (int j = k + 1; j + 3 < n; j += 4)
            {
                float32x4_t vakj = vld1q_f32(&(A[k][j]));
                float32x4_t vaij = vld1q_f32(&(A[i][j]));
                float32x4_t vx = vmulq_f32(vakj, vaik);
                vaij = vsubq_f32(vaij, vx);
                vst1q_f32(&(A[i][j]), vaij);
            }

            for (int j = (n & ~3); j < n; j++)
            {
                A[i][j] -= A[i][k] * A[k][j];
            }
        }
    }
}



 


 

void f_pro_alignment_optimized()
{
    for(int k = 0; k < n; k++)
    {
        float32x4_t vt = vmovq_n_f32(A[k][k]);
        int j = k + 1;
        
        // 对齐内存访问
        while((k * n + j) % 4 != 0)
        {
            A[k][j] = A[k][j] * 1.0 / A[k][k];
            j++;
        }

        // 循环展开和SIMD优化
        for(; j + 3 < n; j += 4)
        {
            va = vld1q_f32(&A[k][j]);
            va = vdivq_f32(va, vt);
            vst1q_f32(&A[k][j], va);
        }

        // 处理剩余部分
        for(; j < n; j++)
        {
            A[k][j] = A[k][j] * 1.0 / A[k][k];
        }
        
        A[k][k] = 1.0;

        for(int i = k + 1; i < n; i++)
        {
            vaik = vmovq_n_f32(A[i][k]);
            int j = k + 1;

            // 对齐内存访问
            while((i * n + j) % 4 != 0)
            {
                A[i][j] = A[i][j] - A[k][j] * A[i][k];
                j++;
            }

            // 循环展开和SIMD优化
            for(; j + 3 < n; j += 4)
            {
                vakj = vld1q_f32(&A[k][j]);
                vaij = vld1q_f32(&A[i][j]);
                vx = vmulq_f32(vakj, vaik);
                vaij = vsubq_f32(vaij, vx);
                vst1q_f32(&A[i][j], vaij);
            }

            // 处理剩余部分
            for(; j < n; j++)
            {
                A[i][j] = A[i][j] - A[k][j] * A[i][k];
            }
            
            A[i][k] = 0.0;
        }
    }
}


void getResult()
{
    for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cout << A[i][j] << " ";
		}
		cout << endl;
	}
}



int main()
{

    struct timeval head,tail;

    init();
    gettimeofday(&head, NULL);//开始计时
    f_pro();
    gettimeofday(&tail, NULL);//结束计时
    seconds = ((tail.tv_sec - head.tv_sec)*1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;//单位 ms
    cout<<"f_pro: "<<seconds<<" ms"<<endl;

    init();
    gettimeofday(&head, NULL);//开始计时
    f_pro_cache();
    gettimeofday(&tail, NULL);//结束计时
    seconds = ((tail.tv_sec - head.tv_sec)*1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;//单位 ms
    cout<<"f_pro_cache: "<<seconds<<" ms"<<endl;


    init();
    gettimeofday(&head, NULL);//开始计时
    f_ordinary();
    gettimeofday(&tail, NULL);//结束计时
    double seconds = ((tail.tv_sec - head.tv_sec)*1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;//单位 ms
    cout<<"f_ordinary: "<<seconds<<" ms"<<endl;


    init();
    gettimeofday(&head, NULL);//开始计时
    f_ordinary_cache();
    gettimeofday(&tail, NULL);//结束计时
    seconds = ((tail.tv_sec - head.tv_sec)*1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;//单位 ms
    cout<<"f_ordinary_cache: "<<seconds<<" ms"<<endl;

    init();
    gettimeofday(&head, NULL);//开始计时
    f_pro_alignment();
    gettimeofday(&tail, NULL);//结束计时
    seconds = ((tail.tv_sec - head.tv_sec)*1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;//单位 ms
    cout<<"f_pro_alignment: "<<seconds<<" ms"<<endl;


    init();
    gettimeofday(&head, NULL);//开始计时
    f_pro_division();
    gettimeofday(&tail, NULL);//结束计时
    seconds = ((tail.tv_sec - head.tv_sec)*1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;//单位 ms
    cout<<"f_pro_division: "<<seconds<<" ms"<<endl;


    init();
    gettimeofday(&head, NULL);//开始计时
    f_pro_elimination();
    gettimeofday(&tail, NULL);//结束计时
    seconds = ((tail.tv_sec - head.tv_sec)*1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;//单位 ms
    cout<<"f_pro_elimination: "<<seconds<<" ms"<<endl;

}
