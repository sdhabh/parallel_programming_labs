# include <iostream>
# include <pthread.h>
# include <sys/time.h>
# include <arm_neon.h> // use Neon
using namespace std;

const int n = 1000;
float A[n][n];
int NUM_THREADS = 7; //工作线程数量
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
			A[i][j] = rand() % 100;
	}

	for (int i = 0; i < n; i++)
	{
		int k1 = rand() % n;
		int k2 = rand() % n;
		for (int j = 0; j < n; j++)
		{
			A[i][j] += A[0][j];
			A[k1][j] += A[k2][j];
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


struct threadParam_t1
{
	int k; //消去的轮次
	int t_id; // 线程 id
};

void* threadFunc1(void* param)
{

	float32x4_t vx = vmovq_n_f32(0);
	float32x4_t vaij = vmovq_n_f32(0);
	float32x4_t vaik = vmovq_n_f32(0);
	float32x4_t vakj = vmovq_n_f32(0);


	threadParam_t1* p = (threadParam_t1*)param;
	int k = p->k; //消去的轮次
	int t_id = p->t_id; //线程编号
	int i = k + t_id + 1; //获取自己的计算任务
	for (int m = k + 1 + t_id; m < n; m += worker_count)
	{
		vaik = vmovq_n_f32(A[i][k]);
		int j;
		for (j = k + 1; j + 4 <= n; j += 4)
		{
			vakj = vld1q_f32(&(A[k][j]));
			vaij = vld1q_f32(&(A[i][j]));
			vx = vmulq_f32(vakj, vaik);
			vaij = vsubq_f32(vaij, vx);
			vst1q_f32(&A[i][j], vaij);
		}
		for (; j < n; j++)
			A[m][j] = A[m][j] - A[m][k] * A[k][j];

		A[m][k] = 0;
	}


	pthread_exit(NULL);

}

void pthread_dynamic() {

	float32x4_t va = vmovq_n_f32(0);
	float32x4_t vt = vmovq_n_f32(0);

	for (int k = 0; k < n; k++)
	{
		vt = vmovq_n_f32(A[k][k]);
		int j;
		for (j = k + 1; j + 4 <= n; j += 4)
		{
			va = vld1q_f32(&(A[k][j]));
			va = vdivq_f32(va, vt);
			vst1q_f32(&(A[k][j]), va);
		}

		for (; j < n; j++)
		{
			A[k][j] = A[k][j] * 1.0 / A[k][k];

		}
		A[k][k] = 1.0;

		//创建工作线程，进行消去操作

		pthread_t* handles = new pthread_t[worker_count];// 创建对应的 Handle
		threadParam_t1* param = new threadParam_t1[worker_count];// 创建对应的线程数据结构

		//分配任务
		for (int t_id = 0; t_id < worker_count; t_id++)
		{
			param[t_id].k = k;
			param[t_id].t_id = t_id;
		}
		//创建线程
		for (int t_id = 0; t_id < worker_count; t_id++)
			pthread_create(&handles[t_id], NULL, threadFunc1, (void*)&param[t_id]);

		//主线程挂起等待所有的工作线程完成此轮消去工作
		for (int t_id = 0; t_id < worker_count; t_id++)
			pthread_join(handles[t_id], NULL);

	}
}


struct threadParam_t2
{
	int t_id; //线程 id
};

//信号量定义
sem_t sem_main;
sem_t* sem_workerstart = new sem_t[NUM_THREADS]; // 每个线程有自己专属的信号量
sem_t* sem_workerend = new sem_t[NUM_THREADS];



//线程函数定义
void* threadFunc2(void* param)
{

	float32x4_t vx = vmovq_n_f32(0);
	float32x4_t vaij = vmovq_n_f32(0);
	float32x4_t vaik = vmovq_n_f32(0);
	float32x4_t vakj = vmovq_n_f32(0);


	threadParam_t2* p = (threadParam_t2*)param;
	int t_id = p->t_id;
	for (int k = 0; k < n; k++)
	{
		sem_wait(&sem_workerstart[t_id]); // 阻塞，等待主线完成除法操作（操作自己专属的信号量）
		//循环划分任务
		for (int i = k + 1 + t_id; i < n; i += NUM_THREADS)
		{
			//消去
			vaik = vmovq_n_f32(A[i][k]);
			int j;
			for (j = k + 1; j + 4 <= n; j += 4)
			{
				vakj = vld1q_f32(&(A[k][j]));
				vaij = vld1q_f32(&(A[i][j]));
				vx = vmulq_f32(vakj, vaik);
				vaij = vsubq_f32(vaij, vx);
				vst1q_f32(&A[i][j], vaij);
			}
			for (; j < n; j++)
				A[i][j] = A[i][j] - A[i][k] * A[k][j];

			A[i][k] = 0.0;
		}

		sem_post(&sem_main); // 唤醒主线程
		sem_wait(&sem_workerend[t_id]); //阻塞，等待主线程唤醒进入下一轮
	}

	pthread_exit(NULL);

}

void pthread_static_sem() {
	//初始化信号量
	sem_init(&sem_main, 0, 0);
	for (int i = 0; i < NUM_THREADS; ++i)
	{
		sem_init(&sem_workerstart[i], 0, 0);
		sem_init(&sem_workerend[i], 0, 0);
	}

	//创建线程
	pthread_t* handles = new pthread_t[NUM_THREADS];// 创建对应的 Handle
	threadParam_t2* param = new threadParam_t2[NUM_THREADS];// 创建对应的线程数据结构
	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
	{
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], NULL, threadFunc2, (void*)&param[t_id]);
	}


	float32x4_t va = vmovq_n_f32(0);
	float32x4_t vt = vmovq_n_f32(0);

	for (int k = 0; k < n; ++k)
	{
		//主线程做除法操作
		vt = vmovq_n_f32(A[k][k]);
		int j;
		for (j = k + 1; j + 4 <= n; j += 4)
		{
			va = vld1q_f32(&(A[k][j]));
			va = vdivq_f32(va, vt);
			vst1q_f32(&(A[k][j]), va);
		}

		for (; j < n; j++)
			A[k][j] = A[k][j] * 1.0 / A[k][k];

		A[k][k] = 1.0;


		//开始唤醒工作线程
		for (int t_id = 0; t_id < NUM_THREADS; ++t_id)
			sem_post(&sem_workerstart[t_id]);

		//主线程睡眠（等待所有的工作线程完成此轮消去任务）
		for (int t_id = 0; t_id < NUM_THREADS; ++t_id)
			sem_wait(&sem_main);

		// 主线程再次唤醒工作线程进入下一轮次的消去任务
		for (int t_id = 0; t_id < NUM_THREADS; ++t_id)
			sem_post(&sem_workerend[t_id]);

	}

	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
		pthread_join(handles[t_id], NULL);

	//销毁所有信号量
	sem_destroy(&sem_main);
	sem_destroy(sem_workerstart);
	sem_destroy(sem_workerend);
}

//barrier 定义
pthread_barrier_t barrier_Divsion;
pthread_barrier_t barrier_Elimination;


//线程函数定义
void* threadFunc3(void* param)
{
	float32x4_t va = vmovq_n_f32(0);
	float32x4_t vx = vmovq_n_f32(0);
	float32x4_t vaij = vmovq_n_f32(0);
	float32x4_t vaik = vmovq_n_f32(0);
	float32x4_t vakj = vmovq_n_f32(0);
	float32x4_t vt = vmovq_n_f32(0);


	threadParam_t2* p = (threadParam_t2*)param;
	int t_id = p->t_id;

	for (int k = 0; k < n; ++k)
	{

		vt = vmovq_n_f32(A[k][k]);
		if (t_id == 0)
		{
			int j;
			for (j = k + 1; j + 4 <= n; j += 4)
			{
				va = vld1q_f32(&(A[k][j]));
				va = vdivq_f32(va, vt);
				vst1q_f32(&(A[k][j]), va);
			}

			for (; j < n; j++)
			{
				A[k][j] = A[k][j] * 1.0 / A[k][k];
			}
			A[k][k] = 1.0;
		}

		//第一个同步点
		pthread_barrier_wait(&barrier_Divsion);

		//循环划分任务
		for (int i = k + 1 + t_id; i < n; i += NUM_THREADS)
		{
			//消去
			vaik = vmovq_n_f32(A[i][k]);
			int j;
			for (j = k + 1; j + 4 <= n; j += 4)
			{
				vakj = vld1q_f32(&(A[k][j]));
				vaij = vld1q_f32(&(A[i][j]));
				vx = vmulq_f32(vakj, vaik);
				vaij = vsubq_f32(vaij, vx);
				vst1q_f32(&A[i][j], vaij);
			}
			for (; j < n; j++)
				A[i][j] = A[i][j] - A[i][k] * A[k][j];

			A[i][k] = 0;
		}
		// 第二个同步点
		pthread_barrier_wait(&barrier_Elimination);

	}
	pthread_exit(NULL);
}

void pthread_static_barrier() {
	//初始化barrier
	pthread_barrier_init(&barrier_Divsion, NULL, NUM_THREADS);
	pthread_barrier_init(&barrier_Elimination, NULL, NUM_THREADS);


	//创建线程
	pthread_t* handles = new pthread_t[NUM_THREADS];// 创建对应的 Handle
	threadParam_t2* param = new threadParam_t2[NUM_THREADS];// 创建对应的线程数据结构
	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
	{
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], NULL, threadFunc3, (void*)&param[t_id]);
	}


	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
		pthread_join(handles[t_id], NULL);

	//销毁所有的 barrier
	pthread_barrier_destroy(&barrier_Divsion);
	pthread_barrier_destroy(&barrier_Elimination);
}



int main()
{
	init();
	struct timeval head, tail;
	double seconds;
	gettimeofday(&head, NULL);//开始计时
	pthread_dynamic();
	gettimeofday(&tail, NULL);//结束计时
	seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;//单位 ms
	cout << "pthread_dynamic: " << seconds << " ms" << endl;

	gettimeofday(&head, NULL);//开始计时
	pthread_static_sem();
	gettimeofday(&tail, NULL);//结束计时
	seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;//单位 ms
	cout << "pthread_static_sem: " << seconds << " ms" << endl;

	gettimeofday(&head, NULL);//开始计时
	pthread_static_barrier();
	gettimeofday(&tail, NULL);//结束计时
	seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;//单位 ms
	cout << "pthread_static_barrier: " << seconds << " ms" << endl;



}
