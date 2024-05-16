#include <iostream>
#include <chrono>
#define N 1000
#define ROOP 500000

__global__ void add(int *culc_buff, int *buff)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    culc_buff[tid] += buff[tid];
}

int main(int argc, char **argv)
{
    int *send_buff_h, *send_buff_d, *culc_buff, *recv_buff_h;
    // メモリ確保
    cudaHostAlloc((void **)&send_buff_h, sizeof(int) * N, cudaHostAllocDefault);
    cudaMallocHost((void **)&recv_buff_h, sizeof(int) * N);

    for (int i = 0; i < ROOP; i++)
    {
        send_buff_h[i] = 1;
    }
    cudaMalloc((void **)&send_buff_d, sizeof(int) * N);
    cudaMalloc((void **)&culc_buff, sizeof(int) * N);
    cudaMemset(culc_buff, 0, sizeof(int) * N);

    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < ROOP; i++)
    {
        add<<<N, 1>>>(culc_buff, send_buff_h);
    }

    auto end = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "time : " << elapsed << "ms" << std::endl;

    cudaMemcpy(recv_buff_h, culc_buff, sizeof(int) * N, cudaMemcpyDeviceToHost);
    std::cout << "value : " << recv_buff_h[0] << std::endl;

    cudaFree(send_buff_d);
    cudaFree(culc_buff);
    cudaFreeHost(send_buff_h);
    cudaFreeHost(recv_buff_h);
}