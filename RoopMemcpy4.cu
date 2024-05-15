#include <iostream>
#include <cuda.h>
#include <chrono>

#define ROOP 3000000
#define SIZE 10000
#define PARA 13
int main(int argc, char **argv)
{
    int **send_buff_h, **send_buff_d, **recv_buff_h, **recv_buff_d;
    cudaStream_t *st;
    cudaMallocHost((void **)&recv_buff_h, sizeof(size_t) * PARA);
    cudaMallocHost((void **)&send_buff_d, sizeof(size_t) * PARA);
    cudaMallocHost((void **)&recv_buff_d, sizeof(size_t) * PARA);
    cudaMallocHost((void **)&send_buff_h, sizeof(size_t) * PARA);
    cudaMallocHost((void **)&st, sizeof(cudaStream_t) * PARA);
    // メモリ確保
    for (int i = 0; i < PARA; i++)
    {
        cudaMalloc((void **)&send_buff_d[i], sizeof(int) * SIZE);
        cudaMalloc((void **)&recv_buff_d[i], sizeof(int) * SIZE);
        cudaMallocHost((void **)&send_buff_h[i], sizeof(int) * SIZE);
        cudaMallocHost((void **)&recv_buff_h[i], sizeof(int) * SIZE);
        cudaStreamCreate(&st[i]);
    }

    // init
    for (int j = 0; j < PARA; j++)
    {
        for (int i = 0; i < SIZE; i++)
        {
            send_buff_h[j][i] = i;
        }
        cudaMemcpy(send_buff_d[j], send_buff_h[j], sizeof(int) * SIZE, cudaMemcpyHostToDevice);
    }

    // メモリコピー
    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < ROOP; i++)
    {
        for (int j = 0; j < PARA; j++)
        {
            cudaMemcpyAsync(recv_buff_d[j], send_buff_d[j], sizeof(int) * SIZE, cudaMemcpyDeviceToDevice, st[j]);
        }
    }

    for (int j = 0; j < PARA; j++)
    {
        cudaStreamSynchronize(st[j]);
        cudaMemcpy(recv_buff_h[j], recv_buff_d[j], sizeof(int) * SIZE, cudaMemcpyDeviceToHost);
    }
    auto end = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "time : " << elapsed << "ms" << std::endl;

    // メモリ解放
    for (int j = 0; j < PARA; j++)
    {
        cudaFree(send_buff_d[j]);
        cudaFree(recv_buff_d[j]);
        cudaFreeHost(send_buff_h[j]);
        cudaFreeHost(recv_buff_h[j]);
    }
    cudaFreeHost(send_buff_d);
    cudaFreeHost(recv_buff_d);
    cudaFreeHost(send_buff_h);
    cudaFreeHost(recv_buff_h);
}