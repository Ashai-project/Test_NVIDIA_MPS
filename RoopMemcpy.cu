#include <iostream>
#include <cuda.h>
#include <chrono>

#define ROOP 1000000
#define SIZE 10000
int main(int argc, char **argv)
{
    int *send_buff_h, *send_buff_d, *recv_buff_h, *recv_buff_d;
    // メモリ確保
    cudaMalloc((void **)&send_buff_d, sizeof(int) * SIZE);
    cudaMalloc((void **)&recv_buff_d, sizeof(int) * SIZE);
    cudaMallocHost((void **)&send_buff_h, sizeof(int) * SIZE);
    cudaMallocHost((void **)&recv_buff_h, sizeof(int) * SIZE);

    // init
    for (int i = 0; i < SIZE; i++)
    {
        send_buff_h[i] = i;
    }
    cudaMemcpy(send_buff_d, send_buff_h, sizeof(int) * SIZE, cudaMemcpyHostToDevice);

    // メモリコピー
    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < ROOP; i++)
    {
        cudaMemcpy(recv_buff_d, send_buff_d, sizeof(int) * SIZE, cudaMemcpyDeviceToDevice);
    }
    cudaMemcpy(recv_buff_h, recv_buff_d, sizeof(int) * SIZE, cudaMemcpyDeviceToHost);
    auto end = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "time : " << elapsed << "ms" << std::endl;
    std::cout << "recv[0] : " << recv_buff_h[0] << std::endl;
    std::cout << "recv[1] : " << recv_buff_h[1] << std::endl;
    std::cout << "recv[2] : " << recv_buff_h[2] << std::endl;

    // メモリ解放
    cudaFree(send_buff_d);
    cudaFree(recv_buff_d);
    cudaFreeHost(send_buff_h);
    cudaFreeHost(recv_buff_h);
}