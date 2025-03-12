#include <iostream>
#include <cuda_runtime.h>

int main() {
    // 初始化 CUDA
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA 初始化失败: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    // 获取 GPU 设备数量
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "未找到支持 CUDA 的 GPU 设备！" << std::endl;
        return 1;
    }

    // 遍历所有 GPU 设备
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);

        // 输出计算能力
        std::cout << "GPU " << i << ": " << deviceProp.name << std::endl;
        std::cout << "计算能力: sm_" << deviceProp.major<<"(major), sm_" << deviceProp.minor <<"(minor)"<< std::endl;
        std::cout << "maxBlockSize: " << deviceProp.maxBlockSize[0] << std::endl;
        std::cout << "-----------------------------" << std::endl;
    }

    return 0;
}