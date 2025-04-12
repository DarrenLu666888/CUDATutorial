## 5_reduce

见 5_reduce/readme.md

## 6_warp_level_reduce

[warp shuffle functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=shuffle#warp-shuffle-functions)

讲真，官方文档虽然是英文，但可读性还是很好的

1. warp是在block里面的，不过是硬件层面的，block属于逻辑层面的
2. warp shuffle functions/intrinsics 所有active的threads同步进行，不需要syncwarp
3. 这个版本并没有用smem，而是直接用sum，放在每个线程的私有寄存器上
   1. 怎么确保这个sum一定是在私有寄存器上？：[CUDA变量存储与原子操作 - z_ining - 博客园](https://www.cnblogs.com/ining/p/17110813.html) 核函数符合存储在寄存器中但不能进入被核函数分配的寄存器空间中的变量将存储在本地内存-local mem中，sum是第一个定义的局部变量，因此**应该**会存储在私有寄存器中
   2. 同1. ，如何查看某个变量存储在哪里呢？要看中间生成的PTX代码哩，[CUDA 基础 - 03 访存- register、global、local - 知乎](https://zhuanlan.zhihu.com/p/565199964)；
      `nvcc -ptx -arch=sm_86 6_warp_level_reduce.cu -o 6_warp_level_reduce.ptx` 看了确实
   3. 那每个线程有多少个私有寄存器呢？每个SM又有多少个shared memory？SM架构图可以看出smem的大小，也能看出一个warp的RegisterFile的大小，但一个线程的私有寄存器是多少呢？32个线程均分，还是编译器按照需求来分配（分配的时候lock给该thread?)
      1. 首先，每个thread可用register数是可以设置的[1. Introduction — CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=Register%2520Files#maximum-number-of-registers-per-thread)
      2. 由于SIMT执行模型，同一block内的所有线程（包括同一warp）必须使用 **相同数量的寄存器** ，该数目由编译时确定。因此，一个warp内的每个线程私有寄存器数量完全一致，但具体数值由代码逻辑和上述约束决定。
4. 最后把每个wrap的计算结果放到smem上后，再用第一个warp从smem加载到私用寄存器上再来一次warpShuffle，很nice
5. warp内共享数据方式：[CUDA 中的 warp 理解 - 知乎](https://zhuanlan.zhihu.com/p/22702033325)
   1. smem
   2. warp shuffle

## 7_histogram

因为不同线程可能会往gmem的同一个地方写入数据，因此需要原子加，但每次都是从gmem的hist_data中读取数据，然后往gmem的bin_data中写入数据，开销太大，需要优化（7_1_histogram）：

1. 用smem存储一个block中的bin_data统计结果，然后再原子加到gmem的bin_data中

个人优化思考（其实reduce中很多思路都可以用到这里）：

1. 一个block里就256个thread用了smem的256个int，太浪费了，但数据的范围是0~255，故可以考虑增加blockSize

```
./7_1_histogram_bs1024 && ./7_1_histogram_bs256 
the ans is right
histogram + shared_mem + multi_value latency = 1.289536 ms
the ans is right
histogram + shared_mem + multi_value latency = 1.686624 ms
```

先想到这里

## 8_copy_if

[Warp Vote Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=__activemask#warp-vote-functions)

[ffs](https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ffs.html)

[proc](https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_popc.html)

[lanemask_lt](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=lanemask_lt#special-registers-lanemask-lt)

[Warp Shuffle Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=__shfl_sync#warp-shuffle-functions)

通过warp level的一些api操作，来计算出global offset，挺新奇的，尤其是ptx的使用和rank的计算

## 9_gelu

这里有点问题，应该对分配的device地址进行内存对齐，而不是host地址！

对齐和合并访问可以看《CUDA C编程权威指南》

剩下的就是可以利用一些新的硬件特性，比如硬件层面上对half的支持：

[4. Half Precision Intrinsics — CUDA Math API Reference Manual 12.8 documentation](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__HALF.html#)


## 10_fused op

这个给了算子融合的简单示例，并用了

1. 结构体来将一连串重复的计算封装起来，便于kernel使用不同的fused模式
2. shared mem来存储bias
3. 向量化访存

## 11_softmax

数据规模：

row=1000， col=1024

Grid(1, 125)

Block(32, 8)

共125*8=1000个warp，每个warp32个线程，共32000个线程，每个线程1024/32=32个数据

切分方式：

每rows_per_thread行一个warp负责，一个warp共warp_width=32个线程，32个线程将rows_per_thread行的所有列对应的数据进行切分，每个线程负责cols_per_thread*rows_per_thread个数据；

所以分的块（一个线程处理的数据）是rows_per_thread*cols_per_thread个数据

对于每个线程，每一行有cols_per_thread个数据，对这些数据可以采用向量化读取的方式


## 12_measure_GPU_peak_perf

评估GPU的算力，最重要的是要去除访存的时间，仅仅考虑计算的时间

Q1: why use 4 fma instruction to get GPU peak performance?

A1: we use >2(3or4) fma instruction to hide for loop comparsion and addition instruction overhead

Q2: why use 4 dependant fma instruction to get GPU peak performance, can we use 4 independant ones?

A2: yes, we can use 2/3/4 independant ones

[NVIAIA A800显卡详细配置-GPU算力平台](http://www.suanlicloud.com/article/22.html)


## 13_CUDA_stream
