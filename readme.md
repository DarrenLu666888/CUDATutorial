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
