### nvprof到ncu迁移

```
======== Warning: Skipping profiling on device 0 since profiling is not supported on devices with compute capability 7.5 and higher.
                  Use NVIDIA Nsight Compute for GPU profiling and NVIDIA Nsight Systems for GPU tracing and CPU sampling.
                  Refer https://developer.nvidia.com/tools-overview for more details.
```

#### `branch_efficiency` -> `smsp__sass_average_branch_targets_threads_uniform.pct`

用branch_efficiency指标检查内核的线程分化影响

nvprof

```
$ nvprof --metrics branch_efficiency ./simpleDivergence
```

ncu

```
$ sudo /usr/local/NVIDIA-Nsight-Compute/ncu --metrics smsp__sass_average_branch_targets_threads_uniform.pct ./simpleDivergence
==PROF== Connected to process 20936 (/home/hexu/git/Professional.CUDA.C.Programming/CodeSamples/chapter03/simpleDivergence)
/home/hexu/git/Professional.CUDA.C.Programming/CodeSamples/chapter03/./simpleDivergence using Device 0: NVIDIA GeForce RTX 2060
Data size 64 Execution Configure (block 64 grid 1)
==PROF== Profiling "warmingup(float*)" - 1: 0%....50%....100% - 1 pass
warmup      <<<    1   64 >>> elapsed 0.119083 sec
==PROF== Profiling "mathKernel1(float*)" - 2: 0%....50%....100% - 1 pass
mathKernel1 <<<    1   64 >>> elapsed 0.008643 sec
==PROF== Profiling "mathKernel2(float*)" - 3: 0%....50%....100% - 1 pass
mathKernel2 <<<    1   64 >>> elapsed 0.007262 sec
==PROF== Profiling "mathKernel3(float*)" - 4: 0%....50%....100% - 1 pass
mathKernel3 <<<    1   64 >>> elapsed 0.007343 sec
==PROF== Profiling "mathKernel4(float*)" - 5: 0%....50%....100% - 1 pass
mathKernel4 <<<    1   64 >>> elapsed 0.006955 sec
==PROF== Disconnected from process 20936
[20936] simpleDivergence@127.0.0.1
  warmingup(float*), 2024-Aug-28 11:14:50, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    smsp__sass_average_branch_targets_threads_uniform.pct                                %                            100
    ---------------------------------------------------------------------- --------------- ------------------------------

  mathKernel1(float*), 2024-Aug-28 11:14:50, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    smsp__sass_average_branch_targets_threads_uniform.pct                                %                             80
    ---------------------------------------------------------------------- --------------- ------------------------------

  mathKernel2(float*), 2024-Aug-28 11:14:50, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    smsp__sass_average_branch_targets_threads_uniform.pct                                %                            100
    ---------------------------------------------------------------------- --------------- ------------------------------

  mathKernel3(float*), 2024-Aug-28 11:14:50, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    smsp__sass_average_branch_targets_threads_uniform.pct                                %                          71.43
    ---------------------------------------------------------------------- --------------- ------------------------------

  mathKernel4(float*), 2024-Aug-28 11:14:50, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    smsp__sass_average_branch_targets_threads_uniform.pct                                %                            100
    ---------------------------------------------------------------------- --------------- ------------------------------
$
```


#### `gld_throughput` -> `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second`

用gld_throughput指标检查内核的内存读取效率

nvprof

```
$ nvprof --metrics gld_throughput ./sumMatrix 32 32
```

ncu

```
$ sudo /usr/local/NVIDIA-Nsight-Compute/ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second ./sumMatrix 32 32
==PROF== Connected to process 25364 (/home/hexu/git/Professional.CUDA.C.Programming/CodeSamples/chapter03/sumMatrix)
==PROF== Profiling "sumMatrixOnGPU2D" - 1: 0%....50%....100% - 1 pass
sumMatrixOnGPU2D <<<(512,512), (32,32)>>> elapsed 189.383030 ms
==PROF== Disconnected from process 25364
[25364] sumMatrix@127.0.0.1
  sumMatrixOnGPU2D(float*, float*, float*, int, int), 2024-Aug-27 14:55:57, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second                   Gbyte/second                          64.43
    ---------------------------------------------------------------------- --------------- ------------------------------
$
```

#### `gld_efficiency` -> `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct`

用gld_efficiency指标检测全局加载效率，即被请求的全局加载吞吐量占所需的全局加载吞吐量的比值。
它衡量了应用程序的加载操作利用设备内存带宽的程度。

nvprof

```
$ nvprof --metrics gld_efficiency ./sumMatrix 32 32 
```

ncu

```
$ sudo /usr/local/NVIDIA-Nsight-Compute/ncu --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct ./sumMatrix 32 32
==PROF== Connected to process 26982 (/home/hexu/git/Professional.CUDA.C.Programming/CodeSamples/chapter03/sumMatrix)
==PROF== Profiling "sumMatrixOnGPU2D" - 1: 0%....50%....100% - 1 pass
sumMatrixOnGPU2D <<<(512,512), (32,32)>>> elapsed 286.754847 ms
==PROF== Disconnected from process 26982
[26982] sumMatrix@127.0.0.1
  sumMatrixOnGPU2D(float*, float*, float*, int, int), 2024-Aug-27 15:27:02, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct                        %                            100
    ---------------------------------------------------------------------- --------------- ------------------------------
$
```


#### `achieved_occupancy` -> `sm__warps_active.avg.pct_of_peak_sustained_active`

achieved_occupancy参数：每个sm在每个cycle能够达到的最大activewarp 占总warp的比例。

nvprof

```
$ nvprof --metrics achieved_occupancy ./sumMatrix 64 2
```

ncu

```
$ sudo /usr/local/NVIDIA-Nsight-Compute/ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./sumMatrix 64 2
==PROF== Connected to process 27152 (/home/hexu/git/Professional.CUDA.C.Programming/CodeSamples/chapter03/sumMatrix)
==PROF== Profiling "sumMatrixOnGPU2D" - 1: 0%....50%....100% - 1 pass
sumMatrixOnGPU2D <<<(256,8192), (64,2)>>> elapsed 166.579008 ms
==PROF== Disconnected from process 27152
[27152] sumMatrix@127.0.0.1
  sumMatrixOnGPU2D(float*, float*, float*, int, int), 2024-Aug-27 15:33:28, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    sm__warps_active.avg.pct_of_peak_sustained_active                                    %                          93.26
    ---------------------------------------------------------------------- --------------- ------------------------------
$
```

#### `inst_per_warp` -> `smsp__average_inst_executed_per_warp.ratio`

用inst_per_warp指标来查看每个线程束上执行指令数量的平均值。

nvprof

```
$ nvprof --metrics inst_per_warp ./reduceInteger
```

ncu

```
$ sudo /usr/local/NVIDIA-Nsight-Compute/ncu --metrics smsp__average_inst_executed_per_warp.ratio ./reduceInteger
==PROF== Connected to process 1900 (/home/hexu/git/Professional.CUDA.C.Programming/CodeSamples/chapter03/reduceInteger)
/home/hexu/git/Professional.CUDA.C.Programming/CodeSamples/chapter03/./reduceInteger starting reduction at device 0: NVIDIA GeForce RTX 2060     with array size 16777216  grid 32768 block 512
cpu reduce      elapsed 0.033248 sec cpu_sum: 2139353471
==PROF== Profiling "reduceNeighbored" - 1: 0%....50%....100% - 1 pass
gpu Neighbored  elapsed 0.129067 sec gpu_sum: 2139353471 <<<grid 32768 block 512>>>
==PROF== Profiling "reduceNeighboredLess" - 2: 0%....50%....100% - 1 pass
gpu Neighbored2 elapsed 0.019528 sec gpu_sum: 2139353471 <<<grid 32768 block 512>>>
==PROF== Profiling "reduceInterleaved" - 3: 0%....50%....100% - 1 pass
gpu Interleaved elapsed 0.021165 sec gpu_sum: 2139353471 <<<grid 32768 block 512>>>
==PROF== Profiling "reduceUnrolling2" - 4: 0%....50%....100% - 1 pass
gpu Unrolling2  elapsed 0.016253 sec gpu_sum: 2139353471 <<<grid 16384 block 512>>>
==PROF== Profiling "reduceUnrolling4" - 5: 0%....50%....100% - 1 pass
gpu Unrolling4  elapsed 0.016773 sec gpu_sum: 2139353471 <<<grid 8192 block 512>>>
==PROF== Profiling "reduceUnrolling8" - 6: 0%....50%....100% - 1 pass
gpu Unrolling8  elapsed 0.012840 sec gpu_sum: 2139353471 <<<grid 4096 block 512>>>
==PROF== Profiling "reduceUnrollWarps8" - 7: 0%....50%....100% - 1 pass
gpu UnrollWarp8 elapsed 0.012494 sec gpu_sum: 2139353471 <<<grid 4096 block 512>>>
==PROF== Profiling "reduceCompleteUnrollWarps8" - 8: 0%....50%....100% - 1 pass
gpu Cmptnroll8  elapsed 0.018790 sec gpu_sum: 2139353471 <<<grid 4096 block 512>>>
==PROF== Profiling "reduceCompleteUnroll" - 9: 0%....50%....100% - 1 pass
gpu Cmptnroll   elapsed 0.016863 sec gpu_sum: 2139353471 <<<grid 4096 block 512>>>
==PROF== Disconnected from process 1900
[1900] reduceInteger@127.0.0.1
  reduceNeighbored(int*, int*, unsigned int), 2024-Aug-27 17:26:39, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    smsp__average_inst_executed_per_warp.ratio                                   inst/warp                         786.19
    ---------------------------------------------------------------------- --------------- ------------------------------

  reduceNeighboredLess(int*, int*, unsigned int), 2024-Aug-27 17:26:39, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    smsp__average_inst_executed_per_warp.ratio                                   inst/warp                         326.75
    ---------------------------------------------------------------------- --------------- ------------------------------

  reduceInterleaved(int*, int*, unsigned int), 2024-Aug-27 17:26:39, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    smsp__average_inst_executed_per_warp.ratio                                   inst/warp                         306.25
    ---------------------------------------------------------------------- --------------- ------------------------------

  reduceUnrolling2(int*, int*, unsigned int), 2024-Aug-27 17:26:39, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    smsp__average_inst_executed_per_warp.ratio                                   inst/warp                         342.38
    ---------------------------------------------------------------------- --------------- ------------------------------

  reduceUnrolling4(int*, int*, unsigned int), 2024-Aug-27 17:26:39, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    smsp__average_inst_executed_per_warp.ratio                                   inst/warp                         390.38
    ---------------------------------------------------------------------- --------------- ------------------------------

  reduceUnrolling8(int*, int*, unsigned int), 2024-Aug-27 17:26:39, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    smsp__average_inst_executed_per_warp.ratio                                   inst/warp                         462.38
    ---------------------------------------------------------------------- --------------- ------------------------------

  reduceUnrollWarps8(int*, int*, unsigned int), 2024-Aug-27 17:26:39, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    smsp__average_inst_executed_per_warp.ratio                                   inst/warp                         365.38
    ---------------------------------------------------------------------- --------------- ------------------------------

  reduceCompleteUnrollWarps8(int*, int*, unsigned int), 2024-Aug-27 17:26:39, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    smsp__average_inst_executed_per_warp.ratio                                   inst/warp                         328.38
    ---------------------------------------------------------------------- --------------- ------------------------------

  void reduceCompleteUnroll<512u>(int*, int*, unsigned int), 2024-Aug-27 17:26:39, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    smsp__average_inst_executed_per_warp.ratio                                   inst/warp                         314.38
    ---------------------------------------------------------------------- --------------- ------------------------------
$
```


#### `dram_read_throughput` -> `dram__bytes_read.sum.per_second`

dram_read_throughput表示设备内存读取吞吐量指标

nvprof

```
$ nvprof --metrics dram_read_throughput ./reduceInteger
```

ncu

```
$ sudo /usr/local/NVIDIA-Nsight-Compute/ncu --metrics dram__bytes_read.sum.per_second ./reduceInteger
==PROF== Connected to process 27454 (/home/hexu/git/Professional.CUDA.C.Programming/CodeSamples/chapter03/reduceInteger)
/home/hexu/git/Professional.CUDA.C.Programming/CodeSamples/chapter03/./reduceInteger starting reduction at device 0: NVIDIA GeForce RTX 2060     with array size 16777216  grid 32768 block 512
cpu reduce      elapsed 0.032732 sec cpu_sum: 2139353471
==PROF== Profiling "reduceNeighbored" - 1: 0%....50%....100% - 1 pass
gpu Neighbored  elapsed 0.126561 sec gpu_sum: 2139353471 <<<grid 32768 block 512>>>
==PROF== Profiling "reduceNeighboredLess" - 2: 0%....50%....100% - 1 pass
gpu Neighbored2 elapsed 0.019401 sec gpu_sum: 2139353471 <<<grid 32768 block 512>>>
==PROF== Profiling "reduceInterleaved" - 3: 0%....50%....100% - 1 pass
gpu Interleaved elapsed 0.015214 sec gpu_sum: 2139353471 <<<grid 32768 block 512>>>
==PROF== Profiling "reduceUnrolling2" - 4: 0%....50%....100% - 1 pass
gpu Unrolling2  elapsed 0.014083 sec gpu_sum: 2139353471 <<<grid 16384 block 512>>>
==PROF== Profiling "reduceUnrolling4" - 5: 0%....50%....100% - 1 pass
gpu Unrolling4  elapsed 0.009558 sec gpu_sum: 2139353471 <<<grid 8192 block 512>>>
==PROF== Profiling "reduceUnrolling8" - 6: 0%....50%....100% - 1 pass
gpu Unrolling8  elapsed 0.010215 sec gpu_sum: 2139353471 <<<grid 4096 block 512>>>
==PROF== Profiling "reduceUnrollWarps8" - 7: 0%....50%....100% - 1 pass
gpu UnrollWarp8 elapsed 0.016290 sec gpu_sum: 2139353471 <<<grid 4096 block 512>>>
==PROF== Profiling "reduceCompleteUnrollWarps8" - 8: 0%....50%....100% - 1 pass
gpu Cmptnroll8  elapsed 0.014875 sec gpu_sum: 2139353471 <<<grid 4096 block 512>>>
==PROF== Profiling "reduceCompleteUnroll" - 9: 0%....50%....100% - 1 pass
gpu Cmptnroll   elapsed 0.014720 sec gpu_sum: 2139353471 <<<grid 4096 block 512>>>
==PROF== Disconnected from process 27454
[27454] reduceInteger@127.0.0.1
  reduceNeighbored(int*, int*, unsigned int), 2024-Aug-28 16:57:20, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum.per_second                                           Gbyte/second                           9.11
    ---------------------------------------------------------------------- --------------- ------------------------------

  reduceNeighboredLess(int*, int*, unsigned int), 2024-Aug-28 16:57:20, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum.per_second                                           Gbyte/second                          11.56
    ---------------------------------------------------------------------- --------------- ------------------------------

  reduceInterleaved(int*, int*, unsigned int), 2024-Aug-28 16:57:20, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum.per_second                                           Gbyte/second                          13.07
    ---------------------------------------------------------------------- --------------- ------------------------------

  reduceUnrolling2(int*, int*, unsigned int), 2024-Aug-28 16:57:20, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum.per_second                                           Gbyte/second                          22.24
    ---------------------------------------------------------------------- --------------- ------------------------------

  reduceUnrolling4(int*, int*, unsigned int), 2024-Aug-28 16:57:20, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum.per_second                                           Gbyte/second                          36.29
    ---------------------------------------------------------------------- --------------- ------------------------------

  reduceUnrolling8(int*, int*, unsigned int), 2024-Aug-28 16:57:20, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum.per_second                                           Gbyte/second                          58.92
    ---------------------------------------------------------------------- --------------- ------------------------------

  reduceUnrollWarps8(int*, int*, unsigned int), 2024-Aug-28 16:57:20, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum.per_second                                           Gbyte/second                          57.87
    ---------------------------------------------------------------------- --------------- ------------------------------

  reduceCompleteUnrollWarps8(int*, int*, unsigned int), 2024-Aug-28 16:57:21, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum.per_second                                           Gbyte/second                          61.79
    ---------------------------------------------------------------------- --------------- ------------------------------

  void reduceCompleteUnroll<512u>(int*, int*, unsigned int), 2024-Aug-28 16:57:21, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum.per_second                                           Gbyte/second                          60.40
    ---------------------------------------------------------------------- --------------- ------------------------------
```
