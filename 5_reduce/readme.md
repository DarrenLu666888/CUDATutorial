## baseline

一个thread（一个block）来串行处理

## v0

share memory（smem）保存中间结果，避免写回GM的开销

1. smem是不同block独立，block内threads共享
2. 可以使用template `<int blockSize>来传静态smem`大小（不用blockDim的原因）
3. 对smem读写要用__syncthreads()保证读写顺序一致

## V1

使用位运算替代除余：除余数（相对于被除余数）一般需要是2的次幂，有以下考量：

1. 除数是 2的幂次
   1. 除法替代：x/d -> x>>log2(d). （一般log2 的结果是已知的）
   2. 取模替代：x%d -> x&(d-1)
2. 对于常量除法，可以先用cpu计算出除数的倒数，然后用乘法代替
3. 判断奇偶：x%2==0 -> x&1

## V2

smem bank conflict

就是每次迭代 所有线程访问的smem的地址 要尽可能连续（有的可能不会访问smem），即sequential addressing （相对于Interleaved Addressing）

## V3

原来是256个数的求和用256个线程来，现在用128个来，直接在load到smem的时候就执行一次相加；原因就是原来的方案后128个用了一次之后就没用了，不如不用
