### 6.2.1 非空流中的并发内核

通过nvvp测试simpleHyperq时需要把优化关掉，不然核函数被优化，
造成看不出并发行：

编译选项要改成：

```
nvcc -g -G -Xcompiler -O0 -Xptxas -O0 -lineinfo -O0 -o $@ $<
```
