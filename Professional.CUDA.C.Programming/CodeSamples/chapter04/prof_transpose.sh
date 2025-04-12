

for KERNEL in {0..3}
do
    echo "$KERNEL"
# nvprof --devices 0 --metrics gld_throughput,gst_throughput ./transpose $KERNEL 16 16 2048 2048
#    sudo /usr/local/NVIDIA-Nsight-Compute/ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second ./transpose $KERNEL 16 16 2048 2048
# nvprof --devices 0 --metrics --metrics gld_efficiency,gst_efficiency ./transpose $KERNEL 16 16 2048 2048
    sudo /usr/local/NVIDIA-Nsight-Compute/ncu --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct ./transpose $KERNEL 16 16 2048 2048
done
