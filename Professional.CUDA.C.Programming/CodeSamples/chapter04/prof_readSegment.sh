
# nvprof --devices 0 --metrics gld_efficiency ./readSegment 0
sudo /usr/local/NVIDIA-Nsight-Compute/ncu --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct ./readSegment 0

# nvprof --devices 0 --metrics gld_efficiency ./readSegment 11
sudo /usr/local/NVIDIA-Nsight-Compute/ncu --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct ./readSegment 11

# nvprof --devices 0 --metrics gld_efficiency ./readSegment 128
sudo /usr/local/NVIDIA-Nsight-Compute/ncu --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct ./readSegment 128

# nvprof --devices 0 --metrics gld_transactions ./readSegment 0
sudo /usr/local/NVIDIA-Nsight-Compute/ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum ./readSegment 0

# nvprof --devices 0 --metrics gld_transactions ./readSegment 11
sudo /usr/local/NVIDIA-Nsight-Compute/ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum ./readSegment 11

# nvprof --devices 0 --metrics gld_transactions ./readSegment 128
sudo /usr/local/NVIDIA-Nsight-Compute/ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum ./readSegment 128

