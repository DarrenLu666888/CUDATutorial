
# nvprof --devices 0 --metrics gld_efficiency,gst_efficiency ./simpleMathAoS
sudo /usr/local/NVIDIA-Nsight-Compute/ncu --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct ./simpleMathAoS

