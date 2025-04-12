
sudo sh -c 'echo 2 >/proc/sys/kernel/perf_event_paranoid'
nsys profile --stats=true --force-overwrite=true -o ./nsys_sumMatrixGPUManaged ./sumMatrixGPUManaged 

