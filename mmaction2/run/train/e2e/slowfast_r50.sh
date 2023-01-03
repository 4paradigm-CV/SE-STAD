OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PORT=4819 ./tools/dist_train.sh configs/detection/sestad/e2e/slowfast_k400_r50_fpn_fcos.py \
    8 --validate
