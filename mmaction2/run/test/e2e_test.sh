./tools/dist_test.sh configs/detection/sestad/e2e/slowfast_k400_r50_fpn_fcos.py \
    ./work_dirs/ava/e2e/slowfast_k400_r50_8x8_cos_10e_fpn_fcos/epoch_9.pth 8 \
    --eval mAP --out ./work_dirs/ava/e2e/slowfast_k400_r50_8x8_cos_10e_fpn_fcos/results.csv
