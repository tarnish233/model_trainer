python3 build_data.py \
    --data_file /mnt/bn/fulei-v6-hl-nas-mlx/mlx/workspace/model_trainer/data/im_train_test_1012.csv \
    --output_dir /mnt/bn/fulei-v6-hl-nas-mlx/mlx/workspace/model_trainer/data/ \
    --train_ratio 0.8 \
    --dev_ratio 0.1 \
    --test_ratio 0.1 \
    --label2id /mnt/bn/fulei-v6-hl-nas-mlx/mlx/workspace/saas/imbot/data/label2id.json \
    --problem_type single_label_classification
