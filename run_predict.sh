python3 run_predict.py \
    --data_file /mnt/bn/fulei-v6-lq-nas-mlx/mlx/workspace/DCC/data/DCC_customer/customer_dev_0328.json \
    --output_file /mnt/bn/fulei-v6-hl-nas-mlx/mlx/workspace/model_trainer/data/ \
    --model_path /mnt/bn/fulei-v6-lq-nas-mlx/mlx/workspace/DCC/model/customer_macbert_top10_label_0328 \
    --tokenizer_path /mnt/bn/fulei-v6-lq-nas-mlx/mlx/workspace/DCC/model/customer_macbert_top10_label_0328 \
    --label2id /mnt/bn/fulei-v6-lq-nas-mlx/mlx/workspace/DCC/config/customer.json \
    --max_length 512 \
    --problem_type multi_label_classification