# Split Dataset into split_num parts to apply sae
idxes=(0 1 2 3 4 5 6 7)

for idx in "${idxes[@]}"
do
  python src/apply.py \
    --model_path ... \
    --tokenizer_path ... \
    --hidden_size 4096 --latent_size 65536 --k 192 --layer 16 \
    --dataset_name skyworkpreference \
    --data_path ... \
    --batch_size 4 --max_length 1024 \
    --split_index $idx --split_num 8 \
    --apply_threshold 3.0 \
    --device cuda:$idx \
    --output_path ...  > log/$idx.log 2>&1 &
done