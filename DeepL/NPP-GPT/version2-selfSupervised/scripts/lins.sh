seq_len=512
model=GPT4TS

for percent in 100
do
for pred_len in 96
do

python main.py \
    --root_path ./datasets/lins/ \
    --data_path fw80_pow90_loop1.csv \
    --model_id lins_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data custom \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --batch_size 256 \
    --learning_rate 0.001 \
    --train_epochs 10 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --freq 0 \
    --target vol_170101010.tempf \
    --patch_size 16 \
    --stride 8 \
    --percent $percent \
    --gpt_layer 6 \
    --itr 3 \
    --model $model \
    --patience 3 \
    --cos 1 \
    --tmax 10 \
    --is_gpt 1

done
done