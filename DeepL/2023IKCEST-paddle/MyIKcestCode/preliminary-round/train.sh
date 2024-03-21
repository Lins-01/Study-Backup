python run_seq_cls.py \
  --model_name_or_path ernie-3.0-xbase-zh \
  --dataset cmnli \
  --output_dir ./best_models \
  --export_model_dir best_models/ \
  --do_train \
  --do_eval \
  --do_export \
  --config=default.yml