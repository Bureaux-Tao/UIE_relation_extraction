# 数据生成1
python bio2doccano.py

# 数据生成2
python colabeler2doccano.py \
--colabeler_file ../data/ori_data/lungCT/export.json \
--doccano_file ../data/doccano_data/test.jsonl

# 数据生成3
python doccano2prompt.py \
--doccano_file ../data/doccano_data/lungCT.jsonl \
--task_type ext \
--save_dir ../data/target_data/lungCT/ \
--negative_ratio 3 \
--splits 0.7 0.3 0

# finetune训练
python train.py \
--batch_size 32 \
--learning_rate 1e-5 \
--train_path ./data/target_data/lungCT/train.txt \
--dev_path ./data/target_data/lungCT/dev.txt \
--max_seq_len 256 \
--num_epochs 20 \
--shots 0 \
--early_stop_patience 5 \
--best_weight_dir ./checkpoints/ \
--best_weight_name lung_bc.pt \
--train_history_dir ./history/ \
--train_history_name lung_bc.csv