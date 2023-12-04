### Dummy init command for blinks RWKV-LM (to setup init models)
> Make sure to abort after it initialized

```bash
python train.py --load_model "" --proj_dir "models/L24-D2048" \
--n_layer 24 --n_embd 2048 \
--micro_bsz 64  --pre_ffn 0 --head_qk 0 \
--data_file "../data/enwik8.npy" --data_type "numpy" --vocab_size 50277 \
--ctx_len 2048 --epoch_steps 1 --epoch_count 1 --epoch_begin 0 --epoch_save 1 \
--lr_init 8e-4 --lr_final 2e-4 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
--accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2_offload --grad_cp 0

python train.py --load_model "" --proj_dir "models/L12-D2560" \
--n_layer 12 --n_embd 2560 \
--micro_bsz 64  --pre_ffn 0 --head_qk 0 \
--data_file "../data/enwik8.npy" --data_type "numpy" --vocab_size 50277 \
--ctx_len 2048 --epoch_steps 1 --epoch_count 1 --epoch_begin 0 --epoch_save 1 \
--lr_init 8e-4 --lr_final 2e-4 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
--accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2_offload --grad_cp 0

python train.py --load_model "" --proj_dir "models/L12-D2816" \
--n_layer 12 --n_embd 2816 \
--micro_bsz 64  --pre_ffn 0 --head_qk 0 \
--data_file "../data/enwik8.npy" --data_type "numpy" --vocab_size 50277 \
--ctx_len 2048 --epoch_steps 1 --epoch_count 1 --epoch_begin 0 --epoch_save 1 \
--lr_init 8e-4 --lr_final 2e-4 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
--accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2_offload --grad_cp 0

python train.py --load_model "" --proj_dir "models/L24-D1024" \
--n_layer 24 --n_embd 1024 \
--micro_bsz 64  --pre_ffn 0 --head_qk 0 \
--data_file "../data/enwik8.npy" --data_type "numpy" --vocab_size 50277 \
--ctx_len 2048 --epoch_steps 1 --epoch_count 1 --epoch_begin 0 --epoch_save 1 \
--lr_init 8e-4 --lr_final 2e-4 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
--accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2_offload --grad_cp 0

python train.py --load_model "" --proj_dir "models/L24-D5120" \
--n_layer 24 --n_embd 5120 \
--micro_bsz 64  --pre_ffn 0 --head_qk 0 \
--data_file "../data/enwik8.npy" --data_type "numpy" --vocab_size 50277 \
--ctx_len 2048 --epoch_steps 1 --epoch_count 1 --epoch_begin 0 --epoch_save 1 \
--lr_init 8e-4 --lr_final 2e-4 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
--accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2_offload --grad_cp 0

python train.py --load_model "" --proj_dir "models/L24-D4096" \
--n_layer 24 --n_embd 4096 \
--micro_bsz 64  --pre_ffn 0 --head_qk 0 \
--data_file "../data/enwik8.npy" --data_type "numpy" --vocab_size 50277 \
--ctx_len 2048 --epoch_steps 1 --epoch_count 1 --epoch_begin 0 --epoch_save 1 \
--lr_init 8e-4 --lr_final 2e-4 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
--accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2_offload --grad_cp 0
```

## L12-D2560 init model is uploaded to HF
https://huggingface.co/picocreator/memory-size-experiment-for-rwkv/resolve/main/L12-D2560-init.pth

## L24-D1024 init model
https://huggingface.co/picocreator/memory-size-experiment-for-rwkv/resolve/main/L24-D1024-init.pth