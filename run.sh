CUDA_VISIBLE_DEVICES=0,1,2,3 ray start --head
CUDA_VISIBLE_DEVICES=4,5,6,7 ray start --address=127.0.0.1:6379
trinity run --config examples/grpo_gsm8k/gsm8k.yaml
