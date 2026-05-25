CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=22666 pydiff/train.py -opt options/train_lolv1_pgc.yaml --launcher pytorch

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=22666 pydiff/train.py -opt options/train_lolv2_pgc.yaml --launcher pytorch

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=22666 pydiff/train.py -opt options/train_lolsyn_pgc.yaml --launcher pytorch

