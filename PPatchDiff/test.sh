CUDA_VISIBLE_DEVICES=0 python pydiff/train.py -opt options/infer_lolv1_pgc.yaml
CUDA_VISIBLE_DEVICES=0 python pydiff/train.py -opt options/infer_lolv2_pgc.yaml
CUDA_VISIBLE_DEVICES=0 python pydiff/train.py -opt options/infer_lolsyn_pgc.yaml