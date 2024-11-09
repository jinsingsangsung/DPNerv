
CUDA_VISIBLE_DEVICES=0 python3 main.py --cfg_path cfgs/DNeRV_Beauty.yaml --output_dir ./experiments/my_dnerv1 --time_str my_dnerv1
CUDA_VISIBLE_DEVICES=1 python3 main.py --cfg_path cfgs/DNeRV_Bosph.yaml --output_dir ./experiments/my_dnerv2 --time_str my_dnerv2
CUDA_VISIBLE_DEVICES=2 python3 main.py --cfg_path cfgs/DNeRV_HoneyB.yaml --output_dir ./experiments/my_dnerv3 --time_str my_dnerv3
CUDA_VISIBLE_DEVICES=3 python3 main.py --cfg_path cfgs/DNeRV_Jockey.yaml --output_dir ./experiments/my_dnerv4 --time_str my_dnerv4

CUDA_VISIBLE_DEVICES=0 python3 main.py --cfg_path cfgs/DNeRV_ReadySet.yaml --output_dir ./experiments/my_dnerv5 --time_str my_dnerv5
CUDA_VISIBLE_DEVICES=1 python3 main.py --cfg_path cfgs/DNeRV_ShakeNDry.yaml --output_dir ./experiments/my_dnerv6 --time_str my_dnerv6
CUDA_VISIBLE_DEVICES=2 python3 main.py --cfg_path cfgs/DNeRV_Yacht.yaml --output_dir ./experiments/my_dnerv7 --time_str my_dnerv7
CUDA_VISIBLE_DEVICES=3 python3 main.py --cfg_path cfgs/DNeRV_Bunny.yaml --output_dir ./experiments/my_dnerv_bunny --time_str my_dnerv_bunny


CUDA_VISIBLE_DEVICES=0 python3 main.py --cfg_path cfgs/PNeRV_Beauty.yaml --output_dir ./experiments/my_pnerv1 --time_str my_pnerv1
CUDA_VISIBLE_DEVICES=1 python3 main.py --cfg_path cfgs/PNeRV_Bosph.yaml --output_dir ./experiments/my_pnerv2 --time_str my_pnerv2
CUDA_VISIBLE_DEVICES=2 python3 main.py --cfg_path cfgs/PNeRV_HoneyB.yaml --output_dir ./experiments/my_pnerv3 --time_str my_pnerv3
CUDA_VISIBLE_DEVICES=3 python3 main.py --cfg_path cfgs/PNeRV_Jockey.yaml --output_dir ./experiments/my_pnerv4 --time_str my_pnerv4

CUDA_VISIBLE_DEVICES=0 python3 main.py --cfg_path cfgs/PNeRV_ReadySet.yaml --output_dir ./experiments/my_pnerv5 --time_str my_pnerv5
CUDA_VISIBLE_DEVICES=1 python3 main.py --cfg_path cfgs/PNeRV_ShakeNDry.yaml --output_dir ./experiments/my_pnerv6 --time_str my_pnerv6
CUDA_VISIBLE_DEVICES=2 python3 main.py --cfg_path cfgs/PNeRV_Yacht.yaml --output_dir ./experiments/my_pnerv7 --time_str my_pnerv7
CUDA_VISIBLE_DEVICES=3 python3 main.py --cfg_path cfgs/PNeRV_Bunny.yaml --output_dir ./experiments/my_pnerv_bunny --time_str my_pnerv_bunny
