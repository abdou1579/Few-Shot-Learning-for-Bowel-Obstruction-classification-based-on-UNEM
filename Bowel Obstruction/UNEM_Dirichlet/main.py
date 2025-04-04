import os
import random
import clip
import argparse
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
from src.utils import load_cfg_from_cfg_file, merge_cfg_from_list, Logger, get_log_file
from src.eval_few_shot import Evaluator_few_shot
from src.utrain_few_shot import U_Trainer_few_shot

torch.cuda.empty_cache()
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Main')
    cfg = load_cfg_from_cfg_file('config/main_config.yaml')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    dataset_config = 'config/datasets_config/config_{}.yaml'.format(
        cfg.dataset)
    method_config = 'config/methods_config/{}.yaml'.format(cfg.method)
    cfg.update(load_cfg_from_cfg_file(method_config))
    cfg.update(load_cfg_from_cfg_file(dataset_config))
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    cfg.n_class = cfg.num_classes_test

    return cfg


def main():
    args = parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
    torch.cuda.set_device(args.device)

    # create model
    model, preprocess = clip.load(args.backbone, device)
    log_file = get_log_file(log_path=args.log_path,
                            dataset=args.dataset, method=args.name_method)
    logger = Logger(__name__, log_file)

    if args.u_train:
        u_trainer = U_Trainer_few_shot(device=device, args=args, log_file=log_file)
        u_trainer.run_training(model=model, preprocess=preprocess)
        return

    if args.shots > 0:  # few-shot setting
        evaluator = Evaluator_few_shot(device=device, args=args, log_file=log_file)
    else:
        raise ValueError('Invalid number of shots')
    evaluator.run_full_evaluation(model=model, preprocess=preprocess)


if __name__ == "__main__":
    main()
