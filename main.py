comet_support = True
try:
    from comet_ml import Experiment
except ImportError as e:
    print("Comet ML is not installed, ignore the comet experiment monitor")
    comet_support = False
from models import SiamDTI
from time import time
from utils import set_seed, graph_collate_func, mkdir
from configs import get_cfg_defaults
from dataloader import DTIDataset
from torch.utils.data import DataLoader
from trainer import Trainer

import torch
import argparse
import warnings, os
import pandas as pd
import pdb


parser = argparse.ArgumentParser(description="SiamDTI for DTI prediction")
parser.add_argument('--cfg', default='configs/SiamDTI_LL4.yaml', help="path to config file", type=str)
parser.add_argument('--data', default='biosnap', type=str, metavar='TASK',
                    help='dataset')
parser.add_argument('--split', default='random', type=str, metavar='S', help="split task", choices=['S1', 'S2'])
parser.add_argument('--seed', default='1', type=int)
parser.add_argument('--outputDir', default='./result/S1_biosnap_em256/', type=str, help="split task")
parser.add_argument('--cuda', default='0', type=str, help="cuda")
args = parser.parse_args()

device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')
def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.SOLVER.SEED = args.seed
    set_seed(cfg.SOLVER.SEED)
    suffix = str(int(time() * 1000))[6:]
    cfg.RESULT.OUTPUT_DIR = args.outputDir
    print(cfg.RESULT.OUTPUT_DIR)
    mkdir(args.outputDir)
    experiment = None
    print(f"Config yaml: {args.cfg}")
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")

    dataFolder = f'./datasets/{args.data}'
    dataFolder = os.path.join(dataFolder, str(args.split))


    train_path = os.path.join(dataFolder, 'train.csv')
    val_path = os.path.join(dataFolder, "val.csv")
    test_path = os.path.join(dataFolder, "test.csv")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    train_dataset = DTIDataset(df_train.index.values, df_train)
    val_dataset = DTIDataset(df_val.index.values, df_val)
    test_dataset = DTIDataset(df_test.index.values, df_test)


    if cfg.COMET.USE and comet_support:
        experiment = Experiment(
            project_name=cfg.COMET.PROJECT_NAME,
            workspace=cfg.COMET.WORKSPACE,
            auto_output_logging="simple",
            log_graph=True,
            log_code=False,
            log_git_metadata=False,
            log_git_patch=False,
            auto_param_logging=False,
            auto_metric_logging=False
        )
        hyper_params = {
            "LR": cfg.SOLVER.LR,
            "Output_dir": args.outputDir,
        }
        experiment.log_parameters(hyper_params)
        if cfg.COMET.TAG is not None:
            experiment.add_tag(cfg.COMET.TAG)
        experiment.set_name(f"{args.data}_{suffix}")

    params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS,
              'drop_last': True, 'collate_fn': graph_collate_func}

    training_generator = DataLoader(train_dataset, **params)
    params['shuffle'] = False
    params['drop_last'] = False
    #if not cfg.DA.TASK:
    val_generator = DataLoader(val_dataset, **params)
    test_generator = DataLoader(test_dataset, **params)


    model = SiamDTI(**cfg).to(device)   

    opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)

    torch.backends.cudnn.benchmark = True

    trainer = Trainer(model, opt, device, training_generator, val_generator, test_generator, 
                          
                          experiment=experiment, **cfg)

    result = trainer.train()

    with open(os.path.join(args.outputDir, "model_architecture.txt"), "w") as wf:
        wf.write(str(model))

    print()
    print(f"Directory for saving result: {args.outputDir}")

    return result


if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
