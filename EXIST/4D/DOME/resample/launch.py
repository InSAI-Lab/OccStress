import subprocess
import sys,os
import pickle
from tqdm import tqdm


def run_command(command: str) -> None:
    """Run a command kill actions if it fails

    Args:
        command: command to run
    """
    ret_code = subprocess.call(command, shell=True)
    if ret_code != 0:
        print(f"[bold red]Error: `{command}` failed. Exiting...")
        # sys.exit(1)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--n_rank', type=int, default=1)
    parser.add_argument('--dst', type=str, required=True)
    parser.add_argument('--imageset', type=str, default='../data/nuscenes_infos_train_temporal_v3_scene.pkl')
    parser.add_argument('--input_dataset', type=str, default='gts')
    parser.add_argument('--data_path', type=str, default='../data/nuscenes')
    

    return parser.parse_args()

args=parse_args()
with open(args.imageset, 'rb') as f:
    data = pickle.load(f)
nusc_infos = data['infos']


for i in tqdm(range(args.rank,len(nusc_infos),args.n_rank),desc='lauch processing'):
    print('#'*50,'idx',i)
    dst=os.path.join(args.dst,f'src_scene-{i+1:04d}')
    cmd = [
        "python", "main.py","--idx",f"{i}",
        "--dst",f"{dst}",
        "--imageset",f"{args.imageset}",
        "--input_dataset",f"{args.input_dataset}",
        "--data_path",f"{args.data_path}",
    ]
    cmd = " ".join(cmd)
    print('@'*50,cmd)
    run_command(cmd)





