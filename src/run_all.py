"""Execute the complete CPU experiment and regenerate the HTML report."""
from pathlib import Path
import argparse,os,subprocess,sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--download',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    (root/'data').mkdir(exist_ok=True)
    env=os.environ.copy();env.setdefault('OPENBLAS_NUM_THREADS','4');env.setdefault('OMP_NUM_THREADS','4');env.setdefault('NUMBA_NUM_THREADS','8')
    commands=[]
    if a.download:commands.append(['download_possm.py'])
    commands += [['prepare_possm.py',f'data/possm_{s}','--output',f'data/{s}_cache'] for s in ['scene00','scene01']]
    commands += [['evaluate_possm.py'],['focus_learning.py','--output','results/learning'],['transfer_focus.py'],['verify_results.py'],['build_report.py']]
    for command in commands:
        args=[sys.executable,str(root/'src'/command[0]),*command[1:]]
        print('Running:',' '.join(args),flush=True)
        subprocess.run(args,cwd=root,env=env,check=True)


if __name__=='__main__':main()
