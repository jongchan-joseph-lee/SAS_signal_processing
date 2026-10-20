"""Reproduce remaining autofocus and detection experiments on public PoSSM."""
from pathlib import Path
import argparse,os,subprocess,sys


def main():
 p=argparse.ArgumentParser();p.add_argument('--download',action='store_true');p.add_argument('--report-only',action='store_true');a=p.parse_args()
 root=Path(__file__).resolve().parents[1];env=os.environ.copy()
 for k,v in {'OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2','NUMBA_NUM_THREADS':'6'}.items():env.setdefault(k,v)
 commands=[]
 if not a.report_only:
  if a.download:commands.append(['acquire_all_scenes.py'])
  for s in ['scene00','scene04']:
   if not (root/f'data/{s}_cache/range_compressed.npy').exists():commands.append(['prepare_possm.py',f'data/possm_{s}','--output',f'data/{s}_cache'])
  commands += [['protocol_v3.py'],['evaluate_autofocus_v3.py','--stage','develop'],['point_detection_v3.py','--stage','train'],
   ['evaluate_autofocus_v3.py','--stage','test'],['point_detection_v3.py','--stage','test'],['raw_detection_bridge_v3.py'],['verify_v3.py']]
 commands.append(['build_report_v3.py','--bundle'])
 for cmd in commands:
  print('Running:',' '.join(cmd),flush=True);subprocess.run([sys.executable,str(root/'src'/cmd[0]),*cmd[1:]],cwd=root,env=env,check=True)


if __name__=='__main__':main()
