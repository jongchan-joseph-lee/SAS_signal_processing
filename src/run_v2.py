"""Reproduce the stage-1 v2 checkpoint; raw acquisition optional."""
from pathlib import Path
import argparse,os,subprocess,sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--download',action='store_true');a=p.parse_args();root=Path(__file__).resolve().parents[1]
    env=os.environ.copy();env.setdefault('OPENBLAS_NUM_THREADS','4');env.setdefault('NUMBA_NUM_THREADS','8')
    commands=[['acquire_all_scenes.py']] if a.download else []
    commands += [['prepare_possm.py',f'data/possm_{s}','--output',f'data/{s}_cache'] for s in ['scene00','scene01']]
    commands += [[s] for s in ['diagnose_slc.py','diagnose_taper.py','validate_slc_v2.py','stress_points_v2.py','audit_sources_v2.py','diagnose_gain.py','verify_v2.py']]
    commands += [['build_report_v2.py','--bundle']]
    for cmd in commands:
        args=[sys.executable,str(root/'src'/cmd[0]),*cmd[1:]];print('Running:',' '.join(args),flush=True);subprocess.run(args,cwd=root,env=env,check=True)


if __name__=='__main__':main()
