"""Read complete PoSSM scenes and cache matched-filter outputs."""
from pathlib import Path
import argparse
import json
import time

import numpy as np

from gdf import read_gdf,measurement_geometry
from imaging import pulse_compress


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("folder",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    start=time.perf_counter()
    config,kernel,pings,manifest=read_gdf(args.folder.glob("*.gdf"))
    raw=np.stack([p["iq"] for p in pings])
    tx,rx,speeds=measurement_geometry(config,pings)
    print(f"Raw {raw.shape}, kernel {len(kernel)} samples",flush=True)
    # GDF calls this KBID=1 matched-filter kernel. Its correlation convention
    # is checked on known scene00 development scatterers, not test scenes.
    compressed=pulse_compress(raw.reshape(-1,raw.shape[-1]),kernel,mode="correlation")
    args.output.mkdir(parents=True,exist_ok=True)
    np.save(args.output/"range_compressed.npy",compressed)
    np.savez(args.output/"geometry.npz",tx=tx,rx=rx,speeds=speeds,kernel=kernel)
    summary={"config":config,"manifest":manifest,"raw_shape":list(raw.shape),
             "compressed_shape":list(compressed.shape),"pulse_compression":"FFT convolution with conjugate-reversed supplied kernel",
             "sample_offset_initial":len(kernel)-1,"seconds":time.perf_counter()-start}
    (args.output/"metadata.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k not in ["config","manifest"]},indent=2),flush=True)


if __name__=="__main__":main()
