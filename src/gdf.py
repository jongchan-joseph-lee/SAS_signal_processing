"""PoSSM GDF reader, based on PSU_GDF_Format_20240612.pdf.

The source archive is model-derived, noise-free underwater SAS data, not a
measurement. Complex raw samples use int16 / 2**15; the supplied matched
filter kernel uses int32 / 2**30. No pickle or executable metadata is loaded.
"""
from pathlib import Path
import hashlib
import json
import struct

import numpy as np

SYNC = 0x46444731


def read_gdf(paths):
    config = None
    kernel = None
    pings = []
    manifest = []
    for path in sorted(map(Path, paths)):
        content = path.read_bytes()
        manifest.append({"file": path.name, "bytes": len(content),
                         "sha256": hashlib.sha256(content).hexdigest()})
        offset = 0
        while offset < len(content):
            if offset+12 > len(content):
                raise ValueError("Truncated standard block header")
            sync, length, kind, version, checksum = struct.unpack_from("<IIBBH", content, offset)
            if sync != SYNC or offset+12+length > len(content):
                raise ValueError(f"Invalid GDF block at {path}:{offset}")
            payload = memoryview(content)[offset+12:offset+12+length]
            if kind == 1:
                system, channels, speed, advance, interval, fs, fc, spare, datatype = struct.unpack_from("<HHHHIIIHH", payload)
                if datatype != 0x0401:
                    raise ValueError(f"Unsupported datatype {datatype}")
                tx = np.frombuffer(payload[24:36], dtype="<i4").astype(float)/1e6
                rx = np.frombuffer(payload[36:36+channels*12], dtype="<i4").reshape(channels,3).astype(float)/1e6
                this = {"system": system, "channels": channels,
                        "speed_mps": speed/1000, "advance_m": advance/1000,
                        "interval_s": interval/1e6, "fs_hz": fs, "fc_hz": fc,
                        "datatype": datatype, "tx_local_m": tx.tolist(), "rx_local_m": rx.tolist()}
                if config is not None and config != this:
                    raise ValueError("Sensor configuration differs between files")
                config = this
            elif kind == 2:
                kbid, spare, count = struct.unpack_from("<HHI", payload)
                arr = np.frombuffer(payload[8:8+count*8], dtype="<i4").reshape(count,2).astype(np.float64)/2**30
                this_kernel = arr[:,0]+1j*arr[:,1]
                if kernel is not None and not np.array_equal(kernel, this_kernel):
                    raise ValueError("Kernel changes between files")
                kernel = this_kernel
                if kbid != 1:
                    raise ValueError("Expected KBID=1 supplied matched filter")
            elif kind == 3:
                number, sec, nsec, samples, data_offset, data_length, nav_offset, nav_length = struct.unpack_from("<8I", payload)
                start = 32+nav_offset
                nav_sec, nav_nsec, nav_size, nav_type, nav_version = struct.unpack_from("<IIHBB", payload, start)
                nav = payload[start+12:start+12+nav_size]
                sync1,sync2,nav_message_size,spare,nav_checksum = struct.unpack_from("<BBHHH",nav)
                if (sync1,sync2)!=(71,78) or (sum(nav[8:]) & 0xffff)!=nav_checksum:
                    raise ValueError("Navigation message checksum mismatch")
                values = struct.unpack_from("<9f", nav,8)
                names = ["altitude_m","depth_m","roll_rad","pitch_rad","heading_rad",
                         "xvel_mps","yvel_mps","zvel_mps","sound_speed_mps"]
                navigation = dict(zip(names, values))
                data_start = 32+data_offset
                if config is None or data_length != samples*config["channels"]*4:
                    raise ValueError("Sonar payload length does not match channel/sample count")
                data = np.frombuffer(payload[data_start:data_start+data_length],dtype="<i2").reshape(config["channels"],samples,2)
                iq = (data[:,:,0].astype(np.float32)+1j*data[:,:,1].astype(np.float32))/2**15
                pings.append({"number": number, "time_s": sec+nsec*1e-9,
                              "navigation": navigation, "iq": iq})
            offset += 12+length
    pings.sort(key=lambda p:p["number"])
    if len({p["number"] for p in pings})!=len(pings):
        raise ValueError("Duplicate ping numbers")
    if config is None or kernel is None or not pings:
        raise ValueError("Missing configuration, kernel or sonar data")
    return config, kernel, pings, manifest


def rotation(roll,pitch,heading):
    cr,sr=np.cos(roll),np.sin(roll)
    cp,sp=np.cos(pitch),np.sin(pitch)
    ch,sh=np.cos(heading),np.sin(heading)
    return np.array([[ch*cp,ch*sp*sr-sh*cr,ch*sp*cr+sh*sr],
                     [sh*cp,sh*sp*sr+ch*cr,sh*sp*cr-ch*sr],
                     [-sp,cp*sr,cp*cr]])


def measurement_geometry(config,pings):
    tx_local=np.array(config["tx_local_m"])
    rx_local=np.array(config["rx_local_m"])
    txs=[];rxs=[];speeds=[]
    for ping in pings:
        nav=ping["navigation"]
        rot=rotation(nav["roll_rad"],nav["pitch_rad"],nav["heading_rad"])
        # PoSSM has ideal linear navigation. x=forward, y=starboard, z=down.
        origin=np.array([(ping["number"]-1)*config["advance_m"],0.,-nav["altitude_m"]])
        tx=rot@tx_local+origin
        rx=(rot@rx_local.T).T+origin
        txs.extend([tx]*len(rx));rxs.extend(rx)
        speeds.extend([nav["sound_speed_mps"]]*len(rx))
    return np.asarray(txs),np.asarray(rxs),np.asarray(speeds)


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("folder",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    config,kernel,pings,manifest=read_gdf(args.folder.glob("*.gdf"))
    summary={"config":config,"kernel_samples":len(kernel),"kernel_first":str(kernel[:4]),
             "pings":len(pings),"raw_shape":[len(pings),*pings[0]["iq"].shape],
             "ping_number_range":[pings[0]["number"],pings[-1]["number"]],
             "first_navigation":pings[0]["navigation"],"last_navigation":pings[-1]["navigation"],
             "manifest":manifest}
    print(json.dumps(summary,indent=2))
    if args.output:args.output.write_text(json.dumps(summary,indent=2)+"\n")
