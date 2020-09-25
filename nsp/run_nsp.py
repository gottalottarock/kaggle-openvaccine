import json
from multiprocessing import Pool
import subprocess
import os

DATA_FILE = os.environ.get("DATA_FILE",'../data/train.json')
NUM_PROCESSES = int(os.environ.get("NUM_PROCESSES", 3))


def run_nsp_on(js):
    print(js['id']+" START")
    name = js['id']
    ss = js['structure']
    seq = js['sequence']
    command = '/nsp-1.7/.build/nsp assemble -name {} -seq {} -ss "{}" -n 1'.format( name, seq, ss)
    process = subprocess.call(command, shell=True)
    print(command)
    print(js['id']+" DONE")
    return process

if __name__ == '__main__':
    records = []
    with open(DATA_FILE,'r') as f:
        for line in f:
            if len(line.strip()):
                records.append(json.loads(line))
    with Pool(NUM_PROCESSES) as pool:
        print(pool.map(run_nsp_on, records))
