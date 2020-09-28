import json
from multiprocessing import Pool
import subprocess
from pathlib import Path
import os

DATA_FILE = os.environ.get("DATA_FILE", "../data/train.json")
INIT_DIR = os.environ.get("INIT_DIR")
NUM_PROCESSES = int(os.environ.get("NUM_PROCESSES", 3))


def run_opt(js):
    print(js["id"] + " START")
    name = js["id"]
    ss = js["structure"]
    seq = js["sequence"]
    path = str(Path(INIT_DIR) / (name + ".pred1.pdb"))
    command = '/nsp-1.7/.build/nsp opt -name {} -seq {} -ss "{}" -init {}'.format(
        name, seq, ss, path
    )
    process = subprocess.call(command, shell=True)
    print(js["id"] + " DONE")
    return process


if __name__ == "__main__":
    records = []
    with open(DATA_FILE, "r") as f:
        for line in f:
            if len(line.strip()):
                records.append(json.loads(line))
    files = set()
    for file in Path(INIT_DIR).glob("*.pdb"):
        files.add(file.name[:12])
    print("Records before: {}".format(len(records)))
    records = [record for record in records if record["id"] in files]
    print("Records after: {}".format(len(records)))
    with Pool(NUM_PROCESSES) as pool:
        print(pool.map(run_opt, records))
