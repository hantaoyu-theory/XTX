import yaml
import subprocess
import sys

# Path to your YAML config and train.py
YAML_PATH = 'experiments.yaml'
TRAIN_SCRIPT = 'train.py'
DATA_PATH = 'train.csv.gz'  # Change if needed

with open(YAML_PATH, 'r') as f:
    config = yaml.safe_load(f)

for exp in config['experiments']:
    cmd = [sys.executable, TRAIN_SCRIPT, '--data', DATA_PATH]
    for k, v in exp.items():
        cmd.append(f'--{k}')
        cmd.append(str(v))
    print('Launching:', ' '.join(cmd))
    subprocess.Popen(cmd)

print('All experiments launched!')
