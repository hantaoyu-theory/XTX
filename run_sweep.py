import argparse
import itertools
import os
import subprocess
import sys
import time
from datetime import datetime

import yaml


def cartesian_product(grid):
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', default='sweep.yaml')
    ap.add_argument('--max-parallel', type=int, default=None,
                    help='Override launch.max_parallel from sweep.yaml')
    ap.add_argument('--dry-run', action='store_true',
                    help='Print commands only, do not launch')
    ap.add_argument('--limit', type=int, default=None,
                    help='Run only the first N combinations')
    args = ap.parse_args()

    with open(args.sweep, 'r') as f:
        cfg = yaml.safe_load(f)

    script = cfg.get('script', 'train.py')
    data = cfg.get('data', 'train.csv.gz')
    outdir_root = cfg.get('outdir_root', 'results')
    name_template = cfg.get('name_template', 'run-{i}')
    fixed = cfg.get('fixed', {})
    grid = cfg.get('grid', {})

    launch = cfg.get('launch', {})
    max_parallel = args.max_parallel or int(launch.get('max_parallel', 1))
    dry_run = args.dry_run or bool(launch.get('dry_run', False))
    limit = args.limit if args.limit is not None else launch.get('limit', None)
    if isinstance(limit, str) and limit.lower() == 'null':
        limit = None

    # Expand grid
    combos = list(cartesian_product(grid))
    if limit is not None:
        combos = combos[: int(limit)]

    # Prepare commands
    procs = []
    ensure_dir(outdir_root)

    for i, params in enumerate(combos, start=1):
        all_params = {**fixed, **params}
        name = name_template.format(i=i, **all_params)
        outdir = os.path.join(outdir_root, name)
        ensure_dir(outdir)

        # Build command
        cmd = [sys.executable, script, '--data', data, '--outdir', outdir]
        for k, v in all_params.items():
            cmd.extend([f'--{k}', str(v)])

        # Logs
        log_path = os.path.join(outdir, 'run.log')
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'[{ts}] Prepared: {" ".join(cmd)}')

        if dry_run:
            continue

        # Launch with limited parallelism
        stdout = open(log_path, 'w')
        stderr = subprocess.STDOUT
        procs.append((subprocess.Popen(cmd, stdout=stdout, stderr=stderr), stdout))

        # Throttle to max_parallel
        while len([p for p, _ in procs if p.poll() is None]) >= max_parallel:
            # Reap finished
            alive = []
            for p, f in procs:
                if p.poll() is None:
                    alive.append((p, f))
                else:
                    f.close()
            procs = alive
            time.sleep(1)

    # Wait for all to finish
    for p, f in procs:
        p.wait()
        f.close()

    print('Sweep complete.')


if __name__ == '__main__':
    main()
