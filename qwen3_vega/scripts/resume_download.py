"""Restart only this repository's active data-preparation launcher; preserve all downloads."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time


def snapshot():
    result = {}
    for directory in Path('/proc').glob('[0-9]*'):
        try:
            raw = (directory / 'cmdline').read_bytes()
            stat = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if not raw:
            continue  # Exited zombies have no files or launcher locks left open.
        result[int(directory.name)] = (int(stat[1]), stat[19], raw.decode(errors='replace').strip('\0').split('\0'))
    return result


def descendants(table, root):
    selected = {root}
    while True:
        children = {pid for pid, (ppid, _, _) in table.items() if ppid in selected}
        if children <= selected:
            return selected
        selected |= children


def send(pid, sig):
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass  # A finished curl may exit between the process snapshot and the signal.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_disk', type=Path)
    parser.add_argument('gpus', help='The same four GPUs passed to start.sh')
    parser.add_argument('--workers', type=int, default=8, choices=range(1,9))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    work = (args.data_disk / 'qwen3-vega').resolve()
    table = snapshot()
    entry = str(repo / 'start.sh')
    roots = [pid for pid, (_, _, cmd) in table.items()
             if entry in cmd and '--worker' in cmd and str(work) in cmd]
    if len(roots) > 1:
        raise SystemExit('Multiple matching launchers; left unchanged.')
    if roots:
        root = roots[0]
        family = descendants(table, root)
        prepare = str(repo / 'qwen3_vega/scripts/prepare_data.py')
        active = [pid for pid in family if prepare in table[pid][2] and str(work/'data') in table[pid][2]]
        if len(active) != 1:
            raise SystemExit('This launcher is not in data preparation. No training or evaluation was stopped.')
        allowed = descendants(table, active[0]) | {root}
        # Only the known one_click parent is allowed between prepare_data and start.sh.
        parent = table[active[0]][0]
        expected = str(repo / 'qwen3_vega/scripts/one_click.py')
        if parent not in table or not any(v == expected or v.endswith('/qwen3_vega/scripts/one_click.py') or v == 'qwen3_vega/scripts/one_click.py' for v in table[parent][2]):
            raise SystemExit('Unrecognized download parent; left unchanged.')
        allowed.add(parent)
        if family != allowed:
            raise SystemExit('Launcher has other active stages; left unchanged.')
        print('Restarting only download launcher PIDs:', sorted(family), flush=True)
        if args.dry_run:
            return
        # Freeze the two parents first to prevent them starting the next stage during shutdown.
        for pid in (root, parent):
            send(pid, signal.SIGSTOP)
        try:
            fresh = snapshot()
            if not descendants(fresh, root) <= family:
                raise SystemExit('Launcher changed stage during inspection; left running.')
            for pid in sorted(family - {root, parent}, reverse=True):
                current = fresh.get(pid)
                if current and current[1] == table[pid][1]:
                    send(pid, signal.SIGTERM)
            for pid in (parent, root):
                send(pid, signal.SIGTERM)
        finally:
            for pid in (parent, root):
                send(pid, signal.SIGCONT)
        deadline = time.monotonic()+10
        while time.monotonic()<deadline:
            now = snapshot()
            remaining = [pid for pid in family if pid in now and now[pid][1] == table[pid][1]]
            if not remaining:
                break
            time.sleep(.5)
        else:
            raise SystemExit('A download process is still exiting; files retained. Rerun this command after it exits.')
    elif args.dry_run:
        print('No matching launcher; would start the existing workspace.')
        return
    env = dict(os.environ, HOUND_SOURCE='modelscope', VEGA_DOWNLOAD_WORKERS=str(args.workers), VEGA_RANGE_ARCHIVES='1')
    subprocess.run(['bash', entry, str(args.data_disk.resolve()), args.gpus], env=env, check=True, cwd=repo)


if __name__ == '__main__':
    main()
