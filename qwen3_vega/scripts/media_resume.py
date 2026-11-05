"""Durable progress and extraction receipts for interrupted media preparation."""
import json
from pathlib import Path
import re


def archive_priority(item, archives):
    target = archives / Path(item['name']).name
    if target.is_file() and target.stat().st_size == item['bytes']:
        return 2
    part = target.with_name(target.name+'.part')
    offset = part.stat().st_size if part.is_file() else 0
    received = offset
    folder = part.with_name(part.name+'.ranges')
    if folder.is_dir():
        for entry in folder.iterdir():
            match = re.fullmatch(r'(\d+)-(\d+)', entry.name)
            if match:
                start,end = map(int,match.groups())
                if offset <= start <= end < item['bytes'] and entry.stat().st_size == end-start+1:
                    received += end-start+1
    if received == item['bytes']:
        return 2
    return 0 if part.exists() or folder.exists() else 1


def restored_coverage(receipt, base, identity):
    if not receipt.is_file():
        return None
    saved = json.loads(receipt.read_text())
    if saved['identity'] != identity:
        return None
    for relative, size in saved['files']:
        path = base/relative
        if not path.is_file() or path.stat().st_size != size:
            return None
    return set(saved['covered'])
