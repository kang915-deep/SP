"""Report durable Hound download progress without reading archive contents."""
import os
from pathlib import Path
import re
import sys

from hound_archive_manifest import ITEMS


def show(data):
    archives = data / "downloads"
    if not archives.is_dir():
        raise SystemExit(f"找不到下载目录 {archives}")
    total = received = complete = 0
    partial, pending = [], []
    for item in ITEMS:  # The downloader follows manifest order, not numeric order.
        size = item["bytes"]
        name = Path(item["name"]).name
        target = archives / name
        part = archives / (name + ".part")
        folder = archives / (name + ".part.ranges")
        if target.is_file():
            count = target.stat().st_size
        else:
            offset = part.stat().st_size if part.is_file() else 0
            count = offset
            if folder.is_dir():
                for entry in folder.iterdir():
                    match = re.fullmatch(r"(\d+)-(\d+)", entry.name)
                    if match:
                        start, end = map(int, match.groups())
                        if offset <= start <= end < size and entry.stat().st_size == end-start+1:
                            count += end-start+1
        if count > size:
            raise SystemExit(f"{name} 的文件大小与清单不符，请核对下载目录")
        total += size
        received += count
        if count == size:
            complete += 1
        else:
            pending.append(name)
            if count:
                partial.append(f"{name}（{count/size:.1%}）")
    print(f"Hound 已下齐 {complete}/{len(ITEMS)} 包，还剩 {len(pending)} 包（含部分下载的包）")
    print(f"已落盘 {received/2**30:.2f}/{total/2**30:.2f} GiB，还需下载 {(total-received)/2**30:.2f} GiB")
    if partial:
        print("部分落盘的包：" + "、".join(partial))
    elif pending:
        print("清单中下一个未下齐的包：" + pending[0])
    # Identify the transfer actually in flight, even before its first range is committed.
    proc = Path("/proc")
    if proc.is_dir():
        for pid in proc.iterdir():
            if not pid.name.isdigit():
                continue
            try:
                argv = (pid / "cmdline").read_bytes().split(b"\0")
            except (PermissionError, FileNotFoundError, ProcessLookupError):
                continue
            if os.path.basename(argv[0]) != b"curl":
                continue
            for arg in argv:
                match = re.search(rb"/train_300k/(chunk_\d+\.tar\.gz)(?:\?|$)", arg)
                if match:
                    print("本机正在请求：" + match[1].decode())
    if (data / "media_ready.json").is_file():
        print("训练媒体已就绪")
    elif not pending:
        print("Hound 下载已齐，训练媒体验收尚未完成")


if __name__ == "__main__":
    show(Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd() / "qwen3-vega/data")
