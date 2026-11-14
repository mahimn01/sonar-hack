"""Fetch only some files out of a big ZIP that lives on a web server.

Why: the gesture video archive on Zenodo is one 22.9 GB zip, but we only need
a few classes. A zip keeps a table of contents at its end, so we read that
table with small HTTP "range" requests and then pull each wanted file with ONE
range request each. Nothing else is downloaded.

How to run (standard library only, nothing to install):
  python3 -I fetch_zip_members.py <zip-url> --list
      read-only: prints a per-folder summary (count and size), writes nothing
  python3 -I fetch_zip_members.py <zip-url> --match 'class_(01|02|18|19)/' --out <dir> [--max-files N]
      downloads every member whose path matches the regular expression into <dir>

What it changes: writes the selected files under --out (never outside it).
"""
import argparse
import io
import os
import re
import struct
import sys
import time
import urllib.request
import zipfile
import zlib

UA = {"User-Agent": "Mozilla/5.0 research-client/1.0"}


def http_range(url: str, start: int, end: int, tries: int = 4) -> bytes:
    """Return bytes start..end (inclusive) of the remote file, retrying on errors."""
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except Exception as exc:  # network hiccup or rate limit: wait and retry
            if attempt == tries - 1:
                raise
            print(f"  retry after error: {exc}", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


class HttpRangeFile(io.RawIOBase):
    """Read-only seekable file backed by HTTP range requests (used for the table of contents)."""

    def __init__(self, url: str) -> None:
        req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.url = r.geturl()  # keep the final address after redirects
            self.size = int(r.headers["Content-Range"].split("/")[1])
        self.pos = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = offset if whence == 0 else (self.pos + offset if whence == 1 else self.size + offset)
        return self.pos

    def readinto(self, b) -> int:
        if self.pos >= self.size or len(b) == 0:
            return 0
        data = http_range(self.url, self.pos, min(self.pos + len(b), self.size) - 1)
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def fetch_member(url: str, info: zipfile.ZipInfo) -> bytes:
    """Download one zip member with a single range request and return its unpacked bytes."""
    # The local header is 30 bytes + file name + extra field; ask for a little slack.
    slack = 30 + len(info.filename.encode()) + 1024
    blob = http_range(url, info.header_offset, info.header_offset + slack + info.compress_size)
    if blob[:4] != b"PK\x03\x04":
        raise ValueError(f"bad local header for {info.filename}")
    name_len, extra_len = struct.unpack("<HH", blob[26:30])
    start = 30 + name_len + extra_len
    raw = blob[start : start + info.compress_size]
    if info.compress_type == zipfile.ZIP_STORED:
        data = raw
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(raw, -15)
    else:
        raise ValueError(f"unsupported compression {info.compress_type} for {info.filename}")
    if zlib.crc32(data) != info.CRC:
        raise ValueError(f"checksum mismatch for {info.filename}")
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("--list", action="store_true", help="only print a per-folder summary")
    ap.add_argument("--match", default=None, help="regular expression on the member path")
    ap.add_argument("--out", default=None, help="output directory")
    ap.add_argument("--max-files", type=int, default=0, help="stop after this many files (0 = no limit)")
    ap.add_argument("--depth", type=int, default=2, help="folder depth for the --list summary")
    args = ap.parse_args()

    # Step 1: read the table of contents (a few hundred kB).
    remote = HttpRangeFile(args.url)
    members = [i for i in zipfile.ZipFile(io.BufferedReader(remote, buffer_size=1 << 20)).infolist() if not i.is_dir()]
    print(f"archive: {remote.size/1e9:.2f} GB, {len(members)} files")

    if args.list or not args.match:
        # Step 2a: summary only.
        groups: dict[str, list[int]] = {}
        for i in members:
            key = "/".join(i.filename.split("/")[: args.depth])
            g = groups.setdefault(key, [0, 0])
            g[0] += 1
            g[1] += i.compress_size
        for key in sorted(groups):
            print(f"  {key}: {groups[key][0]} files, {groups[key][1]/1e6:.0f} MB")
        return

    # Step 2b: download the matching members, one request each.
    if not args.out:
        sys.exit("--out is required with --match")
    wanted = [i for i in members if re.search(args.match, i.filename)]
    if args.max_files:
        wanted = wanted[: args.max_files]
    print(f"selected {len(wanted)} files, {sum(i.compress_size for i in wanted)/1e6:.0f} MB")
    out_root = os.path.abspath(args.out)
    for n, info in enumerate(wanted, 1):
        dest = os.path.abspath(os.path.join(out_root, info.filename))
        if not dest.startswith(out_root + os.sep):
            raise ValueError(f"unsafe path in archive: {info.filename}")
        if os.path.exists(dest) and os.path.getsize(dest) == info.file_size:
            continue  # already downloaded
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        data = fetch_member(remote.url, info)
        with open(dest, "wb") as fh:
            fh.write(data)
        print(f"  [{n}/{len(wanted)}] {info.filename} ({len(data)/1e6:.1f} MB)")
        time.sleep(0.6)  # stay well under the server's request-rate limit


if __name__ == "__main__":
    main()
