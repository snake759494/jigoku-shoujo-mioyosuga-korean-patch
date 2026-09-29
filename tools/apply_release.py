"""원본 ISO 의 해시를 확인하고 xdelta 를 적용한 뒤 결과 해시까지 검사한다.

사용:
  python tools/apply_release.py --xdelta <xdelta3.exe> --source <원본 ISO> --patch <xdelta> --output <새 ISO>

기존 출력 파일은 덮어쓰지 않는다. 원본·패치·결과 중 하나라도 해시가 다르면 실패로 끝낸다.
"""
import argparse, hashlib, os, subprocess, sys

SOURCE_SIZE = 1534787584
SOURCE_SHA256 = "942edac33f7b8be0ab571fb5fd92d93fb6a3ffd9d6229a2f6441ed0f3bb3e0c1"
PATCH_SHA256 = "0917540aca47d45be1d1ba23cfdcaefd850a4154bb726158df79e98bbda3a070"
OUTPUT_SIZE = 1534787584
OUTPUT_SHA256 = "07c3ff72d45eb63e7e631a9966f975b20e679cfe91c14d48b686676aee7efc39"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def check(label, path, size, digest):
    n = os.path.getsize(path)
    if size is not None and n != size:
        print(f"{label}: 크기 {n:,} != {size:,}"); return False
    d = sha256(path)
    if d != digest:
        print(f"{label}: SHA-256 불일치\n  실제 {d}\n  기대 {digest}"); return False
    print(f"{label}: OK ({n:,} B)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xdelta", required=True, help="xdelta3 실행 파일")
    ap.add_argument("--source", required=True, help="수정되지 않은 일본판 ISO")
    ap.add_argument("--patch", required=True, help="릴리즈 xdelta")
    ap.add_argument("--output", required=True, help="만들 ISO (기존 파일이면 중단)")
    a = ap.parse_args()
    if os.path.exists(a.output):
        print(f"출력 파일이 이미 있습니다: {a.output}"); sys.exit(1)
    if not (check("원본 ISO", a.source, SOURCE_SIZE, SOURCE_SHA256) and check("xdelta", a.patch, None, PATCH_SHA256)):
        sys.exit(1)
    r = subprocess.run([a.xdelta, "-d", "-s", a.source, a.patch, a.output])
    if r.returncode != 0:
        print(f"xdelta 적용 실패 (exit {r.returncode})"); sys.exit(1)
    if not check("결과 ISO", a.output, OUTPUT_SIZE, OUTPUT_SHA256):
        sys.exit(1)
    print("완료")


if __name__ == "__main__":
    main()
