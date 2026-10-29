# =========================================================
# SHA-256 checksums for every image scored.
# Run from the directory holding the image folders, e.g.:
#   python make_checksums.py contextShortcutData furextra drygrass noise_probe
# Writes checksums/benchmark_sha256.txt with lines "<sha256>  <relative/path>".
# Verify an independently assembled set with:
#   python make_checksums.py --verify checksums/benchmark_sha256.txt <dirs...>
# =========================================================
import sys, os, hashlib

EXTS = ('.jpg', '.jpeg', '.jfif', '.png', '.bmp', '.webp')

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def walk(dirs):
    for d in dirs:
        for root, _, files in os.walk(d):
            for f in sorted(files):
                if f.lower().endswith(EXTS):
                    p = os.path.join(root, f)
                    yield p.replace(os.sep, '/')

def main():
    args = sys.argv[1:]
    if args and args[0] == '--verify':
        ref, dirs = args[1], args[2:]
        want = {}
        for line in open(ref):
            h, _, p = line.strip().partition('  ')
            want[p] = h
        have = {p: sha256(p) for p in walk(dirs)}
        missing = [p for p in want if p not in have]
        extra   = [p for p in have if p not in want]
        bad     = [p for p in want if p in have and have[p] != want[p]]
        print(f"reference {len(want)} | found {len(have)} | "
              f"missing {len(missing)} | unexpected {len(extra)} | mismatched {len(bad)}")
        for label, lst in [("MISSING", missing), ("UNEXPECTED", extra), ("MISMATCH", bad)]:
            for p in lst[:20]:
                print(f"  {label}: {p}")
        sys.exit(1 if (missing or bad) else 0)
    dirs = args or ['.']
    os.makedirs('checksums', exist_ok=True)
    n = 0
    with open('checksums/benchmark_sha256.txt', 'w') as out:
        for p in walk(dirs):
            out.write(f"{sha256(p)}  {p}\n"); n += 1
    print(f"wrote checksums/benchmark_sha256.txt with {n} entries")

if __name__ == '__main__':
    main()
