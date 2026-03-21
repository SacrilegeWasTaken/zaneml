import re
import sys

def get_version(path="build.zig.zon"):
    with open(path) as f:
        content = f.read()
    match = re.search(r'\.version\s*=\s*"([^"]+)"', content)
    if not match:
        print("version not found", file=sys.stderr)
        sys.exit(1)
    print(match.group(1))

get_version()
