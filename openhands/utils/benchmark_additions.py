import fcntl
import json
import os
from pathlib import Path


def safe_append(path: Path, text: str):
    # Open in append+ mode so writes always go to end
    with path.open('a+') as f:
        # Acquire exclusive lock (blocks until free)
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(text)
            f.flush()  # ensure it hits disk
        finally:
            # Release the lock
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def kill_instance(output_file: str):
    with open(output_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            assert 'sid' in data, f"Missing 'sid' in line: {line}"
            sid_to_kill = 'openhands-runtime-' + data['sid']
            break

    print('sid to kill: ', sid_to_kill)
    print(f"docker ps -q --filter 'name={sid_to_kill}'")
    os.system(f"docker ps -q --filter 'name={sid_to_kill}' | xargs -r docker kill")
