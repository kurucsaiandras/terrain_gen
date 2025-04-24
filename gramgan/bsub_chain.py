import sys
import re
import subprocess

def read_script() -> str:
    script = ""
    for line in sys.stdin:
        script += line
    return script

def submit_script(script: str) -> int:

    result = subprocess.run(
        ["bsub"],
        input=script.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    output = result.stdout.decode()
    error = result.stderr.decode()
    print(output, file=sys.stdout)
    print(error, file=sys.stderr)

    if result.returncode != 0:
        sys.exit(result.returncode)
    
    m = re.search(
        r"Job <(\d+)>",
        output,
    )

    return int(m.group(1))

if __name__ == "__main__":
    script = read_script()
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    prev_id = None

    for i in range(count):
        augmented_script = re.sub(
            r"#BSUB\s+-J\s+([a-zA-Z0-9_]+)",
            lambda m: m.group(0) + f"_{i}",
            script
        )

        if prev_id is not None:
            augmented_script += f"#BSUB -w done({prev_id})\n"

        prev_id = submit_script(augmented_script)
        
    