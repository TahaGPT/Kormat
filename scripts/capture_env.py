"""Write results/env/cpu_laptop.json: hardware + library versions of the machine all numbers come from."""
import json
import platform
import subprocess
import sys

from kormat.common.paths import RESULTS
from kormat.common.runlog import capture_env, git_sha


def cpuinfo() -> dict:
    out = {}
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                out["cpu_model"] = line.split(":", 1)[1].strip()
            if line.startswith("flags") and "flags" not in out:
                fl = line.split(":", 1)[1].split()
                out["flags_of_interest"] = [f for f in ("avx", "avx2", "fma", "avx512f", "avx512_vnni", "f16c") if f in fl]
    except OSError:
        pass
    try:
        out["lscpu"] = {k.strip(): v.strip() for k, v in (l.split(":", 1) for l in subprocess.check_output(["lscpu"], text=True).splitlines() if ":" in l)
                        if k.strip() in ("Thread(s) per core", "Core(s) per socket", "CPU max MHz", "CPU min MHz", "L2 cache", "L3 cache")}
    except Exception:
        pass
    return out


def main():
    env = capture_env()
    env.update(cpuinfo())
    env["git_sha"] = git_sha()
    env["python_executable"] = sys.executable
    env["kernel"] = platform.release()
    try:
        import torch
        env["torch_num_interop_threads"] = torch.get_num_interop_threads()
        env["torch_mkldnn"] = torch.backends.mkldnn.is_available()
    except Exception:
        pass
    p = RESULTS / "env" / "cpu_laptop.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(env, indent=2))
    print(json.dumps(env, indent=2))


if __name__ == "__main__":
    main()
