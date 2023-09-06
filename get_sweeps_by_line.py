import shutil
from pathlib import Path
import re

destination_folder = Path("/home/wombat_share/laurito/elk_reporters/sweeps_llama_2_7b_hf")
destination_folder.mkdir(parents=True, exist_ok=True)


def remove_ansi_escape_codes(s):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', s).replace("\n", "").strip()

with open("not-133-sweep-out-14.txt","r") as fi:
    lines = []
    for ln in fi:
        key = "Saving sweep results to"
        if ln.startswith(key):
            source = remove_ansi_escape_codes(ln[len(key):])
            name = source.split("/")[-1]
            destination = destination_folder.joinpath(name)
            
            if Path(source).exists():
                shutil.copy(source, destination)
            else:
                print("path does not exist", source)