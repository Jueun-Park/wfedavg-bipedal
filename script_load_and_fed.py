from os import system

if __name__ == "__main__":
    program = "wfedavg_load_and_fed.py"
    for base_index in range(4):
        system(f"python {program} --base-index {base_index}")
