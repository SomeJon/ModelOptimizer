import json
import os
import math

LOADED_TESTS_PATH = './data/loaded_tests.json'
SLURM_TESTS_DIR = "./slurm_tests"
TESTS_PER_FILE = 60  # Adjust as needed
START_NUMBER = 15


def split_loaded_tests():
    # Ensure the output directory exists
    os.makedirs(SLURM_TESTS_DIR, exist_ok=True)

    # Load the loaded_tests.json
    with open(LOADED_TESTS_PATH, 'r') as f:
        loaded_tests = json.load(f)

    total_tests = len(loaded_tests)
    num_files = math.ceil(total_tests / TESTS_PER_FILE)

    for i in range(num_files):
        start_idx = i * TESTS_PER_FILE
        end_idx = start_idx + TESTS_PER_FILE
        slurm_tests = loaded_tests[start_idx:end_idx]

        slurm_tests_filename = f"slurm_tests{i+START_NUMBER}.json"
        slurm_tests_path = os.path.join(SLURM_TESTS_DIR, slurm_tests_filename)

        with open(slurm_tests_path, 'w') as f:
            json.dump(slurm_tests, f, indent=4)

        print(f"Generated {slurm_tests_filename} with tests {start_idx + 1} to {min(end_idx, total_tests)}")


if __name__ == "__main__":
    split_loaded_tests()
