import argparse
import json
import os
from ModelRunner.runnable_test import run_tests


def main(config_path, output_dir):
    """
    Main function to run all tests from the config file.

    Args:
        config_path (str): Path to the slurm_tests{number}.json file.
        output_dir (str): Directory to save the results.
    """
    # Load the tests from the config file
    with open(config_path, 'r') as f:
        pending_tests = json.load(f)

    completed_tests = []

    # Run all tests
    completed_tests, successful_tests = run_tests(pending_tests, completed_tests)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract the base file name from the config path
    config_filename = os.path.basename(config_path)  # e.g., slurm_tests1.json

    # Define the results file path with the same name
    results_path = os.path.join(output_dir, config_filename)

    # Save the completed tests results
    with open(results_path, 'w') as f:
        json.dump(completed_tests, f, indent=4)

    print(f"Results saved to {results_path}")

    return completed_tests


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run optimizer tests from a Slurm JSON configuration.")
    parser.add_argument("--config", type=str, required=True, help="Path to the slurm_tests{number}.json file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the results.")
    args = parser.parse_args()

    main(args.config, args.output_dir)
