"""Run the repository's self-contained demonstration from Python."""

import argparse
from pathlib import Path

from borehole_fracture_analysis.demo import run_demo


def main() -> None:
    """Parse an optional output directory and print every generated artifact."""

    parser = argparse.ArgumentParser(description="Run the borehole fracture demo")
    parser.add_argument("--output", type=Path, default=Path("artifacts/demo-python"))
    arguments = parser.parse_args()
    outputs = run_demo(arguments.output)
    for output_name, output_value in outputs.items():
        print(f"{output_name}: {output_value}")


if __name__ == "__main__":
    main()
