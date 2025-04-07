import os
from pathlib import Path

# This script modifies .lab files that contain phoneme timing information
# The goal is to adjust the timing when the audio has been sped up by 2x
# It divides all start/end timestamps by 4 to match the new audio duration
# This is used to process lab files in the pb2007_speedx2 dataset

def modify_lab_files(directory: Path):
    """
    Modify all .lab files in the specified directory by dividing the start and end
    interval values by 4. The modified files overwrite the original files.

    Parameters:
    - directory (Path): The path to the directory containing .lab files, also directory where the output will be written.
    """
    if not directory.exists():
        print(f"Error: The directory {directory} does not exist.")
        return

    if not directory.is_dir():
        print(f"Error: The path {directory} is not a directory.")
        return

    # Iterate over all .lab files in the directory
    lab_files = list(directory.glob("*.lab"))
    if not lab_files:
        print(f"No .lab files found in {directory}.")
        return

    print(f"Found {len(lab_files)} .lab file(s) in {directory}.\n")

    for lab_file in lab_files:
        print(f"Processing file: {lab_file.name}")
        try:
            # Read all lines from the .lab file
            with lab_file.open("r", encoding="utf-8") as file:
                lines = file.readlines()

            modified_lines = []
            line_number = 0
            for line in lines:
                line_number += 1
                stripped_line = line.strip()
                if not stripped_line:
                    # Skip empty lines
                    modified_lines.append(line)
                    continue

                parts = stripped_line.split()
                if len(parts) != 3:
                    print(f"  Warning: Line {line_number} in {lab_file.name} does not have exactly 3 parts. Skipping line.")
                    modified_lines.append(line)
                    continue

                try:
                    start, end, label = parts
                    start_new = int(int(start) / 4)
                    end_new = int(int(end) / 4)
                    modified_line = f"{start_new} {end_new} {label}\n"
                    modified_lines.append(modified_line)
                except ValueError:
                    print(f"  Warning: Non-integer start/end values on line {line_number} in {lab_file.name}. Skipping line.")
                    modified_lines.append(line)
                    continue

            # Write the modified lines back to the same .lab file
            with lab_file.open("w", encoding="utf-8") as file:
                file.writelines(modified_lines)

            print(f"  Successfully modified {lab_file.name}.\n")

        except Exception as e:
            print(f"  Error processing {lab_file.name}: {e}\n")

    print("All .lab files have been processed.")

if __name__ == "__main__":
    # Define the target directory
    target_directory = Path("./datasets/pb2007_speedx2/lab/")
    modify_lab_files(target_directory)
