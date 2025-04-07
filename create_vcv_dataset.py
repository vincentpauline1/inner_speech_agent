# This script processes audio files and their corresponding label files to create repeated VCV sequences
# It takes VCV audio files from pb2007_speedx2 dataset and repeats each file 3 times
# For each repetition, it adjusts the timing labels accordingly to match the concatenated audio
# The output is saved in vcv_speedx2 dataset with repeated audio and adjusted labels

import os
import sys
import soundfile as sf
import numpy as np
from pathlib import Path
import logging

def setup_logging():
    """Configure the logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def get_item_ids(start: int, end: int) -> list:
    """
    Generate a list of item IDs formatted as 'item_XXXX'.

    Args:
        start (int): Starting item number (inclusive).
        end (int): Ending item number (inclusive).

    Returns:
        list: List of formatted item IDs.
    """
    return [f"item_{i:04d}" for i in range(start, end + 1)]

def read_label_file(label_path: Path) -> list:
    """
    Read a label file and return a list of label entries.

    Args:
        label_path (Path): Path to the label file.

    Returns:
        list: List of tuples containing (start_time, end_time, label).
    """
    labels = []
    with label_path.open('r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                logging.warning(f"Invalid label format in {label_path}: {line.strip()}")
                continue
            start, end, label = parts
            try:
                labels.append((int(start), int(end), label))
            except ValueError:
                logging.warning(f"Non-integer timestamp in {label_path}: {line.strip()}")
    return labels

def write_label_file(labels: list, label_path: Path):
    """
    Write label entries to a label file.

    Args:
        labels (list): List of tuples containing (start_time, end_time, label).
        label_path (Path): Path to the output label file.
    """
    with label_path.open('w') as f:
        for start, end, label in labels:
            f.write(f"{start} {end} {label}\n")

def concatenate_audio(audio_data: np.ndarray, repetitions: int) -> np.ndarray:
    """
    Concatenate audio data multiple times.

    Args:
        audio_data (np.ndarray): Original audio data.
        repetitions (int): Number of times to repeat the audio.

    Returns:
        np.ndarray: Concatenated audio data.
    """
    return np.tile(audio_data, repetitions)

def adjust_labels(original_labels: list, original_duration: int, repetitions: int) -> list:
    """
    Adjust label timings for concatenated audio.

    Args:
        original_labels (list): Original label entries.
        original_duration (int): Duration of the original audio in milliseconds.
        repetitions (int): Number of repetitions.

    Returns:
        list: Adjusted label entries.
    """
    new_labels = []
    for rep in range(repetitions):
        time_shift = rep * original_duration
        for start, end, label in original_labels:
            new_start = start + time_shift
            new_end = end + time_shift
            new_labels.append((new_start, new_end, label))
    return new_labels

def process_item(item_id: str, 
                wav_input_dir: Path, 
                lab_input_dir: Path, 
                wav_output_dir: Path, 
                lab_output_dir: Path, 
                repetitions: int =3):
    """
    Process a single item: concatenate audio and adjust labels.

    Args:
        item_id (str): The item identifier.
        wav_input_dir (Path): Directory containing input wav files.
        lab_input_dir (Path): Directory containing input label files.
        wav_output_dir (Path): Directory to save concatenated wav files.
        lab_output_dir (Path): Directory to save adjusted label files.
        repetitions (int): Number of times to repeat the audio.
    """
    wav_input_path = wav_input_dir / f"{item_id}.wav"
    lab_input_path = lab_input_dir / f"{item_id}.lab"
    wav_output_path = wav_output_dir / f"{item_id}.wav"
    lab_output_path = lab_output_dir / f"{item_id}.lab"

    # Check if input files exist
    if not wav_input_path.exists():
        logging.warning(f"WAV file not found: {wav_input_path}. Skipping.")
        return
    if not lab_input_path.exists():
        logging.warning(f"LABEL file not found: {lab_input_path}. Skipping.")
        return

    try:
        # Read audio
        audio_data, samplerate = sf.read(wav_input_path)
        logging.debug(f"Read WAV file: {wav_input_path} with samplerate {samplerate}")

        # Concatenate audio
        concatenated_audio = concatenate_audio(audio_data, repetitions)
        logging.debug(f"Concatenated audio length: {len(concatenated_audio)} samples")

        # Write concatenated audio
        sf.write(wav_output_path, concatenated_audio, samplerate)
        logging.info(f"Saved concatenated WAV: {wav_output_path}")

        # Read labels
        original_labels = read_label_file(lab_input_path)
        if not original_labels:
            logging.warning(f"No valid labels found in {lab_input_path}. Skipping label processing.")
            return

        # Determine original duration from labels
        original_duration = max(end for _, end, _ in original_labels)
        logging.debug(f"Original duration from labels: {original_duration} ms")

        # Adjust labels for repetitions
        new_labels = adjust_labels(original_labels, original_duration, repetitions)
        logging.debug(f"Total labels after adjustment: {len(new_labels)}")

        # Write new label file
        write_label_file(new_labels, lab_output_path)
        logging.info(f"Saved adjusted LABEL: {lab_output_path}")

    except Exception as e:
        logging.error(f"Error processing {item_id}: {e}")

def main():
    setup_logging()
    logging.info("Starting the audio and label concatenation process.")

    # Define directories
    base_dir = Path(".")  # Assuming the script is run from the parent directory
    wav_input_dir = base_dir / "datasets" / "pb2007_speedx2" / "wav"
    lab_input_dir = base_dir / "datasets" / "pb2007_speedx2" / "lab"
    wav_output_dir = base_dir / "datasets" / "vcv_speedx2" / "wav"
    lab_output_dir = base_dir / "datasets" / "vcv_speedx2" / "lab"

    # Create output directories if they don't exist
    wav_output_dir.mkdir(parents=True, exist_ok=True)
    lab_output_dir.mkdir(parents=True, exist_ok=True)
    logging.debug(f"Output directories ensured: {wav_output_dir}, {lab_output_dir}")

    # Define item range
    START_ITEM = 18
    END_ITEM = 309
    item_ids = get_item_ids(START_ITEM, END_ITEM)
    logging.info(f"Processing items from {item_ids[0]} to {item_ids[-1]}.")

    # Process each item
    for item_id in item_ids:
        logging.info(f"Processing {item_id}...")
        process_item(
            item_id=item_id,
            wav_input_dir=wav_input_dir,
            lab_input_dir=lab_input_dir,
            wav_output_dir=wav_output_dir,
            lab_output_dir=lab_output_dir,
            repetitions=3
        )

    logging.info("Processing completed successfully.")

if __name__ == "__main__":
    main()
