from tqdm import tqdm  # Progress bar for loops
from communicative_agent import CommunicativeAgent  # Import the communicative agent model
from lib.dataset_wrapper import Dataset  # Dataset wrapper for handling data
from lib import utils  # Utility functions
from scipy.io import wavfile  # For saving generated audio files
import numpy as np  # Numerical operations
from external import lpcynet  # LPCNet for speech synthesis

# Define datasets to be used in training/testing
DATASETS = [
    ["pb2007"],  # Single dataset case
    ["gb2016", "th2016"],  # Multiple datasets case
]

NB_FOLDS = 5  # Number of cross-validation folds

# List of jerk loss weight values used during training
JERK_LOSS_WEIGHTS = [
    0,     # No jerk loss
    0.15,  # Jerk loss applied
]

def export_agent_repetitions(agent, agent_name, datasets_name, datasplits):
    """
    Generates and exports repeated speech samples from the agent.
    
    Args:
        agent (CommunicativeAgent): The agent performing speech repetitions.
        agent_name (str): Identifier for the agent.
        datasets_name (list): Names of the datasets used.
        datasplits (dict): Data splits (training, validation, test).
    """
    for dataset_name in datasets_name:
        print("%s repeats %s" % (agent_name, dataset_name))
        
        # Load dataset
        dataset = Dataset(dataset_name)
        
        # Extract required sound data from dataset
        sound_type = agent.sound_quantizer.config["dataset"]["data_types"]
        items_sound = dataset.get_items_data(sound_type)
        items_source = dataset.get_items_data("source")

        # Define export directories for different representations of repetitions
        repetition_art_export_dir = "./datasets/%s/kfold_art_%s" % (dataset_name, agent_name)
        repetition_cepstrum_export_dir = "./datasets/%s/kfold_cepstrum_%s" % (dataset_name, agent_name)
        repetition_wav_export_dir = "./datasets/%s/kfold_wav_%s" % (dataset_name, agent_name)

        # Create directories if they don't exist
        utils.mkdir(repetition_art_export_dir)
        utils.mkdir(repetition_cepstrum_export_dir)
        utils.mkdir(repetition_wav_export_dir)

        # Get test samples from the dataset split
        test_items = datasplits[dataset_name][2]

        # Process each test sample
        for item_name in tqdm(test_items):  # Show progress bar for processing
            item_sound = items_sound[item_name]  # Retrieve original sound sample
            
            # Generate repetition from the agent
            repetition = agent.repeat(item_sound)

            # Export estimated articulatory parameters
            repetition_art = repetition["art_estimated"]
            repetition_art_file_path = "%s/%s.bin" % (repetition_art_export_dir, item_name)
            repetition_art.tofile(repetition_art_file_path)

            # Export cepstral representation of the repeated sound
            repetition_cepstrum = repetition["sound_repeated"]
            repetition_cepstrum_file_path = "%s/%s.bin" % (repetition_cepstrum_export_dir, item_name)
            repetition_cepstrum.tofile(repetition_cepstrum_file_path)

            # Prepare input for speech synthesis
            item_source = items_source[item_name]
            repetition_lpcnet_features = np.concatenate((repetition_cepstrum, item_source), axis=1)
            repetition_lpcnet_features = dataset.cut_item_silences(item_name, repetition_lpcnet_features)

            # Synthesize waveform using LPCNet
            repetition_wav = lpcynet.synthesize_frames(repetition_lpcnet_features)
            repetition_wav_file_path = "%s/%s.wav" % (repetition_wav_export_dir, item_name)
            wavfile.write(repetition_wav_file_path, 16000, repetition_wav)  # Save audio file

def main():
    """
    Main function to iterate over datasets, folds, and jerk loss weights,
    loading the corresponding agent and exporting its repeated speech samples.
    """
    for datasets_name in DATASETS:
        datasets_key = ",".join(datasets_name)  # Create a key representing the dataset combination

        for i_fold in range(NB_FOLDS):  # Iterate through cross-validation folds
            for jerk_loss_weight in JERK_LOSS_WEIGHTS:  # Iterate through different jerk loss weight settings
                # Define the path where the agent's model is stored
                save_path = "out/communicative_agent/kfold-%s-jerk=%s-%s" % (datasets_key, jerk_loss_weight, i_fold)
                
                # Reload the trained agent from storage
                agent = CommunicativeAgent.reload(save_path)
                
                # Retrieve dataset splits associated with the agent
                agent_datasplits = agent.sound_quantizer.datasplits
                
                # Construct agent identifier based on dataset and jerk weight
                agent_name = "%s-jerk=%s" % (datasets_key, jerk_loss_weight)
                
                # Export the agent's speech repetitions
                export_agent_repetitions(agent, agent_name, datasets_name, agent_datasplits)

# Entry point for script execution
if __name__ == "__main__":
    main()
