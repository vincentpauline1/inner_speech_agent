"""
Script for exporting speech repetitions from trained CommunicativeAgent models using k-fold cross validation.

This script loads trained models and generates repetitions of speech sequences in multiple formats:
- Articulatory trajectories (.bin files)
- Cepstral features (.bin files) 
- Synthesized audio (.wav files)

The models were trained with different configurations:
- Multiple dataset combinations
- K-fold cross validation splits
- Various jerk loss regularization weights

The exported repetitions can be used for analysis and evaluation of model performance.
"""

from tqdm import tqdm
from communicative_agent import CommunicativeAgent
from lib.dataset_wrapper import Dataset
from lib import utils
from scipy.io import wavfile
import numpy as np
from external import lpcynet

# Dataset combinations to evaluate
DATASETS = [
    ["pb2007"],  # Single speaker dataset
    ["gb2016", "th2016"],  # Multi-speaker datasets
]

NB_FOLDS = 5  # Number of cross-validation folds

# Jerk loss weights used during training to regularize articulator movements
JERK_LOSS_WEIGHTS = [
    0,     # No jerk loss
    0.15,  
]

def export_agent_repetitions(agent, agent_name, datasets_name, datasplits):
    """
    Generate and export speech repetitions from a trained agent in multiple formats.
    
    For each test sequence, exports:
    1. Estimated articulatory trajectories (.bin)
    2. Cepstral features of repeated speech (.bin)
    3. Synthesized audio waveforms (.wav)
    
    Args:
        agent (CommunicativeAgent): Trained agent model to generate repetitions
        agent_name (str): Identifier for the agent (used in output paths)
        datasets_name (list): Names of datasets to process
        datasplits (dict): Train/val/test splits for each dataset
    """
    for dataset_name in datasets_name:
        print(f" {agent_name} repeats {dataset_name}")
        
        # Load dataset and extract features
        dataset = Dataset(dataset_name)
        sound_type = agent.sound_quantizer.config["dataset"]["data_types"]
        items_sound = dataset.get_items_data(sound_type)
        items_source = dataset.get_items_data("source")  # Source features for synthesis

        # Setup export directories for different feature types
        base_path = f"./datasets/{dataset_name}/kfold_{agent_name}"
        export_dirs = {
            "art": f"{base_path}_art",
            "cepstrum": f"{base_path}_cepstrum", 
            "wav": f"{base_path}_wav"
        }
        for dir_path in export_dirs.values():
            utils.mkdir(dir_path)

        # Process test sequences from this dataset's split
        test_items = datasplits[dataset_name][2]
        for item_name in tqdm(test_items, desc="Processing sequences"):
            # Generate repetition using agent
            repetition = agent.repeat(items_sound[item_name])
            
            # Export articulatory trajectories
            repetition_art = repetition["art_estimated"]
            repetition_art.tofile(f"{export_dirs['art']}/{item_name}.bin")

            # Export cepstral features
            repetition_cepstrum = repetition["sound_repeated"]
            repetition_cepstrum.tofile(f"{export_dirs['cepstrum']}/{item_name}.bin")

            # Synthesize and export audio using LPCNet
            item_source = items_source[item_name]
            repetition_lpcnet_features = np.concatenate((repetition_cepstrum, item_source), axis=1)
            repetition_lpcnet_features = dataset.cut_item_silences(item_name, repetition_lpcnet_features)
            
            repetition_wav = lpcynet.synthesize_frames(repetition_lpcnet_features)
            wavfile.write(f"{export_dirs['wav']}/{item_name}.wav", 16000, repetition_wav)

def main():
    """
    Main execution function that:
    1. Iterates through dataset combinations
    2. Loads models trained with different configurations:
        - K-fold splits
        - Jerk loss weights
    3. Exports repetitions for each model
    """
    for datasets_name in DATASETS:
        datasets_key = ",".join(datasets_name)
        
        # Process each fold and jerk weight configuration
        for i_fold in range(NB_FOLDS):
            for jerk_loss_weight in JERK_LOSS_WEIGHTS:
                # Load trained model
                model_path = f"out/communicative_agent/kfold-{datasets_key}-jerk={jerk_loss_weight}-{i_fold}"
                agent = CommunicativeAgent.reload(model_path)
                
                # Export repetitions using consistent naming
                agent_name = f"{datasets_key}-jerk={jerk_loss_weight}"
                export_agent_repetitions(
                    agent,
                    agent_name, 
                    datasets_name,
                    agent.sound_quantizer.datasplits
                )

if __name__ == "__main__":
    main()
