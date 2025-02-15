"""
Training script for the CommunicativeAgent model.

This script handles the training of multiple CommunicativeAgent models with different 
configurations, including:
- Multiple training runs with different random seeds
- Various jerk loss weights for regularization
- Different dataset combinations

The trained models and their metrics are saved for later analysis and evaluation.
"""

import os
import sys
import pickle

# Add project root to Python path
print("Current working directory:", os.getcwd())
sys.path.insert(0, "/Users/vpaulien/Inner_Speech/agent")

from lib import utils
from lib.nn.data_scaler import DataScaler
from communicative_agent import CommunicativeAgent
from trainer import Trainer
import torch

# Training configuration constants
NB_TRAINING = 5  # Number of training runs with different random seeds
DATASETS_NAME = ["pb2007"]  # Dataset(s) to use for instance ["pb2007", "msak0", "fsew0"]
FRAME_PADDING = [2]  # Padding frames around each sequence
JERK_LOSS_WEIGHTS = [0.1, 0.15]  # Weights for jerk regularization loss
NB_DERIVATIVES = [0]  # Number of derivatives to compute for articulatory features
ART_TYPE = "art_params"  # Type of articulatory parameters to use


def train_agent(agent, save_path):
    """
    Train a single CommunicativeAgent model and save the results.

    Args:
        agent (CommunicativeAgent): The agent instance to train
        save_path (str): Directory path to save the trained model and metrics

    The function:
    1. Sets up training components (dataloaders, optimizers, loss functions)
    2. Initializes and configures the trainer
    3. Trains the model and records metrics
    4. Saves the trained model and metrics to disk
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Training %s" % save_path)
    if os.path.isdir(save_path):
        print("Model already trained, skipping...")
        print()
        return

    # Get training components from agent
    dataloaders = agent.get_dataloaders()
    optimizers = agent.get_optimizers()
    losses_fn = agent.get_losses_fn()

    # Initialize sound scalers for both synthesizer and agent
    sound_scalers = {
        "synthesizer": DataScaler.from_standard_scaler(
            agent.synthesizer.sound_scaler
        ).to(device),
        "agent": DataScaler.from_standard_scaler(agent.sound_scaler).to(device),
    }

    # Configure and initialize trainer
    trainer = Trainer(
        agent.nn,
        optimizers,
        *dataloaders,
        losses_fn,
        agent.config["training"]["max_epochs"],
        agent.config["training"]["patience"],
        agent.synthesizer,
        sound_scalers,
        "./out/checkpoint.pt",
    )
    
    # Train model and record metrics
    metrics_record = trainer.train()

    # Save trained model and metrics
    utils.mkdir(save_path)
    agent.save(save_path)
    with open(save_path + "/metrics.pickle", "wb") as f:
        pickle.dump(metrics_record, f)


def main():
    """
    Main training loop that:
    1. Loads model configurations
    2. Iterates through training runs with different random seeds
    3. Tests different jerk loss weights
    4. Trains and saves models for each configuration
    """
    # Load final model and quantizer configurations
    final_configs = utils.read_yaml_file("communicative_agent/communicative_final_configs.yaml")
    final_quantizer_configs = utils.read_yaml_file("quantizer/quantizer_final_configs.yaml")

    # Iterate through model configurations
    for config_name, config in final_configs.items():
        quantizer_name = config_name.split("-")[0]
        quantizer_config = final_quantizer_configs["%s-cepstrum" % quantizer_name]

        # Run multiple training iterations with different random seeds
        for i_training in range(NB_TRAINING):
            # Configure quantizer with current random seed
            quantizer_config["dataset"]["datasplit_seed"] = i_training
            quantizer_signature = utils.get_variable_signature(quantizer_config)

            # Test different jerk loss weights
            for jerk_loss_weight in JERK_LOSS_WEIGHTS:
                # Update config with current settings
                config["sound_quantizer"]["name"] = "%s-%s" % (quantizer_signature, i_training)
                config["training"]["jerk_loss_weight"] = jerk_loss_weight

                # Initialize and train agent
                agent = CommunicativeAgent(config)
                signature = agent.get_signature()
                save_path = "out/communicative_agent/%s-%s" % (signature, i_training)
                train_agent(agent, save_path)


if __name__ == "__main__":
    main()
