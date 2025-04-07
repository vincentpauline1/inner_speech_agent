"""
Training script for the ImitativeAgent model.

This script handles the training of multiple ImitativeAgent models with different 
configurations, including:
- Multiple training runs with different random seeds
- Various jerk loss weights and ceiling values for regularization 
- Different dataset combinations

The trained models and their metrics are saved for later analysis and evaluation.
"""

import os
import sys
import pickle

# Add project root to Python path
print("current path:", os.getcwd())
sys.path.insert(0, "/mnt/c/Users/vpaul/OneDrive - CentraleSupelec/Inner_Speech/agent/")

from lib import utils
from lib.nn.data_scaler import DataScaler
from imitative_agent import ImitativeAgent
import torch
from trainer import Trainer

# Training configuration constants
NB_TRAINING = 5  # Number of training runs with different random seeds
ART_MODALITY = "ema"  # Type of articulatory parameters to use
DATASETS_NAME = ["pb2007"]  # Dataset(s) to use  for instance ["pb2007", "msak0", "fsew0"]
JERK_LOSS_CEILS = [0.014]  # Ceiling values for jerk regularization loss
JERK_LOSS_WEIGHTS = [1]  # Weights for jerk regularization loss for instance [0.1, 0.15]


def main():
    """
    Main training loop that:
    1. Loads model configurations
    2. Iterates through training runs with different random seeds
    3. Tests different jerk loss parameters
    4. Trains and saves models for each configuration
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for i_training in range(NB_TRAINING):
        for dataset_name in DATASETS_NAME:
            for jerk_loss_ceil in JERK_LOSS_CEILS:
                for jerk_loss_weight in JERK_LOSS_WEIGHTS:
                    # Load and update agent configuration
                    agent_config = utils.read_yaml_file(
                        "imitative_agent/imitative_config.yaml"
                    )
                    agent_config["dataset"]["names"] = [dataset_name]
                    agent_config["training"]["jerk_loss_ceil"] = jerk_loss_ceil
                    agent_config["training"]["jerk_loss_weight"] = jerk_loss_weight
                    agent_config["synthesizer"]["name"] = "ea587b76c95fecef01cfd16c7f5f289d-0/"
                        # "dn=%s-hl=256,256,256,256-in=%s-out=cepstrum-0"
                        # % (dataset_name, ART_MODALITY)

                    # Initialize agent and setup save path
                    agent = ImitativeAgent(agent_config)
                    signature = agent.get_signature()
                    save_path = "out/imitative_agent/%s-%s" % (signature, i_training)

                    print("Training %s (i_training=%s)" % (signature, i_training))
                    if os.path.isdir(save_path):
                        print("Already done")
                        print()
                        continue

                    # Get training components from agent
                    dataloaders = agent.get_dataloaders()
                    optimizers = agent.get_optimizers()
                    losses_fn = agent.get_losses_fn()

                    # Initialize sound scalers for both synthesizer and agent
                    sound_scalers = {
                        "synthesizer": DataScaler.from_standard_scaler(
                            agent.synthesizer.sound_scaler
                        ).to(device),
                        "agent": DataScaler.from_standard_scaler(agent.sound_scaler).to(
                            device
                        ),
                    }

                    # Configure and initialize trainer
                    trainer = Trainer(
                        agent.nn,
                        optimizers,
                        *dataloaders,
                        losses_fn,
                        agent_config["training"]["max_epochs"],
                        agent_config["training"]["patience"],
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


if __name__ == "__main__":
    main()
