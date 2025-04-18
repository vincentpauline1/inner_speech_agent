#!/usr/bin/env python3
"""
Training script for the inner speech agent model to learn both modes (inner/overt) concurrently.

This script trains an imitative agent that learns to map between speech sounds and 
articulatory movements. It supports training with different loss functions including
jerk loss for smoother articulator trajectories.
"""

import os
import sys
import pickle
print("current path:", os.getcwd())
sys.path.insert(0, "/mnt/c/Users/vpaul/Inner_Speech/agent/")

from lib import utils
from lib.nn.data_scaler import DataScaler
from imitative_agent import ImitativeAgent
import torch
from trainer import Trainer

# Training configuration
NB_TRAINING = 1  # Number of training runs with different random seeds
ART_MODALITY = "ema"  # Articulatory modality: EMA (Electromagnetic Articulography) data
DATASETS_NAME = ["pb2007"]  # Dataset to use for training (for instance ["pb2007", "msak0", "fsew0"])
JERK_LOSS_CEILS = [0.014]  # Maximum allowed jerk loss
JERK_LOSS_WEIGHTS = [1]  # Weight of jerk loss in total loss function


def main():
    """
    Main training loop that:
    1. Initializes the inner speech agent with the specified configuration
    2. Sets up data loaders, optimizers, and loss functions
    3. Trains the model using the Trainer class
    4. Saves the trained model and training metrics
    """
    # Set device based on GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Iterate through training configurations
    for i_training in range(NB_TRAINING):
        for dataset_name in DATASETS_NAME:
            for jerk_loss_ceil in JERK_LOSS_CEILS:
                for jerk_loss_weight in JERK_LOSS_WEIGHTS:
                    # Load and customize agent configuration
                    agent_config = utils.read_yaml_file(
                        "imitative_agent_inner_speech_silent_single_backprop/imitative_config.yaml"
                    )
                    agent_config["dataset"]["names"] = [dataset_name]
                    agent_config["training"]["jerk_loss_ceil"] = jerk_loss_ceil
                    agent_config["training"]["jerk_loss_weight"] = jerk_loss_weight
                    agent_config["synthesizer"]["name"] =  "ea587b76c95fecef01cfd16c7f5f289d-0/" # Initialize the synthesizer with the pretrained synthesizer model
                        # "dn=%s-hl=256,256,256,256-in=%s-out=cepstrum-0"
                        # % (dataset_name, ART_MODALITY)
                    

                    agent = ImitativeAgent(agent_config)
                    signature = agent.get_signature()
                    save_path = "out/imitative_agent_inner_speech_silent_overt_11/%s-%s" % (signature, i_training)

                    print("Training %s (i_training=%s)" % (signature, i_training))
                    if os.path.isdir(save_path):
                        print("Already done")
                        print()
                        continue

                    # Set up training components
                    dataloaders = agent.get_dataloaders()
                    optimizers = agent.get_optimizers()
                    losses_fn = agent.get_losses_fn()

                    # Initialize sound scalers for data normalization
                    sound_scalers = {
                        "synthesizer": DataScaler.from_standard_scaler(
                            agent.synthesizer.sound_scaler
                        ).to(device),
                        "agent": DataScaler.from_standard_scaler(agent.sound_scaler).to(
                            device
                        ),
                    }

                    # Initialize trainer and start training
                    trainer = Trainer(
                        agent.nn,
                        optimizers,
                        *dataloaders,
                        losses_fn,
                        agent_config["training"]["max_epochs"],
                        agent_config["training"]["patience"],
                        agent.synthesizer,
                        sound_scalers,
                        "./out/imitative_agent_inner_speech_silent_dataloader_directexpanded/checkpoint.pt",
                    )
                    metrics_record = trainer.train()

                    # Save trained model and metrics
                    utils.mkdir(save_path)
                    agent.save(save_path)
                    with open(save_path + "/metrics.pickle", "wb") as f:
                        pickle.dump(metrics_record, f)
                    print(f"Training completed. Model and metrics saved to {save_path}")


if __name__ == "__main__":
    main()
