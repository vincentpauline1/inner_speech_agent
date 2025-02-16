#!/usr/bin/env python3
"""
Script to learn inner speech by fine-tuning the imitative agent.

This script loads a pre-trained overt speech model and fine-tunes it for inner speech
by selectively training only the inner speech output layer while keeping other parameters frozen.

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

# Number of training runs with different random seeds
NB_TRAINING = 30


def main():
    """
    Main training loop that:
    1. Loads a pre-trained overt speech model
    2. Freezes all parameters except the inner speech output layer
    3. Fine-tunes the model using the specified loss functions
    4. Saves the fine-tuned model and training metrics
    """
    # Set device based on GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for i_training in range(NB_TRAINING):
        # Load base configuration
        agent_config = utils.read_yaml_file(
            './imitative_agent_inner_speech_silent_finetune/imitative_config.yaml'
        )
        
        # Define paths for loading pre-trained model components
        load_path = agent_config["model"]["overt_agent_path"]
        nn_weights_path = os.path.join(load_path, 'nn_weights.pt')
        sound_scaler_path = os.path.join(load_path, 'sound_scaler.pickle')
        datasplits_path = os.path.join(load_path, 'datasplits.pickle')

        # Iterate through training configurations                    
        for dataset_name in agent_config["dataset"]["names"]:
            for jerk_loss_ceil in agent_config["training"]["jerk_loss_ceil"]:
                for jerk_loss_weight in agent_config["training"]["jerk_loss_weight"]:
                    # Reload config for each combination to ensure clean state
                    agent_config = utils.read_yaml_file(
                        './imitative_agent_inner_speech_silent_finetune/imitative_config.yaml'
                    )
                    agent_config["dataset"]["names"] = [dataset_name]
                    agent_config["training"]["jerk_loss_ceil"] = jerk_loss_ceil
                    agent_config["training"]["jerk_loss_weight"] = jerk_loss_weight

                    # Initialize agent with current configuration
                    agent = ImitativeAgent(agent_config)

                    # Load pre-trained model weights
                    if os.path.exists(nn_weights_path):
                        agent.nn.load_state_dict(torch.load(nn_weights_path, map_location=device), strict=False)
                        print(f"Loaded existing agent's weights from {nn_weights_path}")
                    else:
                        print(f"Neural network weights not found at {nn_weights_path}")
                        return
                    
                    # Load sound normalization parameters (sound scaler)
                    if os.path.exists(sound_scaler_path):
                        with open(sound_scaler_path, 'rb') as f:
                            agent.sound_scaler = pickle.load(f)
                        print(f"Loaded sound scaler from {sound_scaler_path}")
                    else:
                        print(f"Sound scaler not found at {sound_scaler_path}")
                        return

                    # Load the data splits to ensure consistent splits
                    if os.path.exists(datasplits_path):
                        with open(datasplits_path, 'rb') as f:
                            agent.datasplits = pickle.load(f)
                        print(f"Loaded data splits from {datasplits_path}")
                    else:
                        print(f"Data splits not found at {datasplits_path}")
                        return

                    # Freeze all parameters except inner speech output layer
                    for param in agent.nn.inverse_model.parameters():
                        param.requires_grad = False

                    for param in agent.nn.inverse_model.output_layer_inner1.parameters():
                        param.requires_grad = True
                        
                    # Freeze the direct model parameters
                    for param in agent.nn.direct_model.parameters():
                        param.requires_grad = False               

                    # Generate unique signature for this training run
                    signature = agent.get_signature()
                    save_path = "out/imitative_agent_inner_speech_silent_finetune_nojerk7/%s-%s" % (signature, i_training)

                    print(f"Fine-tuning {signature} (i_training={i_training})")
                    if os.path.isdir(save_path):
                        print("Already done")
                        print()
                        continue

                    # Set up training components
                    dataloaders = agent.get_dataloaders()
                    
                    # Configure optimizer only for inner speech output layer
                    optimizers = {
                        'inverse_model': torch.optim.Adam(
                            list(agent.nn.inverse_model.output_layer_inner1.parameters()),
                            lr=agent_config['training']['learning_rate']
                        ),
                        'direct_model': None  # Direct model remains frozen
                    }                
                    losses_fn = agent.get_losses_fn()
                    scheduler = agent.get_scheduler(optimizers)
                    
                    # Initialize sound scalers for data normalization
                    sound_scalers = {
                        "synthesizer": DataScaler.from_standard_scaler(
                            agent.synthesizer.sound_scaler
                        ).to(device),
                        "agent": DataScaler.from_standard_scaler(agent.sound_scaler).to(
                            device
                        ),
                    }

                    # Define checkpoint path for model saving
                    new_checkpoint_path = os.path.join('./out/imitative_agent_inner_speech_silent_finetune/','checkpoint.pt') 

                    # Initialize trainer and start fine-tuning
                    trainer = Trainer(
                        agent.nn,
                        optimizers,
                        scheduler,
                        *dataloaders,
                        losses_fn,
                        agent_config["training"]["max_epochs"],
                        agent_config["training"]["patience"],
                        agent.synthesizer,
                        sound_scalers,
                        new_checkpoint_path,
                    )
                    metrics_record = trainer.train()
                    
                    # Save fine-tuned model and training metrics
                    utils.mkdir(save_path)
                    agent.save(save_path)
                    with open(save_path + "/metrics.pickle", "wb") as f:
                        pickle.dump(metrics_record, f)


if __name__ == "__main__":
    main()
