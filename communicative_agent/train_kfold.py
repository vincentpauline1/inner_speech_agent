"""
K-fold cross validation training script for the CommunicativeAgent model.
Trains models on different dataset splits with varying jerk loss weights to evaluate robustness.
"""
import os
import pickle

from lib import utils
from lib.nn.data_scaler import DataScaler
from communicative_agent import CommunicativeAgent
import torch
from trainer import Trainer

# Training configuration
NB_FOLDS = 5  # Number of cross-validation folds
DATASETS = [["pb2007"], ["gb2016", "th2016"]]  # Dataset combinations to evaluate
JERK_LOSS_WEIGHTS = [0, 0.15]  # Jerk regularization weights to test


def train_agent(agent, save_path):
    """
    Train a CommunicativeAgent model and save results.
    
    Args:
        agent (CommunicativeAgent): Agent instance to train
        save_path (str): Directory path to save trained model and metrics
    """
    print("Training %s" % save_path)
    if os.path.isdir(save_path):
        print("Already done")
        print()
        return
    
    # Set device for training (GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    # Initialize trainer with model components and training parameters
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
    2. Iterates through dataset combinations
    3. Performs k-fold cross validation
    4. Tests different jerk loss weights
    """
    # Load final model configurations
    final_configs = utils.read_yaml_file("communicative_agent/communicative_final_configs.yaml")

    # Iterate through dataset combinations
    for datasets_name in DATASETS:
        datasets_key = ",".join(datasets_name)
        config = final_configs[datasets_key]

        # Perform k-fold cross validation
        for i_fold in range(NB_FOLDS):
            # Test different jerk loss weights
            for jerk_loss_weight in JERK_LOSS_WEIGHTS:
                # Configure model for current fold and jerk weight
                config["sound_quantizer"]["name"] = "kfold-%s-%s" % (datasets_key, i_fold)
                config["training"]["jerk_loss_weight"] = jerk_loss_weight

                # Initialize and train agent
                agent = CommunicativeAgent(config)
                save_path = "out/communicative_agent/kfold-%s-jerk=%s-%s" % (
                    datasets_key, jerk_loss_weight, i_fold
                )
                train_agent(agent, save_path)


if __name__ == "__main__":
    main()
