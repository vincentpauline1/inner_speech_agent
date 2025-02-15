"""
Hyperparameter optimization for the CommunicativeAgent model using Hyperopt.
Performs automated tuning of model architecture and training parameters to minimize validation loss.
"""
import os
import pickle
import numpy as np
from hyperopt import tpe, hp, fmin

from lib import utils
from lib.nn.data_scaler import DataScaler
from communicative_agent import CommunicativeAgent
import torch
from trainer import Trainer


def train_agent(agent, save_path):
    """
    Train a CommunicativeAgent model and save results.
    
    Args:
        agent: CommunicativeAgent instance to train
        save_path: Directory path to save trained model and metrics
        
    Returns:
        metrics_record: Dictionary containing training/validation metrics history
    """
    print("Training %s" % (save_path))
    if os.path.isdir(save_path):
        print("Model already trained, skipping...")
        print()
        return
        
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get data loaders, optimizers and loss functions from agent
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

    return metrics_record


def train_with_hyperparameters(hyperparameters):
    """
    Train a model with a specific set of hyperparameters.
    
    Args:
        hyperparameters: Dictionary of hyperparameter values to test
        
    Returns:
        final_validation_loss: Best validation loss achieved during training
    """
    # Load base configuration
    agent_config = utils.read_yaml_file("communicative_agent/communicative_config.yaml")

    # Update inverse model hyperparameters
    agent_config["training"]["inverse_model_learning_rate"] = hyperparameters[
        "inverse_model_learning_rate"
    ]
    agent_config["model"]["inverse_model"]["num_layers"] = int(
        hyperparameters["inverse_model_num_layers"]
    )
    agent_config["model"]["inverse_model"]["hidden_size"] = int(
        2 ** hyperparameters["inverse_model_hidden_size"]
    )
    agent_config["model"]["inverse_model"]["dropout_p"] = hyperparameters[
        "inverse_model_dropout_p"
    ]

    # Update direct model hyperparameters if not using synthesizer
    if (
        "use_synth_as_direct_model" not in agent_config["model"]
        or not agent_config["model"]["use_synth_as_direct_model"]
    ):
        agent_config["training"]["direct_model_learning_rate"] = hyperparameters[
            "direct_model_learning_rate"
        ]
        agent_config["model"]["direct_model"]["dropout_p"] = hyperparameters[
            "direct_model_dropout_p"
        ]
        agent_config["model"]["direct_model"]["hidden_layers"] = [
            int(2 ** hyperparameters["direct_model_dim_hidden_layers"])
        ] * int(hyperparameters["direct_model_nb_hidden_layers"])

    # Initialize and train agent
    agent = CommunicativeAgent(agent_config)
    signature = agent.get_signature()
    save_path = f"out/communicative_agent/{signature}"

    metrics_record = train_agent(agent, save_path)
    
    # Return best validation loss achieved
    final_validation_loss = min(
        metrics_record["validation"]["inverse_model_repetition_error"]
    )
    print(f"Validation loss for current config: {final_validation_loss:.4f}")
    return final_validation_loss


def main():
    """
    Main function to run hyperparameter optimization.
    Defines search space and runs optimization using Tree of Parzen Estimators (TPE).
    """
    base_config = utils.read_yaml_file("communicative_agent/communicative_config.yaml")

    # Define search space for inverse model hyperparameters
    hyperparameters_space = {
        "inverse_model_learning_rate": hp.loguniform(
            "inverse_model_learning_rate", np.log(1e-4), np.log(1e-2)
        ),
        "inverse_model_num_layers": hp.quniform("inverse_model_num_layers", 1, 2, 1),
        "inverse_model_hidden_size": hp.quniform("inverse_model_hidden_size", 5, 6, 1),
        "inverse_model_dropout_p": hp.uniform("inverse_model_dropout_p", 0, 0.9),
    }

    # Add direct model hyperparameters if not using synthesizer
    if (
        "use_synth_as_direct_model" not in base_config["model"]
        or not base_config["model"]["use_synth_as_direct_model"]
    ):
        hyperparameters_space.update({
            "direct_model_learning_rate": hp.loguniform(
                "direct_model_learning_rate", np.log(1e-4), np.log(1e-2)
            ),
            "direct_model_dropout_p": hp.uniform("direct_model_dropout_p", 0, 0.9),
            "direct_model_dim_hidden_layers": hp.quniform(
                "direct_model_dim_hidden_layers", 6, 9, 1
            ),
            "direct_model_nb_hidden_layers": hp.quniform(
                "direct_model_nb_hidden_layers", 1, 4, 1
            ),
        })

    # Run hyperparameter optimization
    best_config = fmin(
        fn=train_with_hyperparameters,
        space=hyperparameters_space,
        algo=tpe.suggest,
        max_evals=100,
    )
    print("best config:")
    print(best_config)


if __name__ == "__main__":
    main()
