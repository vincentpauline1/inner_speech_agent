"""
ImitativeAgent: A neural network-based agent for speech repetition and estimation.

During overt speech mode, the agent produces both repetition and estimation while in inner speech mode only the estimation should remain.

The learning mechanism of inner speech follow the Vygotskian developmental theory of inner speech.
Such that inner speech occurs after the overt speech is mastered. To that extent, the inner speech agent is learned by fine-tuning the imitative agent.


The agent uses a bidirectional architecture with:
- An inverse model (sound->articulation) implemented as an LSTM
- A direct model (articulation->sound) implemented as a feedforward network for the sound estimation
- A pretrained synthesizer to convert articulation to sound for the sound repetition
"""

import torch
import pickle
import yaml
import os,sys
from lib.dataset_wrapper import Dataset
from sklearn.preprocessing import StandardScaler

# Configure paths and imports
print("current path:", os.getcwd())

# Add custom directory path to access agent-related modules
sys.path.insert(0, "/mnt/c/Users/vpaul/Documents/Inner_Speech/agent/")

# Import custom modules for the agent
from lib.base_agent import BaseAgent
from lib.sound_dataloader_inner_speech import get_dataloaders
from lib.nn.simple_lstm_ft import SimpleLSTM
from lib.nn.feedforward import FeedForward
from lib.nn.loss import ceil_loss, compute_jerk_loss, compute_speed_loss 

from imitative_agent_nn import ImitativeAgentNN
from synthesizer.synthesizer import Synthesizer

# Path to trained synthesizer directory
SYNTHESIZERS_PATH = os.path.join(os.path.dirname(__file__), "../out/synthesizer")


class ImitativeAgent(BaseAgent):
    """
    Agent that learns to imitate (repeat and estimate) speech by mapping between acoustic and articulatory representations.
    
    Attributes:
        config (dict): Configuration parameters for the model architecture and training
        sound_scaler (StandardScaler): Normalizes acoustic features
        datasplits (dict): Train/val/test splits for each dataset
        device (str): Computing device (CPU/GPU) for model execution
        nn (ImitativeAgentNN): Neural network architecture combining inverse and direct models
    """
    def __init__(self, config, load_nn=True):
        self.config = config
        self.sound_scaler = StandardScaler()
        self.datasplits = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if load_nn:
            print(self.config["model"])
            # Initialize synthesizer from saved model
            self.synthesizer = Synthesizer.reload(
                "%s/%s" % (SYNTHESIZERS_PATH, config["synthesizer"]["name"])
            )
            print(self.config["model"])
            self._build_nn(self.config["model"])
            self.nn.eval()

    def _build_nn(self, model_config):
        """
        Constructs the neural network architecture with inverse and direct models.
        
        The inverse model maps sound->articulation using an LSTM.
        The direct model maps articulation->sound estimation using a feedforward network.
        """
        self.art_dim = self.synthesizer.art_dim # Articulation dimension
        self.sound_dim = self.synthesizer.sound_dim # Sound dimension

        # Define inverse model using an LSTM network for sound-to-articulation mapping
        # Inverse model : 18D Sound cepstrum barks  -> 6D Articulatory features
        inverse_model = SimpleLSTM(
            self.sound_dim,
            self.art_dim,
            model_config["inverse_model"]["hidden_size"],
            model_config["inverse_model"]["num_layers"],
            model_config["inverse_model"]["dropout_p"],
            model_config["inverse_model"]["bidirectional"],
        )

        # Define direct model using a feedforward network for articulation-to-sound mapping
        # Direct model : 6D Articulatory features -> 18D Sound cepstrum barks
        direct_model = FeedForward(
            self.art_dim,
            self.sound_dim,
            model_config["direct_model"]["hidden_layers"],
            model_config["direct_model"]["activation"],
            model_config["direct_model"]["dropout_p"],
            model_config["direct_model"]["batch_norm"],
        )

        # Initialize the ImitativeAgentNN with the inverse and direct models
        self.nn = ImitativeAgentNN(inverse_model, direct_model).to(self.device)

    def get_dataloaders(self, inner_speech_ratio = 1):
        """
        Creates dataloaders for training/validation/testing whith specified inner speech ratio.
        
        Args:
            inner_speech_ratio (float): Proportion of inner speech examples (default=1 for pure inner speech)
        
        Returns:
            dict: DataLoaders for each split
        """
        datasplits, dataloaders = get_dataloaders(
            self.config["dataset"], self.sound_scaler, self.datasplits, inner_speech_ratio
        )
        self.datasplits = datasplits
        return dataloaders

    def get_optimizers(self):
        """
        Defines and returns optimizers for the inverse and direct models, based on the learning rate in config.
        """
        return {
            "inverse_model": torch.optim.Adam(
                self.nn.inverse_model.parameters(),
                lr=self.config["training"]["learning_rate"],
            ),
            "direct_model": torch.optim.Adam(
                self.nn.direct_model.parameters(),
                lr=self.config["training"]["learning_rate"],
            ),
        }
    
    def get_scheduler(self, optimizers):
        """
        Sets a OneCycleLR or ReduceLROnPlateau scheduler for the inverse model's optimizer to adapt learning rates during training.
        """
        # Use OneCycleLR scheduler for the inverse model        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizers["inverse_model"], max_lr=self.config["training"]["learning_rate"], total_steps=self.config["training"]["max_epochs"], anneal_strategy='linear')

        # Reduce learning rate when a metric has stopped improving
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers["inverse_model"], mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6)
        
        return scheduler
    

    def get_losses_fn(self):
        """
        Defines loss functions for model training.
        
        Combines reconstruction loss with penalties for:
        - Jerk (third derivative of articulation)
        - Velocity (first derivative of articulation)
        """
        art_scaler_var = torch.FloatTensor(self.synthesizer.art_scaler.var_)

        def inverse_model_loss(art_seqs_pred, sound_seqs_pred, sound_seqs, seqs_mask):
            # Compute reconstruction error
            reconstruction_error = (sound_seqs_pred - sound_seqs) ** 2
            reconstruction_loss = reconstruction_error[seqs_mask].mean()

            # Apply penalties on articulator movement
            art_seqs_pred = art_seqs_pred * art_scaler_var
            jerk_loss = compute_jerk_loss(art_seqs_pred, seqs_mask)
            vel_loss = compute_speed_loss(art_seqs_pred, seqs_mask)

            # Combine losses with configured weights
            total_loss = reconstruction_loss + (
                ceil_loss(jerk_loss, self.config["training"]["jerk_loss_ceil"])
                * self.config["training"]["jerk_loss_weight"]
            )

            return total_loss, reconstruction_loss, jerk_loss, vel_loss

        def mse(sound_seqs_pred, sound_seqs, seqs_mask):
            """Simple MSE loss for sound prediction"""
            reconstruction_error = (sound_seqs_pred - sound_seqs) ** 2
            return reconstruction_error[seqs_mask].mean()

        return {"inverse_model": inverse_model_loss, "mse": mse}

    def save(self, save_path):
        """
        Saves the configuration, scaler, dataset splits, and model weights to specified path.
        """
        with open(save_path + "/config.yaml", "w") as f:
            yaml.safe_dump(self.config, f)
        with open(save_path + "/sound_scaler.pickle", "wb") as f:
            pickle.dump(self.sound_scaler, f)
        with open(save_path + "/datasplits.pickle", "wb") as f:
            pickle.dump(self.datasplits, f)
        torch.save(self.nn.state_dict(), save_path + "/nn_weights.pt")

    @staticmethod
    def reload(save_path, load_nn=True):
        """
        Reloads a saved model, restoring configuration, scaler, dataset splits, and model weights.
        
        Args:
            save_path (str): Directory containing saved model
            load_nn (bool): Whether to load neural network weights
        """
        with open(save_path + "/config.yaml", "r") as f:
            config = yaml.safe_load(f)
            print(config)
        agent = ImitativeAgent(config)

        with open(save_path + "/sound_scaler.pickle", "rb") as f:
            agent.sound_scaler = pickle.load(f)
        with open(save_path + "/datasplits.pickle", "rb") as f:
            agent.datasplits = pickle.load(f)
        if load_nn:
            agent.nn.load_state_dict(torch.load(save_path + "/nn_weights.pt"))
            agent.nn.eval()

        return agent

    def repeat_overt(self, sound_seq):
        """
        Repeats an input sequence in overt speech mode.
        Adds a 0 feature flag to indicate overt speech mode.
        """
        # Add feature equal to 0 to sound_seqs for overt speech
        time_steps, nb_features = sound_seq.shape
        
        # Create overt speech flag (0)
        zero_feature = torch.zeros(time_steps).unsqueeze(-1).float() # Shape [batch_size, 1, 1]
        
        # Prepare input by scaling and concatenating flag
        nn_input = torch.FloatTensor(self.sound_scaler.transform(sound_seq)).to(self.device)
        nn_input = torch.cat((nn_input, zero_feature), dim=-1)

        # Generate predictions
        with torch.no_grad():
            sound_seq_estimated_unscaled, art_seq_estimated_unscaled = self.nn(
                nn_input[None, :, :]
            )
            
        # Post-process predictions
        sound_seq_estimated_unscaled = sound_seq_estimated_unscaled[0].cpu().numpy()
        art_seq_estimated_unscaled = art_seq_estimated_unscaled[0].cpu().numpy()
        
        # Inverse transform predictions
        art_seq_estimated = self.synthesizer.art_scaler.inverse_transform(
            art_seq_estimated_unscaled
        )
        sound_seq_estimated = self.sound_scaler.inverse_transform(
            sound_seq_estimated_unscaled
        )
        sound_seq_repeated = self.synthesizer.synthesize(art_seq_estimated)
        
        return {
            "sound_repeated": sound_seq_repeated,
            "sound_estimated": sound_seq_estimated,
            "art_estimated": art_seq_estimated,
        }

    def repeat_inner(self,sound_seq):
        """
        Repeats an input sequence in inner speech mode.
        Adds a 1 feature flag to indicate inner speech mode.
        """
        time_steps, nb_features = sound_seq.shape
        
        # Create inner speech flag (1)
        one_feature = torch.ones(time_steps).unsqueeze(-1).float()
        
        # Prepare input
        nn_input = torch.FloatTensor(self.sound_scaler.transform(sound_seq))
        nn_input = torch.cat((nn_input, one_feature), dim=-1)

        # Generate and post-process predictions
        with torch.no_grad():
            sound_seq_estimated_unscaled, art_seq_estimated_unscaled = self.nn(
                nn_input[None, :, :]
            )
        
        sound_seq_estimated_unscaled = sound_seq_estimated_unscaled[0].cpu().numpy()
        art_seq_estimated_unscaled = art_seq_estimated_unscaled[0].cpu().numpy()
        
        # Inverse transform predictions
        art_seq_estimated = self.synthesizer.art_scaler.inverse_transform(
            art_seq_estimated_unscaled
        )
        sound_seq_estimated = self.sound_scaler.inverse_transform(
            sound_seq_estimated_unscaled
        )
        sound_seq_repeated = self.synthesizer.synthesize(art_seq_estimated)
        
        return {
            "sound_repeated": sound_seq_repeated,
            "sound_estimated": sound_seq_estimated,
            "art_estimated": art_seq_estimated,
        }

    def repeat_overt_datasplit(self, datasplit_index=None):
        """
        Repeats all sequences in a datasplit using overt speech mode.
        
        Args:
            datasplit_index (int, optional): Index of split to process. If None, processes all data.
        """
        agent_features = {}
        sound_type = self.config["dataset"]["sound_type"]

        for dataset_name in self.config["dataset"]["names"]:
            dataset_features = {}
            dataset = Dataset(dataset_name)
            
            # Get items to process
            if datasplit_index is None:
                items_name = dataset.get_items_name(sound_type)
            else:
                items_name = self.datasplits[dataset_name][datasplit_index]

            # Process each item
            items_sound = dataset.get_items_data(sound_type)
            for item_name in items_name:
                item_sound = items_sound[item_name]
                repetition = self.repeat_overt(item_sound)
                
                # Store results
                for repetition_type, repetition_data in repetition.items():
                    if repetition_type not in dataset_features:
                        dataset_features[repetition_type] = {}
                    dataset_features[repetition_type][item_name] = repetition_data

            agent_features[dataset_name] = dataset_features
        return agent_features
    
    def repeat_inner_datasplit(self, datasplit_index=None):
        """
        Repeats all sequences in a datasplit using inner speech mode.
        
        Args:
            datasplit_index (int, optional): Index of split to process. If None, processes all data.
        """
        agent_features = {}
        sound_type = self.config["dataset"]["sound_type"]

        for dataset_name in self.config["dataset"]["names"]:
            dataset_features = {}
            dataset = Dataset(dataset_name)
            
            # Get items to process
            if datasplit_index is None:
                items_name = dataset.get_items_name(sound_type)
            else:
                items_name = self.datasplits[dataset_name][datasplit_index]

            # Process each item
            items_sound = dataset.get_items_data(sound_type)
            for item_name in items_name:
                item_sound = items_sound[item_name]
                repetition = self.repeat_inner(item_sound)
                
                # Store results
                for repetition_type, repetition_data in repetition.items():
                    if repetition_type not in dataset_features:
                        dataset_features[repetition_type] = {}
                    dataset_features[repetition_type][item_name] = repetition_data

            agent_features[dataset_name] = dataset_features
        return agent_features


    # def invert_art(self, sound_seq):
    #     nn_input = torch.FloatTensor(self.sound_scaler.transform(sound_seq)).to("cuda")
    #     with torch.no_grad():
    #         art_seq_estimated_unscaled = self.nn.inverse_model(
    #             nn_input[None, :, :]
    #         )
    #     art_seq_estimated_unscaled = art_seq_estimated_unscaled[0].cpu().numpy()
    #     art_seq_estimated = self.synthesizer.art_scaler.inverse_transform(
    #         art_seq_estimated_unscaled
    #     )
    #     return art_seq_estimated
