import torch
import yaml
import os

from lib.base_agent import BaseAgent
from lib.nn.simple_lstm import SimpleLSTM
from lib.nn.feedforward import FeedForward
from lib.nn.loss import compute_jerk_loss

from communicative_agent_nn import CommunicativeAgentNN
from synthesizer.synthesizer import Synthesizer
from quantizer.quantizer import Quantizer

# Paths to saved synthesizer and quantizer models
SYNTHESIZERS_PATH = os.path.join(os.path.dirname(__file__), "../out/synthesizer")
QUANTIZERS_PATH = os.path.join(os.path.dirname(__file__), "../out/quantizer")


class CommunicativeAgent(BaseAgent):
    """
    Main agent class that combines inverse model (sound->articulation) and direct model (articulation->sound)
    """
    def __init__(self, config, load_nn=True):
        """Initialize agent with config and optionally load neural networks"""
        self.config = config
        # Handle special case where synthesizer is used as direct model
        if "use_synth_as_direct_model" in self.config["model"]:
            if self.config["model"]["use_synth_as_direct_model"]:
                if "direct_model" in self.config["model"]:
                    del self.config["model"]["direct_model"]
            else:
                del self.config["model"]["use_synth_as_direct_model"]

        # Load pretrained synthesizer and quantizer models
        self.synthesizer = Synthesizer.reload(
            "%s/%s" % (SYNTHESIZERS_PATH, config["synthesizer"]["name"]),
            load_nn=load_nn,
        )
        self.sound_quantizer = Quantizer.reload(
            "%s/%s" % (QUANTIZERS_PATH, config["sound_quantizer"]["name"]),
            load_nn=load_nn,
        )
        self.nb_speakers = len(self.sound_quantizer.config["dataset"]["names"])
        self.sound_scaler = self.sound_quantizer.data_scaler
        self.datasplits = self.sound_quantizer.datasplits
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if load_nn:
            self._build_nn(self.config["model"])
            self.nn.eval()

    def _build_nn(self, model_config):
        """Build the neural network architecture"""
        self.art_dim = self.synthesizer.art_dim
        self.sound_dim = self.synthesizer.sound_dim
        embedding_dim = self.sound_quantizer.config["model"]["embedding_dim"]

        # Build LSTM-based inverse model (sound->articulation)
        inverse_model = SimpleLSTM(
            embedding_dim,
            self.art_dim,
            model_config["inverse_model"]["hidden_size"],
            model_config["inverse_model"]["num_layers"],
            model_config["inverse_model"]["dropout_p"],
            model_config["inverse_model"]["bidirectional"],
        )

        # Build direct model (articulation->sound) - either feedforward or use synthesizer
        if "use_synth_as_direct_model" not in self.config["model"]:
            direct_model = FeedForward(
                self.art_dim,
                self.sound_dim,
                model_config["direct_model"]["hidden_layers"],
                model_config["direct_model"]["activation"],
                model_config["direct_model"]["dropout_p"],
                model_config["direct_model"]["batch_norm"],
            )
        else:
            direct_model = self.synthesizer.nn

        # Combine models into single neural network
        self.nn = CommunicativeAgentNN(
            inverse_model, direct_model, self.sound_quantizer.nn
        ).to(self.device)

    def get_dataloaders(self):
        """Get data loaders from quantizer"""
        return self.sound_quantizer.get_dataloaders()

    def get_optimizers(self):
        """Configure optimizers for training"""
        optimizers = {}
        # Optimizer for inverse model
        optimizers["inverse_model"] = torch.optim.Adam(
            self.nn.inverse_model.parameters(),
            lr=self.config["training"]["inverse_model_learning_rate"],
        )
        # Optimizer for direct model if not using synthesizer
        if "use_synth_as_direct_model" not in self.config["model"]:
            optimizers["direct_model"] = torch.optim.Adam(
                self.nn.direct_model.parameters(),
                lr=self.config["training"]["direct_model_learning_rate"],
            )

        return optimizers

    def get_losses_fn(self):
        """Define loss functions for training"""
        art_scaler_var = torch.FloatTensor(self.synthesizer.art_scaler.var_).to(self.device)

        def inverse_model_loss(art_seqs_pred, sound_seqs_pred, sound_seqs, seqs_mask):
            """Loss function for inverse model combining reconstruction and jerk loss"""
            reconstruction_error = (sound_seqs_pred - sound_seqs) ** 2
            reconstruction_loss = reconstruction_error[seqs_mask].mean()

            art_seqs_pred = art_seqs_pred * art_scaler_var
            jerk_loss = compute_jerk_loss(art_seqs_pred, seqs_mask)

            total_loss = (
                reconstruction_loss
                + jerk_loss * self.config["training"]["jerk_loss_weight"]
            )

            return total_loss, reconstruction_loss, jerk_loss

        def mse(seqs_pred, seqs, seqs_mask):
            """Mean squared error loss function"""
            reconstruction_error = (seqs_pred - seqs) ** 2
            reconstruction_loss = reconstruction_error[seqs_mask].mean()
            return reconstruction_loss

        return {"inverse_model": inverse_model_loss, "mse": mse}

    def save(self, save_path):
        """Save model config and weights"""
        with open(save_path + "/config.yaml", "w") as f:
            yaml.safe_dump(self.config, f)
        torch.save(self.nn.state_dict(), save_path + "/nn_weights.pt")

    @staticmethod
    def reload(save_path, load_nn=True):
        """Reload saved model from path"""
        with open(save_path + "/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        agent = CommunicativeAgent(config, load_nn=load_nn)

        if load_nn:
            agent.nn.load_state_dict(torch.load(save_path + "/nn_weights.pt", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
            agent.nn.eval()

        return agent

    def repeat(self, sound_seq):
        """Process sound sequence through model pipeline to get articulation and reconstructed sound"""
        nn_input = torch.FloatTensor(self.sound_scaler.transform(sound_seq)).to(self.device)[
            None, :, :
        ]
        with torch.no_grad():
            # Encode sound to discrete units
            _, sound_unit_seq, _, _ = self.nn.sound_quantizer.encode(nn_input)
            # Convert to articulation parameters
            art_seq_estimated_unscaled = self.nn.inverse_model(sound_unit_seq)
            # Convert back to sound features
            sound_seq_estimated_unscaled = self.nn.direct_model(
                art_seq_estimated_unscaled
            )
            # _, art_unit_seq, _, _ = self.nn.art_quantizer.encode(
            #     art_seq_estimated_unscaled
            # )

        # Convert outputs to numpy and rescale
        sound_seq_estimated_unscaled = sound_seq_estimated_unscaled[0].cpu().numpy()
        art_seq_estimated_unscaled = art_seq_estimated_unscaled[0].cpu().numpy()
        art_seq_estimated = self.synthesizer.art_scaler.inverse_transform(
            art_seq_estimated_unscaled
        )
        sound_seq_estimated = self.sound_scaler.inverse_transform(
            sound_seq_estimated_unscaled
        )
        sound_unit_seq = sound_unit_seq[0].cpu().numpy()
        # art_unit_seq = art_unit_seq[0].cpu().numpy()

        # Synthesize final sound from articulation parameters
        sound_seq_repeated = self.synthesizer.synthesize(art_seq_estimated)
        return {
            "sound_units": sound_unit_seq,
            "sound_repeated": sound_seq_repeated,
            "sound_estimated": sound_seq_estimated,
            "art_estimated": art_seq_estimated,
            # "art_units": art_unit_seq,
        }

    def get_grad(self, input):
        """Calculate gradient of articulation parameters with respect to sound units"""
        # Preprocess input
        nn_input = torch.FloatTensor(self.sound_scaler.transform(input)).to(self.device)[
            None, :, :
        ]
        with torch.no_grad():
            _, sound_unit_seq, _, _ = self.nn.sound_quantizer.encode(nn_input)
        sound_unit_seq.requires_grad_(True)
        
        # Get articulation predictions
        art_seqs_pred = self.nn.inverse_model(sound_unit_seq)

        # Calculate sum of absolute articulation values
        art_seqs_pred_abs= torch.abs(art_seqs_pred)
        art_seqs_sum = torch.sum(art_seqs_pred_abs, dim=2, keepdim=False)
 
        grad_output = torch.ones_like(art_seqs_sum)

        # Calculate jacobian
        jacobian = torch.autograd.grad(art_seqs_sum,sound_unit_seq, grad_outputs = grad_output, retain_graph=True, allow_unused=True)
        # Sum the absolute impact (jacobian) across the feature dimension
        jacobian = torch.cat(jacobian, dim=-1) # Now dimension [timesteps, cepstrum_features] tensor instead of tuple
        jacobian = torch.sum(torch.abs(torch.tensor(jacobian)), dim=2)  # Now dimension [batch_size,timesteps]


        return jacobian