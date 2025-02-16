import torch
import numpy as np
import os, sys
from tqdm import tqdm
print("current path:", os.getcwd())
sys.path.insert(0, "/mnt/c/Users/vpaul/Inner_Speech/agent/")

from lib.early_stopping import EarlyStopping
from lib.training_record import TrainingRecord, EpochMetrics

class Trainer:
    """
    Trainer class for fine-tuning an imitative agent model for inner speech.
    
    This trainer handles both overt and inner speech modes, with the ability to 
    selectively train the inner speech output layer while keeping other parameters frozen.
    
    Args:
        nn (ImitativeAgentNN): Neural network model containing inverse and direct models
        optimizers (dict): Optimizers for model training {'inverse_model': optimizer, 'direct_model': optimizer}
        scheduler: Learning rate scheduler
        train_dataloader: DataLoader for training data
        validation_dataloader: DataLoader for validation data 
        test_dataloader: DataLoader for test data
        losses_fn (dict): Loss functions for training
        max_epochs (int): Maximum number of training epochs
        patience (int): Early stopping patience
        synthesizer: Speech synthesizer model
        sound_scalers (dict): Data scalers for sound normalization
        checkpoint_path (str): Path to save model checkpoints
        inner_speech_ratio (float): Ratio of inner speech samples in each batch (default: 1)
        device (str): Device to run training on (default: 'cuda' if available else 'cpu')
    """
    def __init__(
        self,
        nn,
        optimizers,
        scheduler,
        train_dataloader,
        validation_dataloader,
        test_dataloader,
        losses_fn,
        max_epochs,
        patience,
        synthesizer,
        sound_scalers,
        checkpoint_path,
        inner_speech_ratio=1,  
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.nn = nn.to(device)
        self.optimizers = optimizers
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader
        self.losses_fn = losses_fn
        self.max_epochs = max_epochs
        self.patience = patience
        self.synthesizer = synthesizer
        self.sound_scalers = sound_scalers
        self.checkpoint_path = checkpoint_path
        self.inner_speech_ratio = inner_speech_ratio
        self.device = device

    def train(self):
        """
        Main training loop implementing early stopping based on validation performance.
        
        Returns:
            dict: Training record containing metrics for all epochs
        """
        training_record = TrainingRecord()
        early_stopping = EarlyStopping(
            patience=self.patience, verbose=True, path=self.checkpoint_path
        )

        for epoch in range(1, self.max_epochs + 1):
            print(f"== Epoch {epoch} ==")
            # In this implementation, the agent already knows the overt speech mode and tries to learn the inner speech projection/linear layer transformation
            # The inner speech ratio is set to 1 for the entire training
            # self.inner_speech_ratio = 1
            # Fixed inner speech ratio of 1 for fine-tuning
            print(f"== Inner speech ratio: {self.inner_speech_ratio} ==")
            
            # Training phase
            train_metrics = self.epoch_fn(self.train_dataloader, epoch, is_training=True)
            training_record.save_epoch_metrics("train", train_metrics)

            # Validation phase
            validation_metrics = self.epoch_fn(self.validation_dataloader, epoch, is_training=False)
            training_record.save_epoch_metrics("validation", validation_metrics)

            # Optional test phase
            if self.test_dataloader is not None:
                test_metrics = self.epoch_fn(self.test_dataloader, epoch, is_training=False)
                training_record.save_epoch_metrics("test", test_metrics)

            # Early stopping check based on validation error
            early_stopping(
                validation_metrics.metrics["inverse_model_repetition_error"], self.nn
            )
            # self.scheduler.step(validation_metrics.metrics["inverse_model_repetition_error"]) # Uncomment this line if scheduler is used

            if early_stopping.early_stop:
                print("Early stopping")
                break

        # Load the best model after early stopping or completion
        self.nn.load_state_dict(torch.load(self.checkpoint_path))
        return training_record.record

    def epoch_fn(self, dataloader, epoch, is_training=False):
        """
        Processes one epoch of data, handling both inner and overt speech modes.
        
        Args:
            dataloader: DataLoader providing batches
            epoch (int): Current epoch number
            is_training (bool): Whether this is a training phase
            
        Returns:
            EpochMetrics: Metrics collected during the epoch
        """
        nb_batch = len(dataloader)
        epoch_record = EpochMetrics(nb_batch)
        
        if not is_training:
            self.nn.eval()
        
        for batch in tqdm(dataloader, total=nb_batch, leave=False):
            # Unpack and prepare batch data
            sound_seqs, seqs_len, seqs_mask, _ = batch
            sound_seqs = sound_seqs.to(self.device)
            seqs_mask = seqs_mask.to(self.device)

            batch_size, time_steps, nb_features = sound_seqs.shape


            # Dynamically generate the mode flags based on the current inner_speech_ratio
            # Inner speech if flag == 1, overt speech if flag == 0
            num_inner = int(batch_size * self.inner_speech_ratio)
            num_overt = batch_size - num_inner
            mode_flags = [1] * num_inner + [0] * num_overt  # 1=inner, 0=overt
            np.random.shuffle(mode_flags)

            # Add mode flags to input sequences
            mode_flags = torch.tensor([[i]*time_steps for i in mode_flags])
            mode_flags = mode_flags.to(self.device).unsqueeze(-1).float()
            sound_seqs = torch.cat((sound_seqs, mode_flags), dim=-1) # Shape [batch_size, time_steps, nb_features + 1]

            # Create masks for inner and overt speech samples
            overt_mask = mode_flags[:,0,-1] == 0
            inner_mask = mode_flags[:,0,-1] == 1

            total_loss = 0
            if is_training:
                self.optimizers["inverse_model"].zero_grad()
                self.nn.train()
            else:
                self.nn.eval()

            # Process overt speech samples if present
            if overt_mask.any():
                print("Processing overt speech samples")
                direct_loss = self.step_direct_model(
                    sound_seqs, seqs_len, seqs_mask, overt_mask, epoch_record, is_training
                )
                inverse_loss = self.step_inverse_model(
                    sound_seqs, seqs_len, seqs_mask, overt_mask, epoch_record, is_training
                )
                total_loss += direct_loss + inverse_loss

            # Process inner speech samples if present
            if inner_mask.any():
                print("Processing inner speech samples")
                # For inner speech, train inverse model inner linear layer but keep direct model frozen
                self.nn.inverse_model.train()
                self.nn.direct_model.eval()
                
                inner_loss = self.step_inner_speech(
                    sound_seqs, seqs_len, seqs_mask, inner_mask, epoch_record, is_training
                )
                total_loss += inner_loss

            if is_training:

                total_loss.backward()  # Backpropagate the total combined loss
                self.optimizers["inverse_model"].step() # Update the inverse model inner output layer parameters

        return epoch_record

    def step_direct_model(
        self, sound_seqs, seqs_len, seqs_mask, overt_mask, epoch_record, is_training
    ):
        """
        Processes direct model step.
        Estimates articulatory sequences and synthesizes corresponding estimated sounds.
        """
        overt_mask = overt_mask.nonzero(as_tuple=True)[0]

        # Generate articulatory sequences and synthesize sounds
        art_seqs_estimated, lstm_output = self.nn.inverse_model(sound_seqs, seqs_len=seqs_len)
        sound_seqs_produced = self.synthesizer.synthesize_cuda(art_seqs_estimated)
        
        # Apply sound scaling transformations
        sound_seqs_produced = self.sound_scalers["synthesizer"].inverse_transform(
            sound_seqs_produced
        )
        sound_seqs_produced = self.sound_scalers["agent"].transform(sound_seqs_produced)

        mode_flags = sound_seqs[:,:,-1].unsqueeze(-1)  # Shape [batch_size, time_steps, 1]

        sound_seqs_estimated = self.nn.direct_model(art_seqs_estimated)

        # Calculate direct model loss
        direct_model_loss = self.losses_fn["mse"](
            sound_seqs_estimated[overt_mask], sound_seqs_produced[overt_mask], seqs_mask[overt_mask]
        )

        epoch_record.add("direct_model_estimation_error", direct_model_loss.item())
        return direct_model_loss

    def step_inverse_model(
        self, sound_seqs, seqs_len, seqs_mask, overt_mask, epoch_record, is_training
    ):
        """
        Processes inverse model step.
        Computes estimation error, jerk, and repetition error metrics.
        """
        overt_mask = overt_mask.nonzero(as_tuple=True)[0]

        # Generate articulatory and sound sequences
        art_seqs_estimated, lstm_output = self.nn.inverse_model(sound_seqs, seqs_len=seqs_len)

        mode_flags = sound_seqs[:,:,-1].unsqueeze(-1)  # Shape [batch_size, time_steps, 1]

        sound_seqs_estimated = self.nn.direct_model(art_seqs_estimated)

        # Calculate inverse model losses
        inverse_total, inverse_estimation_error, inverse_jerk = self.losses_fn[
            "inverse_model"
        ](art_seqs_estimated[overt_mask], sound_seqs_estimated[overt_mask], sound_seqs[:,:,:-1][overt_mask], seqs_mask[overt_mask])

        # Record metrics
        epoch_record.add(
            "inverse_model_estimation_error", inverse_estimation_error.item()
        )
        epoch_record.add("inverse_model_jerk", inverse_jerk.item())

        # Calculate repetition error using synthesized sounds
        sound_seqs_produced = self.synthesizer.synthesize_cuda(
            art_seqs_estimated.detach()
        )
        repetition_error = self.losses_fn["mse"](
            sound_seqs_produced[overt_mask], sound_seqs[:,:,:-1][overt_mask], seqs_mask[overt_mask]
        )
        epoch_record.add("inverse_model_repetition_error", repetition_error.item())
        return inverse_total

    def step_inner_speech(
        self, sound_seqs, seqs_len, seqs_mask, inner_mask, epoch_record, is_training
    ):
        """
        Processes inner speech samples with additional constraints:
        1. Null articulation constraint: encourages silent repetition from the synthesized sounds
        2. Loudness constraint: minimizes acoustic energy
        """
        inner_mask = inner_mask.nonzero(as_tuple=True)[0]

        # Generate articulatory and sound sequences
        art_seqs_estimated = self.nn.inverse_model(sound_seqs, seqs_len=seqs_len)
        nb_batch, time_steps, nb_features = art_seqs_estimated.shape
        sound_seqs_estimated = self.nn.direct_model(art_seqs_estimated)

        # Calculate inverse model losses
        inverse_total, inverse_estimation_error, inverse_jerk, velocity_loss = self.losses_fn[
            "inverse_model"
        ](art_seqs_estimated[inner_mask], sound_seqs_estimated[inner_mask], sound_seqs[:,:,:-1][inner_mask], seqs_mask[inner_mask])

        # Apply null articulation constraint
        art_seqs_copy = art_seqs_estimated.clone()
        sound_seqs_produced = self.synthesizer.synthesize_cuda(
            art_seqs_copy.detach()
        )    


        ###########################
        # Nul art loss constraint #
        ###########################

        nul_art_seq = [1.8451, -3.6259, -11.1342, 0.5492, -1.5493, 9.3482]
        nul_art_seqs = torch.tensor([[list(nul_art_seq)] * time_steps] * nb_batch)    
        silence_seqs_produced = self.synthesizer.synthesize_cuda(
            nul_art_seqs
        )
        
        # Calculate silence error
        repetition_silence_error = self.losses_fn["mse"](
            sound_seqs_produced[inner_mask], silence_seqs_produced[inner_mask], seqs_mask[inner_mask]
        )

        # Record metrics
        epoch_record.add(
            "inverse_model_estimation_error", inverse_estimation_error.item()
        )

        inverse_estimation_error += repetition_silence_error #+ 10*inverse_jerk + 10*velocity_loss

        #######################
        # Loudness constraint #
        #######################

        # Compute loudness loss directly on the synthesized sounds
        # Convert Bark coefficients to energy
        # Assuming Bark coefficients are in linear scale; if in log scale, apply appropriate transformation
        # Square the Bark coefficients to get power (since amplitude squared is power)


        # power = sound_seqs_produced[inner_mask] ** 2  # Shape: [batch_size, time_steps, bark_bins]
        # # Sum power across Bark bands to get total power per time step
        # total_power = power.sum(dim=-1)  # Shape: [batch_size, time_steps]
        # # Compute loudness (could also take square root if desired)
        # loudness = total_power  # Shape: [batch_size, time_steps]

        # # Apply sequence mask
        # loudness = loudness * seqs_mask[inner_mask]  # Mask out padding

        # # Compute the mean loudness loss
        # loudness_loss = loudness.mean()

        # # Add loudness loss to inverse estimation error
        # inverse_estimation_error += loudness_loss
# 
        # inverse_total += repetition_silence_error


        epoch_record.add("inverse_model_jerk", inverse_jerk.item())
        epoch_record.add("inverse_model_repetition_error", repetition_silence_error.item())

        return inverse_estimation_error
