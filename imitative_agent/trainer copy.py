import torch
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to Python path
print("current path:", os.getcwd())
sys.path.insert(0, "/mnt/c/Users/vpaul/OneDrive - CentraleSupelec/Inner_Speech/agent/")

from lib.early_stopping import EarlyStopping
from lib.training_record import TrainingRecord, EpochMetrics


class Trainer:
    """
    Trainer class for the ImitativeAgent model that handles training, validation and testing.
    
    Implements training loop with early stopping, loss computation, and optimization for both
    inverse model (sound->articulation) and direct model (articulation->sound).
    
    Args:
        nn: Neural network model to train
        optimizers: Dict of optimizers for model components
        train_dataloader: DataLoader for training data
        validation_dataloader: DataLoader for validation data
        test_dataloader: DataLoader for test data
        losses_fn: Dict of loss functions
        max_epochs: Maximum number of training epochs
        patience: Early stopping patience
        synthesizer: Synthesizer model for generating sounds
        sound_scalers: Dict of scalers for sound features
        checkpoint_path: Path to save model checkpoints
        device: Device to run training on (cuda/cpu)
    """
    def __init__(
        self,
        nn,
        optimizers,
        train_dataloader,
        validation_dataloader,
        test_dataloader,
        losses_fn,
        max_epochs,
        patience,
        synthesizer,
        sound_scalers,
        checkpoint_path,
        device= "cuda" if torch.cuda.is_available() else "cpu" 
    ):
        self.nn = nn.to(device)
        self.optimizers = optimizers
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader
        self.losses_fn = losses_fn
        self.max_epochs = max_epochs
        self.patience = patience
        self.synthesizer = synthesizer
        self.sound_scalers = sound_scalers
        self.checkpoint_path = checkpoint_path
        self.device = device

    def train(self):
        """
        Main training loop that trains both direct and inverse models with early stopping.
        
        Returns:
            dict: Training record containing metrics for all epochs
        """
        training_record = TrainingRecord()
        early_stopping = EarlyStopping(
            patience=self.patience, verbose=True, path=self.checkpoint_path
        )

        for epoch in range(1, self.max_epochs + 1):
            print("== Epoch %s ==" % epoch)

            # Run training epoch
            train_metrics = self.epoch_train(self.train_dataloader)
            training_record.save_epoch_metrics("train", train_metrics)

            # Run validation epoch
            validation_metrics = self.epoch_evaluate(self.validation_dataloader)
            training_record.save_epoch_metrics("validation", validation_metrics)

            # Run test epoch if test data provided
            if self.test_dataloader is not None:
                test_metrics = self.epoch_evaluate(self.test_dataloader)
                training_record.save_epoch_metrics("test", test_metrics)

            # Check early stopping criteria
            early_stopping(
                validation_metrics.metrics["inverse_model_repetition_error"], self.nn
            )

            if early_stopping.early_stop:
                print("Early stopping")
                break
            else:
                print()

        # Load best model weights
        self.nn.load_state_dict(torch.load(self.checkpoint_path))
        return training_record.record

    def epoch_train(self, dataloader):
        """
        Runs one epoch of training for both direct and inverse models.
        
        Args:
            dataloader: DataLoader containing training data
            
        Returns:
            EpochMetrics containing training metrics
        """
        nb_batch = len(dataloader)
        epoch_record = EpochMetrics(nb_batch)

        for batch in tqdm(dataloader, total=nb_batch, leave=False):
            # Process batch data
            sound_seqs, seqs_len, seqs_mask = batch
            sound_seqs = sound_seqs.to(self.device)
            seqs_mask = seqs_mask.to(self.device)

            # Train direct and inverse models
            self.step_direct_model(
                sound_seqs, seqs_len, seqs_mask, epoch_record, is_training=True
            )
            self.step_inverse_model(
                sound_seqs, seqs_len, seqs_mask, epoch_record, is_training=True
            )

        return epoch_record

    def epoch_evaluate(self, dataloader):
        """
        Runs one epoch of evaluation for both direct and inverse models.
        
        Args:
            dataloader: DataLoader containing validation/test data
            
        Returns:
            EpochMetrics containing evaluation metrics
        """
        nb_batch = len(dataloader)
        epoch_record = EpochMetrics(nb_batch)

        self.nn.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, total=nb_batch, leave=False):
                sound_seqs, seqs_len, seqs_mask = batch
                sound_seqs = sound_seqs.to(self.device)
                seqs_mask = seqs_mask.to(self.device)

                # Evaluate direct and inverse models
                self.step_direct_model(
                    sound_seqs, seqs_len, seqs_mask, epoch_record, is_training=False
                )
                self.step_inverse_model(
                    sound_seqs, seqs_len, seqs_mask, epoch_record, is_training=False
                )

        return epoch_record

    def step_direct_model(
        self, sound_seqs, seqs_len, seqs_mask, epoch_record, is_training
    ):
        """
        Performs one optimization step for the direct model.
        
        Args:
            sound_seqs: Input sound sequences
            seqs_len: Sequence lengths
            seqs_mask: Sequence masks
            epoch_record: Metrics recorder
            is_training: Whether in training mode
        """
        if is_training:
            self.nn.inverse_model.eval()
            self.nn.direct_model.train()
            self.nn.direct_model.requires_grad_(True)
        # Generate articulation sequence and synthesize sound
        with torch.no_grad():
            art_seqs_estimated = self.nn.inverse_model(sound_seqs, seqs_len=seqs_len)
        sound_seqs_produced = self.synthesizer.synthesize_cuda(art_seqs_estimated)
        sound_seqs_produced = self.sound_scalers["synthesizer"].inverse_transform(
            sound_seqs_produced
        )
        # Scale sound features
        sound_seqs_produced = self.sound_scalers["agent"].transform(sound_seqs_produced)


        # Optimize direct model
        if is_training:
            self.optimizers["direct_model"].zero_grad()
        sound_seqs_estimated = self.nn.direct_model(art_seqs_estimated)
        direct_model_loss = self.losses_fn["mse"](
            sound_seqs_estimated, sound_seqs_produced, seqs_mask
        )
        if is_training:
            direct_model_loss.backward()
            self.optimizers["direct_model"].step()

        epoch_record.add("direct_model_estimation_error", direct_model_loss.item())

    def step_inverse_model(
        self, sound_seqs, seqs_len, seqs_mask, epoch_record, is_training
    ):
        """
        Performs one optimization step for the inverse model.
        
        Estimates articulation from sound units and optimizes inverse model using
        both direct model estimation error and repetition error (→ through synthesizer).

        Args:
            sound_seqs: Input sound sequences
            seqs_len: Sequence lengths 
            seqs_mask: Sequence masks
            epoch_record: Metrics recorder
            is_training: Whether in training mode
        """
        # Inverse model training/evaluation
        # (inverse model estimation → direct model estimation vs. perceived sound)
        if is_training:
            self.nn.inverse_model.train()
            self.nn.direct_model.eval()
            self.nn.direct_model.requires_grad_(False)

            self.optimizers["inverse_model"].zero_grad()

        # Forward pass through inverse and direct models
        art_seqs_estimated = self.nn.inverse_model(sound_seqs, seqs_len=seqs_len)
        sound_seqs_estimated = self.nn.direct_model(art_seqs_estimated)

        # Compute and optimize inverse model losses
        inverse_total, inverse_estimation_error, inverse_jerk = self.losses_fn[
            "inverse_model"
        ](art_seqs_estimated, sound_seqs_estimated, sound_seqs, seqs_mask)
        if is_training:
            inverse_total.backward()
            self.optimizers["inverse_model"].step()

        # Record metrics
        epoch_record.add(
            "inverse_model_estimation_error", inverse_estimation_error.item()
        )
        epoch_record.add("inverse_model_jerk", inverse_jerk.item())

        # Inverse model repetition error
        # (inverse model estimation → synthesizer vs. perceived sound)
        sound_seqs_produced = self.synthesizer.synthesize_cuda(
            art_seqs_estimated.detach()
        )

        repetition_error = self.losses_fn["mse"](
            sound_seqs_produced, sound_seqs, seqs_mask
        )
        epoch_record.add("inverse_model_repetition_error", repetition_error.item())

        #################################################################""

        print(f" shape art seqs estimated : {art_seqs_estimated.shape}")
        print(f" shape sound_seqs_estimated : {sound_seqs_estimated.shape}")
        print(f" shape sound_seqs_produced : {sound_seqs_produced.shape}")
        # print(sound_seqs_produced[0])
        # dataset = Dataset("pb2007")
        # items_cepstrum = dataset.get_items_data("cepstrum", cut_silences=False)
        # items_source = dataset.get_items_data("source", cut_silences=False)
        
        batch_size, seq_len, num_features = sound_seqs_estimated.shape

        item_sounds_unscaled = self.sound_scalers["agent"].inverse_transform(sound_seqs)
        for i in range(batch_size):
            # item_name = item_names[i]
            # item_cepstrum = items_cepstrum[item_name]
            # item_source = items_source[item_name]
            # item_wave = dataset.get_item_wave(item_name)

            item_art_seq_est = art_seqs_estimated[i,:,:]
            item_sound_seq_est = sound_seqs_estimated[i,:, :]
            item_sound_seq_produced = sound_seqs_produced[i,:]
            item_sound_seq = sound_seqs[i,:,:]
            item_sound_unscaled = item_sounds_unscaled[i,:,:]

            # item_sound = np.concatenate((item_sound_seq, item_source), axis=1)
            # item_wave = lpcynet.synthesize_frames(item_sound_seq)
            # item_cepstrum_lpcnet= lpcynet.analyze_frames(item_wave)

            # item_sound_seq = item_sound_seq.numpy()
            # print(f"item cepstrum shape : {torch.tensor(item_cepstrum.shape)}")
            # print("Type of item_cepstrum:", type(item_cepstrum[0][0]))

            print(f" sound seq shape:{torch.tensor(item_sound_seq).shape}")
            print("Type of sound seq:", type(item_sound_seq[0][0]))

            # print(f" item wave :{torch.tensor(item_wave).shape}")
            # print("Type of item wave:", type(item_wave[0]))

            print(f"art seq est shape:{item_art_seq_est.shape}")
            print(f" Type of art seq est: {type(item_art_seq_est[0][0])}")

            print(f" sound seq est shape: {item_sound_seq_est.shape}")
            print(f" Type of sound seq est:{type(item_sound_seq_est[0][0])}")

            print(f"sound seq produced shape:{item_sound_seq_produced.shape}")
            print(f" Type of sound seq produced:{type(item_sound_seq_produced[0])}")

            # print(f"item wave  shape:{item_wave.shape}")
            # print(f" Type of item wave:{type(item_wave[0])}")

            print("Max item sound seq produced bark:", torch.max(item_sound_seq_produced))
            print("Min item sound produced bark:", torch.min(item_sound_seq_produced))
            print("Is float of sound seq produced bark:", torch.is_floating_point(item_sound_seq_produced[0][0]))

            print("Max item train:", torch.max(item_sound_seq))
            print("Min item train:", torch.min(item_sound_seq))
            print("Type of item_cepstrum_training:", type(item_sound_seq[0][0]))
            # Create a figure and axes
            plt.figure(figsize=(10, 8))

            # First subplot
            plt.subplot(3, 1, 1)
            plt.title("Estimated cepstrum original (scaled) with untrained agent")
            plt.imshow(item_sound_seq_est.detach().numpy().T, origin="lower")

            # Second subplot
            plt.subplot(3, 1, 2)
            plt.title("Training cepstrum (scaled) taken from dataloader")
            plt.imshow(item_sound_seq.T, origin="lower")

            # Third subplot
            plt.subplot(3, 1, 3)
            plt.title("Cepstrum explicitly scaled after repetition with untrained agent")
            plt.imshow(self.sound_scalers["agent"].transform(item_sound_seq_produced).detach().numpy().T, origin="lower")
            # Show the plot
            plt.tight_layout()  # Adjusts subplots to fit into figure area.
        plt.show()
        vincent
        print(f" modules sound quantizer: {dir(self.nn)}")

        ##########################################################""""

        repetition_error = self.losses_fn["mse"](
            sound_seqs_produced, sound_seqs, seqs_mask
        )
        epoch_record.add("inverse_model_repetition_error", repetition_error.item())
