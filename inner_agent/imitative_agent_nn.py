from torch import nn

import torch

class ImitativeAgentNN(nn.Module):
    """
    Initializes the ImativeAgentNN with specified inverse and direct models.
    """
    def __init__(self, inverse_model, direct_model):
        super(ImitativeAgentNN, self).__init__()
        self.inverse_model = inverse_model
        self.direct_model = direct_model

    def forward(self, sound_seqs):
        art_seqs_pred = self.inverse_model(sound_seqs)
        sound_seqs_pred = self.direct_model(art_seqs_pred)
        return sound_seqs_pred, art_seqs_pred
