import torch.nn as nn

class DEEPCOX_ARCHITECTURE(nn.Module):
    """
    Deep Cox architecture used in SIDISH for survival prediction.
    This network integrates pretrained encoder representations (from the VAE)
    with a Cox proportional hazards regression layer for modeling survival risk.
    """
    
    def __init__(self, hidden, encoder, dropout):
        """
        Args:
            hidden (int): Number of hidden units in the intermediate layer.
            encoder (nn.Module): Pretrained encoder (e.g., from SIDISH VAE).
            dropout (float): Dropout rate for regularization.
        """
        
        super(DEEPCOX_ARCHITECTURE, self).__init__()

        # Extract encoder layers except the final output layer.
        self.encoder_layer = nn.Sequential(*list(encoder.model.encoder.children())[:-1])
        self.af1 = nn.Tanh()
        self.dr1 = nn.Dropout(dropout)

        # Add a new fully connected layer for risk prediction
        self.new_layer = nn.Linear(self.encoder_layer[-1].out_features, hidden)
        self.dr2 = nn.Dropout(dropout)
        self.af2 = nn.Tanh()
        self.final_layer = nn.Linear(hidden, 1, bias=False)

    def forward(self, x):
        x_ = self.af1(self.dr1(self.encoder_layer(x)))
        x__ = self.af2(self.dr2(self.new_layer(x_)))
        final_x = self.final_layer(x__)
        return final_x
