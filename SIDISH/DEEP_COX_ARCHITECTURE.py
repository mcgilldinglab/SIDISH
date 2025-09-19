import torch.nn as nn

class DEEPCOX_ARCHITECTURE(nn.Module):
    
    def __init__(self, hidden, encoder, dropout):
        super(DEEPCOX_ARCHITECTURE, self).__init__()
        self.encoder_layer = nn.Sequential(*list(encoder.model.encoder.children())[:-1])
        self.af1 = nn.Tanh()
        self.dr1 = nn.Dropout(dropout)
        self.new_layer = nn.Linear(self.encoder_layer[-1].out_features, hidden)
        
        #self.new_layer = nn.Linear(last_out_features, hidden)
        self.dr2 = nn.Dropout(dropout)
        self.af2 = nn.Tanh()
        self.final_layer = nn.Linear(hidden, 1, bias=False)

    def forward(self, x):
        x_ = self.af1(self.dr1(self.encoder_layer(x)))
        x__ = self.af2(self.dr2(self.new_layer(x_)))
        final_x = self.final_layer(x__)

        return final_x
    
    
    

