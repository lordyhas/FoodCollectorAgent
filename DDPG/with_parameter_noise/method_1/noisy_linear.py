import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        """
        in_features  : dimension d'entrée
        out_features : dimension de sortie
        sigma_init   : échelle initiale du bruit
        """
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Paramètres déterministes
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))

        # Paramètres du bruit
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Buffers pour le bruit échantillonné
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        # Initialisation classique : uniform entre -bound et bound pour mu
        bound = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        # sigma initial est fixé à sigma_init
        self.weight_sigma.data.fill_(self.sigma_init)
        self.bias_sigma.data.fill_(self.sigma_init)

    def reset_noise(self):
        # Génère un bruit gaussien pour weight et bias
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, input):
        if self.training:
            # En mode entraînement, on ajoute du bruit
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # En mode évaluation, on utilise uniquement la partie déterministe
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)
