import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from vector_quantize_pytorch import LFQ, VectorQuantize
import math


class Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, kernel_size=3, stride=2):
        super(Encoder, self).__init__()
        
        
        # FM mod
        self.strided_conv_1 = nn.Conv1d(in_channels=112, out_channels=512, kernel_size=kernel_size, padding=1)
        self.strided_conv_2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=kernel_size, padding=0)
        
        self.residual_conv_1 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=kernel_size, padding=1)
        self.residual_conv_2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=kernel_size, padding=1)
        
        self.proj = nn.Conv1d(in_channels=512, out_channels=16, kernel_size=3)
        
    def forward(self, x):
        
        x = self.strided_conv_1(x)
        x = self.strided_conv_2(x)
        
        x = F.relu(x)
        y = self.residual_conv_1(x)
        y = y+x
        
        x = F.relu(y)
        y = self.residual_conv_2(x)
        #print(y.shape)
        #print(x.shape)
        y = y+x
        
        y = self.proj(y)
        return y


class VQEmbeddingEMA(nn.Module):
    """
    After every epoch, run this for random restart:
    random_restart()
    reset_usage()
    """
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.999, epsilon=1e-5, usage_threshold=1.0e-9):
        super(VQEmbeddingEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.usage_threshold = usage_threshold
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        
        # init_bound = 1 / n_embeddings
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        # embedding.uniform_(-init_bound, init_bound)  # try other types on initialization 
        # Xavier initalization is designed to keep the scale of the gradients roughly the same in all layers
        nn.init.xavier_uniform_(embedding)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

        # initialize usage buffer for each code as fully utilized
        self.register_buffer('usage', torch.ones(self.n_embeddings), persistent=False)

    def update_usage(self, min_enc):
        self.usage[min_enc] = self.usage[min_enc] + 1  # if code is used add 1 to usage
        self.usage /= 2 # decay all codes usage

    def reset_usage(self):
        print("Reset usage of embeddings between epochs\n")
        self.usage.zero_() #  reset usage between epochs

    def random_restart(self):
        #  randomly restart all dead codes below threshold with random code in codebook
        dead_codes = torch.nonzero(self.usage < self.usage_threshold).squeeze(1)
        print("Are there any dead codes on this epoch? ", len(dead_codes))  # torch.any(dead_codes != 0))
        rand_codes = torch.randperm(self.n_embeddings)[0:len(dead_codes)].to('cpu')  # mps things
        with torch.no_grad():
            self.embedding[dead_codes] = self.embedding[rand_codes]

    def encode(self, x):
        x_flat = x.detach().reshape(-1, self.embedding_dim)

        distances = (-torch.cdist(x_flat, self.embedding, p=2)) ** 2

        indices = torch.argmin(distances.float(), dim=-1)
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)
        return quantized, indices.view(x.size(0), x.size(1))
    
    def retrieve_random_codebook(self, random_indices):
        quantized = F.embedding(random_indices, self.embedding)
        quantized = quantized.transpose(1, 3)
        return quantized

    def forward(self, x):
        # M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, self.embedding_dim)
        
        distances = (-torch.cdist(x_flat, self.embedding, p=2)) ** 2

        # find closest encodings
        # min_encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        # min_encodings = torch.zeros(
        #     min_encoding_indices.shape[0], self.n_embeddings).type_as(z)
        ### min_encodings.scatter_(1, min_encoding_indices, 1)

        indices = torch.argmin(distances.float(), dim=-1)
        encodings = F.one_hot(indices, self.n_embeddings).float()
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)
        
        if self.training:
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)
            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n + self.n_embeddings * self.epsilon) * n
            dw = torch.matmul(encodings.t(), x_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw
            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        self.update_usage(indices)
        
        codebook_loss = F.mse_loss(x.detach(), quantized)
        e_latent_loss = F.mse_loss(x, quantized.detach())
        commitment_loss = self.commitment_cost * e_latent_loss

        # preserve gradients
        quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, commitment_loss, codebook_loss, perplexity
    

class Decoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, kernel_size=3, stride=2):
        super(Decoder, self).__init__()
        
        '''
        self.in_proj = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)
        
        self.residual_conv_1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_1, padding=1)
        self.residual_conv_2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_2, padding=0)
        
        self.strided_t_conv_1 = nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_3, stride, padding=0)
        self.strided_t_conv_2 = nn.ConvTranspose2d(hidden_dim, output_dim, kernel_4, stride, padding=0)
        '''
        # FM mod
        self.in_proj = nn.Conv1d(in_channels=16, out_channels=512, kernel_size=3)

        self.residual_conv_1 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=kernel_size, padding=1)
        self.residual_conv_2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=kernel_size, padding=1)

        self.strided_t_conv_1 = nn.ConvTranspose1d(in_channels=512, out_channels=512, kernel_size=kernel_size, stride=1, padding=0, dilation=1)
        self.strided_t_conv_2 = nn.ConvTranspose1d(in_channels=512, out_channels=112, kernel_size=kernel_size, stride=1, padding=0, dilation=2)
        
    def forward(self, x):

        x = self.in_proj(x)
        
        y = self.residual_conv_1(x)
        y = y+x
        x = F.relu(y)
        
        y = self.residual_conv_2(x)
        y = y+x
        y = F.relu(y)
        
        y = self.strided_t_conv_1(y)
        y = self.strided_t_conv_2(y)
        return y
    
class Model(nn.Module):
    def __init__(self, Encoder, Codebook, Decoder):
        super(Model, self).__init__()
        self.encoder = Encoder
        self.codebook = Codebook
        self.decoder = Decoder
                
    def forward(self, x, epoch):
        z = self.encoder(x)
        # warm up model with no quantization
        if epoch >= 10:
            z_quantized, commitment_loss, codebook_loss, perplexity = self.codebook(z)
            x_hat = self.decoder(z_quantized)
        else:
            x_hat = self.decoder(z)
            commitment_loss, codebook_loss, perplexity = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
        
        return x_hat, commitment_loss, codebook_loss, perplexity

    

class ModelResidualVQ(nn.Module):
    def __init__(self, Encoder, Codebook, Decoder, n_embeddings=512, latent_dim=16):
        super(ModelResidualVQ, self).__init__()
        self.encoder = Encoder
        self.codebook = Codebook
        self.decoder = Decoder
        self.n_embeddings = n_embeddings
        self.latent_dim = latent_dim
        
    def forward(self, x, epoch):
        z = self.encoder(x)
        print("x.size() before passing X through the encoder", x.size())
        z = self.encoder(x)
        z = z.detach().reshape(-1, self.latent_dim)
        print("z.size() after passing X through the encoder and after flattening into 2D", z.size())
        # warm up model with no quantization
        # if epoch >= 8:
        # Previous: z_quantized, commitment_loss, codebook_loss, perplexity = self.codebook(z)
        # return_loss_breakdown=False:
        z_quantized, indices, commitment_loss = self.codebook(z)
        #z_quantized, indices, total_loss, loss_breakdown = self.codebook(z)
        #commitment_loss, codebook_loss = loss_breakdown['commitment'], loss_breakdown['codebook_diversity']  # -entropy(avg_prob).mean() in the lib, so positive mean entropy?
        #perplexity = torch.exp(codebook_loss)
        avg_probs = torch.mean(F.one_hot(indices, self.n_embeddings).float(), dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        codebook_loss = torch.tensor(0.0)
        z_quantized = z_quantized.reshape(x.shape[0], self.latent_dim, -1)  # Reshape to (batch_size, latent_dim, seq_length)
        print("z_quantized.size() after passing through the codebook and reshaping", z_quantized.size())
        x_hat = self.decoder(z_quantized)
        return x_hat, commitment_loss, codebook_loss, perplexity

def calculate_beta_log(n, total_iterations=50, initial_beta=0.35, final_beta=0.001, smoothing_factor=0.1):
    # a scaling factor based on the initial and final beta values
    scale_factor = (np.log(initial_beta) - np.log(final_beta)) / np.log(total_iterations) * smoothing_factor
    
    # beta value for the current iteration
    beta = np.exp(np.log(initial_beta) - scale_factor * np.log(n))
    
    if beta < final_beta:
        beta = final_beta
    return beta


class LFQAutoEncoder(nn.Module):
    # Lookup Free Quantization AutoEncoder
    def __init__(self, codebook_size=512, latent_dim=16, **vq_kwargs):
        super().__init__()
        quantize_dim = latent_dim  # Set quantize_dim to 16

        # Encoder
        self.encode = nn.Sequential(
            nn.Conv1d(112, 64, kernel_size=3, stride=1, padding=1),  # Conv1d layer
            nn.MaxPool1d(kernel_size=2, stride=2),                    # MaxPool1d layer
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),   # Conv1d layer
            nn.MaxPool1d(kernel_size=2, stride=2),                    # MaxPool1d layer
            nn.GroupNorm(4, 128, affine=False),
            nn.Conv1d(128, quantize_dim, kernel_size=1)               # Conv1d layer with kernel_size=1
        )

        # Quantization layer (use the latent_dim=16)
        self.quantize = LFQ(dim=quantize_dim, **vq_kwargs)

        # Decoder
        self.decode = nn.Sequential(
            nn.Conv1d(quantize_dim, 128, kernel_size=3, stride=1, padding=1),  # Output: [30, 128, 7]
            nn.Upsample(scale_factor=2, mode="nearest"),  # Output: [30, 128, 14]
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),  # Output: [30, 64, 14]
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),  # Output: [30, 64, 28]
            nn.Conv1d(64, 112, kernel_size=3, stride=1, padding=2),  # Adjust padding to 2 for sequence length 30
        )

    def forward(self, x):
        #print(f"Input shape: {x.shape}")  # Should be [batch_size, channels, sequence_length] [30, 112, 30]

        # Encoder
        x = self.encode(x)
        #print(f"Encoded shape: {x.shape}")  # Shape after encoding [30, 16, 7]
        x = x.permute(0, 2, 1)

        # Quantization (output should have same shape as input to decoder)
        x, indices, entropy_aux_loss = self.quantize(x)
        #print(f"Quantized shape: {x.shape}")  # Shape after quantization
        x = x.permute(0, 2, 1)

        # Decoder
        x = self.decode(x)
        #print(f"Decoded shape: {x.shape}")  # Shape after decoding
        return x.clamp(-1, 1), indices, entropy_aux_loss

"""
    # from vector_quantize_pytorch library examples
    def __init__(
        self,
        codebook_size,
        n_embeddings=512,
        **vq_kwargs
    ):
        super().__init__()
        assert math.log2(codebook_size).is_integer()
        quantize_dim = int(math.log2(codebook_size))

        self.encode = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # In general norm layers are commonly used in Resnet-based encoder/decoders
            # explicitly add one here with affine=False to avoid introducing new parameters
            nn.GroupNorm(4, 32, affine=False),
            nn.Conv2d(32, quantize_dim, kernel_size=1),
        )

        self.quantize = LFQ(dim=quantize_dim, **vq_kwargs)

        self.decode = nn.Sequential(
            nn.Conv2d(quantize_dim, 32, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
        )
        self.n_embeddings = n_embeddings
        return

    def forward(self, x):
        x = self.encode(x)
        x, indices, entropy_aux_loss = self.quantize(x)
        x = self.decode(x)

        avg_probs = torch.mean(F.one_hot(indices, self.n_embeddings).float(), dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return x.clamp(-1, 1), indices, entropy_aux_loss, perplexity
"""

class QSTGAE(nn.Module):
    # https://github.com/ZhouKanglei/STGAE
    def __init__(self, n_embeddings=512, latent_dim=16, **vq_kwargs):
        super(QSTGAE, self).__init__()
        self.n_embeddings = n_embeddings
        self.latent_dim = latent_dim
        # encoder
        # self.gcn = 
        # self.tcn =
        # self.res = residual block
        # quantizer
        self.quantizer = VectorQuantize(dim=latent_dim, accept_image_fmap=False, **vq_kwargs)

    def forward(self, x, epoch):

        # encoding block (can be repeated several times)
        h = self.gcn(x)
        encoded_x = self.tcn(h) + self.res(x)

        # quantization
        z = self.quantizer(encoded_x)

        # decoding block (can be repeated several times)
        x_hat = self.decoder(z)
