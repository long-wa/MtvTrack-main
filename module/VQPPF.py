import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResidualLayer(nn.Module):
    """
    One residual layer inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    """

    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1,
                      stride=1, bias=False)
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x


class ResidualStack(nn.Module):
    """
    A stack of residual layers inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList(
            [ResidualLayer(in_dim, h_dim, res_h_dim)]*n_res_layers)

    def forward(self, x):  # (32,128,8,8)
        for layer in self.stack:
            x = layer(x)
        x = F.relu(x)  # (32,128,8,8)
        return x

class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQPPF.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e  # 512
        self.e_dim = e_dim  # 64
        self.beta = beta  # 0.25

        self.embedding = nn.Embedding(self.n_e, self.e_dim)  # (512,64)  embedding space of e_d
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.embedding_z = nn.Embedding(self.n_e, self.e_dim)  # embedding space of e_c
        self.embedding_z.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z_list):
        """
        Quantization of latent representations for templates and search regions from encoder outputs
        one-hot vector that is the index of the closest embedding vector e_j

        """
        total_embedding_loss = None
        total_perplexity = None
        min_encoding_list = []
        min_encoding_indices_list = []
        z_q_list = []
        for i, z in enumerate(z_list):
            # reshape z -> (batch, height, width, channel) and flatten
            z = z.permute(0, 2, 3, 1).contiguous()
            z_flattened = z.view(-1, self.e_dim)
            # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

            # a = torch.sum(z_flattened ** 2, dim=1, keepdim=True)  # (16,1)
            # b = torch.sum(self.embedding.weight**2, dim=1)  # (512,)
            # c = torch.matmul(z_flattened, self.embedding.weight.t())  # (16,512)
            # e = a+b  # (16,512)
            # f = e+c  # (16,512)
            if i == 0 :  # where 0 represents the latent representation of the template, and the others denote the latent representations of the search regions
                d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                    torch.sum(self.embedding_z.weight ** 2, dim=1) - 2 * \
                    torch.matmul(z_flattened,
                                 self.embedding_z.weight.t())
            else:
                d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                    torch.sum(self.embedding.weight**2, dim=1) - 2 * \
                    torch.matmul(z_flattened, self.embedding.weight.t())

            # find closest encodings
            min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
            min_encodings = torch.zeros(
                min_encoding_indices.shape[0], self.n_e).to(device)
            min_encodings.scatter_(1, min_encoding_indices, 1)

            # get quantized latent vectors
            if i == 0:
                z_q = torch.matmul(min_encodings, self.embedding_z.weight).view(z.shape)
            else:
                z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

            # compute loss for embedding
            loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                torch.mean((z_q - z.detach()) ** 2)

            # preserve gradients
            z_q = z + (z_q - z).detach()

            # perplexity
            e_mean = torch.mean(min_encodings, dim=0)
            perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

            # ===================
            z_q_list.append(z_q)
            min_encoding_list.append(min_encodings)
            min_encoding_indices_list.append(min_encoding_indices)
            if i == 0:
                total_embedding_loss = loss
                total_perplexity = perplexity
            else:
                total_embedding_loss += loss
                total_perplexity += perplexity

        # return loss, z_q, perplexity, min_encodings, min_encoding_indices
        return total_embedding_loss, z_q_list, total_perplexity / len(z_list), min_encoding_list, min_encoding_indices_list




class Encoder(nn.Module):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta
    maps to the latent space x -> z.
    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, stride_is_4=False):
        super(Encoder, self).__init__()
        kernel = 4
        stride = 2
        if stride_is_4:
            self.conv_stack = nn.Sequential(
                nn.Conv2d(in_dim, h_dim * 2, kernel_size=kernel,
                          stride=stride, padding=1),
                nn.ReLU(),
                nn.Conv2d(h_dim * 2, h_dim, kernel_size=kernel,
                          stride=stride, padding=1),
                nn.ReLU(),
                nn.Conv2d(h_dim, h_dim, kernel_size=kernel-1,
                          stride=stride-1, padding=1),
                ResidualStack(
                    h_dim, h_dim, res_h_dim, n_res_layers)

            )
        else:
            self.conv_stack = nn.Sequential(
                nn.Conv2d(in_dim, h_dim *2, kernel_size=kernel,
                          stride=stride, padding=1),
                nn.ReLU(),
                nn.Conv2d(h_dim * 2, h_dim, kernel_size=kernel-1,
                          stride=stride-1, padding=1),
                nn.ReLU(),
                nn.Conv2d(h_dim, h_dim, kernel_size=kernel-1,
                          stride=stride-1, padding=1),
                ResidualStack(
                    h_dim, h_dim, res_h_dim, n_res_layers)

            )

    def forward(self, x):
        return self.conv_stack(x)

class Decoder(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi
    maps back to the original space z -> x.
    """

    def __init__(self, output_dim, in_dim, h_dim, n_res_layers, res_h_dim, decode_x=False):
        super(Decoder, self).__init__()
        kernel = 4
        stride = 2

        if decode_x:
            self.inverse_conv_stack = nn.Sequential(
                nn.ConvTranspose2d(
                    in_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1),
                ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
                nn.ConvTranspose2d(h_dim, h_dim * 2,
                                   kernel_size=kernel, stride=stride, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(h_dim * 2, output_dim, kernel_size=kernel,
                                   stride=stride, padding=1)
            )
        else:
            self.inverse_conv_stack = nn.Sequential(
                nn.ConvTranspose2d(
                    in_dim, h_dim, kernel_size=kernel - 1, stride=stride - 1, padding=1),
                ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
                nn.ConvTranspose2d(h_dim, h_dim * 2,
                                   kernel_size=kernel - 1, stride=stride - 1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(h_dim * 2, output_dim, kernel_size=kernel,
                                   stride=stride, padding=1)
            )
    def forward(self, x):
        return self.inverse_conv_stack(x)

class VQPPF(nn.Module):
    """
    This is the implementation of the VQPPF module
    Component:
        encoder_z: Encoder for processing template information
        encoder_x: Encoder for processing search area information
        pre_quantization_conv: Channel conversion for preparing quantization execution
        vector_quantization: Quantification module for potential variables in the template and search area at all moments
        decoder: Decoder outputting time-varying target states
        decoder_z: Additional constraint terms during training for template information
        decoder_x: Additional constraint terms during training for search area information
        mu1 and mu2: Trainable weight parameters for the mixture of static initial impressions and dynamic evolution patterns

    """
    def __init__(self, input_dim, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta, save_img_embedding_map=False):
        super(VQPPF, self).__init__()

        self.encoder_z = Encoder(input_dim, h_dim, n_res_layers, res_h_dim)
        self.encoder_x = Encoder(input_dim, h_dim, n_res_layers, res_h_dim, stride_is_4=True)
        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = Decoder(input_dim, embedding_dim, h_dim, n_res_layers, res_h_dim)
        self.decoder_z = Decoder(input_dim, embedding_dim, h_dim, n_res_layers, res_h_dim)
        self.decoder_x = Decoder(input_dim, embedding_dim, h_dim, n_res_layers, res_h_dim, decode_x=True)

        self.mu1 = nn.Parameter(nn.Parameter(torch.ones(1)))
        self.mu2 = nn.Parameter(nn.Parameter(torch.ones(1)))

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, z, x_list, z_tv, train_first_phase=True):

        z_e = self.encoder_z(z)
        z_e = self.pre_quantization_conv(z_e)

        x_e_list=[z_e]
        for x in x_list:
            x_e = self.encoder_x(x)
            x_e = self.pre_quantization_conv(x_e)
            x_e_list.append(x_e)


        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(
            x_e_list)  # z_q:list:4,
        # z_q_fusion =torch.mean(torch.stack(z_q), dim=0)
        if train_first_phase:
            z_q_fusion = self.mu1 * z_q[0] + self.mu2 * (z_q[1] + z_q[2] + z_q[3])

            re_tv = self.decoder(z_q_fusion)
            L_t = self.calculate_recon_loss(z_tv, re_tv)
            re_z = self.decoder_z(z_q[0])
            L_z = self.calculate_recon_loss(z, re_z)
            L_d = 0
            for z_qx, x_lable in zip(z_q[1:], x_list[1:]):
                re_x = self.decoder_x(z_qx)
                L_d += self.calculate_recon_loss(x_lable, re_x,)
            L = L_t + L_z + L_d/3
            total_loss = L + embedding_loss
            return {"total_loss": total_loss, "perplexity": perplexity, "z_q": z_q}
        else:
            re_z = self.decoder_z(z_q[0])
            L_z = self.calculate_recon_loss(z, re_z)
            L_d = 0
            for z_qx, x_lable in zip(z_q[1:], x_list[1:]):
                re_x = self.decoder_x(z_qx)
                L_d += self.calculate_recon_loss(x_lable, re_x, )
            L = L_z + L_d / 3
            total_loss = L + embedding_loss
            return {"total_loss": total_loss, "perplexity": perplexity, "z_q": z_q}
    def encode(self, z, x_list):
        z_e = self.encoder_z(z)
        z_e = self.pre_quantization_conv(z_e)
        x_e_list = [z_e]
        for x in x_list:
            x_e = self.encoder_x(x)
            x_e = self.pre_quantization_conv(x_e)
            x_e_list.append(x_e)
        embedding_loss, z_q, _, _, _ = self.vector_quantization(
            x_e_list)  # z_q:list:3,

        return embedding_loss, z_q

    def decode(self, z_q_list, tv_label):
        # z_q_fusion = torch.mean(torch.stack(z_q_list), dim=0)
        z_q_fusion = self.mu1 * z_q_list[0] + self.mu2 * (z_q_list[1] + z_q_list[2] +  z_q_list[3])
        re_vt = self.decoder(z_q_fusion)
        L_t = self.calculate_recon_loss(tv_label, re_vt)
        return re_vt, L_t

    def decodeInference(self, z_q_list):
        # z_q_fusion = torch.mean(torch.stack(z_q_list), dim=0)  # (4,64,2,2)
        z_q_fusion = self.mu1 * z_q_list[0] + self.mu2 * (z_q_list[1] + z_q_list[2] + z_q_list[3])  # (4,64,2,2)
        re_vt = self.decoder(z_q_fusion)  # (4,768,8,8 )
        return re_vt

    def encoderZ(self, template_patch):
        z_e = self.encoder_z(template_patch)
        z_e = [self.pre_quantization_conv(z_e)]
        _, z_q, _, _, _ = self.vector_quantization(z_e)
        return z_q[0]

    def encoderX(self, search_patch):
        z_e = self.encoder_x(search_patch)
        z_e = [self.pre_quantization_conv(z_e)]
        _, z_q, _, _, _ = self.vector_quantization(z_e)
        return z_q[0]

    def calculate_recon_loss(self, x_ori, x_rec):
        sample_var = torch.var(x_ori)
        recon_loss = torch.mean((x_ori - x_rec) ** 2) / sample_var
        return recon_loss



