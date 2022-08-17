#This code is based on: https://github.com/SimonKohl/probabilistic_unet

from unet.unet_model import UNet
from utils import init_weights,init_weights_orthogonal_normal, l2_regularisation
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl
import torch
import torch.nn as nn
from persistence_ws_boundary import *
from discrete_morse_theory import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    """
    A convolutional neural network, consisting of len(num_filters) times a block of no_convs_per_block convolutional layers,
    after each block a pooling operation is performed. And after each convolutional layer a non-linear (ReLU) activation function is applied.
    """
    def __init__(self, input_channels, num_filters, no_convs_per_block, initializers, padding=True, posterior=False):
        super(Encoder, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.num_filters = num_filters

        if posterior:
            #To accomodate for the mask that is concatenated at the channel axis, we increase the input_channels.
            self.input_channels += 1
        layers = []
        for i in range(len(self.num_filters)):
            """
            Determine input_dim and output_dim of conv layers in this block. The first layer is input x output,
            All the subsequent layers are output x output.
            """
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = num_filters[i]
            
            if i != 0:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))
            
            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=int(padding)))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_per_block-1):
                layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=int(padding)))
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

        self.layers.apply(init_weights)

    def forward(self, input):
        output = self.layers(input)
        return output

class AxisAlignedConvGaussian(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix.
    """
    def __init__(self, input_channels, num_filters, no_convs_per_block, latent_dim, initializers, posterior=False):
        super(AxisAlignedConvGaussian, self).__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        self.num_filters = num_filters
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim
        self.posterior = posterior
        if self.posterior:
            self.name = 'Posterior'
        else:
            self.name = 'Prior'
        self.encoder = Encoder(self.input_channels, self.num_filters, self.no_convs_per_block, initializers, posterior=self.posterior)
        self.conv_layer = nn.Conv2d(num_filters[-1], 2 * self.latent_dim, (1,1), stride=1)
        self.show_img = 0
        self.show_seg = 0
        self.show_concat = 0
        self.show_enc = 0
        self.sum_input = 0

        nn.init.kaiming_normal_(self.conv_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.conv_layer.bias)

    def forward(self, input, segm=None):

        #If segmentation is not none, concatenate the mask to the channel axis of the input
        if segm is not None:
            self.show_img = input
            self.show_seg = segm
            input = torch.cat((input, segm), dim=1)
            self.show_concat = input
            self.sum_input = torch.sum(input)

        encoding = self.encoder(input)
        self.show_enc = encoding

        #We only want the mean of the resulting hxw image
        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)

        #Convert encoding to 2 x latent dim and split up for mu and log_sigma
        mu_log_sigma = self.conv_layer(encoding)

        #We squeeze the second dimension twice, since otherwise it won't work when batch size is equal to 1
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)

        mu = mu_log_sigma[:,:self.latent_dim]
        log_sigma = mu_log_sigma[:,self.latent_dim:]
        # print("mean is {}, and deviation is {}".format(mu, log_sigma))

        #This is a multivariate normal with diagonal covariance matrix sigma
        #https://github.com/pytorch/pytorch/pull/11178
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)),1)
        return mu, dist

class ProbabilisticUnet(nn.Module):
    """
    A probabilistic UNet (https://arxiv.org/abs/1806.05034) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: is a list consisint of the amount of filters layer
    latent_dim: dimension of the latent space
    no_cons_per_block: no convs per block in the (convolutional) encoder of prior and posterior
    """

    def __init__(self, input_channels=1, num_classes=1, num_filters=[32,64,128,192], latent_dim=1):
        super(ProbabilisticUnet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.no_convs_per_block = 3
        self.initializers = {'w':'he_normal', 'b':'normal'}
        self.z_prior_sample = 0

        self.unet = UNet(self.input_channels, self.num_classes, self.num_filters).to(device)
        self.prior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, self.no_convs_per_block, self.latent_dim,  self.initializers,).to(device)
        self.posterior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, self.no_convs_per_block, self.latent_dim, self.initializers, posterior=True).to(device)

    def forward(self, patch, segm, training=True):
        """
        Construct prior latent space for patch and run patch through UNet,
        in case training is True also construct posterior latent space
        """
        if training:
            self.mu, self.posterior_latent_space = self.posterior.forward(patch, segm)
        self.mu, self.prior_latent_space = self.prior.forward(patch)
        self.unet_features = torch.sigmoid(self.unet.forward(patch))

        return self.unet_features

    def sample(self, testing=False):
        """
        Sample a segmentation by reconstructing from a prior sample
        and combining this with UNet features
        """
        if testing == False:
            z_prior = self.prior_latent_space.rsample()
            self.z_prior_sample = z_prior
        else:
            #You can choose whether you mean a sample or the mean here. For the GED it is important to take a sample.
            #z_prior = self.prior_latent_space.base_dist.loc 

            num_preds = 20
            for i in range(num_preds):
                z_prior = self.prior_latent_space.sample()
                self.z_prior_sample = z_prior
                if (z_prior <= 0) | (z_prior >= 1):
                    continue
                else:
                    return z_prior, persistence_boundary(self.unet_features,z_prior)
                    break;
        # return self.unet_features

    def sample_vessel(self, testing=False, use_prior_mean = False):
        """
        Sample a segmentation by reconstructing from a prior sample
        and combining this with UNet features
        """
        if testing == False:
            z_prior = self.prior_latent_space.rsample()
            self.z_prior_sample = z_prior
        else:
            #You can choose whether you mean a sample or the mean here. For the GED it is important to take a sample.
            #z_prior = self.prior_latent_space.base_dist.loc 
            num_preds = 20
            for i in range(num_preds):
                z_prior = self.prior_latent_space.sample()
                self.z_prior_sample = z_prior
                if (z_prior <= 0) | (z_prior >= 1):
                    continue
                else:
                    return z_prior, dmt(self.unet_features,z_prior)
                    break;

    def reconstruct(self, use_posterior_mean=False, calculate_posterior=False, z_posterior=None):
        """
        Reconstruct a segmentation from a posterior sample (decoding a posterior sample) and UNet feature map
        use_posterior_mean: use posterior_mean instead of sampling z_q
        calculate_posterior: use a provided sample or sample from posterior latent space
        """
        if use_posterior_mean:
            # z_posterior = self.posterior_latent_space.loc
            z_posterior = self.mu
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
        return persistence_boundary(self.unet_features, z_posterior)

    def reconstruct_vessel(self, use_posterior_mean=False, calculate_posterior=False, z_posterior=None):
        """
        Reconstruct a segmentation from a posterior sample (decoding a posterior sample) and UNet feature map
        use_posterior_mean: use posterior_mean instead of sampling z_q
        calculate_posterior: use a provided sample or sample from posterior latent space
        """
        if use_posterior_mean:
            # z_posterior = self.posterior_latent_space.loc
            z_posterior = self.mu
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
        return dmt(self.unet_features, z_posterior)

    def kl_divergence(self, analytic=True, calculate_posterior=False, z_posterior=None):
        """
        Calculate the KL divergence between the posterior and prior KL(Q||P)
        analytic: calculate KL analytically or via sampling from the posterior
        calculate_posterior: if we use samapling to approximate KL we can sample here or supply a sample
        """
        if analytic:
            #Neeed to add this to torch source code, see: https://github.com/pytorch/pytorch/issues/13545
            kl_div = kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space)
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
            log_posterior_prob = self.posterior_latent_space.log_prob(z_posterior)
            log_prior_prob = self.prior_latent_space.log_prob(z_posterior)
            kl_div = log_posterior_prob - log_prior_prob
        return kl_div

    def elbo(self, segm, pretrain, weight_reconstruction, weight_kl, dataset, analytic_kl=True, reconstruct_posterior_mean=False):
        """
        Calculate the evidence lower bound of the log-likelihood of P(Y|X)
        """

        criterion = nn.BCELoss(size_average = False, reduce=False, reduction=None)
        z_posterior = self.posterior_latent_space.rsample()
        
        #Here we use the posterior sample sampled above
        if pretrain == 'True':
            self.kl = 0
            self.mean_reconstruction_loss = 0
        else:
            self.kl = torch.mean(self.kl_divergence(analytic=analytic_kl, calculate_posterior=False, z_posterior=z_posterior))
            if dataset == 'DRIVE':
                self.reconstruction = self.reconstruct_vessel(use_posterior_mean=reconstruct_posterior_mean, calculate_posterior=False, z_posterior=z_posterior)     
            else: 
                self.reconstruction = self.reconstruct(use_posterior_mean=reconstruct_posterior_mean, calculate_posterior=False, z_posterior=z_posterior)      
            reconstruction_loss = criterion(input=self.unet_features , target=segm) * torch.from_numpy(self.reconstruction).cuda()
            # self.reconstruction_loss = torch.sum(reconstruction_loss)
            self.mean_reconstruction_loss = torch.mean(reconstruction_loss)

        seg_loss = criterion(input=self.unet_features, target=segm)
        # self.seg_loss = torch.sum(seg_loss)
        self.mean_seg_loss = torch.mean(seg_loss)

        return -(self.mean_seg_loss + weight_reconstruction * self.mean_reconstruction_loss + weight_kl * self.kl)
