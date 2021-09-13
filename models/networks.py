import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


##############################################################################
# Define networks (including generator, discriminator, domain-agnostic classifier)
##############################################################################
def define_G(input_nc, output_nc, ngf, norm='instance', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator
    Returns a generator
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, n_scale=2, norm='instance', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator
    Returns a discriminator
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    net = Discriminator(input_nc, ndf, n_scale, norm_layer=norm_layer)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_C(input_nc, ndf, n_scale=2, norm='instance', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a domain-agnostic classifier
    Returns a domain-agnostic classifier
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    net = DomainAgnosticClassifier(input_nc, ndf, norm_layer=norm_layer)
    return init_net(net, init_type, init_gain, gpu_ids)



##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class GANLoss_List(nn.Module):
    """Define different GAN objectives.

    The GANLoss_List class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss_List class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss_List, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction_list, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor list) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """
        target_tensor_list = []
        for i in range(len(prediction_list)):
            if target_is_real:
                target_tensor = self.real_label
            else:
                target_tensor = self.fake_label
            target_tensor_list.append(target_tensor.expand_as(prediction_list[i]))
        return target_tensor_list

    def __call__(self, prediction_list, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor_list = self.get_target_tensor(prediction_list, target_is_real)
            loss = 0.0
            for i in range(len(prediction_list)):
                loss += self.loss(prediction_list[i], target_tensor_list[i])
        elif self.gan_mode == 'wgangp':
            loss = 0.0
            for i in range(len(prediction_list)):
                if target_is_real:
                    loss += -prediction_list[i].mean()
                else:
                    loss += prediction_list[i].mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer
    """
    def __init__(self, sigma=0.02, is_relative_detach=True):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        return x + torch.normal(mean=0, std=self.sigma, size=x.size()).cuda()


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
        """
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        ### Encoder architecture
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),

            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),

            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 2),
            nn.ReLU(True),

            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 4),
            nn.ReLU(True),

            ResnetBlockDilated(ngf * 4, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            ResnetBlockDilated(ngf * 4, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            ResnetBlockDilated(ngf * 4, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            ResnetBlockDilated(ngf * 4, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            ResnetBlockDilated(ngf * 4, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        )
        

        ### Translation decoder architecture
        self.tr_decoder = nn.Sequential(
            ResnetBlockDilated(ngf * 4, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            ResnetBlockDilated(ngf * 4, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            ResnetBlockDilated(ngf * 4, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            ResnetBlockDilated(ngf * 4, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            
            nn.ConvTranspose2d(ngf * 4, ngf * 2,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
            norm_layer(ngf * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 2, ngf,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),

            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        )


        ### Reconstruction decoder architecture
        self.re_decoder = nn.Sequential(
            GaussianNoise(0.02),
            ResnetBlockDilated(ngf * 4, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            GaussianNoise(0.02),
            ResnetBlockDilated(ngf * 4, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            ResnetBlockDilated(ngf * 4, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            ResnetBlockDilated(ngf * 4, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),

            nn.ConvTranspose2d(ngf * 4, ngf * 2,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
            norm_layer(ngf * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 2, ngf,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),

            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        )


    def forward(self, input):
        """Standard forward"""
        """Return translation output, reconstruction output, encoder output"""

        return self.tr_decoder(self.encoder(input)), self.re_decoder(self.encoder(input)), self.encoder(input)
        # Size:        [1, 3, img_w, img_h]                  [1, 3, img_w, img_h]           [1, img_w, 64, 64]

class ResnetBlockDilated(nn.Module):
    """Define a Resnet dilated block"""

    def __init__(self, dim, norm_layer, use_dropout, use_bias, dilation=2):
        """Initialize the Resnet dilated block"""

        super(ResnetBlockDilated, self).__init__()
        self.conv_block = self.build_conv_block(dim, norm_layer, use_dropout, use_bias, dilation)

    def build_conv_block(self, dim, norm_layer, use_dropout, use_bias, dilation):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
            dilation(int)       -- dilation rate if use dilated_conv
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 2
        
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias, dilation=dilation), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias, dilation=dilation), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class DomainAgnosticClassifier(nn.Module):
    """Defines a domain-agnostic classifier"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.InstanceNorm2d):
        """Construct a domain-agnostic classifier
        """
        super(DomainAgnosticClassifier, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf * 4, kernel_size=kw, stride=2, padding=padw),
            norm_layer(ndf*4),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [
            nn.Conv2d(ndf * 4, ndf * 2, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [
            nn.Conv2d(ndf * 2, ndf * 2, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.Conv2d(ndf * 2, 2, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return torch.mean(self.model(input), dim=[0, 2, 3]).reshape(1, 1, 1, 2)


class NumSteps(nn.Module):
    """Num steps conv + ins_bn + leakyrelu"""
    
    def __init__(self, input_nc,ndf=64, n_scale=2, norm_layer=nn.InstanceNorm2d):
        """Construct a multi-layer conv+bn+lrelu"""
        
        super(NumSteps, self).__init__()
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True)
        ]

        for n in range(1, n_scale):
            sequence += [
                nn.Conv2d(ndf, ndf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf),
                nn.LeakyReLU(0.2, True)
            ]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class OneStepDilated(nn.Module):
    """One steps dilated_conv + ins_bn + leakyrelu"""
    
    def __init__(self, input_nc, ndf=64, norm_layer=nn.InstanceNorm2d):
        """Construct one-layer conv+bn+lrelu"""
        
        super(OneStepDilated, self).__init__()
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=3, dilation=2),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True)
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class DiscriminatorDown(nn.Module):
    """Equal to dis_down in tensorflow inplementation of ForkGAN"""

    def __init__(self, input_nc, ndf=64, n_scale=2, norm_layer=nn.InstanceNorm2d):
        """Construct dis_down"""

        super(DiscriminatorDown, self).__init__()
        self.num_steps = NumSteps(input_nc, ndf, n_scale, norm_layer)
        self.one_step_dilated = OneStepDilated(input_nc, ndf, norm_layer)
        self.n_scale = n_scale

    def forward(self, inputs):
        backpack = inputs[0]
        for i in range(self.n_scale):
            if i == self.n_scale - 1:
                inputs[i] = self.num_steps(backpack)
            else:
                inputs[i] = self.one_step_dilated(inputs[i+1])
        return inputs


class FinalConv(nn.Module):
    """Equal to final_conv in tensorflow inplementation of ForkGAN"""

    def __init__(self, input_nc, n_scale):
        """Construct a final_conv layer for every scale"""

        super(FinalConv, self).__init__()
        self.conv = nn.Conv2d(input_nc, 1, kernel_size=4, stride=1, padding=1)
        self.n_scale = n_scale

    def forward(self, inputs):
        for i in range(self.n_scale):
            inputs[i] = self.conv(F.pad(inputs[i], (0, 1, 0, 1)))
        return inputs


class Discriminator(nn.Module):
    """Equal to discriminator in tensorflow inplementation of ForkGAN"""

    def __init__(self, input_nc, ndf=64, n_scale=2, norm_layer=nn.InstanceNorm2d):
        """Construct discriminator including four dis_down and one final_conv"""

        super(Discriminator, self).__init__()
        self.disdown_1 = DiscriminatorDown(input_nc, ndf*1, n_scale, norm_layer)
        self.disdown_2 = DiscriminatorDown(ndf*1, ndf*2, n_scale, norm_layer)
        self.disdown_3 = DiscriminatorDown(ndf*2, ndf*4, n_scale, norm_layer)
        self.disdown_4 = DiscriminatorDown(ndf*4, ndf*8, n_scale, norm_layer)
        self.final_conv = FinalConv(ndf*8, n_scale=2)
        self.n_scale = n_scale

    def forward(self, input):
        inputs = []
        for i in range(self.n_scale):
            inputs.append(F.interpolate(input, size=(input.size(2)//(2**i), input.size(3)//(2**i)), 
                            mode='bicubic', align_corners=True))
        outs_1 = self.disdown_1(inputs)
        outs_2 = self.disdown_2(outs_1)
        outs_3 = self.disdown_3(outs_2)
        outs_4 = self.disdown_4(outs_3)
        outs = self.final_conv(outs_4)
        return outs



# real_A = torch.rand([1, 3, 256, 256])
# real_B = torch.rand([1, 3, 256, 256])

# ##############################################################################
# # Generator
# ##############################################################################
# netG_A = define_G(3, 3, 64, 'instance', False, 'normal', 0.02, [0])
# netG_B = define_G(3, 3, 64, 'instance', False, 'normal', 0.02, [0])

# fake_B, rec_realA, realA_percep = netG_A(real_A)
# fake_A_, rec_fakeB, fakeB_percep = netG_B(fake_B)
# fake_A, rec_realB, realB_percep = netG_B(real_B)
# fake_B_, rec_fakeA, fakeA_percep = netG_A(fake_A)

# print('Generator Test Successfully!')

# percep_loss = torch.mean(torch.abs(torch.mean(realA_percep, dim=3) - torch.mean(fakeB_percep, dim=3))) +\
#                 torch.mean(torch.abs(torch.mean(realB_percep, dim=3) - torch.mean(fakeA_percep, dim=3)))
# print(percep_loss)

# # ##############################################################################
# # # Domain-Agnostic Classifier
# # ##############################################################################
# # netC = define_C(256, 64, 2, 'instance', 'normal', 0.02, [0])

# # realA_percep_logit = netC(realA_percep)
# # realB_percep_logit = netC(realB_percep)
# # fakeA_percep_logit = netC(fakeA_percep)
# # fakeB_percep_logit = netC(fakeB_percep)

# # print('Domain-Agnostic Classifier Test Successfully!')


# # ##############################################################################
# # # Discriminator
# # ##############################################################################
# # netD_A = define_D(3, 64, 2, 'instance', 'normal', 0.02, [0])
# # netD_B = define_D(3, 64, 2, 'instance', 'normal', 0.02, [0])
# # netDrec_A = define_D(3, 64, 2, 'instance', 'normal', 0.02, [0])
# # netDrec_B = define_D(3, 64, 2, 'instance', 'normal', 0.02, [0])
# # netDref_A = define_D(3, 64, 2, 'instance', 'normal', 0.02, [0])
# # netDref_B = define_D(3, 64, 2, 'instance', 'normal', 0.02, [0])

# # print('Discriminator Test Successfully!')



# # gpu_ids = [0,1,2]
# # device = torch.device('cuda:{}'.format(gpu_ids[0]))
# # criterionGAN_list = GANLoss_List('lsgan').to(device)

# # loss_G_A = criterionGAN_list(netD_A(fake_B), True)
# # loss_G_B = criterionGAN_list(netD_B(fake_A), True)

# # loss_G_A_rec = criterionGAN_list(netDrec_A(rec_realA), True)
# # loss_G_B_rec = criterionGAN_list(netDrec_B(rec_realB), True)

# # loss_G_A_ref = criterionGAN_list(netDref_A(rec_fakeB), True)
# # loss_G_B_ref = criterionGAN_list(netDref_B(rec_fakeA), True)