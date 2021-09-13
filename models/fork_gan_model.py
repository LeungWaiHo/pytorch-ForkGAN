import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class ForkGANModel(BaseModel):
    """
    This class implements the ForkGAN model, for learning image-to-image translation without paired data.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the ForkGAN class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G', 'G_adv_total', 'G_adv', 'G_rec', 'G_recfake', 'G_cycle', 'G_rec', 'G_percep', 'cls', 'G_cls']
        self.loss_names.append('D')
        self.loss_names.append('RC')
        self.loss_names.append('RF')
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # visual_names_A = ['real_A', 'fake_B', 'rec_realA', 'rec_fakeB']
        # visual_names_B = ['real_B', 'fake_A', 'rec_realB', 'rec_fakeA']
        visual_names_A = ['realA_percep', 'fakeB_percep']
        visual_names_B = ['realB_percep', 'fakeA_percep']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'C', 'D_A', 'D_B', 'RC_A', 'RC_B', 'RF_A', 'RF_B']
        else:  # during test time, only load G
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators and domain-agnostic classifier
            # define discriminators
            self.netD_A = networks.define_D(opt.input_nc, opt.ndf, opt.n_scale, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.n_scale, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netRC_A = networks.define_D(opt.input_nc, opt.ndf, opt.n_scale, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netRC_B = networks.define_D(opt.input_nc, opt.ndf, opt.n_scale, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netRF_A = networks.define_D(opt.input_nc, opt.ndf, opt.n_scale, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netRF_B = networks.define_D(opt.input_nc, opt.ndf, opt.n_scale, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # define domain-agnostic classifier
            self.netC = networks.define_C(256, opt.ndf, opt.n_scale, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            
            # define loss functions
            self.criterionGAN_list = networks.GANLoss_List(opt.gan_mode).to(self.device)  # define GAN loss list.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionReconstruction = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_C = torch.optim.Adam(itertools.chain(self.netC.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_RC = torch.optim.Adam(itertools.chain(self.netRC_A.parameters(), self.netRC_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_RF = torch.optim.Adam(itertools.chain(self.netRF_A.parameters(), self.netRF_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_C)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_RC)
            self.optimizers.append(self.optimizer_RF)
            

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.A_label = torch.zeros([self.real_A.size(0)], dtype=torch.long).to(self.device)
        self.B_label = torch.ones([self.real_B.size(0)], dtype=torch.long).to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # Obtain outputs from each generator G_A and G_B
        self.fake_B, self.rec_realA, self.realA_percep  = self.netG_A(self.real_A)  # G_A(A)
        self.fake_A_, self.rec_fakeB, self.fakeB_percep = self.netG_B(self.fake_B)  # G_B(G_A(A))
        self.fake_A, self.rec_realB, self.realB_percep  = self.netG_B(self.real_B)  # G_B(B)
        self.fake_B_, self.rec_fakeA, self.fakeA_percep = self.netG_A(self.fake_A)  # G_A(G_B(B))
        if self.isTrain:
            # Obtain outputs from domain-agnostic classifier
            self.realA_percep_logit = self.netC(self.realA_percep)
            self.realB_percep_logit = self.netC(self.realB_percep)
            self.fakeA_percep_logit = self.netC(self.fakeA_percep)
            self.fakeB_percep_logit = self.netC(self.fakeB_percep)
        
    
    ##########################################################################
    # Define generator loss
    ##########################################################################
    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A, _, _ = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B, _, _ = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0
        
        # Initialize generator's loss items
        self.loss_G_adv_total = 0.0
        self.loss_G_adv = 0.0
        self.loss_G_rec = 0.0
        self.loss_G_recfake = 0.0

        for i in range(self.opt.n_d):
            # Generator adversarial loss
            self.loss_G_A = self.criterionGAN_list(self.netD_A(self.fake_B), True)
            self.loss_G_B = self.criterionGAN_list(self.netD_B(self.fake_A), True)

            # Generator adversarial reconstruction loss
            self.loss_G_A_rec = self.criterionGAN_list(self.netRC_A(self.rec_realA), True)
            self.loss_G_B_rec = self.criterionGAN_list(self.netRC_B(self.rec_realB), True)

            # Generator adversarial refine loss
            self.loss_G_A_ref = self.criterionGAN_list(self.netRF_A(self.rec_fakeB), True)
            self.loss_G_B_ref = self.criterionGAN_list(self.netRF_B(self.rec_fakeA), True)

            self.loss_G_adv += 0.5 * (self.loss_G_A + self.loss_G_B)
            self.loss_G_rec += 0.5 * (self.loss_G_A_rec + self.loss_G_B_rec)
            self.loss_G_recfake += 0.5 * (self.loss_G_A_ref + self.loss_G_B_ref)
            
            # GAN adversarial loss
            self.loss_G_adv_total += self.loss_G_adv + self.loss_G_rec + self.loss_G_recfake

        # Generator classification loss
        self.loss_G_cls = F.cross_entropy(self.fakeA_percep_logit.reshape(-1, 2), self.A_label)*0.5 +\
                            F.cross_entropy(self.fakeB_percep_logit.reshape(-1, 2), self.B_label)*0.5

        # Generator perceptual loss
        self.loss_G_percep = torch.mean(torch.abs(torch.mean(self.realA_percep, dim=3) - torch.mean(self.fakeB_percep, dim=3))) +\
                                    torch.mean(torch.abs(torch.mean(self.realB_percep, dim=3) - torch.mean(self.fakeA_percep, dim=3)))

        # Generator cycle loss
        self.loss_G_cycle_A = self.criterionCycle(self.real_A, self.fake_A_)
        self.loss_G_cycle_B = self.criterionCycle(self.real_B, self.fake_B_)
        self.loss_G_cycle = self.loss_G_cycle_A + self.loss_G_cycle_B

        # Generator reconstruction loss
        self.loss_G_rec_A = self.criterionReconstruction(self.rec_realA, self.real_A)
        self.loss_G_rec_B = self.criterionReconstruction(self.rec_realB, self.real_B)
        self.loss_G_rec = self.loss_G_rec_A + self.loss_G_rec_B

        # combined loss and calculate gradients
        self.loss_G = self.loss_G_adv_total + self.loss_G_cls + self.loss_G_percep +\
                         self.opt.gamma * self.loss_G_cycle + self.opt.eps * self.loss_G_rec
        self.loss_G.backward(retain_graph=True)

        

    ##########################################################################
    # Define domain-agnostic classifier loss
    ##########################################################################
    def backward_C(self):
        """Calculate domain-agnostic classifier loss for generator and itself"""
        self.loss_cls = F.cross_entropy(self.realA_percep_logit.reshape(-1, 2), self.A_label)*0.25 +\
                            F.cross_entropy(self.realB_percep_logit.reshape(-1, 2), self.B_label)*0.25 +\
                                F.cross_entropy(self.fakeB_percep_logit.reshape(-1, 2), self.A_label)*0.25 +\
                                    F.cross_entropy(self.fakeA_percep_logit.reshape(-1, 2), self.B_label)*0.25
        self.loss_cls.backward()


    ##########################################################################
    # Define discriminator loss (same as CycleGAN)  ===>  translation
    ##########################################################################
    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN_list(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN_list(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D

    def backward_D(self):
        """Calculate GAN loss for discriminator D
        We also call loss_D.backward() to calculate the gradients for each discriminator.
        """
        self.loss_D_list = []
        for i in range(self.opt.n_d):
            fake_B = self.fake_B_pool.query(self.fake_B)
            self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
            fake_A = self.fake_A_pool.query(self.fake_A)
            self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
            
            self.loss_D_item = self.loss_D_A + self.loss_D_B
            self.loss_D_item.backward()
            self.loss_D_list.append(self.loss_D_item)
        self.loss_D = sum(self.loss_D_list)


    ##########################################################################
    # Define reconstruction loss  ===>  reconstruction
    ##########################################################################
    def backward_RC_basic(self, netRC, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netRC (network)      -- the discriminator RC
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        """
        # Real
        pred_real = netRC(real)
        loss_RC_real = self.criterionGAN_list(pred_real, True)
        # Fake
        pred_fake = netRC(fake.detach())
        loss_RC_fake = self.criterionGAN_list(pred_fake, False)
        # Combined loss and calculate gradients
        loss_RC = (loss_RC_real + loss_RC_fake) * 0.5
        return loss_RC

    def backward_RC(self):
        """Calculate GAN loss for discriminator RC
        We also call loss_RC.backward() to calculate the gradients for each discriminator.
        """
        self.loss_RC_list = []
        for i in range(self.opt.n_d):
            rec_realA = self.fake_B_pool.query(self.rec_realA)
            self.loss_RC_A = self.backward_RC_basic(self.netRC_A, self.real_A, rec_realA)
            rec_realB = self.fake_A_pool.query(self.rec_realB)
            self.loss_RC_B = self.backward_RC_basic(self.netRC_B, self.real_B, rec_realB)

            self.loss_RC_item = self.loss_RC_A + self.loss_RC_B
            self.loss_RC_item.backward()
            self.loss_RC_list.append(self.loss_RC_item)
        self.loss_RC = sum(self.loss_RC_list)


    ##########################################################################
    # Define refinement loss  ===>  refinement
    ##########################################################################
    def backward_RF_basic(self, netRF, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netRF (network)      -- the discriminator RF
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        """
        # Real
        pred_real = netRF(real)
        loss_RF_real = self.criterionGAN_list(pred_real, True)
        # Fake
        pred_fake = netRF(fake.detach())
        loss_RF_fake = self.criterionGAN_list(pred_fake, False)
        # Combined loss and calculate gradients
        loss_RF = (loss_RF_real + loss_RF_fake) * 0.5
        return loss_RF

    def backward_RF(self):
        """Calculate GAN loss for discriminator RF
        We also call loss_RF.backward() to calculate the gradients for each discriminator.
        """
        self.loss_RF_list = []
        for i in range(self.opt.n_d):
            rec_fakeB = self.fake_B_pool.query(self.rec_fakeB)
            self.loss_RF_A = self.backward_RF_basic(self.netRF_A, self.real_B, rec_fakeB)
            rec_fakeA = self.fake_A_pool.query(self.rec_fakeA)
            self.loss_RF_B = self.backward_RF_basic(self.netRF_B, self.real_A, rec_fakeA)

            self.loss_RF_item = self.loss_RF_A + self.loss_RF_B
            self.loss_RF_item.backward()
            self.loss_RF_list.append(self.loss_RF_item)
        self.loss_RF = sum(self.loss_RF_list)


    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        
        # C
        self.set_requires_grad([self.netC], True)
        self.optimizer_C.zero_grad()   # set C's gradients to zero
        self.backward_C()              # calculate gradients for C
        self.optimizer_C.step()        # update C's weights
        
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D()              # calculate graidents for D
        self.optimizer_D.step()        # update D_A and D_B's weights

        # RC_A and RC_B
        self.set_requires_grad([self.netRC_A, self.netRC_B], True)
        self.optimizer_RC.zero_grad()   # set RC_A and RC_B's gradients to zero
        self.backward_RC()              # calculate graidents for RC
        self.optimizer_RC.step()        # update RC_A and RC_B's weights

        # RF_A and RF_B
        self.set_requires_grad([self.netRF_A, self.netRF_B], True)
        self.optimizer_RF.zero_grad()   # set RF_A and RF_B's gradients to zero
        self.backward_RF()              # calculate graidents for RF
        self.optimizer_RF.step()        # update RF_A and RF_B's weights