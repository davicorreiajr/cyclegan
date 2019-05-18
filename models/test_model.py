from .base_model import BaseModel
from . import networks


class TestModel(BaseModel):
    """ This TesteModel can be used to generate CycleGAN results for only one direction."""
    def __init__(self, opt):
        """Initialize the TestModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags
        """
        assert(not opt.isTrain)
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        self.loss_names = []
        # specify the images you want to save/display.
        self.visual_names = ['real_A', 'fake_B']
        # specify the models you want to save to the disk. Only one generator is needed.
        self.model_names = ['G_B']

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.norm,
                                      opt.use_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see <BaseModel.load_networks>
        setattr(self, 'netG_B', self.netG)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.real_A = input['A'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        self.fake_B = self.netG(self.real_A)

    def optimize_parameters(self):
        pass
