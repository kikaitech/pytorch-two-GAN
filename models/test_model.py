from torch.autograd import Variable
from collections import OrderedDict
import util.util as util
from . import networks


class TestModel():
    def name(self):
        return 'TestModel'

    def initialize(self, opt):
        assert(not opt.isTrain)
        self.gpu_ids = opt.gpu_ids
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join('../pytorch-two-GAN/checkpoints/', 'soccer_seg_detection_pix2pix')

        self.seg_netG  = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                          opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.detec_netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                            opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        save_seg = '%s_net_%s.pth' % ('seg_latest', 'G')
        save_seg = os.path.join(self.save_dir, save_seg)
        self.seg_netG.load_state_dict(torch.load(save_seg, map_location=lambda storage, loc: storage.cuda()))

        save_detec = '%s_net_%s.pth' % ('detec_latest', 'G')
        save_detec = os.path.join(self.save_dir, save_detec)
        self.detec_netG.load_state_dict(torch.load(save_detec, map_location=lambda storage, loc: storage.cuda()))

    def set_input(self, input):
        # we need to use single_dataset mode
        input_A = input['A']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], non_blocking=True)
        self.input_A = input_A
        self.image_paths = input['A_paths']

    def test(self):
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG(self.real_A)
        torch.jit.trace(self.seg_netG, (self.real_A)).save('./seg_netG_realA.pt')

        fake_B = (self.fake_B + 1.0)/2.0
        input_A = (self.real_A + 1.0)/2.0
        self.fake_C = (fake_B * input_A) * 2.0 - 1
     
        self.fake_D = self.detec_netG(self.fake_C)
        torch.jit.trace(self.detec_netG, (self.fake_C)).save('./detec_netG_fake_C.pt')
    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        fake_D = util.tensor2im(self.fake_D.data)
        fake_C = util.tensor2im(self.fake_C.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('fake_C', fake_C),
                            ('fake_D', fake_D)])
