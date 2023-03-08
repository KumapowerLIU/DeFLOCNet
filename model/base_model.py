import os
import torch


class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.model_names = None
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        # self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.device = (
            torch.device("cuda:{}".format(self.gpu_ids[0]))
            if self.gpu_ids
            else torch.device("cpu")
        )
        self.input_img = None
        self.gt_local = None
        self.gt = None
        self.mask = None
        self.inv_mask = None
        self.sketch = None
        self.color = None
        self.mask_color = None
        self.input_noise = None
        self.input_sketch = None
        self.input_color = None
        self.crop_x = None
        self.crop_y = None

    def name(self):
        return 'BaseModel'

    def set_input(self, **kwargs):
        pass

    def forward(self):
        pass

    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_errors(self):
        pass

    def save(self, label):
        pass

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def save_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = "%s_net_%s.pth" % (which_epoch, name)
                save_path = os.path.join(self.save_dir, save_filename).replace(
                    "\\", "/"
                )
                net = getattr(self, "net" + name)
                optimize = getattr(self, "optimizer_" + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(
                        {
                            "net": net.module.cpu().state_dict(),
                            "optimize": optimize.state_dict(),
                        },
                        save_path,
                    )
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    # helper loading function that can be used by subclasses
    def load_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = "%s_net_%s.pkl" % (which_epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)

                net = getattr(self, "net" + name)

                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                state_dict = torch.load(
                    load_path.replace("\\", "/"), map_location=str(self.device)
                )
                if self.isTrain:
                    optimize = getattr(self, "optimizer_" + name)
                    if "optimize" in state_dict:
                        optimize.load_state_dict(state_dict["optimize"])
                if 'net' in state_dict:
                    net.load_state_dict(state_dict["net"])
                else:
                    net.load_state_dict(state_dict)

    #
    # # helper saving function that can be used by subclasses
    # def save_network(self, network, network_label, epoch_label, gpu_ids):
    #     save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    #     save_path = os.path.join(self.save_dir, save_filename)
    #     torch.save(network.cpu().state_dict(), save_path)
    #     if len(gpu_ids) and torch.cuda.is_available():
    #         network.cuda(gpu_ids[0])
    #
    # # helper loading function that can be used by subclasses
    # def load_network(self, network, network_label, epoch_label):
    #     save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    #     save_path = os.path.join(self.save_dir, save_filename)
    #     network.load_state_dict(torch.load(save_path))

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
