import random
import torch
import torch.utils.data
from PIL import Image
from glob import glob
import util.util as util

class Places2(torch.utils.data.Dataset):
    def __init__(self, img_root, sk_root, mask_root,color_root,color_mask, in_sk_root, img_transform, mask_transform,
                 split='train'):
        super(Places2, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.Train=False

        # use about 8M images in the challenge dataset
        if split == 'train':
            self.paths = sorted(glob('{:s}/*'.format(img_root),
                              recursive=True))
            self.sk_path=sorted(glob('{:s}/*'.format(sk_root),
                              recursive=True))
            self.color_path=sorted(glob('{:s}/*'.format(color_root),
                              recursive=True))
            self.in_sk_path=sorted(glob('{:s}/*.png'.format(in_sk_root),
                              recursive=True))


            self.Train=True

        else:
            self.paths=sorted(glob('{:s}/*'.format(sk_root),
                              recursive=True))
            self.sk_path = sorted(glob('{:s}/*'.format(img_root),
                              recursive=True))
            self.color_path=sorted(glob('{:s}/*'.format(color_root),
                              recursive=True))
            self.in_sk_path=sorted(glob('{:s}/*.png'.format(in_sk_root),
                              recursive=True))


        self.mask_paths = sorted(glob('{:s}/*'.format(mask_root)))
        self.color_mask_path = sorted(glob('{:s}/*'.format(color_mask)))

        self.N_mask = len(self.mask_paths)
        self.N_mask_co = len(self.color_mask_path)


    def __getitem__(self, index):

        mask_path=self.mask_paths[random.randint(0, self.N_mask - 1)]
        mask_path_c = self.color_mask_path [random.randint(0, self.N_mask_co- 1)]

        sketch_img=Image.open(self.paths[index])
        color_img=Image.open(self.color_path[0])
        gt_img = Image.open(self.sk_path [0])
        in_sk_img=Image.open(self.in_sk_path[index])
        if self.Train==True:
            mask = Image.open(mask_path)
            mask_c = Image.open(mask_path_c)
        else:
            # mask = Image.open(mask_path)
            # mask_c = Image.open(mask_path_c)
            mask = Image.open(self.mask_paths[0])
            mask_c = Image.open(self.color_mask_path[0])




        rotate_set_data=random.randint(0,1)
        if rotate_set_data==1 and self.Train==True:
            gt_img = gt_img.transpose(Image.FLIP_LEFT_RIGHT)
            sketch_img = sketch_img.transpose(Image.FLIP_LEFT_RIGHT)
            color_img = color_img.transpose(Image.FLIP_LEFT_RIGHT)
            in_sk_img = in_sk_img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            mask_c = mask_c.transpose(Image.FLIP_LEFT_RIGHT)



        gt_img = self.img_transform(gt_img.convert('RGB'))
        sketch_img = self.mask_transform(sketch_img.convert('RGB'))
        color_img = self.img_transform(color_img.convert('RGB'))
        mask = self.mask_transform(mask.convert('RGB'))
        mask_c = self.mask_transform(mask_c.convert('RGB'))
        in_sk_img= self.mask_transform(in_sk_img.convert('RGB'))
        return gt_img, sketch_img, color_img, mask, mask_c, in_sk_img

    def __len__(self):
        return len(self.paths)