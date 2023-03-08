import cv2
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from ui.ui import Ui_Form
from ui.mouse_event import GraphicsScene
from ntpath import basename
from PIL import Image
import numpy as np
import time
from model.models import create_model
import torch
import os
import torchvision
import torchvision.transforms as transforms
from config.test_config import TestConfig

basic_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
import sys

sys.path.append(basic_dir)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Ex(QWidget, Ui_Form):
    def __init__(self, model):
        super().__init__()
        self.demo_save_dir = './demo_results'
        if os.path.exists(self.demo_save_dir) is False:
            os.makedirs(self.demo_save_dir)
        self.setupUi(self)
        self.show()
        self.model = model

        self.output_img = None

        self.mat_img = None

        self.ld_mask = None
        self.ld_sk = None

        self.modes = [0, 0, 0]
        self.mouse_clicked = False
        self.scene = GraphicsScene(self.modes)
        self.graphicsView.setScene(self.scene)
        self.graphicsView.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.result_scene = QGraphicsScene()
        self.graphicsView_2.setScene(self.result_scene)
        self.graphicsView_2.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_2.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.dlg = QColorDialog(self.graphicsView)
        self.color = None

    def trans_to_pytorch_img(self, img):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
        return transform(img)

    def trans_to_pytorch_mask(self, mask):
        transform = transforms.Compose([transforms.ToTensor()])
        return transform(mask)

    def mode_select(self, mode):
        for i in range(len(self.modes)):
            self.modes[i] = 0
        self.modes[mode] = 1

    def open(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File",
                                                  QDir.currentPath())
        if fileName:
            image = QPixmap(fileName)
            mat_img_2 = cv2.imread(fileName)
            name = basename(str(fileName))
            name = name.split('.')[0]
            mat_img = cv2.imread(fileName)
            if image.isNull():
                QMessageBox.information(self, "Image Viewer",
                                        "Cannot load %s." % fileName)
                return

            # redbrush = QBrush(Qt.red)
            # blackpen = QPen(Qt.black)
            # blackpen.setWidth(5)
            self.image = image.scaled(self.graphicsView.size(), Qt.IgnoreAspectRatio)
            mat_img = cv2.resize(mat_img, (256, 256), interpolation=cv2.INTER_CUBIC)
            self.mat_img = Image.fromarray(cv2.cvtColor(mat_img, cv2.COLOR_BGR2RGB))

            # arrange
            mat_img_2 = cv2.resize(mat_img_2, (256, 256), interpolation=cv2.INTER_CUBIC)
            mat_img_2 = mat_img_2 / 127.5 - 1
            self.mat_img_2 = np.expand_dims(mat_img_2, axis=0)

            self.scene.reset()
            if len(self.scene.items()) > 0:
                self.scene.reset_items()
            self.scene.addPixmap(self.image)
            if len(self.result_scene.items()) > 0:
                self.result_scene.removeItem(self.result_scene.items()[-1])
            self.result_scene.addPixmap(self.image)

    def clip(self, tens):

        t = tens.clone()
        #         t=(t+1)/2.0
        t[t > 0.5] = 1
        t[t < 0.5] = 0

        return t

    def mask_mode(self):
        self.mode_select(0)

    def sketch_mode(self):
        self.mode_select(1)

    def stroke_mode(self):
        if not self.color:
            self.color_change_mode()
        self.scene.get_stk_color(self.color)
        self.mode_select(2)

    def color_change_mode(self):
        self.dlg.exec_()
        self.color = self.dlg.currentColor().name()
        print('Color:', self.color)
        self.pushButton_4.setStyleSheet("background-color: %s;" % self.color)
        self.scene.get_stk_color(self.color)

    def complete(self):

        sketch = self.make_sketch(self.scene.sketch_points)
        stroke, stroke_down = self.make_stroke(self.scene.stroke_points)
        mask = self.make_mask(self.scene.mask_points)
        stroke_down = np.concatenate([stroke_down[:, :, 2:3], stroke_down[:, :, 1:2], stroke_down[:, :, :1]], axis=2)
        mask = Image.fromarray(mask.astype('uint8')).convert('RGB')
        sketch = Image.fromarray(sketch.astype('uint8')).convert('RGB')
        stroke = Image.fromarray(stroke.astype('uint8')).convert('RGB')

        sketch = self.trans_to_pytorch_img(sketch)
        stroke = self.trans_to_pytorch_img(stroke)
        mask = self.trans_to_pytorch_mask(mask)
        if not type(self.ld_mask) == type(None):
            ld_mask = np.expand_dims(self.ld_mask[:, :, 0:1], axis=0)
            ld_mask[ld_mask > 0] = 1
            ld_mask[ld_mask < 1] = 0
            mask = mask + ld_mask
            mask[mask > 0] = 1
            mask[mask < 1] = 0
            mask = np.asarray(mask, dtype=np.uint8)

        if not type(self.ld_sk) == type(None):
            sketch = sketch + self.ld_sk
            sketch[sketch > 0] = 1

        input_image = self.trans_to_pytorch_img(self.mat_img)

        input_image = torch.unsqueeze(input_image, 0)
        sketch = torch.unsqueeze(sketch, 0)
        color = torch.unsqueeze(stroke, 0)
        mask_co = torch.unsqueeze(mask.clone(), 0)
        in_sk = sketch

        mask = mask.cuda()
        mask = mask[0]
        mask = torch.unsqueeze(mask, 0)
        mask = torch.unsqueeze(mask, 1)
        mask = mask.byte()
        start_t = time.time()
        with torch.no_grad():
            self.model.set_input(input_image, sketch, color, mask, mask_co)
            self.model.forward()
        end_t = time.time()
        print('inference time : {}'.format(end_t - start_t))

        output_dict = self.model.get_current_visuals()
        pic = (torch.cat([output_dict["input_image"], output_dict["fake"], output_dict["sketch"], output_dict["color"]],
                         dim=0) + 1) / 2.0
        torchvision.utils.save_image(pic, self.demo_save_dir + '/output.png', nrow=2)
        torchvision.utils.save_image((output_dict["sketch"] + 1) / 2.0, self.demo_save_dir +'/sketch.png')
        torchvision.utils.save_image((output_dict["color"] + 1) / 2.0, self.demo_save_dir + '/color.png')

        result = ((output_dict["fake"].cpu() + 1) / 2.0) * 255
        result = np.asarray(result[0, :, :, :], dtype=np.uint8)
        result = result.transpose(1, 2, 0)
        self.output_img = result.copy()
        result = np.concatenate([result[:, :, :1], result[:, :, 1:2], result[:, :, 2:3]], axis=2)
        qim = QImage(result.data, result.shape[1], result.shape[0], result.strides[0], QImage.Format_RGB888)
        self.result_scene.removeItem(self.result_scene.items()[-1])
        self.result_scene.addPixmap(QPixmap.fromImage(qim))

    def make_mask(self, pts):
        if len(pts) > 0:
            mask = np.zeros((256, 256, 3))
            for pt in pts:
                cv2.line(mask, pt['prev'], pt['curr'], (255, 255, 255), 12)
        else:
            mask = np.zeros((256, 256, 3))
        return mask

    def make_sketch(self, pts):
        if len(pts) > 0:
            sketch = np.ones((256, 256, 3)) * 255
            for pt in pts:
                cv2.line(sketch, pt['prev'], pt['curr'], (0, 0, 0), 2)
        else:
            sketch = np.ones((256, 256, 3)) * 255
        return sketch

    def make_stroke(self, pts):
        if len(pts) > 0:
            stroke = np.ones((256, 256, 3)) * 127.5
            stroke_down = np.ones((256, 256, 3)) * 255.0
            for pt in pts:
                c = pt['color'].lstrip('#')
                color = tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))
                color = (color[0], color[1], color[2])
                cv2.line(stroke, pt['prev'], pt['curr'], color, 4)
                cv2.line(stroke_down, pt['prev'], pt['curr'], color, 4)
        else:
            stroke = np.ones((256, 256, 3)) * 127.5
            stroke_down = np.ones((256, 256, 3)) * 255.0
        return stroke, stroke_down

    def arrange(self):
        image = np.asarray((self.mat_img_2[0] + 1) * 127.5, dtype=np.uint8)
        if len(self.scene.mask_points) > 0:
            for pt in self.scene.mask_points:
                cv2.line(image, pt['prev'], pt['curr'], (255, 255, 255), 12)
        if len(self.scene.stroke_points) > 0:
            for pt in self.scene.stroke_points:
                c = pt['color'].lstrip('#')
                color = tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))
                color = (color[2], color[1], color[0])
                cv2.line(image, pt['prev'], pt['curr'], color, 4)
        if len(self.scene.sketch_points) > 0:
            for pt in self.scene.sketch_points:
                cv2.line(image, pt['prev'], pt['curr'], (0, 0, 0), 2)
        # cv2.imwrite('tmp.png', image)
        image = QPixmap('tmp.png')
        self.scene.history.append(3)
        self.scene.addPixmap(image)

    def save_img(self):
        if type(self.output_img):
            fileName, _ = QFileDialog.getSaveFileName(self, "Save File",
                                                      QDir.currentPath())
            self.output_img = np.concatenate(
                [self.output_img[:, :, 2:3], self.output_img[:, :, 1:2], self.output_img[:, :, :1]], axis=2)
            cv2.imwrite(fileName + '.png', self.output_img)

    def undo(self):
        self.scene.undo()

    def clear(self):
        self.scene.reset_items()
        self.scene.reset()
        if type(self.image):
            self.scene.addPixmap(self.image)

    def nature(self):
        self.model.opt.name = 'nature'
        self.model.loadnature(62)
        print('nature model ready')

    def face(self):
        self.model.opt.name = 'face'
        self.model.load_networks(40)
        print('face model ready')


if __name__ == '__main__':
    opt = TestConfig().create_config()
    model = create_model(opt)
    app = QApplication(sys.argv)
    ex = Ex(model)
    sys.exit(app.exec_())
