from torch.utils.data import Dataset
from torchvision import transforms
import cv2
from PIL import Image
import random
import numpy as np
hyp = {
    'hsv_h': 0.015,
    'hsv_s': 0.7, 
    'hsv_v': 0.4, 
}
from transformers import AutoProcessor
# processor = AutoProcessor.from_pretrained('../weights/vit-base-patch16-224-in21k')

class MyDataSet(Dataset):
    def __init__(self, dataset_type, embeddings, labels):
        """
        dataset_type: ['train', 'dev', 'test']
        """

        # self.inputs = inputs
        # self.inputs["labels"] = labels[:,:60]
        self.sample_list = list()
        self.dataset_type = dataset_type
        for i,embedding in enumerate(embeddings):
            self.sample_list.append((embedding,labels[i]))

    def __getitem__(self, index):
        inputs_embeds,labels = self.sample_list[index]
        return  {"inputs_embeds":inputs_embeds, "labels":labels}
        # d = dict()
        # for key in self.inputs.keys():
        #     d[key] = self.inputs[key][index]
        # return d


    def __len__(self):
        return len(self.sample_list)
        # return len(self.inputs["input_ids"])

class MyDataSet1(Dataset):
    def __init__(self, dataset_type, text_inputs, image_inputs):
        """
        dataset_type: ['train', 'dev', 'test']
        """
        self.inputs = text_inputs
        self.inputs["pixel_values"] = image_inputs['pixel_values']

    def __getitem__(self, index):
        d = dict()
        for key in self.inputs.keys():
            d[key] = self.inputs[key][index]
        return d


    def __len__(self):
        return len(self.inputs["input_ids"])


class MyDataSet2(Dataset):
    def __init__(self, inputs, dataset_type='train', data='2015'):
        """
        dataset_type: ['train', 'dev', 'test']
        """
        
        self.inputs = inputs
        self.dataset_type = dataset_type
        self.images_path = []
        # with open("datasets/finetune/dualc/{}/input_{}.txt".format(data, self.dataset_type), "r", encoding="utf-8") as file:
        #     # 使用for循环逐行读取文件内容
        #     for line in file:
        #         # 去除每行末尾的换行符（如果有的话）
        #         line = line.rstrip()
        #         # 处理每行内容，这里只是简单地打印出来
        #         self.images_path.append(line)
        # print(self.dataset_type, " ", len(self.images_path), len(self.inputs['pixel_values']))
        # assert len(self.images_path) == len(self.inputs['pixel_values'])

    # def __getitem__(self, index):
    #     d = dict()
    #     for key in self.inputs.keys():
    #         if key=='pixel_values' and self.dataset_type in ['train'] and random.random()<0.2:
    #             tmp = cv2.imread(self.images_path[index])
    #             H, W, _ = tmp.shape
    #             min_size = min(H ,W)
    #             r = np.random.uniform(-1, 1, 3) * [hyp["hsv_h"], hyp["hsv_s"], hyp["hsv_v"]] + 1
    #             augment_hsv(np.array(tmp), r)
    #             img = Image.fromarray(np.uint8(tmp))
    #             transforms.RandomHorizontalFlip(0.5)(img)
    #             rate = random.random()
    #             if rate<0.3:
    #                 img = transforms.RandomRotation(degrees=15)(img)
    #             if rate<0.66:
    #                 img = transforms.CenterCrop((min_size, min_size))(img)
    #             else:
    #                 img = transforms.GaussianBlur(7, 3)(img)
    #             pixel_values = processor([img],return_tensors="pt")["pixel_values"]
    #             d[key] = pixel_values[0]

    #         else:
    #             d[key] = self.inputs[key][index]
    #     return d
    
    def __getitem__(self, index):
        d = dict()
        for key in self.inputs.keys():
            d[key] = self.inputs[key][index]
        return d


    def __len__(self):
        length = len(self.inputs["input_ids"])
        return length

def augment_hsv(im, r):
    """Applies HSV color-space augmentation to an image with random gains for hue, saturation, and value."""

    hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_RGB2HSV))
    dtype = im.dtype  # uint8

    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB, dst=im)  # no return needed