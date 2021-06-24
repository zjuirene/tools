#批量裁剪至256*256

import torch.utils.data as data
from glob import glob
from PIL import Image
import torchvision.transforms as transforms
import argparse
import os
import imageio
import  numpy  as np
import pandas as pd
def get_all_path(open_file_path):
    rootdir = open_file_path
    path_list = []
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    for i in range(0, len(list)):
        com_path = os.path.join(rootdir, list[i])
        #print(com_path)
        if os.path.isfile(com_path):
            path_list.append(com_path)
        if os.path.isdir(com_path):
            path_list.extend(get_all_path(com_path))
    #print(path_list)
    return path_list


class DataSet(data.Dataset):
    def __init__(self, img_dir, resize):
        super(DataSet, self).__init__()
        # self.img_paths = glob('{:s}/*'.format(img_dir))
        self.img_paths = get_all_path(img_dir)
        self.transform = transforms.Compose([
        transforms.Resize(int(resize * 76 / 64)),
        transforms.RandomCrop(resize),
        transforms.RandomHorizontalFlip()])
        self.data_dir = img_dir

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, '/Users/liailin/Downloads/DM-GAN/data/birds/CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, '/Users/liailin/Downloads/DM-GAN/data/birds/CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox



    def __getitem__(self, item):

        img = Image.open(self.img_paths[item]).convert('RGB')
        if os.path.basename(self.data_dir)=='birdGT':
            width, height = img.size
            self.bbox = self.load_bbox()
            (filepath, tempfilename) = os.path.split(self.img_paths[item])
            (filepath1, tempfilename1) = os.path.split(filepath)
            key=tempfilename1+'/'+os.path.splitext(tempfilename)[0]
            bbox = self.bbox[key]
            r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)
            y1 = np.maximum(0, center_y - r)
            y2 = np.minimum(height, center_y + r)
            x1 = np.maximum(0, center_x - r)
            x2 = np.minimum(width, center_x + r)
            img = img.crop([x1, y1, x2, y2])

        img = self.transform(img)

        return img, self.img_paths[item]

    def __len__(self):
        return len(self.img_paths)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='flowerGT')
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--save_dir', type=str, default='flowerGT256')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    dataset = DataSet(args.img_dir, args.resize)
    print('dataset:', len(dataset))

    for i in range(len(dataset)):
        img, path = dataset[i]
        path = os.path.basename(path)
        print('Processing:', path)

        imageio.imwrite(args.save_dir+'/{:s}'.format(path), img)




