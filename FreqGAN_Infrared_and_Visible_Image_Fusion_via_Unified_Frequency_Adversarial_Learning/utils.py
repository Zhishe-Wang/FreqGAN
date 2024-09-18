import numpy as np
import os
from args import args
import torch
from os import listdir
from os.path import join
from imageio import imsave
from torchvision import transforms
import cv2
from imageio import imwrite


def load_dataset(ir_imgs_path, vi_imgs_path, BATCH_SIZE, num_imgs=None):
    if num_imgs is None:
        num_imgs = len(ir_imgs_path)
    ir_imgs_path = ir_imgs_path[:num_imgs]
    vi_imgs_path = vi_imgs_path[:num_imgs]
    # random
    mod = num_imgs % BATCH_SIZE
    print('BATCH SIZE %d.' % BATCH_SIZE)
    print('Train images number %d.' % num_imgs)
    print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        ir_imgs_path = ir_imgs_path[:-mod]
        vi_imgs_path = vi_imgs_path[:-mod]
    batches = int(len(ir_imgs_path) // BATCH_SIZE)
    return ir_imgs_path, vi_imgs_path, batches


def make_floor(path1, path2):
    path = os.path.join(path1, path2)
    if os.path.exists(path) is False:
        os.makedirs(path)
    return path


def get_train_images_auto(paths, height=args.hight, width=args.width, mode='RGB'):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, mode=mode)
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = np.reshape(image, [image.shape[2], image.shape[0], image.shape[1]])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    images = images / 255
    return images


def save_images(path, data, out):
    w, h = out.shape[0], out.shape[1]
    if data.shape[1] == 1:
        data = data.reshape([data.shape[2], data.shape[3]])
    ori = data[0:w, 0:h]
    imwrite(path, ori)


def get_image(path, height=256, width=256, mode='L'):
    global image
    if mode == 'L':
        image = cv2.imread(path, 0)
    elif mode == 'RGB':
        image = cv2.cvtColor(path, cv2.COLOR_BGR2RGB)

    if height is not None and width is not None:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return image


def get_test_images(paths, height=None, width=None, mode='L'):
    ImageToTensor = transforms.Compose([transforms.ToTensor()])
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, mode=mode)

        w, h = image.shape[0], image.shape[1]
        w_s = 128 - w % 128
        h_s = 128 - h % 128
        image = cv2.copyMakeBorder(image, 0, w_s, 0, h_s, cv2.BORDER_CONSTANT,
                                   value=128)
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = ImageToTensor(image).float().numpy() * 255
    images.append(image)
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    images = images / 255
    return images


def list_images(directory):
    images = []
    names = []
    dir = listdir(directory)
    dir.sort()
    for file in dir:
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
        name1 = name.split('.')
        names.append(name1[0])
    return images
