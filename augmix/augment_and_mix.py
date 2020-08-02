# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Reference implementation of AugMix's data augmentation method in numpy."""
import augmentations
import numpy as np
from PIL import Image
from sys import argv as args
import cv2

# CIFAR-10 constants
MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2023, 0.1994, 0.2010]


def normalize(image):
  """Normalize input image channel-wise to zero mean and unit variance."""
  image = image.transpose(2, 0, 1)  # Switch to channel-first
  mean, std = np.array(MEAN), np.array(STD)
  image = (image - mean[:, None, None]) / std[:, None, None]
  return image.transpose(1, 2, 0)


def apply_op(image, op, severity):
  image = np.clip(image * 255., 0, 255).astype(np.uint8)
  pil_img = Image.fromarray(image)  # Convert to PIL.Image
  pil_img = op(pil_img, severity)
  return np.asarray(pil_img) / 255.


def augment_and_mix(image, severity=3, width=3, depth=-1, alpha=1.):
  """Perform AugMix augmentations and compute mixture.

  Args:
    image: Raw input image as float32 np.ndarray of shape (h, w, c)
    severity: Severity of underlying augmentation operators (between 1 to 10).
    width: Width of augmentation chain
    depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
      from [1, 3]
    alpha: Probability coefficient for Beta and Dirichlet distributions.

  Returns:
    mixed: Augmented and mixed image.
  """
  ws = np.float32(
      np.random.dirichlet([alpha] * width))
  m = np.float32(np.random.beta(alpha, alpha))
  imglst = []
  size = 0
  mix = np.zeros_like(image)
  imglst.append(mix)
  print(size, 'start')
  size += 1
  for i in range(width):
    image_aug = image.copy()
    imglst.append(image_aug)
    print(size, 'copy')
    size += 1
    depth = depth if depth > 0 else np.random.randint(1, 4)
    for _ in range(3):
      op = np.random.choice(augmentations.augmentations)
      image_aug = apply_op(image_aug, op, severity)
      imglst.append(image_aug)
      print(size, 'augment')
      size += 1
    # Preprocessing commutes since all coefficients are convex
    A = (ws[i] * normalize(image_aug))
    print(mix.shape, A.shape)
    print('image shape: ', image.shape)
    np.add(mix, A, out=mix, casting='unsafe')
    imglst.append(mix)
    print(size, 'mix')
    size += 1

  mixed = (1 - m) * normalize(image) + m * mix
  return mixed, imglst

if __name__ == '__main__':
  npimage = cv2.imread(args[1], 1)
  # image = Image.new("RGB", image.size, (255, 255, 255))
  # npimage = np.asarray(image)[...,:3]
  # print(type(image))
  # print('hihi')
  image_out, imglist = augment_and_mix(npimage)
  print(len(imglist))
  for imagg in imglist:
    cv2.imshow('augmix img', imagg)#np.hstack((image_out, npimage)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  cv2.imshow('augmix img', image_out)#np.hstack((image_out, npimage)))
  cv2.waitKey(0)
  cv2.destroyAllWindows()
