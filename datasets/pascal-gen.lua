--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Script to compute list of ImageNet filenames and classes
--
--  This generates a file gen/imagenet.t7 which contains the list of all
--  ImageNet training and validation images and their classes. This script also
--  works for other datasets arragned with the same layout.
--

local sys = require 'sys'
local ffi = require 'ffi'
local py = require('fb.python')

--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Script to compute list of ImageNet filenames and classes
--
--  This generates a file gen/imagenet.t7 which contains the list of all
--  ImageNet training and validation images and their classes. This script also
--  works for other datasets arragned with the same layout.
--

local sys = require 'sys'
local ffi = require 'ffi'
local M = {}
--opt.data = '/opt/caffe/data/pascal/VOC2012/JPEGImages/'
local function findClasses()
   py = require ('fb.python')
   py.exec([=[import pickle
import numpy as np
with open('datasets/PascalLabel_train.pickle', 'r') as f1:
    trainMultilabels = pickle.load(f1)
with open('datasets/PascalLabel_val.pickle', 'r') as f2:
    valMultilabels = pickle.load(f2)]=])
   trainMultilabels = py.eval('trainMultilabels')
   valMultilabels = py.eval('valMultilabels')
   return trainMultilabels, valMultilabels
end

local function findImages(labeldict, opt)
   local imagePath = torch.CharTensor()
   local imageClass = torch.LongTensor()

   local maxLength = -1
   local imagePaths = {}
   
   local nInstance = 0

   local nClass = opt.nClasses
   for index, labels in pairs(labeldict) do
      --assert(opt.nClasses==(#labels)[1])
      nInstance = nInstance + 1
   end
   local imageClass = torch.LongTensor(nInstance, opt.nClasses)

   local id = 1
   for index, labels in pairs(labeldict) do
      path = index .. '.jpg'
      maxLength = math.max(maxLength, #path + 1)
      table.insert(imagePaths, path)
      imageClass[id] = labels:long()
      id = id + 1
   end

   local nImages = #imagePaths
   local imagePath = torch.CharTensor(nImages, maxLength):zero()
   for i, path in ipairs(imagePaths) do
      ffi.copy(imagePath[i]:data(), path)
   end

   return imagePath, imageClass
end

function M.exec(opt, cacheFile)
   -- find the image path names
   local imagePath = torch.CharTensor()  -- path to each image in dataset
   local imageClass = torch.LongTensor() -- class index of each image (class index in self.classes)

   --local trainDir = paths.concat(opt.data,'JPEGImages')
   --local valDir = paths.concat(opt.data,'JPEGImages')
   local classList = {'__background__', 'aeroplane', 'bicycle', 'bird', 'boat',
   'bottle', 'bus', 'car', 'cat', 'chair',
   'cow', 'diningtable', 'dog', 'horse',
   'motorbike', 'person', 'pottedplant',
   'sheep', 'sofa', 'train', 'tvmonitor'}
   --assert(paths.dirp(trainDir), 'train directory not found: ' .. trainDir)
   --assert(paths.dirp(valDir), 'val directory not found: ' .. valDir)

   print("=> Generating list of images from pickle")
   local trainMultilabels, valMultilabels = findClasses()

   print(" | finding all validation images")
   local valImagePath, valImageClass = findImages(valMultilabels, opt)

   print(" | finding all training images")
   local trainImagePath, trainImageClass = findImages(trainMultilabels, opt)

   local info = {
      basedir = opt.data,
      classList = classList,
      train = {
         imagePath = trainImagePath,
         imageClass = trainImageClass,
      },
      val = {
         imagePath = valImagePath,
         imageClass = valImageClass,
      },
   }

   print(" | saving list of images to " .. cacheFile)
   torch.save(cacheFile, info)
   return info
end

return M



