--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Multi-threaded data loader
--

local datasets = require 'datasets/init'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local M = {}
local DataLoader = torch.class('resnet.DataLoader', M)

function DataLoader.create(opt)
   -- The train and val loader
   local loaders = {}

   for i, split in ipairs{'train', 'val'} do
      local dataset = datasets.create(opt, split)
      loaders[i] = M.DataLoader(dataset, opt, split)
   end

   return table.unpack(loaders)
end

function DataLoader:__init(dataset, opt, split)
   local function init()
      require('datasets/' .. opt.dataset)
   end
   local function main(idx)
      _G.dataset = dataset
      _G.preprocess = dataset:preprocess()
      return dataset:size()
   end

   local threads, sizes = Threads(opt.nThreads, init, main)
   self.nCrops = (split == 'val' and opt.tenCrop) and 10 or 1
   self.threads = threads
   self.__size = sizes[1][1]
   self.batchSize = math.floor(opt.batchSize / self.nCrops)
   --<<<<<<< HEAD
   --added
   self.nClasses = opt.nClasses
   self.dataset = opt.dataset
   --=======
   -->>>>>>> e8fb31378fd8dc188836cf1a7c62b609eb4fd50a
end

function DataLoader:size()
   return math.ceil(self.__size / self.batchSize)
end

function DataLoader:run()
   local threads = self.threads
   local size, batchSize = self.__size, self.batchSize
   local perm = torch.randperm(size)

   local idx, sample = 1, nil
   ---added
   local nClasses = self.nClasses
   local dataset = self.dataset

   local function enqueue()
      while idx <= size and threads:acceptsjob() do
         local indices = perm:narrow(1, idx, math.min(batchSize, size - idx + 1))
         threads:addjob(
            function(indices, nCrops)
               local sz = indices:size(1) 
               local batch, imageSize
               local targe = nil
               if dataset == 'pascal' then
                  target = torch.IntTensor(sz, nClasses)
               else
                  target = torch.IntTensor(sz)
               end
               for i, idx in ipairs(indices:totable()) do
                  local sample = _G.dataset:get(idx)
                  local input = _G.preprocess(sample.input)
                  if not batch then
                     imageSize = input:size():totable()
                     if nCrops > 1 then table.remove(imageSize, 1) end
                     batch = torch.FloatTensor(sz, nCrops, table.unpack(imageSize))
                  end
                  batch[i]:copy(input)
                  
                  target[i] = sample.target
               end
               collectgarbage()
               return {
                  input = batch:view(sz * nCrops, table.unpack(imageSize)),
                  target = target,
               }
            end,
            function(_sample_)
               sample = _sample_
            end,
            indices,
            self.nCrops
         )
         idx = idx + batchSize
      end
   end

   local n = 0
   local function loop()
      enqueue()
      if not threads:hasjob() then
         return nil
      end
      threads:dojob()
      if threads:haserror() then
         threads:synchronize()
      end
      enqueue()
      n = n + 1
      return n, sample
   end

   return loop
end

return M.DataLoader
