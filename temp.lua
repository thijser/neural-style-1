require 'torch'
require 'nn'
require 'image'
require 'optim'
require 'cutorch'
require 'cunn'

require 'loadcaffe'
local cmd = torch.CmdLine()
local function main(params)
  cutorch.setDevice(1)
  local loadcaffe_backend = 'nn'
  local cnn = loadcaffe.load('models/VGG_ILSVRC_19_layers-deploy.prototxt', 'models/VGG_ILSVRC_19_layers.caffemodel', loadcaffe_backend):float()
  cnn:cuda()
  targetImage_caffe = image.load('tank.jpg', 3)
  targetImage_caffe = targetImage_caffe:cuda() 

print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
print(targetImage_caffe)
print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
 netimage=cnn:forward(targetImage_caffe)


end

local params = cmd:parse(arg)
main(params)
