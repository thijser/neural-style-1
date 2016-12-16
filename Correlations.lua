require 'torch'
require 'nn'
require 'image'
require 'optim'

require 'loadcaffe'


local cmd = torch.CmdLine()

-- Basic options
cmd:option('-pre_image', 'examples/inputs/seated-nude.jpg',
           'Style target image')
cmd:option('-style_blend_weights', 'nil')
cmd:option('-post_image', 'examples/inputs/tubingen.jpg',
           'Content target image')
cmd:option('-image_size', 512, 'Maximum height / width of generated image')
cmd:option('-gpu', 0, 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')
cmd:option('-edit_image', 'examples/inputs/seated-nude.jpg',
           'Style target image')
-- Optimization options
cmd:option('-content_weight', 5e0)
cmd:option('-style_weight', 1e2)
cmd:option('-tv_weight', 1e-3)
cmd:option('-num_iterations', 1000)
cmd:option('-normalize_gradients', false)
cmd:option('-init', 'image', 'random|image')
cmd:option('-optimizer', 'lbfgs', 'lbfgs|adam')
cmd:option('-learning_rate', 1e1)

-- Output options
cmd:option('-print_iter', 50)
cmd:option('-save_iter', 100)
cmd:option('-output_image', 'out.png')

-- Other options
cmd:option('-style_scale', 1.0)
cmd:option('-original_colors', 0)
cmd:option('-pooling', 'max', 'max|avg')
cmd:option('-proto_file', 'models/VGG_ILSVRC_19_layers_deploy.prototxt')
cmd:option('-model_file', 'models/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-backend', 'nn', 'nn|cudnn|clnn')
cmd:option('-cudnn_autotune', false)
cmd:option('-seed', -1)

cmd:option('-content_layers', 'relu4_2', 'layers for content')
cmd:option('-style_layers', 'relu1_1,relu2_1,relu3_1,relu4_1,relu5_1', 'layers for style')

local function main(params)
  if params.gpu >= 0 then
    if params.backend ~= 'clnn' then
      require 'cutorch'
      require 'cunn'
      cutorch.setDevice(params.gpu + 1)
    else
      require 'clnn'
      require 'cltorch'
      cltorch.setDevice(params.gpu + 1)
    end
  else
    params.backend = 'nn'
  end

  if params.backend == 'cudnn' then
    require 'cudnn'
    if params.cudnn_autotune then
      cudnn.benchmark = true
    end
    cudnn.SpatialConvolution.accGradParameters = nn.SpatialConvolutionMM.accGradParameters -- ie: nop
  end
  
  local loadcaffe_backend = params.backend
  if params.backend == 'clnn' then loadcaffe_backend = 'nn' end
  local cnn = loadcaffe.load(params.proto_file, params.model_file, loadcaffe_backend):float()
  if params.gpu >= 0 then
    if params.backend ~= 'clnn' then
      cnn:cuda()
    else
      cnn:cl()
    end
  end
  print("loading pre image")
  --load pre image 
  local pre_image = image.load(params.pre_image, 3)
  pre_image = image.scale(pre_image, params.image_size, 'bilinear')
  local pre_image_caffe = preprocess(pre_image):float()
    print("loading post image")
  --load post image 
    local post_image = image.load(params.post_image, 3)
  post_image = image.scale(post_image, params.image_size, 'bilinear')
  local post_image_caffe = preprocess(post_image):float()
  
  print("selecting gpu")
  if params.gpu >= 0 then
    if params.backend ~= 'clnn' then
      pre_image_caffe = pre_image_caffe:cuda()
      post_image_caffe = post_image_caffe:cuda() 
    else
            pre_image_caffe = pre_image_caffe:cl()
            post_image_caffe = post_image_caffe:cl()
    end
   end
  

  -- Set up the network, inserting style and content loss modules
  local CorrectionLoss = {}
  print("setting up network && inserting loss modules")
  local next_content_idx, next_style_idx = 1, 1
  local net = nn.Sequential()
  if params.tv_weight > 0 then
    local tv_mod = nn.TVLoss(params.tv_weight):float()
    if params.gpu >= 0 then
      if params.backend ~= 'clnn' then
        tv_mod:cuda()
      else
        tv_mod:cl()
      end
    end
    net:add(tv_mod)
  end

  for i = 1, #cnn do
  	print("now setting up layer: ".. i)
        local layer = cnn:get(i)
      local name = layer.name
      local layer_type = torch.type(layer)
      local is_pooling = (layer_type == 'cudnn.SpatialMaxPooling' or layer_type == 'nn.SpatialMaxPooling')
      if is_pooling and params.pooling == 'avg' then
        assert(layer.padW == 0 and layer.padH == 0)
        local kW, kH = layer.kW, layer.kH
        local dW, dH = layer.dW, layer.dH
        local avg_pool_layer = nn.SpatialAveragePooling(kW, kH, dW, dH):float()
        if params.gpu >= 0 then
          if params.backend ~= 'clnn' then
            avg_pool_layer:cuda()
          else
            avg_pool_layer:cl()
          end
        end
        local msg = 'Replacing max pooling at layer %d with average pooling'
        print(string.format(msg, i))
        net:add(avg_pool_layer)
      else
        net:add(layer)
      end
      
      if name == pre_image[next_content_idx] then
        print("Setting up content layer", i, ":", layer.name)
        local target = net:forward(content_image_caffe):clone()
        local norm = params.normalize_gradients
        local loss_module = nn.StyleLoss(params.content_weight, target, norm):float()
        if params.gpu >= 0 then
          if params.backend ~= 'clnn' then
            loss_module:cuda()
          else
            loss_module:cl()
          end
        end
        net:add(loss_module)
        table.insert(content_losses, loss_module)
        next_content_idx = next_content_idx + 1
      end  
 
      if name == post_image[next_style_idx] then
       print("Setting up style layer  ", i, ":", layer.name)
       local gram = GramMatrix():float()
       if params.gpu >= 0 then
         if params.backend ~= 'clnn' then
           gram = gram:cuda()
         else
           gram = gram:cl()
         end
       end
       local target = nil
       for i = 1, #style_images_caffe do
         local target_features = net:forward(style_images_caffe[i]):clone()
         local target_i = gram:forward(target_features):clone()
         target_i:div(target_features:nElement())
         target_i:mul(style_blend_weights[i])
         if i == 1 then
           target = target_i
         else
           target:add(target_i)
         end
       end
       local norm = params.normalize_gradients
       local loss_module = nn.StyleLoss(params.style_weight, target, norm):float()
       if params.gpu >= 0 then
         if params.backend ~= 'clnn' then
           loss_module:cuda()
        else
           loss_module:cl()
         end
       end
       net:add(loss_module)
       table.insert(style_losses, loss_module)
        next_style_idx = next_style_idx + 1
      end
    end
  end
  

  -- We don't need the base CNN anymore, so clean it up to save memory.
--  cnn = nil
--  for i=1,#net.modules do
--    local module = net.modules[i]
--    if torch.type(module) == 'nn.SpatialConvolutionMM' then
--        -- remove these, not used, but uses gpu memory
--        module.gradWeight = nil
--        module.gradBias = nil
--    end--
--  end
--  collectgarbage()
end

--prepare image for caffe 
function preprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):mul(256.0)
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img:add(-1, mean_pixel)
  return img
end

local TVLoss, parent = torch.class('nn.TVLoss', 'nn.Module')

function TVLoss:__init(strength)
  parent.__init(self)
  self.strength = strength
  self.x_diff = torch.Tensor()
  self.y_diff = torch.Tensor()
end
function TVLoss:updateOutput(input)
  self.output = input
  return self.output
end

-- TV loss backward pass inspired by kaishengtai/neuralart
function TVLoss:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input):zero()
  local C, H, W = input:size(1), input:size(2), input:size(3)
  self.x_diff:resize(3, H - 1, W - 1)
  self.y_diff:resize(3, H - 1, W - 1)
  self.x_diff:copy(input[{{}, {1, -2}, {1, -2}}])
  self.x_diff:add(-1, input[{{}, {1, -2}, {2, -1}}])
  self.y_diff:copy(input[{{}, {1, -2}, {1, -2}}])
  self.y_diff:add(-1, input[{{}, {2, -1}, {1, -2}}])
  self.gradInput[{{}, {1, -2}, {1, -2}}]:add(self.x_diff):add(self.y_diff)
  self.gradInput[{{}, {1, -2}, {2, -1}}]:add(-1, self.x_diff)
  self.gradInput[{{}, {2, -1}, {1, -2}}]:add(-1, self.y_diff)
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end



local params = cmd:parse(arg)
main(params)
