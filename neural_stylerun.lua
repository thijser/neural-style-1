require("neural_style.lua")
require 'optim'
require 'torch'

local cmd = torch.CmdLine()
cmd:option('-style_image', 'tt.png,tankcol.jpg',
           'Style target image')
cmd:option('-style_blend_weights', 'nil')
cmd:option('-content_image', 'LShH540.jpg',
           'Content target image')
cmd:option('-image_size', 768, 'Maximum height / width of generated image')
cmd:option('-gpu', 0, 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')

-- Optimization options
cmd:option('-content_weight', 5e0)

cmd:option('-style_weight', 4e2)
cmd:option('-tv_weight', 1e-3)
cmd:option('-num_iterations', 800)
cmd:option('-normalize_gradients', false)
cmd:option('-init', 'image', 'random|image')
cmd:option('-optimizer', 'lbfgs', 'lbfgs|adam')
cmd:option('-learning_rate', 1e1)

-- Output options
cmd:option('-print_iter', 50)
cmd:option('-save_iter', 50)
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


local params = cmd:parse(arg)
main(params)
