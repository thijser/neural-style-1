require ("imageSelectorroul.lua")


local cmd = torch.CmdLine()
cmd:option('-gpu', 0, 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')
cmd:option('-backend', 'nn', 'nn|cudnn|clnn')
cmd:option('-cudnn_autotune', false)
cmd:option('-image_size', 64, 'Maximum height / width of generated image')


cmd:option('-target_image', 'out/prepro.png')
cmd:option('-avaible_images', 'in.jpg,tankbw.jpg,hawaii.jpg,aeaecb2791801e2bfeb37f281599a885.jpg,tt.png,tt2.jpg')
cmd:option('-image_count',5)
cmd:option('-proto_file', 'models/VGG_ILSVRC_19_layers-deploy.prototxt')
cmd:option('-model_file', 'models/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-selectorGenerations', 200999)
cmd:option('-timestop',600)
cmd:option('-selectorWidth', 4)
cmd:option('-selectorDepth', 5)
cmd:option('-neural_Content_eval_Layer' ,'conv5_4') 
cmd:option('-colweight' , 90000)
cmd:option('mode','roulmate') --roulmate,topmate,top,roul,rand
cmd:option('mutatechance',1)




local params = cmd:parse(arg)
t=neuralEval(params,{"/home/thijser/neural-style-1/t/Pictures/banana/ActiOn_1.jpg","/home/thijser/neural-style-1/t/Pictures/banana/ActiOn_2.jpg"})
print(t)
