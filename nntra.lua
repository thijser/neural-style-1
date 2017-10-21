require 'torch'
require 'nn'
require 'image'
require 'optim'
require 'neural_style.lua'
require 'loadcaffe'


local img = image.load('out_prepost_99.png', 3)
local loc = image.load('inbw.jpg', 3)


function TransferBlack(img,orig)

--
--	imgt=img
    img=image.scale(img,orig:size()[3],orig:size()[2],'bilinear')
	imgo=orig:double()
	imgt=image.rgb2lab(img)
	imgo=image.rgb2lab(imgo)
	imgt[1]=imgo[1]

	img=image.lab2rgb(imgt)
	
return img
end

local store= TransferBlack(img,loc)
image.save('tra22.png',store)

