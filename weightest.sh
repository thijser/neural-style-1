for i in {0..40}
do
	echo $i
sleep 1
killall -9  /home/thijser/torch/install/bin/luajit
killall -9 /home/thijser/torch-cl/install/bin/luajit
	exc="th neural_stylerun.lua"
	contentweight="-content_weight 5"
	styleweight="-style_weight $((2 ^ $i))"
	content="-content_image in/crown.jpg"
	out="-output_image out/"$i"out.png"
	style="-style_image t/Pictures/Mandarinfish/ActiOn_64.jpg,t/Pictures/Mandarinfish/ActiOn_34.jpg,t/Pictures/Mandarinfish/ActiOn_92.jpg,t/Pictures/Mandarinfish/ActiOn_94.jpg,t/Pictures/Mandarinfish/ActiOn_40.jpg,t/Pictures/Mandarinfish/ActiOn_44.jpg,t/Pictures/Mandarinfish/ActiOn_73.jpg,t/Pictures/Mandarinfish/ActiOn_65.jpg,t/Pictures/Mandarinfish/ActiOn_17.jpg,t/Pictures/Mandarinfish/ActiOn_29.jpg,t/Pictures/Mandarinfish/ActiOn_3.jpg	

"

 	exc="$exc $content $style $styleweight $contentweight $out" 
	echo $exc
	$exc > nul

	
done
