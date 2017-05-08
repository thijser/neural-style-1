echo "killing all other torch instances"
killall -9  /home/thijser/torch/install/bin/luajit

cd t 

if [ "$2" -eq "0" ]; then 
	echo "using pre-existing images"
else 
	python imsearch.py $1 $2
fi 
cd ..

c="th imageSelector.lua -avaible_images $(find t/Pictures -type f \( -iname \*.jpg -o -iname \*.png \) -printf '%p,' | sed 's/,$//')"
echo $c
eval $c
