cd t 
rm Pictures/* -r
python imsearch.py $1 $2
cd ..

g="th neural_style.lua -content_image in.jpg -style_image in.jpg"
c="th neural_style.lua -content_image in.jpg -style_image $(find t/Pictures -type f \( -iname \*.jpg -o -iname \*.png \) -printf '%p,' | sed 's/,$//')"
echo $c
eval $c
