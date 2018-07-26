mkdir images -p && mkdir results -p;
rm images/content1.png -rf;
rm images/style1.png -rf;
rm results/demo_result_example1.png
cd images;
axel -n 1 http://freebigpictures.com/wp-content/uploads/shady-forest.jpg --output=content1.png;
axel -n 1 https://vignette.wikia.nocookie.net/strangerthings8338/images/e/e0/Wiki-background.jpeg/revision/latest?cb=20170522192233 --output=style1.png;
convert -resize 25% content1.png content1.png;
convert -resize 50% style1.png style1.png;
cd ..;
python demo.py;
