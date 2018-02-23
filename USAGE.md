### License
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).


### Install requirements

- OS: Ubuntu 16.04
- CUDA: 9.1
- Python: **Python 2 from Anaconda2**
- Python Library Dependency
  1. `conda install pytorch torchvision cuda90 -y -c pytorch`
  2. `conda install -y -c anaconda pip`
  3. `pip install scikit-umfpack`
  4. `pip install -U setuptools`
  5. `pip install cupy`
  6. `pip install pynvrtc`

### Download pretrained networks

- Download pretrained networks via the following [link](https://drive.google.com/open?id=1ENgQm9TgabE1R99zhNf5q6meBvX6WFuq).
- Unzip and store the model files under `models`.

### Example 1: Transfer the style of a style photo to a content photo.
- `mkdir images && mkdir results`
- Go to the image folder: `cd images`
- Download content image 1: `axel -n 1 http://freebigpictures.com/wp-content/uploads/shady-forest.jpg --output=content1.png`
- Download style image 1: `axel -n 1 https://vignette.wikia.nocookie.net/strangerthings8338/images/e/e0/Wiki-background.jpeg/revision/latest?cb=20170522192233 --output=style1.png`
- These images are huge. We need to resize them first. Run
  - `convert -resize 25% content1.png content1.png`
  - `convert -resize 50% style1.png style1.png`
- Go to the root folder: `cd ..`

- Test the photorealistic image stylization code `python demo.py`


### Example 2: Transfer the style of a style photo to a content photo with semantic label maps.

By default, our algorithm performs the global stylization. In order to give users control to decide the content–style correspondences for better stylization effects, we also support the spatial control through manully drawing label maps. 

#### Prepare label maps

- Install the tool [labelme](https://github.com/wkentaro/labelme) and run the following command to start it: `labelme`

- Start labeling regions (drawing polygons) in the content and style image. The corresponding regions (e.g., sky-to-sky) should have the same label.

- The labeling result is saved in a ".json" file. By running the following command, you will get the `label.png` under `path/example_json`, which is the label map used in our code. `label.png` is a 1-channel image (usually looks totally black) consists of consecutive labels starting from 0.

```
labelme_json_to_dataset example.json -o path/example_json
```  

#### Stylize with label maps

Now, we have four inputs and set their paths in `demo.py`:
```
python demo.py \
   --content_image_path PATH-TO-YOUR-CONTENT-IMAGE \ 
   --content_seg_path PATH-TO-YOUR-CONTENT-LABEL \ 
   --style_image_path PATH-TO-YOUR-STYLE-IMAGE \ 
   --style_seg_path PATH-TO-YOUR-STYLE-LABEL \ 
   --output_image_path PATH-TO-YOUR-OUTPUT
```

Below is a 3-label transferring example (images and labels are from the [DPST](https://github.com/luanfujun/deep-photo-styletransfer) work by Luan et al.):

![](transfer_with_label.png)

### Docker image

We also provide a docker image for testing the code. 

  1. Build the docker image `docker build -t your-docker-image:v1.0 .`
  2. Run an interactive session `docker run -v YOUR_PATH:YOUR_PATH --runtime=nvidia -i -t your-docker-image:v1.0 /bin/bash`
  3. `cd YOUR_PATH`
  4. `./demo.sh`

## Acknowledgement

- We express gratitudes to the great work [DPST](https://www.cs.cornell.edu/~fujun/files/style-cvpr17/style-cvpr17.pdf) by Luan et al. and their [Torch](https://github.com/luanfujun/deep-photo-styletransfer) and [Tensorflow](https://github.com/LouieYang/deep-photo-styletransfer-tf) implementations.
