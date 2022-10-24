# operetta_tool - python library

#### The operetta_tool is a python library created for handling and annotation raw images from the Opera Phenix platform used for ML / AI applications

<p align="right">
<img  src="https://github.com/jkubis96/Operetta_tool/blob/main/fig/logo_jbs.PNG?raw=true" alt="drawing" width="250" />
</p>


### Author: Jakub Kubiś 

<div align="left">
 Institute of Bioorganic Chemistry<br />
 Polish Academy of Sciences<br />
 Department of Molecular Neurobiology<br />
</div>


## Description


<div align="justify"> The Operetta_tool is a python library created for handling and annotation images from the Opera Phenix platform used for ML / AI applications. </div>

</br>

## Installation

#### In command line write:

```
pip install Operetta_tool
```

## Opera data prepare as input for the operetta_tool



#### 1. Conduct raw Opera images composition into *.tiff format via BIOP plugin for Opera in ImageJ

* BIOP [https://github.com/BIOP/ijp-operetta-importer?fbclid=IwAR1L6uXqVh9crz1jJ7gdqxPd4o2jfQ3VkLVzk9uokuSlSo1MKqdVPudHyK4]


<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/main/fig/biop.bmp" alt="drawing" width="600" />
</p>

###### Figure 1 Image compositing using the BIOP plugin in ImageJ


#### 2. Conduct z-projection of the *.tiff format image via Z-PROJECTION in ImageJ


<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/main/fig/projection.bmp" alt="drawing" width="600" />
</p>

###### Figure 2 Image z-projection using the Z-PROJECTION in ImageJ



## Usage



#### 1. Import library

```
import Operetta_tool
```
    
    

#### 2. Split images from different chanels

```
operetta.operetta_annotation.split_chanels(path_to_images, path_to_save)
```

* path_to_images - path to directory including raw Opera images ['Images' directory]
* path_to_save - path to directory for splited channels save

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/main/fig/ff.bmp" alt="drawing" width="600" />
</p>

###### Figure 1 Directories with images from different channels


#### 3. Obtaining images coordinates

```
opera_coordinates = operetta.operetta_annotation.xml_load(path_to_opera_xml)
```
* path_to_opera_xml - path to Index data of Opera in xml format



#### 4. Adaptation of the Opera images coordinates to the coordinates in the overview image

```
image_dictinary, img_length, img_width = operetta.operetta_annotation.image_sequences(opera_coordinates)
```
* opera coordinates - data frame of indexes from Opera XML log



#### 5. Display of composite photo from Opera (z-projection) with grid in places of single photos and indexes

```
operetta.operetta_annotation.image_grid(path_to_opera_projection, img_length, img_width, resize_factor)
```

* path_to_opera_projection - path to image (z-projection) 
* img_length - number of pictures for images included in y-axis [from 'image_sequences' function]
* img_width - number of pictures for images included in x-axis [from 'image_sequences' function]
* resize_factor - factor of picture resize [modified when the pictures with default value 100 are too small or too big, depending on the magnification of the lens (x20, x40, x60, ...)]

##### Image from x20

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/main/fig/select2.bmp" alt="drawing" width="600" />
</p>

##### Image from x63

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/main/fig/select1.bmp" alt="drawing" width="600" />
</p>

##### Figures 2 & 3 Images composition from Opera (z-projection) with grid and indexes 


#### 7. Separation of selected stacks of images by indexes into separate directories

```
operetta.operetta_annotation.select_pictures(image_dictinary, path_to_images, path_to_save, numbers_of_pictures)
```


* path_to_images - path to directory including raw Opera images ['Images' or separate channels directory]
* path_to_save - path to the directory for saving chosen by indexes images stacks
* numbers_of_pictures - list of choosen pictures [1,2,3,10,11,21,...]



<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/main/fig/se.bmp" alt="drawing" width="600" />
</p>

##### Figure 4 Directories with separated chosen stacks of raw Opera images