# operetta_tool - python library

##### New version 1.1.3 allow more complex processing of images as data input for AI algorithms

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
pip install Operetta-tool
```



## Usage


#### 1. Import required libraries

```
from operetta import operetta_annotation
import cv2
```


#### 2. Load images metadata from xml file

```
xml_file = operetta_annotation.xml_load(path_to_opera_xml)
```
* path_to_opera_xml - path to Index data of Opera in xml format (usual 'Images/Index.idx.xml')


#### 3. Detection of outlier images not included in the square of the main photo

```
xml_file, figure = operetta_annotation.detect_outlires(xml_file, list_of_out = [])
```

##### This function displays the arrangement of individual images and their location on the map of the main photo

* xml_file - input = prevoiusly loaded xml_file / output = function return xml_file reduced images selected by user based on figure
* figure - the map of each image location on the main photo
* list_of_out - list of the indexes for outliers images selected by the user from figure

##### Example:

###### First run of function to display images' location map
```
xml_file, figure = operetta_annotation.detect_outlires(xml_file, list_of_out = [])

```

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/main/fig/before_outlires.bmp" alt="drawing" width="600" />
</p>


###### Second run of function to reduce outliers number of images and display images' location map
```
xml_file, figure = operetta_annotation.detect_outlires(xml_file, list_of_out = [0])

```

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/main/fig/after_outlires.bmp" alt="drawing" width="600" />
</p>



#### 4. Adaptation of the Opera images coordinates to the coordinates in the overview image

```
imgs, img_length, img_width = operetta_annotation.image_sequences(opera_coordinates)
```

* imgs - a set of images in the correct order for subsequent analyses
* img_length - number of pictures for images included in y-axis 
* img_width - number of pictures for images included in x-axis 


#### 5. Composition of the raw images on the x,  y, z axis into the main photo in *.tiff format including all z-stack slices

```   
operetta_annotation.image_concatenate(path_to_images, imgs, img_length, img_width, scale_factor, chanels, n_thread) 
```

* chanels - list of channels included in composition analysis, eg.['ch2', 'ch3']
* scale_factor - the factor for scaling pictures with different basal brightness intensities. Default 50
* path_to_images - path to directory including raw Opera images ['Images' directory]
* n_thread - the number of processor threads involved in the analysis adapted to the device on which the analysis will be performed. The more threads, the faster the analysis
* imgs - a set of images in the correct order [from 'image_sequences' function]
* img_length - number of pictures for images included in y-axis [from 'image_sequences' function]
* img_width - number of pictures for images included in x-axis [from 'image_sequences' function]


##### The results for each channel are saved in separate *.tiff files



#### 6. Image projection in the z-axis (z-projection) of completed images stacks in *.tiff format 

```   
projection = operetta_annotation.z_projection(path_to_tiff, color)
```

* path_to_tiff - path to the completed image of the z-axis stack in *.tiff format
* color - color scale of image projection [red, green, blue, magenta or grey]

##### Example:

###### First projection: chanel 2 - Alexa488
```
projection1 = operetta_annotation.z_projection('chanel_ch2.tiff', 'green')
cv2.imwrite('projection_chanel_ch2.png', projection1)
```

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/main/fig/projection1.bmp" alt="drawing" width="600" />
</p>


###### Second projection: chanel 3 - Alexa647
```
projection2 = operetta_annotation.z_projection('chanel_ch3.tiff', 'red')
cv2.imwrite('projection_chanel_ch3.png', projection2)
```

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/main/fig/projection2.bmp" alt="drawing" width="600" />
</p>



#### 7. Merge images projections in one complex picture

```   
merged_projection = operetta_annotation.merge_images(image_list, intensity_factors)
```

* mage_list - list of projections to merge, eg. [projection1, projection2]
* intensity_factors - list of important factors to each projection (which channel results should be more or less visible). Recommended is 1 to 1. The number of factors depends on the number of image projections to be merged and is set for each separately, eg. [1,1]

##### Example:

###### Result of merged channels 

```
image_list = [projection2, projection1]
intensity_factors = [1,1]

merged_projection = operetta_annotation.merge_images(image_list, intensity_factors)
cv2.imwrite('merged_projection.png', merged_projection)
```

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/main/fig/merged.bmp" alt="drawing" width="600" />
</p>



#### 8. Display of composite photo from Opera (z-projection) with grid in places of single photos and indexes

```
operetta_annotation.image_grid(path_to_opera_projection, img_length, img_width)
```

* path_to_opera_projection - path to image (z-projection) 
* imgs - a set of images in the correct order [from 'image_sequences' function]
* img_length - number of pictures for images included in y-axis [from 'image_sequences' function]
* img_width - number of pictures for images included in x-axis [from 'image_sequences' function]

##### Examples:

##### Example of above image on lens x40

<p align="center">
<img  src="https://github.com/jkubis96/Operetta_tool/blob/main/fig/select3.bmp?raw=true" alt="drawing" width="600" />
</p>


##### Other lens examples:

##### Image on lens x20

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/main/fig/select2.bmp" alt="drawing" width="600" />
</p>

##### Image on lens x63

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/main/fig/select1.bmp" alt="drawing" width="600" />
</p>




#### 9. Split images from different chanels

```
operetta_annotation.split_chanels(path_to_images, path_to_save)
```

* path_to_images - path to directory including raw Opera images ['Images' directory]
* path_to_save - path to directory for splited channels save


###### Figure 3 Directories with images from different channels

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/main/fig/ff.bmp" alt="drawing" width="600" />
</p>




#### 10. Separation of selected stacks of images by indexes into separate directories

```
operetta_annotation.select_pictures(image_dictinary, path_to_images, path_to_save, numbers_of_pictures)
```


* path_to_images - path to directory including raw Opera images ('Images') or separate channels directory, eg. ('ch1','ch2')
* path_to_save - path to the directory for saving chosen by indexes images stacks
* numbers_of_pictures - list of choosen pictures [1,2,3,10,11,21,...]


##### Directories with separated chosen stacks of raw Opera images

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/main/fig/se.bmp" alt="drawing" width="600" />
</p>

