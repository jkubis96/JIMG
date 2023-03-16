# operetta_tool - python library

##### New version 1.2.5 allow easier and more complex processing of images as data input for AI algorithms

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

#### 4. Detection of lack of images in sequence included in main image square

```
xml_file, figure_in = operetta_annotation.repair_blanks(xml_file)
```

##### This function completes missing image elements and saves their coordinates (x and y axis) to the xml metadata file. 
##### In subsequent analysis, these points will be filled with a black square in the main photo

* xml_file - input = prevoiusly loaded xml_file / output = function return xml_file supplemented with missing elements


##### Example:

###### Before (one missing element)


<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/main/fig/opera1.bmp" alt="drawing" width="600" />
</p>


###### After (repaired)


<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/main/fig/opera2.bmp" alt="drawing" width="600" />
</p>


#### 5. Adaptation of the Opera images coordinates to the coordinates in the overview image

```
imgs, img_length, img_width = operetta_annotation.image_sequences(opera_coordinates)
```

* imgs - a set of images in the correct order for subsequent analyses
* img_length - number of pictures for images included in y-axis 
* img_width - number of pictures for images included in x-axis 


#### 6. Composition of the raw images on the x,  y, z axis into the main photo in *.tiff format including all z-stack slices

```   
operetta_annotation.mage_concatenate(path_to_images, imgs, img_length, img_width, overlap, chanels, n_thread) 
```

* chanels - list of channels included in composition analysis, eg.['ch2', 'ch3']
* overlap - the percentage of image overlap defined earlier when taking a picture on the microscope [eg. 0.05 or 0.1 (5% or 10%)]
* path_to_images - path to directory including raw Opera images ['Images' directory]
* n_thread - the number of processor threads involved in the analysis adapted to the device on which the analysis will be performed. The more threads, the faster the analysis
* imgs - a set of images in the correct order [from 'image_sequences' function]
* img_length - number of pictures for images included in y-axis [from 'image_sequences' function]
* img_width - number of pictures for images included in x-axis [from 'image_sequences' function]


##### The results for each channel are saved in separate *.tiff files



#### 7. Image projection in the z-axis (z-projection) of completed images stacks in *.tiff format 

```   
projection = operetta_annotation.z_projection(path_to_tiff)
```

* path_to_tiff - path to the completed image of the z-axis stack in *.tiff format

###### Displayed parameters for z-projection adjustment:

* size - the size of the displayed image (adjustable with the scroll of the mouse during real time)
* gamma
* threshold
* brightness 
* contrast
* color - color scale of image projection [red, green, blue, magenta or grey]
* method - method for the z projection [avg, max, min, std] (methods of averaging the z-stack values)

###### Displayed options for z-projection adjustment:

* apply - apply changes to the parameters and display a new image (excluding 'size')
* auto - automative adjustment of and display a new image (excluding 'size' and 'color'). The color should be chosen manually and click 'Apply'
* save - the adjusted z-projection image is saved to a variable in the python

##### Example:

###### First projection: chanel 2 - Alexa488
```
projection1 = operetta_annotation.z_projection('chanel_ch2.tiff')
cv2.imwrite('projection_chanel_ch2.png', projection1)
```

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/main/fig/adjust_z_stack1.bmp" alt="drawing" width="600" />
</p>




###### Second projection: chanel 3 - Alexa647
```
projection2 = operetta_annotation.z_projection('chanel_ch3.tiff')
cv2.imwrite('projection_chanel_ch3.png', projection2)
```

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/main/fig/adjust_z_stack2.bmp" alt="drawing" width="600" />
</p>



#### 8. Merge images projections in one complex picture

```   
merged_projection = operetta_annotation.merge_images(image_list)
```

* image_list - list of projections to merge, eg. [projection1, projection2]

###### Displayed parameters for image merging adjustment:

* size - the size of the displayed image (adjustable with the scroll of the mouse during real time)
* Images intensity: Img_0, Img_1, ... - depending on number of input images

###### Displayed options for image merging adjustment:

* apply - apply changes to the parameters and display a new image (excluding 'size')
* save - the adjusted and merged image is saved to a variable in the python

##### Example:

###### Result of merged channels 

```
image_list = [projection2, projection1]

merged_projection = operetta_annotation.merge_images(image_list)
cv2.imwrite('merged_projection.png', merged_projection)
```

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/main/fig/merged_new.bmp" alt="drawing" width="600" />
</p>



#### 9. Display of composite photo from Opera (z-projection) with grid in places of single photos and their indexes

```
numbers_of_pictures = operetta_annotation.image_grid(path_to_opera_projection, img_length, img_width)
```

* path_to_opera_projection - path to image (z-projection) 
* imgs - a set of images in the correct order [from 'image_sequences' function]
* img_length - number of pictures for images included in y-axis [from 'image_sequences' function]
* img_width - number of pictures for images included in x-axis [from 'image_sequences' function]


###### Displayed parameters for image merging adjustment:

* size - the size of the displayed image (adjustable with the scroll of the mouse during real time)
* Enter id of image: user provides comma-separated numbers of the index for the part of the image that part raw data will be used for further analysis (e.g. for further analysis by AI/ML algorithms)

###### Displayed options for image merging adjustment:

* apply - saving the list of indexes of chosen parts (images) to the python variable -> (numbers_of_pictures)


##### Example:


```
numbers_of_pictures = operetta_annotation.image_grid('projection_chanel_ch2.png', img_length, img_width)
```

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/main/fig/grid.bmp" alt="drawing" width="600" />
</p>


#### Examples of images grid for various microscope lens:

##### Image on lens x20

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/main/fig/select2.bmp" alt="drawing" width="600" />
</p>

##### Image on lens x40

<p align="center">
<img  src="https://github.com/jkubis96/Operetta_tool/blob/main/fig/select3.bmp?raw=true" alt="drawing" width="600" />
</p>


##### Image on lens x63

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/main/fig/select1.bmp" alt="drawing" width="600" />
</p>




#### 10. Split images from different chanels

```
operetta_annotation.split_chanels(path_to_images, path_to_save)
```

* path_to_images - path to directory including raw Opera images ['Images' directory]
* path_to_save - path to directory for splited channels save


###### Figure 3 Directories with images from different channels

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/main/fig/ff.bmp" alt="drawing" width="600" />
</p>




#### 11. Separation of selected stacks of images by indexes into separate directories

```
operetta_annotation.select_pictures(image_dictinary, path_to_images, path_to_save, numbers_of_pictures)
```

* image_dictinary (imgs) - a set of images in the correct order [from 'image_sequences' function]
* path_to_images - path to directory including raw Opera images ('Images') or separate channels directory, eg. ('ch1','ch2')
* path_to_save - path to the directory for saving chosen by indexes images stacks
* numbers_of_pictures - list of choosen pictures [1,2,3,10,11,21,...]


##### Example:


```
operetta_annotation.select_pictures(imgs, 'ch2', 'selected_images', numbers_of_pictures)
```


##### Directories with separated chosen stacks of raw Opera images

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/main/fig/se.bmp" alt="drawing" width="600" />
</p>

