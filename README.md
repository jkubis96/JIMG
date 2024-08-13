# JIMG - python library

##### New version 1.4.9 allows easier and more complex processing of images from Opera Phenix Plus High-Content Screening System.

<p align="right">
<img  src="https://github.com/jkubis96/Logos/blob/main/logos/jbs_current.png?raw=true"drawing" width="250" />
</p>


### Author: Jakub Kubiś 

<div align="left">
 Institute of Bioorganic Chemistry<br />
 Polish Academy of Sciences<br />
 Department of Molecular Neurobiology<br />
</div>


## Description


<div align="justify"> The JIMG is a Python library created for handling high-resolution images from Opera Phenix Plus High-Content Screening System (including raw images concatenation, z-projection, channels merging, and image resizing).  Additionally, we have created options for annotation parts of images and choosing them for further analysis eg. an analysis by ML / AI algorithms of differences between healthy and diseased parts of the brain related to the disease but not explicitly defined or obvious to observers. </div>

</br>

## Installation

#### In command line write:

```
pip install JIMG==1.4.9
```



## Usage


#### 1. Import required libraries

```
from JIMG import jimg 
import cv2
```

</br>


#### 2. Load images metadata from xml file

```
xml_file, metadata = jimg.xml_load('Images/Index.idx.xml')
```

* path_to_opera_xml - path to Index data of Opera in xml format (usual 'Images/Index.idx.xml')

##### This function load metadata about images from current project included in Index.idx.xml. 
* xml_file - contains information about the position of each raw image required for the next functions
* metadata - contains information about the number of channels, signal wavelength of channels, resolution, etc.


##### Metadata:

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.2.0.0/fig/metadata.bmp" alt="drawing" />
</p>

</br>


#### 3. Detection of outlier images not included in the square of the main image

```
xml_file, figure = jimg.detect_outlires(xml_file, list_of_out = [])
```

##### This function displays the arrangement of individual images and their location on the map of the main image

* xml_file - input = prevoiusly loaded xml_file / output = function return xml_file reduced images selected by user based on figure
* figure - the map of each image location on the main image
* list_of_out - list of the indexes for outliers images selected by the user from figure

##### Example:

###### First run of function to display images' location map
```
xml_file, figure = jimg.detect_outlires(xml_file, list_of_out = [])

```

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.1.0.0/fig/before_outlires.bmp" alt="drawing"  />
</p>


###### Second run of function to reduce outliers number of images and display images' location map
```
xml_file, figure = jimg.detect_outlires(xml_file, list_of_out = [0])

```

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.1.0.0/fig/after_outlires.bmp" alt="drawing" />
</p>


</br>


#### 4. Detection of lack of images in sequence included in main image 

```
xml_file, figure_in = jimg.repair_blanks(xml_file)
```

##### This function completes missing image elements and saves their coordinates (x and y axis) to the xml metadata file.
 
##### In subsequent analysis, these points will be filled with a black square in the main photo

* xml_file - input = prevoiusly loaded xml_file / output = function return xml_file supplemented with missing elements


##### Example:

###### Before (one missing element)


<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.1.0.0/fig/opera1.bmp" alt="drawing"  />
</p>


###### After (repaired)


<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.1.0.0/fig/opera2.bmp" alt="drawing" />
</p>

</br>


#### 5. Adaptation of the Opera images coordinates to the coordinates in the overview image

```
image_queue, img_length, img_width = jimg.image_sequences(opera_coordinates = xml_file)
```

* imgs - a set of images in the correct order for subsequent analyses
* img_length - number of pictures for images included in y-axis 
* img_width - number of pictures for images included in x-axis 


</br>


#### 6. Composition of the raw images on the x, y, z axis into the main photo in *.tiff format including all z-stack slices

```   
metadata_out = jimg.image_concatenate(path_to_images, path_to_save, image_queue, metadata, img_length, img_width, overlap, channels, resize, n_proc = 4, par_type = 'processes'):
```

* metadata - metadata file loaded in 'xml_load' function
* channels - list of channels included in composition analysis, eg.['ch2', 'ch3']
* overlap - the percentage of image overlap defined earlier when taking a picture on the microscope [eg. 0.05 or 0.1 (5% or 10%)]
* path_to_images - path to directory including raw Opera images ['Images' directory]
* path_to_save - path to directory in which the output *.tiff files will saved
> * WARNING! In this function path_to_images / path_to_save should be full path\
>	The full path can be obtained using os.getcwd() + 'directory name' joined using os.path.join() eg. full_path = os.path.join(os.getcwd(), 'Images')
* n_proc - the number of processor threads or cores involved in the analysis adapted to the device on which the analysis will be performed. The more threads / cores, the faster the analysis
* image_queue - a set of images in the correct order [from 'image_sequences' function]
* img_length - number of pictures for images included in y-axis [from 'image_sequences' function]
* img_width - number of pictures for images included in x-axis [from 'image_sequences' function]
* par_type - type of parallelization ['processes', 'threads']
* metadata_out - inputted metadata corrected by resize factor


##### The results for each channel are saved in separate *.tiff files


</br>


#### 6.1. Function for resizing images '*.tiff' files after concatenation

##### This function can be used when the original image is very big and it can be hard to conduct z-projection on a PC with worse hardware


```   
resized_metadata = jimg.resize_tiff(channels, metadata, prefix = 'resized' , height = None, width = None, resize_factor = None)
```


* metadata - metadata file loaded in 'xml_load' and corrected in 'image_concatenate' function or additionally corrected by 'resize_tiff' function

##### The metadata is required for this function due to the change in image scale which will be important for setting scale bar in the 'add_scalebar' function

* channels - list of channels for resize eg.['ch2', 'ch3']
* prefix - prefix to the new image name saved in working dir after resized [default: 'resized']

##### Warning! You can set only one parameters to resize image! If you set height then width will change proportionally and vice versa.

##### It is important for maintaining proportions in the context of biological rights.

* height - new hight of the image [default: None]
* width - new width of the image [default: None]
* resize_factor - factor of resize both image parameters [height/resize_factor x width/resize_factor]  [default: None]

</br>


#### 7. Image projection in the z-axis (z-projection) of completed images stacks in *.tiff format 

```   
projection = jimg.z_projection(path_to_tiff, stack_check = True)
```

* path_to_tiff - path to the completed image of the z-axis stack in *.tiff format
* stack_check - is the option for checking the outliers in images on the z-axis. This function requires more RAM memory and makes that z-projection last longer. [default: True]

###### Displayed parameters for z-projection adjustment:

* size - the size of the displayed image (adjustable with the scroll of the mouse during real time)
* gamma
* threshold
* brightness 
* contrast
* color - color scale of image projection ["grey", "blue", "green", "red", "magenta", 'yellow', 'cyan']
* method - method for the z projection ["avg", "max", "min", "sdt", "median"] (methods of averaging the z-stack values)

###### Displayed options for z-projection adjustment:

* apply - apply changes to the parameters and display a new image (excluding 'size')
* auto - automative adjustment of and display a new image (excluding 'size' and 'color'). The color should be chosen manually and click 'Apply'
* save - the adjusted z-projection image is saved to a variable in the python

##### Example:

###### First projection: channel 1 - DAPI
```
projection1 = jimg.z_projection('channel_ch1.tiff', stack_check = True)
cv2.imwrite('projection_channel_ch1.png', projection1)
```

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.1.0.0/fig/projection1.bmp" alt="drawing" />
</p>


###### Second projection: channel 2 - Alexa488
```
projection2 = jimg.z_projection('channel_ch2.tiff', stack_check = True)
cv2.imwrite('projection_channel_ch2.png', projection2)
```

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.1.0.0/fig/projection2.bmp" alt="drawing" />
</p>



###### Third projection: channel 3 - Alexa647
```
projection3 = jimg.z_projection('channel_ch3.tiff', stack_check = True)
cv2.imwrite('projection_channel_ch3.png', projection3)
```

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.1.0.0/fig/projection3.bmp" alt="drawing" />
</p>

</br>


#### 8. Merge images projections in one complex picture

```   
merged_projection = jimg.merge_images(image_list)
```

* image_list - list of projections to merge, eg. [projection1, projection2, projection3]

###### Displayed parameters for image merging adjustment:

* size - the size of the displayed image (adjustable with the scroll of the mouse during real time)
* Images intensity: Img_0, Img_1, ... - depending on number of input images

###### Displayed options for image merging adjustment:

* apply - apply changes to the parameters and display a new image (excluding 'size')
* save - the adjusted and merged image is saved to a variable in the python

##### Example:

###### Result of merged channels 

```
image_list = [projection1, projection2, projection3]

merged_projection = jimg.merge_images(image_list)
cv2.imwrite('merged_projection.png', merged_projection)
```

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.1.0.0/fig/merged.bmp" alt="drawing" />
</p>

</br>


#### 8.1. Resize projection - *if required*

```   
resized_image, resized_metadata = jimg.resize_projection(image, metadata = None, height = None, width = None, resize_factor = None)

```

##### If you resize the projection before the 'add_scalebar' function and you need the scale bar on the image, the metadata is required for this function due to the change in image scale which will be important for setting the scale bar in the 'add_scalebar' function

##### If you resize the projection after adding a scale bar or you don't need a scale bar you can omit the metadata and set the None value.

* image - image after 'z_projection' or 'merge_images' 
* metadata - metadata file loaded in 'xml_load' [default: None]

##### Warning! You can set only one parameters to resize image! If you set height then width will change proportionally and vice versa.

##### It is important for maintaining proportions in the context of biological rights.

* height - new hight of the image [default: None]
* width - new width of the image [default: None]
* resize_factor - factor of resize both image parameters [height/resize_factor x width/resize_factor]  [default: None]

</br>


#### 8.2. Add scalebar - *if required*

```   
scaled_image = jimg.add_scalebar(image, metadata)
```

* image - image after 'z_projection' or 'merge_images'
* metadata - metadata file loaded in 'xml_load' or function


###### Displayed parameters for image merging adjustment:

* Scale length - the size of displayed scale [um]
* Scalebar thickness - thickness of the scale bar line [px]
* Color - the color of the scale bar
* Font size - the size of the scale fonts
* Position - position of the scale bar on the image
* Horizontal position - adjustment of the location of the scale bar in the horizontal position
* Vertical position - adjustment of the location of the scale bar in the vertical position


###### Displayed options for image merging adjustment:

* apply - apply changes to the parameters and display a new image (excluding 'size')
* save - the adjusted and merged image is saved to a variable in the python

##### Example:

###### Result with scale bar

```
merged_projection_scale = jimg.add_scalebar(merged_projection, metadata)
cv2.imwrite('merged_projection_scale.png', merged_projection_scale)
```

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.1.0.0/fig/scale%20bar.bmp" alt="drawing" />
</p>

</br>

#### 8.3. Loading previously saved image

```   
loaded_image = cv2.imread('image.png', cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
```

###### It is important to load images in such a way as to save the depth and colors of the projected image

</br>


#### 9. Display of composite photo from Opera (z-projection) with grid in places of single photos and their indexes

```
numbers_of_pictures = jimg.image_grid(path_to_opera_projection, img_length, img_width)
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
numbers_of_pictures = jimg.image_grid('merged_projection.png', img_length, img_width)
```

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.1.0.0/fig/grid.bmp" alt="drawing" />
</p>


#### Examples of images grid for various microscope magnification:

##### Image magnification x20

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.1.0.0/fig/select2.bmp" alt="drawing" />
</p>

##### Image magnification x40

<p align="center">
<img  src="https://github.com/jkubis96/JIMG/raw/v.1.0.0/fig/select3.bmp" alt="drawing" width="600" />
</p>


##### Image magnification x63

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.1.0.0/fig/select1.bmp" alt="drawing" />
</p>


</br>


#### 10. Split images from different channels

```
jimg.split_channels(path_to_images, path_to_save)
```

* path_to_images - path to directory including raw Opera images ['Images' directory]
* path_to_save - path to directory for splited channels save


###### Directories with images from different channels

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.1.0.0/fig/ff.bmp" alt="drawing" />
</p>


</br>

#### 11. Separation of selected stacks of images by indexes into separate directories

```
jimg.select_pictures(image_dictionary, path_to_images, path_to_save, numbers_of_pictures, chennels)
```

* image_dictionary (image_queue) - a set of images in the correct order [from 'image_sequences' function]
* path_to_images - path to directory including raw Opera images ('Images') or separate channels directory, eg. ('ch1','ch2')
* path_to_save - path to the directory for saving chosen by indexes images stacks
* numbers_of_pictures - list of choosen pictures [1,2,3,10,11,21,...]
* channels - list of channels for choosing images eg.['ch2', 'ch3']


##### Example:


```
jimg.select_pictures(image_queue, 'Images', 'selected_images', numbers_of_pictures, ['ch2'])
```


##### Directories with separated chosen stacks of raw Opera images

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.1.0.0/fig/se.bmp"drawing"  />
</p>

##### The high-resolution wild images selected in this way can be used for further analysis using ML/AI algorithms, depending on the topic under study


#### Have fun JBS
