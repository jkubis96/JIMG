# JIMG - tool and application for high-resolution image management


<br />



<p align="right">
    <img src="https://github.com/jkubis96/Logos/blob/main/logos/jbs_current.png?raw=true" alt="drawing" width="250" />
    <img src="https://github.com/jkubis96/Logos/blob/main/logos/jbi_current.png?raw=true" alt="drawing" width="250" />
</p>




<br />

#### JBioImaging is a part of JBioSystem responsible for application creation in the biological images handling

<br />

<br />


### Author: Jakub Kubiś 

<div align="left">
 Institute of Bioorganic Chemistry<br />
 Polish Academy of Sciences<br />
 Department of Molecular Neurobiology<br />
</div>


<br />


## Description


<div align="justify"> 

This tool was created for handling high-resolution images from the Opera Phenix Plus High-Content
Screening System, including operations such as concatenating raw series of images, z-projection,
channel merging, image resizing, etc. Additionally, we have included options for annotating specific
parts of images and selecting them for further analysis, for example, teaching ML/AI algorithms.

Certain elements of this tool can be adapted for data analysis and annotation in other imaging systems.
For more information, please feel free to contact us!
			
</div>


<br />

## Table of contents

[Installation](#installation) \
[Usage](#usage) \
[Images operations](#dz) 

1. [Start](#start) \
1.1 [Project manager](#project-manager) \
1.2 [Loading metadata](#loading-metadata) \
1.3 [Concat images](#concat-im) \
1.4 [Z-projection](#z-projection) \
1.5 [Images manager](#img-manager) \
1.6 [Merge images](#img-merge) \
1.7 [Add scale-bar](#add-scalebar) \
1.8 [Annotate image](#annotate-image) \
1.9 [Annotate raw](#annotate-raw) \
1.10 [Exit](#exit) 

2. [Reset](#reset_) 
3. [Contact](#contact) 
4. [Manual](#manual) 
5. [License](#license) 
6. [Exit](#exc)
7. [Console code](#code)\
7.1 [Images and metadata handling](#metadata-code)\
7.2 [Image concatenation](#concatenation-code)\
7.3 [Image adjustment](#adjust-code)




<br />

#### Previous version of code: [JIMG v.1.4.9](https://github.com/jkubis96/JIMG/tree/v.1.0.0)


<br />

## Installation <a id="installation"></a>

<br />


#### Python users:

CMD:
```
pip install JIMG>=2.1.7
```


<br />


#### For Windows users .exe:

* [Download](https://www.mediafire.com/file/a9r5lzrvljfytkn/JIMG-v.2.1.7.exe/file)


<br />


#### Docker container:


```
docker pull jkubis96/jimg:v2.1.7
```


<br />
<br />


## Usage <a id="usage"></a>



### Run application 

<br />

#### Python users:


Python console:

```
from JIMG.app.load_app import run
run()
```


<br />


#### For Windows users:

Run current version of application.exe file


<br />


<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.2.0.0/fig/1%20front.bmp" alt="drawing"/>
</p>


<br />

#### Docker container:


```
docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw --rm jkubis96/jimg:v2.1.7
```


<br />
<br />


### Images operations <a id="dz"></a>

<br />


This chapter below will present all operations that the application allowed to conduct on high-resolution images from a microscope (especially High Content Screening such as Opera Phoenix).

<br />


<p align="center">
<img  src="https://github.com/jkubis96/JIMG/raw/v.2.0.0/fig/before_zoom.bmp" alt="drawing" />
</p>

<br />


<span><font color="green">Displayed images can be resized with Window size / Size sliders and also zoomed by pressing and holding the 'Z' button and scrolling by mouse or touchpad in selected part of the image</font></span>


<br />

<br />


### 1. Start <a id="start"></a>

This runs the options for image operations and management

<br />


<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.2.0.0/fig/1.1%20run.bmp" alt="drawing"/>
</p>


<br />

<br />


#### 1.1 Project manager <a id="project-manager"></a>

Options for loading and saving project

<br />


<p align="center">
<img  src="https://github.com/jkubis96/JIMG/blob/v.2.0.0/fig/1.p.png?raw=true" alt="drawing"  />
</p>


<br />


* Load project - loading previous saved project with *.pjn extension

<br />


<p align="center">
<img  src="https://github.com/jkubis96/JIMG/blob/v.2.0.0/fig/1.pl.png?raw=true" alt="drawing" />
</p>

<br />


> * Browse - select path to the project [*.pjm]

> * Load - load project metadata

> * Back - back to the previous application window

<br />



* Save current - save the currently edited project


<br />

<p align="center">
<img  src="https://github.com/jkubis96/JIMG/blob/v.2.0.0/fig/1.ps.png?raw=true" alt="drawing" />
</p>


<br />


> * Browse - select directory to the project save

> * Save - save project metadata

> * Back - back to the previous application window

<br />




* Back - back to the previous application window



<br />

<br />


#### 1.2 Loading metadata <a id="loading-metadata"></a>

Options for loading and adjusting raw microscope images metadata (coordinates, size, channels, etc.)

<br />


<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.2.0.0/fig/1.1.1%20load%20metadata.bmp" alt="drawing" />
</p>


<br />


* Browse path - select path to the metadata - Index.inx.xml 

<br />


* Load metadata - loading metadata to the application

<br />


* Display core - displaying the core of the raw images by their coordinates and repairing lacking elements of the image (full image must be rectangular)

<br />

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.2.0.0/fig/1.1.1.1%20display%20core.bmp" alt="drawing" />
</p>


<br />


* Core reduce - manually adjusting the images included in the full image core (full image must be rectangular)



<br />

<br />


> #### 1.2.1 Core reduce - options


<br />

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.2.0.0/fig/1.1.1.2%20cores%20reducing.bmp" alt="drawing" />
</p>


<br />

> * Window size - the size of the displayed core image

> * IDs - the list of the images for excluding from the image core 

> * Reduce - applying the IDs list of the raw images for reducing

> * Return - reversing changes in the image core 

> * Save - saving changes in the image core

> * Back - back to the previous application window

<br />



* Back - back to the previous application window 

<br />

<br />


#### 1.3 Concat images <a id="concat-im"></a>

This runs the options for images concatenation


<br />


<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.2.0.0/fig/1.1.2%20concatenate%20images.bmp" alt="drawing" />
</p>


<br />



* Browse input - select the path to the raw images (the same location as for Index.inx.xml)

<br />


* Browse save - select the path to the directory for saving concatenated *.tiff files

<br />


* Images overlap value - the value for overlap of the raw images (value is set during images making via microscope)

<br />


* Select channels - select the channel or channels for which the *.tiff files should be created

<br />



* Preferable multiprocessing - select the multiprocessing option for images concatenation [processes / threads]

> In the case of the application for Windows, the multiprocessing option is threading (available for this type of application)
> If the concatenation process should be faster (processes option) run the application from the Python console

<br />



* Number of cores/threads - select the number of cores/threads [default: 0,5 * available cores or 0.75*available threads]

> The option depends on the same case as the above 'Preferable multiprocessing'

<br />


* Resize factor for concatenated image - select the value for resize. Default: 1 (no changes in the original image)

> Options for reducing RAM usage and full image size (shape and weight)

<br />



* Start concatenate - run the concatenation process

> It can last several minutes and depends on image size, number of image layers, and multiprocessing options.

<br />


* Back - back to the previous application window 

<br />

<br />


#### 1.4 Z-projection <a id="z-projection"></a>


This runs the options for z-projection of the image from the *.tiff file


<br />


<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.2.0.0/fig/1.1.3.1%20tiff%20load%20-%20z%20projection.bmp" alt="drawing" />
</p>


<br />


* Browse path - select the path to the *.tiff file

<br />


* Load - load selected file

<br />

<br />


> #### 1.4.1 Z-selection

> Options for selecting particular slices that will excluded from the Z-projection



<br />


<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.2.0.0/fig/1.1.3.2%20slices%20selection%20-%20z%20projection.bmp" alt="drawing" />
</p>


<br />


> * Window size - the size of the displayed image


<br />


> * Slice - number of currently displayed slice



<br />


> * Enter numbers of slices to remove - IDs of slices that will excluded from the Z-projection



<br />


> * Apply - accept slices to remove


<br />

> * Back - back to the previous application window 


<br />


* Projection - run projection options

<br />

<br />


> #### 1.4.2 Z-projection

> Options for Z-projection conducting

> > Channel 1

<br />


<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.2.0.0/fig/1.1.3.3%20z%20projection.bmp" alt="drawing" />
</p>


<br />


> > Channel 2

<br />


<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.2.0.0/fig/1.1.3.3%20z%20projection%202.bmp" alt="drawing" />
</p>


<br />



> * Window size - the size of the displayed image


<br />


> * Gamma - the value of gamma



<br />


> * Min - the minimum value of the pixel


<br />


> * Max - the maximum value of the pixel


<br />


> * Brightness - the value of brightness


<br />


> * Contrast - the value of contrast


<br />


> * CLAHE - run the CLAHE algorithm

<br />


> * Color - the color of the projection

<br />


> * Projection method - the method of the projection

<br />


> * Apply - apply and display changes

<br />


> * Save - save the projection

> > After saving the projection will be visible in the 'Images manager'



<br />

> * Back - back to the previous application window 


<br />




* Back - back to the previous application window 

<br />

<br />



#### 1.5 Images manager <a id="img-manager"></a>

Options for images management and operation

<br />


<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.2.0.0/fig/1.1.4%20manager.bmp" alt="drawing" />
</p>



<br />


* Window size - the size of the displayed image


<br />


* Add - open a window for adding to the manager image in formats of *.tiff, *.tif, *.jpg, *.jpeg, *.png

<br />


* Remove - remove the selected image from the manager

<br />


* Display - display the selected image from the manager

<br />


* Resize - resize the selected image from the manager

<br />

<br />

> #### 1.5.1 Resize options


<br />


<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.2.0.0/fig/1.1.4.1%20manager%20-%20resize.bmp" alt="drawing" />
</p>


<br />



> * Height - a new height of the image


<br />


> * Width - a new width of the image

<br />


> * Resize factor - a factor of resizing (dividing) both [height x width] values


<br />


> * Resize - apply resizing to the image


<br />


> * Save - save the resized image


<br />


> * Back - back to the previous application window 


<br />
<br />



> #### 1.5.2 Rotate options


<br />



<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.2.0.0/fig/1.1.4.3%20manager%20-%20rotate.bmp" alt="drawing" />
</p>


<br />



> * Rotate ° - set the degree of angle of the image rotation  [0°, 90°, 180°, 270°]


<br />


> * Mirror type ° - set the side for the image mirroring  [horizontal, vertical, horizontal/vertical].


<br />


> * Rotate - apply rotation/mirroring changes


<br />


> * Save - save rotation/mirroring changes

> > After saving the projection will be visible in the 'Images manager'


<br />

<br />



* Save - save the selected image from the manager

<br />

<br />



> #### 1.5.3 Save options


<br />


<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.2.0.0/fig/1.1.4.2%20manager%20-%20save.bmp" alt="drawing" />
</p>


<br />



> * Browse - the path to the save directory


<br />


> * File name - a file name for the saving image 

<br />


> * Extension - the extension of the saving file [*.png, *.tiff, *.tif]


<br />


> * Save - save the image


<br />


> * Back - back to the previous application window 


<br />


* Back - back to the previous application window 


<br />

<br />


#### 1.6 Merge images <a id="img-merge"></a>

Options for multiple images merging

<br />


<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.2.0.0/fig/1.1.5.1%20merge%20select.bmp" alt="drawing" />
</p>



<br />


* Window size - the size of the displayed image


<br />


* Display - display the selected image


<br />



* Merge - run options (window) for selected images merging


<br />

<br />


> #### 1.6.1 Merge options

<br />

> The number of images must be higher than 1 and the images must have the same shape [height x width]


<br />

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.2.0.0/fig/1.1.5.2%20merge%20images.bmp" alt="drawing" />
</p>


<br />



> * Size - the size of the merged image


<br />


> * Images intensity - intensity of particular images [Img_0, Img_1, ...] in the merging image


<br />



> * Apply - apply the changes in the intensity of particular images


<br />



> * Save - save the merging image


<br />


> * Back - back to the previous application window 


<br />



* Save - run the save option as same as in 'Images Manager'


<br />


* Back - back to the previous application window 


<br />


<br />


#### 1.7 Add scale-bar <a id="add-scalebar"></a>

Options for multiple images merging

<br />


<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.2.0.0/fig/1.1.6%20scalebar.bmp" alt="drawing" />
</p>


<br />

* Window size - the size of the displayed image

<br />


* Display - display the selected image


<br />


* Add scale - add a scale-bar to the selected image


<br />

<br />


> #### 1.7.1 Scale-bar options


<br />



<br />

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.2.0.0/fig/1.1.6.1%20add%20scale.bmp" alt="drawing" />
</p>


<br />



> * Window size - the size of the displayed image


<br />


> * μm/px - scale size in the μm of images on pixel


<br />



> * Scale length [μm] - the length of the scale displayed on the image


<br />


> * Scale thickness [px] - the width in the pixels of the scale displayed on the image


<br />


> * Color - the color of the scale displayed on the image


<br />



> * Font size - size of the signature font under the scale


<br />



> * Position - the position of the scale-bar on the image


<br />


> * Horizontal position - the horizontal shift of the original position of the scale-bar


<br />


> * Vertical position - the vertical shift of the original position of the scale-bar


<br />


> * Apply - apply the changes to the scale-bar settings


<br />



> * Save - save the image with the scale-bar settings


<br />


> * Back - back to the previous application window 


<br />



* Save - run the save option as same as in 'Images Manager'


<br />

<br />



#### 1.8 Annotate image <a id="annotate-image"></a>

Options for image annotation

<br />


<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.2.0.0/fig/1.1.7%20annotate.bmp" alt="drawing" />
</p>


<br />

* Window size - the size of the displayed image

<br />


* Display - display the selected image


<br />


* Annotate - draw annotation on the selected image


<br />

<br />


> #### 1.8.1 Annotation options


<br />



<br />

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.2.0.0/fig/1.1.7.1%20annotate%20-%20annotate.bmp" alt="drawing" />
</p>


<br />

> For drawing on the image the 'D' button must be pressed on the keyboard. The drawing is conducted by mouse or touchpad.



> * Size - the size of the image for drawing


<br />


> * Line width - the width of lines drawn on the image


<br />



> * Line color - the color of the drawn lines


<br />


> * Undo - undo changes in line drawing


<br />


> * Apply - apply changes in line drawing


<br />



> * Save - save the image with drawn annotation

<br />


> > Saved results of annotation 

<br />


<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.2.0.0/fig/1.1.7.2%20annotate%20-%20results.bmp" alt="drawing" />
</p>



<br />


> > _annotated_image[n] - image with drawn annotation


<br />


> > _annotated_image[n]_lines - raw annotation lines


<br />


> > _annotated_image[n]_mask - raw mask for drawn annotation


<br />


> * Close - close the annotation window and back to the previous application window



<br />

<br />


#### 1.9 Annotate raw <a id="annotate-raw"></a>

Options for raw images annotation

<br />


<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.2.0.0/fig/1.1.8%20annotation%20raw.bmp" alt="drawing" />
</p>


<br />

* Window size - the size of the displayed image

<br />


* Display - display the selected image


<br />


* Annotate raw - draw annotation on the selected image


<br />

<br />


> #### 1.9.1 Selecting images for annotate


<br />



<br />

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.2.0.0/fig/1.1.8.1%20annotation%20raw%20-%20select%20images.bmp" alt="drawing" />
</p>


<br />

> Based on the grid and numeration, the raw images are selected for annotation


<br />


> * Size - the size of the image for selection


<br />



> * Grid color - the color of the grid


<br />


> * Font color - the color of the numeric font


<br />


> * Apply - apply changes in the color of the grid and font


<br />



> * Save - accept the selected images and start the annotation

<br />

<br />


> #### 1.9.2 Annotation panel


<br />

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.2.0.0/fig/1.1.8.1.1%20annotation%20raw%20-%20raw.bmp" alt="drawing" />
</p>


<br />




> * Adjust - run panel for raw image projection adjustment


<br />

> > This panel is the same as in 1.4.2 Z-projection chapter.
> > After Apply changes the Close button will save these changes for all current annotated images.


<br />

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.2.0.0/fig/1.1.8.1.1.1%20annotation%20raw%20-%20raw%20-%20adjust.bmp" alt="drawing" />
</p>


<br />


> * Annotate - run panel for raw image annotation

<br />

> > This panel is the same as in 1.8.1 Annotation options chapter.
> > After Apply changes the Close button will save annotation for current image.


<br />

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.2.0.0/fig/1.1.8.1.1.2%20annotation%20raw%20-%20raw%20-%20annotate.bmp" alt="drawing" />
</p>



<br />


> * Next - change for the next image in the selected images queue


> > If the image will not be annotated, the Next button means, that the user allows to prepare a mask of the whole image (it means that the whole image is taken into consideration in the study)



<br />



> * Previous - change for the previous image in the selected images queue


<br />

 
> * Discard - exclude the image from the analysis



<br />


> * Save - save whole progress of annotation and back to previous application window for saving or select other images

> > <span style="color:red">!!! If next, the results will not save with the 'Save selection' button, and the next analysis will be run with the 'Annotate raw' button, the results of the current analysis will be lost !!!</span>


<br />


> * Back -  back to the previous application window 


<br />




* Save selection - save results obtained from 'Annotate raw' analysis


<br />

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.2.0.0/fig/1.1.8.1.2%20annotation%20raw%20-%20raw%20-%20save%20results.bmp" alt="drawing" />
</p>


<br />

> The results of all annotations of raw images will be saved. An image selection from the image manager list is not required.

<br />


> * Browse - select the path to the save directory 


<br />


> * File name - provide a name for the directory in which the annotation results will be saved


<br />


> * Extension -  select image extension for grid map and Z-projection image of raw annotated images [*.png, *.tiff, *.tif]


<br />


> * Save -  save results in the given directory with the given name. It can last a while...


<br />


> * Back -  back to the previous application window 


<br />


<br />


> #### 1.9.3 Annotation results

<br />


> * Main save directory for annotation results of selected images
 
<br />

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.2.0.0/fig/1.1.8.1.2.1%20annotation%20raw%20-%20raw%20-%20save%20results%20outside.bmp" alt="drawing" />
</p>


<br />

> > It contains directories for each selected and not excluded from the analysis selected image. Numbers are related to the numbers on the grid image map (image with 'grid_' prefix).  Additionally, this directory is included Z-projection without the grid.


<br />


> * Directories with annotated images (directories with 'img_' prefix)


<br />

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.2.0.0/fig/1.1.8.1.2.2%20annotation%20raw%20-%20raw%20-%20save%20results%20inside.bmp" alt="drawing" />
</p>


<br />

> > It contains a Z-projection of the current image with a given number (annotated_projection), their raw annotation (annotation), masks (mask_8bit, mask_16bit, and mask_binary), and all raw images that belong to the Z axis (slices) that were not excluded in '1.4.1 Z-selection' chapter step of whole image adjustment 



<br />

<br />

#### 1.10 Exit <a id="exit"></a>

<br />


> Exit from 'Images operations' and return to Main menu


<br />

<br />


### 2. Reset <a id="reset_"></a>


<br />


> The Reset button allows removing all current metadata to start a new analysis without re-running the whole application.

<br />

<br />

### 3. Contact <a id="contact"></a>


<br />


> It opens window with contact information.

<br />

<br />

### 4. Manual <a id="manual"></a>


<br />


> It opens the default web browser with the GitHub directory with manual (present webpage)


<br />

<br />


### 5. License <a id="license"></a>


<br />


> It opens product License [MIT](https://github.com/jkubis96/Operetta_tool/blob/v.2.0.0/LICENSE)

<br />

<br />


### 6. Exit <a id="exc"></a>


<br />


> It closes the whole application. All not saved results will be lost!

<br />




### 7. Console code <a id="code"></a>




<br />


#### 7.1 Images and metadata handling <a id="metadata-code"></a>

Some of the functions can be run from the Python console and used for creating automatic or semiautomatic pipelines for image analysis.

<br />

Loading library:

```
from JIMG.functions import jimg as jg
```


<br />


##### 7.1.1 Metadata loading

<br />

Function:

```
xml_load(path_to_xml)
```

> Description:\
  This function loads the images index file and collects metadata.


> Args:

  >* path_to_xml (str) - path to a image metadata

> Returns:

  >* image_info (pd.DataFrame) - list of images with numeration and coordinates
  >* metadata (dict) - images information


<br />

Example:

```
img_info, metadata =  jg.xml_load(path_to_xml = path_to_inx)
```

<br />

Output:


<br />

> Metadata

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.2.0.0/fig/metadata.bmp" alt="drawing" />
</p>


<br />

> Image info

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.2.0.0/fig/img_info.bmp" alt="drawing" />
</p>



<br />


##### 7.1.2 Metadat repairing


<br />

Function:

```
repair_image(image_info, dispaly_plot = True)
```

> Description:\
  This function is used for repairing microscope image-taking errors.
  The full images (core of the image) obtained from microscopes such as the Opera Pheonix consist of many smaller (raw) images.
  Sometimes microscope takes photos out of the targeted place or misses one photo and then the full photo can not be merged.
  This function allows for automatic repair of the core of the full image.
  If the core is still not appropriate use manual repair by manual_outlires() function.
    


> Args:

  > * image_info (pd.DataFrame) - list of images with numeration obtained from the xml_load() function
  > * dispaly_plot (bool) - show the graph in the console. Default: True
  
> Returns:

  > * fig - location and numeration of raw images in the main core of the full image
  > * image_info (DataFrame) - adjusted image_info


<br />



Function:

```
manual_outlires(image_info, list_of_out = [], dispaly_plot = False)
```

> Description:\
   This function is used for repairing microscope image-taking errors.
   The full images (core of the image) obtained from microscopes such as the Opera Pheonix consist of many smaller (raw) images.
   Sometimes microscope takes photos out of the targeted place or misses one photo and then the full photo can not be merged.
   This function allows for checking how the raw images are placed and manually removing some of them from further analysis.
    


> Args:

  > * image_info (pd.DataFrame) - list of images with numeration obtained from the xml_load() function
  > * list_of_out (list) - list with numbers of images to exclude. If '[]' only the graph will presented
  >> *in first - run user should provide an empty list to check the graph and decide, which potential images should be excluded       
  > * dispaly_plot (bool) - show the graph in the console. Default: False


> Returns:
  > * fig - location and numeration of raw images in the main core of the full image
  > * image_info (DataFrame) - adjusted image_info

  
<br />




Examples:

```
img_info, figure1 = jg.repair_image(image_info=img_info, dispaly_plot = True)

img_info, figure2 = jg.manual_outlires(image_info = img_info, list_of_out = [], dispaly_plot = False)

```

<br />

Output:


<br />

> Befroe repairing ex. 1

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.2.0.0/fig/lack.bmp" alt="drawing" />
</p>


<br />

> After repairing ex. 1

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.2.0.0/fig/lack_filled.bmp" alt="drawing" />
</p>



<br />


<br />

> Befroe repairing ex. 2

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.2.0.0/fig/outlires.bmp" alt="drawing" />
</p>


<br />

> After repairing ex. 2

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.2.0.0/fig/outlires_repaired.bmp" alt="drawing" />
</p>



<br />


##### 7.1.2 Split channels

<br />

Function:

```
split_channels(path_to_images, path_to_save)
```

> Description:\
  This function goes to a directory with raw images obtained from Opera Phoenix and divides them based on image channel number into separate directories.    


> Args:

  > * path_to_images (str) - path to a images
  > * path_to_save (str) - path to save directories with raw images divided by channels 

> Returns:

  > Directories: ch1, ch2, ...


<br />


Example:

```
jg.split_channels(path_to_images = 'Images', path_to_save = '')
```

<br />

Output:


<br />

> Splited directories

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG/v.2.0.0/fig/channel_split.bmp" alt="drawing" />
</p>


<br />




#### 7.2 Image concatenation <a id="concatenation-code"></a>



<br />

Function:

```
image_sequences(image_info)
```

> Description:\
  This function calculates the image queue in the full image core.
  The images in metadata usually are in a different order than the order of the images taken by the software of microscope.
  This function allows the proper images queue, length, and width necessary for images to concatenate into a full image core.
    
    
> Args:

  > * image_info (pd.DataFrame) - list of images with numeration obtained from the xml_load() function and repired by repair_image() / manual_outlires()

> Returns:

  > * image_queue (pd.DataFrame) - image_info with additional raw numeration (queue of images taken by the microscope) of images in the full image core
  > * img_length (int) - length (number of raw images) included in the full image core
  > * img_width (int) - width (number of raw images) included in the full image core


<br />



Function:

```
# Functions for getting info about available cores/threads 

get_number_of_cores()
get_number_of_threads()


image_concatenate(path_to_images, path_to_save, image_queue, metadata, img_length, img_width, overlap, channels, resize = 2, n_proc = 4, par_type = 'processes')
```

> Description:\
  This function is used to create a full microscope image by concatenation raw images in a parallel way.
  The full image core is based on image metadata and raw images occurrence modified by manual_outlires() and repair_image() functions.

    


> Args:

  > * path_to_images (str) - path to raw images
  > * path_to_save (str) - path to save concatenated the full image in *.tiff format
  >> *WARNING! In this function path_to_images / path_to_save should be full path\
  >>  The full path can be obtained using os.getcwd() + 'directory name' joined using os.path.join() eg. full_path = os.path.join(os.getcwd(), 'Images')
  > * image_queue (pd.DataFrame) - data frame with calculated raw images queue from image_sequences() function
  > * metadata (dict) - metadata for the microscope image obtained from xml_load() function
  > * img_length (int) - length (number of raw images) included in the full image core
  > * img_width (int) - width (number of raw images) included in the full image core
  > * overlap (float) - overlap of raw images to their neighbor images' horizontal and vertical axis
  >> *eg. 0.05 <-- 5% overlap
  > * channels (list) - list of channels to create the concatenated full image. The image for every channel will be saved as a separate file. Information about available channels in metadata loaded from xml_load()
  >> *eg. ['ch1','ch2']
  > * resize (int) - resize factor for the full image size (dividing by factor height x width of every raw image)
  > * n_proc (int) - number of processes/threads for the image concatenatenation process conducted. Depends on 'par_type'.
  >> *avaiable number of threads / cores avaiable from get_number_of_cores() / get_number_of_threads()
  > * par_type (str) - parallelization method ['threads', 'processes']. Default: 'processes'
         


> Returns:

  > Image: The full image concatenated of raw single images with given by user concatenation setting saved in *.tiff format in the given directory.

  
<br />





Example 1:

```
image_queue, img_length, img_width = jg.image_sequences(image_info)


# Multiprocessing (cores) - parallelization

n_cor = jg.get_number_of_cores()
n_cor = n_cor - 2

path_to_images os.path.join(os.getcwd(), 'Images')
path_to_save = os.getcwd()
overlap = 0.05
channels = ['ch1', 'ch2']

jg.image_concatenate(path_to_images, path_to_save, image_queue, metadata, img_length, img_width, overlap, channels, resize = 2, n_proc = n_cor, par_type = 'processes'):

```

<br />


Example 2:

```
image_queue, img_length, img_width = jg.image_sequences(image_info)


# Multithreads (threads) - parallelization

n_threads = jg.get_number_of_threads()
n_threads = n_threads - 2

path_to_images os.path.join(os.getcwd(), 'Images')
path_to_save = os.getcwd()
overlap = 0.05
channels = ['ch1', 'ch2']

jg.image_concatenate(path_to_images, path_to_save, image_queue, metadata, img_length, img_width, overlap, channels, resize = 2, n_proc = n_threads, par_type = 'threads'):

```


<br />



#### 7.3 Image adjustment <a id="adjust-code"></a>


<br />


##### 7.3.1 Tiff operation

<br />

Function:

```
load_tiff(path_to_tiff)
```

> Description:\
  This function is used for loading *.tiff files. 
  When the image is not 16-bit, that function will convert it to the 16-bit image. 


> Args:

  > * path_to_tiff (str) - path to *.tiff file   

> Returns:

  > * stack (np.ndarray) - loaded image returned to a variable



<br />

Function:

```
read_tiff_meta(file_path)
```

> Description:\
  This function allows load metadata included in *.tiff file.


> Args:

  > * file_path (str) - path to the *.tiff file

> Returns:

  > * z - z-spacing [µm]
  > * y - resolution in y-axis pixels [µm/px]
  > * x - resolution in y-axis pixels [µm/px]


<br />



Function:

```
resize_tiff(image, metadata = None, height = None, width = None, resize_factor = None)
```

> Description:\
  This function gets previously loaded *.tiff file (3d-array) and resizes each image in the Z-axis.
  
  !WARNING!
       
      You can change only one parameter in each resizing operation.
      This restriction is designed to preserve the biological proportions
      When you set more than one parameter, only the first parameter
      in the queue will be changed in a single resizing operation.
      The queue is set up: first height, second width, and last resize factor.


> Args:

  > * image (np.ndarray) - input *. tiff image (3d-array)
  > * metadata (dict | None) - metadata for the image from xml_load() function. If None the metadata correction is ommited.
  > * height (int) - new height value 
  > * width (int) - new width value
  > * resize_factor (int) - resize factor (dividing original height x width)
       

> Returns:

   > if metadata == None:
            > * resized_image (np.ndarray) - resized image in *.tiff format
        
   > if metadata != None:
            > * resized_image (np.ndarray) - resized image in *.tiff format
            > * res_metadata (dict) -  metadata corrected by the resolution changes

 

<br />


Function:

```
save_tiff(tiff_image, path_to_save, metadata = None)
```


> Description:\
  This function gets previously loaded *.tiff file (3d-array) and saves it.


> Args:

  > * tiff_image (np.ndarray) - input *. tiff image (3d-array)
  > * path_to_save (str) - path to save *.tiff. Required: file name with *.tiff extension
  > * metadata (dict | None) - metadata to the file from xml_load() function or after using resize_tiff()
  >> *if metadata == None, any metadata will not attached to the *.tiff file


> Returns:

  > Saved file under the given path


<br />



Example:

```
tiff_file = jg.load_tiff(path_to_tiff = 'channel_ch1.tiff')


# loading resolution information from *.tiff file
z, y, x, = jg.read_tiff_meta(file_path = 'res.tiff')

print(z)
print(y)
print(x)

#################################################


# loading whole metadata to the image

path_to_inx = 'Images/Index.idx.xml'

_, metadata =  jg.xml_load(path_to_xml = path_to_inx)


# adjust resolution information in metadata (if the tiff file during concatenation was resized; in the function: image_concatenate( ) the value of resize > 1 - not original resolution)

metadata['X_resolution[um/px]'] = x
metadata['Y_resolution[um/px]'] = y


# resize 

resized_tiff, res_metadata = jg.resize_tiff(image = tiff_file, metadata = metadata, height = None, width = None, resize_factor = 2)

# save

jg.save_tiff(tiff_image = jg.resized_tiff, path_to_save = 'resized_tiff.tiff', metadata = res_metadata)

```

<br />



##### 7.3.2 Image operations (z-projection, adjustment, loading, saving, merging)

<br />

Function:

```
display_preview(projection)
```

> Description:\
  This function allows you to quickly preview images.     



> Args:

  > * image (np.ndarray) - input image

> Returns:

  > Image: display inputted image



<br />

Function:

```
z_projection(tiff_object, projection_type = 'avg')
```

> Description:\
  This function conducts Z projection of the stacked (3D array) image, eg. loaded to a variable with load_tiff()


> Args:

  > * tiff_object (np.ndarray) - stacked (3D) image 
  > * projection_type (str) - type of the stacked image projection of Z axis ['avg', 'median', 'min', 'max', 'std']
    

> Returns:

  > * img (np.ndarray) - image projection returned to a variable



<br />



Function:

```
equalizeHist_16bit(image_eq)
```

> Description:\
  This function conducts global histogram equalization on the inputted image.
 

> Args:

  > * image_eq (np.ndarray) - input image


> Returns:

  > * image_eq_16 (np.ndarray) - image after the global histograme equalization adjustment
 

<br />


Function:

```
clahe_16bit(img, kernal = (100, 100))
```


> Description:\
  This function conducts CLAHE algorithm on the inputted image.


> Args:

  > * img (np.ndarray) - input image
  > * kernal (tuple) - the size of the kernel as the field of CLAHE algorithm adjustment through the whole image in the subsequent iterations eg. (100,100)

> Returns:

  > * img (np.ndarray) - image after the CLAHE adjustment


<br />


Function:

```
adjust_img_16bit(img, color = 'gray', max_intensity = 65535, min_intenisty = 0, brightness = 100, contrast = 1, gamma = 1)
```


> Description:\
  This function allows manually adjusting image parameters and returns the adjusted image.


> Args:

  > * img (np.ndarray) - input image
  > * color (str) - color of the image (RGB) ['green', 'blue', 'red', 'yellow', 'magenta', 'cyan']
  > * max_intensity (int) - upper threshold for pixel value. The pixel that exceeds this value will change to the set value
  > * min_intenisty (int) - lower threshold for pixel value. The pixel that is down to this value will change to 0
  > * brightness (int) - value for image brightness [0-200]. Default: 100 (base value)
  > * contrast (float | int) - value for image contrast [0-5]. Default: 1 (base value)
  > * gamma (float | int) - value for image brightness [0-5]. Default: 1 (base value)

> Returns:

  > * img_gamma (np.ndarray) - image after the parameters adjustment


<br />


Function:

```
resize_projection(image, metadata = None, height = None, width = None, resize_factor = None):
```


> Description:\
  This function gets an image and resizes it.

  !WARNING!
       
      You can change only one parameter in each resizing operation.
      This restriction is designed to preserve the biological proportions
      When you set more than one parameter, only the first parameter
      in the queue will be changed in a single resizing operation.
      The queue is set up: first height, second width, and last resize factor.


> Args:

  > * image (np.ndarray) - input image
  > * metadata (dict | None) - metadata for the image from xml_load() function. If None the metadata correction is ommited
  > * height (int) - new height value 
  > * width (int) - new width value
  > * resize_factor (int) - resize factor (dividing original height x width)
       
       

> Returns:

  > * img_gamma (np.ndarray) - image after the parameters adjustment


<br />


Function:

```
save_image(image, path_to_save)
```


> Description:\
  This function gets an image and saves it.


> Args:

  > * image (np.ndarray) - input image
  > * path_to_save (str) - path to save. Required: file name with *.png, *.tiff or *.tif extension
      

> Returns:

  > Saved file under the given path


<br />



Function:

```
load_image(path)
```


> Description:\
  This function allows the load of the image. When the image is not 16-bit, that function will convert it to the 16-bit image.     


> Args:

  > * path (str) - path to the image
  

> Returns:

  > * img (np.ndarray) - 16-bit image loaded to a variable


<br />


Function:


```
merge_images(image_list:list, intensity_factors:list = [])
```


> Description:\
  This function allows the merging of image projections from different channels.   


> Args:

  > * image_list (list(np.ndarray)) - list of images for merging
  >> *all images in the list must be in the same shape and size!!!      
  > * intensity_factors (list(float)) - list of intensity values for every image provided in image_list. Base value for each image should be 1.
  >> *value < 1 decrease intensity 
  >> *value > 1 increase intensity 
  

> Returns:

  > * result (np.ndarray) - image after the merging


<br />




Examples:


```
projection = jg.z_projection(tiff_object = tiff_file, projection_type = 'median')


jg.display_preview(projection)
 

eq_pro = jg.equalizeHist_16bit(projection)


jg.display_preview(eq_pro)


clahe_pro = jg.clahe_16bit(eq_pro, kernal = (100, 100))


jg.display_preview(clahe_pro)


adj_image = jg.adjust_img_16bit(clahe_pro, color = 'blue', max_intensity = 65535, min_intenisty = 0, brightness = 100, contrast = 3, gamma = 1)


jg.display_preview(adj_image)


resized = jg.resize_projection(adj_image, metadata = None, height = None, width = None, resize_factor = 2)


jg.display_preview(resized)


jg.save_image(image = resized, path_to_save = 'resized.png')


loaded_image = load_image(path = 'resized.png')



# merging images 

projection_ch1 = jg.load_image(path = 'projection_ch1.png')

projection_ch2 = jg.load_image(path = 'projection_ch2.png')


merged_image = jg.merge_images(image_list = [projection_ch1, projection_ch2], intensity_factors = [1,1])

jg.display_preview(merged_image)


jg.save_image(image = merged_image, path_to_save = 'merged_image.png')

```

<br />


### Have fun JBS