# JIMG - tool and application for high-resolution image management


<br />



<p align="right">
    <img src="https://github.com/jkubis96/Operetta_tool/blob/v.2.0.0/icons/jbs_icon.png?raw=true" alt="drawing" width="250" />
    <img src="https://github.com/jkubis96/Operetta_tool/blob/v.2.0.0/icons/jbi_icon.png?raw=true" alt="drawing" width="250" />
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








<br />

<br />

## Installation <a id="installation"></a>

<br />


#### Python users:

CMD:
```
pip install Operetta-tool
```


<br />


#### For Windows users:

* [Download](https://download944.mediafire.com/txbbi6k01zggKXihNoHPRFSPRtP8CDizS-d13Ue25aC3V3VRQgRJU6_0t1VFo0_DAXz_Qdk3wX-8fNiQ9OUfnEdNei5T6fqKZKQVFC9y5o5uASGBKttYhrXLqwDWV6-BgLq_dnyIB0PMhUeVoQ7QB8IaPF1_6Oo2lJo3r4suVNwuwA8/47v297dlda78zcm/JIMG+v.2.0.exe)


<br />
<br />


## Usage <a id="usage"></a>



### Run application 

<br />

#### Python users:



```
pip install Operetta-tool
```


<br />


#### For Windows users:

Run current version of application.exe file


<br />


<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/v.2.0.0/fig/1%20front.bmp" alt="drawing"/>
</p>


<br />


### Images operations <a id="dz"></a>

<br />


This chapter below will present all operations that the application allowed to conduct on high-resolution images from a microscope (especially High Content Screening such as Opera Phoenix).

<br />


<p align="center">
<img  src="https://github.com/jkubis96/Operetta_tool/raw/v.2.0.0/fig/before_zoom.bmp" alt="drawing" />
</p>

<br />


<span style="color:green">Displayed images can be resized with Window size / Size sliders and also zoomed by pressing and holding the 'Z' button and scrolling by mouse or touchpad in selected part of the image</span>


<br />

<br />


### 1. Start <a id="start"></a>

This runs the options for image operations and management

<br />


<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/v.2.0.0/fig/1.1%20run.bmp" alt="drawing"/>
</p>


<br />

<br />


#### 1.1 Project manager <a id="project-manager"></a>

Options for loading and saving project

<br />


<p align="center">
<img  src="https://github.com/jkubis96/Operetta_tool/blob/v.2.0.0/fig/1.p.png?raw=true" alt="drawing"  />
</p>


<br />


* Load project - loading previous saved project with *.pjn extension

<br />


<p align="center">
<img  src="https://github.com/jkubis96/Operetta_tool/blob/v.2.0.0/fig/1.pl.png?raw=true" alt="drawing" />
</p>

<br />


  * Browse - select path to the project [*.pjm]

 * Load - load project metadata

 * Back - back to the previous application window

<br />



* Save current - save the currently edited project


<br />

<p align="center">
<img  src="https://github.com/jkubis96/Operetta_tool/blob/v.2.0.0/fig/1.ps.png?raw=true" alt="drawing" />
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
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/v.2.0.0/fig/1.1.1%20load%20metadata.bmp" alt="drawing" />
</p>


<br />


* Browse path - select path to the metadata - Index.inx.xml 

<br />


* Load metadata - loading metadata to the application

<br />


* Display core - displaying the core of the raw images by their coordinates and repairing lacking elements of the image (full image must be rectangular)

<br />

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/v.2.0.0/fig/1.1.1.1%20display%20core.bmp" alt="drawing" />
</p>


<br />


* Core reduce - manually adjusting the images included in the full image core (full image must be rectangular)



<br />

<br />


> #### 1.2.1 Core reduce - options


<br />

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/v.2.0.0/fig/1.1.1.2%20cores%20reducing.bmp" alt="drawing" />
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
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/v.2.0.0/fig/1.1.2%20concatenate%20images.bmp" alt="drawing" />
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
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/v.2.0.0/fig/1.1.3.1%20tiff%20load%20-%20z%20projection.bmp" alt="drawing" />
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
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/v.2.0.0/fig/1.1.3.2%20slices%20selection%20-%20z%20projection.bmp" alt="drawing" />
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
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/v.2.0.0/fig/1.1.3.3%20z%20projection.bmp" alt="drawing" />
</p>


<br />


> > Channel 2

<br />


<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/v.2.0.0/fig/1.1.3.3%20z%20projection%202.bmp" alt="drawing" />
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
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/v.2.0.0/fig/1.1.4%20manager.bmp" alt="drawing" />
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
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/v.2.0.0/fig/1.1.4.1%20manager%20-%20resize.bmp" alt="drawing" />
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


* Save - save the selected image from the manager

<br />

<br />


> #### 1.5.2 Save options


<br />


<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/v.2.0.0/fig/1.1.4.2%20manager%20-%20save.bmp" alt="drawing" />
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
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/v.2.0.0/fig/1.1.5.1%20merge%20select.bmp" alt="drawing" />
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
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/v.2.0.0/fig/1.1.4.2%20manager%20-%20save.bmp" alt="drawing" />
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
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/v.2.0.0/fig/1.1.6%20scalebar.bmp" alt="drawing" />
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
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/v.2.0.0/fig/1.1.6.1%20add%20scale.bmp" alt="drawing" />
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
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/v.2.0.0/fig/1.1.7%20annotate.bmp" alt="drawing" />
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
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/v.2.0.0/fig/1.1.7.1%20annotate%20-%20annotate.bmp" alt="drawing" />
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


> Saved results of annotation 

<br />


<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/v.2.0.0/fig/1.1.7.2%20annotate%20-%20results.bmp" alt="drawing" />
</p>



<br />


>> _annotated_image[n] - image with drawn annotation


<br />


>> _annotated_image[n]_lines - raw annotation lines


<br />


>> _annotated_image[n]_mask - raw mask for drawn annotation


<br />


> * Close - close the annotation window and back to the previous application window



<br />

<br />


#### 1.9 Annotate raw <a id="annotate-raw"></a>

Options for raw images annotation

<br />


<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/v.2.0.0/fig/1.1.8%20annotation%20raw.bmp" alt="drawing" />
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
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/v.2.0.0/fig/1.1.8.1%20annotation%20raw%20-%20select%20images.bmp" alt="drawing" />
</p>


<br />

> Based on the grid and numeration, the raw images are selected for annotation



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
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/v.2.0.0/fig/1.1.8.1.1%20annotation%20raw%20-%20raw.bmp" alt="drawing" />
</p>


<br />




> * Adjust - run panel for raw image projection adjustment


<br />

> This panel is the same as in 1.4.2 Z-projection chapter.
> After Apply changes the Close button will save these changes for all current annotated images.


<br />

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/v.2.0.0/fig/1.1.8.1.1.1%20annotation%20raw%20-%20raw%20-%20adjust.bmp" alt="drawing" />
</p>


<br />


> * Annotate - run panel for raw image annotation

<br />

> This panel is the same as in 1.8.1 Annotation options chapter.
> After Apply changes the Close button will save annotation for current image.


<br />

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/v.2.0.0/fig/1.1.8.1.1.2%20annotation%20raw%20-%20raw%20-%20annotate.bmp" alt="drawing" />
</p>



<br />


> * Next - change for the next image in the selected images queue


> If the image will not be annotated, the Next button means, that the user allows to prepare a mask of the whole image (it means that the whole image is taken into consideration in the study)



<br />



> * Previous - change for the previous image in the selected images queue


<br />

 
> * Discard - exclude the image from the analysis



<br />


> * Save - save whole progress of annotation and back to previous application window for saving or select other images

> <span style="color:red">!!! If next, the results will not save with the 'Save selection' button, and the next analysis will be run with the 'Annotate raw' button, the results of the current analysis will be lost !!!</span>


<br />


> * Back -  back to the previous application window 


<br />




* Save selection - save results obtained from 'Annotate raw' analysis


<br />

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/v.2.0.0/fig/1.1.8.1.2%20annotation%20raw%20-%20raw%20-%20save%20results.bmp" alt="drawing" />
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
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/v.2.0.0/fig/1.1.8.1.2.1%20annotation%20raw%20-%20raw%20-%20save%20results%20outside.bmp" alt="drawing" />
</p>


<br />

> It contains directories for each selected and not excluded from the analysis selected image. Numbers are related to the numbers on the grid image map (image with 'grid_' prefix).  Additionally, this directory is included Z-projection without the grid.


<br />


> * Directories with annotated images (directories with 'img_' prefix)


<br />

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/Operetta_tool/v.2.0.0/fig/1.1.8.1.2.2%20annotation%20raw%20-%20raw%20-%20empty_annotation.bmp" alt="drawing" />
</p>


<br />

> It contains a Z-projection of the current image with a given number (annotated_projection), their raw annotation (annotation), masks (mask_8bit, mask_16bit, and mask_binary), and all raw images that belong to the Z axis (slices) that were not excluded in '1.4.1 Z-selection' chapter step of whole image adjustment 



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




