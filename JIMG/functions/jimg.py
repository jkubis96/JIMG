import multiprocessing
import shutil
import os
import numpy as np
import re
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import tifffile as tiff
from joblib import Parallel, delayed
import warnings
import tkinter as tk 
import copy





 #       _  ____   _         _____              _                      
 #      | ||  _ \ (_)       / ____|            | |                    
 #      | || |_) | _   ___ | (___   _   _  ___ | |_  ___  _ __ ___   
 #  _   | ||  _ < | | / _ \ \___ \ | | | |/ __|| __|/ _ \| '_ ` _ \  
 # | |__| || |_) || || (_) |____) || |_| |\__ \| |_|  __/| | | | | | 
 #  \____/ |____/ |_| \___/|_____/  \__, ||___/ \__|\___||_| |_| |_|  
 #                                   __/ |                                   
 #                                  |___/      



warnings.filterwarnings("ignore", category=RuntimeWarning)


############################ <Operation fun> ###################################


def get_number_of_cores():
    
    
    try:

        num_cores = os.cpu_count()
        if num_cores is not None:
            return int(num_cores/2)

        num_cores = multiprocessing.cpu_count()
        return int(num_cores/2)

    except Exception as e:
        print(f"Error while getting the number of cores: {e}")
        return None
    
    

def get_number_of_threads():
    
    
    try:

        num_cores = os.cpu_count()
        if num_cores is not None:
            return int(num_cores)

        num_cores = multiprocessing.cpu_count()
        return int(num_cores)

    except Exception as e:
        print(f"Error while getting the number of cores: {e}")
        return None




def get_screan():
    

    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    root.destroy()
    
    return screen_width, screen_height







def resize_to_screen_tiff(tiff_file, factor = 4):
    
    screen_width, screen_height = get_screan()
    

    screen_width = screen_width*factor
    screen_height = screen_height*factor
    

    h = tiff_file[0].shape[0]
    w = tiff_file[0].shape[1]
    
    
    if screen_width < w:
        h = tiff_file[0].shape[0]
        w = tiff_file[0].shape[1]
        
        ww = int((screen_width/w) * w)
        hh = int((screen_width/w) * h)
        

        tiff_file_new_shape = np.zeros((tiff_file.shape[0], hh, ww),  dtype=np.uint16)

        for n, i in enumerate(tiff_file):
            tiff_file_new_shape[n] = cv2.resize(tiff_file[n], (ww, hh))
        
        
        tiff_file = tiff_file_new_shape
        h = tiff_file[0].shape[0]
        w = tiff_file[0].shape[1]
            
        
        
    if screen_height < h:
        h = tiff_file[0].shape[0]
        w = tiff_file[0].shape[1]
        
        ww = int((screen_height/h) * w)
        hh = int((screen_height/h) * h)

    
    
        tiff_file_new_shape = np.zeros((tiff_file.shape[0], hh, ww),  dtype=np.uint16)

        for n, i in enumerate(tiff_file):
            tiff_file_new_shape[n] = cv2.resize(tiff_file[n], (ww, hh))
            
        tiff_file = tiff_file_new_shape
            
        
    return tiff_file
    

    


def resize_to_screen_img(img_file, factor = 1):
    
    screen_width, screen_height = get_screan()
    

    screen_width = int(screen_width*factor)
    screen_height = int(screen_height*factor)
    

    h = int(img_file.shape[0])
    w = int(img_file.shape[1])
    
    
    if screen_width < w:
        h = img_file.shape[0]
        w = img_file.shape[1]
        
        ww = int((screen_width/w) * w)
        hh = int((screen_width/w) * h)
        

        img_file = cv2.resize(img_file, (ww, hh))
        
        h = img_file.shape[0]
        w = img_file.shape[1]
            
        
        
    if screen_height < h:
        h = img_file.shape[0]
        w = img_file.shape[1]
        
        ww = int((screen_height/h) * w)
        hh = int((screen_height/h) * h)

    
    
        img_file = cv2.resize(img_file, (ww, hh))
        
        
        
    
    return img_file
        
        



############################ <\Operation fun> ##################################



################################ Main code #####################################






def resize_tiff(image, metadata, height = None, width = None, resize_factor = None):
    
    """
    This function gets previously loaded *.tiff file (3d-array) and resizes each image in the Z-axis.
    
    Args:
       
       image (np.ndarray) - input *. tiff image (3d-array)
       metadata (dict | None) - metadata for the image from xml_load() function. If None the metadata correction is ommited.
       height (int) - new height value 
       width (int) - new width value
       resize_factor (int) - resize factor (dividing original height x width)
       
       !WARNING!
       
           You can change only one parameter in each resizing operation.
           This restriction is designed to preserve the biological proportions
           When you set more than one parameter, only the first parameter
           in the queue will be changed in a single resizing operation.
           The queue is set up: first height, second width, and last resize factor.

     

    Returns:
        if metadata == None:
            resized_image (np.ndarray) - resized image in *.tiff format
        
        if metadata != None:
            resized_image (np.ndarray) - resized image in *.tiff format
            res_metadata (dict) -  metadata corrected by the resolution changes
       
    """
        
    try:
        
        if metadata != None:
            res_metadata = copy.deepcopy(metadata)
        else:
            res_metadata = None

        
        if height != None and  width == None:
            h = image.shape[1]
            w = image.shape[2]
            
            wh = int(height/h * w)
            
            
            
            resized_image = np.empty((image.shape[0], height, wh)).astype(np.uint16)
            
            for n in range(image.shape[0]):
                resized_image[n] = cv2.resize(image[n], (wh, height))
                
            if metadata != None:
                res_metadata['X_resolution[um/px]'] = res_metadata['X_resolution[um/px]']*(height/h)
                res_metadata['Y_resolution[um/px]'] = res_metadata['Y_resolution[um/px]']*(height/h)
                
            
           
                
            print('Resized successfully')
            print('Current resolution is ' + str(resized_image.shape[2]) + 'x' + str(resized_image.shape[1]))
            
        elif width != None and height == None:
            h = image.shape[1]
            w = image.shape[2]
            
            wh = int(width/w * h)
            
            
            resized_image = np.empty((image.shape[0], wh, width)).astype(np.uint16)
            
            for n in range(image.shape[0]):
                resized_image[n] = cv2.resize(image[n], (width, wh))
                
            if metadata != None:
                res_metadata['X_resolution[um/px]'] = res_metadata['X_resolution[um/px]']*(width/w)
                res_metadata['Y_resolution[um/px]'] = res_metadata['Y_resolution[um/px]']*(width/w)
                
            
            print('Resized successfully')
            print('Current resolution is ' + str(resized_image.shape[2]) + 'x' + str(resized_image.shape[1]))
            
        elif width == None and height == None and resize_factor != None:
            h = image.shape[1]
            w = image.shape[2]
            
            wh = int(w / resize_factor)
            hw = int(h / resize_factor)
            
            
            resized_image = np.empty((image.shape[0], hw, wh)).astype(np.uint16)
            
            for n in range(image.shape[0]):
                resized_image[n] = cv2.resize(image[n], (wh, hw))
            
            if metadata != None:

                res_metadata['X_resolution[um/px]'] = res_metadata['X_resolution[um/px]']/resize_factor
                res_metadata['Y_resolution[um/px]'] = res_metadata['Y_resolution[um/px]']/resize_factor
               
               
            print('Resized successfully')
            print('Current resolution is ' + str(resized_image.shape[2]) + 'x' + str(resized_image.shape[1]))
            
        elif width != None and height != None:
            print('Resized with [width x hight] both parameters is not allowed')
            print('Choose one parameter and second will be scaled')
            print('Rescaling on both parameters without scale can modifie biological diversity')
        
        else:
            print('Resized unsuccessfully')
            print('Provided wrong parameters or lack of parameters')
                       
        if metadata != None:
            return resized_image, res_metadata
        else:
            return resized_image
    
    except:
        print("Something went wrong. Check the function input data and try again!")
        
     



def save_image(image, path_to_save):
    
    """
    This function gets an image and saves it.
    
    Args:
       
       image (np.ndarray) - input image
       path_to_save (str) - path to save. Required: file name with *.png, *.tiff or *.tif extension
      
    Returns:
        Saved file under the given path
       
    """
    
    try:
        if len(path_to_save) == 0 or '.png' not in path_to_save or '.tiff' not in path_to_save or '.tif' not in path_to_save:
            print('\nThe path is not provided or the file extension is not *.png, *.tiff or *.tif')
        else:
            cv2.imwrite(path_to_save, image)
        
    except:
        print("Something went wrong. Check the function input data and try again!")

        
     
        
     
def save_tiff(tiff_image:np.ndarray, path_to_save:str = '', metadata = None):
    
    """
    This function gets previously loaded *.tiff file (3d-array) and saves it.
    
    Args:
       
       tiff_image (np.ndarray) - input *. tiff image (3d-array)
       path_to_save (str) - path to save *.tiff. Required: file name with *.tiff extension
       metadata (dict | None) - metadata to the file from xml_load() function or after using resize_tiff()
           * if metadata == None, any metadata will not attached to the *.tiff file

    Returns:
        Saved file under the given path
       
    """
    
    try:
    
        if len(path_to_save) == 0 or '.tiff' not in path_to_save:
            print('\nThe path is not provided or the file extension is not *.tiff')
        else:
            if metadata == None:
                tiff.imwrite(str(path_to_save), tiff_image,
                                imagej=True) 
            else:
                try:
                    tiff.imwrite(str(path_to_save), tiff_image,
                                    imagej=True,
                                    resolution=(metadata['X_resolution[um/px]'], metadata['Y_resolution[um/px]']),
                                    metadata={'spacing': metadata['z_spacing'], 
                                              'unit': 'um', 
                                              'axes' : 'ZYX', 
                                              'PhysicalSizeX': metadata['X_resolution[um/px]'],
                                              'PhysicalSizeXUnit': 'um',
                                              'PhysicalSizeY': metadata['Y_resolution[um/px]'],
                                              'PhysicalSizeYUnit': 'um',
                                              'magnification[x]': metadata['magnification[x]'][0]}) 
                except:
                    
                    if not isinstance(metadata, dict):
                        tiff.imwrite(str(path_to_save), tiff_image,
                                    imagej= True) 
                    else:
                        tiff.imwrite(str(path_to_save), tiff_image,
                                    imagej= True,
                                    metadata = metadata) 
           
                    
    except:
        print("Something went wrong. Check the function input data and try again!")

    

     
def resize_projection(image, metadata = None, height = None, width = None, resize_factor = None):
    
    """
    This function gets an image and resizes it.
    
    Args:
       
       image (np.ndarray) - input image
       metadata (dict | None) - metadata for the image from xml_load() function. If None the metadata correction is ommited
       height (int) - new height value 
       width (int) - new width value
       resize_factor (int) - resize factor (dividing original height x width)
       
       !WARNING!
       
           You can change only one parameter in each resizing operation.
           This restriction is designed to preserve the biological proportions
           When you set more than one parameter, only the first parameter
           in the queue will be changed in a single resizing operation.
           The queue is set up: first height, second width, and last resize factor.

     

    Returns:
        if metadata == None:
            Image: Resized image 
        
        if metadata != None:
            Image: Resized image 
            Metadata: Metadata corrected by the resolution changes.
       
    """
    
    try:
        
        cmet = None
        if height != None and  width == None:
            h = image.shape[0]
            w = image.shape[1]
            
            wh = int(height/h * w)
            
            
            image = cv2.resize(image, (wh, height))
            if metadata != None:
                
                cmet = copy.deepcopy(metadata)
                cmet['X_resolution[um/px]'] = cmet['X_resolution[um/px]']*(height/h)
                cmet['Y_resolution[um/px]'] = cmet['Y_resolution[um/px]']*(height/h)
                
            print('Resized successfully')
            print('Current resolution is ' + str(image.shape[1]) + 'x' + str(image.shape[0]))
            
        elif width != None and height == None:
            h = image.shape[0]
            w = image.shape[1]
            
            wh = int(width/w * h)
            
            
            image = cv2.resize(image, (width, wh))
            if metadata != None:
                
                cmet = copy.deepcopy(metadata)

                cmet['X_resolution[um/px]'] = cmet['X_resolution[um/px]']*(width/w)
                cmet['Y_resolution[um/px]'] = cmet['Y_resolution[um/px]']*(width/w)
            
            print('Resized successfully')
            print('Current resolution is ' + str(image.shape[1]) + 'x' + str(image.shape[0]))
            
        elif width == None and height == None and resize_factor != None:
            h = image.shape[0]
            w = image.shape[1]
            
            wh = int(w / resize_factor)
            hw = int(h / resize_factor)
            
            image = cv2.resize(image, (wh, hw))
            if metadata != None:
                
                cmet = copy.deepcopy(metadata)
    
                cmet['X_resolution[um/px]'] = cmet['X_resolution[um/px]']/resize_factor
                cmet['Y_resolution[um/px]'] = cmet['Y_resolution[um/px]']/resize_factor
               
            print('Resized successfully')
            print('Current resolution is ' + str(image.shape[1]) + 'x' + str(image.shape[0]))
            
        elif width != None and height != None:
            print('Resized with [width x hight] both parameters is not allowed')
            print('Choose one parameter and second will be scaled')
            print('Rescaling on both parameters without scale can modifie biological diversity')
        
        else:
            print('Resized unsuccessfully')
            print('Provided wrong parameters or lack of parameters')
               
        
        if cmet != None:
            return image, cmet
        else:
            return image

    
    except:
        print("Something went wrong. Check the function input data and try again!")
        
            
    
def split_channels(path_to_images:str, path_to_save:str):
    
    """
    This function goes to a directory with raw images obtained from Opera Phoenix and divides them based on image channel number into separate directories.    
    
    Args:
       
       path_to_images (str) - path to a images
       path_to_save (str) - path to save directories with raw images divided by channels 


    Returns:
        Directories: ch1, ch2, ...
       
    """
    
    try:
    
        channels=os.listdir(path_to_images)
        channels=[re.sub('.*-','',x)  for x in channels if 'tiff' in x]
        channels=[re.sub('sk.*','',x)  for x in channels if 'tiff' in x]
        
        channels = np.unique(channels).tolist()
        
        for ch in channels:
                
            
            if not os.path.exists(os.path.join(path_to_save, str(ch))):
                os.mkdir(os.path.join(path_to_save, str(ch)))
        
      
            
            images_list=os.listdir(path_to_images)
        
            images_list=[x for x in images_list if str(ch) in x]
            images_list = images_list + ['Index.idx.xml']
            
            if not os.path.exists(os.path.join(path_to_save, str(ch))):
                os.mkdir(os.path.join(path_to_save, str(ch)))
                
            for image in images_list:
                shutil.copy(os.path.join(path_to_images,image),os.path.join(os.path.join(path_to_save, str(ch))))
                
    except:
        print("Something went wrong. Check the function input data and try again!")
    
    
    
    
def xml_load(path_to_xml:str):
    
    """
    This function loads the images index file and collects metadata.
    
    Args:
       
       path_to_xml (str) - path to a image metadata


    Returns:
        image_info (pd.DataFrame) - list of images with numeration and coordinates
        metadata (dict) - images information
       
    """
    
    try:
    
        name = []
        x = []
        y = []
        x_res = []
        y_res = []
        channel_num = []
        channel_name = []
        max_intensity = []
        z_spacing = []
        excitation_wavelength = []
        emissio_wavelength = []
        magnification = []
        
        df = {'name':name, 'x':x, 'y':y}
        
        
        with open(path_to_xml) as topo_file:
            topo_file= topo_file.readlines()
            
            for line in topo_file:
                if str('PositionX') in line:
                    df['x'].append(float(re.sub('<PositionX Unit="m">','', re.sub('</PositionX>','',line)).replace(' ', '')))
                elif str('PositionY') in line:
                    df['y'].append(float(re.sub('<PositionY Unit="m">','', re.sub('</PositionY>','',line)).replace(' ', '')))
                elif str('URL') in line:
                    df['name'].append(re.sub('</URL>','', re.sub('<URL>','',line)).replace(' ', ''))
                elif str('<ImageResolutionX Unit="m">') in line and float(re.sub('</ImageResolutionX>','', re.sub('<ImageResolutionX Unit="m">','',line)).replace(' ', '')) not in x_res:
                    x_res.append(float(re.sub('</ImageResolutionX>','', re.sub('<ImageResolutionX Unit="m">','',line)).replace(' ', '')))
                elif str('<ImageResolutionY Unit="m">') in line and float(re.sub('</ImageResolutionY>','', re.sub('<ImageResolutionY Unit="m">','',line)).replace(' ', '')) not in y_res:
                    y_res.append(float(re.sub('</ImageResolutionY>','', re.sub('<ImageResolutionY Unit="m">','',line)).replace(' ', '')))
                elif str('<ChannelID>') in line and re.sub('</ChannelID>','', re.sub('<ChannelID>','',line)).replace(' ', '') not in channel_num:
                    channel_num.append(re.sub('</ChannelID>','', re.sub('<ChannelID>','',line)).replace(' ', ''))
                elif str('<ChannelName>') in line and re.sub('</ChannelName>','', re.sub('<ChannelName>','',line)).replace(' ', '') not in channel_name:
                    channel_name.append(re.sub('</ChannelName>','', re.sub('<ChannelName>','',line)).replace(' ', ''))
                elif str('<MaxIntensity>') in line and int(re.sub('</MaxIntensity>','', re.sub('<MaxIntensity>','',line)).replace(' ', '')) not in max_intensity:
                    max_intensity.append(int(re.sub('</MaxIntensity>','', re.sub('<MaxIntensity>','',line)).replace(' ', '')))
                elif str('<AbsPositionZ Unit="m">') in line and float(re.sub('</AbsPositionZ>','', re.sub('<AbsPositionZ Unit="m">','',line)).replace(' ', '')) not in z_spacing:
                    z_spacing.append(float(re.sub('</AbsPositionZ>','', re.sub('<AbsPositionZ Unit="m">','',line)).replace(' ', '')))
                elif str('<MainExcitationWavelength Unit="nm">') in line and int(re.sub('</MainExcitationWavelength>','', re.sub('<MainExcitationWavelength Unit="nm">','',line)).replace(' ', '')) not in excitation_wavelength:
                    excitation_wavelength.append(int(re.sub('</MainExcitationWavelength>','', re.sub('<MainExcitationWavelength Unit="nm">','',line)).replace(' ', '')))
                elif str('<MainEmissionWavelength Unit="nm">') in line and int(re.sub('</MainEmissionWavelength>','', re.sub('<MainEmissionWavelength Unit="nm">','',line)).replace(' ', '')) not in emissio_wavelength:
                    emissio_wavelength.append(int(re.sub('</MainEmissionWavelength>','', re.sub('<MainEmissionWavelength Unit="nm">','',line)).replace(' ', '')))
                elif str('<ObjectiveMagnification Unit="">') in line and int(re.sub('</ObjectiveMagnification>','', re.sub('<ObjectiveMagnification Unit="">','',line)).replace(' ', '')) not in magnification:
                    magnification.append(int(re.sub('</ObjectiveMagnification>','', re.sub('<ObjectiveMagnification Unit="">','',line)).replace(' ', '')))
           


        image_info = pd.DataFrame(df)
        image_info['name'] = [re.sub('p.*', '', x) for x in image_info['name']]
        
        image_info['y'] = image_info['y']*-1
        
        
        image_info = image_info.drop_duplicates()
        image_info['num'] = range(1,len(image_info['name'])+1)
        
        image_info = image_info.reset_index(drop = True)
        
    
        metadata = {'channel_name':channel_name, 'channel_number':channel_num, 'X_resolution[um/px]': float(1./(x_res[0]*1000000)), 'Y_resolution[um/px]': float(1./(y_res[0]*1000000)), 'max_intensity[um]':max_intensity, 'z_spacing':np.mean(z_spacing), 'excitation_wavelength[nm]':excitation_wavelength, 'emissio_wavelength[nm]':emissio_wavelength, 'magnification[x]':magnification}
        
        
        return image_info, metadata  
    
    except:
        print("Something went wrong. Check the function input data and try again!")
    
    
    


def manual_outlires(image_info:pd.DataFrame, list_of_out:list = [], dispaly_plot = False):
    
    """
    This function is used for repairing microscope image-taking errors.
    The full images (core of the image) obtained from microscopes such as the Opera Pheonix consist of many smaller (raw) images.
    Sometimes microscope takes photos out of the targeted place or misses one photo and then the full photo can not be merged.
    This function allows for checking how the raw images are placed and manually removing some of them from further analysis.
    
    
    Args:
       
       image_info (pd.DataFrame) - list of images with numeration obtained from the xml_load() function
       list_of_out (list) - list with numbers of images to exclude. If '[]' only the graph will presented
           * in first-run user should provide an empty list to check the graph and decide, which potential images should be excluded       
       
        dispaly_plot (bool) - show the graph in the console. Default: False


    Returns:
        fig - location and numeration of raw images in the main core of the full image
        image_info (pd.DataFrame) - adjusted image_info
       
    """
    
    try:
        
        
        
        if len(list_of_out) != 0:
            image_info = image_info[~image_info.index.isin(list_of_out)]
        
        def outlires_image_detect(image_info):
               
            x  = np.interp(image_info['x'], (min(image_info['x']), max(image_info['x'])), (0, 100))
            y = np.interp(image_info['y'], (min(image_info['y']), max(image_info['y'])), (0, 100))
            
            
            fig, ax = plt.subplots(figsize=(30, 30))
            ax.scatter(x , y , s= 700, c='red', alpha=0.7, marker = 's',  edgecolor="none")
            
            ax.set_axis_off()
            ax.invert_yaxis()
    
            for n, i in enumerate(image_info.index):
                ax.annotate(i, (x[n] , y[n] ), xytext=(0, 0), textcoords="offset pixels", ha='center', va='center',
                     fontsize=8, fontweight='bold', color='yellow')
                
                
            physical_size = (16, 14)  
            pixels_per_inch = 300 
            pixel_size = tuple(int(physical_size[i] * pixels_per_inch) for i in (1, 0))
                
            fig.set_size_inches(*physical_size)
            fig.set_dpi(pixels_per_inch)
            
            plt.close(fig)
            
            
            if dispaly_plot == True:
                canvas = FigureCanvas(fig)
                canvas.draw()
                image = np.array(canvas.renderer.buffer_rgba())
        
        
                cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        
                    
                cv2.resizeWindow('Image', *pixel_size)
                cv2.imshow('Image', image)
                
        
                
                cv2.waitKey(0)
                
                
                cv2.destroyAllWindows()
            
            return fig

    
        fig = outlires_image_detect(image_info)
        
        return image_info, fig
    
    except:
        print("Something went wrong. Check the function input data and try again!")
    


    

def repair_image(image_info:pd.DataFrame, dispaly_plot = True):
     
    """
    This function is used for repairing microscope image-taking errors.
    The full images (core of the image) obtained from microscopes such as the Opera Pheonix consist of many smaller (raw) images.
    Sometimes microscope takes photos out of the targeted place or misses one photo and then the full photo can not be merged.
    This function allows for automatic repair of the core of the full image.
    If the core is still not appropriate use manual repair by manual_outlires() function.
    
    
    Args:
       
       image_info (pd.DataFrame) - list of images with numeration obtained from the xml_load() function
       dispaly_plot (bool) - show the graph in the console. Default: True


    Returns:
       fig - location and numeration of raw images in the main core of the full image
       image_info (pd.DataFrame) - adjusted image_info
       
    """
   
    try:
   
        x = np.unique(image_info['x'])
        
        y = np.unique(image_info['y'])
        
        
        
        #
        x_n = []
        for ix in x:
           x_n.append(len(image_info['x'][image_info['x'] == ix]))
           
        mfqx = [abs(xv) for xv in x_n]
           
        mfqx = max(set(mfqx),key = mfqx.count)
        
           
        df_x = pd.DataFrame({'x':x, 'n': x_n})
        #
        
        
        df_x = df_x.sort_values('x').reset_index(drop = True)
        
        
        
        avg_dist_x = []
        for nx in df_x['x'].index:
            if nx != max(df_x['x'].index):
                avg_dist_x.append(abs(float(df_x['x'][nx]) - float(df_x['x'][nx+1])))
                
            elif nx == max(df_x['x'].index):
                avg_dist_x.append(abs(float(df_x['x'][nx-1]) - float(df_x['x'][nx])))
        
                   
        avg_dist_x_median = np.median(avg_dist_x)
        
        
        df_x['dist'] = avg_dist_x
        
        
        #
        y_n = []
        for iy in y:
           y_n.append(len(image_info['y'][image_info['y'] == iy]))
        
        
        mfqy = [abs(xy) for xy in y_n]
           
        mfqy = max(set(mfqy),key = mfqy.count)
        
        
        df_y = pd.DataFrame({'y':y, 'n': y_n})
        
        
              
        df_y = df_y.sort_values('y').reset_index(drop = True)
        
        
        
        avg_dist_y = []
        for nx in df_y['y'].index:
            if nx != max(df_y['y'].index):
                avg_dist_y.append(abs(float(df_y['y'][nx]) - float(df_y['y'][nx+1])))
                
            elif nx == max(df_y['y'].index):
                avg_dist_y.append(abs(float(df_y['y'][nx-1]) - float(df_y['y'][nx])))
        
                   
        avg_dist_y_median = np.median(avg_dist_y)
        
        
        df_y['dist'] = avg_dist_y
        
        #
        df_x_un = df_x        
        df_y_un = df_y
        
        df_x = df_x[(df_x['n'] < mfqx) | (df_x['n'] > mfqx) & (df_x['dist'] < avg_dist_x_median*1.5)]
        
        df_y = df_y[(df_y['n'] < mfqy) | (df_y['n'] > mfqy) & (df_y['dist'] < avg_dist_y_median*1.5)]
        
        
     
   
        b = 0
        
        image_info['XY'] = image_info['x'].astype(str) + image_info['y'].astype(str)
       
                
        
        for xi in df_x_un['x']:
            for yi in df_y['y']:
                if str(str(xi)+str(yi)) not in list(image_info['XY']):
                    b += 1
                    new_row = {'name' : 'blank' + str(b), 'x': float(xi) , 'y': float(yi), 'num': 'NULL'}
                    image_info = pd.concat([image_info, pd.DataFrame([new_row])], ignore_index=True)
        
        
        
        image_info['XY'] = image_info['x'].astype(str) + image_info['y'].astype(str)
        
        for yi in df_y_un['y']:
            for xi in df_x['x']:
                if str(str(xi)+str(yi)) not in list(image_info['XY']):
                    b += 1
                    new_row = {'name' : 'blank' + str(b), 'x': float(xi) , 'y': float(yi), 'num': 'NULL'}
                    image_info = pd.concat([image_info, pd.DataFrame([new_row])], ignore_index=True)
        
        

        
        
        
        def outlires_image_detect(image_info):
               
            x  = np.interp(image_info['x'], (min(image_info['x']), max(image_info['x'])), (0, 100))
            y = np.interp(image_info['y'], (min(image_info['y']), max(image_info['y'])), (0, 100))
            
            
            fig, ax = plt.subplots(figsize=(30, 30))
            ax.scatter(x , y , s= 700, c='red', alpha=0.7, marker = 's',  edgecolor="none")
            
            ax.set_axis_off()
            ax.invert_yaxis()
    
    
            
            for n, i in enumerate(image_info.index):
                ax.annotate(i, (x[n] , y[n] ), xytext=(0, 0), textcoords="offset pixels", ha='center', va='center',
                     fontsize=8, fontweight='bold', color='yellow')
                
                
            physical_size = (16, 14)  
            pixels_per_inch = 300  
            pixel_size = tuple(int(physical_size[i] * pixels_per_inch) for i in (1, 0))
                
            fig.set_size_inches(*physical_size)
            fig.set_dpi(pixels_per_inch)
            
            plt.close(fig)
            
            
            if dispaly_plot == True:
                canvas = FigureCanvas(fig)
                canvas.draw()
                image = np.array(canvas.renderer.buffer_rgba())
        
        
                cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        
                    
                cv2.resizeWindow('Image', *pixel_size)
                cv2.imshow('Image', image)
                
        
                cv2.waitKey(0)
                
                cv2.destroyAllWindows()
                
            
            return fig
    
        fig = outlires_image_detect(image_info)
        image_info = image_info.drop(columns=['XY'])
        
        return image_info, fig
    
    except:
        print("Something went wrong. Check the function input data and try again!")


    
  
    
def image_sequences(image_info:pd.DataFrame):
    
    """
    This function calculates the image queue in the full image core.
    The images in metadata usually are in a different order than the order of the images taken by the software of microscope.
    This function allows the proper images queue, length, and width necessary for images to concatenate into a full image core.
    
    
    Args:
       
        image_info (pd.DataFrame) - list of images with numeration obtained from the xml_load() function and repaired by repair_image() / manual_outlires()

    Returns:
        image_queue (pd.DataFrame) - image_info with additional raw numeration (queue of images taken by the microscope) of images in the full image core
        img_length (int) - length (number of raw images) included in the full image core
        img_width (int) - width (number of raw images) included in the full image core

       
    """
    
    try:
    
        y = list(np.unique(image_info['y']))
        y.sort()
    
    
        queue_images=[]
    
        for table in y:
            tmp = image_info[image_info['y'] == table]
            tmp = tmp.sort_values(by=['x'])
            queue_images = queue_images + list(tmp['name'])
    
        image_queue=pd.DataFrame()
        image_queue['queue']=queue_images
        image_queue['image_num']=range(1,len(image_queue['queue'])+1)
        
        img_length = len(list(np.unique(image_info['y'])))
        img_width = len(list(np.unique(image_info['x'])))
        
        return image_queue, img_length, img_width
    
    except:
        print("Something went wrong. Check the function input data and try again!")




def image_concatenate(path_to_images:str, path_to_save:str, imgs:pd.DataFrame, metadata, img_length:int, img_width:int, overlap:int, channels:list, resize:int = 2, n_proc:int = 4, par_type = 'processes'):
     
    """
     This function is used to create a full microscope image by concatenation raw images in a parallel way.
     The full image core is based on image metadata and raw images occurrence modified by manual_outlires() and repair_image() functions.

      
      
      Args:
          
         path_to_images (str) - path to raw images
         path_to_save (str) - path to save concatenated the full image in *.tiff format
         
             * WARNING! In this function path_to_images / path_to_save should be full path
               The full path can be obtained using os.getcwd() + 'directory name' joined 
               using os.path.join() eg. full_path = os.path.join(os.getcwd(), 'Images')
         
         image_queue (pd.DataFrame) - data frame with calculated raw images queue from image_sequences() function
         metadata (dict) - metadata for the microscope image obtained from xml_load() function
         img_length (int) - length (number of raw images) included in the full image core
         img_width (int) - width (number of raw images) included in the full image core
         overlap (float) - overlap of raw images to their neighbor images' horizontal and vertical axis
          * eg. 0.05 <-- 5% overlap
         channels (list) - list of channels to create the concatenated full image. The image for every channel will be saved as a separate file. Information about available channels in metadata loaded from xml_load()
          * eg. ['ch1','ch2']
         resize (int) - resize factor for the full image size (dividing by factor height x width of every raw image)
         n_proc (int) - number of processes/threads for the image concatenatenation process conducted. Depends on 'par_type'.
          * avaiable number of threads / cores avaiable from get_number_of_cores() / get_number_of_threads()
         par_type (str) - parallelization method ['threads', 'processes']. Default: 'processes'
         
    
      Returns:
          Image: The full image concatenated of raw single images with given by user concatenation setting saved in *.tiff format in the given directory.
         
    """

    init_path = os.getcwd()
    
    res_metadata = copy.deepcopy(metadata)
 
    
    try:
    
        os.chdir(path_to_images)         
            
        
        def par_1(q, img_width, imgs, black_img, st, overlap, resize, tmp_img):
            img_width_copy = copy.deepcopy(img_width)
            imgs_copy = copy.deepcopy(imgs)
            black_img_copy = copy.deepcopy(black_img)
            st_copy = copy.deepcopy(st)
            overlap_copy = copy.deepcopy(overlap)
            resize_copy = copy.deepcopy(resize)
            tmp_img_copy = copy.deepcopy(tmp_img)


            stop = img_width_copy * (q + 1)
            start = img_width_copy * q
            tmp = imgs_copy['queue'][start:stop]
    
            list_p = []
            for t in tmp:
                if 'blank' in t:
                    list_p.append(str(t))
                else:
                    list_p.append(
                        str([f for f in tmp_img_copy if str(re.sub('\n', '', str(t)) + 'p') in f and str('p' + st_copy) in f][0]))
           
            
            data = []
            for img in list_p:
                if os.path.exists(img):
                    data.append(cv2.imread(img, cv2.IMREAD_ANYDEPTH))
                else:
                    data.append(black_img_copy)
    
            row, col = data[0].shape
            for n, i in enumerate(data):
                if resize_copy > 1:
                    original_height, original_width = data[n].shape[:2]
    
                    new_width = original_width // resize_copy
                    new_height = original_height // resize_copy
                    if overlap_copy > 0:
                        data[n] = cv2.resize(data[n][:, int(col * overlap_copy / 2):-int(col * overlap_copy / 2)], (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                    else:
                        data[n] = cv2.resize(data[n], (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                    
                else:
                    if overlap_copy > 0:
                        data[n] = data[n][:, int(col * overlap_copy / 2):-int(col * overlap_copy / 2)]
    
            data = np.concatenate(data, axis=1)
            
            
            if overlap_copy > 0:
                row, col = data.shape  
                data = data[int(row*overlap_copy/2):-int(row*overlap_copy/2), :]
    
            return q, data
        
        
        
        images_list=os.listdir(os.getcwd())
        
    
        deep = np.unique([re.sub('-.*','', re.sub('.*p', '', n)) for n in images_list if '.tiff' in n])
               
        
        for ch in channels:
            
            images_tmp = []
         
            tmp_img = [i for i in images_list if ch in i]
            
            black_img = cv2.imread(tmp_img[0], cv2.IMREAD_ANYDEPTH)
            black_img.fill(0) 
            
            
            for st in deep:
                
                with Parallel(n_jobs=n_proc, prefer=par_type) as parallel:
                    data = parallel(delayed(par_1)(q, img_width, imgs, black_img, st, overlap, resize, tmp_img) 
                                    for q in range(0, img_length))
                               
                data.sort(key=lambda x: x[0])

                data = [result[1] for result in data]
                
                data = np.concatenate(data, axis = 0)
                
                images_tmp.append(data)
                
                del data
        
    
        
            data = np.stack(images_tmp)
            
            
            res_metadata['X_resolution[um/px]'] = res_metadata['X_resolution[um/px]']/resize
            res_metadata['Y_resolution[um/px]'] = res_metadata['Y_resolution[um/px]']/resize
            
            os.chdir(path_to_save)    
            tiff.imwrite('channel_' + str(ch) + '.tiff', data,
                           imagej=True,
                           resolution=(res_metadata['X_resolution[um/px]'], res_metadata['Y_resolution[um/px]']),
                           metadata={'spacing': res_metadata['z_spacing'], 
                                     'unit': 'um', 
                                     'axes' : 'ZYX', 
                                     'PhysicalSizeX': res_metadata['X_resolution[um/px]'],
                                     'PhysicalSizeXUnit': 'um',
                                     'PhysicalSizeY': res_metadata['Y_resolution[um/px]'],
                                     'PhysicalSizeYUnit': 'um',
                                     'Channel': ch,
                                     'magnification[x]': res_metadata['magnification[x]'][0]}) 
                
            
            
            del data
            
            from joblib.externals.loky import get_reusable_executor
            get_reusable_executor().shutdown(wait=True)
            
            os.chdir(path_to_images)    
            
            
            
        os.chdir(init_path)  
    
    except:
        os.chdir(init_path)   
        print("Something went wrong. Check the function input data and try again! \nCheck that the number of channels you want to assemble matches the number of data channels!")








def load_tiff(path_to_tiff:str):
    
    """
      This function is used for loading *.tiff files. 
      When the image is not 16-bit, that function will convert it to the 16-bit image.     
     
      Args:
         
         path_to_tiff (str) - path to *.tiff file    
    
      Returns:
          stack (np.ndarray) - loaded image returned to a variable
         
    """
    
    try:
        stack = tiff.imread(path_to_tiff)
        
        
        if stack.dtype != 'uint16':
            
            stack = stack.astype(np.uint16)
            
            for n, _ in enumerate(stack):
    
                min_val = np.min(stack[n])
                max_val = np.max(stack[n])
                
                stack[n] = ((stack[n] - min_val) / (max_val - min_val) * 65535).astype(np.uint16)
                
                stack[n] = np.clip(stack[n], 0, 65535)
        
    
    
        return stack
    
    except:
        print("Something went wrong. Check the function input data and try again!")






    



def z_projection(tiff_object, projection_type = 'avg'):
    
    """
      This function conducts Z projection of the stacked (3D array) image, eg. loaded to a variable with load_tiff()
     
      Args:
         
         tiff_object (np.ndarray) - stacked (3D) image 
         projection_type (str) - type of the stacked image projection of Z axis ['avg', 'median', 'min', 'max', 'std']
    
      Returns:
          
          img (np.ndarray) - image projection returned to a variable
         
    """
    try:
    
        if projection_type == 'avg':
            img = np.mean(tiff_object, axis=0).astype(np.uint16) 
        elif projection_type == 'max':
            img = np.max(tiff_object, axis=0).astype(np.uint16)  
        elif projection_type == 'min': 
            img = np.min(tiff_object, axis=0).astype(np.uint16)
        elif projection_type == 'std':
            img = np.std(tiff_object, axis=0).astype(np.uint16) 
        elif projection_type == 'median':
            img = np.median(tiff_object, axis=0).astype(np.uint16)    
            
        return img
    
    except:
        print("Something went wrong. Check the function input data and try again!")
 





def clahe_16bit(img, kernal = (100, 100)):
    
    """
      This function conducts CLAHE algorithm on the inputted image.
     
      Args:
         
         img (np.ndarray) - input image
    
      Returns:
         img (np.ndarray) - image after the CLAHE adjustment
         kernal (tuple) - the size of the kernel as the field of CLAHE algorithm adjustment through the whole image in the subsequent iterations eg. (100,100)
         
    """
    
    try:
    
        img = img.copy()
        
        img8bit = img.copy()
        
    
        min_val = np.min(img8bit)
        max_val = np.max(img8bit)
        
        img8bit = ((img8bit - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    
        
        clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=kernal)
        img8bit = clahe.apply(img8bit)
        
        
        img8bit = img8bit/255
        
            
        img = img*img8bit
    
        
        min_val = np.min(img)
        max_val = np.max(img)
        
        img = ((img - min_val) / (max_val - min_val) * 65535).astype(np.uint16)
        
    
    
        return img
    
    except:
        print("Something went wrong. Check the function input data and try again!")






def equalizeHist_16bit(image_eq):
    
 
    """
      This function conducts global histogram equalization on the inputted image.
     
      Args:
         
         image_eq (np.ndarray) - input image
    
      Returns:
          image_eq_16 (np.ndarray) - image after the global histograme equalization adjustment
         
    """
    
    try:

        image = image_eq.copy()
    
        min_val = np.min(image)
        max_val = np.max(image)
        
        scaled_image = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        
        
        eq_image = cv2.equalizeHist(scaled_image)
        
        
        eq_image_bin = (eq_image/255)
        
        
        image_eq_16 = image * eq_image_bin
        image_eq_16 = (image_eq_16/ np.max(image_eq_16)) * 65535
        image_eq_16[image_eq_16 > (65535/2)]  += (65535 - np.max(image_eq_16))
        image_eq_16 = image_eq_16.astype(np.uint16)
        
        
        return image_eq_16
    
         
    except:
        print("Something went wrong. Check the function input data and try again!")







def adjust_img_16bit(img, color = 'gray', max_intensity:int = 65535, min_intenisty:int = 0, brightness:int = 1000, contrast = 1.0, gamma = 1.0):

    """
      This function allows manually adjusting image parameters and returns the adjusted image.
     
      Args:
         
         img (np.ndarray) - input image
         color (str) - color of the image (RGB) ['green', 'blue', 'red', 'yellow', 'magenta', 'cyan']
         max_intensity (int) - upper threshold for pixel value. The pixel that exceeds this value will change to the set value
         min_intenisty (int) - lower threshold for pixel value. The pixel that is down to this value will change to 0
         brightness (int) - value for image brightness [900-2000]. Default: 1000 (base value)
         contrast (float | int) - value for image contrast [0-5]. Default: 1 (base value)
         gamma (float | int) - value for image brightness [0-5]. Default: 1 (base value)

    
      Returns:
          img_gamma (np.ndarray) - image after the parameters adjustment
         
    """
    
    try:
    
        img = img.copy()

        img = img.astype(np.uint64)  
        
        img = np.clip(img, 0, 65535)
        
        
        #brightness
        if brightness != 1000:
            factor = -1000 + brightness
            side = factor/abs(factor)
            img[img > 0] = img[img > 0] + ((img[img > 0]*abs(factor)/100)*side)
            img = np.clip(img, 0, 65535)

        
       
        #contrast
        if contrast != 1:
            img = ((img - np.mean(img)) * contrast) + np.mean(img)
            img = np.clip(img, 0, 65535)
            
            
        #gamma
        if gamma != 1:
        
            max_val = np.max(img)
            
            image_array = img.copy()/max_val
            
            image_array = np.clip(image_array , 0, 1)
           
            corrected_array = image_array ** (1/gamma)
            
            img = corrected_array*max_val
           
            del image_array, corrected_array
            
            img = np.clip(img, 0, 65535)

            

        
            
        img = ((img/np.max(img))*65535).astype(np.uint16)  
        
        
        
        # max intenisty
        if max_intensity != 65535:
            img[img >= max_intensity] = 65535
        
        
        # min intenisty
        if min_intenisty != 0:
            img[img <= min_intenisty] = 0


        img_gamma = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint16)


        if color == 'green':
            img_gamma[:,:,1] = img
        
        
        elif color == 'red':
            img_gamma[:,:,2] = img
        
            
        elif color == 'blue':
            img_gamma[:,:,0] = img
        
            
        elif color == 'magenta':
            img_gamma[:,:,0] = img
            img_gamma[:,:,2] = img
            
        elif color == 'yellow':
            img_gamma[:,:,1] = img
            img_gamma[:,:,2] = img
        
        elif color == 'cyan':
            img_gamma[:,:,0] = img
            img_gamma[:,:,1] = img
         
        elif color == 'gray':
            img_gamma[:,:,0] = img
            img_gamma[:,:,1] = img
            img_gamma[:,:,2] = img
            
                 
                 
                 
        return img_gamma
    
    except:
        print("Something went wrong. Check the function input data and try again!")



            




def merge_images(image_list:list, intensity_factors:list = []):
    
    """
      This function allows the merging of image projections from different channels.
     
      Args:
         
        image_list (list(np.ndarray)) - list of images for merging
          * all images in the list must be in the same shape and size!!!
          
        intensity_factors (list(float)) - list of intensity values for every image provided in image_list. Base value for each image should be 1.
          * value < 1 decrease intensity 
          * value > 1 increase intensity 
             

      Returns:
         result (np.ndarray) - image after the merging
         
    """
    
    
    try:
        
        result = None
        
        if len(intensity_factors) == 0:
             
            intensity_factors = []
            for bt in range(len(image_list)):
                intensity_factors.append(1)
       
        
        for i, image in enumerate(image_list):
            if result is None:
                result = image.astype(np.uint64) * intensity_factors[i]
            else:
                result = cv2.addWeighted(result, 1, image.astype(np.uint64) * intensity_factors[i], 1, 0)
        
        result = np.clip(result, 0, 65535)
        
        result = result.astype(np.uint16)
        
        return result
    
    
    except:
        print("Something went wrong. Check the function input data and try again!")








def load_image(path):
    
    """
    This function allows the load of the image. When the image is not 16-bit, that function will convert it to the 16-bit image.     
      
    Args:
         
         path (str) - path to the image
    
    Returns:
        img (np.ndarray) - 16-bit image loaded to a variable
         
    """
    
    try:
    
        img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        
        # convert to 16 bit (the function are working on 16 bit images!)
        if img.dtype != 'uint16':
            
            min_val = np.min(img)
            max_val = np.max(img)
            
            img = ((img - min_val) / (max_val - min_val) * 65535).astype(np.uint16)
            
            img = np.clip(img, 0, 65535)
                
            
        return img
    
    except:
        print("Something went wrong. Check the function input data and try again!")







def read_tiff_meta(file_path):

    """
      This function allows load metadata included in *.tiff file.
     
      Args:
         
         file_path (str) - path to the *.tiff file

    
      Returns:
          List: [z, y, x]
          
          z - z-spacing [µm]
          y - resolution in y-axis pixels [µm/px]
          x - resolution in y-axis pixels [µm/px]
         
    """
    
    try:

        def _xy_voxel_size(tags, key):
            assert key in ['XResolution', 'YResolution']
            if key in tags:
                num_pixels, units = tags[key].value
                return units / num_pixels
            # return default
            return 1.
    
        with tiff.TiffFile(file_path) as t:
            image_metadata = t.imagej_metadata
            if image_metadata is not None:
                z = image_metadata.get('spacing', 1.)
            else:
                # default voxel size
                z = 1.
    
            tags = t.pages[0].tags
            # parse X, Y resolution
            y = _xy_voxel_size(tags, 'YResolution')
            x = _xy_voxel_size(tags, 'XResolution')
            # return voxel size
            
            if z == 1 and y == 1 and x == 1:
                return [0,0,0]
            else:
                return [z, 1./y, 1./x]

    except:
        print("Something went wrong. Check the function input data and try again!")





def rotate_image(img, rotate:int):
    
    """
      This function allows for angular rotation of the image.
     
      Args:
         
         img (np.ndarray) - image for rotation
         rotate (int) - degree of rotation [avaiable: 90, 180, 270]

    
      Returns:
          
          Rotated image (np.ndarray)
         
    """
    
    try:
        
        if rotate == 0:
            r = 0
        elif rotate == 90:
            r = -1
        elif rotate == 180:
            r = 2
        elif rotate == 180:
            r = 2
        elif rotate == 270:
            r = 1
        else:
            print("Wrong argument - rotate!")
            return None

        
        img = img.copy()
        
        img = np.rot90(img.copy(), k=r)
    
        return img
    
    except:
        print("Something went wrong. Check the function input data and try again!")




def mirror_image(img, rotate:str):
    
    """
      This function allows for mirroring of the image.

      Args:
          img (np.ndarray) - image for mirroring
          rotate (str) - type of mirroring to apply
          
          Available options are:
             'h'  - horizontal mirroring
             'v'  - vertical mirroring
             'hv' - both horizontal and vertical mirroring
    
      Returns:
          
          Mirrored image (np.ndarray)
         
    """
    
    try:
        
        if rotate == 'h':
            img = np.fliplr(img.copy())
        elif rotate == 'v':
            img = np.flipud(img.copy())
        elif rotate == 'hv':
            img = np.flipud(np.fliplr(img.copy()))
        else:
            print("Wrong argument - rotate!")
            return None

        return img
    
    except:
        print("Something went wrong. Check the function input data and try again!")




def display_preview(image):
    
    
    """
    This function allows you to quickly preview images.     
    
    
      Args:
         
         image (np.ndarray) - input image
    
      Returns:
          Image: display inputted image
         
    """
    
    try:
        
        res_sc = resize_to_screen_img(image.copy(), factor=0.8)
        
        cv2.imshow('Display', res_sc)
        
        
        cv2.waitKey(0) & 0xFF
        
        cv2.destroyAllWindows()
        
    except:
        print("Something went wrong. Check the function input data and try again!")



    

############################### Main code / ####################################


 #       _  ____   _         _____              _                        
 #      | ||  _ \ (_)       / ____|            | |                    
 #      | || |_) | _   ___ | (___   _   _  ___ | |_  ___  _ __ ___   
 #  _   | ||  _ < | | / _ \ \___ \ | | | |/ __|| __|/ _ \| '_ ` _ \  
 # | |__| || |_) || || (_) |____) || |_| |\__ \| |_|  __/| | | | | | 
 #  \____/ |____/ |_| \___/|_____/  \__, ||___/ \__|\___||_| |_| |_|  
 #                                   __/ |                                   
 #                                  |___/      



    