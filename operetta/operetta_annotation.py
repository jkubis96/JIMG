import shutil
import os
import numpy as np
import re
import pandas as pd
import cv2
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image as im
import h5py
from tqdm import tqdm
import tifffile as tiff
from joblib import Parallel, delayed
import math
import gc
import warnings
import tkinter as tk 
from tkinter import ttk, Text
from skimage import io, filters
import pkg_resources
from PIL import ImageFont, ImageDraw, Image
import copy




warnings.filterwarnings("ignore", category=RuntimeWarning)



def resize_tiff(channels:list, metadata, prefix = 'resized' , height = None, width = None, resize_factor = None):
    
    if metadata == None:
        print('Metadata is required. Load metadata')
    
    else:
        
        try:
            
            
            for ch in channels:
                
                res_metadata = copy.deepcopy(metadata)
                
                path_to_tiff = str('channel_' + str(ch) + '.tiff')
                
                if not os.path.exists(path_to_tiff):
                    print('\nImage ' + path_to_tiff +  ' does not exist')
                else:
    
                    image = tiff.imread(path_to_tiff)
                    
                    if height != None and  width == None:
                        h = image.shape[1]
                        w = image.shape[2]
                        
                        wh = int(height/h * w)
                        
                        
                        
                        resized_image = np.empty((image.shape[0], height, wh)).astype(np.uint16)
                        
                        for n in range(image.shape[0]):
                            resized_image[n] = cv2.resize(image[n], (wh, height))
                            
                            
                        res_metadata['X_resolution[m]'][0] = res_metadata['X_resolution[m]'][0]*height/h
                        res_metadata['Y_resolution[m]'][0] = res_metadata['Y_resolution[m]'][0]*height/h
                        
                        
                        tiff.imsave(str(prefix + '_' + path_to_tiff), resized_image,
                                        imagej=True,
                                        resolution=(res_metadata['X_resolution[m]'][0]*1000000, res_metadata['Y_resolution[m]'][0]*1000000),
                                        metadata={'spacing': res_metadata['z_spacing'], 'unit': 'um'}) 
                            
                        print('Resized succesfully')
                        print('Current resolution is ' + str(resized_image.shape[2]) + 'x' + str(resized_image.shape[1]))
                        
                    elif width != None and height == None:
                        h = image.shape[1]
                        w = image.shape[2]
                        
                        wh = int(width/w * h)
                        
                        
                        resized_image = np.empty((image.shape[0], wh, width)).astype(np.uint16)
                        
                        for n in range(image.shape[0]):
                            resized_image[n] = cv2.resize(image[n], (width, wh))
                            
                
                        res_metadata['X_resolution[m]'][0] = res_metadata['X_resolution[m]'][0]*width/w
                        res_metadata['Y_resolution[m]'][0] = res_metadata['Y_resolution[m]'][0]*width/w
                        
                        tiff.imsave(str(prefix + '_' + path_to_tiff), resized_image,
                                        imagej=True,
                                        resolution=(res_metadata['X_resolution[m]'][0]*1000000, res_metadata['Y_resolution[m]'][0]*1000000),
                                        metadata={'spacing': res_metadata['z_spacing'], 'unit': 'um'}) 
                        
                        print('Resized succesfully')
                        print('Current resolution is ' + str(resized_image.shape[2]) + 'x' + str(resized_image.shape[1]))
                        
                    elif width == None and height == None and resize_factor != None:
                        h = image.shape[1]
                        w = image.shape[2]
                        
                        wh = int(w / resize_factor)
                        hw = int(h / resize_factor)
                        
                        
                        resized_image = np.empty((image.shape[0], hw, wh)).astype(np.uint16)
                        
                        for n in range(image.shape[0]):
                            resized_image[n] = cv2.resize(image[n], (wh, hw))
                        
                
                        res_metadata['X_resolution[m]'][0] = res_metadata['X_resolution[m]'][0]/resize_factor
                        res_metadata['Y_resolution[m]'][0] = res_metadata['Y_resolution[m]'][0]/resize_factor
                        
                        tiff.imsave(str(prefix + '_' + path_to_tiff), resized_image,
                                        imagej=True,
                                        resolution=(res_metadata['X_resolution[m]'][0]*1000000, res_metadata['Y_resolution[m]'][0]*1000000),
                                        metadata={'spacing': res_metadata['z_spacing'], 'unit': 'um'}) 
                           
                        print('Resized succesfully')
                        print('Current resolution is ' + str(resized_image.shape[2]) + 'x' + str(resized_image.shape[1]))
                        
                    elif width != None and height != None:
                        print('Resized with [width x hight] both parameters is not allowed')
                        print('Choose one parameter and second will be scaled')
                        print('Rescaling on both parameters without scale can modifie biological diversity')
                    
                    else:
                        print('Resized unsuccessfully')
                        print('Provided wrong parameters or lack of parameters')
                           
                    
                    
            
            
            return res_metadata
        
        except:
            print("Something went wrong. Check the function input data and try again!")
            
     

     
def resize_projection(image, metadata = None, height = None, width = None, resize_factor = None):
    
    try:
        if height != None and  width == None:
            h = image.shape[0]
            w = image.shape[1]
            
            wh = int(height/h * w)
            
            
            image = cv2.resize(image, (wh, height))
            if metadata != None:
                metadata['X_resolution[m]'][0] = metadata['X_resolution[m]'][0]*height/h
                metadata['Y_resolution[m]'][0] = metadata['Y_resolution[m]'][0]*height/h
                
            print('Resized succesfully')
            print('Current resolution is ' + str(image.shape[1]) + 'x' + str(image.shape[0]))
            
        elif width != None and height == None:
            h = image.shape[0]
            w = image.shape[1]
            
            wh = int(width/w * h)
            
            
            image = cv2.resize(image, (width, wh))
            if metadata != None:
    
                metadata['X_resolution[m]'][0] = metadata['X_resolution[m]'][0]*width/w
                metadata['Y_resolution[m]'][0] = metadata['Y_resolution[m]'][0]*width/w
            
            print('Resized succesfully')
            print('Current resolution is ' + str(image.shape[1]) + 'x' + str(image.shape[0]))
            
        elif width == None and height == None and resize_factor != None:
            h = image.shape[0]
            w = image.shape[1]
            
            wh = int(w / resize_factor)
            hw = int(h / resize_factor)
            
            image = cv2.resize(image, (wh, hw))
            if metadata != None:
    
                metadata['X_resolution[m]'][0] = metadata['X_resolution[m]'][0]/resize_factor
                metadata['Y_resolution[m]'][0] = metadata['Y_resolution[m]'][0]/resize_factor
               
            print('Resized succesfully')
            print('Current resolution is ' + str(image.shape[1]) + 'x' + str(image.shape[0]))
            
        elif width != None and height != None:
            print('Resized with [width x hight] both parameters is not allowed')
            print('Choose one parameter and second will be scaled')
            print('Rescaling on both parameters without scale can modifie biological diversity')
        
        else:
            print('Resized unsuccessfully')
            print('Provided wrong parameters or lack of parameters')
               
        
    
        return image, metadata
    
    except:
        print("Something went wrong. Check the function input data and try again!")
        
            
    
def split_channels(path_to_images:str, path_to_save:str):
    
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
    
    
def xml_load(path_to_opera_xml:str):
    
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
        
        
        with open(path_to_opera_xml) as topo_file:
            for line in topo_file:
               if line.startswith('    <Image Version="1">'):
                   break
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
           
            
           
                    
        
        df = pd.DataFrame(df)
        df['name'] = [re.sub('p.*', '', x) for x in df['name']]
        
        df['y'] = df['y']*-1
        
        
        df = df.drop_duplicates()
        df['num'] = range(1,len(df['name'])+1)
        
        df = df.reset_index(drop = True)
        
    
        metadata = {'channel_name':channel_name, 'channel_number':channel_num, 'X_resolution[m]': x_res, 'Y_resolution[m]': y_res, 'max_intensity[nm]':max_intensity, 'z_spacing':np.mean(z_spacing), 'excitation_wavelength[nm]':excitation_wavelength, 'emissio_wavelength[nm]':emissio_wavelength, 'magnification[x]':magnification}
        
        
        return df, metadata  
    
    except:
        print("Something went wrong. Check the function input data and try again!")
    
    




def detect_outlires(xml_file:pd.DataFrame, list_of_out:list = []):
    
    try:
        
        if len(list_of_out) != 0:
            xml_file = xml_file[~xml_file.index.isin(list_of_out)]
        
        def outlires_image_detect(xml_file):
               
            x  = np.interp(xml_file['x'], (min(xml_file['x']), max(xml_file['x'])), (0, 100))
            y = np.interp(xml_file['y'], (min(xml_file['y']), max(xml_file['y'])), (0, 100))
            
            
            fig, ax = plt.subplots(figsize=(30, 30))
            ax.scatter(x , y , s= 700, c='red', alpha=0.7, marker = 's',  edgecolor="none")
            
            ax.set_axis_off()
            ax.invert_yaxis()
    
            for n, i in enumerate(xml_file.index):
                ax.annotate(i, (x[n] , y[n] ), xytext=(0, 0), textcoords="offset pixels", ha='center', va='center',
                     fontsize=8, fontweight='bold', color='yellow')
                
                
            physical_size = (16, 14)  
            pixels_per_inch = 300 
            pixel_size = tuple(int(physical_size[i] * pixels_per_inch) for i in (1, 0))
                
            fig.set_size_inches(*physical_size)
            fig.set_dpi(pixels_per_inch)
            
            canvas = FigureCanvas(fig)
            canvas.draw()
            image = np.array(canvas.renderer.buffer_rgba())
    
    
            cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    
                
            cv2.resizeWindow('Image', *pixel_size)
            cv2.imshow('Image', image)
            
    
            
            cv2.waitKey(0)
            
            
            cv2.destroyAllWindows()
            
            return fig
    
        fig = outlires_image_detect(xml_file)
        
        return xml_file, fig
    
    except:
        print("Something went wrong. Check the function input data and try again!")
    


def repair_blanks(xml_file:pd.DataFrame):
   
    try:
   
        x = np.unique(xml_file['x'])
        
        y = np.unique(xml_file['y'])
        
        
        
        #
        x_n = []
        for ix in x:
           x_n.append(len(xml_file['x'][xml_file['x'] == ix]))
           
        mfqx = [abs(xv) for xv in x_n]
           
        mfqx = max(set(mfqx),key = mfqx.count)
        
           
        df_x = pd.DataFrame({'x':x, 'n': x_n})
        #
        
        
        
        #
        y_n = []
        for iy in y:
           y_n.append(len(xml_file['y'][xml_file['y'] == iy]))
        
        
        mfqy = [abs(xy) for xy in y_n]
           
        mfqy = max(set(mfqy),key = mfqy.count)
        
        
        df_y = pd.DataFrame({'y':y, 'n': y_n})
        
        #
        
        df_x = df_x[df_x['n'] < mfqx]
        
        df_y = df_y[df_y['n'] < mfqy]
        
        xtml = xml_file.copy()
        
        xtml['XY'] = xtml['x'].astype(str) + xtml['y'].astype(str)
        
        b = 0
        for xi in df_x['x']:
            for yi in df_y['y']:
                if str(str(xi)+str(yi)) not in list(xtml['XY']):
                    b += 1
                    new_row = {'name' : 'blank' + str(b), 'x': float(xi) , 'y': float(yi), 'num': 'NULL'}
                    xml_file = xml_file.append(new_row, ignore_index=True)
    
                
        
        
        
        def outlires_image_detect(xml_file):
               
            x  = np.interp(xml_file['x'], (min(xml_file['x']), max(xml_file['x'])), (0, 100))
            y = np.interp(xml_file['y'], (min(xml_file['y']), max(xml_file['y'])), (0, 100))
            
            
            fig, ax = plt.subplots(figsize=(30, 30))
            ax.scatter(x , y , s= 700, c='red', alpha=0.7, marker = 's',  edgecolor="none")
            
            ax.set_axis_off()
            ax.invert_yaxis()
    
            for n, i in enumerate(xml_file.index):
                ax.annotate(i, (x[n] , y[n] ), xytext=(0, 0), textcoords="offset pixels", ha='center', va='center',
                     fontsize=8, fontweight='bold', color='yellow')
                
                
            physical_size = (16, 14)  
            pixels_per_inch = 300  
            pixel_size = tuple(int(physical_size[i] * pixels_per_inch) for i in (1, 0))
                
            fig.set_size_inches(*physical_size)
            fig.set_dpi(pixels_per_inch)
            
            canvas = FigureCanvas(fig)
            canvas.draw()
            image = np.array(canvas.renderer.buffer_rgba())
    
    
            cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    
                
            cv2.resizeWindow('Image', *pixel_size)
            cv2.imshow('Image', image)
            
    
            
            cv2.waitKey(0)
            
            
            cv2.destroyAllWindows()
            
            return fig
    
        fig = outlires_image_detect(xml_file)
        
        return xml_file, fig
    
    except:
        print("Something went wrong. Check the function input data and try again!")

  
    
def image_sequences(opera_coordinates:pd.DataFrame):
    
    try:
    
        y = list(np.unique(opera_coordinates['y']))
        y.sort()
    
    
        queue_images=[]
    
        for table in y:
            tmp = opera_coordinates[opera_coordinates['y'] == table]
            tmp = tmp.sort_values(by=['x'])
            queue_images = queue_images + list(tmp['name'])
    
        image_dictinary=pd.DataFrame()
        image_dictinary['queue']=queue_images
        image_dictinary['image_num']=range(1,len(image_dictinary['queue'])+1)
        
        img_length = len(list(np.unique(opera_coordinates['y'])))
        img_width = len(list(np.unique(opera_coordinates['x'])))
        
        return image_dictinary, img_length, img_width
    
    except:
        print("Something went wrong. Check the function input data and try again!")



def image_concatenate(path_to_images:str, imgs:pd.DataFrame, metadata, img_length:int, img_width:int, overlap:int, channels:list, n_thread:int):
     
    try:
        
        for obj in gc.get_objects():   
            if isinstance(obj, h5py.File):  
                try:
                    obj.close()
                except:
                    pass 
                
                
        if os.path.exists(os.path.join(path_to_images, 'images.h5')):
            os.remove(os.path.join(path_to_images,'images.h5'))
            
        if os.path.exists(os.path.join(path_to_images,'images2.h5')):
            os.remove(os.path.join(path_to_images,'images2.h5'))
        
        def par_1(q, path_to_images, img_width, imgs, black_img, st, ch, overlap):
            stop =  img_width * (q+1)
            start = img_width * q
            tmp = imgs['queue'][start:stop]
            
            list_p = []
            for t in tmp:
                if 'blank' in t:
                    list_p.append(str(t))
                else:
                    list_p.append(str([f for f in tmp_img if str(re.sub('\n','', str(t)) + 'p') in f and str('p'+st) in f][0]))
                
            data = []
            for img in list_p:
                if os.path.exists(os.path.join(path_to_images, img)):
                    data.append(cv2.imread(os.path.join(path_to_images, img), cv2.IMREAD_ANYDEPTH))
                else:
                    data.append(black_img)
                 
    
    
            row, col = data[0].shape
            for n,i in enumerate(data):
                data[n] = data[n][:, int(col*overlap/2):-int(col*overlap/2)]
                         
    
    
            
            data = np.concatenate(data, axis = 1)
            images_tmp.create_dataset('lane_' + str(q) + '-deep_' + str(st) + '-channel_' + str(ch),  data=data)
            del data
        
        
        
        images_list=os.listdir(path_to_images)
        deep = np.unique([re.sub('-.*','', re.sub('.*p', '', n)) for n in images_list if '.tiff' in n])
        
       
        
        for ch in channels:
            images_tmp2 = h5py.File(os.path.join(path_to_images, 'images2.h5'),   mode = 'a')
        
            tmp_img = [i for i in images_list if ch in i]
            
            black_img = cv2.imread(os.path.join(path_to_images, tmp_img[0]), cv2.IMREAD_ANYDEPTH)
            black_img.fill(0) 
            for st in tqdm(deep):
                images_tmp = h5py.File(os.path.join(path_to_images, 'images.h5'),   mode = 'a')
        
               
                Parallel(n_jobs=n_thread, prefer="threads")(delayed(par_1)(q, path_to_images, img_width, imgs, black_img, st, ch, overlap) for q in range(0,img_length))
        
               
                    
                data = []
                for q in range(0,img_length):
                    data.append(images_tmp[[f for f in images_tmp.keys() if 'lane_' + str(q) + '-deep' in f][0]][:])
                  
        
                images_tmp.close()
        
                os.remove(os.path.join(path_to_images, 'images.h5'))
                
             
                    
                row, col = data[0].shape
                for n,i in enumerate(data):
                    data[n] = data[n][int(row*overlap/2):-int(row*overlap/2), :]
                    
         
                    
            
                   
                 
                data = np.concatenate(data, axis = 0)
                
                images_tmp2.create_dataset('deep_' + str(st) + '-channel_' + str(ch),  data=data)
        
            data = []
            for q in tqdm(images_tmp2.keys()):
    
    
    
          
                data.append(images_tmp2[q][:])
                    
    
            data = np.stack(data)
            
        
            tiff.imsave('channel_' + str(ch) + '.tiff', data,
                           imagej=True,
                           resolution=(metadata['X_resolution[m]'][0]*1000000, metadata['Y_resolution[m]'][0]*1000000),
                           metadata={'spacing': metadata['z_spacing'], 'unit': 'um'}) 
                
            images_tmp2.close()
            
            del data
            
            os.remove(os.path.join(path_to_images, 'images2.h5'))
            
    except:
        print("Something went wrong. Check the function input data and try again! \nCheck that the number of channels you want to assemble matches the number of data channels!")





def z_projection(path_to_tiff:str, stack_check:str):

    if not os.path.exists(path_to_tiff):
        
        print('\nImage does not exist. Check the correctness of the path to image')
        
    else:
        
        global stack

        stack = tiff.imread(path_to_tiff)
        
        if stack_check == True:
        
            median_image = np.median(stack, axis=0)
    
            # Calculate the pixel-wise difference between each image and the median image
            diff_stack = np.abs(stack - median_image)
            
            del median_image
    
            # Threshold the difference stack
            threshold_value = filters.threshold_otsu(diff_stack)
            thresholded_stack = diff_stack > threshold_value
            
            del diff_stack, threshold_value
    
            # Drop outliers
            mean_intensity = np.mean(thresholded_stack, axis=(1,2))
            
            del thresholded_stack
            
            outlier_indices = list(np.where(mean_intensity > np.mean(mean_intensity) + np.std(mean_intensity))[0]) + list(np.where(mean_intensity < np.mean(mean_intensity) - np.std(mean_intensity))[0]) 
            
            stack = np.delete(stack, outlier_indices, axis=0)
            
           
        
        stack = stack.astype(np.uint64)  
        
    
                
       
        window = tk.Tk()
        
        window.geometry("500x800")
        window.title("Z-PROJECTION")
    
        window.iconbitmap(pkg_resources.resource_filename("operetta", "jbsicon.ico"))
       
        
        txt1 = tk.Label(window, text="Adjust parameters", anchor="w", justify="left")
        txt1.pack()
       
        tk.Label(window, text="").pack()
        tk.Label(window, text="").pack()
        
        
        label1 = tk.Label(window, text="Size", anchor="w")
        label1.pack()
        
        # Create a slider widget
        
        size = tk.DoubleVar()
        slider1 = tk.Scale(window, from_=0, to=100, orient=tk.HORIZONTAL, length=400, variable=size)
        slider1.set(50)
        slider1.pack()
        
        def update_slider(event):
            slider_value = slider1.get()
            slider_value += event.delta/120 
            size.set(slider_value)
        
        slider1.bind("<MouseWheel>", update_slider)
        
        tk.Label(window, text="").pack()
        
        label2 = tk.Label(window, text="Gamma", anchor="w")
        label2.pack()
        
        gamma = tk.DoubleVar()
        slider2 = tk.Scale(window, from_=0, to=99, orient=tk.HORIZONTAL, length=400, variable=gamma)
        slider2.set(10)
        slider2.pack()
        
        tk.Label(window, text="").pack()
        
        label3 = tk.Label(window, text="Threshold", anchor="w")
        label3.pack()
        
        label3_min = tk.Label(window, text="Min", anchor="w")
        label3_min.pack()
        
        threshold_min = tk.DoubleVar()
        slider3_min = tk.Scale(window, from_=0, to=32767, orient=tk.HORIZONTAL, length=400, variable=threshold_min)
        slider3_min.set(0)
        slider3_min.pack()
        
        label3_max = tk.Label(window, text="Max", anchor="w")
        label3_max.pack()
        
        threshold_max = tk.DoubleVar()
        slider3_max = tk.Scale(window, from_=0, to=65535, orient=tk.HORIZONTAL, length=400, variable=threshold_max)
        slider3_max.set(int(65535))
        slider3_max.pack()
        
        tk.Label(window, text="").pack()
        
        label5 = tk.Label(window, text="Brightness", anchor="w")
        label5.pack()
        
        brightness = tk.DoubleVar()
        slider5 = tk.Scale(window, from_=0, to=300, orient=tk.HORIZONTAL, length=400, variable=brightness)
        slider5.set(150)
        slider5.pack()
        
        
        label6 = tk.Label(window, text="Contrast", anchor="w")
        label6.pack()
        
        contrast = tk.DoubleVar()
        slider6 = tk.Scale(window, from_=0, to=30, orient=tk.HORIZONTAL, length=400, variable=contrast)
        slider6.set(10)
        slider6.pack()
        
        
        tk.Label(window, text="").pack()
        label4 = tk.Label(window, text="Color", anchor="w")
        label4.pack()
        
        items = ['grey', "blue", "green", "red", "magenta", 'yellow', 'cyan']
    
        combobox = ttk.Combobox(window, values=items)
        
        combobox.current(0)
        
        combobox.pack()
        
        
        tk.Label(window, text="").pack()
        
        label4 = tk.Label(window, text="Projection method", anchor="w")
        label4.pack()
        
        projections = ["avg", "max", "min", "sdt", "median"]
    
        projections_type = ttk.Combobox(window, values=projections)
        
        projections_type.current(0)
        
        projections_type.pack()
        
       
        button_finished = tk.BooleanVar(value=False)
        
        def active_changes():
                
               
                global img_gamma
                
                if projections_type.get() == 'avg':
                    img = np.mean(stack, axis=0).astype(np.uint64) 
                elif projections_type.get() == 'max':
                    img = np.max(stack, axis=0).astype(np.uint64)  
                elif projections_type.get() == 'min': 
                    img = np.min(stack, axis=0).astype(np.uint64)
                elif projections_type.get() == 'std':
                    img = np.std(stack, axis=0).astype(np.uint64) 
                elif projections_type.get() == 'median':
                    img = np.median(stack, axis=0).astype(np.uint64)    
                
               
                
                
                color = combobox.get()
                
                
                
                img[img >  int(threshold_max.get())] = int(threshold_max.get())
                
             

                img[img < int(threshold_min.get())] = 0
                
                #brightness
                img[img > 0] = img[img > 0] + int(brightness.get()*200 - 30000)
                
                
                #contrast
                img = cv2.multiply(img,  int(contrast.get()/ 10))
    
                img[img < 0] = 0

                #gamma
    
                
                factor = np.max(img)
                img_norm = img / np.max(img)
                img = np.power(img_norm, (gamma.get()+1)/10)
                
                img = (img * (factor))
                
                del img_norm
                
                
                img = (img / np.max(img)) * 65535

                img = np.round(img).astype(np.uint16)
            
                
                img_gamma = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint16)
                blurred_image = cv2.GaussianBlur(img, (1001, 1001), 0)
                img = cv2.addWeighted(img, 1.5, blurred_image, -0.25, 0)
                
                   
               
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
                 
                elif color == 'grey':
                    img_gamma[:,:,0] = img
                    img_gamma[:,:,1] = img
                    img_gamma[:,:,2] = img
            
            
                
                button_finished.set(True)
           
    
        
        tk.Label(window, text="").pack()
        button = tk.Button(window, text="Apply", command=active_changes)
        
        
        
        
        button.pack()
        
        
        
        def display_image():
            
            if button_finished.get():
                tmp = img_gamma
                height, width = tmp.shape[:2]
                resized_image = cv2.resize(tmp, (int(width/(50-size.get()*0.5)), int(height/(50-size.get()*0.5))))
                cv2.imshow('Image',resized_image)
                
            window.after(100, display_image)
           
        
        display_image()
        
        
        def auto_adjust():
            
            global stack
            
            for n, d in enumerate(stack):   
                stack[n] = (stack[n] / np.mean(stack[n])) * (((2**16)-1) /  np.log(np.mean(stack[n])))
                stack[n][stack[n] > ((2**16)-1)] = ((2**16)-1)
      
        
            stack[stack > ((2**16)-1)] = ((2**16)-1)
            stack = stack.astype(np.uint64)   
            
            projection = np.average(stack, axis=0)

            tmin = int(np.mean(projection))/1.1
            tmax = int(np.max(projection))

            
            gamma.set(10)         
            slider2.set(10)
                     
            threshold_min.set(tmin)
            slider3_min.set(tmin)
            
            threshold_max.set(tmax)
            slider3_max.set(tmax)
            
            
            brightness.set(150)
            slider5.set(150)
      
            contrast.set(10)
            slider6.set(10)
            
            projections_type.current(0)
           
            active_changes()
            
            
        tk.Label(window, text="").pack()
        buttonauto = tk.Button(window, text="Auto", command=auto_adjust)
        buttonauto.pack()
        
        
        def close_window():
            window.destroy()
            
        tk.Label(window, text="").pack()
        
        button2 = tk.Button(window, text="Save", command=close_window)
        
        button2.pack()
        
        
        
        active_changes()
        window.mainloop()
    
        cv2.destroyAllWindows()
    
        
        return img_gamma






def merge_images(image_list:list):
    
    try:
    
        intensity_factors = []
        for bt in range(len(image_list)):
            intensity_factors.append(1)
            
        def merge1():
            global result
            result = None
            
            for i, image in enumerate(image_list):
                if result is None:
                    result = image.astype(np.uint64) * intensity_factors[i]
                else:
                    result = cv2.addWeighted(result, 1, image.astype(np.uint64) * intensity_factors[i], 1, 0)
            
            result[result > ((2**16)-1)] = ((2**16)-1)
            result = result.astype(np.uint16)
        
            
    
        
        window = tk.Tk()
        
        window.geometry("500x600")
        window.title("MERGE CHANNELS")
    
        window.iconbitmap(pkg_resources.resource_filename("operetta", "jbsicon.ico"))
       
    
       
        tk.Label(window, text="").pack()
        tk.Label(window, text="").pack()
        
        
        label1 = tk.Label(window, text="Size", anchor="w")
        label1.pack()
        
        # Create a slider widget
        
        size = tk.DoubleVar()
        slider1 = tk.Scale(window, from_=0, to=100, orient=tk.HORIZONTAL, length=400, variable=size)
        slider1.set(50)
        slider1.pack()
        
        def update_slider(event):
            slider_value = slider1.get()
            slider_value += event.delta/120 
            size.set(slider_value)
        
        slider1.bind("<MouseWheel>", update_slider)
        
        tk.Label(window, text="").pack()
        
        label2 = tk.Label(window, text="Images intensity", anchor="w")
        label2.pack()
        
        slider_values = {}
    
        for bt in range(len(image_list)):
            slider_values[str('b' + str(bt))] = tk.DoubleVar()
            tk.Label(window, text="").pack()
            tk.Label(window, text=str("Img_" + str(bt)), anchor="w").pack()
            tk.Scale(window, from_=0, to=20, orient=tk.HORIZONTAL, length=400, variable=slider_values[str('b' + str(bt))]).pack()
            slider_values[str('b' + str(bt))].set(10)
    
        
    
    
        tk.Label(window, text="").pack()
        
        def merge2():
            intensity_factors = []
            for bt in range(len(image_list)):
                intensity_factors.append((slider_values[str('b' + str(bt))].get()/10))
                
            global result
            result = None
            
            for i, image in enumerate(image_list):
                if result is None:
                    result = image.astype(np.uint64) * intensity_factors[i]
                else:
                    result = cv2.addWeighted(result, 1, image.astype(np.uint64) * intensity_factors[i], 1, 0)
            
            result[result > ((2**16)-1)] = ((2**16)-1)
            result = result.astype(np.uint16)
        
        button = tk.Button(window, text="Apply", command=merge2)
        
        button.pack()
        
        merge1()
        
        
        def display_image():
    
            tmp = result
            height, width = tmp.shape[:2]
            resized_image = cv2.resize(tmp, (int(width/(50-size.get()*0.5)), int(height/(50-size.get()*0.5))))
            cv2.imshow('Image',resized_image)
            
            window.after(100, display_image)
           
        
        display_image()
        
        
        def close_window():
            window.destroy()
        
        tk.Label(window, text="").pack()
        
        button2 = tk.Button(window, text="Save", command=close_window)
        
        button2.pack()
        
        
        
        
        window.mainloop()
    
        cv2.destroyAllWindows()
        
      
        
        return result
    
    except:
        print("Something went wrong. Check the function input data and try again!")




def image_grid(path_to_opera_projection:str, img_length:int, img_width:int):

    if not os.path.exists(path_to_opera_projection):
        
        print('\nImage does not exist. Check the correctness of the path to image')

    else:
        global tmp
        cv2.namedWindow('Image')
        
        resize_factor = 10
        image = cv2.imread(path_to_opera_projection)
        tmp = image
        image = cv2.resize(image, (int(img_width* resize_factor), int(img_length* resize_factor)))  
    
        
        def nothing(x):
            pass
        
        def resize(image, img_length, img_width, resize_factor):
        
            
        
            for sqr in range(0,img_length+3):
                for sqr2 in range(1,img_width+3):
        
        
                    start_point = (sqr*resize_factor, sqr*resize_factor)
                
                    end_point = (sqr2*resize_factor, sqr2*resize_factor)
                
                    color = (255, 0, 0)
                
                    thickness = int(1*resize_factor/100)
                    
                    
                    image2 = cv2.rectangle(image, start_point, end_point, color, thickness)
                    
        
        
            num_pic=0
            for sqr in range(1,img_length+1):
                for sqr2 in range(0,img_width):
                    num_pic=num_pic+1
        
                    org= (int((sqr2*resize_factor)+resize_factor*0.3) ,int((sqr*resize_factor)-resize_factor*0.3))
                    
                    fontScale = float(1*resize_factor/100)
                       
                    color = (255,255,255)
                      
                    thickness = int(2*resize_factor/100)
                    
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    
                    
                    image2 = cv2.putText(image2, str(num_pic), org, font, 
                               fontScale, color, thickness, cv2.LINE_AA)
                    
           
                   
                    
            return image2
        
        resize_table = pd.DataFrame()
        resize_table['range'] = range(0,101)
        resize_table['height'] = list(itertools.repeat(img_length*resize_factor, 101))
        resize_table['width'] = list(itertools.repeat(img_width*resize_factor, 101))
        resize_table['resize_factor'] = list(itertools.repeat(resize_factor, 101))
        resize_table['factor'] = 0
        resize_table['factor'][resize_table['range'].isin(range(0,101))] = range(0,101)
        resize_table['height'] =  resize_table['height'] + (resize_table['height'] * resize_table['factor'])/resize_factor
        resize_table['width'] = resize_table['width'] + (resize_table['width'] * resize_table['factor'])/resize_factor
        resize_table['resize_factor'] =  resize_table['resize_factor'] + (resize_table['resize_factor'] * resize_table['factor'])/resize_factor
    
        
        image2 = resize(image, img_length, img_width, resize_factor)
        
        
        
        window = tk.Tk()
        
        window.geometry("500x600")
        window.title("IMAGE SELECTION")

        window.iconbitmap(pkg_resources.resource_filename("operetta", "jbsicon.ico"))
        
        txt1 = tk.Label(window, text="Images selection", anchor="w", justify="left")
        txt1.pack()
       
        tk.Label(window, text="").pack()
        tk.Label(window, text="").pack()
        
        
        label1 = tk.Label(window, text="Size", anchor="w")
        label1.pack()
        
        # Create a slider widget
        
        size = tk.DoubleVar()
        slider1 = tk.Scale(window, from_=0, to=100, orient=tk.HORIZONTAL, length=400, variable=size)
        slider1.set(50)
        slider1.pack()
        
        def update_slider(event):
            slider_value = slider1.get()
            slider_value += event.delta/120 
            size.set(slider_value)
        
        slider1.bind("<MouseWheel>", update_slider)
        
        tk.Label(window, text="").pack()
        
        
        
        label = tk.Label(window, text="Enter id of image:")
        label.pack()
        
        tk.Label(window, text="").pack()
        
        t = Text(window, height=15, width=50)
        t.pack()
           
           
        def display_image():
            
            rf = size.get()
        
  
            image = tmp
            image = cv2.resize(image, (int(resize_table['width'][resize_table['range'] == rf][rf]), int(resize_table['height'][resize_table['range'] == rf][rf])))  
            image2 = resize(image, img_length, img_width, int(resize_table['resize_factor'][resize_table['range'] == rf][rf]))

             
            cv2.imshow('Image',image2)
                
            window.after(100, display_image)
           
        
        display_image()
        
        
        def close_window():
            global image_list
            image_list = t.get("1.0", tk.END)
            image_list = re.sub(r"\s+", "", image_list)
            image_list = image_list.split(',')
            image_list = [item for item in image_list if item != ""]
            image_list = [int(x) for x in image_list]
            window.destroy()
        
        tk.Label(window, text="").pack()
        
        button2 = tk.Button(window, text="Apply", command=close_window)
        
        button2.pack()

        
        window.mainloop()

        cv2.destroyAllWindows()
   

        return image_list
    
    

def select_pictures(image_dictinary:pd.DataFrame, path_to_images:str, path_to_save:str, numbers_of_pictures:list, chennels:list):
    
    try:
    
        selected = image_dictinary[image_dictinary['image_num'].isin(numbers_of_pictures)]
        selected = selected.reset_index()
        
        if not os.path.exists(path_to_save):
            os.mkdir(path_to_save)
        
        for n, num in enumerate(selected['image_num']):
            
            images_list=os.listdir(path_to_images)
        
            images_list=[x for x in images_list if str(re.sub('\n','', (str(selected['queue'][selected['image_num'] == num][n]))) + 'p') in x]
            
            images_list = [x for x in images_list if re.sub('sk.*','',re.sub('.*-','',x)) in chennels]
            
            if not os.path.exists(os.path.join(path_to_save,'img_' + str(num))):
                os.mkdir(os.path.join(path_to_save,'img_' + str(num)))
                
            for image in images_list:
                shutil.copy(os.path.join(path_to_images,image),os.path.join(path_to_save,'img_' + str(num)))
                
    except:
        print("Something went wrong. Check the function input data and try again!")  
            
            
def add_scalebar(image, metadata):

    try:
        global image2
        global met
        
        met = metadata.copy()

        image2 = image.copy()
        
        h = image2.shape[0]
        w = image2.shape[1]
        
        nw = None
        nh = None
        
        
        if w > 20000:
            rw = 20000/w
            
            nw = int(w*rw)
            nh = int(h*rw)
            
            image2 = cv2.resize(image2, (nw, nh))
            metadata['X_resolution[m]'][0] = metadata['X_resolution[m]'][0]*rw
            metadata['Y_resolution[m]'][0] = metadata['Y_resolution[m]'][0]*rw
            h = image2.shape[0]
            w = image2.shape[1]
    
                               
        if h > 20000:
            rh = 20000/h
            
            nw = int(w*rh)
            nh = int(h*rh)
            
            image2 = cv2.resize(image2, (nw, nh))
            metadata['X_resolution[m]'][0] = metadata['X_resolution[m]'][0]*rh
            metadata['Y_resolution[m]'][0] = metadata['Y_resolution[m]'][0]*rh
            h = image2.shape[0]
            w = image2.shape[1]
           
                                
        if w != image2.shape[1] or h != image2.shape[0]:
            print('Resolution of the image was too large')
            print('Current resolution is ' + str(nw) + 'x' + str(nh))

            
        
        window = tk.Tk()
        
        window.geometry("500x800")
        window.title("SCALE-BAR")
    
        window.iconbitmap(pkg_resources.resource_filename("operetta", "jbsicon.ico"))
       
        
        txt1 = tk.Label(window, text="Scale parameters", anchor="w", justify="left")
        txt1.pack()
       
        tk.Label(window, text="").pack()
        tk.Label(window, text="").pack()
        
        
        label1 = tk.Label(window, text="Size", anchor="w")
        label1.pack()
        
        # Create a slider widget
        
        size = tk.DoubleVar()
        slider1 = tk.Scale(window, from_=0, to=100, orient=tk.HORIZONTAL, length=400, variable=size)
        slider1.set(50)
        slider1.pack()
        
        def update_slider(event):
            slider_value = slider1.get()
            slider_value += event.delta/120 
            size.set(slider_value)
        
        slider1.bind("<MouseWheel>", update_slider)
        
        tk.Label(window, text="").pack()
        
        label2 = tk.Label(window, text="Scale length [um]", anchor="w")
        label2.pack()
        
        length = tk.DoubleVar()
        l = [50, 100, 250, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000]
    
        length = ttk.Combobox(window, values=l)
        length.current(4)
        length.pack()
       
        
        tk.Label(window, text="").pack()
        
        label3 = tk.Label(window, text="Scalebar thickness[px]", anchor="w")
        label3.pack()
        
        
        thickness = tk.DoubleVar()
        items = [10, 15, 25, 30, 35, 50, 75, 100]
    
        thickness = ttk.Combobox(window, values=items)
        thickness.current(3)
        thickness.pack()

        

        tk.Label(window, text="").pack()
        label4 = tk.Label(window, text="Color", anchor="w")
        label4.pack()
        
        items = ['white', 'grey', "blue", "green", "red", "magenta", 'yellow', 'cyan', 'black']
    
        combobox = ttk.Combobox(window, values=items)
        
        combobox.current(0)
        
        combobox.pack()
        
        
        tk.Label(window, text="").pack()
        label5 = tk.Label(window, text="Font size", anchor="w")
        label5.pack()
        
        fonts = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10]
    
        font = ttk.Combobox(window, values=fonts)
        
        font.current(10)
        
        font.pack()
        
        
        
        tk.Label(window, text="").pack()
        
        label6 = tk.Label(window, text="Position", anchor="w")
        label6.pack()
        
        position_bar = ["right_bottom", "right_top", "left_top", "left_bottom"]
    
        position_type = ttk.Combobox(window, values=position_bar)
        
        position_type.current(0)
        
        position_type.pack()
        
        tk.Label(window, text="").pack()
        label7 = tk.Label(window, text="Horizontal position", anchor="w")
        label7.pack()
        
        horizontal = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10]
    
        hor = ttk.Combobox(window, values=horizontal)
        
        hor.current(10)
        
        hor.pack()
        
        tk.Label(window, text="").pack()
        label7 = tk.Label(window, text="Vertical position", anchor="w")
        label7.pack()
        
        vertical = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10]
    
        ver = ttk.Combobox(window, values=vertical)
        
        ver.current(10)
        
        ver.pack()
        
       
        button_finished = tk.BooleanVar(value=False)
        
        def active_changes():
            

                scale_color = (65535, 65535, 65535)
               
                global image3
                
                image3 = image2.copy()
                
               
                scale_length_um = int(length.get())  
                pixels_per_um = met['X_resolution[m]'][0]*1000000   
                scale_length_px = int(scale_length_um * pixels_per_um)
 
                # Calculate the position of the scale bar
                image_height = image3.shape[0]
                image_width = image3.shape[1]
                

                
                scale_position = (int(image_width*0.96) - scale_length_px, int(image_height*0.96))  # Adjust the position as needed

 
                # Draw the scale bar on the image
                if str(combobox.get()) == 'grey':
                    scale_color = (32768, 32768, 32768)
                elif str(combobox.get()) == 'red':
                    scale_color = (0, 0, 65535)
                elif str(combobox.get()) == 'green':
                    scale_color = (0, 65535, 0)
                elif str(combobox.get()) == 'blue':
                    scale_color = (65535, 0, 0)
                elif str(combobox.get()) == 'magenta':
                    scale_color = (65535, 0, 65535)
                elif str(combobox.get()) == 'yellow':
                    scale_color = (0, 65535, 65535)
                elif str(combobox.get()) == 'cyan':
                    scale_color = (65535, 65535, 0)
                elif str(combobox.get()) == 'black':
                    scale_color = (0, 0, 0)
                elif str(combobox.get()) == 'white':
                    scale_color = (65535, 65535, 65535)
                  


                font_typ = ImageFont.truetype('arialbd.ttf', int(np.log2(image_height)* float(font.get())))

                scale_thickness = int(thickness.get())  
                
                if str(position_type.get()) == 'left_top':
                    scale_position = (int(image_width*0.04) + scale_length_px + int(hor.get())*int(np.log10(image_width)*10), int(image_height*0.04) + int(ver.get())*int(np.log10(image_height)*10)) 
                    cv2.rectangle(image3, scale_position, (scale_position[0] + scale_length_px, scale_position[1] ), scale_color, scale_thickness)
                    overlay = np.zeros((image3.shape[0], image3.shape[1]), dtype = np.uint16)
                    overlay_rgb = np.repeat(overlay[:, :, np.newaxis], 3, axis=2).astype(np.uint16)
                    img_pil = Image.fromarray(overlay_rgb, mode='RGB')
                    draw = ImageDraw.Draw(img_pil)
                    draw.text((scale_position[0] + 10, scale_position[1] + int(np.log10(image_height)*15)), f'{scale_length_um} \u03BCm', fill = str(combobox.get()), font = font_typ)
                    img_pil = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR).astype(np.uint16)
                    img_pil = cv2.normalize(img_pil, None, 0, 65535, cv2.NORM_MINMAX)
                    image3 = cv2.add(image3, img_pil) 
                    
                elif str(position_type.get()) == 'left_bottom':
                    scale_position = (int(image_width*0.04) + scale_length_px + int(hor.get())*int(np.log10(image_width)*10), int(image_height*0.96) - int(ver.get())*int(np.log10(image_height)*10))  
                    cv2.rectangle(image3, scale_position, (scale_position[0] + scale_length_px, scale_position[1] ), scale_color, scale_thickness)
                    overlay = np.zeros((image3.shape[0], image3.shape[1]), dtype = np.uint16)
                    overlay_rgb = np.repeat(overlay[:, :, np.newaxis], 3, axis=2).astype(np.uint16)
                    img_pil = Image.fromarray(overlay_rgb, mode='RGB')
                    draw = ImageDraw.Draw(img_pil)
                    draw.text((scale_position[0] + 10, scale_position[1] + int(np.log10(image_height)*15)), f'{scale_length_um} \u03BCm', fill = str(combobox.get()), font = font_typ)
                    img_pil = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR).astype(np.uint16)
                    img_pil = cv2.normalize(img_pil, None, 0, 65535, cv2.NORM_MINMAX)
                    image3 = cv2.add(image3, img_pil) 
                    
                elif str(position_type.get()) == 'right_top':
                    scale_position = (int(image_width*0.96) - scale_length_px + int(hor.get())*int(np.log10(image_width)*10), int(image_height*0.04) + int(ver.get())*int(np.log10(image_height)*10))  
                    cv2.rectangle(image3, scale_position, (scale_position[0] + scale_length_px, scale_position[1] ), scale_color, scale_thickness)
                    overlay = np.zeros((image3.shape[0], image3.shape[1]), dtype = np.uint16)
                    overlay_rgb = np.repeat(overlay[:, :, np.newaxis], 3, axis=2).astype(np.uint16)
                    img_pil = Image.fromarray(overlay_rgb, mode='RGB')
                    draw = ImageDraw.Draw(img_pil)
                    draw.text((scale_position[0] + 10, scale_position[1] + int(np.log10(image_height)*15)), f'{scale_length_um} \u03BCm', fill = str(combobox.get()), font = font_typ)
                    img_pil = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR).astype(np.uint16)
                    img_pil = cv2.normalize(img_pil, None, 0, 65535, cv2.NORM_MINMAX)
                    image3 = cv2.add(image3, img_pil) 
                    
                elif str(position_type.get()) == 'right_bottom':
                    scale_position = (int(image_width*0.96) - scale_length_px + int(hor.get())*int(np.log10(image_width)*10), int(image_height*0.96) - int(ver.get())*int(np.log10(image_height)*10))  
                    cv2.rectangle(image3, scale_position, (scale_position[0] + scale_length_px, scale_position[1] ), scale_color, scale_thickness)
                    overlay = np.zeros((image3.shape[0], image3.shape[1]), dtype = np.uint16)
                    overlay_rgb = np.repeat(overlay[:, :, np.newaxis], 3, axis=2).astype(np.uint16)
                    img_pil = Image.fromarray(overlay_rgb, mode='RGB')
                    draw = ImageDraw.Draw(img_pil)
                    draw.text((scale_position[0] + 10, scale_position[1] + int(np.log10(image_height)*15)), f'{scale_length_um} \u03BCm', fill = str(combobox.get()), font = font_typ)
                    img_pil = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR).astype(np.uint16)
                    img_pil = cv2.normalize(img_pil, None, 0, 65535, cv2.NORM_MINMAX)
                    image3 = cv2.add(image3, img_pil) 

                
                button_finished.set(True)
           
    
        
        tk.Label(window, text="").pack()
        button = tk.Button(window, text="Apply", command=active_changes)
        
        
        
        
        button.pack()
        
        
        
        def display_image():
            
            if button_finished.get():
                tmp = image3
                height, width = tmp.shape[:2]
                resized_image = cv2.resize(tmp, (int(width/(50-size.get()*0.5)), int(height/(50-size.get()*0.5))))
                cv2.imshow('Image',resized_image)
                
            window.after(100, display_image)
           
        
        display_image()
        
        
       
        
        def close_window():
            window.destroy()
            
        tk.Label(window, text="").pack()
        
        button2 = tk.Button(window, text="Save", command=close_window)
        
        button2.pack()
        
        
        
        active_changes()
        window.mainloop()
    
        cv2.destroyAllWindows()
    
        
        return image3

    except:
        print("Something went wrong. Check the function input data and try again!")
