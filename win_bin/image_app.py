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
from tkinter import ttk, Text, filedialog
from PIL import ImageFont, ImageDraw, Image, ImageTk
import copy
import webbrowser
import sys
import pickle
import threading



 #       _  ____   _         _____              _                      
 #      | ||  _ \ (_)       / ____|            | |                    
 #      | || |_) | _   ___ | (___   _   _  ___ | |_  ___  _ __ ___   
 #  _   | ||  _ < | | / _ \ \___ \ | | | |/ __|| __|/ _ \| '_ ` _ \  
 # | |__| || |_) || || (_) |____) || |_| |\__ \| |_|  __/| | | | | | 
 #  \____/ |____/ |_| \___/|_____/  \__, ||___/ \__|\___||_| |_| |_|  
 #                                   __/ |                                   
 #                                  |___/      


# path getting function

def app_path():
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return base_path


_icon_source = app_path()



warnings.filterwarnings("ignore", category=RuntimeWarning)


################################ Main code #####################################


class Metadata:
    def __init__(self, metadata_path = None, 
                 metadata=None, xml=None, 
                 tiffs_path=None,
                 concat_path=None, 
                 saved_tiff_path=None, 
                 images_dict = {'img':[], 'metadata':[], 'img_name':[]}, 
                 tmp_path=None, 
                 channel = None,
                 tmp_xml = None,
                 save_current = None,
                 annotation_series = {'annotated_image':None, 'image_grid':None, 'img_data':None},
                 resize_tmp = {'image':None, 'metadata':None, 'name':None},
                 removal_list = None,
                 project_path = None):
        
        
        self.metadata_path = metadata_path
        self.metadata = metadata
        self.xml = xml
        self.tiffs_path = tiffs_path
        self.concat_path = concat_path
        self.saved_tiff_path = saved_tiff_path
        self.images_dict = images_dict
        self.tmp_path = tmp_path
        self.channel = channel
        self.annotation_series = annotation_series
        self.resize_tmp = resize_tmp
        self.tmp_xml = tmp_xml
        self.save_current = save_current
        self.removal_list = removal_list
        self.project_path = project_path


        



    def add_metadata_path(self, metadata_path):
        self.metadata_path = metadata_path
        
    def add_metadata(self, metadata):
        self.metadata = metadata

    def add_xml(self, xml):
        self.xml = xml
    
    def add_tiffs_path(self, tiffs_path):
        self.tiffs_path = tiffs_path

    def add_concat_path(self, concat_path):
        self.concat_path = concat_path
        
    def add_saved_tiff_path(self, saved_tiff_path):
        self.saved_tiff_path = saved_tiff_path
        
    def add_image(self, img, img_name, metadata):
        self.images_dict['img'].append(img)
        self.images_dict['img_name'].append(img_name)
        self.images_dict['metadata'].append(metadata)
        
    def add_tmp_path(self, tmp_path):
        self.tmp_path = tmp_path
        
    def add_channel(self, channel):
        self.channel = channel
        
          
    def add_annotated_raw(self, ann_img, img_grid, img_dict):
        self.annotation_series['annotated_image'] = ann_img
        self.annotation_series['image_grid'] = img_grid
        self.annotation_series['img_data'] = img_dict
        
        
    def add_resize(self, img, met, nam):
        self.resize_tmp['image'] = img
        self.resize_tmp['metadata'] = met
        self.resize_tmp['name'] = nam


    
    def add_tmp_xml(self, tmp_xml):
        self.tmp_xml = tmp_xml
        
        
    def add_save_current(self, save_current):
        self.save_current = save_current
        
    
    def add_rm_list(self, removal_list):
        self.removal_list = removal_list
        
        
    def add_project_path(self, project_path):
        self.project_path = project_path
        
        
        
    def copy(self):
        return Metadata(self.metadata_path, 
                       self.metadata, 
                       self.xml, 
                       self.tiffs_path, 
                       self.concat_path, 
                       self.saved_tiff_path, 
                       self.images_dict, 
                       self.tmp_path,
                       self.channel,
                       self.annotation_series,
                       self.resize_tmp,
                       self.tmp_xml,
                       self.save_current,
                       self.removal_list,
                       self.project_path)







def get_number_of_cores():
    try:

        num_cores = os.cpu_count()
        if num_cores is not None:
            return num_cores

        num_cores = multiprocessing.cpu_count()
        return num_cores

    except Exception as e:
        print(f"Error while getting the number of cores: {e}")
        return None




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
                            
                            
                        res_metadata['X_resolution[um/px]'] = res_metadata['X_resolution[um/px]']*(height/h)
                        res_metadata['Y_resolution[um/px]'] = res_metadata['Y_resolution[um/px]']*(height/h)
                        
                        
                        tiff.imwrite(str(prefix + '_' + path_to_tiff), resized_image,
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
                        
                            
                        print('Resized successfully')
                        print('Current resolution is ' + str(resized_image.shape[2]) + 'x' + str(resized_image.shape[1]))
                        
                    elif width != None and height == None:
                        h = image.shape[1]
                        w = image.shape[2]
                        
                        wh = int(width/w * h)
                        
                        
                        resized_image = np.empty((image.shape[0], wh, width)).astype(np.uint16)
                        
                        for n in range(image.shape[0]):
                            resized_image[n] = cv2.resize(image[n], (width, wh))
                            
                
                        res_metadata['X_resolution[um/px]'] = res_metadata['X_resolution[um/px]']*(width/w)
                        res_metadata['Y_resolution[um/px]'] = res_metadata['Y_resolution[um/px]']*(width/w)
                        
                        tiff.imwrite(str(prefix + '_' + path_to_tiff), resized_image,
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
                                                  'magnification[x]': res_metadata['magnification[x]']}) 
                        
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
                        
                
                        res_metadata['X_resolution[um/px]'] = res_metadata['X_resolution[um/px]']/resize_factor
                        res_metadata['Y_resolution[um/px]'] = res_metadata['Y_resolution[um/px]']/resize_factor
                        
                        tiff.imwrite(str(prefix + '_' + path_to_tiff), resized_image,
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
                                                  'magnification[x]': res_metadata['magnification[x]']}) 
                        
                           
                        print('Resized successfully')
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
               
        
    
        return image, cmet
    
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
    
    
    
    
def mirror_function(img, rotate:str):
    
    if rotate == 'h':
        img = np.fliplr(img.copy())
    elif rotate == 'v':
        img = np.flipud(img.copy())
    elif rotate == 'hv':
        img = np.flipud(np.fliplr(img.copy()))
        
    return img



def rotate_function(img, rotate:int):
    
    img = img.copy()
    
    img = np.rot90(img.copy(), k=rotate)

    return img


    
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
        
    
        metadata = {'channel_name':channel_name, 'channel_number':channel_num, 'X_resolution[um/px]': float(1./(x_res[0]*1000000)), 'Y_resolution[um/px]': float(1./(y_res[0]*1000000)), 'max_intensity[um]':max_intensity, 'z_spacing':np.mean(z_spacing), 'excitation_wavelength[nm]':excitation_wavelength, 'emissio_wavelength[nm]':emissio_wavelength, 'magnification[x]':magnification}
        
        
        return df, metadata  
    
    except:
        print("Something went wrong. Check the function input data and try again!")
    
    
    


def manual_outlires(xml_file:pd.DataFrame, list_of_out:list = [], dispaly_plot = False):
    
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

    
        fig = outlires_image_detect(xml_file)
        
        return xml_file, fig
    
    except:
        print("Something went wrong. Check the function input data and try again!")
    


    

def repair_image(xml_file:pd.DataFrame, dispaly_plot = True):
   
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
           y_n.append(len(xml_file['y'][xml_file['y'] == iy]))
        
        
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
        
        xml_file['XY'] = xml_file['x'].astype(str) + xml_file['y'].astype(str)
       
                
        
        for xi in df_x_un['x']:
            for yi in df_y['y']:
                if str(str(xi)+str(yi)) not in list(xml_file['XY']):
                    b += 1
                    new_row = {'name' : 'blank' + str(b), 'x': float(xi) , 'y': float(yi), 'num': 'NULL'}
                    xml_file = pd.concat([xml_file, pd.DataFrame([new_row])], ignore_index=True)
        
        
        
        xml_file['XY'] = xml_file['x'].astype(str) + xml_file['y'].astype(str)
        
        for yi in df_y_un['y']:
            for xi in df_x['x']:
                if str(str(xi)+str(yi)) not in list(xml_file['XY']):
                    b += 1
                    new_row = {'name' : 'blank' + str(b), 'x': float(xi) , 'y': float(yi), 'num': 'NULL'}
                    xml_file = pd.concat([xml_file, pd.DataFrame([new_row])], ignore_index=True)
        
        

        
        
        
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




def image_concatenate(path_to_images:str, path_to_save:str, imgs:pd.DataFrame, metadata, img_length:int, img_width:int, overlap:int, channels:list, resize:int = 2, n_proc:int = 4, par_type = 'processes'):
     
    init_path = os.getcwd()
    
    res_metadata = copy.deepcopy(metadata)
 
    
    try:
    
        os.chdir(path_to_images)         
            
        
        def par_1(q, img_width, imgs, black_img, st, ch, overlap, resize):
            stop = img_width * (q + 1)
            start = img_width * q
            tmp = imgs['queue'][start:stop]
    
            list_p = []
            for t in tmp:
                if 'blank' in t:
                    list_p.append(str(t))
                else:
                    list_p.append(
                        str([f for f in tmp_img if str(re.sub('\n', '', str(t)) + 'p') in f and str('p' + st) in f][0]))
           
            
            data = []
            for img in list_p:
                if os.path.exists(img):
                    data.append(cv2.imread(img, cv2.IMREAD_ANYDEPTH))
                else:
                    data.append(black_img)
    
            row, col = data[0].shape
            for n, i in enumerate(data):
                if resize > 1:
                    original_height, original_width = data[n].shape[:2]
    
                    new_width = original_width // resize
                    new_height = original_height // resize
                    if overlap > 0:
                        data[n] = cv2.resize(data[n][:, int(col * overlap / 2):-int(col * overlap / 2)], (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                    else:
                        data[n] = cv2.resize(data[n], (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                    
                else:
                    if overlap > 0:
                        data[n] = data[n][:, int(col * overlap / 2):-int(col * overlap / 2)]
    
            data = np.concatenate(data, axis=1)
            
            
            if overlap > 0:
                row, col = data.shape  
                data = data[int(row*overlap/2):-int(row*overlap/2), :]
    
            return data
        
        
        
        images_list=os.listdir(os.getcwd())
        
    
        deep = np.unique([re.sub('-.*','', re.sub('.*p', '', n)) for n in images_list if '.tiff' in n])
               
        
        for ch in channels:
            
            images_tmp = []
         
            tmp_img = [i for i in images_list if ch in i]
            
            black_img = cv2.imread(tmp_img[0], cv2.IMREAD_ANYDEPTH)
            black_img.fill(0) 
            
            for st in deep:
                
                data = Parallel(n_jobs=n_proc, prefer=par_type)(delayed(par_1)(q, img_width, imgs, black_img, st, ch, overlap, resize) for q in range(0,img_length))
               
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
            
            os.chdir(path_to_images)    
            
            
            
        os.chdir(init_path)    
    
    except:
        os.chdir(init_path)   
        print("Something went wrong. Check the function input data and try again! \nCheck that the number of channels you want to assemble matches the number of data channels!")



def get_screan():
    

    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    root.destroy()
    
    return screen_width, screen_height




def load_tiff(path_to_tiff:str):
    
    
    stack = tiff.imread(path_to_tiff)
    
    
    if stack.dtype != 'uint16':
        
        stack = stack.astype(np.uint16)
        
        for n, _ in enumerate(stack):

            min_val = np.min(stack[n])
            max_val = np.max(stack[n])
            
            stack[n] = ((stack[n] - min_val) / (max_val - min_val) * 65535).astype(np.uint16)
            
            stack[n] = np.clip(stack[n], 0, 65535)
    


    return stack




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
    



def resize_to_screen_img(img_file, factor = 4):
    
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
    
    
   
    
    



def z_projection(tiff_object, projection_type = 'avg'):
    
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








def clahe_16bit(img, kernal = (100, 100)):
    
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




def equalizeHist_16bit(image_eq):

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





def adjust_img_16bit(img, color = 'gray', max_intensity = 65535, min_intenisty = 0, brightness = 1000, contrast = 1, gamma = 1):

    
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




def merge_images(image_list:list, intensity_factors:list = []):
    
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





def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    
    # convert to 16 bit (the function are working on 16 bit images!)
    if img.dtype != 'uint16':
        
        min_val = np.min(img)
        max_val = np.max(img)
        
        img = ((img - min_val) / (max_val - min_val) * 65535).astype(np.uint16)
        
        img = np.clip(img, 0, 65535)
            
        
    return img





def image_grid_resize(image_grid, img_length:int, img_width:int, font_color:str = 'white', grid_color:str = 'blue'):
    
    grid_color = grid_color.lower()
    font_color = font_color.lower()

    img_length_single, img_width_single = image_grid.shape[:2]
    
    
    img_width_single //= img_width
    img_length_single //= img_length

    image2 = image_grid.copy()  
    
    
    
    

    if grid_color == 'green':
        grid_color_num = (0, 2**16, 0)
    
    
    elif grid_color == 'red':
        grid_color_num = (0, 0, 2**16)
    
        
    elif grid_color == 'blue':
        grid_color_num = (2**16, 0, 0)
    
        
    elif grid_color == 'magenta':
        grid_color_num = (2**16, 0, 2**16)

        
    elif grid_color == 'yellow':
        grid_color_num = (0, 2**16, 2**16)


    elif grid_color == 'cyan':
        grid_color_num = (2**16, 2**16, 0)

    else:
        grid_color_num = (2**16, 2**16,  2**16)
        


    if font_color == 'green':
        font_color_num = (0, 2**16, 0)
    
    elif font_color == 'red':
        font_color_num = (0, 0, 2**16)
    
    elif font_color == 'blue':
        font_color_num = (2**16, 0, 0)
    
    elif font_color == 'magenta':
        font_color_num = (2**16, 0, 2**16)

    elif font_color == 'yellow':
        font_color_num = (0, 2**16, 2**16)

    elif font_color == 'cyan':
        font_color_num = (2**16, 2**16, 0)

    else:
        font_color_num = (2**16, 2**16,  2**16)

        
        

    thickness = int(max(1, int(min(img_length_single, img_width_single) / 20)))
    font_scale = float(min(img_length_single, img_width_single) / 120)

    num_pic = 0
    fontScale = font_scale
    thickness = int(thickness*0.8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for sqr2 in range(img_length):
        for sqr in range(img_width):
            start_point = (sqr * img_width_single, sqr2 * img_length_single)
            end_point = ((sqr + 1) * img_width_single, (sqr2 + 1) * img_length_single)
            image2 = cv2.rectangle(image2, start_point, end_point, grid_color_num, thickness)
            
            num_pic += 1
            org = (int((sqr * img_width_single) + img_width_single * 0.1),
                   int((sqr2 * img_length_single) + img_length_single * 0.7))
            
            image2 = cv2.putText(image2, str(num_pic), org, font, fontScale, font_color_num, thickness, cv2.LINE_AA)



    return image2

    
    
    
    

def select_pictures(image_dictinary:pd.DataFrame, path_to_images:str, path_to_save:str, numbers_of_pictures:list, chennels:list, rm_slice_list = None):
    
    try:
    
        selected = image_dictinary[image_dictinary['image_num'].isin(list(numbers_of_pictures))]
        selected = selected.reset_index()
        
        if not os.path.exists(path_to_save):
            os.mkdir(path_to_save)
        
        for n, num in enumerate(selected['image_num']):
            
            images_list=os.listdir(path_to_images)
        
            images_list=[x for x in images_list if str(re.sub('\n','', (str(selected['queue'][selected['image_num'] == num][n]))) + 'p') in x]
            
            images_list = [x for x in images_list if re.sub('sk.*','',re.sub('.*-','',x)) in chennels]
            
            if not os.path.exists(os.path.join(path_to_save,'img_' + str(num))):
                os.mkdir(os.path.join(path_to_save,'img_' + str(num)))
            
                    
            for inx, image in enumerate(images_list):
                if None == rm_slice_list:
                    shutil.copy(os.path.join(path_to_images,image),os.path.join(path_to_save,'img_' + str(num)))
                else:
                    if inx not in rm_slice_list:
                        shutil.copy(os.path.join(path_to_images,image),os.path.join(path_to_save,'img_' + str(num)))

                
    except:
        print("Something went wrong. Check the function input data and try again!")  
            
            
            
            
            
   


def read_tiff_meta(file_path):


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




def get_pos(event, j, p, flags, param):
    global zoom_factor
    global x_cord
    global y_cord

    
    x_cord = j
    y_cord = p
    
    
     


def add_cord(image, final_shape , window_size, x_cord, y_cord):
    global zoom_factor
    global zoomed_h
    global zoomed_w
    global x_start
    global y_start
    
    xx = None
    yy = None
    xw = None
    yw = None
    
    
    h, w = image.shape[:2]
    
    if x_cord is not None and y_cord is not None:
        

 
        
        zoomed_x = int(x_cord  * w / zoom_factor / (int(w /(100/(window_size)))))
        zoomed_y = int(y_cord  * h / zoom_factor / (int(h /(100/(window_size)))))
        
      
        if zoom_factor == 1:
            xx = int(zoomed_x)
            yy = int(zoomed_y)
            
            xw = int(xx * final_shape[1] / w)
            yw = int(yy * final_shape[0] / h)
        else:
            

            xx = int(zoomed_x + x_start)
            yy = int(zoomed_y + y_start)
            
            
            xw = int(xx * final_shape[1] / w)
            yw = int(yy * final_shape[0] / h)
            

            
    
    return xx, yy, xw, yw

        
        

    
def zoom_in(event, z, k, flags, param):
    global zoom_factor
    global x
    global y
    x = z
    y = k
    
    if event == cv2.EVENT_MOUSEWHEEL:
        # Check if the mouse wheel is scrolled up (positive) or down (negative)
        if flags > 0:
            # Zoom in
            zoom_factor *= 1.1
        else:
            # Zoom out
            zoom_factor /= 1.1

        # Ensure zoom factor is within reasonable limits
        zoom_factor = max(1, min(zoom_factor, 5))



   
  
    

# Function to update the zoomed region
def update_zoomed_region(image, window_size, x, y):
    global zoom_factor
    
    global zoomed_h
    global zoomed_w
    global x_start
    global y_start
        
    try:
        # Calculate the zoomed region based on the mouse cursor position
        h, w = image.shape[:2]
        
        x = int(x * w / (int(w/(100/(window_size)))))
        y = int(y * h / (int(h/(100/(window_size)))))
        
        
        zoomed_h = int(h / zoom_factor)
        zoomed_w = int(w / zoom_factor)
        
        # Ensure the zoomed region stays within the image boundaries
        x_start = max(0, min(x - zoomed_w // 2, w - zoomed_w))
        y_start = max(0, min(y - zoomed_h // 2, h - zoomed_h))
        
        # Extract the zoomed region
        zoomed_region = image[y_start:y_start+zoomed_h, x_start:x_start+zoomed_w]
        
        resized_image = cv2.resize(zoomed_region, (int(w/(100/(window_size))), int(h/(100/(window_size)))))
        

        
        return resized_image
    
    except:
        
        height, width = image.shape[:2]
        
        resized_image = cv2.resize(image, (int(width/(100/(window_size))), int(height/(100/(window_size)))))
        
        return resized_image
            




def tiff_reduce_app(path_to_tiff, parent_window = None):
       
    if not os.path.exists(path_to_tiff):
        
        print('\nImage does not exist. Check the correctness of the path to image')
        return [], None
        
    else:
        
        # Initialize variables
        global app_metadata
        global image_list_red
        global tiff_file_red
        global tiff_file_red_return
        
    
        
        tmp = load_tiff(path_to_tiff=path_to_tiff)
        
        # reducing image for display
        
        tiff_file_red = resize_to_screen_tiff(tmp.copy(), factor = 4)
        
        del tmp
        
        tiff_file_red_return = tiff_file_red.copy()
        
        
        global window
        global zoom_factor
        global x
        global y
        
        x = 1 
        y =1
        zoom_factor = 1.0
        
        
        def prep_slice():
            global tiff_file_red
            
            for n, _ in enumerate(tiff_file_red):
                tmp = equalizeHist_16bit(tiff_file_red[n])
                tmp = ((tmp - np.mean(tmp)) * 3) + np.mean(tmp)
                tmp = np.clip(tmp, 0, 65535)
                
                tiff_file_red[n] = tmp
                
                del tmp
            
            
            
        
    
        def display_image():
            global window
            global zoom_factor
            global x
            global y
            
            
    
            resized_image = update_zoomed_region(tiff_file_red[int(image_slice.get()) - 1], size.get(), x, y)
            
            cv2.imshow('Tiff selection', resized_image)
            
            key = cv2.waitKey(100) & 0xFF
            if key == ord('z'):
                cv2.setMouseCallback('Tiff selection', zoom_in)
            else:
                cv2.setMouseCallback('Tiff selection', lambda *args: None)  
    
    
    
            window.after(100, display_image)
            
    
        
        def cls_win():
            global app_metadata
            global image_list_red
            global tiff_file_red
            global window
            
            app_metadata.saved_tiff_path = None
            
            image_list_red = []
            tiff_file_red = None
            
            window.destroy()
        
        
        
        
        def close_window():
            global image_list_red
            global tiff_file_red_return
            global window
            global app_metadata
            

            image_list_red = t.get("1.0", tk.END)
            image_list_red = re.sub(r"\s+", "", image_list_red)
            image_list_red = re.sub(r"\.", ",", image_list_red)
            image_list_red = re.sub(r"\n+", "", image_list_red)
            image_list_red = image_list_red.split(',')
            image_list_red = [item for item in image_list_red if item != ""]
            image_list_red = [(int(x) - 1) for x in image_list_red]
            image_list_red = list(set(image_list_red))
            
            
            if all(elem in list(range(tiff_file_red_return.shape[0])) for elem in image_list_red):
                if len(image_list_red) > 0:
                    tiff_file_red_return = tiff_file_red_return[[x for x in range(tiff_file_red_return.shape[0]) if x not in image_list_red]]
                    
                
                if app_metadata.removal_list == None:
                    app_metadata.add_rm_list(image_list_red)
                else:
                    image_list_red = list(set(app_metadata.removal_list + image_list_red))
                    app_metadata.add_rm_list(image_list_red)
    
                    
                
                
                cv2.destroyAllWindows()
        
                window.destroy()
            
            else:
                
                error_text = (
                    "The provided image numbers is not\n"
                    "included in the images list.\n"
                    "Please check the entered numbers!"
                )
                
                error_win(error_text, parent_window = None)
            
        
        # basic show
        
        if parent_window == None:
            window = tk.Tk()
        else:
            window = tk.Toplevel(parent_window)
        
        
        def validate_input(event):
            content = event.widget.get("1.0", tk.END).strip()
            
            if all(char.isdigit() or char in {',', ' ', '\n'} for char in content):
                return True
            else:
                event.widget.delete("end-2c", "end-1c")
                return False
            
        
        window.geometry("500x600")
        window.title("Z-selection")
    
        window.iconbitmap(os.path.join(_icon_source,'jbi_icon.ico'))
        
        txt1 = tk.Label(window, text="Slice selection", anchor="w", justify="left")
        txt1.pack()
       
        tk.Label(window, text="").pack()
        tk.Label(window, text="").pack()
        
        
        tk.Label(window, text="Window size", anchor="w").pack()
        
        # Create a slider widget
        
        size = tk.DoubleVar()
        slider1 = tk.Scale(window, from_=1, to=50, orient=tk.HORIZONTAL, length=400, variable=size)
        slider1.set(24)
        slider1.pack()
        
     
       
        tk.Label(window, text="").pack()
        
        
        tk.Label(window, text="Slice", anchor="w").pack()
        
        image_slice = tk.DoubleVar()
        slider2 = tk.Scale(window, from_=1, to=int(tiff_file_red.shape[0]) , orient=tk.HORIZONTAL, length=400, variable=image_slice)
        slider2.set(1)
        slider2.pack()
        
        tk.Label(window, text="").pack()
        
        
        label = tk.Label(window, text="Enter numbers of slices to remove:")
        label.pack()
        
        tk.Label(window, text="").pack()
        
        t = Text(window, height=8, width=50)
        t.bind("<KeyRelease>", validate_input)
        t.pack()
        
        display_image()
        
        tk.Label(window, text="").pack()
        
        button2 = tk.Button(window, text="Enhance", command=prep_slice, width=15, height=1)
        
        button2.pack()
        
        tk.Label(window, text="").pack()
        
        button2 = tk.Button(window, text="Apply", command=close_window, width=15, height=1)
        
        button2.pack()
        
        
        tk.Label(window, text="").pack()
        
        button3 = tk.Button(window, text="Back", command=cls_win, width=15, height=1)
        
        button3.pack()
    
    
    
        window.mainloop()
        
        cv2.destroyAllWindows()
        
      
    
        return image_list_red, tiff_file_red_return
    


    




def z_projection_app(path_to_tiff:str, reduced_tiff, rm_tiff, parent_window = None):

    if not os.path.exists(path_to_tiff):
        
        print('\nImage does not exist. Check the correctness of the path to image')
        return None
        
    
    
    global tiff_file_app
    
    tiff_file_app = reduced_tiff
    

    global returned_projection

    returned_projection = None
    
    global window_projection
    global zoom_factor
    global x
    global y
    x = 1 
    y = 1
    zoom_factor = 1.0
    
    def cls_win():
        global window_projection
        global returned_projection
        
        returned_projection = None
        window_projection.destroy()
    
    
    
    def close_window():
        global window_projection
        global returned_projection
        global tiff_file_app
    
                
        returned_projection = z_projection(tiff_object = tiff_file_app, projection_type = projections_type.get())
                
        
        if h_var.get() == True:
            returned_projection =  equalizeHist_16bit(returned_projection)


        if c_var.get() == True:
            returned_projection = clahe_16bit(returned_projection, kernal = (100, 100))
            
        
        returned_projection = adjust_img_16bit(img = returned_projection, color = combobox.get(), max_intensity = int(threshold_max.get()), min_intenisty = int(threshold_min.get()), brightness = int(brightness.get()), contrast = float(contrast.get()), gamma = float(gamma.get()))
                        

        window_projection.destroy()
        
        
    
          

    def active_changes():
        
        global tiff_file_app, stacked_app
                    
        
        stacked_app = z_projection(tiff_object = tiff_file_app, projection_type = projections_type.get())
        

        if h_var.get() == True:
            stacked_app =  equalizeHist_16bit(stacked_app)

        if c_var.get() == True:
            stacked_app = clahe_16bit(stacked_app, kernal = (100, 100))
            
            
  
        
        stacked_app = adjust_img_16bit(img = stacked_app ,color = combobox.get(), max_intensity = int(threshold_max.get()), min_intenisty = int(threshold_min.get()), brightness = int(brightness.get()), contrast = float(contrast.get()), gamma = float(gamma.get()))
              
        
    def reset__():
        slider2.set(1)
        slider3_min.set(0)
        slider3_max.set(int(65535))
        slider5.set(1000)
        slider6.set(1)
        c_var.set(False)
        h_var.set(False)

        active_changes()
        




        
    def display_image():
        global window_projection
        global zoom_factor
        global stacked_app
        global x
        global y


        
        
        
        resized_image = update_zoomed_region(stacked_app.copy(), size.get(), x, y)

        cv2.imshow('Z-projection',resized_image) 
        
        key = cv2.waitKey(100) & 0xFF
        if key == ord('z'):
            cv2.setMouseCallback('Z-projection', zoom_in)
        else:
            cv2.setMouseCallback('Z-projection', lambda *args: None)  


            
        window_projection.after(100, display_image)
        
  

  
    if parent_window == None:
        window_projection = tk.Tk()
    else:
        window_projection = tk.Toplevel(parent_window)   

    
    window_projection.geometry("500x710")
    window_projection.title("Z-projection")

    window_projection.iconbitmap(os.path.join(_icon_source, 'jbi_icon.ico'))
    
    
    tk.Label(window_projection, text="Window size", anchor="w").pack()



    # Create a slider widget        
    size = tk.DoubleVar()
    slider1 = tk.Scale(window_projection, from_=1, to=50, orient=tk.HORIZONTAL, length=400, variable=size)
    slider1.set(24)
    slider1.pack()
    
   
    
    
    tk.Label(window_projection, text="Gamma", anchor="w").pack()



    gamma = tk.DoubleVar()
    slider2 = tk.Scale(window_projection, from_=0, to=5, resolution=0.1, orient=tk.HORIZONTAL, length=400, variable=gamma)
    slider2.set(1)
    slider2.pack()
    
    
    tk.Label(window_projection, text="Threshold", anchor="w").pack()
    
    tk.Label(window_projection, text="Min", anchor="w").pack()
    
    threshold_min = tk.DoubleVar()
    slider3_min = tk.Scale(window_projection, from_=0, to=32767, orient=tk.HORIZONTAL, length=400, variable=threshold_min)
    slider3_min.set(0)
    slider3_min.pack()
    
    tk.Label(window_projection, text="Max", anchor="w").pack()
    
    threshold_max = tk.DoubleVar()
    slider3_max = tk.Scale(window_projection, from_=0, to=65535, orient=tk.HORIZONTAL, length=400, variable=threshold_max)
    slider3_max.set(int(65535))
    slider3_max.pack()
    
    
    tk.Label(window_projection, text="Brightness", anchor="w").pack()

    
    brightness = tk.DoubleVar()
    slider5 = tk.Scale(window_projection, from_=900, to=2000, orient=tk.HORIZONTAL, length=400, variable=brightness)
    slider5.set(1000)
    slider5.pack()
    
    

    
    tk.Label(window_projection, text="Contrast", anchor="w").pack()
    
    contrast = tk.DoubleVar()
    slider6 = tk.Scale(window_projection,  from_=0, to=5, resolution=0.1, orient=tk.HORIZONTAL, length=400, variable=contrast)
    slider6.set(1)
    slider6.pack()
    
    
    
    
    tk.Label(window_projection, text="Initialize image adjustment:").pack()
    
    
    h_var = tk.BooleanVar()
    h = tk.Checkbutton(window_projection, text="equalizeHis", variable=h_var)
    h.pack()
    
    
    c_var = tk.BooleanVar()
    c = tk.Checkbutton(window_projection, text="CLAHE", variable=c_var)
    c.pack()
    
            
    tk.Label(window_projection, text="Color", anchor="w").pack()
    
    items = ['gray', "blue", "green", "red", "magenta", 'yellow', 'cyan']

    combobox = ttk.Combobox(window_projection, values=items)
    
    combobox.current(0)
    
    combobox.pack()
    
    
    
    label4 = tk.Label(window_projection, text="Projection method", anchor="w")
    label4.pack()
    
    projections = ["avg", "max", "min", "std", "median"]

    projections_type = ttk.Combobox(window_projection, values=projections)
    
    projections_type.current(0)
    
    projections_type.pack()
        
 
    #######################################################################
    
    tk.Label(window_projection, text="").pack()

    
    button = tk.Button(window_projection, text="Apply", command=active_changes, width=15, height=1)

    button.pack()
    

    
    
    button2 = tk.Button(window_projection, text="Save", command=close_window, width=15, height=1)
    
    button2.pack()
    


    
    button3 = tk.Button(window_projection, text="Reset", command=reset__, width=15, height=1)
    
    button3.pack()
    

    
    button4 = tk.Button(window_projection, text="Back", command=cls_win, width=15, height=1)
    
    button4.pack()
    
       
    active_changes()
    
    display_image()
    
  
    window_projection.mainloop()

    cv2.destroyAllWindows()

    
    return returned_projection
    
    




def merge_images_app(image_list:list):
    
    global result_merge
    result_merge = None
    
    global window_merge
    global zoom_factor
    global x
    global y
    x = 1 
    y = 1
    zoom_factor = 1.0
    
    
    global final_list
    final_list = image_list
    
    global lower_list
    
    lower_list = []
    for img in image_list:
        lower_list.append(resize_to_screen_img(img, factor = 4))
    
    
    # basci merge 
    
    global merged_projection
    merged_projection =  merge_images(image_list = lower_list)
    
    
    def merge_inside():
        global merged_projection
        global lower_list
        
        intensity_factors = []
        for bt in range(len(lower_list)):
            intensity_factors.append((slider_values[str('b' + str(bt))].get()/10))
        
        merged_projection =  merge_images(image_list = lower_list, intensity_factors = intensity_factors)
        
    
    def cls_win():
        global window_merge
        window_merge.destroy()
        
     
    def close_window():
        
        global result_merge
        global window_merge
        global final_list
        global lower_list
        
        del lower_list
        
        
        intensity_factors = []
        for bt in range(len(image_list)):
            intensity_factors.append((slider_values[str('b' + str(bt))].get()/10))
        
        result_merge =  merge_images(image_list = final_list, intensity_factors = intensity_factors)
        
        
        window_merge.destroy()
        
        


    window_merge = tk.Tk()
    
    window_merge.geometry("500x600")
    window_merge.title("Merge channels")

    window_merge.iconbitmap(os.path.join(_icon_source,'jbi_icon.ico'))
   

   
    tk.Label(window_merge, text="").pack()
    tk.Label(window_merge, text="").pack()
    
    
    label1 = tk.Label(window_merge, text="Size", anchor="w")
    label1.pack()
    
    # Create a slider widget
    
    size = tk.DoubleVar()
    slider1 = tk.Scale(window_merge, from_=1, to=50, orient=tk.HORIZONTAL, length=400, variable=size)
    slider1.set(24)
    slider1.pack()
    
    

 
    
    
    tk.Label(window_merge, text="").pack()
    
    label2 = tk.Label(window_merge, text="Images intensity", anchor="w")
    label2.pack()
    
    slider_values = {}

    for bt in range(len(image_list)):
        slider_values[str('b' + str(bt))] = tk.DoubleVar()
        tk.Label(window_merge, text="").pack()
        tk.Label(window_merge, text=str("Img_" + str(bt)), anchor="w").pack()
        tk.Scale(window_merge, from_=0, to=20, orient=tk.HORIZONTAL, length=400, variable=slider_values[str('b' + str(bt))]).pack()
        slider_values[str('b' + str(bt))].set(10)

    


    tk.Label(window_merge, text="").pack()
    
    
    button = tk.Button(window_merge, text="Apply", command=merge_inside, width=15, height=1)
    
    button.pack()
    

        
        
    def display_image():
        global window_merge
        global zoom_factor
        global merged_projection
        global x
        global y


        
        resized_image = update_zoomed_region(merged_projection.copy(), size.get(), x, y)

        cv2.imshow('Merge images',resized_image) 
        
        
        key = cv2.waitKey(100) & 0xFF
        if key == ord('z'):
            cv2.setMouseCallback('Merge images', zoom_in)
        else:
            cv2.setMouseCallback('Merge images', lambda *args: None)  


        
            
        window_merge.after(100, display_image)
       
    
    display_image()
    
    
    tk.Label(window_merge, text="").pack()
    
    button2 = tk.Button(window_merge, text="Save", command=close_window, width=15, height=1)
    
    button2.pack()
    
    
    tk.Label(window_merge, text="").pack()
    
    button3 = tk.Button(window_merge, text="Back", command=cls_win, width=15, height=1)
    
    button3.pack()
    
    
    
    window_merge.mainloop()

    cv2.destroyAllWindows()
    
  
    
    return result_merge


        



def image_selection_app(input_image, img_length:int, img_width:int):
    
    
    global check_name
    
    if '_rotated_' in check_name:
        error_text = ('\nYou try use rotated image!\n'
                      'The queue of raw images may be different!\n'
                      'If you want to use rotated image be sure\n'
                      'that the image was transformed again to\n'
                      'the primary image position!')
        
        error_win(error_text, parent_window = None,  color= 'yellow', win_name= 'Warning')
        
        
    elif '_loaded' in check_name:
        error_text = ('\nYou try use loaded image!\n'
                      'The queue of raw images may be different!\n'
                      'If you want to use loaded image be sure\n'
                      'that the image is original image\n'
                      )
        
        error_win(error_text, parent_window = None,  color= 'yellow', win_name= 'Warning')

    
        
    global image
    global final_image
    
    
    global iml
    global imw
    
    imw = img_width
    iml = img_length
     

    
    image = input_image.copy()
    
    image = resize_to_screen_img(image, factor = 4)
    
    final_image = input_image.copy()


    
    global window_selection
    global zoom_factor
    global x
    global y
    x = 1 
    y = 1
    zoom_factor = 1.0
    
    
    def change_colors():
        
        global image
         
        global iml
        global imw
        
        image = image_grid_resize(image, img_length = iml, img_width = imw, grid_color = str(grid_box.get()), font_color = str(font_box.get()))

        
        
            
    def display_image():
        global window_selection
        global zoom_factor
        global image
        global x
        global y


        
        resized_image = update_zoomed_region(image.copy(), size.get(), x, y)

        cv2.imshow('Selection images',resized_image) 
        
        
        key = cv2.waitKey(100) & 0xFF
        if key == ord('z'):
            cv2.setMouseCallback('Selection images', zoom_in)
        else:
            cv2.setMouseCallback('Selection images', lambda *args: None)  


            
        window_selection.after(100, display_image)
        
        
    def win_cls():
        global image_list
        global final_image
        global window_selection
        
        image_list = []
        final_image = None
        window_selection.destroy()
        cv2.destroyAllWindows()
        
        
        

    
    def close_window():
        
        global window_selection
        global image_list
        global final_image
         
        global iml
        global imw
        
        final_image = image_grid_resize(final_image, img_length = iml, img_width = imw, grid_color = str(grid_box.get()), font_color = str(font_box.get()))


        image_list = t.get("1.0", tk.END)
        image_list = re.sub(r"\s+", "", image_list)
        image_list = re.sub(r"\.", ",", image_list)
        image_list = image_list.split(',')
        image_list = [item for item in image_list if item != ""]
        image_list = [int(x) for x in image_list]
        window_selection.destroy()
    

    def validate_input(event):
        content = event.widget.get("1.0", tk.END).strip()
        
        if all(char.isdigit() or char in {',', ' ', '\n'} for char in content):
            return True
        else:
            event.widget.delete("end-2c", "end-1c")
            return False
        
        
    window_selection = tk.Tk()
    
    window_selection.geometry("500x625")
    window_selection.title("Image selection")

    window_selection.iconbitmap(os.path.join(_icon_source,'jbi_icon.ico'))
    
    txt1 = tk.Label(window_selection, text="Images selection", anchor="w", justify="left")
    txt1.pack()
   
    tk.Label(window_selection, text="").pack()
    
    
    label1 = tk.Label(window_selection, text="Size", anchor="w")
    label1.pack()
    
    # Create a slider widget
    
    size = tk.DoubleVar()
    slider1 = tk.Scale(window_selection, from_=1, to=50, orient=tk.HORIZONTAL, length=400, variable=size)
    slider1.set(24)
    slider1.pack()
    
    
    tk.Label(window_selection, text="").pack()
    tk.Label(window_selection, text="Grid color", anchor="w").pack()
    
    grid_colors = ['white', "blue", "green", "red", "magenta", 'yellow', 'cyan']

    grid_box = ttk.Combobox(window_selection, values=grid_colors)
    
    grid_box.current(1)
    
    grid_box.pack()
    
    
    
    tk.Label(window_selection, text="").pack()
    tk.Label(window_selection, text="Font color", anchor="w").pack()
    
    font_colors = ['white', "blue", "green", "red", "magenta", 'yellow', 'cyan']

    font_box = ttk.Combobox(window_selection, values=font_colors)
    
    font_box.current(0)
    
    font_box.pack()
    

    
    tk.Label(window_selection, text="").pack()
    
    
    
    label = tk.Label(window_selection, text="Enter IDs of image:")
    label.pack()
    
    tk.Label(window_selection, text="").pack()
    
    t = Text(window_selection, height=10, width=50)
    t.bind("<KeyRelease>", validate_input)
    t.pack()
    
        
    
    
    tk.Label(window_selection, text="").pack()
    
    button1 = tk.Button(window_selection, text="Apply", command=change_colors, width=15, height=1)
    
    button1.pack()
    
    tk.Label(window_selection, text="").pack()

    button2 = tk.Button(window_selection, text="Save", command=close_window, width=15, height=1)
    
    button2.pack()
    
    
        
    tk.Label(window_selection, text="").pack()

    button3 = tk.Button(window_selection, text="Back", command=win_cls, width=15, height=1)
    
    button3.pack()
    

    change_colors()
    display_image()
    
    window_selection.mainloop()

    cv2.destroyAllWindows()
    


    return final_image, image_list
    





            
def add_scalebar(image, px_to_um, parent_window = None):
    
    

    global image2    
    global window_bar
    global zoom_factor
    global x
    global y
    x = 1 
    y = 1
    zoom_factor = 1.0
    
    global px
    
    global return_image
    return_image = None
    
    
    image2 = image

    
    
    
    def active_changes():
        
            global px
        
            px_ = re.sub(r'\n', '', px.get("1.0", tk.END))
            px_ = re.sub(r',', '.', px_)
            px_ = re.sub(r"\s+", "", px_)
            
            
            
            
            dec = True
            for char in px_:
                if not (char.isdigit() or char == '.'):
                    dec = False
                    break
                

           
            global image3
            global image2
            
            image3 = image2.copy()
            
            
            if dec == True and float(px_) > 0:
                
                px_ = float(px_)
                
                
                h = image3.shape[0]
                w = image3.shape[1]
                
                nw = None
                nh = None
                
                
                if w > 20000:
                    rw = 20000/w
                    
                    nw = int(w*rw)
                    nh = int(h*rw)
                    
                    image3 = cv2.resize(image3, (nw, nh))
                    px_ = px_*rw
                    h = image3.shape[0]
                    w = image3.shape[1]

                                       
                if h > 20000:
                    rh = 20000/h
                    
                    nw = int(w*rh)
                    nh = int(h*rh)
                    
                    image3 = cv2.resize(image3, (nw, nh))
                    px_ = px_*rh
                    h = image3.shape[0]
                    w = image3.shape[1]
                   
                                        
                if w != image3.shape[1] or h != image3.shape[0]:
                    print('Resolution of the image was too large')
                    print('Current resolution is ' + str(nw) + 'x' + str(nh))
                
                
                
                
            
                scale_length_um = int(length.get())  
                pixels_per_um = px_   
                scale_length_px = int(scale_length_um * pixels_per_um)
     
                # Calculate the position of the scale bar
                image_height = image3.shape[0]
                image_width = image3.shape[1]
                
    
                
                scale_position = (int(image_width*0.96) - scale_length_px, int(image_height*0.96))  
    
     
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
    
                global final_image
                
                final_image = image3.copy()
                
            elif dec == False:
                error_text = ('\nThe px ratio should be float or int number!')
                error_win(error_text, parent_window = window_bar)
        
                
            
            image3 = resize_to_screen_img(image3, factor = 4)
    
    
    
            
    def display_image():
        global window_bar
        global zoom_factor
        global image3
        global x
        global y


        
        resized_image = update_zoomed_region(image3, size.get(), x, y)

        cv2.imshow('Scale-bar',resized_image) 
        
        
        key = cv2.waitKey(100) & 0xFF
        if key == ord('z'):
            cv2.setMouseCallback('Scale-bar', zoom_in)
        else:
            cv2.setMouseCallback('Scale-bar', lambda *args: None)  


            
        window_bar.after(100, display_image)
        


    
    def close_window():
        
        global window_bar
        global return_image
        
        return_image = None
        
        window_bar.destroy()
        
        
        
    def save_():
        
        global window_bar
        global final_image
        global return_image
        
        
        return_image = final_image
        
        window_bar.destroy()
        
    

    ###########################################################################
        
    if parent_window == None:
        window_bar = tk.Tk()
    else:
        window_bar = tk.Toplevel(parent_window)
        

    window_bar.geometry("500x725")
    window_bar.title("Scale-bar")

    window_bar.iconbitmap(os.path.join(_icon_source,'jbi_icon.ico'))
   

    
    
    label1 = tk.Label(window_bar, text="Window size", anchor="w")
    label1.pack()
    
    # Create a slider widget
    
    size = tk.DoubleVar()
    slider1 = tk.Scale(window_bar, from_=0, to=50, orient=tk.HORIZONTAL, length=400, variable=size)
    slider1.set(24)
    slider1.pack()
    
    
    tk.Label(window_bar, text="").pack()

    
    txt1 = tk.Label(window_bar, text="Scale parameters", anchor="w", justify="left")
    txt1.pack()
    
    tk.Label(window_bar, text="").pack()

    
    tk.Label(window_bar, text="µm/px:").pack()
    px = tk.Text(window_bar, height=1, width=10)
    if px_to_um != None:
        px.insert(tk.END, str(px_to_um))
    else:
        px.insert(tk.END, "0")

    px.pack()
    
    tk.Label(window_bar, text="").pack()

    
    label2 = tk.Label(window_bar, text="Scale length [µm]", anchor="w")
    label2.pack()
    
    length = tk.DoubleVar()
    l = [50, 100, 250, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 10000]

    length = ttk.Combobox(window_bar, values=l)
    length.current(4)
    length.pack()
   
    tk.Label(window_bar, text="").pack()

    
    label3 = tk.Label(window_bar, text="Scale-bar thickness [px]", anchor="w")
    label3.pack()
    
    
    thickness = tk.DoubleVar()
    items = [10, 15, 25, 30, 35, 50, 75, 100, 150, 200, 250]

    thickness = ttk.Combobox(window_bar, values=items)
    thickness.current(3)
    thickness.pack()

    tk.Label(window_bar, text="").pack()

    label4 = tk.Label(window_bar, text="Color", anchor="w")
    label4.pack()
    
    items = ['white', 'grey', "blue", "green", "red", "magenta", 'yellow', 'cyan', 'black']

    combobox = ttk.Combobox(window_bar, values=items)
    
    combobox.current(0)
    
    combobox.pack()
    
    tk.Label(window_bar, text="").pack()

    label5 = tk.Label(window_bar, text="Font size", anchor="w")
    label5.pack()
    
    fonts = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,12,15,18,25]

    font = ttk.Combobox(window_bar, values=fonts)
    
    font.current(10)
    
    font.pack()
    
    
    tk.Label(window_bar, text="").pack()

    label6 = tk.Label(window_bar, text="Position", anchor="w")
    label6.pack()
    
    position_bar = ["right_bottom", "right_top", "left_top", "left_bottom"]

    position_type = ttk.Combobox(window_bar, values=position_bar)
    
    position_type.current(0)
    
    position_type.pack()
    
    tk.Label(window_bar, text="").pack()

    label7 = tk.Label(window_bar, text="Horizontal position", anchor="w")
    label7.pack()
    
    horizontal = [-15,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,15]

    hor = ttk.Combobox(window_bar, values=horizontal)
    
    hor.current(11)
    
    hor.pack()
    
    
    tk.Label(window_bar, text="").pack()

    label7 = tk.Label(window_bar, text="Vertical position", anchor="w")
    label7.pack()
    
    vertical = [-15,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,15]

    ver = ttk.Combobox(window_bar, values=vertical)
    
    ver.current(11)
    
    ver.pack()
    

    
    tk.Label(window_bar, text="").pack()
    
    button = tk.Button(window_bar, text="Apply", command=active_changes, width=15, height=1)
    

    
    button.pack()
    
   
    
    button2 = tk.Button(window_bar, text="Save", command=save_, width=15, height=1)
    
    button2.pack()
    
    
    
    
    button3 = tk.Button(window_bar, text="Back", command=close_window, width=15, height=1)
    
    button3.pack()
    
    
    
    
    active_changes()
    display_image()
    
    
    window_bar.mainloop()

    cv2.destroyAllWindows()

    
    return return_image






def draw_annotation(input_image, main_window = None):

    global image
    global final_image
    global final_shape
    global mask 
    global filled_mask
    
    
    final_image = input_image.copy()
    
    
    mask = np.zeros_like(final_image)
    
    
    final_shape = final_image.shape[:2]
    
    
    image = resize_to_screen_img(input_image.copy(), factor = 4)
    

    global window_annotation
    global zoom_factor
    global x
    global y
    global x_cord
    global y_cord
    
    global cords
    
    cords = []
    
    x_cord = None
    y_cord = None
    x = 1 
    y = 1
    zoom_factor = 1.0
    
    
    global cord_xy
    
    cord_xy = {'x':[],'y':[], 'xw':[], 'yw': []}
    
    
    def euclidean_distance(x1, y1, x2, y2):
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    
    def close_():
        global window_annotation
        global final_image, mask, filled_mask
        
        final_image, mask, filled_mask = None, None, None
        
        cv2.destroyAllWindows()
        window_annotation.destroy()
 
    
  
    
    def display_image():
        global window_annotation
        global zoom_factor
        global cord_xy
        global image
        global x
        global y
        global x_cord
        global y_cord
        global cords
        global l_color
        global final_shape
        global cur_siz
        global resized_image


            
        resized_image = update_zoomed_region(image.copy(), size.get(), x, y)


        cv2.imshow('Annotation images',resized_image) 
        
        
        key = cv2.waitKey(50) & 0xFF
        if key == ord('z'):
            cv2.setMouseCallback('Annotation images', zoom_in)
            
        elif key == ord('d'):
            cv2.setMouseCallback('Annotation images', get_pos)
            x_tmp, y_tmp, x_tmpw, y_tmpw = add_cord(image, final_shape, size.get(), x_cord, y_cord)
            if x_tmp != None and y_tmp != None:
                cord_xy['x'].append(int(x_tmp))
                cord_xy['y'].append(int(y_tmp))
                cord_xy['xw'].append(int(x_tmpw))
                cord_xy['yw'].append(int(y_tmpw))
                x_tmp = None
                y_tmp = None
                x_tmpw = None
                y_tmpw = None
                x_cord = None
                y_cord = None
                
                
                for i in range(len(cord_xy['x'])):
                    if i + 1 < max(range(len(cord_xy['x']))):
                        if i > 0 and euclidean_distance(cord_xy['x'][i], cord_xy['y'][i], cord_xy['x'][i + 1], cord_xy['y'][i + 1]) < euclidean_distance(cord_xy['x'][i-1], cord_xy['y'][i-1], cord_xy['x'][i], cord_xy['y'][i])*25:
                            cv2.line(image, (int(cord_xy['x'][i]), int(cord_xy['y'][i])), (int(cord_xy['x'][i+1]), int(cord_xy['y'][i+1])), (2**16, 2**16, 2**16), thickness=10)
            

        elif key == ord('q'):
            close_window()
            

            
        else:
           
            
            if str(line_box.get()) == 'grey':
                l_color = (32768, 32768, 32768)
            elif str(line_box.get()) == 'red':
                l_color = (0, 0, 65535)
            elif str(line_box.get()) == 'green':
                l_color = (0, 65535, 0)
            elif str(line_box.get()) == 'blue':
                l_color = (65535, 0, 0)
            elif str(line_box.get()) == 'magenta':
                l_color = (65535, 0, 65535)
            elif str(line_box.get()) == 'yellow':
                l_color = (0, 65535, 65535)
            elif str(line_box.get()) == 'cyan':
                l_color = (65535, 65535, 0)
            elif str(line_box.get()) == 'black':
                l_color = (0, 0, 0)
            else:
                l_color = (65535, 65535, 65535)
            

                
            cv2.setMouseCallback('Annotation images', lambda *args: None)
            if len(cord_xy['x']) > 0:
                   cords.append(cord_xy)
            cord_xy = {'x':[],'y':[], 'xw':[], 'yw': []}
            x_tmp = None
            y_tmp = None
            x_tmpw = None
            y_tmpw = None
            
            
            for c in cords:
                for i in range(len(c['x'])):
                    if i + 1 < max(range(len(c['x']))):
                        if i > 0 and euclidean_distance(c['x'][i], c['y'][i], c['x'][i + 1], c['y'][i + 1]) < euclidean_distance(c['x'][i-1], c['y'][i-1], c['x'][i], c['y'][i])*25:
                            cv2.line(image, (int(c['x'][i]), int(c['y'][i])), (int(c['x'][i+1]), int(c['y'][i+1])), l_color, thickness=int(line.get()))
            

            
            
            
        window_annotation.after(50, display_image)
       
        
       
    
    def del_():
        global cords
        global final_image
        global image
        global cord_xy
        
        if len(cords) > 0:
            cords = cords[0:len(cords)-1]
        image = resize_to_screen_img(final_image.copy(), factor = 4)
        cord_xy = {'x':[],'y':[], 'xw':[], 'yw': []}

        
    
    def close_window():
    
        global window_annotation
        global final_image
        global mask
        global filled_mask
        
        
        if len(cords) > 0:
            accept()
        else:
            final_image, mask, filled_mask = None, None, None


        cv2.destroyAllWindows()
        window_annotation.destroy()
        



    def accept():
        
        global cords
        global l_color
        global final_image
        global mask
        global image
        
        fc =  final_image.shape[1] / image.shape[1]
                
        
        
        for c in cords:
            for i in range(len(c['x'])):
                if i + 1 < max(range(len(c['x']))):
                    if i > 0 and euclidean_distance(c['xw'][i], c['yw'][i], c['xw'][i + 1], c['yw'][i + 1]) < euclidean_distance(c['xw'][i-1], c['yw'][i-1], c['xw'][i], c['yw'][i])*25:
                        cv2.line(final_image, (int(c['xw'][i]), int(c['yw'][i])), (int(c['xw'][i+1]), int(c['yw'][i+1])), l_color, thickness=int(line.get()*fc))
                        cv2.line(mask, (int(c['xw'][i]), int(c['yw'][i])), (int(c['xw'][i+1]), int(c['yw'][i+1])), l_color, thickness=int(line.get()*fc))


    # app
    global window_annotation
    
    if main_window is None:
        window_annotation = tk.Tk()
    else:
        window_annotation = tk.Toplevel(main_window)
    
    window_annotation.geometry("500x510")
    window_annotation.title("Annotation image")

    window_annotation.iconbitmap(os.path.join(_icon_source,'jbi_icon.ico'))
    
    txt1 = tk.Label(window_annotation, text="Images selection", anchor="w", justify="left")
    txt1.pack()
   
    tk.Label(window_annotation, text="").pack()
    
    
    label1 = tk.Label(window_annotation, text="Size", anchor="w")
    label1.pack()
    
    # Create a slider widget
    
    size = tk.DoubleVar()
    slider1 = tk.Scale(window_annotation, from_=1, to=50, orient=tk.HORIZONTAL, length=400, variable=size)
    slider1.set(23)
    slider1.pack()
    
    
    
    tk.Label(window_annotation, text="").pack()
    
    
    label2 = tk.Label(window_annotation, text="Line width", anchor="w")
    label2.pack()
    
    # Create a slider widget
    
    line = tk.DoubleVar()
    slider2 = tk.Scale(window_annotation, from_=10, to=50, orient=tk.HORIZONTAL, length=400, variable=line)
    slider2.set(10)
    slider2.pack()
    
    
    tk.Label(window_annotation, text="").pack()
    tk.Label(window_annotation, text="Line color", anchor="w").pack()
    
    line_colors = ['white', "blue", "green", "red", "magenta", 'yellow', 'cyan', 'grey']

    line_box = ttk.Combobox(window_annotation, values=line_colors)
    
    line_box.current(3)
    
    line_box.pack()
    
    
    
            
    
    tk.Label(window_annotation, text="").pack()
    
    tk.Button(window_annotation, text="Undo", command=del_, width=15, height=1).pack()
    

    
    tk.Label(window_annotation, text="").pack()
    
    tk.Button(window_annotation, text="Apply", command=accept, width=15, height=1).pack()

    
    
    tk.Label(window_annotation, text="").pack()

    tk.Button(window_annotation, text="Save", command=close_window, width=15, height=1).pack()


      
    tk.Label(window_annotation, text="").pack()

    tk.Button(window_annotation, text="Close", command=close_, width=15, height=1).pack()



    display_image()
    
    window_annotation.mainloop()
    

    cv2.destroyAllWindows()
    
    
    # creating full mask
    
    if isinstance(mask, np.ndarray):
    
        filled_mask = cv2.cvtColor(mask.copy(), cv2.COLOR_BGR2GRAY)
        
        filled_mask = cv2.convertScaleAbs(filled_mask, alpha=(255/65535))
    
        contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        filled_mask = cv2.fillPoly(filled_mask, pts = contours, color=(255,255,255))
         
        filled_mask = filled_mask.astype(np.uint16)*(65535/255)


    return final_image, mask, filled_mask
    





def tiff_annotation(path_to_images:str, image_list:list, image_dictinary, metadata, grid = None):
    
    global app_metadata
    
    
    global res_dic
    res_dic = None
    

    global img_names
    global final_images
    global final_annotation
    global final_masks
    global projection_images

    
    global il
    il = sorted(list(set(image_list)))
    
    
    
    global idic
    idic = image_dictinary
    
    
    il = [ocr for ocr in il if ocr in list(idic['image_num'])]

    
    img_names = il
    projection_images = [None]*len(il)
    final_images = [None]*len(il)
    final_annotation = [None]*len(il)
    final_masks = [None]*len(il)
    
    
    global ni
    
    ni = 0
    
    global nim
    
    nim = il[ni]
    
    
    global x
    global y
    global zoom_factor
    
    x = 1 
    y = 1
    zoom_factor = 1.0
    
    
    global pro_type
    global color_type
    global cluth_bool
    global his_bool
    global contrast_type
    global brightness_type
    global threshold_max_type
    global threshold_min_type
    global gamma_type
    
    pro_type = 'avg'
    color_type = 'gray'
    cluth_bool = False
    contrast_type = 1
    his_bool = False
    brightness_type = 1000
    threshold_max_type = 2**16 - 1
    threshold_min_type = 0
    gamma_type = 1
    
    global annotation_window
   

    
    def rm_im():
        global il
        global ni
        global nim
        global img_names
        global final_images
        global projection_images
        global final_masks
        global final_annotation
        global var_im
        
        if len(img_names) > 1:

            if ni - 1 >= 0:
                il = [l for l in il if l != nim]
                change_nim_down()
                img_names.pop(ni+1)
                final_images.pop(ni+1)
                projection_images.pop(ni+1)
                final_annotation.pop(ni+1)
                final_masks.pop(ni+1)
                
            else: 
                il = [l for l in il if l != nim]
                change_nim_up()
                img_names.pop(ni-1)
                final_images.pop(ni-1)
                projection_images.pop(ni-1)
                final_annotation.pop(ni-1)
                final_masks.pop(ni-1)
                # ni -= 1
                
            var_im.set(str(ni+1) + ' of ' + str(len(final_images)) + ' - ref.num. ' + str(nim))

    
    
    def close_save():
        global res_dic 
        global annotation_window
        global final_images
        global final_masks
        global img_names
        global final_annotation

        
        res_dic = {'images':final_images, 'projections':projection_images, 'annotations':final_annotation, 'masks':final_masks,'images_num':img_names}


        cv2.destroyAllWindows()
        annotation_window.destroy()

        

            
    
    def z_pro_tiff(images):
        
        global projection_images
        global ni
        global pro_type
        global color_type
        global cluth_bool
        global his_bool
        global contrast_type
        global brightness_type
        global threshold_max_type
        global threshold_min_type
        global gamma_type
        
 
        stacked = z_projection(tiff_object = images, projection_type = str(pro_type))
        
        
        if his_bool == True:
            stacked =  equalizeHist_16bit(stacked)
        

        if cluth_bool == True:
            stacked = clahe_16bit(stacked, kernal = (100, 100))
            

        stacked = adjust_img_16bit(img = stacked, color = str(color_type), max_intensity = int(threshold_max_type), min_intenisty = int(threshold_min_type), brightness = int(brightness_type), contrast = float(contrast_type), gamma = float(gamma_type))
                        
        projection_images[ni] = stacked
        
        return stacked
    
        
        
        
            
    def images_queue(images_lists, remove_list = app_metadata.removal_list):
    
        images = []
        for inx, img in enumerate(images_list):
            if None == remove_list:
                images.append(cv2.imread(os.path.join(path_to_images, img), cv2.IMREAD_ANYDEPTH))
            else:
                if inx not in remove_list:
                    images.append(cv2.imread(os.path.join(path_to_images, img), cv2.IMREAD_ANYDEPTH))
        
        images = np.stack(images)
        
        return images
        
    
    def images_selection(event=None):
        
        global nim
        global ni
        global zp
        global il
        global idic

        global pro_type
        global color_type
        global cluth_bool
        global his_bool
        global contrast_type
        global brightness_type
        global threshold_max_type
        global threshold_min_type
        global gamma_type
        global images_list
        global channel
        global app_metadata
       
        selected = idic[idic['image_num'].isin(il)]
        selected = selected.reset_index()
        
        
        app_metadata.add_channel(str(channel.get()))
       
        images_list=os.listdir(path_to_images)
        
        tmp_str = str(re.sub('\n','', (str(selected.loc[selected['image_num'] == nim, 'queue'].values[0]))) + 'p')
    
        images_list=[x for x in images_list if tmp_str in x]
        
        images_list = [x for x in images_list if re.sub('sk.*','',re.sub('.*-','',x)) in str(channel.get())]
        
        images = images_queue(images_list)
        
        zp = z_pro_tiff(images)
            
          
       
    def annotate():
        
        global zp
        global annotation_window
        global ni 
        global nim 
        global img_names
        global final_images
        global final_masks
        global final_annotation
        
        annotation_window.destroy()
        
        
        zp, annotation, full_mask  = draw_annotation(zp)
        
        if isinstance(zp, np.ndarray):
        
            img_names[ni] = nim
            final_images[ni] = zp
            final_masks[ni] = full_mask
            final_annotation[ni] = annotation

        
        main_win()
     
        
    def change_nim_up():
        global il
        global ni
        global nim
        global var_im
        if ni + 1 < len(il):
            ni += 1
            nim = il[ni]
            images_selection()
            
        var_im.set(str(ni+1) + ' of ' + str(len(final_images)) + ' - ref.num. ' + str(nim))
        
        
            
            
            
    def change_nim_down():
        global il
        global ni
        global nim
        global var_im

        if ni - 1 >= 0:
            ni -= 1
            nim = il[ni]
            images_selection()
            
        var_im.set(str(ni+1) + ' of ' + str(len(final_images)) + ' - ref.num. ' + str(nim))
            
         

    def display_image_ann():
        global annotation_window
        global zoom_factor
        global zp
        global x
        global y
        global after_id
        
        global image_names
        global final_images
        global ni
        global dis

  
 

        if isinstance(final_images[ni], np.ndarray):
            resized_image = update_zoomed_region(final_images[ni], size.get(), x, y)
        else:
            resized_image = update_zoomed_region(zp, size.get(), x, y)

        cv2.imshow('Annotation images',resized_image) 
        
        
        if isinstance(grid, np.ndarray):
            hg, wg = grid.shape[:2]
            display_grid = cv2.resize(grid.copy(), (int(int(wg/(100/(grid_size.get())))), int(int(hg/(100/(grid_size.get()))))))
            cv2.imshow('Grid', display_grid) 

            
        
        key = cv2.waitKey(100) & 0xFF
        if key == ord('z'):
            cv2.setMouseCallback('Annotation images', zoom_in)
        else:
            cv2.setMouseCallback('Annotation images', lambda *args: None)  


        annotation_window.after(100, display_image_ann)

        
        
                
                
    def projection_window(): 
        
        global anotation_window
        
         
        
        def apply_fun():
            

            
            global pro_type
            global color_type
            global cluth_bool
            global his_bool
            global contrast_type
            global brightness_type
            global threshold_max_type
            global threshold_min_type
            global gamma_type
             
             
            pro_type = projections_type.get()
            color_type = combobox.get()
            cluth_bool = c_var.get()
            his_bool = h_var.get()
            contrast_type = contrast.get()
            brightness_type = brightness.get()
            threshold_max_type = threshold_max.get()
            threshold_min_type = threshold_min.get()
            gamma_type = gamma.get()
            
            
            images_selection()
            
            
            
        def reset__():
            slider2.set(1)
            slider3_min.set(0)
            slider3_max.set(int(65535))
            slider5.set(1000)
            slider6.set(1)
            c_var.set(False)
            h_var.set(False)

            apply_fun()
               
            
        

        
        def clw():
            
            global pro_type
            global color_type
            global cluth_bool
            global his_bool
            global contrast_type
            global brightness_type
            global threshold_max_type
            global threshold_min_type
            global gamma_type
             
             
            pro_type = projections_type.get()
            color_type = combobox.get()
            cluth_bool = c_var.get()
            his_bool = h_var.get()
            contrast_type = contrast.get()
            brightness_type = brightness.get()
            threshold_max_type = threshold_max.get()
            threshold_min_type = threshold_min.get()
            gamma_type = gamma.get()
            
            
            images_selection()
            
            window_projection_tiff.destroy()
            cv2.destroyAllWindows()

            
            
                
        window_projection_tiff = tk.Toplevel(annotation_window)
        
        window_projection_tiff.geometry("500x625")
        window_projection_tiff.title("Annotation projection")
        
        window_projection_tiff.iconbitmap(os.path.join(_icon_source,'jbi_icon.ico'))
        
        
       
        
        tk.Label(window_projection_tiff, text="Gamma", anchor="w").pack()
        
        
        
        gamma = tk.DoubleVar()
        slider2 = tk.Scale(window_projection_tiff,  from_=0, to=5, resolution=0.1, orient=tk.HORIZONTAL, length=400, variable=gamma)
        slider2.set(1)
        slider2.pack()
        
        
        tk.Label(window_projection_tiff, text="Threshold", anchor="w").pack()
        
        tk.Label(window_projection_tiff, text="Min", anchor="w").pack()
        
        threshold_min = tk.DoubleVar()
        slider3_min = tk.Scale(window_projection_tiff, from_=0, to=32767, orient=tk.HORIZONTAL, length=400, variable=threshold_min)
        slider3_min.set(0)
        slider3_min.pack()
        
        tk.Label(window_projection_tiff, text="Max", anchor="w").pack()
        
        threshold_max = tk.DoubleVar()
        slider3_max = tk.Scale(window_projection_tiff, from_=0, to=65535, orient=tk.HORIZONTAL, length=400, variable=threshold_max)
        slider3_max.set(int(65535))
        slider3_max.pack()
        
                
        tk.Label(window_projection_tiff, text="Brightness", anchor="w").pack()
        
        
        brightness = tk.DoubleVar()
        slider5 = tk.Scale(window_projection_tiff, from_=900, to=2000, orient=tk.HORIZONTAL, length=400, variable=brightness)
        slider5.set(1000)
        slider5.pack()
                
        
        tk.Label(window_projection_tiff, text="Contrast", anchor="w").pack()
        
        contrast = tk.DoubleVar()
        slider6 = tk.Scale(window_projection_tiff,  from_=0, to=5, resolution=0.1, orient=tk.HORIZONTAL, length=400, variable=contrast)
        slider6.set(1)
        slider6.pack()
        
        
        
        tk.Label(window_projection_tiff, text="Initialize image adjustment:").pack()
        
        h_var = tk.BooleanVar()
        h = tk.Checkbutton(window_projection_tiff, text="equalizeHis", variable=h_var)
        h.pack()
        
        c_var = tk.BooleanVar()
        c = tk.Checkbutton(window_projection_tiff, text="CLAHE", variable=c_var)
        c.pack()
        
        
        tk.Label(window_projection_tiff, text="Color", anchor="w").pack()
        
        items = ['gray', "blue", "green", "red", "magenta", 'yellow', 'cyan']
        
        combobox = ttk.Combobox(window_projection_tiff, values=items)
        
        combobox.current(0)
        
        combobox.pack()
        
        
        
        label4 = tk.Label(window_projection_tiff, text="Projection method", anchor="w")
        label4.pack()
        
        projections = ["avg", "max", "min", "std", "median"]
        
        projections_type = ttk.Combobox(window_projection_tiff, values=projections)
        
        projections_type.current(0)
        
        projections_type.pack()
        
        
        #######################################################################
        
         
        tk.Label(window_projection_tiff, text="").pack()
        button = tk.Button(window_projection_tiff, text="Apply", command=apply_fun, width=15, height=1)
        
        button.pack()
        
        
        button2 = tk.Button(window_projection_tiff, text="Reset", command=reset__, width=15, height=1)
        
        button2.pack()
        
        
        button3 = tk.Button(window_projection_tiff, text="Close", command=clw, width=15, height=1)
        
        button3.pack()
        
       
        window_projection_tiff.mainloop()
       
       
       
    def main_win():  
        
        global ni
        global final_images
        global var_im
        global channel
        global annotation_window
        
        
        def win_close():
            
            global annotation_window
            
            annotation_window.destroy()
        
        global size
        global grid_size
        global dis
        global annotation_window
        

        annotation_window = tk.Tk()
        
        annotation_window.geometry("500x670")
        annotation_window.title("Annotation")
        
        annotation_window.iconbitmap(os.path.join(_icon_source,'jbi_icon.ico'))
        
                
        tk.Label(annotation_window, text="").pack()

        
        label1 = tk.Label(annotation_window, text="Image size", anchor="w")
        label1.pack()
        
        
        # Create a slider widget
        
        size = tk.DoubleVar()
        slider1 = tk.Scale(annotation_window, from_=1, to=50, orient=tk.HORIZONTAL, length=400, variable=size)
        slider1.set(23)
        slider1.pack()
        
        
          
        tk.Label(annotation_window, text="").pack()
        
        label2 = tk.Label(annotation_window, text="Grid size", anchor="w")
        label2.pack()
        
        
        # Create a slider widget
        
        grid_size = tk.DoubleVar()
        slider1 = tk.Scale(annotation_window, from_=1, to=50, orient=tk.HORIZONTAL, length=400, variable=grid_size)
        slider1.set(23)
        slider1.pack()
        
        
        #####
        tk.Label(annotation_window, text="").pack()
        
        tk.Label(annotation_window, text="").pack()

        
        tk.Label(annotation_window, text="Image number:").pack()

        var_im = tk.StringVar()

        var_im.set(str(ni+1) + ' of ' + str(len(final_images)) + ' - ref.num. ' + str(nim))
        
        tk.Label(annotation_window, textvariable=var_im).pack()
        
        
        #######
        
        
        tk.Label(annotation_window, text="").pack()

        
        tk.Label(annotation_window, text="Channnel:").pack()
        
        chennels = ['ch' + re.sub('\n', '', x) for x in app_metadata.metadata['channel_number']]

        channel = ttk.Combobox(annotation_window, values=chennels)
        
        if app_metadata.channel != None:
            for n, i in enumerate(chennels):
                if i == app_metadata.channel:
                    break
                
            channel.current(n)
        
        else:
                
            channel.current(0)
        
        channel.pack()
        
        
        channel.bind("<<ComboboxSelected>>", images_selection)

        
        
        
        tk.Label(annotation_window, text="").pack()
        button = tk.Button(annotation_window, text="Adjust", command=projection_window, width=15, height=1)
        
        button.pack()
        
        
        tk.Label(annotation_window, text="").pack()
        button = tk.Button(annotation_window, text="Annotate", command=annotate, width=15, height=1)
        
        button.pack()
         
         
        tk.Label(annotation_window, text="").pack()
        button = tk.Button(annotation_window, text="Next", command=change_nim_up, width=15, height=1)
        
        button.pack()
         
         
        tk.Label(annotation_window, text="").pack()
        button = tk.Button(annotation_window, text="Previous", command=change_nim_down, width=15, height=1)
        
        button.pack()
        
        
        
        tk.Label(annotation_window, text="").pack()
        button = tk.Button(annotation_window, text="Discard", command=rm_im, width=15, height=1)
        
        button.pack()
        

        
        tk.Label(annotation_window, text="").pack()
        
        button2 = tk.Button(annotation_window, text="Save", command=close_save, width=15, height=1)
        
        button2.pack()
        
        
        
        tk.Label(annotation_window, text="").pack()
        
        button3 = tk.Button(annotation_window, text="Back", command=win_close, width=15, height=1)
        
        button3.pack()
                
        images_selection()
        
        display_image_ann()
       
        annotation_window.mainloop()
    
        cv2.destroyAllWindows()


    main_win()
   
    return res_dic

#########################################################################################

#########################################################################################
# information / warnings / error window

def error_win(error_text, parent_window = None, color = 'tomato', win_name = 'Error'):
    
    if parent_window == None:
        error_win_ = tk.Tk()
    else:
        error_win_ = tk.Toplevel(parent_window)
    
   
    def close_win_():
        error_win_.destroy()
        
    
    error_win_.configure(bg=color)

    error_win_.geometry("450x250")
    error_win_.title(win_name)
    
    error_win_.attributes("-topmost", True)

    error_win_.iconbitmap(os.path.join(_icon_source,'jbi_icon.ico'))

    tk.Label(error_win_, text="", bg = color).pack()
    

    tk.Label(error_win_, text=error_text, anchor="w", justify="left", bg = color,  font=('Helvetica', 10, 'bold')).pack()
    
    tk.Label(error_win_, text="", bg = color).pack()
    tk.Label(error_win_, text="", bg = color).pack()


    tk.Button(error_win_, text="OK", command=close_win_, width=6, height=2, bg=color).pack()
    
    
    error_win_.mainloop()
    
 
    


############################################################### --WINDOWS -- ###############################################################
# metadata loading

def metadata_window():
    
    global app_metadata
    global app_run
    global meta_win
    
    
    global zoom_factor
    global x
    global y
    x = 1 
    y = 1
    zoom_factor = 1.0
    
    
    
    def exit_win():
        
        global meta_win
        global reduction_win
        cv2.destroyAllWindows()

        try:
            meta_win.destroy()
        except: 
            pass
        
        try:
            reduction_win.destroy()
        except:
            pass

    def metadata_inputr():
        
        global app_metadata
        
        metadata_path = filedialog.askopenfilename()
        if metadata_path:
            input_box.delete(0, 'end')  
            input_box.insert('end', metadata_path)
            app_metadata.add_metadata_path(metadata_path)
            
            
            
    def repair_and_display():
        if True in np.unique(app_metadata.metadata_path != None) and True in np.unique(app_metadata.xml != None):
            try:
                xml_file, fig = repair_image(app_metadata.xml, dispaly_plot = True)
                app_metadata.add_xml(xml_file)
            except:
                error_text = ('\nYou must first provide the path\n'
                             'to the Index file and load it!!!')
                error_win(error_text, parent_window = None)
        else:
            error_text = ('\nYou must first provide the path\n'
                         'to the Index file and load it!!!')
            error_win(error_text, parent_window = None)
                
            
            
            
            
    def manual_adjustemnt():
        if True in np.unique(app_metadata.metadata_path != None) and True in np.unique(app_metadata.xml != None):
            
            
            cv2.destroyAllWindows()
            meta_win.destroy()
            
            
            def reduce_():
                global app_metadata
                global cores_fig
                global xml_tmp
                global reduction_list
                
                
                out_list = reduction_list.get("1.0", tk.END)
                out_list = re.sub(r"\s+", "", out_list)
                out_list = re.sub(r"\.", ",", out_list)
                out_list = out_list.split(',')
                out_list = [item for item in out_list if item != ""]
                out_list = [int(x) for x in out_list]
                
                
                xml_tmp, cores_fig = manual_outlires(app_metadata.tmp_xml, out_list, dispaly_plot = False)
                
                app_metadata.add_tmp_xml(xml_tmp)
                
                cores_fig = FigureCanvas(cores_fig)
                cores_fig.draw()
                cores_fig = np.array(cores_fig.renderer.buffer_rgba())
                cores_fig = resize_to_screen_img(cores_fig, factor = 2)
                
                
            def reset_():
                global app_metadata
                global cores_fig

                app_metadata.add_tmp_xml(app_metadata.xml)
                _, cores_fig = manual_outlires(app_metadata.tmp_xml, list_of_out = [], dispaly_plot = False)   
                
                cores_fig = FigureCanvas(cores_fig)
                cores_fig.draw()
                cores_fig = np.array(cores_fig.renderer.buffer_rgba())
                cores_fig = resize_to_screen_img(cores_fig, factor = 2)
               
                

            
                        
            def display_image():
                global reduction_win
                global app_metadata
                global cores_fig
                global zoom_factor
                global x
                global y

                cv2.destroyAllWindows()
                
                _, cores_fig = manual_outlires(app_metadata.tmp_xml, list_of_out = [], dispaly_plot = False)
                
                cores_fig = FigureCanvas(cores_fig)
                cores_fig.draw()
                cores_fig = np.array(cores_fig.renderer.buffer_rgba())
                cores_fig = resize_to_screen_img(cores_fig, factor = 2)

                
                def dis_loop():
                    global cores_fig
                    global zoom_factor
                    global x
                    global y
                    
                    t = True
             
                    resized_image = update_zoomed_region(cores_fig.copy(), size.get(), x, y)
             
                    cv2.imshow('Display', resized_image) 
                    
                    key = cv2.waitKey(50) & 0xFF
                    if key == ord('z'):
                        cv2.setMouseCallback('Display', zoom_in)
                    elif cv2.getWindowProperty('Display',cv2.WND_PROP_VISIBLE) < 1: 

                        reduction_win.destroy()
                        cv2.destroyAllWindows()
                        
                        t = False

                   
                    
                    if t == True:
                        reduction_win.after(1, dis_loop)
                
                dis_loop()
                    

            def win_cls():
                global reduction_win
                cv2.destroyAllWindows()
                reduction_win.destroy()
                met_win_()
                
                
            def save_changes():
                global app_metadata
                global meta_win
                global reduction_win

                
                
                xml_tmp, _ = repair_image(app_metadata.tmp_xml, dispaly_plot = False)
                
                app_metadata.add_xml(xml_tmp)
                app_metadata.add_tmp_xml(None)
                cv2.destroyAllWindows()
                reduction_win.destroy()
                
                error_text = ('\nResults saved successfully!!!')
                error_win(error_text, parent_window = None, color= 'green', win_name='Information')
                
                met_win_()

                    

    

            def window_reduce_():
                global reduction_win
                global size
                global reduction_list

                
                reduction_win = tk.Tk()
                
                reduction_win.geometry("500x625")
                reduction_win.title("Cores reducing")
            
                reduction_win.iconbitmap(os.path.join(_icon_source,'jbi_icon.ico'))
                
                
                 
                tk.Label(reduction_win, text="", anchor="w").pack()
                
                tk.Label(reduction_win, text="Window size", anchor="w").pack()
                
                # Create a slider widget
                
                size = tk.DoubleVar()
                slider1 = tk.Scale(reduction_win, from_=1, to=50, orient=tk.HORIZONTAL, length=400, variable=size)
                slider1.set(24)
                slider1.pack()
                
            
                tk.Label(reduction_win, text="").pack()
                
          

                tk.Label(reduction_win, text="").pack()
                
                
                label = tk.Label(reduction_win, text="Enter IDs of cores to reduce:")
                label.pack()
                
                tk.Label(reduction_win, text="").pack()
                
                reduction_list = Text(reduction_win, height=15, width=50)
                reduction_list.pack()
                   
                
                tk.Label(reduction_win, text="").pack()
                
                button1 = tk.Button(reduction_win, text="Reduce", command=reduce_, width=15, height=1)
                
                button1.pack()
                
                tk.Label(reduction_win, text="").pack()
            
                button2 = tk.Button(reduction_win, text="Return", command=reset_, width=15, height=1)
                
                button2.pack()
                
                
                tk.Label(reduction_win, text="").pack()
            
                button3 = tk.Button(reduction_win, text="Save", command=save_changes, width=15, height=1)
                
                button3.pack()
                
                
                tk.Label(reduction_win, text="").pack()
            
                button3 = tk.Button(reduction_win, text="Back", command=win_cls, width=15, height=1)
                
                button3.pack()
                
                reset_()

                display_image()
                
                reduction_win.mainloop()
            
                cv2.destroyAllWindows()
                
                
            window_reduce_()
                
            
            
           
        else:
            error_text = ('\nYou must first provide the path\n'
                         'to the Index file and load it!!!')
            error_win(error_text, parent_window = None)
    
            
            
    
    def load_met():
        
        global app_metadata
        if True in np.unique(app_metadata.metadata_path != None):
            try:
                xlm_data, metadata_data = xml_load(app_metadata.metadata_path)
                app_metadata.add_metadata(metadata_data)
                app_metadata.add_xml(xlm_data)
                
                
                del xlm_data, metadata_data
    
        
            except:
                error_text = ('\nYou must first provide the path\n'
                             'to the Index file and load it!!!')
                error_win(error_text, parent_window = None)
        else:
            error_text = ('\nYou must first provide the path\n'
                         'to the Index file and load it!!!')
            error_win(error_text, parent_window = None)
            
    
    def met_win_():
        global input_box
        global app_metadata
        global meta_win
        
        meta_win = tk.Tk()
        
        meta_win.geometry("500x400")
        meta_win.title("Load metadata")
    
        meta_win.iconbitmap(os.path.join(_icon_source,'jbi_icon.ico'))
    
        tk.Label(meta_win, text="").pack()
        
        tk.Label(meta_win, text="Path to the Index file:").pack()
    
    
        input_box = tk.Listbox(meta_win, width=70, height=1)
        input_box.pack(pady=1)
        
        if app_metadata.metadata_path != None:
            input_box.delete(0, 'end')  
            input_box.insert('end', app_metadata.metadata_path)
        
        tk.Label(meta_win, text="").pack()
        
        button1 = tk.Button(meta_win, text="Browse\npath", command=metadata_inputr, width=20, height=2)
        button1.pack()
        
        tk.Label(meta_win, text="").pack()
        
        
        
        
        button2 = tk.Button(meta_win, text="Load\nmetadata", command=load_met, width=20, height=2)
        button2.pack()
        
        
        
        tk.Label(meta_win, text="").pack()
    
        
    
        button3 = tk.Button(meta_win, text="Display\ncore", command=repair_and_display, width=20, height=2)
        button3.pack()
    
        tk.Label(meta_win, text="").pack()
    
    
    
        button4 = tk.Button(meta_win, text="Core\nreduce", command=manual_adjustemnt, width=20, height=2)
        button4.pack()
        
        
        tk.Label(meta_win, text="").pack()
    
    
    
        button5 = tk.Button(meta_win, text="Back", command=exit_win, width=20, height=2)
        button5.pack()
        
        
    
        meta_win.mainloop()
        
        
    met_win_()






# images concatenation

def concatenate_window():
    global app_run
    global app_metadata
    
   
        
    def exit_win():
        
        con_win.destroy()   
    
    
    def browse_inputr():
        
        global input_path
        input_path = filedialog.askdirectory()
        if input_path:
            input_box.delete(0, 'end')  
            input_box.insert('end', input_path)
            app_metadata.add_tiffs_path(input_path)
         

    def browse_save():
        
        global save_path
        save_path = filedialog.askdirectory()
        if save_path:
            save_box.delete(0, 'end') 
            save_box.insert('end', save_path)
            app_metadata.add_concat_path(save_path)

         
        
        
    def conc():
        
        overlap = overlap_val.get("1.0", tk.END)
        overlap = re.sub(',', '.', overlap)
        overlap = str(overlap.replace(' ', ''))
        overlap = str(overlap.replace('\n', ''))

        

        n_proc = cores.get("1.0", tk.END)
        n_proc = re.sub(',', '.', n_proc)
        n_proc = str(n_proc.replace(' ', ''))
        n_proc = str(n_proc.replace('\n', ''))

        
        
        resize = int(res.get())
        
        
        selected_indices = listbox.curselection()
        channels = [listbox.get(index) for index in selected_indices]
        

        dec = True
        for char in overlap:
            if not (char.isdigit() or char == '.'):
                dec = False
                break
        
        
        dec2 = True
        for char in n_proc:
            if not (char.isdigit()):
                dec2 = False
                break


            
        if dec == True and dec2 == True and app_metadata.tiffs_path != None and app_metadata.concat_path != None and len(channels) > 0:
            
            progress_var.set(10)
            progress_bar.update()

            def run_concatenate():
                dic, img_length, img_width = image_sequences(app_metadata.xml)
                image_concatenate(str(app_metadata.tiffs_path), 
                                  str(app_metadata.concat_path), 
                                  dic, 
                                  app_metadata.metadata, 
                                  img_length, 
                                  img_width, 
                                  float(overlap), 
                                  channels, 
                                  resize, 
                                  int(n_proc), 
                                  str('threads'))


                con_win.after(0, lambda: progress_var.set(100))  
                con_win.after(0, lambda: progress_bar.update())  

            threading.Thread(target=run_concatenate).start()
            
         
        
        elif dec == False:

            error_text = ('\nYou must provide the images overlap value (float number)\n'
                          'which was set previously in the image-creating system!!!')
            error_win(error_text, parent_window = con_win)
            
        elif dec2 == False:
            error_text = ('\nYou must provide the right number of threads!!!')
            error_win(error_text, parent_window = con_win)
            
        elif app_metadata.tiffs_path == None:
            error_text = ('\nProvide the path to the image!!!')
            error_win(error_text, parent_window = con_win)
            
        elif app_metadata.concat_path == None:
            error_text = ('\nProvide the path to save!!!')
            error_win(error_text, parent_window = con_win)
        
        elif len(channels) == 0:
            error_text = ('\nSelect channels!!!')
            error_win(error_text, parent_window = con_win)
    


    con_win = tk.Tk()
    

    con_win.geometry("500x670")
    con_win.title("Concatenate images")

    con_win.iconbitmap(os.path.join(_icon_source,'jbi_icon.ico'))

    
    tk.Label(con_win, text="Path to the raw images folder:").pack()


    input_box = tk.Listbox(con_win, width=70, height=1)
    input_box.pack(pady=1)
    
    if app_metadata.tiffs_path != None:
        input_box.delete(0, 'end')  
        input_box.insert('end', app_metadata.tiffs_path)
    

    button1 = tk.Button(con_win, text="Browse\ninput", command=browse_inputr, width=20, height=2)
    button1.pack()
    
    tk.Label(con_win, text="").pack()
    
    
    
    tk.Label(con_win, text="Path to save *.tiff file:").pack()


    
    save_box = tk.Listbox(con_win, width=70, height=1)
    save_box.pack(pady=1)
    
    
    if app_metadata.concat_path != None:
        save_box.delete(0, 'end')  
        save_box.insert('end', app_metadata.concat_path)
    
    button2 = tk.Button(con_win, text="Browse\nsave", command=browse_save, width=20, height=2)
    button2.pack()
    
    tk.Label(con_win, text="").pack()
    
    
    
    tk.Label(con_win, text="Settings for the images concatenation :").pack()
    
    
    
    tk.Label(con_win, text="Images overlap value:").pack()
    overlap_val = tk.Text(con_win, height=1, width=10)
    overlap_val.insert(tk.END, "0")
    overlap_val.pack()
    
    tk.Label(con_win, text="").pack()

    tk.Label(con_win, text="Select channels:").pack()


    channels = []
    for ch in app_metadata.metadata['channel_number']:
        ch = re.sub('\n', '', ch)
        channels.append('ch'+ch)
        
    listbox = tk.Listbox(con_win, selectmode=tk.MULTIPLE, exportselection=False, height=4)
    for item in channels:
        listbox.insert(tk.END, item)
    listbox.pack(pady=5)
    

        
    tk.Label(con_win, text="Number of threads:").pack()
    cores = tk.Text(con_win, height=1, width=10)
    
    prop = int(get_number_of_cores()*0.75)
    
    cores.insert(tk.END, str(prop))
    cores.pack()
    
    
    tk.Label(con_win, text="").pack()
    
    tk.Label(con_win, text="Resize factor for concatenated image:").pack()
    
    res = ttk.Combobox(con_win, values=[1,2,3,4,5,6,7,8])
    res.current(0)
    res.pack()
    
    
    tk.Label(con_win, text="").pack()


    button3 = tk.Button(con_win, text="Start\nconcatenate", command=conc, width=20, height=2)
    button3.pack()

    
    progress_var = tk.IntVar()
    progress_bar = ttk.Progressbar(con_win, variable=progress_var, mode="determinate", maximum=100, length=300)
    progress_bar.pack(pady=10)


    

    button5 = tk.Button(con_win, text="Back", command=exit_win, width=20, height=2)
    button5.pack()
    

    con_win.mainloop()
    
        
        
       

        
        

    
# projection window


def tiff_win_app():
    
    global app_run
    global app_metadata
    global linf
    
    linf = False
     
    def exit_win():
        
        tiff_win.destroy()    
    
    
    def browse_tiff():
        
        global input_path
        input_path = filedialog.askopenfilename()
        if input_path:
            if '.tiff' in input_path or '.tif' in input_path:
                tiff_box.delete(0, 'end')  
                tiff_box.insert('end', input_path)
                app_metadata.add_saved_tiff_path(input_path)
            else:
                error_text = ('\nProvide the path to the *.tiff image!!!')
                error_win(error_text, parent_window = tiff_win)
         
    
        
    def load():
        global reduced_tiffs
        global tiff_win
        global current_metadata
        global linf
        
        
            
        if app_metadata.saved_tiff_path != None:
            
            tiff_win.destroy()
            
            reduced_tiffs = tiff_reduce_app(app_metadata.saved_tiff_path, parent_window = None)
            
            try:
                z, y, x = read_tiff_meta(app_metadata.saved_tiff_path)
                current_metadata = {'X_resolution[um/px]':x, 'Y_resolution[um/px]':y, 'spacing':z , 'unit': 'um'}
                
            except:
                current_metadata = {'X_resolution[um/px]':None, 'Y_resolution[um/px]':None, 'spacing':None , 'unit': None}

            linf = True
            main_win()
            
            

        else:
            error_text = ('\nProvide the path to the *.tiff image!!!')
            error_win(error_text, parent_window = tiff_win)
            
     
    def project():
        global linf
        global reduced_tiffs
        global tiff_win
        global app_metadata
        global current_metadata
        
        if linf == True and app_metadata.saved_tiff_path != None:

            tiff_win.destroy()
    
            projection = z_projection_app(path_to_tiff = app_metadata.saved_tiff_path, reduced_tiff = reduced_tiffs[1], rm_tiff=reduced_tiffs[0], parent_window = None)
            
            if isinstance(projection, np.ndarray):
                img_name = os.path.basename(app_metadata.saved_tiff_path)
                img_name = re.sub(r'\.[^.]*$', '', img_name) 
                
                n = 0
                while(True):
                    n += 1
                    tmp_img_name = img_name + '_projection' + str(n)
                    if tmp_img_name not in app_metadata.images_dict['img_name']:
                        break
        
                app_metadata.add_image(projection, tmp_img_name, current_metadata)
            
            main_win()
            
        else:
            
            error_text = ('\nYou must first load *.tiff!!!')
            error_win(error_text, parent_window = tiff_win)
            
            
            
    

    def main_win():
        global tiff_win
        global tiff_box

        tiff_win = tk.Tk()
            
        tiff_win.geometry("500x380")
        tiff_win.title("Tiff load")
    
        tiff_win.iconbitmap(os.path.join(_icon_source,'jbi_icon.ico'))
    
        tk.Label(tiff_win, text="").pack()
        
        tk.Label(tiff_win, text="").pack()


        tk.Label(tiff_win, text="Path to the *.tiff file:").pack()
    
    
        tiff_box = tk.Listbox(tiff_win, width=70, height=1)
        tiff_box.pack(pady=1)
        
        if app_metadata.saved_tiff_path != None:
            tiff_box.delete(0, 'end')  
            tiff_box.insert('end', app_metadata.saved_tiff_path)
            
            
        tk.Label(tiff_win, text="").pack()

        
        button1 = tk.Button(tiff_win, text="Browse\ninput", command=browse_tiff, width=20, height=2)
        button1.pack()
        
        tk.Label(tiff_win, text="").pack()
        
    
        button2 = tk.Button(tiff_win, text="Load", command=load, width=20, height=2)
        button2.pack()
    
        tk.Label(tiff_win, text="").pack()
        
        
        button3 = tk.Button(tiff_win, text="Projection", command=project, width=20, height=2)
        button3.pack()
    
        tk.Label(tiff_win, text="").pack()
        
    
    
        button4 = tk.Button(tiff_win, text="Back", command=exit_win, width=20, height=2)
        button4.pack()
        
        
    
    
        tiff_win.mainloop()
        
    main_win()
    
    
    
# images manager


def img_manager_win():
     
    global app_run
    global app_metadata
    global load_win
    
    
    global zoom_factor
    global x
    global y
    x = 1 
    y = 1
    zoom_factor = 1.0
    
    
    def save_image():
        
        
        if len(file_listbox.curselection()) > 0:
            
            app_metadata.add_save_current(file_listbox.get(file_listbox.curselection()[0]))
        
            
            global load_win
            global lab_name
            
            def exit_win():
                sv_win.destroy()
                
             
            def save_browse():
                
                save_path = filedialog.askdirectory()
                if save_path:
                    sv_box.delete(0, 'end')  
                    sv_box.insert('end', save_path)
                    app_metadata.add_tmp_path(save_path)
                
                
            def im_save():
                global lab_name
                global app_metadata
                

                
                file = lab_name.get("1.0", tk.END)
                file = re.sub(r'\n', '', file)
                file = re.sub(r'\s', '_', file)
                
                if len(file) > 0 and app_metadata.tmp_path != None:
                
                    init_path = os.getcwd()
                    
                    os.chdir(app_metadata.tmp_path)
                    
                    
                    n = 0
                    while(True):
                        n += 1
                        tmp_file = file + '_' + str(n) + '.' + str(type_box.get())
                        if os.path.exists(tmp_file) == False:
                            break
                    
    
                    cv2.imwrite(tmp_file, app_metadata.images_dict['img'][app_metadata.images_dict['img_name'].index(app_metadata.save_current)])
                    
                    
                    

                    
                    os.chdir(init_path) 
                    
                    error_text = ('\nResults saved successfully!!!')
                    error_win(error_text, parent_window = None, color= 'green', win_name='Information')
                        

                    
                elif len(file) == 0:
                    error_text = ('\nProvide the file name!!!')
                    error_win(error_text, parent_window = sv_win)
                
                else:
                    error_text = ('\nProvide the path to the directory!!!')
                    error_win(error_text, parent_window = sv_win)
    
    

            
            sv_win = tk.Toplevel(load_win)
        
            sv_win.geometry("500x400")
            sv_win.title("Save image")
        
            sv_win.iconbitmap(os.path.join(_icon_source,'jbi_icon.ico'))
            
            
            sv_win.transient(load_win)

            sv_win.grab_set()
            
            
            tk.Label(sv_win, text="", anchor="w").pack()
            
            tk.Label(sv_win, text="Path to the save directory:").pack()
        
        
            sv_box = tk.Listbox(sv_win, width=70, height=1)
            sv_box.pack(pady=1)
            
            if app_metadata.tmp_path != None:
                sv_box.delete(0, 'end')  
                sv_box.insert('end', app_metadata.tmp_path)
            
    
            
            button1 = tk.Button(sv_win, text="Browse", command=save_browse, width=20, height=2)
            button1.pack()
            
            tk.Label(sv_win, text="").pack()
            
            
            label = tk.Label(sv_win, text="File name:")
            label.pack()
            
            tk.Label(sv_win, text="").pack()
            
            lab_name = Text(sv_win, height=1, width=50)
            lab_name.pack()
            
            
            tk.Label(sv_win, text="").pack()
    
            img_type = ['png', 'tiff', 'tif']
    
            type_box = ttk.Combobox(sv_win, values=img_type)
            
            type_box.current(0)
            
            type_box.pack()
            
            tk.Label(sv_win, text="").pack()
            
        
            button2 = tk.Button(sv_win, text="Save", command=im_save, width=20, height=2)
            button2.pack()
        
            tk.Label(sv_win, text="").pack()
            
            
          
        
            button5 = tk.Button(sv_win, text="Back", command=exit_win, width=20, height=2)
            button5.pack()
            
            
            sv_win.mainloop()
            
            app_metadata.add_save_current('png')
            
                 
        else:
            
            error_text = ('\nLoad and / or select the image!!!')
            error_win(error_text, parent_window = load_win)
        
        
        
       
            
    
    def prep_inimg():
        global app_metadata
        global inimg
        
        inimg = resize_to_screen_img(app_metadata.images_dict['img'][app_metadata.images_dict['img_name'].index(file_listbox.get(file_listbox.curselection()[0]))].copy(), factor = 4)

    

    def display_image_tmp():
        global inimg
        global app_metadata
        global zoom_factor
        global x
        global y
        
        t = True
 
        resized_image = update_zoomed_region(inimg, size.get(), x, y)
        
        cv2.imshow('Display',resized_image) 
        
        key = cv2.waitKey(50) & 0xFF
        if key == ord('z'):
            cv2.setMouseCallback('Display', zoom_in)
        elif cv2.getWindowProperty('Display',cv2.WND_PROP_VISIBLE) < 1: 

            load_win.destroy()
            cv2.destroyAllWindows()
            main_win() 
            
            t = False

        else:
            cv2.setMouseCallback('Display', lambda *args: None)  

        
        if t == True:
            load_win.after(1, display_image_tmp)
        
 
   
    def display_img_():
        
        
        if len(file_listbox.curselection()) > 0:
            prep_inimg()
            display_image_tmp()
        else:
            
            error_text = ('\nLoad and / or select the image!!!')
            error_win(error_text, parent_window = load_win)

       
         
    
    def rm_img(images_dict, img_name_to_remove):
        if img_name_to_remove in images_dict['img_name']:
            index_to_remove = images_dict['img_name'].index(img_name_to_remove)
            del images_dict['img'][index_to_remove]
            del images_dict['metadata'][index_to_remove]
            del images_dict['img_name'][index_to_remove]
    
   
        
        
    def img_list():
        global file_listbox
        file_listbox.delete(0, tk.END)  
        for filename in app_metadata.images_dict['img_name']:
            file_listbox.insert(tk.END, filename)
            
        
        
        
    def run_rm_img():
        if len(file_listbox.curselection()) > 0:
            global app_metadata
            selected_indices = file_listbox.curselection()
            imgs = [file_listbox.get(index) for index in selected_indices]
            for h in imgs:
                rm_img(app_metadata.images_dict, h)
                
            img_list()
        else:
            
            error_text = ('\nSelect image!!!')
            error_win(error_text, parent_window = load_win)
         
         

    def exit_win():
        global load_win
        cv2.destroyAllWindows()
        load_win.destroy() 

        
    def add_image():
        global load_win
        global input_path
        input_path = filedialog.askopenfilename()
        
        if len(input_path) > 0 and True in [x in str(input_path)  for x in ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif']]:
            
            init_path = os.getcwd()
            

            pth_img, img_nam = os.path.split(input_path)

            os.chdir(pth_img)      
            
            img = load_image(img_nam)
            
            img_name = os.path.basename(input_path)
            img_name = re.sub(r'\.[^.]*$', '', img_name) 
            
            n = 0
            while(True):
                n += 1
                tmp_img_name = img_name + '_loaded' + str(n)
                if tmp_img_name not in app_metadata.images_dict['img_name']:
                    break
    
            app_metadata.add_image(img, tmp_img_name, None)
            
            os.chdir(init_path)
            
            img_list()
                
               
        elif len(input_path) == 0:
            
            error_text = ('\nThe path was not provided!!!')
            error_win(error_text, parent_window = load_win)
            
        else:
            
            error_text = ('\nThe file extension is not available to load!!!')
            error_win(error_text, parent_window = load_win)
           
            
    def rotate_():
          
        if len(file_listbox.curselection()) > 0:
            global load_win   
            global app_metadata
            
            
            
            global metadata_to_rotate
            metadata_to_rotate = app_metadata.images_dict['metadata'][app_metadata.images_dict['img_name'].index(file_listbox.get(file_listbox.curselection()[0]))]
            
            global image_to_rotate 
            image_to_rotate = app_metadata.images_dict['img'][app_metadata.images_dict['img_name'].index(file_listbox.get(file_listbox.curselection()[0]))]
            
            global inimg
            inimg = resize_to_screen_img(image_to_rotate.copy(), factor = 4)
            
            global name_to_rotate 
            name_to_rotate = app_metadata.images_dict['img_name'][app_metadata.images_dict['img_name'].index(file_listbox.get(file_listbox.curselection()[0]))]
            
            
           
            def display_image_rot():
                global load_win
                global zoom_factor
                global zp
                global x
                global y
                global after_id
                global inimg
                

                
          
                if isinstance(app_metadata.resize_tmp['image'], np.ndarray):
                    resized_image = update_zoomed_region(app_metadata.resize_tmp['image'], size.get(), x, y)
                else:
                    resized_image = update_zoomed_region(inimg, size.get(), x, y)

                cv2.imshow('Rotate image',resized_image) 
                
                    
                
                key = cv2.waitKey(100) & 0xFF
                if key == ord('z'):
                    cv2.setMouseCallback('Rotate image', zoom_in)
                else:
                    cv2.setMouseCallback('Rotate image', lambda *args: None)  


                rotate_win.after(100, display_image_rot)
            
            
            def save_():
                global im_to_save
                
                if isinstance(app_metadata.resize_tmp['image'], np.ndarray):
                    
                    app_metadata.add_image(im_to_save, app_metadata.resize_tmp['name'], app_metadata.resize_tmp['metadata'])
                    
                    app_metadata.add_resize(None, None, None)
                    
                    global rotate_win
                    
                    cv2.destroyAllWindows()
                    rotate_win.destroy()
                    
                    img_list()
                    
                    
                
                else:
                    
                    error_text = ('\nNothing selected to save!\n'
                                  'Firstly resize some image!')
                    
                    error_win(error_text, parent_window = rotate_win)
                    

            
            
            def exit_win():
                
                cv2.destroyAllWindows()
                rotate_win.destroy()
                
             
           
            def rotate_run():
                global app_metadata
                global metadata_to_rotate
                global image_to_rotate 
                global name_to_rotate 
                global im_to_save
                global inimg
                
                

                if rotate_box.get() == "0°":
                    r = 0
                elif rotate_box.get() == "90°":
                    r = -1
                elif rotate_box.get() == "180°":
                    r = 2
                elif rotate_box.get() == "180°":
                    r = 2
                elif rotate_box.get() == "270°":
                    r = 1
                    

                tmp_img = rotate_function(inimg, r)
                im_to_save = rotate_function(image_to_rotate, r)

              

                if mirror_box.get() == "horizontal":
                    tmp_img = mirror_function(tmp_img, 'h')
                    im_to_save = mirror_function(im_to_save, 'h')

                elif mirror_box.get() == "vertical":
                    tmp_img = mirror_function(tmp_img, 'v')
                    im_to_save = mirror_function(im_to_save, 'v')

                elif mirror_box.get() == "horizontal/vertical":
                    tmp_img = mirror_function(tmp_img, 'hv')
                    im_to_save = mirror_function(im_to_save, 'hv')


                

                    
                n = 0
                while(True):
                    n += 1
                    tmp_img_name = name_to_rotate + '_rotated_' + str(n)
                    if tmp_img_name not in app_metadata.images_dict['img_name']:
                        break
               
               
                app_metadata.add_resize(tmp_img, metadata_to_rotate, tmp_img_name)
                    
      
                        
                       
    
            global rotate_win
            
            rotate_win = tk.Toplevel(load_win)
            
            # rotate_win = tk.Tk()

        
            rotate_win.geometry("500x500")
            rotate_win.title("Rotate image")
        
            rotate_win.iconbitmap(os.path.join(_icon_source,'jbi_icon.ico'))
            
            
            tk.Label(rotate_win, text="", anchor="w").pack()
            
            
            text = (
                '    You can rotate or mirror your image.\n'
                '    Please note that annotating the single images in their raw\n'
                '    form will not be possible after rotating or mirroring.\n\n\n'
                '    !The option Annotate raw for this image should be not use!\n'
            )
            
        
        
            tk.Label(rotate_win, text=text, anchor="w", justify="center").pack()
            
            tk.Label(rotate_win, text="").pack()
            
            label1 = tk.Label(rotate_win, text="Rotate °:", anchor="w")
            label1.pack()
            
           
            rotate_type = ["0°", "90°", "180°", "270°"]

            rotate_box = ttk.Combobox(rotate_win, values=rotate_type)
            
            rotate_box.current(0)
            
            rotate_box.pack()
            
            
            tk.Label(rotate_win, text="").pack()
            
            label1 = tk.Label(rotate_win, text="Mirror type:", anchor="w")
            label1.pack()
            
           
            mirror_type = ["----------------------", "horizontal", "vertical", "horizontal/vertical"]

            mirror_box = ttk.Combobox(rotate_win, values=mirror_type)
            
            mirror_box.current(0)
            
            mirror_box.pack()
            
            
            

            tk.Label(rotate_win, text="").pack()

            
            button4 = tk.Button(rotate_win, text="Rotate", command=rotate_run, width=20, height=2)
            button4.pack()
            
            
            tk.Label(rotate_win, text="").pack()

            
            button5 = tk.Button(rotate_win, text="Save", command=save_, width=20, height=2)
            button5.pack()
            
            
            tk.Label(rotate_win, text="").pack()

            
        
            button6 = tk.Button(rotate_win, text="Back", command=exit_win, width=20, height=2)
            button6.pack()
            
    
            display_image_rot()
            
            rotate_win.mainloop()
            
            cv2.destroyAllWindows()

            
                 
        else:
            
            error_text = ('\nLoad and / or select the image!!!')
            error_win(error_text, parent_window = load_win)  
            
            
            
    def resize_():
          
        if len(file_listbox.curselection()) > 0:
            global load_win   
            global app_metadata
            
            
            
            global metadata_to_resize
            metadata_to_resize = app_metadata.images_dict['metadata'][app_metadata.images_dict['img_name'].index(file_listbox.get(file_listbox.curselection()[0]))]
            
            global image_to_resize 
            image_to_resize = app_metadata.images_dict['img'][app_metadata.images_dict['img_name'].index(file_listbox.get(file_listbox.curselection()[0]))]
            
            
            global name_to_resize 
            name_to_resize = app_metadata.images_dict['img_name'][app_metadata.images_dict['img_name'].index(file_listbox.get(file_listbox.curselection()[0]))]
            
            
            
            h = image_to_resize.shape[0]
            w = image_to_resize.shape[1]
            
            
            
            def save_():
                
                if isinstance(app_metadata.resize_tmp['image'], np.ndarray):
                    
                    app_metadata.add_image(app_metadata.resize_tmp['image'], app_metadata.resize_tmp['name'], app_metadata.resize_tmp['metadata'])
                    
                    app_metadata.add_resize(None, None, None)
                    
                    global resize_win
                    
                    resize_win.destroy()
                    
                    img_list()
                    
                    
                
                else:
                    
                    error_text = ('\nNothing selected to save!\n'
                                  'Firstly resize some image!')
                    
                    error_win(error_text, parent_window = resize_win)
                    

            
            
            def exit_win():
                resize_win.destroy()
                
             
           
            def resize_run():
                global app_metadata
                global metadata_to_resize
                global image_to_resize 
                global name_to_resize 
                
            
            
                h1 = image_to_resize.shape[0]
                w1 = image_to_resize.shape[1]
                
                
                
                resize_fac = resize_txt.get("1.0", tk.END)
                resize_fac = str(resize_fac.replace(' ', ''))
                resize_fac = str(resize_fac.replace('\n', ''))

                
                dec_resize = True
                for char in resize_fac:
                    if not (char.isdigit()):
                        dec_resize = False
                        break
                    
                
                
                height_fac = height_txt.get("1.0", tk.END)
                height_fac = str(height_fac.replace(' ', ''))
                height_fac = str(height_fac.replace('\n', ''))

                
                dec_height = True
                for char in height_fac:
                    if not (char.isdigit()):
                        dec_height = False
                        break
                    
                    
                width_fac = width_txt.get("1.0", tk.END)
                width_fac = str(width_fac.replace(' ', ''))
                width_fac = str(width_fac.replace('\n', ''))

                
                dec_width = True
                for char in height_fac:
                    if not (char.isdigit()):
                        dec_width = False
                        break
                    


                if dec_height == True and h1 != int(height_fac):
 

                    res_im, res_met = resize_projection(image_to_resize, metadata = metadata_to_resize, height = int(height_fac), width = None, resize_factor = None)
                    
                    
                    hres = res_im.shape[0]
                    wres = res_im.shape[1]
                
                    
                    n = 0
                    while(True):
                        n += 1
                        tmp_img_name = name_to_resize + '_resized_' + str(n)
                        if tmp_img_name not in app_metadata.images_dict['img_name']:
                            break
                    
                    
                    app_metadata.add_resize(res_im, res_met, tmp_img_name)
                    
                    error_text = (f'\nResized sucesfully!\n'
                                  f'Current size is height {hres} x width {wres}')
                    
                    error_win(error_text, parent_window = resize_win,  color= 'green', win_name= 'Info')
                    
                    
                    
                elif dec_width == True and w1 != int(width_fac):
 
                    res_im, res_met = resize_projection(image_to_resize, metadata = metadata_to_resize, height = None, width = int(width_fac), resize_factor = None)
                    
                    
                    hres = res_im.shape[0]
                    wres = res_im.shape[1]
                
                    
                    n = 0
                    while(True):
                        n += 1
                        tmp_img_name = name_to_resize + '_resized_' + str(n)
                        if tmp_img_name not in app_metadata.images_dict['img_name']:
                            break
                    
                    
                    app_metadata.add_resize(res_im, res_met, tmp_img_name)
                    
                    error_text = (f'\nResized sucesfully!\n'
                                  f'Current size is height {hres} x width {wres}')
                    
                    error_win(error_text, parent_window = resize_win,  color= 'green', win_name= 'Info')
                                       
                    
                elif dec_resize == True and int(resize_fac) != int(1):
 
                    res_im, res_met = resize_projection(image_to_resize, metadata = metadata_to_resize, height = None, width = None, resize_factor = int(resize_fac))
                    
                    
                    hres = res_im.shape[0]
                    wres = res_im.shape[1]
                
                                        
                    n = 0
                    while(True):
                        n += 1
                        tmp_img_name = name_to_resize + '_resized_' + str(n)
                        if tmp_img_name not in app_metadata.images_dict['img_name']:
                            break
                    
                    
                    app_metadata.add_resize(res_im, res_met, tmp_img_name)
                    
                    error_text = (f'\nResized sucesfully!\n'
                                  f'Current size is height {hres} x width {wres}')
                    
                    error_win(error_text, parent_window = resize_win,  color= 'green', win_name= 'Info')
                    
                    
                else:
                    res_im, res_met = None, None
                    
                    error_text = ('\nThe image is unable to resize with these settings!')
                    error_win(error_text, parent_window = resize_win)
                        
                       
    
            global resize_win
            
            resize_win = tk.Toplevel(load_win)
        
            resize_win.geometry("500x560")
            resize_win.title("Resize image")
        
            resize_win.iconbitmap(os.path.join(_icon_source,'jbi_icon.ico'))
            
            
            tk.Label(resize_win, text="", anchor="w").pack()
            
            
            text = (
                '    You can change only one parameter in each resizing operation.\n'
                '    This restriction is designed to preserve the biological proportions\n'
                '    When you set more than one parameter, only the first parameter \n'
                '    in the queue will be changed in a single resizing operation.\n'
                '    The queue is set up: first height, second width, and last resize factor.\n'
              
            )
            
        
        
            tk.Label(resize_win, text=text, anchor="w", justify="center").pack()
            
            tk.Label(resize_win, text="").pack()
            
            tk.Label(resize_win, text="Height:").pack()
            
            global height_txt
            height_txt = tk.Text(resize_win, height=1, width=10)
            height_txt.insert(tk.END, str(h))
            height_txt.pack()
            
            
                           
            tk.Label(resize_win, text="").pack()
            
            tk.Label(resize_win, text="Width:").pack()
            
            global width_txt
            width_txt = tk.Text(resize_win, height=1, width=10)
            width_txt.insert(tk.END, str(w))
            width_txt.pack()
            
            
            tk.Label(resize_win, text="").pack()
            
            tk.Label(resize_win, text="Resize factor:").pack()
            
            global resize_txt
            resize_txt = tk.Text(resize_win, height=1, width=10)
            resize_txt.insert(tk.END, '1')
            resize_txt.pack()

            tk.Label(resize_win, text="").pack()

            
            button4 = tk.Button(resize_win, text="Resize", command=resize_run, width=20, height=2)
            button4.pack()
            
            
            tk.Label(resize_win, text="").pack()

            
            button5 = tk.Button(resize_win, text="Save", command=save_, width=20, height=2)
            button5.pack()
            
            
            tk.Label(resize_win, text="").pack()

            
        
            button6 = tk.Button(resize_win, text="Back", command=exit_win, width=20, height=2)
            button6.pack()
            
    
    
            resize_win.mainloop()
            
            cv2.destroyAllWindows()

            
                 
        else:
            
            error_text = ('\nLoad and / or select the image!!!')
            error_win(error_text, parent_window = load_win)
        
        
        

    

    def main_win():
        global file_listbox
        global load_win
        global size


        load_win = tk.Tk()
        
    
        load_win.geometry("650x745")
        load_win.title("Images manager")
    
        load_win.iconbitmap(os.path.join(_icon_source,'jbi_icon.ico'))
        
        
        tk.Label(load_win, text="", anchor="w").pack()
        
        tk.Label(load_win, text="Window size", anchor="w").pack()
        
        
        # Create a slider widget
        size = tk.DoubleVar()
        slider1 = tk.Scale(load_win, from_=1, to=50, orient=tk.HORIZONTAL, length=400, variable=size)
        slider1.set(24)
        slider1.pack()
        
    
        tk.Label(load_win, text="").pack()
        
        tk.Label(load_win, text="Images manager").pack()
        
        
        file_listbox = tk.Listbox(load_win, selectmode=tk.SINGLE, width=90)
        file_listbox.pack(pady=10)


        
    
        button1 = tk.Button(load_win, text="Add", command=add_image, width=20, height=2)
        button1.pack()
        
        tk.Label(load_win, text="").pack()
        
    
        button2 = tk.Button(load_win, text="Remove", command=run_rm_img, width=20, height=2)
        button2.pack()
    
        tk.Label(load_win, text="").pack()
        
        
        button3 = tk.Button(load_win, text="Display", command=display_img_, width=20, height=2)
        button3.pack()
    
        tk.Label(load_win, text="").pack()
        
        
        
        button4 = tk.Button(load_win, text="Resize", command=resize_, width=20, height=2)
        button4.pack()
    
        tk.Label(load_win, text="").pack()
        
        
        
        button4_1 = tk.Button(load_win, text="Rotate", command=rotate_, width=20, height=2)
        button4_1.pack()
    
        tk.Label(load_win, text="").pack()
        

        
        button5 = tk.Button(load_win, text="Save", command=save_image, width=20, height=2)
        button5.pack()
    
        tk.Label(load_win, text="").pack()
        
    
        button6 = tk.Button(load_win, text="Back", command=exit_win, width=20, height=2)
        button6.pack()
        
        
    
        img_list()

        load_win.mainloop()
        
        cv2.destroyAllWindows()
        
        
        
        
    main_win()
    
    
    
# merge images window

def img_merge_win():
    
    global app_run
    global app_metadata
    global merge_win
    
    
    global zoom_factor
    global x
    global y
    x = 1 
    y = 1
    zoom_factor = 1.0
    
    def merge_():
        
        
        if len(file_listbox.curselection()) > 0:
            global app_metadata
            selected_indices = file_listbox.curselection()
            image_list = [file_listbox.get(index) for index in selected_indices]
            image_names = image_list
            metadata = app_metadata.images_dict['metadata'][app_metadata.images_dict['img_name'].index(image_names[0])]
            image_list = [app_metadata.images_dict['img'][app_metadata.images_dict['img_name'].index(i)] for i in image_list]
    
            if len(image_list) > 1:
                shapes_list = [s.shape for s in image_list]
                if all(elem == shapes_list[0] for elem in shapes_list):
                    
                    merge_win.destroy()
                    merge_image = merge_images_app(image_list)
                    
                    if isinstance(merge_image, np.ndarray):
                        
                        img_name = 'merged_image_'
                        
                        n = 0
                        while(True):
                            n += 1
                            tmp_img_name = img_name + str(n)
                            if tmp_img_name not in app_metadata.images_dict['img_name']:
                                break
                
                        app_metadata.add_image(merge_image, tmp_img_name, metadata)
                        
                    main_win()
                    
                else:
                    
                    error_text = ('\nSelected images could not be merged!!!\n'
                                  'Images have different shapes!!!')
                    error_win(error_text, parent_window = merge_win)


            else:
                
                error_text = ('\nThe number of images should be more than one!!!')
                error_win(error_text, parent_window = merge_win)

                
        else:
            
            error_text = ('\nSelect images!!!')
            error_win(error_text, parent_window = merge_win)
         

        
    
    
    def save_image():
        
        
        if len(file_listbox.curselection()) > 0:
            
            app_metadata.add_save_current(file_listbox.get(file_listbox.curselection()[0]))
        
            
            global merge_win
            global lab_name
            
            def exit_win():
                sv_win.destroy()
                
             
            def save_browse():
                
                save_path = filedialog.askdirectory()
                if save_path:
                    sv_box.delete(0, 'end')  
                    sv_box.insert('end', save_path)
                    app_metadata.add_tmp_path(save_path)
                
                
            def im_save():
                global lab_name
                global app_metadata
                

                
                file = lab_name.get("1.0", tk.END)
                file = re.sub(r'\n', '', file)
                file = re.sub(r'\s', '_', file)
                
                if len(file) > 0 and app_metadata.tmp_path != None:
                
                    init_path = os.getcwd()
                    
                    os.chdir(app_metadata.tmp_path)
                    
                    
                    n = 0
                    while(True):
                        n += 1
                        tmp_file = file + '_' + str(n) + '.' + str(type_box.get())
                        if os.path.exists(tmp_file) == False:
                            break
                    
    
                    cv2.imwrite(tmp_file, app_metadata.images_dict['img'][app_metadata.images_dict['img_name'].index(app_metadata.save_current)])
                    
                    
                    

                    
                    os.chdir(init_path) 
                    
                    error_text = ('\nResults saved successfully!!!')
                    error_win(error_text, parent_window = None, color= 'green', win_name='Information')
                        

                    
                elif len(file) == 0:
                    error_text = ('\nProvide the file name!!!')
                    error_win(error_text, parent_window = sv_win)
                
                else:
                    error_text = ('\nProvide the path to the directory!!!')
                    error_win(error_text, parent_window = sv_win)
    
    

            
            sv_win = tk.Toplevel(merge_win)
        
            sv_win.geometry("500x400")
            sv_win.title("Save image")
        
            sv_win.iconbitmap(os.path.join(_icon_source,'jbi_icon.ico'))
            
            
            sv_win.transient(merge_win)

            sv_win.grab_set()
            
            
            tk.Label(sv_win, text="", anchor="w").pack()
            
            tk.Label(sv_win, text="Path to the save directory:").pack()
        
        
            sv_box = tk.Listbox(sv_win, width=70, height=1)
            sv_box.pack(pady=1)
            
            if app_metadata.tmp_path != None:
                sv_box.delete(0, 'end')  
                sv_box.insert('end', app_metadata.tmp_path)
            
    
            
            button1 = tk.Button(sv_win, text="Browse", command=save_browse, width=20, height=2)
            button1.pack()
            
            tk.Label(sv_win, text="").pack()
            
            
            label = tk.Label(sv_win, text="File name:")
            label.pack()
            
            tk.Label(sv_win, text="").pack()
            
            lab_name = Text(sv_win, height=1, width=50)
            lab_name.pack()
            
            
            tk.Label(sv_win, text="").pack()
    
            img_type = ['png', 'tiff', 'tif']
    
            type_box = ttk.Combobox(sv_win, values=img_type)
            
            type_box.current(0)
            
            type_box.pack()
            
            tk.Label(sv_win, text="").pack()
            
        
            button2 = tk.Button(sv_win, text="Save", command=im_save, width=20, height=2)
            button2.pack()
        
            tk.Label(sv_win, text="").pack()
            
            
          
        
            button5 = tk.Button(sv_win, text="Back", command=exit_win, width=20, height=2)
            button5.pack()
            
            
            sv_win.mainloop()
            
            app_metadata.add_save_current('png')
            
                 
        else:
            
            error_text = ('\nSelect the image!!!')
            error_win(error_text, parent_window = merge_win)

        

    
    
    def prep_inimg():
        global app_metadata
        global inimg
        
        inimg = resize_to_screen_img(app_metadata.images_dict['img'][app_metadata.images_dict['img_name'].index(file_listbox.get(file_listbox.curselection()[0]))].copy(), factor = 4)



    def display_image_tmp():
        global inimg
        global app_metadata
        global zoom_factor
        global x
        global y
        
        t = True
 
        resized_image = update_zoomed_region(inimg, size.get(), x, y)
        
        cv2.imshow('Display',resized_image) 
        
        key = cv2.waitKey(50) & 0xFF
        if key == ord('z'):
            cv2.setMouseCallback('Display', zoom_in)
        elif cv2.getWindowProperty('Display',cv2.WND_PROP_VISIBLE) < 1: 

            merge_win.destroy()
            cv2.destroyAllWindows()
            main_win() 
            
            t = False

        else:
            cv2.setMouseCallback('Display', lambda *args: None)  

        
        if t == True:
            merge_win.after(1, display_image_tmp)
            
        
   
 
   
    def display_img_():
        if len(file_listbox.curselection()) > 0:
            prep_inimg()
            display_image_tmp()
        else:
            
            error_text = ('\nSelect the image!!!')
            error_win(error_text, parent_window = merge_win)

       
      
        
    def img_list():
        global file_listbox
        file_listbox.delete(0, tk.END)  
        for filename in app_metadata.images_dict['img_name']:
            file_listbox.insert(tk.END, filename)
            
        
 

    def exit_win():
        global merge_win
        cv2.destroyAllWindows()
        merge_win.destroy() 
        
        

    def main_win():
        global file_listbox
        global merge_win
        global size


        merge_win = tk.Tk()
        
    
        merge_win.geometry("650x560")
        merge_win.title("Merge images")
    
        merge_win.iconbitmap(os.path.join(_icon_source,'jbi_icon.ico'))
        
        
        tk.Label(merge_win, text="", anchor="w").pack()
        
        tk.Label(merge_win, text="Window size", anchor="w").pack()
        
        # Create a slider widget
        
        size = tk.DoubleVar()
        slider1 = tk.Scale(merge_win, from_=1, to=50, orient=tk.HORIZONTAL, length=400, variable=size)
        slider1.set(24)
        slider1.pack()
        
    
        tk.Label(merge_win, text="").pack()
        
        tk.Label(merge_win, text="Images manager").pack()
        
        
        file_listbox = tk.Listbox(merge_win, selectmode=tk.MULTIPLE, width=90)
        file_listbox.pack(pady=10)


        
    
        button1 = tk.Button(merge_win, text="Display", command=display_img_, width=20, height=2)
        button1.pack()
        
        tk.Label(merge_win, text="").pack()
        
    
        button2 = tk.Button(merge_win, text="Merge", command=merge_, width=20, height=2)
        button2.pack()
    
        tk.Label(merge_win, text="").pack()
        
        
        button3 = tk.Button(merge_win, text="Save", command=save_image, width=20, height=2)
        button3.pack()
    
        tk.Label(merge_win, text="").pack()
        
        
        button4 = tk.Button(merge_win, text="Back", command=exit_win, width=20, height=2)
        button4.pack()
    

    
        img_list()

        merge_win.mainloop()
        
        cv2.destroyAllWindows()
         
        
    main_win()



#scalebar window


def img_scale_win():
    
    global app_run
    global app_metadata
    global sc_win
    
    
    global zoom_factor
    global x
    global y
    
    x = 1 
    y = 1
    zoom_factor = 1.0
    
    
    
    def scalebar_():
        
        
        if len(file_listbox.curselection()) > 0:
            global app_metadata
            
            metadata = app_metadata.images_dict['metadata'][app_metadata.images_dict['img_name'].index(file_listbox.get(file_listbox.curselection()[0]))]
            image = app_metadata.images_dict['img'][app_metadata.images_dict['img_name'].index(file_listbox.get(file_listbox.curselection()[0]))]
            name = app_metadata.images_dict['img_name'][app_metadata.images_dict['img_name'].index(file_listbox.get(file_listbox.curselection()[0]))]
            
 
    
            sc_win.destroy()
            
        
            
            if metadata == None:
                px_to_um = None
            else:
                px_to_um = metadata['X_resolution[um/px]']
            
            result_image = add_scalebar(image, px_to_um, parent_window = None)


            if isinstance(result_image, np.ndarray):
                
            
                        
                img_name = name + '_scale-bar_'
                
                n = 0
                while(True):
                    n += 1
                    tmp_img_name = img_name + str(n)
                    if tmp_img_name not in app_metadata.images_dict['img_name']:
                        break
                    
                            
                app_metadata.add_image(result_image, tmp_img_name, None)
                            
            main_win()
                    
             
                
        else:
            
            error_text = ('\nSelect the image!!!')
            error_win(error_text, parent_window = sc_win)
         

        
    
    
    def save_image():
        
        
        if len(file_listbox.curselection()) > 0:
            
            app_metadata.add_save_current(file_listbox.get(file_listbox.curselection()[0]))
        
            
            global sc_win
            global lab_name
            
            def exit_win():
                sv_win.destroy()
                
             
            def save_browse():
                
                save_path = filedialog.askdirectory()
                if save_path:
                    sv_box.delete(0, 'end')  
                    sv_box.insert('end', save_path)
                    app_metadata.add_tmp_path(save_path)
                
                
            def im_save():
                global lab_name
                global app_metadata
                

                
                file = lab_name.get("1.0", tk.END)
                file = re.sub(r'\n', '', file)
                file = re.sub(r'\s', '_', file)
                
                if len(file) > 0 and app_metadata.tmp_path != None:
                
                    init_path = os.getcwd()
                    
                    os.chdir(app_metadata.tmp_path)
                    
                    
                    n = 0
                    while(True):
                        n += 1
                        tmp_file = file + '_' + str(n) + '.' + str(type_box.get())
                        if os.path.exists(tmp_file) == False:
                            break
                    
    
                    cv2.imwrite(tmp_file, app_metadata.images_dict['img'][app_metadata.images_dict['img_name'].index(app_metadata.save_current)])
                    
                    
                    

                    
                    os.chdir(init_path) 
                    
                    error_text = ('\nResults saved successfully!!!')
                    error_win(error_text, parent_window = None, color= 'green', win_name='Information')
                        

                    
                elif len(file) == 0:
                    error_text = ('\nProvide the file name!!!')
                    error_win(error_text, parent_window = sv_win)
                
                else:
                    error_text = ('\nProvide the path to the directory!!!')
                    error_win(error_text, parent_window = sv_win)
    
    

            
            sv_win = tk.Toplevel(sc_win)
        
            sv_win.geometry("500x400")
            sv_win.title("Save image")
        
            sv_win.iconbitmap(os.path.join(_icon_source,'jbi_icon.ico'))
            
            
            sv_win.transient(sc_win)

            sv_win.grab_set()
            
            
            
            tk.Label(sv_win, text="", anchor="w").pack()
            
            tk.Label(sv_win, text="Path to the save directory:").pack()
        
        
            sv_box = tk.Listbox(sv_win, width=70, height=1)
            sv_box.pack(pady=1)
            
            if app_metadata.tmp_path != None:
                sv_box.delete(0, 'end')  
                sv_box.insert('end', app_metadata.tmp_path)
            
    
            
            button1 = tk.Button(sv_win, text="Browse", command=save_browse, width=20, height=2)
            button1.pack()
            
            tk.Label(sv_win, text="").pack()
            
            
            label = tk.Label(sv_win, text="File name:")
            label.pack()
            
            tk.Label(sv_win, text="").pack()
            
            lab_name = Text(sv_win, height=1, width=50)
            lab_name.pack()
            
            
            tk.Label(sv_win, text="").pack()
    
            img_type = ['png', 'tiff', 'tif']
    
            type_box = ttk.Combobox(sv_win, values=img_type)
            
            type_box.current(0)
            
            type_box.pack()
            
            tk.Label(sv_win, text="").pack()
            
        
            button2 = tk.Button(sv_win, text="Save", command=im_save, width=20, height=2)
            button2.pack()
        
            tk.Label(sv_win, text="").pack()
            
            
          
        
            button5 = tk.Button(sv_win, text="Back", command=exit_win, width=20, height=2)
            button5.pack()
            
            
            sv_win.mainloop()
            
            app_metadata.add_save_current('png')
            
                 
        else:
            
            error_text = ('\nSelect the image!!!')
            error_win(error_text, parent_window = sc_win)
        

    
    
    def prep_inimg():
        global app_metadata
        global inimg
        
        inimg = resize_to_screen_img(app_metadata.images_dict['img'][app_metadata.images_dict['img_name'].index(file_listbox.get(file_listbox.curselection()[0]))].copy(), factor = 4)

        
    def display_image_tmp():
        global inimg
        global app_metadata
        global zoom_factor
        global x
        global y
        
        t = True
     
        resized_image = update_zoomed_region(inimg, size.get(), x, y)
        
        cv2.imshow('Display',resized_image) 
        
        key = cv2.waitKey(50) & 0xFF
        if key == ord('z'):
            cv2.setMouseCallback('Display', zoom_in)
        elif cv2.getWindowProperty('Display',cv2.WND_PROP_VISIBLE) < 1: 
    
            sc_win.destroy()
            cv2.destroyAllWindows()
            main_win() 
            
            t = False
    
        else:
            cv2.setMouseCallback('Display', lambda *args: None)  
    
        
        if t == True:
            sc_win.after(1, display_image_tmp)
            
            
 
   
    def display_img_():
        if len(file_listbox.curselection()) > 0:
            prep_inimg()
            display_image_tmp()
        else:
            
            error_text = ('\nSelect the image!!!')
            error_win(error_text, parent_window = sc_win)

       
      
        
    def img_list():
        global file_listbox
        file_listbox.delete(0, tk.END)  
        for filename in app_metadata.images_dict['img_name']:
            file_listbox.insert(tk.END, filename)
            



    def exit_win():
        global sc_win
        cv2.destroyAllWindows()
        sc_win.destroy() 
        
        

    def main_win():
        global file_listbox
        global sc_win
        global size


        sc_win = tk.Tk()
        
    
        sc_win.geometry("650x560")
        sc_win.title("Add scale-bar")
    
        sc_win.iconbitmap(os.path.join(_icon_source,'jbi_icon.ico'))
        
        
        tk.Label(sc_win, text="", anchor="w").pack()
        
        tk.Label(sc_win, text="Window size", anchor="w").pack()
        
        # Create a slider widget
        
        size = tk.DoubleVar()
        slider1 = tk.Scale(sc_win, from_=1, to=50, orient=tk.HORIZONTAL, length=400, variable=size)
        slider1.set(24)
        slider1.pack()
        
    
        tk.Label(sc_win, text="").pack()
        
        tk.Label(sc_win, text="Images list").pack()
        
        
        file_listbox = tk.Listbox(sc_win, selectmode=tk.SINGLE, width=90)
        file_listbox.pack(pady=10)


        
    
        button1 = tk.Button(sc_win, text="Display", command=display_img_, width=20, height=2)
        button1.pack()
        
        tk.Label(sc_win, text="").pack()
        
    
        button2 = tk.Button(sc_win, text="Add\nscale", command=scalebar_, width=20, height=2)
        button2.pack()
    
        tk.Label(sc_win, text="").pack()
        
        
        button3 = tk.Button(sc_win, text="Save", command=save_image, width=20, height=2)
        button3.pack()
    
        tk.Label(sc_win, text="").pack()
        
        
        button4 = tk.Button(sc_win, text="Back", command=exit_win, width=20, height=2)
        button4.pack()
    

        img_list()

        sc_win.mainloop()
        
        cv2.destroyAllWindows()
         
        
    main_win()




# single annotation

def img_annotation_image():
    
    global app_run
    global app_metadata
    global an_win
    
    
    global zoom_factor
    global x
    global y
    x = 1 
    y = 1
    zoom_factor = 1.0
    
    
    
    def annotate_():
        
        
        if len(file_listbox.curselection()) > 0:
            global app_metadata
         
            metadata = app_metadata.images_dict['metadata'][app_metadata.images_dict['img_name'].index(file_listbox.get(file_listbox.curselection()[0]))]
            image = app_metadata.images_dict['img'][app_metadata.images_dict['img_name'].index(file_listbox.get(file_listbox.curselection()[0]))]
            name = app_metadata.images_dict['img_name'][app_metadata.images_dict['img_name'].index(file_listbox.get(file_listbox.curselection()[0]))]
            
 
    
            an_win.destroy()
            
        
            result_image, annotation, mask = draw_annotation(image)
            

            if isinstance(result_image, np.ndarray):
                
            
                        
                img_name = name + '_annotated_image_'
                
                n = 0
                while(True):
                    n += 1
                    tmp_img_name = img_name + str(n)
                    if tmp_img_name not in app_metadata.images_dict['img_name']:
                        break
                    
                            
                app_metadata.add_image(result_image, tmp_img_name, metadata)
                app_metadata.add_image(annotation, tmp_img_name + '_lines', metadata)
                app_metadata.add_image(mask, tmp_img_name + '_mask', metadata)


                            
            main_win()
                    
             
                
        else:
            
            error_text = ('\nSelect the image!!!')
            error_win(error_text, parent_window = an_win)
         

        
    
    
    def save_image():
        
        
        if len(file_listbox.curselection()) > 0:
            
            app_metadata.add_save_current(file_listbox.get(file_listbox.curselection()[0]))
        
            
            global an_win
            global lab_name
            
            def exit_win():
                sv_win.destroy()
                
             
            def save_browse():
                
                save_path = filedialog.askdirectory()
                if save_path:
                    sv_box.delete(0, 'end')  
                    sv_box.insert('end', save_path)
                    app_metadata.add_tmp_path(save_path)
                
                
            def im_save():
                global lab_name
                global app_metadata
                

                
                file = lab_name.get("1.0", tk.END)
                file = re.sub(r'\n', '', file)
                file = re.sub(r'\s', '_', file)
                
                if len(file) > 0 and app_metadata.tmp_path != None:
                
                    init_path = os.getcwd()
                    
                    os.chdir(app_metadata.tmp_path)
                    
                    
                    n = 0
                    while(True):
                        n += 1
                        tmp_file = file + '_' + str(n) + '.' + str(type_box.get())
                        if os.path.exists(tmp_file) == False:
                            break
                    
    
                    cv2.imwrite(tmp_file, app_metadata.images_dict['img'][app_metadata.images_dict['img_name'].index(app_metadata.save_current)])
                    
                    
                    

                    
                    os.chdir(init_path) 
                    
                    error_text = ('\nResults saved successfully!!!')
                    error_win(error_text, parent_window = None, color= 'green', win_name='Information')
                        

                    
                elif len(file) == 0:
                    error_text = ('\nProvide the file name!!!')
                    error_win(error_text, parent_window = sv_win)
                
                else:
                    error_text = ('\nProvide the path to the directory!!!')
                    error_win(error_text, parent_window = sv_win)
    
    

            
            sv_win = tk.Toplevel(an_win)
        
            sv_win.geometry("500x400")
            sv_win.title("Save image")
        
            sv_win.iconbitmap(os.path.join(_icon_source,'jbi_icon.ico'))
            
            
            sv_win.transient(an_win)

            sv_win.grab_set()
            
            
            
            tk.Label(sv_win, text="", anchor="w").pack()
            
            tk.Label(sv_win, text="Path to the save directory:").pack()
        
        
            sv_box = tk.Listbox(sv_win, width=70, height=1)
            sv_box.pack(pady=1)
            
            if app_metadata.tmp_path != None:
                sv_box.delete(0, 'end')  
                sv_box.insert('end', app_metadata.tmp_path)
            
    
            
            button1 = tk.Button(sv_win, text="Browse", command=save_browse, width=20, height=2)
            button1.pack()
            
            tk.Label(sv_win, text="").pack()
            
            
            label = tk.Label(sv_win, text="File name:")
            label.pack()
            
            tk.Label(sv_win, text="").pack()
            
            lab_name = Text(sv_win, height=1, width=50)
            lab_name.pack()
            
            
            tk.Label(sv_win, text="").pack()
    
            img_type = ['png', 'tiff', 'tif']
    
            type_box = ttk.Combobox(sv_win, values=img_type)
            
            type_box.current(0)
            
            type_box.pack()
            
            tk.Label(sv_win, text="").pack()
            
        
            button2 = tk.Button(sv_win, text="Save", command=im_save, width=20, height=2)
            button2.pack()
        
            tk.Label(sv_win, text="").pack()
            
            
          
        
            button5 = tk.Button(sv_win, text="Back", command=exit_win, width=20, height=2)
            button5.pack()
            
            
            sv_win.mainloop()
            
            app_metadata.add_save_current('png')
            
                 
        else:
            
            error_text = ('\nSelect the image!!!')
            error_win(error_text, parent_window = an_win)
        
        
    
    def prep_inimg():
        global app_metadata
        global inimg
        
        inimg = resize_to_screen_img(app_metadata.images_dict['img'][app_metadata.images_dict['img_name'].index(file_listbox.get(file_listbox.curselection()[0]))].copy(), factor = 4)


    
    def display_image_tmp():
        global inimg
        global app_metadata
        global zoom_factor
        global x
        global y
        
        t = True
     
        resized_image = update_zoomed_region(inimg, size.get(), x, y)
        
        cv2.imshow('Display',resized_image) 
        
        key = cv2.waitKey(50) & 0xFF
        if key == ord('z'):
            cv2.setMouseCallback('Display', zoom_in)
        elif cv2.getWindowProperty('Display',cv2.WND_PROP_VISIBLE) < 1: 
    
            an_win.destroy()
            cv2.destroyAllWindows()
            main_win() 
            
            t = False
    
        else:
            cv2.setMouseCallback('Display', lambda *args: None)  
    
        
        if t == True:
            an_win.after(1, display_image_tmp)
            
        

   
    def display_img_():
        if len(file_listbox.curselection()) > 0:
            prep_inimg()
            display_image_tmp()
        else:
            
            error_text = ('\nSelect the image!!!')
            error_win(error_text, parent_window = an_win)

       
      
        
    def img_list():
        global file_listbox
        file_listbox.delete(0, tk.END)  
        for filename in app_metadata.images_dict['img_name']:
            file_listbox.insert(tk.END, filename)
            
        
 

    def exit_win():
        global an_win
        cv2.destroyAllWindows()
        an_win.destroy() 
        
        

    def main_win():
        global file_listbox
        global an_win
        global size


        an_win = tk.Tk()
            
        an_win.geometry("650x560")
        an_win.title("Image annotation")
    
        an_win.iconbitmap(os.path.join(_icon_source,'jbi_icon.ico'))
        
        
        tk.Label(an_win, text="", anchor="w").pack()
        
        tk.Label(an_win, text="Window size", anchor="w").pack()
        
        # Create a slider widget
        
        size = tk.DoubleVar()
        slider1 = tk.Scale(an_win, from_=1, to=50, orient=tk.HORIZONTAL, length=400, variable=size)
        slider1.set(24)
        slider1.pack()
        
    
        tk.Label(an_win, text="").pack()
        
        tk.Label(an_win, text="Images list").pack()
        
        
        file_listbox = tk.Listbox(an_win, selectmode=tk.SINGLE, width=90)
        file_listbox.pack(pady=10)


        
    
        button1 = tk.Button(an_win, text="Display", command=display_img_, width=20, height=2)
        button1.pack()
        
        tk.Label(an_win, text="").pack()
        
    
        button2 = tk.Button(an_win, text="Annotate", command=annotate_, width=20, height=2)
        button2.pack()
    
        tk.Label(an_win, text="").pack()
        
        
        button3 = tk.Button(an_win, text="Save", command=save_image, width=20, height=2)
        button3.pack()
    
        tk.Label(an_win, text="").pack()
        
        
        button4 = tk.Button(an_win, text="Back", command=exit_win, width=20, height=2)
        button4.pack()
    

    
        img_list()

        an_win.mainloop()
        
        cv2.destroyAllWindows()
         
        
    main_win()



# raw annotation


def img_annotation_raw():
    
    global app_run
    global app_metadata
    global anr_win
    
    
    global zoom_factor
    global x
    global y
    x = 1 
    y = 1
    zoom_factor = 1.0
    
    
        

    
    def annotate_raw_():
        
        
        if len(file_listbox.curselection()) > 0:
            global app_metadata
            global check_name
            
            check_name = app_metadata.images_dict['img_name'][app_metadata.images_dict['img_name'].index(file_listbox.get(file_listbox.curselection()[0]))]
            
            metadata = app_metadata.images_dict['metadata'][app_metadata.images_dict['img_name'].index(file_listbox.get(file_listbox.curselection()[0]))]
            image = app_metadata.images_dict['img'][app_metadata.images_dict['img_name'].index(file_listbox.get(file_listbox.curselection()[0]))]            
            
           
                

            if not isinstance(app_metadata.xml, pd.DataFrame):
                
                
                anr_win.destroy()
                
                
                warnign = ('WARNING!\n'
                           'Metadata for not found\n'
                           'Load metadata for this projection\n'
                           'Be sure that this projection belongs to this directory\n'
                           'and provide the right path to the XML file!\n'
                           'Next run "Annotation raw" one more time!\n'
                           )
                
                error_win(warnign, parent_window = None, color = 'yellow', win_name='Warning')
                
                
                metadata_window()

                
                main_win()
                
                
            else:
                
                if not isinstance(metadata, dict):
                    metadata = app_metadata.metadata
                
                anr_win.destroy()
                
                if os.path.exists(app_metadata.metadata_path):
                    path_to_images, file_name = os.path.split(app_metadata.metadata_path)
                    path_to_images, file_name = os.path.split(path_to_images)
                    
                    
                    image_dictinary, img_length, img_width = image_sequences(app_metadata.xml)
                    img, img_list = image_selection_app(image, img_length, img_width)
                    
                    if len(img_list) > 0:
    
                        init_path = os.getcwd()
                        
                        os.chdir(path_to_images)
                        
                        grid_im = resize_to_screen_img(img, factor = 2)
                        
                        
                        results = tiff_annotation(file_name, img_list, image_dictinary, metadata, grid = grid_im)
    
                        os.chdir(init_path)
                        
                        if isinstance(results, dict):
                            
                            app_metadata.add_annotated_raw(image, img, results)
                            
                        cv2.destroyAllWindows()
                        main_win()
                    
                    else:
                        
                        error_text = ('\nImages were not selected for annotation!!!')
                        error_win(error_text, parent_window = None)
                     
                        
                        cv2.destroyAllWindows()
                        main_win()
                        
                else:
                    
                    error_text = ('WARNING!\n'
                                  'Your Raw images directory has changed!!!\n'
                                  'Load metadata for this projection\n'
                                  'Be sure that this projection belongs to this directory\n'
                                  'and provide the right path to the XML file!\n'
                                  'Next run "Annotation raw" one more time!\n'
                                  'If in the previous analysis, the core of the image has been changed\n'
                                  'the numbers annotated to the Raw images can be different!!!'
                                  
                                  )
                    
                    
                    error_win(error_text, parent_window = None)
                    
                    metadata_window()
                    

                    main_win()




                
        else:
            
            error_text = ('\nSelect the image!!!')
            error_win(error_text, parent_window = anr_win)
         

        
    
    
    def save_raw_images_():
        global anr_win
        global channel
        
   
        if isinstance(app_metadata.annotation_series['img_data'], dict):
        
            global anr_win
            global lab_name_raw
            
            def exit_win():
                svr_win.destroy()
                
             
            def save_browse():
                
                save_path = filedialog.askdirectory()
                if save_path:
                    sv_box.delete(0, 'end')  
                    sv_box.insert('end', save_path)
                    app_metadata.add_tmp_path(save_path)
                
                
            def im_save():
                global lab_name_raw
    
                
                file = lab_name_raw.get("1.0", tk.END)
                file = re.sub(r'\n', '', file)
                file = re.sub(r'\s', '_', file)
                
                if len(file) > 0 and app_metadata.tmp_path != None:
                
                    init_path = os.getcwd()
                    
                    os.chdir(app_metadata.tmp_path)
                    
                    n = 0
                    while(True):
                        n += 1
                        tmp_file = file + '_' + str(n)
                        if not os.path.exists(tmp_file):
                            os.mkdir(tmp_file)
                            os.chdir(tmp_file)
                            
                            break
                            print(f"Folder '{tmp_file}' created successfully.")
                        else:
                            print(f"Folder '{tmp_file}' already exists.")
                    
                    
                    
                    n = 0
                    while(True):
                        n += 1
                        tmp_file_name = file + '_' + str(n) + '.' + str(type_box.get())
                        if os.path.exists(tmp_file_name) == False:
                            break
                       
                        
                       
                    path_to_images, _ = os.path.split(app_metadata.metadata_path)
                    
                    
                    
                    cv2.imwrite('raw_' + tmp_file_name, app_metadata.annotation_series['annotated_image'])
                    cv2.imwrite('grid_' + tmp_file_name, app_metadata.annotation_series['image_grid'])
                    
                    image_dict, _,_ = image_sequences(app_metadata.xml)
                    
                    select_pictures(
                        image_dict,
                        path_to_images,
                        path_to_save=os.getcwd(),
                        numbers_of_pictures=app_metadata.annotation_series['img_data']['images_num'],
                        chennels=[app_metadata.channel],
                        rm_slice_list = app_metadata.removal_list
                        
                    )                    
                    
                    
                    for n_inx in range(len(app_metadata.annotation_series['img_data']['images_num'])):
                        tmp_img_file = 'img_' + str(app_metadata.annotation_series['img_data']['images_num'][n_inx])
                        if isinstance(app_metadata.annotation_series['img_data']['projections'][n_inx], np.ndarray):
                            cv2.imwrite(os.path.join(tmp_img_file, 'projection.' +  str(type_box.get())), app_metadata.annotation_series['img_data']['projections'][n_inx])
                            if isinstance(app_metadata.annotation_series['img_data']['masks'][n_inx], np.ndarray):
                                
                                cv2.imwrite(os.path.join(tmp_img_file, 'mask_16bit.' + str(type_box.get())), app_metadata.annotation_series['img_data']['masks'][n_inx])
                                
                                bit8_mask = app_metadata.annotation_series['img_data']['masks'][n_inx]/(65535/255)
                                bit8_mask = bit8_mask.astype(np.uint8)
                                cv2.imwrite(os.path.join(tmp_img_file, 'mask_8bit.' + str(type_box.get())), bit8_mask)
                                
                                bit8_mask = bit8_mask/255
                                cv2.imwrite(os.path.join(tmp_img_file, 'mask_binary.' + str(type_box.get())), bit8_mask)
                                
                                del bit8_mask

                            else:

                                cv2.imwrite(os.path.join(tmp_img_file, 'mask_16bit.' + str(type_box.get())), np.full_like(app_metadata.annotation_series['img_data']['projections'][n_inx], 65535).astype(np.uint16))
                                cv2.imwrite(os.path.join(tmp_img_file, 'mask_8bit.' + str(type_box.get())), np.full_like(app_metadata.annotation_series['img_data']['projections'][n_inx], 255).astype(np.uint8))
                                cv2.imwrite(os.path.join(tmp_img_file, 'mask_binary.' + str(type_box.get())), np.full_like(app_metadata.annotation_series['img_data']['projections'][n_inx], 1).astype(np.uint8))


                            if isinstance(app_metadata.annotation_series['img_data']['annotations'][n_inx], np.ndarray):
                                cv2.imwrite(os.path.join(tmp_img_file, 'annotated_projection.' +  str(type_box.get())), app_metadata.annotation_series['img_data']['images'][n_inx])

                                cv2.imwrite(os.path.join(tmp_img_file, 'annotation.' + str(type_box.get())), app_metadata.annotation_series['img_data']['annotations'][n_inx])
        
        
                            else:
                                cv2.imwrite(os.path.join(tmp_img_file, 'annotated_projection.' +  str(type_box.get())), app_metadata.annotation_series['img_data']['projections'][n_inx])
                                cv2.imwrite(os.path.join(tmp_img_file, 'annotation.' + str(type_box.get())), np.zeros_like(app_metadata.annotation_series['img_data']['projections'][n_inx]))
                              
                        else:
                            shutil.rmtree(tmp_img_file)
                            

                    app_metadata.annotation_series['img_data'] = None
                    
                    os.chdir(init_path) 
                    
                    error_text = ('\nResults saved successfully!!!')
                    error_win(error_text, parent_window = None, color= 'green', win_name='Information')
                        

                    
                elif len(file) == 0:
                    error_text = ('\nProvide the file name!!!')
                    error_win(error_text, parent_window = svr_win)
                
                else:
                    error_text = ('\nProvide the path to the directory!!!')
                    error_win(error_text, parent_window = svr_win)
    
    
 
            svr_win = tk.Toplevel(anr_win)
            
        
            svr_win.geometry("500x400")
            svr_win.title("Save images")
        
            svr_win.iconbitmap(os.path.join(_icon_source,'jbi_icon.ico'))
            
            
            svr_win.transient(anr_win)

            svr_win.grab_set()
            
            
            tk.Label(svr_win, text="", anchor="w").pack()
            
            tk.Label(svr_win, text="Path to the save directory:").pack()
        
        
            sv_box = tk.Listbox(svr_win, width=70, height=1)
            sv_box.pack(pady=1)
            
            if app_metadata.tmp_path != None:
                sv_box.delete(0, 'end')  
                sv_box.insert('end', app_metadata.tmp_path)
            
    
            
            button1 = tk.Button(svr_win, text="Browse", command=save_browse, width=20, height=2)
            button1.pack()
            
            tk.Label(svr_win, text="").pack()
            
            
            label = tk.Label(svr_win, text="File name:")
            label.pack()
            
            tk.Label(svr_win, text="").pack()
            
            lab_name_raw = Text(svr_win, height=1, width=50)
            lab_name_raw.pack()
            
            
            tk.Label(svr_win, text="").pack()
    
            img_type = ['png', 'tiff', 'tif']
    
            type_box = ttk.Combobox(svr_win, values=img_type)
            
            type_box.current(0)
            
            type_box.pack()
            
            tk.Label(svr_win, text="").pack()
            
        
            button2 = tk.Button(svr_win, text="Save", command=im_save, width=20, height=2)
            button2.pack()
        
            tk.Label(svr_win, text="").pack()
            
            
          
        
            button5 = tk.Button(svr_win, text="Back", command=exit_win, width=20, height=2)
            button5.pack()
            
    
    
            svr_win.mainloop()
            
                 
        else:
            
            error_text = ('\nThe raw images were not selected and annotated!!!')
            error_win(error_text, parent_window = anr_win)
        

            
    
    def prep_inimg():
        global app_metadata
        global inimg
        
        inimg = resize_to_screen_img(app_metadata.images_dict['img'][app_metadata.images_dict['img_name'].index(file_listbox.get(file_listbox.curselection()[0]))].copy(), factor = 4)

    
    def display_image_tmp():
        global inimg
        global app_metadata
        global zoom_factor
        global x
        global y
        
        t = True
     
        resized_image = update_zoomed_region(inimg, size.get(), x, y)
        
        cv2.imshow('Display',resized_image) 
        
        key = cv2.waitKey(50) & 0xFF
        if key == ord('z'):
            cv2.setMouseCallback('Display', zoom_in)
        elif cv2.getWindowProperty('Display',cv2.WND_PROP_VISIBLE) < 1: 
    
            anr_win.destroy()
            cv2.destroyAllWindows()
            main_win() 
            
            t = False
    
        else:
            cv2.setMouseCallback('Display', lambda *args: None)  
    
        
        if t == True:
            anr_win.after(1, display_image_tmp)

   
    def display_img_():
        if len(file_listbox.curselection()) > 0:
            prep_inimg()
            display_image_tmp()
        else:
            
            error_text = ('\nSelect the image!!!')
            error_win(error_text, parent_window = anr_win)

       
      
        
    def img_list():
        global file_listbox
        file_listbox.delete(0, tk.END)  
        for filename in app_metadata.images_dict['img_name']:
            file_listbox.insert(tk.END, filename)
            
        
 

    def exit_win():
        global anr_win
        cv2.destroyAllWindows()
        anr_win.destroy() 
        
    
    
    def main_win():
        global file_listbox
        global anr_win
        global size


        anr_win = tk.Tk()
        
    
        anr_win.geometry("650x560")
        anr_win.title("Annotate raw images")
    
        anr_win.iconbitmap(os.path.join(_icon_source,'jbi_icon.ico'))
        
        
        tk.Label(anr_win, text="", anchor="w").pack()
        
        tk.Label(anr_win, text="Window size", anchor="w").pack()
        
        # Create a slider widget
        
        size = tk.DoubleVar()
        slider1 = tk.Scale(anr_win, from_=1, to=50, orient=tk.HORIZONTAL, length=400, variable=size)
        slider1.set(24)
        slider1.pack()
        
    
        tk.Label(anr_win, text="").pack()
        
        tk.Label(anr_win, text="Images list").pack()
        
        
        
        file_listbox = tk.Listbox(anr_win, selectmode=tk.SINGLE, width=90)
        file_listbox.pack(pady=10)


        
    
        button1 = tk.Button(anr_win, text="Display", command=display_img_, width=20, height=2)
        button1.pack()
        
        tk.Label(anr_win, text="").pack()
        
    
        button2 = tk.Button(anr_win, text="Annotate\nraw", command=annotate_raw_, width=20, height=2)
        button2.pack()
    
        tk.Label(anr_win, text="").pack()
        
        
        button3 = tk.Button(anr_win, text="Save\nselection", command=save_raw_images_, width=20, height=2)
        button3.pack()
    
        tk.Label(anr_win, text="").pack()
        
        
        button4 = tk.Button(anr_win, text="Back", command=exit_win, width=20, height=2)
        button4.pack()
    

    
        img_list()

        anr_win.mainloop()
        
        cv2.destroyAllWindows()
         
        
    main_win()



def project_manager_win():
    global app_metadata
    
    
    def save_project():
        global pm_win

        
        def exit_win():
            sp_win.destroy()
            
         
        def save_browse():
            global app_metadata
            save_path = filedialog.askdirectory()
            if save_path:
                sv_box.delete(0, 'end')  
                sv_box.insert('end', save_path)
                app_metadata.add_tmp_path(save_path)
    


        def save_object_to_file():
            global app_metadata
            
            if app_metadata.tmp_path != None:
                save_meta = app_metadata.copy()
                save_meta.tiffs_path = None
                save_meta.concat_path = None
                save_meta.saved_tiff_path = None
                save_meta.tmp_path = None
                save_meta.project_path = None
                save_meta.channel = None
                save_meta.annotation_series = None
                save_meta.resize_tmp = None
                save_meta.tmp_xml = None
                save_meta.save_current = None
                save_meta.project_path  = None
                
                
            
      
                with open(os.path.join(app_metadata.tmp_path, 'project_data.pjm'), 'wb') as file:
                    pickle.dump(save_meta, file)
                    
                error_text = ('\nProject saved successfully!!!')
                error_win(error_text, parent_window = None, color= 'green', win_name='Information')
                
            else:
                error_text = ('\nProvide the path to the directory!!!')
                error_win(error_text, parent_window = None)
                    
                
                
         


        
        sp_win = tk.Toplevel(pm_win)
    
        sp_win.geometry("500x250")
        sp_win.title("Save project")
    
        sp_win.iconbitmap(os.path.join(_icon_source,'jbi_icon.ico'))
        
        
        sp_win.transient(pm_win)

        sp_win.grab_set()
        
        
        tk.Label(sp_win, text="", anchor="w").pack()
        
        tk.Label(sp_win, text="Path to the save directory:").pack()
    
    
        sv_box = tk.Listbox(sp_win, width=70, height=1)
        sv_box.pack(pady=1)
        
        if app_metadata.tmp_path != None:
            sv_box.delete(0, 'end')  
            sv_box.insert('end', app_metadata.tmp_path)
        

        
        button1 = tk.Button(sp_win, text="Browse", command=save_browse, width=20, height=2)
        button1.pack()
        
        tk.Label(sp_win, text="").pack()
        
        
    
        button2 = tk.Button(sp_win, text="Save", command=save_object_to_file, width=20, height=2)
        button2.pack()
    
        tk.Label(sp_win, text="").pack()
        
        
      
    
        button5 = tk.Button(sp_win, text="Back", command=exit_win, width=20, height=2)
        button5.pack()
        
        
        sp_win.mainloop()
        
        
        
        
    def load_project():
        global pm_win

        
        def exit_win():
            lp_win.destroy()
            
         
            
        def browse_project():
            
            global app_metadata
            global input_path
            
            input_path = filedialog.askopenfilename()
            if input_path:
                pl_box.delete(0, 'end')  
                pl_box.insert('end', input_path)
                app_metadata.add_project_path(input_path)
              

                
                
        def load_object_from_file():
            global app_metadata
            
            if app_metadata.project_path != None and '.pjm' in app_metadata.project_path:
                with open(app_metadata.project_path, 'rb') as file:
                    app_metadata_tmp = pickle.load(file)
                    app_metadata.metadata_path = app_metadata_tmp.metadata_path
                    app_metadata.metadata =  app_metadata_tmp.metadata
                    app_metadata.xml =  app_metadata_tmp.xml
                    app_metadata.tiffs_path = None
                    app_metadata.concat_path = None
                    app_metadata.saved_tiff_path = None
                    app_metadata.images_dict = app_metadata_tmp.images_dict
                    app_metadata.tmp_path = None
                    app_metadata.annotation_series = {'annotated_image':None, 'image_grid':None, 'img_data':None}
                    app_metadata.resize_tmp = {'image':None, 'metadata':None, 'name':None}
                    app_metadata.removal_list = app_metadata_tmp.removal_list
                    
                    del app_metadata_tmp

                    
                error_text = ('\nProject loaded successfully!!!')
                error_win(error_text, parent_window = None, color= 'green', win_name='Information')
            
            else:
                error_text = ('\nProvide path to the project metadata file with *.pjm extension!!!')
                error_win(error_text, parent_window = None)
                
                
                
         
        
        lp_win = tk.Toplevel(pm_win)
    
        lp_win.geometry("500x250")
        lp_win.title("Load project")
    
        lp_win.iconbitmap(os.path.join(_icon_source,'jbi_icon.ico'))
        
        
        lp_win.transient(pm_win)

        lp_win.grab_set()
        
        
        tk.Label(lp_win, text="", anchor="w").pack()
        
        tk.Label(lp_win, text="Path to the project:").pack()
    
    
        pl_box = tk.Listbox(lp_win, width=70, height=1)
        pl_box.pack(pady=1)
        
        if app_metadata.project_path != None:
            pl_box.delete(0, 'end')  
            pl_box.insert('end', app_metadata.project_path)
        

        
        button1 = tk.Button(lp_win, text="Browse", command=browse_project, width=20, height=2)
        button1.pack()
        
        tk.Label(lp_win, text="").pack()
        
        
    
        button2 = tk.Button(lp_win, text="Load", command=load_object_from_file, width=20, height=2)
        button2.pack()
    
        tk.Label(lp_win, text="").pack()
        
        
      
    
        button5 = tk.Button(lp_win, text="Back", command=exit_win, width=20, height=2)
        button5.pack()
        
        
        lp_win.mainloop()
        

        
                 
            
    
    def main_win():
        global app_metadata
        global pm_win
        
        def clswin():
            
            global pm_win
            pm_win.destroy()
            

        global pm_win
        pm_win = tk.Tk()
        
    
        pm_win.geometry("350x270")
        pm_win.title("Project manager")
    
        pm_win.iconbitmap(os.path.join(_icon_source,'jbi_icon.ico'))
        
                
        tk.Label(pm_win, text="", anchor="w").pack()
        
        
        tk.Label(pm_win, text="Project management", anchor="w").pack()
        
        tk.Label(pm_win, text="", anchor="w").pack()

        
    
        button1 = tk.Button(pm_win, text="Load\nproject", command=load_project, width=20, height=2)
        button1.pack()
        
        tk.Label(pm_win, text="").pack()
        
    
        button2 = tk.Button(pm_win, text="Save\ncurrent", command=save_project, width=20, height=2)
        button2.pack()
    
        tk.Label(pm_win, text="").pack()
        
    
        
        
        button3 = tk.Button(pm_win, text="Back", command=clswin, width=20, height=2)
        button3.pack()
    


        pm_win.mainloop()
        
         
        
    main_win()
    
    

############################################################### --WINDOWS -- ###############################################################








############################################################### --APP -- ###############################################################


def run_app():
    
    def project_manager__():
        app_run.destroy()
        project_manager_win()
        main_window()
        
    
    def quit_():
        global app_run
        app_run.destroy()
        
    
    
    def load_metadata__():
        global app_run
        
        app_run.destroy()
        metadata_window()
        main_window()
        
        
    def concat__():
        global app_run
        
        if isinstance(app_metadata.metadata, dict):
            app_run.destroy()
            concatenate_window()
            main_window()
            
        else:
            
            app_run.destroy()
            warning_text = ('\nThe metadata is not loaded!!!\n'
                            'Load metadata from *.xml file!!!')
            
            error_win(warning_text, parent_window = None, color= 'yellow', win_name= 'Warning')
            
            metadata_window()
            main_window()


        
    def projection__():
        global app_run
        
        app_run.destroy()
        tiff_win_app()
        main_window()
        
        
    def managment__():
        global app_run
        
        app_run.destroy()
        img_manager_win()
        main_window()
        
        
    def merge__():
        global app_run
        
        app_run.destroy()
        img_merge_win()
        main_window()
        
        
    def scalebar__():
        global app_run
        
        app_run.destroy()
        img_scale_win()
        main_window()
        
        
    def base_annotation__():
        global app_run
        
        app_run.destroy()
        img_annotation_image()
        main_window()
        
        
        
    def raw_annotation__():
        global app_run
        
        app_run.destroy()
        img_annotation_raw()
        main_window()
        
        
        
        
    
    global license_w
    global app_run
    
    def main_window():
        
        global app_run
        app_run = tk.Tk()
        
    
        app_run.geometry("350x660")
        app_run.title("Main menu")
    
        app_run.iconbitmap(os.path.join(_icon_source,'jbi_icon.ico'))
        
        tk.Label(app_run, text="").pack()
         
        button = tk.Button(app_run, text="Project\nmanager", command=project_manager__, width=20, height=2)
        button.pack()
    
        tk.Label(app_run, text="").pack()
         
        button = tk.Button(app_run, text="Load\nmetadata", command=load_metadata__, width=20, height=2)
        button.pack()
        
        tk.Label(app_run, text="").pack()
        
        button1 = tk.Button(app_run, text="Concat\nimages", command=concat__, width=20, height=2)
        button1.pack()
    
        tk.Label(app_run, text="").pack()
        
        button2 = tk.Button(app_run, text="Z-projection", command=projection__, width=20, height=2)
        button2.pack()
        
        tk.Label(app_run, text="").pack()
        
        button2_1 = tk.Button(app_run, text="Images\nmanager", command=managment__, width=20, height=2)
        button2_1.pack()
        
        tk.Label(app_run, text="").pack()
        
        button3 = tk.Button(app_run, text="Merge\nimages", command=merge__, width=20, height=2)
        button3.pack()
    
        tk.Label(app_run, text="").pack()
        
        button4 = tk.Button(app_run, text="Add\nscale-bar", command=scalebar__, width=20, height=2)
        button4.pack()
    
        tk.Label(app_run, text="").pack()
        
        
        button5 = tk.Button(app_run, text="Annotate\nimage", command=base_annotation__, width=20, height=2)
        button5.pack()
    
        tk.Label(app_run, text="").pack()
        
        
        button6 = tk.Button(app_run, text="Annotate\nraw", command=raw_annotation__, width=20, height=2)
        button6.pack()
    
        tk.Label(app_run, text="").pack()
        
        button7 = tk.Button(app_run, text="Exit", command=quit_, width=20, height=2)
        button7.pack()
        
        
    
        app_run.mainloop()
    
    main_window()




global app_metadata
app_metadata = Metadata()


def jbs_main_win():
    

    def exit_fun():
        jbs_win.destroy()
        
    def res_():
        
        global app_metadata
 
        app_metadata.metadata_path = None
        app_metadata.metadata = None
        app_metadata.xml = None
        app_metadata.tiffs_path = None
        app_metadata.concat_path = None
        app_metadata.saved_tiff_path = None
        app_metadata.images_dict = {'img':[], 'metadata':[], 'img_name':[]}
        app_metadata.tmp_path = None
        app_metadata.annotation_series = {'annotated_image':None, 'image_grid':None, 'img_data':None}
        app_metadata.resize_tmp = {'image':None, 'metadata':None, 'name':None}
        app_metadata.removal_list = None
        app_metadata.project_path = None
    

        
    def licence():
        
        global jbs_win
        global license_w

        license_w = tk.Toplevel(jbs_win)
        
        
        license_w.title("License")

        license_w.iconbitmap(os.path.join(_icon_source,'jbi_icon.ico'))
            
            
        lic = (
            '   MIT License\n'
            '   Copyright (c) 2024 Jakub Kubis JBS\n'
            '   Permission is hereby granted, free of charge, to any person obtaining a copy\n'
            '   of this software and associated documentation files (the "Software"), to deal\n'
            '   in the Software without restriction, including without limitation the rights\n'
            '   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n'
            '   copies of the Software, and to permit persons to whom the Software is\n'
            '   furnished to do so, subject to the following conditions:\n'

            '   The above copyright notice and this permission notice shall be included in all\n'
            '   copies or substantial portions of the Software.\n'

            '   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n'
            '   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n'
            '   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n'
            '   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n'
            '   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n'
            '   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n'
            '   SOFTWARE.'
         )


        tk.Label(license_w, text=lic, anchor="w", justify="left").pack()
        
        
        license_w.mainloop()
        
    
    def contact():
        
        global jbs_win
        global conw

        conw = tk.Toplevel(jbs_win)
        
        
        conw.title("Contact")

        conw.iconbitmap(os.path.join(_icon_source,'jbi_icon.ico'))
            
            
        con = (
            
            '   Organization:\n'
            '   Institute of Bioorganic Chemistry\n'
            '   Polish Academy of Sciences\n'
            '   Zygmunta Noskowskiego 12/14, 61-704 Poznań\n'
            '   Poland\n\n\n'


            '   Creator:\n'
            '   Jakub Kubis\n'
            '   JBioSystem\n'

            '   Email: jbiosystem@gmail.com'
            
         )


        tk.Label(conw, text=con, anchor="w", justify="left").pack()
        
        
        conw.mainloop()
           
           
            
            
            
    def manual():
        webbrowser.open('https://github.com/jkubis96/JIMG/tree/v.2.0.0', new=2)
        
        
    def start_run():
        
        global jbs_win
        
        jbs_win.destroy()
        
        run_app()
        
        main_win_1st()
        
        
    def main_win_1st():
        global jbs_win
        
        
        jbs_win = tk.Tk()
    
        jbs_win.geometry("825x680")
        jbs_win.title("Image system")
    
        jbs_win.iconbitmap(os.path.join(_icon_source,'jbi_icon.ico'))
    
        ico_path = os.path.join(_icon_source,'jbs_icon.png')
        img_pil = Image.open(ico_path).resize((150, 150))
        img = ImageTk.PhotoImage(img_pil)
    
        label = tk.Label(jbs_win, image=img)
        label.grid(row=0, column=0, rowspan=2, padx=15, pady=15)
        
        
        
        jbs_path = os.path.join(_icon_source,'jbi_icon.png')
        img_pil_jbs = Image.open(jbs_path).resize((150, 150))
        img_jbs = ImageTk.PhotoImage(img_pil_jbs)
    
        label_jbs = tk.Label(jbs_win, image=img_jbs)
        label_jbs.grid(row=1, column=0, rowspan=25, padx=15, pady=15)
    
        text = (
            '\n This tool was created for handling high-resolution images from the Opera Phenix Plus High-Content\n'
			' Screening System, including operations such as concatenating raw series of images, z-projection,\n'
			' channel merging, image resizing, etc. Additionally, we have included options for annotating specific\n' 
			' parts of images and selecting them for further analysis, for example, teaching ML/AI algorithms.\n\n'

			' Certain elements of this tool can be adapted for data analysis and annotation in other imaging systems.\n'
			' For more information, please feel free to contact us!\n'
        )
        
    
    
        tk.Label(jbs_win, text=text, anchor="w", justify="left").grid(row=0, column=5, columnspan=1, pady=5)
       
    
        button1 = tk.Button(jbs_win, text="Start", command=start_run, width=20, height=3)
        button1.grid(row=20, column=4, columnspan=5, pady=10)
        
        button2 = tk.Button(jbs_win, text="Reset", command=res_, width=20, height=3)
        button2.grid(row=22, column=4, columnspan=2, pady=5)
    
        button3 = tk.Button(jbs_win, text="Contact", command=contact, width=20, height=3)
        button3.grid(row=24, column=4, columnspan=2, pady=5)
    
        button4 = tk.Button(jbs_win, text="Manual", command=manual, width=20, height=3)
        button4.grid(row=26, column=4, columnspan=2, pady=5)
    
        button5 = tk.Button(jbs_win, text="License", command=licence, width=20, height=3)
        button5.grid(row=28, column=4, columnspan=2, pady=5)

        button6 = tk.Button(jbs_win, text="Exit", command=exit_fun, width=20, height=3)
        button6.grid(row=30, column=4, columnspan=2, pady=5)
        
        
       
        
        footer = (
            '\n\n'
            'Institute of Bioorganic Chemistry PAS, Poznań, 2024'
            
        )
    
    
        tk.Label(jbs_win, text=footer, anchor="w", justify="left").grid(row=31, column=4, columnspan=2, pady=5)
    
        jbs_win.mainloop()
        
        
        
    main_win_1st()
    
  
    
############################################################### --APP -- ###############################################################

    

# run 

if __name__ == "__main__":
    multiprocessing.freeze_support()
    jbs_main_win()
    

############################### Main code / ####################################


 #       _  ____   _         _____              _                      
 #      | ||  _ \ (_)       / ____|            | |                     
 #      | || |_) | _   ___ | (___   _   _  ___ | |_  ___  _ __ ___   
 #  _   | ||  _ < | | / _ \ \___ \ | | | |/ __|| __|/ _ \| '_ ` _ \  
 # | |__| || |_) || || (_) |____) || |_| |\__ \| |_|  __/| | | | | | 
 #  \____/ |____/ |_| \___/|_____/  \__, ||___/ \__|\___||_| |_| |_|  
 #                                   __/ |                                   
 #                                  |___/      
