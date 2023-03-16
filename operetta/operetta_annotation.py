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
from tkinter import ttk
from skimage import io, filters
import pkg_resources


warnings.filterwarnings("ignore", category=RuntimeWarning)
    
def split_chanels(path_to_images:str, path_to_save:str):
    
    
    
    chanels=os.listdir(path_to_images)
    chanels=[re.sub('.*-','',x)  for x in chanels if 'tiff' in x]
    chanels=[re.sub('sk.*','',x)  for x in chanels if 'tiff' in x]
    
    chanels = np.unique(chanels).tolist()
    
    for ch in chanels:
            
        
        if not os.path.exists(os.path.join(path_to_save, str(ch))):
            os.mkdir(os.path.join(path_to_save, str(ch)))
    
  
        
        images_list=os.listdir(path_to_images)
    
        images_list=[x for x in images_list if str(ch) in x]
        images_list = images_list + ['Index.idx.xml']
        
        if not os.path.exists(os.path.join(path_to_save, str(ch))):
            os.mkdir(os.path.join(path_to_save, str(ch)))
            
        for image in images_list:
            shutil.copy(os.path.join(path_to_images,image),os.path.join(os.path.join(path_to_save, str(ch))))
    
    
    
    
    

def xml_load(path_to_opera_xml:str):
    
    name = []
    x = []
    y = []
    
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
                
                
    
    df = pd.DataFrame(df)
    df['name'] = [re.sub('p.*', '', x) for x in df['name']]
    
    df['y'] = df['y']*-1
    
    
    df = df.drop_duplicates()
    df['num'] = range(1,len(df['name'])+1)
    
    df = df.reset_index(drop = True)
    
    return df



def detect_outlires(xml_file:pd.DataFrame, list_of_out:list = []):
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
            
            
        physical_size = (16, 14)  # Example size in inches
        pixels_per_inch = 300  # Example DPI of display device
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
    


def repair_blanks(xml_file:pd.DataFrame):
   
   
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

  
    
def image_sequences(opera_coordinates:pd.DataFrame):
    
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





def image_concatenate(path_to_images:str, imgs:pd.DataFrame, img_length:int, img_width:int, overlap:int, chanels:list, n_thread:int):
     
    
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
        for n in range(1, len(data)):
            data[n-1] = data[n-1][:, :-int(col*overlap)]
                     


        
        data = np.concatenate(data, axis = 1)
        images_tmp.create_dataset('lane_' + str(q) + '-deep_' + str(st) + '-chanel_' + str(ch),  data=data)
        del data
    
    
    
    images_list=os.listdir(path_to_images)
    deep = np.unique([re.sub('-.*','', re.sub('.*p', '', n)) for n in images_list if '.tiff' in n])
    
   
    
    for ch in chanels:
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
            for n in range(1, len(data)):
                data[n-1] = data[n-1][:-int(row*overlap), :]
                
        
               
             
            data = np.concatenate(data, axis = 0)
            
            images_tmp2.create_dataset('deep_' + str(st) + '-chanel_' + str(ch),  data=data)
    
        data = []
        for q in tqdm(images_tmp2.keys()):



      
            data.append(images_tmp2[q][:])
                

        data = np.stack(data)
        
    
        tiff.imwrite('chanel_' + str(ch) + '.tiff', data, imagej=True)
            
        images_tmp2.close()
        
        del data
        
        os.remove(os.path.join(path_to_images, 'images2.h5'))





def z_projection(path_to_tiff:str):

    if not os.path.exists(path_to_tiff):
        
        print('\nImage does not exist. Check the correctness of the path to image')
        
    else:
        
        global stack
        
        stack = tiff.imread(path_to_tiff)
        
        median_image = np.median(stack, axis=0)

        # Calculate the pixel-wise difference between each image and the median image
        diff_stack = np.abs(stack - median_image)

        # Threshold the difference stack
        threshold_value = filters.threshold_otsu(diff_stack)
        thresholded_stack = diff_stack > threshold_value

        # Drop outliers
        mean_intensity = np.mean(thresholded_stack, axis=(1,2))
        outlier_indices = np.where(mean_intensity > np.mean(mean_intensity) + np.std(mean_intensity))[0]
        
        stack = np.delete(stack, outlier_indices, axis=0)
        
        
        
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
        
        threshold = tk.DoubleVar()
        slider3 = tk.Scale(window, from_=0, to=2000, orient=tk.HORIZONTAL, length=400, variable=threshold)
        slider3.set(150)
        slider3.pack()
        
        tk.Label(window, text="").pack()
        
        label5 = tk.Label(window, text="Brightness", anchor="w")
        label5.pack()
        
        brightness = tk.DoubleVar()
        slider5 = tk.Scale(window, from_=0, to=200, orient=tk.HORIZONTAL, length=400, variable=brightness)
        slider5.set(100)
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
        
        projections = ["avg", "max", "min", "sdt"]
    
        projections_type = ttk.Combobox(window, values=projections)
        
        projections_type.current(0)
        
        projections_type.pack()
        
       
        button_finished = tk.BooleanVar(value=False)
        
        def active_changes():
           
            global img_gamma
            
            if projections_type.get() == 'avg':
                projection = np.average(stack, axis=0)
            elif projections_type.get() == 'max':
                projection = np.max(stack, axis=0)
            elif projections_type.get() == 'min':
                projection = np.min(stack, axis=0)
            elif projections_type.get() == 'std':
                projection = np.std(stack, axis=0)
                
                
            img = projection.copy()
            
            img = img.astype(np.int64)
            
            color = combobox.get()
            
            img = img - int(np.std(img)*(threshold.get()/200))
            img[img <= 0] = 0
            
            
            img = cv2.convertScaleAbs(img, dst = cv2.CV_16U, alpha = int(contrast.get()/ 10),beta =  int(brightness.get() - 100))


            img_norm = img / np.max(img)
            img = np.power(img_norm, (gamma.get()+1)/10)
            
            img = (img * (2**16 - 1)).astype(np.uint16)


            
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
                
            elif color == 'cyan':
                img_gamma[:,:,1] = img
                img_gamma[:,:,2] = img
            
            elif color == 'yellow':
                img_gamma[:,:,0] = img
                img_gamma[:,:,1] = img
             
            elif color == 'grey':
                img_gamma = img
        
        
            
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
            
            projection = np.average(stack, axis=0)

            g = int(float(np.median(projection)/np.std(projection) * 150)*(np.mean(projection)/np.median(projection))**2)

            
            gamma.set(30)         
            slider2.set(30)
                     
            threshold.set(g)
            slider3.set(g)
            
            
            brightness.set(100)
            slider5.set(100)
      
            contrast.set(15) 
            slider6.set(15)
            
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
    
    intensity_factors = []
    for bt in range(len(image_list)):
        intensity_factors.append(1)
        
    def merge1():
        global result
        result = None
        
        for i, image in enumerate(image_list):
            if result is None:
                result = image.astype(float) * intensity_factors[i]
            else:
                result = cv2.addWeighted(result, 1, image.astype(float) * intensity_factors[i], 1, 0)
        
        result = result.astype('uint16')
        
        
 
       
  
    
    window = tk.Tk()
    
    window.geometry("500x600")
    window.title("MERGE CHANELS")

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
                result = image.astype(float) * intensity_factors[i]
            else:
                result = cv2.addWeighted(result, 1, image.astype(float) * intensity_factors[i], 1, 0)
        
        result = result.astype('uint16')
    
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
        
        import tkinter as tk 
        from tkinter import ttk, Text
        
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
    
    

def select_pictures(image_dictinary:pd.DataFrame, path_to_images:str, path_to_save:str, numbers_of_pictures:list):
    
    selected = image_dictinary[image_dictinary['image_num'].isin(numbers_of_pictures)]
    selected = selected.reset_index()
    
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    
    for n, num in enumerate(selected['image_num']):
        
        images_list=os.listdir(path_to_images)
    
        images_list=[x for x in images_list if str(re.sub('\n','', (str(selected['queue'][selected['image_num'] == num][n]))) + 'p') in x]
        
        if not os.path.exists(os.path.join(path_to_save,'img_' + str(num))):
            os.mkdir(os.path.join(path_to_save,'img_' + str(num)))
            
        for image in images_list:
            shutil.copy(os.path.join(path_to_images,image),os.path.join(path_to_save,'img_' + str(num)))