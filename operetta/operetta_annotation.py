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




def image_concatenate(path_to_images:str, imgs:pd.DataFrame, img_length:int, img_width:int, scale_factor:int(50), chanels:list, n_thread:int):
     
    
    for obj in gc.get_objects():   
        if isinstance(obj, h5py.File):  
            try:
                obj.close()
            except:
                pass 
            
            
    if os.path.exists('images.h5'):
        os.remove('images.h5')
        
    if os.path.exists('images2.h5'):
        os.remove('images.h5')
    
    def par_1(start, stop, q, path_to_images, img_width, imgs):
        stop = stop + img_width * (q+1)
        start = start + img_width * q
        tmp = imgs['queue'][start:stop]
        
        list_p = []
        for t in tmp:
            list_p.append(str([f for f in tmp_img if str(re.sub('\n','', str(t)) + 'p') in f and str('p'+st) in f][0]))
            
        data = []
        for img in list_p:
            if os.path.exists(os.path.join(path_to_images, img)):
                data.append(cv2.imread(os.path.join(path_to_images, img), cv2.IMREAD_ANYDEPTH))
            else:
                data.append(black_img)
             
        #factor calculate
        
        q10 = []
        for img in data:
            q10.append(np.quantile(img, 0.33))
            
      
            
        #rescaling standarization
        
        for n, img in enumerate(data):
            data[n] = data[n] - np.mean(q10)
            data[n][data[n] < 0] = 0
            tmp = data[n]
            data[n] =  cv2.normalize(data[n], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            data[n] = cv2.equalizeHist(data[n])
            data[n] = (data[n] - data[n].min()) / (data[n].max() - data[n].min())
            data[n] = np.nan_to_num(data[n])
            data[n] = data[n]*tmp
            
    
        #cocnacte len
        
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
    
            #tmp_img = [i for i in tmp_img if re.sub('\n','', str(img)) in i]
            start = 0
            stop = 0
            Parallel(n_jobs=n_thread, prefer="threads")(delayed(par_1)(start, stop, q, path_to_images, img_width, imgs) for q in range(0,img_length))
    
           
                
            data = []
            for q in range(0,img_length):
                data.append(images_tmp[[f for f in images_tmp.keys() if 'lane_' + str(q) + '-deep' in f][0]][:])
              
    
            images_tmp.close()
    
            os.remove(os.path.join(path_to_images, 'images.h5'))
            
                           
            #rescaling standarization
            q10 = []
            for img in data:
                q10.append(np.quantile(img, 0.33))
                
          
                
            for n, img in enumerate(data):
                data[n] = data[n] - np.mean(q10)
                data[n][data[n] < 0] = 0
                data[n] = cv2.convertScaleAbs(data[n], alpha=scale_factor/np.average(data[n][data[n] > np.quantile(data[n], 0.95)]), beta=0)

              
             
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




def z_projection(path_to_tiff:str, color:str):

      
    stack = tiff.imread(path_to_tiff)
  
    
    projection = np.max(stack, axis=0)
  


    img_rgb = cv2.cvtColor(projection, cv2.COLOR_GRAY2RGB)
    
    if color == 'green':
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        lut[:, 0, 1] = np.arange(256)
        result = cv2.LUT(img_rgb, lut)


    elif color == 'red':
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
    
        for i in range(256):
            lut[i, 0, 2] = i
        
        result = cv2.LUT(img_rgb, lut)
        
    elif color == 'blue':
        lut = np.zeros((256, 1, 3), dtype=np.uint8)

        for i in range(256):
            lut[i, 0, 0] = i
                
        result = cv2.LUT(img_rgb, lut)
        
    elif color == 'magenta':
   
        lut = np.zeros((256, 1, 3), dtype=np.uint8)

        for i in range(256):
            lut[i, 0, 0] = i
            lut[i, 0, 2] = i

       
        result = cv2.LUT(img_rgb, lut)
     
    elif color == 'grey':
        lut = np.zeros((256, 1, 1), dtype=np.uint8)

        for i in range(256):
            lut[i, 0, 0] = i
        
        result = cv2.LUT(img_rgb, lut)


    
    return result


def merge_images(image_list:list, intensity_factors:list):

    result = None
    
    for i, image in enumerate(image_list):
        if result is None:
            result = image.astype(float) * intensity_factors[i]
        else:
            result = cv2.addWeighted(result, 1, image.astype(float) * intensity_factors[i], 1, 0)
    
    result = result.astype('uint8')
    
    return result




def image_grid(path_to_opera_projection:str, img_length:int, img_width:int):


    cv2.namedWindow('Image')
    
    resize_factor = 10
    image = cv2.imread(path_to_opera_projection)
    tmp = image
    image = cv2.resize(image, (int(img_width* resize_factor), int(img_length* resize_factor)))  

    
    def nothing(x):
        pass
    
    def resize(image, img_length, img_width, resize_factor):
    
        
    
        for sqr in range(0,img_length):
            for sqr2 in range(1,img_width+1):
    
    
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
    cv2.imshow('Image', image2)
    
    cv2.namedWindow('Tracebar')
    cv2.createTrackbar('Size', 'Tracebar',0,100, nothing)
    cv2.resizeWindow("Tracebar", 500, 20)
    cv2.createTrackbar('Quit', 'Tracebar',0,1, nothing)
    
    


    rf = 0
    rfch = 0
    ex = 0
    
    ch = None
    while ch != 27:
        cv2.imshow('Image',image2)
        ch = cv2.waitKey(1) & 0xFF
        rfch = (int(cv2.getTrackbarPos('Size','Tracebar'))) 
        q = (int(cv2.getTrackbarPos('Quit','Tracebar'))) 
        if q == 1:
            break
        

        
        if (rf != rfch):
            image = tmp
            image = cv2.resize(image, (int(resize_table['width'][resize_table['range'] == rf][rf]), int(resize_table['height'][resize_table['range'] == rf][rf])))  
            image2 = resize(image, img_length, img_width, int(resize_table['resize_factor'][resize_table['range'] == rf][rf]))

            rf = rfch


    
    cv2.destroyAllWindows()
    
    

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