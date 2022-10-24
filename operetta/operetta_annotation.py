import shutil
import os
import numpy as np
import re
import pandas as pd
import cv2
import pandas as pd
import itertools

    
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
    
    return df


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




def image_grid(path_to_opera_projection:str, img_length:int, img_width:int, resize_factor:int = 100):
    cv2.namedWindow('Image')
    
    image = cv2.imread(path_to_opera_projection)
    image = cv2.resize(image, (int(img_width* resize_factor), int(img_length* resize_factor)))  

    
    def nothing(resize_factor):
        pass
    
    def resize(image, img_length, img_width, resize_factor):
    
        
    
        for sqr in range(0,img_width):
            for sqr2 in range(1,img_length+1):
    
    
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
    resize_table['factor'][resize_table['range'].isin(range(1,51))] = range(-50,0)
    resize_table['factor'][resize_table['range'].isin(range(51,101))] = range(1,51)
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
            image = cv2.imread(path_to_opera_projection)
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
    
        images_list=[x for x in images_list if re.sub('\n','', str(selected['queue'][selected['image_num'] == num][n])) in x]
        
        if not os.path.exists(os.path.join(path_to_save,'img_' + str(num))):
            os.mkdir(os.path.join(path_to_save,'img_' + str(num)))
            
        for image in images_list:
            shutil.copy(os.path.join(path_to_images,image),os.path.join(path_to_save,'img_' + str(num)))