from JIMG import jimg
import os

# metadata repairing & image concatenate

jimg.split_channels(path_to_images = 'Images', path_to_save = '')

    
image_info, metadata = jimg.xml_load(path_to_xml = 'Images/Index.idx.xml')
    
image_info, fig = jimg.repair_blanks(xml_file = image_info)


image_info, fig = jimg.detect_outlires(xml_file = image_info, list_of_out = []) 

   
image_queue, img_length, img_width = jimg.image_sequences(opera_coordinates = image_info)

channels = ['ch1','ch2']
overlap = 0.05
resize = 2
n_proc = 5 
par_type = 'processes'

work_dir = os.getcwd()

path_to_images = os.path.join(work_dir, 'Images')
path_to_save =  work_dir

res_metadata = jimg.image_concatenate(path_to_images , path_to_save , image_queue, metadata, img_length, img_width, overlap, channels, resize, n_proc, par_type)
  

channels = ['ch1']

jimg.resize_tiff(channels, metadata, prefix = 'resized' , height = None, width = None, resize_factor = None)
    
projection1 = jimg.z_projection(path_to_tiff = 'channel_ch1.tiff', stack_check = False)

projection2 = jimg.z_projection(path_to_tiff = 'channel_ch2.tiff', stack_check = False)

merged = jimg.merge_images([projection1,projection2])

     
resized, res_met = jimg.resize_projection(image = merged, metadata = res_metadata, height = None, width = None, resize_factor = 2)
    
scale_im = jimg.add_scalebar(image = resized, metadata = res_met)

scale_im = jimg.add_scalebar(image = merged, metadata = res_metadata)



path_to_opera_projection = 'projection_ch1.png'

numbers_of_pictures = jimg.image_grid(path_to_opera_projection, img_length, img_width)
   

image_dictionary = image_queue
path_to_images = 'Images'
chennels = ['ch1']

jimg.select_pictures(image_dictionary, path_to_images, path_to_save, numbers_of_pictures, chennels)
    
   
            

  