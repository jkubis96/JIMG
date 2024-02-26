# from JIMG.functions import jimg as jg
from JIMG.app.load_app import run
run()



from JIMG.functions import jimg as jg


# Raw images dealing


path_to_inx = 'Images/Index.idx.xml'

#loading metadata
img_info, metadata =  jg.xml_load(path_to_xml = path_to_inx)


#repairing metadata
img_info, figure1 = jg.repair_image(image_info=img_info, dispaly_plot = True)

img_info, figure2 = jg.manual_outlires(image_info = img_info, list_of_out = [], dispaly_plot = False)


# split channels


jg.split_channels(path_to_images = 'Images', path_to_save = '')


# image concatenationm

image_queue, img_length, img_width = jg.image_sequences(img_info)

path_to_images = 'Images'
path_to_save = ''
overlap = 0.05
channels = ['ch1', 'ch2']

jg.image_concatenate(path_to_images, path_to_save, image_queue, metadata, img_length, img_width, overlap, channels, resize = 2, n_proc = 4, par_type = 'processes')




# tiff adjustment


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

# adjust resolution information in metadata (if the tiff file during concatenation was resized)

metadata['X_resolution[um/px]'] = x
metadata['Y_resolution[um/px]'] = y

resized_tiff, res_metadata = jg.resize_tiff(image = tiff_file, metadata = metadata, height = None, width = None, resize_factor = 2)


jg.save_tiff(tiff_image = jg.resized_tiff, path_to_save = 'resized_tiff.tiff', metadata = res_metadata)


# z projection


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



# merging images 

projection_ch1 = jg.load_image(path = 'projection_ch1.png')

projection_ch2 = jg.load_image(path = 'projection_ch2.png')


merged_image = jg.merge_images(image_list = [projection_ch1, projection_ch2], intensity_factors = [1,1])

jg.display_preview(merged_image)


jg.save_image(image = merged_image, path_to_save = 'merged_image.png')









