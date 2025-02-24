GPU_USAGE = True
loadModel = False
trainModel = True
load_entire_dataset = False

name = 'galilei'

name_loadModel = 'model-galilei-seg.h5'

path_data = 'D:\data\eyeSegmentation\GalileiData\GalileiColourSegmentationData'
path_data = '/data/GalileiData/Fischer09062021'

path_save_test = 'res_test'
path_save_train = 'res_train'
path_save_data_test = 'resImg_test'
path_save_data_train = 'resImg_train'

path_train = ''
path_test = 'test/'

seed = 1


N_segClasses = 4 #for Galilei



im_width =128
im_height = 128
im_chan = 3

out_width = 128
out_height = 128

nEpochs = 10
validation_split = 0.1
batch_size = 10

lr = 0.00002

