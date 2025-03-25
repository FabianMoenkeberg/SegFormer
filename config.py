GPU_USAGE = True
load_local_model = True
train_model = False
load_entire_dataset = True
reduced_full_dataset = False
load_additional_data = False

name = 'galilei'

name_loadModel = 'model-galilei-seg.h5'

path_data = '/data_eye/GalileiData/Fischer09062021'
path_data = '/data_eye/eyeSegmentationFemto/applanated/segmented'
path_additional_data = '/data_eye/eyeSegmentationFemto/segmentedTopviewLiquidClean'

path_save_test = 'res_test'
path_save_train = 'res_train'
path_save_data_test = 'resImg_test'
path_save_data_train = 'resImg_train'

path_train = ''
path_test = 'test/'

seed = 1

N_samples = 500
N_samples_add = 2000
N_segClasses = 2 #for Galilei
# N_segClasses = 150

im_width = 256
im_height = 256
im_chan = 3

out_width = im_width//2
out_height = im_height//2

nEpochs = 20
validation_split = 0.1
batch_size = 30

lr = 0.001

