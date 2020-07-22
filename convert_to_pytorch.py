import os
from pprint import pprint
import tensorflow as tf
from inception import Inception3
import torch
import numpy as np
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

model = Inception3()

#CUB model trained in tensorflow
#tf_path = os.path.abspath('/home/m.bharti/svn/FineGrainedClassification/cvpr18-inaturalist-transfer/checkpoints/cub_200/auxlogits_aug_resize/model.ckpt-2810')  # Path to our TensorFlow checkpoint


## iNaturalist path, does not work
#tf_path = os.path.abspath('./checkpoints/inception/inception_v3_iNat_299.ckpt')

print_tensors_in_checkpoint_file(file_name='./checkpoints/inception/inception_v3_iNat_299.ckpt', tensor_name='', all_tensors=True)

reader = pywrap_tensorflow.NewCheckpointReader('./checkpoints/inception/inception_v3_iNat_299.ckpt')
init_vars = reader.get_variable_to_shape_map()

#print_tensors_in_checkpoint_file(file_name='./checkpoints/inception/inception_v3_iNat_299.ckpt', all_tensors=True, tensor_name='')

#init_vars = tf.train.list_variables(tf_path)
#pprint(init_vars)
#print(len(init_vars))

tf_vars = []
for key in init_vars:
    #print("Loading TF weight {} with shape {}".format(name, shape))
    print(key)
    array = reader.get_tensor(key)
   # array1 = tf.train.load_variable(tf_path, name)
    tf_vars.append((key, array))

print("Total vars {}".format(len(tf_vars)))

count =0

total_aux_variables =0
total_logits_variables = 0
# FOr each variable in the PyTorch model
for full_name, array in tf_vars:
    # skip the prefix ('model/') and split the path-like variable name in a list of sub-path
    if full_name  == 'global_step':
        continue

    name = full_name[12:].split('/')

    if full_name == 'InceptionV3/Logits/Conv2d_1c_1x1/biases/Momentum':
        continue

    if full_name == 'InceptionV3/Logits/Conv2d_1c_1x1/weights/Momentum':
        continue

    if full_name == 'InceptionV3/AuxLogits/Conv2d_1b_1x1/BatchNorm/beta/Momentum':
        continue

    if full_name == 'InceptionV3/AuxLogits/Conv2d_1b_1x1/weights/Momentum':
        continue

    if full_name == 'InceptionV3/AuxLogits/Conv2d_2a_5x5/BatchNorm/beta/Momentum':
        continue

    if full_name == 'InceptionV3/AuxLogits/Conv2d_2a_5x5/weights/Momentum':
        continue

    if full_name == 'InceptionV3/AuxLogits/Conv2d_2b_1x1/biases/Momentum':
        continue

    if full_name == 'InceptionV3/AuxLogits/Conv2d_2b_1x1/weights/Momentum':
        continue

    print(full_name)

    # Initiate the pointer from the main model class
    pointer = model

    if name[0] == 'AuxLogits':
        total_aux_variables = total_aux_variables + 1
        pprint(full_name)


    if name[0] == 'Logits':
        total_logits_variables = total_logits_variables + 1
       # pprint(full_name)
       # continue

    # We iterate along the scopes and move our pointer accordingly
    for m_name in name:

        l = [m_name]

        # Convert parameters final names to the PyTorch modules equivalent names
        if l[0] == 'weights':
            pointer = getattr(pointer, 'conv')
            pointer = getattr(pointer, 'weight')

            array = np.transpose(array, (3, 2, 0, 1))

            assert pointer.shape == array.shape
            #print("Initialize PyTorch weight {}".format(name))
            pointer.data = torch.from_numpy(array)
            count = count + 1
        elif l[0] == 'biases': #Batch Normalisation
            pointer = getattr(pointer, 'conv')
            pointer = getattr(pointer, 'bias')
            assert pointer.shape == array.shape
            pointer.data = torch.from_numpy(array)
            count = count + 1
        elif l[0] == 'beta':  # Batch Normalisation
            pointer = getattr(pointer, 'bias')
            assert pointer.shape == array.shape
            pointer.data = torch.from_numpy(array)
            count = count + 1
        elif l[0] == 'gamma': #Batch Normalisation not present for this model
            pointer = getattr(pointer, 'weight')
            pointer.data = torch.from_numpy(array)
            count = count + 1
        elif l[0] == 'moving_mean': #Batch Normalisation
            assert getattr(pointer, 'running_mean').shape == array.shape

            pointer.__setattr__('running_mean', torch.from_numpy(array))
            #pointer = torch.from_numpy(array)
            count = count + 1
        elif l[0] == 'moving_variance': #Batch Normalisation
            assert getattr(pointer, 'running_var').shape == array.shape
            pointer.__setattr__('running_var', torch.from_numpy(array))
            count = count + 1

        else:
            pointer = getattr(pointer, l[0])
            #print("Moving forward {}".format(l[0]))



pprint("Updated {} parameters".format(count))
#torch.save(model.state_dict(), './cub_inceptionv3_again.pth')

torch.save(model.state_dict(), './iNat_inceptionv3.pth')






