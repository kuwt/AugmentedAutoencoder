import cv2
import tensorflow as tf
import numpy as np
import glob
import os
import time
import argparse
import configparser
import matplotlib.pyplot as plt
from auto_pose.ae import factory, utils

from datetime import datetime
import os

parser = argparse.ArgumentParser()
parser.add_argument("experiment_name")
parser.add_argument("-f", "--file_str", required=True, help='folder or filename to image(s)')
# parser.add_argument("-gt_bb", action='store_true', default=False)
arguments = parser.parse_args()
full_name = arguments.experiment_name.split('/')
experiment_name = full_name.pop()
experiment_group = full_name.pop() if len(full_name) > 0 else ''

file_str = arguments.file_str
if os.path.isdir(file_str):
    files = sorted(glob.glob(os.path.join(str(file_str),'*.png'))+glob.glob(os.path.join(str(file_str),'*.jpg')))
else:
    files = [file_str]

workspace_path = os.environ.get('AE_WORKSPACE_PATH')
if workspace_path == None:
    print 'Please define a workspace path:\n'
    print 'export AE_WORKSPACE_PATH=/path/to/workspace\n'
    exit(-1)
log_dir = utils.get_log_dir(workspace_path,experiment_name,experiment_group)
ckpt_dir = utils.get_checkpoint_dir(log_dir)

codebook, dataset, encoder, decoder = factory.build_codebook_from_name(experiment_name, experiment_group, True, True)


rootpath = os.getcwd()
dateTimeObj  = datetime.now()
timestampStr = dateTimeObj.strftime("%d-%b-%Y-%H_%M_%S")
print("------------------------------" + timestampStr)
new_folder = timestampStr
os.mkdir(new_folder)

with tf.Session() as sess:

    factory.restore_checkpoint(sess, tf.train.Saver(), ckpt_dir)

    for i, file in enumerate(files):
        filename_w_ext = os.path.basename(file)
        filename, file_extension = os.path.splitext(filename_w_ext)

        new_folder_sub = rootpath + '/' + new_folder + '/' + filename
        os.mkdir(new_folder_sub)                 

        im = cv2.imread(file)
        im = cv2.resize(im,(128,128))

        #R = codebook.nearest_rotation(sess, im)
        #print R
        
        estimationNum = 5
        Rs = codebook.nearest_rotation(sess, im, estimationNum)
        print Rs
        print(Rs.shape)

       
        for i in range(estimationNum):
            pred_view = dataset.render_rot( Rs[i],downSample = 1)
            path = new_folder_sub + '/result_' + str(i) + '.png'
            cv2.imwrite(path, cv2.resize(pred_view,(256,256)))


        #pred_view = dataset.render_rot( R,downSample = 1)
        
        #plt.imshow(cv2.resize(im/255.,(256,256)))
        #plt.show()

        #plt.imshow(cv2.resize(pred_view,(256,256)))
        #plt.show()
        ######## reconstruct ########
        if im.dtype == 'uint8':
            x = im/255.
        if x.ndim == 3:
            x = np.expand_dims(x, 0)
        reconstr = sess.run(decoder.x,feed_dict={encoder.x:x})
        print(reconstr.shape)
        p = reconstr[0]
        print(p.shape)

        cv2.imwrite(new_folder_sub + '/input.png',im)
        cv2.imwrite(new_folder_sub + '/resized.png',cv2.resize(im,(256,256)))
        cv2.imwrite(new_folder_sub + '/reconstruct.png',p*255)
