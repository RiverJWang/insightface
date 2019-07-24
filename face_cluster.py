import mxnet as mx
import cv2
import numpy as np
import mxnet.ndarray as nd
from sklearn import preprocessing
import os
from sklearn.cluster import DBSCAN


ctx = mx.gpu(0)
sym, arg_params, aux_params = mx.model.load_checkpoint('/home/shtf2/starter/INSIGHTFACE_ROOT/insightface/models/model-r100-ii/model', 0)
# arg_params, aux_params = ch_dev(arg_params, aux_params, ctx)
all_layers = sym.get_internals()
sym = all_layers['fc1_output']
model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
# model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
model.bind(data_shapes=[('data', (1, 3, 112, 112))])
model.set_params(arg_params, aux_params)

def caffe_preprocess_image(image_path):
    base_image = cv2.imread(image_path)
    base_image = cv2.resize(base_image,(96,112))
    # cv2.imshow("a",base_image)
    # cv2.waitKey(0)
    # base_image = cv2.cvtColor(base_image,cv2.COLOR_BGR2RGB)
    base_image = np.asarray(base_image,np.float32)
    base_image = (base_image-127.5)*0.0078125
    return base_image

def mxnet_infer(image_path):
    base_image = cv2.imread(image_path)
    image = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (112, 112))
    image = np.asarray(image, np.float32)
    image = np.transpose(image, (2, 0, 1))
    image = image[np.newaxis,...]
    jerry_data = mx.nd.array(image)
    jerry_data = mx.io.DataBatch(data=(jerry_data,), label=(nd.array([1]),))
    model.forward(jerry_data, is_train=False)
    net_out = model.get_outputs()
    np_out = net_out[0].asnumpy()
    # print(np_out)
    out = preprocessing.normalize(np_out)
    # print(out[0][0:10])
    return out.tolist()[0]

def extract_all_feature():
    base_dir = 'images'
    features = []
    for image in os.listdir(base_dir):
        features.append(mxnet_infer(os.path.join(base_dir,image)))
    return features

def clustering():
    features = extract_all_feature()
    # cluster the embeddings
    print("[INFO] clustering...")
    clt = DBSCAN(metric="euclidean", n_jobs=4)
    clt.fit(features)
    labelIDs = np.unique(clt.labels_)
    numUniqueFaces = len(np.where(labelIDs > -1)[0])
    print("[INFO] # unique faces: {}".format(numUniqueFaces))
    print 'ok'

def compare_two_face():
    feature_one = mxnet_infer('datasets/52.jpg')
    feature_two = mxnet_infer('datasets/394.jpg')
    dist = np.sum(np.square(np.asarray(feature_one)-np.asarray(feature_two)), 0)
    print dist**0.5
compare_two_face()

def extract_child_feature_and_save():
    base_path = '/e/company_file/bbtree/faces_width40'
    dirs = os.listdir(base_path)
    for dir in dirs:
        print('go image dir:', dir)
        item_dir = os.path.join(base_path, dir)
        features = []
        image_count = len(os.listdir(item_dir))
        for i in range(1,image_count+1):
            image_path = item_dir+'/'+str(i)+'.jpg'
            features.append(mxnet_infer(image_path))
        features = np.asarray(features)
        np.save(dir,features)


# extract_child_feature_and_save()
# features = np.load('features.npz.npy')
# print 'ok'




































