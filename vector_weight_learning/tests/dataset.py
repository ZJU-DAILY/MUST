from this import d
import numpy as np
import time
import struct
import json
import re
import os

def to_fvecs(filename, data):
    with open(filename, 'wb') as fp:
        for y in data:
            d = struct.pack('I', len(y))
            fp.write(d)
            for x in y:
                a = struct.pack('f', float(x))
                fp.write(a)

def to_ivecs(filename, data):
    # count = 0
    with open(filename, 'wb') as fp:
        for y in data:
            # count += 1
            # if count > 100:
            #     break
            d = struct.pack('I', len(y))
            fp.write(d)
            for x in y:
                a = struct.pack('I', int(x))
                fp.write(a)

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def ivecs_read_var(fname):
    data = []
    with open(fname, 'rb') as fp:
        tail = fp.seek(0, 2)
        fp.seek(0)
        while fp.tell() != tail:
            cur = fp.read(4)
            d, = struct.unpack('I', cur)
            tmp = []
            for i in range(d):
                cur = fp.read(4)
                id, = struct.unpack('I', cur)
                tmp.append(id)
            data.append(tmp)
    return data

def fvecs_read_var(fname):
    data = []
    with open(fname, 'rb') as fp:
        tail = fp.seek(0, 2)
        fp.seek(0)
        while fp.tell() != tail:
            cur = fp.read(4)
            d, = struct.unpack('I', cur)
            tmp = []
            for i in range(d):
                cur = fp.read(4)
                id, = struct.unpack('f', cur)
                tmp.append(id)
            data.append(tmp)
    return data

def fbin_read(fname):
    data = []
    with open(fname, 'rb') as fp:
        cur = fp.read(4)
        n, = struct.unpack('I', cur)
        cur = fp.read(4)
        d, = struct.unpack('I', cur)
        for i in range(n):
            tmp = []
            for i in range(d):
                cur = fp.read(4)
                id, = struct.unpack('f', cur)
                tmp.append(id)
            data.append(tmp)
    return data

def fvecs_read(filename):
    return ivecs_read(filename).view('float32')

def txt_read(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    meta_info = re.split('[ \n]', lines[0])
    lines.pop(0)
    print(meta_info)
    data = []
    for line in lines:
        tmp = re.split('[ \n]', line)
        tmp.pop()
        tmp = list(map(float, tmp))
        data.append(tmp)

    return data

def to_txt(filename, data):
    f = open(filename, 'w')
    num = len(data)
    dim = len(data[0])
    f.write(str(num) + ' ' + str(dim) + '\n')
    for i in range(num):
        for j in range(dim):
            if j != (dim - 1):
                f.write(str(data[i][j]) + ' ')
            else:
                f.write(str(data[i][j]) + '\n')

def load_vecs(filename):
    _, ext = os.path.splitext(filename)
    if ext == ".fvecs":
        data = fvecs_read_var(filename)
    elif ext == ".ivecs":
        data = ivecs_read_var(filename)
    elif ext == ".fbin":
        data = fbin_read(filename)
    else:
        raise TypeError("Unknown file type: {}".format(ext))

    return data

def DatasetLoader(dataset):
    dataset_root = "../../GraphANNS/doc/dataset/"
    dataset_suffix = "resnet_encode"
    is_composition = 0
    type = "/train/"
    if dataset == 'siftsmall':
        base_modal1 = fvecs_read(dataset_root + dataset + "/" + dataset + "_modal1_base.fvecs")
        base_modal2 = ivecs_read(dataset_root + dataset + "/" + dataset + "_modal2_base.ivecs")
        query_modal1 = fvecs_read(dataset_root + dataset + "/" + dataset + "_modal1_query.fvecs")
        query_modal2 = ivecs_read(dataset_root + dataset + "/" + dataset + "_modal2_query.ivecs")
        gt = ivecs_read_var(dataset_root + dataset + "/" + dataset + "_gt.ivecs")
    elif dataset == 'mitstates':
        base_modal1 = fvecs_read(dataset_root + dataset + type + dataset_suffix + "/" + dataset + "_modal1_base.fvecs")
        base_modal2 = fvecs_read(dataset_root + dataset + type + dataset_suffix + "/" + dataset +"_modal2_base.fvecs")
        query_modal2 = fvecs_read(dataset_root + dataset + type + dataset_suffix + "/" + dataset +"_modal2_query.fvecs")
        gt = ivecs_read_var(dataset_root + dataset + type + dataset +"_gt.ivecs")
        if is_composition:
            query_modal1 = fvecs_read(dataset_root + dataset + type + dataset_suffix + "/" + dataset +"_modal1_2_query.fvecs")
        else:
            query_modal1 = fvecs_read(dataset_root + dataset + type + dataset_suffix + "/" + dataset +"_modal1_query.fvecs")
    elif dataset == 'msong1m':
        base_modal1 = fvecs_read("/home/xxxxx/ANNS/dataset/sample/dataset/msong/msong_sample_base.fvecs")
        query_modal1 = fvecs_read("/home/xxxxx/ANNS/dataset/sample/dataset/msong/msong_sample_query.fvecs")
        base_modal2 = ivecs_read("/home/xxxxx/ANNS/xxx/GraphANNS/doc/dataset/siftsmall/siftsmall_modal2_base.ivecs")
        query_modal2= ivecs_read("/home/xxxxx/ANNS/xxx/GraphANNS/doc/dataset/siftsmall/siftsmall_modal2_query.ivecs")
        gt = ivecs_read_var("/home/xxxxx/ANNS/xxx/GraphANNS/doc/dataset/msong1m/msong1m_sample_gt.ivecs")
    elif dataset == 'uqv1m':
        base_modal1 = fvecs_read("/home/xxxxx/ANNS/dataset/sample/dataset/uqv/uqv_sample_base.fvecs")
        query_modal1 = fvecs_read("/home/xxxxx/ANNS/dataset/sample/dataset/uqv/uqv_sample_query.fvecs")
        base_modal2 = ivecs_read("/home/xxxxx/ANNS/xxx/GraphANNS/doc/dataset/siftsmall/siftsmall_modal2_base.ivecs")
        query_modal2= ivecs_read("/home/xxxxx/ANNS/xxx/GraphANNS/doc/dataset/siftsmall/siftsmall_modal2_query.ivecs")
        gt = ivecs_read_var("/home/xxxxx/ANNS/xxx/GraphANNS/doc/dataset/uqv1m/uqv1m_sample_gt.ivecs")
    elif dataset == 'deep':
        base_modal1 = fvecs_read("/home/xxxxx/ANNS/xxx/GraphANNS/doc/dataset/deep/deep_modal1_sample_base.fvecs")
        query_modal1 = fvecs_read("/home/xxxxx/ANNS/xxx/GraphANNS/doc/dataset/deep/deep_modal1_sample_query.fvecs")
        base_modal2 = ivecs_read("/home/xxxxx/ANNS/xxx/GraphANNS/doc/dataset/siftsmall/siftsmall_modal2_base.ivecs")
        query_modal2= ivecs_read("/home/xxxxx/ANNS/xxx/GraphANNS/doc/dataset/siftsmall/siftsmall_modal2_query.ivecs")
        gt = ivecs_read_var("/home/xxxxx/ANNS/xxx/GraphANNS/doc/dataset/deep/deep_sample_gt.ivecs")
    elif dataset == 'celeba+':
        base_modal1 = fvecs_read("/home/xxxxx/ANNS/xxx/GraphANNS/doc/dataset/celeba+/test/celeba_modal1_base.fvecs")
        query_modal1 = fvecs_read("/home/xxxxx/ANNS/xxx/GraphANNS/doc/dataset/celeba+/test/celeba_modal1_query.fvecs")
        base_modal2 = ivecs_read("/home/xxxxx/ANNS/xxx/GraphANNS/doc/dataset/celeba+/test/celeba_modal2_base.ivecs")
        query_modal2 = ivecs_read("/home/xxxxx/ANNS/xxx/GraphANNS/doc/dataset/celeba+/test/celeba_modal2_query.ivecs")
        base_modal3 = fvecs_read("/home/xxxxx/ANNS/xxx/GraphANNS/doc/dataset/celeba+/test/celeba_modal3_base.fvecs")
        query_modal3 = fvecs_read("/home/xxxxx/ANNS/xxx/GraphANNS/doc/dataset/celeba+/test/celeba_modal3_query.fvecs")
        base_modal4 = fvecs_read("/home/xxxxx/ANNS/xxx/GraphANNS/doc/dataset/celeba+/test/celeba_modal4_base.fvecs")
        query_modal4 = fvecs_read("/home/xxxxx/ANNS/xxx/GraphANNS/doc/dataset/celeba+/test/celeba_modal4_query.fvecs")
        gt = ivecs_read_var("/home/xxxxx/ANNS/xxx/GraphANNS/doc/dataset/celeba+/test/celeba_gt.ivecs")
    else: # supported options: "celeba", "shopping100k"
        base_modal1 = fvecs_read(dataset_root + dataset +"/train/" + dataset_suffix + "/" + dataset + "_modal1_base.fvecs")
        base_modal2 = ivecs_read(dataset_root + dataset +"/train/" + dataset_suffix + "/" + dataset +"_modal2_base.ivecs")
        query_modal2 = ivecs_read(dataset_root + dataset +"/train/" + dataset_suffix + "/" + dataset +"_modal2_query.ivecs")
        gt = ivecs_read_var(dataset_root + dataset +"/train/" + dataset_suffix + "/" + dataset +"_gt.ivecs")
        if is_composition:
            query_modal1 = fvecs_read(dataset_root + dataset +"/train/" + dataset_suffix + "/" + dataset +"_modal1_2_query.fvecs")
        else:
            query_modal1 = fvecs_read(dataset_root + dataset +"/train/" + dataset_suffix + "/" + dataset +"_modal1_query.fvecs")

    return base_modal1, base_modal2, base_modal3, base_modal4, query_modal1, query_modal2, query_modal3, query_modal4, gt