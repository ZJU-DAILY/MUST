import os
import time
import argparse
import numpy as np

import pyssg


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base1", type=str,
                        default="/home/xxxxx/ANNS/xxx/GraphANNS/doc/dataset/siftsmall/siftsmall_modal1_base.fvecs",
                        help="fvecs file for base1 vectors")
    parser.add_argument("--base2", type=str,
                        default="/home/xxxxx/ANNS/xxx/GraphANNS/doc/dataset/siftsmall/siftsmall_modal2_base.ivecs",
                        help="fvecs file for base2 vectors")
    parser.add_argument("--query1", type=str,
                        default="/home/xxxxx/ANNS/xxx/GraphANNS/doc/dataset/siftsmall/siftsmall_modal1_query.fvecs",
                        help="fvecs file for query1 vectors")
    parser.add_argument("--query2", type=str,
                        default="/home/xxxxx/ANNS/xxx/GraphANNS/doc/dataset/siftsmall/siftsmall_modal2_query.ivecs",
                        help="fvecs file for query2 vectors")
    parser.add_argument("--groundtruth", type=str,
                        default="/home/xxxxx/ANNS/xxx/GraphANNS/doc/dataset/siftsmall/siftsmall_gt.ivecs",
                        help="ivecs file for groundtruth")
    parser.add_argument("--graph", type=str,
                        default="./graphs/sift.ssg",
                        help="path SSG graph file")
    parser.add_argument("--k", type=int, default=100,
                        help="how many neighbors to query")
    parser.add_argument("--l", type=int, default=100,
                        help="search param L")
    parser.add_argument("--seed", type=int, default=161803398,
                        help="random seed")
    return parser.parse_args()


def load_vecs(filename):
    _, ext = os.path.splitext(filename)
    if ext == ".fvecs":
        dtype = np.float32
    elif ext == ".ivecs":
        dtype = np.int32
    else:
        raise TypeError("Unknown file type: {}".format(ext))

    data = np.fromfile(filename, dtype=dtype)
    dim = data[0].view(np.int32)
    data = data.reshape(-1, dim + 1).astype(dtype)
    return np.ascontiguousarray(data[:, 1:])


if __name__ == "__main__":
    args = setup_args()

    base1 = load_vecs(args.base1)
    nbases, dim = base1.shape

    base2 = load_vecs(args.base2)
    nbases, dim = base2.shape
    print(base2[0])

    query1 = load_vecs(args.query1)
    nq, _ = query1.shape

    query2 = load_vecs(args.query2)
    nq, _ = query2.shape
    print(query2[0])

    # modal1 = pyssg.OptSet(pyssg.Metric.IP_Float, 1, 0)
    modal2 = pyssg.OptSet(pyssg.Metric.AS_Uint, 1, 0)
    # modal1.load_base_float(query2)
    modal2.load_base_uint(base2)
    # modal1.load_query_float(query2)
    modal2.load_query_uint(query2)
    # res1 = modal1.float_gen_dist(0, 99)
    res2 = modal2.uint_gen_dist(0, 1)
    # print(type(res1))
    # print(res1[0][0])
    print(res2[0][0:10])

    #
    # start = time.time()
    # results = [index.search(q, args.k, args.l) for q in query]
    # elapsed = time.time() - start
    # qps = nq / elapsed
    #
    # results = np.asarray(results)
    # gt = load_vecs(args.groundtruth)[:nq]
    # assert gt.shape == results.shape
    #
    # cnt = 0
    # for ret, gt in zip(results, gt):
    #     cnt += len(np.intersect1d(ret, gt))
    # acc = cnt / nq / args.k * 100
    # print(f"{nq} queries in {elapsed:.4f}s, {qps:.4f}QPS, accuracy {acc:.4f}")
