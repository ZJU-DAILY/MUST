from statistics import median
import torch
import model
import dataset
import sys
import time
import pymswl

k = model.k

def inf_progress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))

    filledLength = round(barLength * iteration / float(total))
    bar = '#' * filledLength + '-' * (barLength - filledLength)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix))
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def eva_recall(res, gt, k):
    count = 0
    for i in range(len(res)):
        res_intersection = list(set(res[i]) & set(gt[i]))
        count += min(len(res_intersection), k)
    return count / (k * len(res))


base_modal1, base_modal2, base_modal3, base_modal4, query_modal1, query_modal2, query_modal3, query_modal4, \
gt = dataset.DatasetLoader('celeba+')
# gt = [gt1[i] for i in range(10000)]

# 'IP_Float_AS_Int_Skip0'
dist_opt = pymswl.OptSet(pymswl.Metric.IP_Float_AS_Int_Skip0, 64, 1)
dist_opt.load_1float_2int_345float(base_modal1, query_modal1, base_modal2, query_modal2, base_modal3, query_modal3, base_modal4, query_modal4)
# modal1.load_base_float(base_modal1)
# modal1.load_query_float(query_modal1)
# modal1_dist = modal1.float_gen_dist(0, len(query_modal1) - 1)
# modal2 = pyssg.OptSet(pyssg.Metric.AS_Uint_Skip0, 1, 0)
# modal2.load_base_uint(base_modal2)
# modal2.load_query_uint(query_modal2)
# modal2_dist = modal2.uint_gen_dist(0, len(query_modal1) - 1)

# gt_modal1_dist = torch.FloatTensor(gt_modal1_dist)
# gt_modal2_dist = torch.FloatTensor(gt_modal2_dist)
# modal1_dist = torch.FloatTensor(modal1_dist)
# modal2_dist = torch.FloatTensor(modal2_dist)
gt_modal1_dist = [dist_opt.modal1_dist_by_id(i, gt[i]) for i in range(len(gt))]
gt_modal2_dist = [dist_opt.modal2_int_dist_by_id(i, gt[i]) for i in range(len(gt))]
gt_modal3_dist = [dist_opt.modal3_dist_by_id(i, gt[i]) for i in range(len(gt))]
gt_modal4_dist = [dist_opt.modal4_dist_by_id(i, gt[i]) for i in range(len(gt))]
gt_len = [len(gt[i]) for i in range(len(gt))]

aggre_func = model.AggregationFunctionModel()
criteria = model.AggregationFunctionLoss()
optimizer = model.torch.optim.Adam([aggre_func.omega_x1, aggre_func.omega_x2, aggre_func.omega_x3, aggre_func.omega_x4], lr=0.2, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
max_recall = 0
opt_x1 = aggre_func.omega_x1.item()
opt_x2 = aggre_func.omega_x2.item()
opt_x3 = aggre_func.omega_x3.item()
opt_x4 = aggre_func.omega_x4.item()
for epoch in range(40):
    print("Epoch: #", epoch)
    # phi_positive = aggre_func(gt_modal1_dist, gt_modal2_dist)
    # phi_negative = aggre_func(modal1_dist, modal2_dist).topk(len(gt[query_id]), largest=True, sorted=True)
    ST1 = time.perf_counter()
    dist_opt.load_weight((aggre_func.omega_x1)[0].item(), (aggre_func.omega_x2)[0].item(), (aggre_func.omega_x3)[0].item(), (aggre_func.omega_x4)[0].item())
    # reca = [dist_opt.float_int_topk_id(i, i, len(gt[i]))[0] for i in range(len(gt))]
    reca = dist_opt.float_int_topk_id(0, len(gt) - 1, gt_len)
    # reca = dist_opt.float_float_topk_id(0, 2, [1,1])
    # print(reca[0])
    # print(dist_opt.modal1_dist_by_id(0, [14447]))
    ST2 = time.perf_counter()
    print("Get top-k time: ", (ST2 - ST1))
    reca_modal1_dist = [dist_opt.modal1_dist_by_id(i, reca[i]) for i in range(len(reca))]
    reca_modal2_dist = [dist_opt.modal2_int_dist_by_id(i, reca[i]) for i in range(len(reca))]
    reca_modal3_dist = [dist_opt.modal3_dist_by_id(i, reca[i]) for i in range(len(reca))]
    reca_modal4_dist = [dist_opt.modal4_dist_by_id(i, reca[i]) for i in range(len(reca))]
    # reca_modal1_dist = torch.FloatTensor(reca_modal1_dist)
    # reca_modal2_dist = torch.FloatTensor(reca_modal2_dist)
    phi_positive = []
    phi_negative = []
    res = []
    T11 = 0
    T22 = 0
    for query_id in range(len(gt)):
        T1 = time.perf_counter()
        cur_gt_modal1_dist = torch.FloatTensor(gt_modal1_dist[query_id])
        cur_gt_modal2_dist = torch.FloatTensor(gt_modal2_dist[query_id])
        cur_gt_modal3_dist = torch.FloatTensor(gt_modal3_dist[query_id])
        cur_gt_modal4_dist = torch.FloatTensor(gt_modal4_dist[query_id])

        # cur_gt_modal1_dist = torch.FloatTensor(dist_opt.modal1_dist_by_id(query_id, gt[query_id]))
        # cur_gt_modal2_dist = torch.FloatTensor(dist_opt.modal2_int_dist_by_id(query_id, gt[query_id]))
        phi_p = aggre_func(cur_gt_modal1_dist, cur_gt_modal2_dist, cur_gt_modal3_dist, cur_gt_modal4_dist)

        phi_positive.append(phi_p)

        T2 = time.perf_counter()
        T11 += T2 - T1

        # d_xx = distance.cosine_distance(tensor_base_data1_norm, qv1)
        d_xx = torch.FloatTensor(reca_modal1_dist[query_id])
        d_yy = torch.FloatTensor(reca_modal2_dist[query_id])
        d_zz = torch.FloatTensor(reca_modal3_dist[query_id])
        d_vv = torch.FloatTensor(reca_modal4_dist[query_id])
        # print(aggre_func(d_xx, d_yy))
        # d_xx = torch.FloatTensor(dist_opt.float_gen_dist(query_id, query_id))[0]
        # d_yy = torch.FloatTensor(dist_opt.int_gen_dist(query_id, query_id))[0]
        # for i in range(base_data1.shape[0]):
        #     d_yy[i] = distance.hamming_similarity(tensor_base_data2[i], qv2)
        # d_topk, id_topk = aggre_func(d_xx, d_yy).topk(len(gt[query_id]), largest=True, sorted=True)
        # id_topk = id_topk.tolist()
        # res.append(id_topk)
        # print(id_topk.tolist())
        phi_negative.append(aggre_func(d_xx, d_yy, d_zz, d_vv))
        # phi_negative.append(d_topk)

        T3 = time.perf_counter()
        T22 += T3 - T2

        inf_progress(query_id, len(gt) - 1, 'Progress', 'Complete', 1, 50)
    print("Compute groundtruth distance time: ", T11, 's')
    print("Compute Top-K distance time: ", T22, 's')

    # print(phi_positive)
    # print(phi_negative)

    optimizer.zero_grad()
    loss = criteria(phi_positive, phi_negative)
    loss.backward()
    # print("omega_x grad: ", aggre_func.omega_x.grad)
    # print("omega_y grad: ", aggre_func.omega_y.grad)
    print("Parameter, omega_x1: %lf, omega_x2: %lf, omega_x3: %lf, omega_x4: %lf" % (aggre_func.omega_x1,
                                                                      aggre_func.omega_x2, aggre_func.omega_x3, aggre_func.omega_x4))

    cur_recall = eva_recall(reca, gt, k)
    print("Recall@%d: %lf" % (k, cur_recall))
    if cur_recall > max_recall:
        max_recall = cur_recall
        opt_x1 = aggre_func.omega_x1.item()
        opt_x2 = aggre_func.omega_x2.item()
        opt_x3 = aggre_func.omega_x3.item()
        opt_x4 = aggre_func.omega_x4.item()
    optimizer.step()

print('Max Recall@%d: %lf' % (k, max_recall))
print('Optimal x1: ', opt_x1)
print('Optimal x2: ', opt_x2)
print('Optimal x3: ', opt_x3)
print('Optimal x4: ', opt_x4)