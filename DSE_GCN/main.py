# -*- coding: utf-8 -*-

import dgl
import time
import os
import torch as th
import numpy as np
import pandas as pd
from utils import *
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score,precision_score,recall_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from model import GRDTI


def get_predicted_sideeffects_for_drug(drug_name, drug_sideeffect_reconstruct, drug_name_to_index,
                                       index_to_sideeffect_name, threshold=0.5):
    # 获取药物索引
    drug_index = drug_name_to_index.get(drug_name)
    if drug_index is None:
        raise ValueError(f"Drug {drug_name} not found in dataset.")
    # 确保 drug_index 是整数类型
    drug_index = int(drug_index)
    # 提取该药物的预测不良反应关联分值
    drug_predictions = drug_sideeffect_reconstruct[drug_index]

    # 筛选出超过阈值的预测不良反应及对应概率
    predicted_sideeffects = {}
    for idx, prediction in enumerate(drug_predictions):
        if prediction > threshold:
            sideeffect_name = index_to_sideeffect_name.get(idx)
            if sideeffect_name:
                # 将概率转换为百分比形式
                probability_percentage = prediction.item() * 100
                predicted_sideeffects[sideeffect_name] = probability_percentage

    return predicted_sideeffects

def load_drug_sideeffect():
    file_path = './data/takeToDict_data.xlsx'
    df = pd.read_excel(file_path)

    # 处理药物名称字典
    drug_dict = {}
    drug_index = 0
    for drug_name in df['drugName']:
        if pd.isnull(drug_name) or drug_index >= 680:
            break
        drug_dict[drug_name] = drug_index
        drug_index += 1

    # 处理不良反应字典
    side_effect_dict = {}
    side_effect_index = 0
    for side_effect_id, group in df.groupby('sideeffectId'):
        if side_effect_index >= 2600:
            break
        side_effect_names = group['sideeffectName'].tolist()
        if len(side_effect_names) > 0:
            side_effect_dict[side_effect_index] = side_effect_names[0]
        else:
            print("Empty side - effect name list found at index", side_effect_index)
        side_effect_index += 1

    return drug_dict, side_effect_dict

def loda_data():
    # network_path = './newdata/'
    network_path = './olddata/'

    drug_drug = np.loadtxt(network_path + 'mat_drug_drug.txt')

    drug_disease = np.loadtxt(network_path + 'mat_drug_disease.txt')
    drug_sideeffect = np.loadtxt(network_path + 'mat_drug_se.txt')

    protein_protein = np.loadtxt(network_path + 'mat_protein_protein.txt')
    protein_disease = np.loadtxt(network_path + 'mat_protein_disease.txt')

    drug_protein = np.loadtxt(network_path + 'mat_drug_protein.txt')

    print("Load data finished.")

    return drug_drug, drug_disease, drug_protein, protein_protein, protein_disease, drug_sideeffect

def ConstructGraph(drug_drug, drug_disease, drug_protein, protein_protein,
                   protein_disease, drug_sideeffect, num_sideeffect):
    num_drug = len(drug_drug)
    num_protein = len(protein_protein)
    num_disease = len(drug_disease.T)
    # num_sideeffect = len(drug_sideeffect.T)

    list_drug = []
    for i in range(num_drug):
        list_drug.append((i, i))

    list_protein = []
    for i in range(num_protein):
        list_protein.append((i, i))

    list_disease = []
    for i in range(num_disease):
        list_disease.append((i, i))

    list_sideeffect = []
    for i in range(num_sideeffect):
        list_sideeffect.append((i, i))

    list_DDI = []
    for row in range(num_drug):
        for col in range(num_drug):
            if drug_drug[row, col] > 0:
                list_DDI.append((row, col))

    list_drug_disease = []
    list_disease_drug = []
    for row in range(num_drug):
        for col in range(num_disease):
            if drug_disease[row, col] > 0:
                list_drug_disease.append((row, col))
                list_disease_drug.append((col, row))

    list_drug_protein = []
    list_protein_drug = []
    for row in range(num_drug):
        for col in range(num_protein):
            if drug_protein[row, col] > 0:
                list_drug_protein.append((row, col))
                list_protein_drug.append((col, row))

    list_PPI = []
    for row in range(num_protein):
        for col in range(num_protein):
            if protein_protein[row, col] > 0:
                list_PPI.append((row, col))

    list_protein_disease = []
    list_disease_protein = []
    for row in range(num_protein):
        for col in range(num_disease):
            if protein_disease[row, col] > 0:
                list_protein_disease.append((row, col))
                list_disease_protein.append((col, row))


    list_drug_sideeffect = []
    list_sideeffect_drug = []
    for row in range(num_drug):
        for col in range(num_sideeffect):
            if drug_sideeffect[row, col] > 0:
                list_drug_sideeffect.append((row, col))
                list_sideeffect_drug.append((col, row))


    g_HIN = dgl.heterograph({('disease', 'disease_disease virtual', 'disease'): list_disease,
                             ('drug', 'drug_drug virtual', 'drug'): list_drug,
                             ('protein', 'protein_protein virtual', 'protein'): list_protein,
                             ('sideeffect', 'sideeffect_sideeffect virtual', 'sideeffect'): list_sideeffect,
                             ('drug', 'drug_drug interaction', 'drug'): list_DDI, \
                             ('protein', 'protein_protein interaction', 'protein'): list_PPI, \
                             ('drug', 'drug_protein interaction', 'protein'): list_drug_protein, \
                             ('protein', 'protein_drug interaction', 'drug'): list_protein_drug, \
                             ('drug', 'drug_sideeffect association', 'sideeffect'): list_drug_sideeffect, \
                             ('sideeffect', 'sideeffect_drug association', 'drug'): list_sideeffect_drug, \
                             ('drug', 'drug_disease association', 'disease'): list_drug_disease, \
                             ('disease', 'disease_drug association', 'drug'): list_disease_drug, \
                             ('protein', 'protein_disease association', 'disease'): list_protein_disease, \
                             ('disease', 'disease_protein association', 'protein'): list_disease_protein
                             })

    g = g_HIN.edge_type_subgraph(['drug_drug interaction', 'protein_protein interaction',
                                  'drug_protein interaction', 'protein_drug interaction',
                                  'drug_sideeffect association', 'sideeffect_drug association',
                                  'drug_disease association', 'disease_drug association',
                                  'protein_disease association', 'disease_protein association'
                                  ])

    return g


def TrainAndEvaluate(DSEtrain, DSEvalid, DSEtest, num_sideeffect, args, drug_drug, drug_disease,
                     drug_protein, protein_protein, protein_disease, all_records):
    drug_name_to_index, index_to_sideeffect_name = load_drug_sideeffect()
    device = th.device(args.device)

    # Numbers of different nodes
    num_disease = len(drug_disease.T)
    num_drug = len(drug_drug)
    num_protein = len(protein_protein)


    drug_sideeffect = th.zeros((num_drug, num_sideeffect))
    mask = th.zeros((num_drug, num_sideeffect)).to(device)
    for ele in DSEtrain:
        drug_sideeffect[ele[0], ele[1]] = ele[2]
        mask[ele[0], ele[1]] = 1

    best_valid_aupr = 0.
    # best_valid_auc = 0
    test_aupr = 0.
    test_auc = 0.
    patience = 0.

    pos = np.count_nonzero(DSEtest[:, 2])
    neg = np.size(DSEtest[:, 2]) - pos
    xy_roc_sampling = []
    xy_pr_sampling = []

    # 根据药物、疾病、蛋白质、不良反应的关系得到一个图结构
    g = ConstructGraph(drug_drug, drug_disease, drug_protein, protein_protein,
                       protein_disease, drug_sideeffect, num_sideeffect)

    drug_drug = th.tensor(drug_drug).to(device)
    # drug_chemical = th.tensor(drug_chemical).to(device)
    drug_disease = th.tensor(drug_disease).to(device)
    drug_protein = th.tensor(drug_protein).to(device)
    protein_protein = th.tensor(protein_protein).to(device)
    # protein_sequence = th.tensor(protein_sequence).to(device)
    protein_disease = th.tensor(protein_disease).to(device)
    drug_sideeffect = drug_sideeffect.to(device)

    model = GRDTI(g, num_disease, num_drug, num_protein, num_sideeffect, args)
    model.to(device)

    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    precision_values, recall_values = [], []  # 用于存储每个epoch的查准率和查全率

    f1_score_values = []  # 新增用于存储每个epoch的F1分数

    mcc_values = []  # 新增用于存储每个epoch的MCC值

    for i in range(args.epochs):

        model.train()
        tloss, dseloss, l2loss, dse_re, DSE_p = model(drug_drug, drug_disease, drug_protein,
                                                     protein_protein, protein_disease,
                                                     drug_sideeffect, mask)

        results = dse_re.detach().cpu()
        optimizer.zero_grad()
        loss = tloss
        loss.backward()
        th.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        model.eval()

        with th.no_grad():
            print("Epoch", i, ":", "Total_loss & DSEloss & L2_loss:", loss.cpu().data.numpy(), ",", dseloss.item(),
                  ",", l2loss.item())

            pred_list = []
            ground_truth = []

            for ele in DSEvalid:
                pred_list.append(results[ele[0], ele[1]])
                ground_truth.append(ele[2])

            valid_auc = roc_auc_score(ground_truth, pred_list)
            valid_aupr = average_precision_score(ground_truth, pred_list)

            # 将概率值转换为二进制标签
            pred_labels = np.where(np.array(pred_list) >= 0.5, 1, 0)

            # 计算查准率和查全率,将zero_division设置为0或1，可以避免NAN
            precision = precision_score(ground_truth, pred_labels, zero_division=0)
            recall = recall_score(ground_truth, pred_labels, zero_division=0)

            precision_values.append(precision)  # 存储每个 epoch 的查准率
            recall_values.append(recall)  # 存储每个 epoch 的查全率

            # 计算F1分数
            if precision + recall > 0:
                f1_score = 2 * (precision * recall / (precision + recall))
            else:
                f1_score = 0
            f1_score_values.append(f1_score)  # 存储每个 epoch 的F1分数

            # 计算TP、TN、FP、FN
            tp = np.sum((np.array(pred_labels) == 1) & (np.array(ground_truth) == 1))
            tn = np.sum((np.array(pred_labels) == 0) & (np.array(ground_truth) == 0))
            fp = np.sum((np.array(pred_labels) == 1) & (np.array(ground_truth) == 0))
            fn = np.sum((np.array(pred_labels) == 0) & (np.array(ground_truth) == 1))

            # 计算MCC
            if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0:
                mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            else:
                mcc = 0
            mcc_values.append(mcc)  # 存储每个 epoch 的MCC值

            if valid_aupr >= best_valid_aupr:
                best_valid_aupr = valid_aupr
                # best_valid_auc = valid_auc
                best_DSE_potential = DSE_p
                patience = 0

                # Calculating AUC & AUPR (pos:neg=1:10)
                db = []
                xy_roc = []
                xy_pr = []
                for ele in DSEtest:
                    db.append([results[ele[0], ele[1]], ele[2]])

                db = sorted(db, key=lambda x: x[0], reverse=True)

                tp, fp = 0., 0.
                for i_db in range(len(db)):
                    if db[i_db][0]:
                        if db[i_db][1]:
                            tp = tp + 1
                        else:
                            fp = fp + 1
                        xy_roc.append([fp / neg, tp / pos])
                        xy_pr.append([tp / pos, tp / (tp + fp)])

                test_auc = 0.
                prev_x = 0.
                for x, y in xy_roc:
                    if x != prev_x:
                        test_auc += (x - prev_x) * y
                        prev_x = x

                test_aupr = 0.
                prev_x = 0.
                for x, y in xy_pr:
                    if x != prev_x:
                        test_aupr += (x - prev_x) * y
                        prev_x = x

                # sampling (pos:neg=1:10) for averaging and plotting
                xy_roc_sampling = []
                xy_pr_sampling = []
                for i_xy in range(len(xy_roc)):
                    if i_xy % 10 == 0:
                        xy_roc_sampling.append(xy_roc[i_xy])
                        xy_pr_sampling.append(xy_pr[i_xy])

            print('Valid auc & aupr:', valid_auc, valid_aupr, ";  ", 'Test auc & aupr:', test_auc, test_aupr, ";  ",
                  "Precision & Recall:", precision, recall, "F1-score:", f1_score, "MCC:", mcc)
            all_drug_predicted_sideeffects = {}
            for drug_name in drug_name_to_index:
                predicted_sideeffects = get_predicted_sideeffects_for_drug(
                    drug_name=drug_name,
                    drug_sideeffect_reconstruct=results,
                    drug_name_to_index=drug_name_to_index,
                    index_to_sideeffect_name=index_to_sideeffect_name,
                    threshold=0.5
                )
                all_drug_predicted_sideeffects[drug_name] = predicted_sideeffects

            # 将每个 epoch 的指标存入字典
            record = {
                "Epoch": i+1,
                "Total Loss": tloss.item(),
                "DSE Loss": dseloss.item(),
                "L2 Loss": l2loss.item(),
                "Valid AUC": valid_auc,
                "Valid AUPR": valid_aupr,
                "Test AUC": test_auc,
                "Test AUPR": test_aupr,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1_score,  # 新增F1分数记录到字典中
                "MCC": mcc,  # 新增MCC记录到字典中
                "predicted_sideeffects": all_drug_predicted_sideeffects
            }
            all_records.append(record)

    return test_auc, test_aupr, xy_roc_sampling, xy_pr_sampling, best_DSE_potential, np.mean(precision_values), np.mean(recall_values), np.mean(f1_score_values), np.mean(mcc_values),all_drug_predicted_sideeffects, all_records


def main(args):
    # 初始化用于存储训练和验证记录的列表
    all_records = []
    # 初始化用于存储所有十折交叉验证结果的列表
    all_k_fold_results = []
    # 初始化用于存储每一折每一轮详细训练记录的嵌套列表
    all_folds_all_records = [[] for _ in range(args.nFold)]
    # 初始化用于存储六个参数值的列表
    auc_list = []
    aupr_list = []
    precision_list = []
    recall_list = []
    f1_score_list = []
    mcc_list = []
    # 加载药物与药物，药物与疾病，药物与蛋白，蛋白与蛋白，蛋白与疾病，药物与不良反应
    drug_d, drug_di, drug_pro, protein_p, protein_di, drug_side = loda_data()

    # 药物与不良反应的数量
    num_sideeffect = len(drug_side.T)

    # sampling
    # drug_side=1.0的点集合
    whole_positive_index = []
    # drug_side=0.0的点集合
    whole_negative_index = []
    for i in range(np.shape(drug_side)[0]):
        for j in range(np.shape(drug_side)[1]):
            if int(drug_side[i][j]) == 1:
                whole_positive_index.append([i, j])
            elif int(drug_side[i][j]) == 0:
                whole_negative_index.append([i, j])

    # pos:neg=1:10
    negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                             size=10 * len(whole_positive_index), replace=False)

    # 把whole_positive_index & whole_negative_index两个点集合打到一个三列的numpy中
    data_set = np.zeros((len(negative_sample_index) + len(whole_positive_index), 3), dtype=int)
    count = 0
    for i in whole_positive_index:
        data_set[count][0] = i[0]
        data_set[count][1] = i[1]
        data_set[count][2] = 1
        count += 1
    for i in negative_sample_index:
        data_set[count][0] = whole_negative_index[i][0]
        data_set[count][1] = whole_negative_index[i][1]
        data_set[count][2] = 0
        count += 1

    # 将数据划分为训练集（80%）和测试集（10%）和验证集（10%）
    train, DSEtest = train_test_split(data_set, test_size=0.1, random_state=None)
    DSEtrain, DSEvalid = train_test_split(train, test_size=0.1, random_state=None)
    for fold_idx in range(args.nFold):  # 进行十折交叉验证
        auc, aupr, xy_roc, xy_pr, DSE_potential, precision_mean, recall_mean, f1_score_mean, mcc_mean, predicted_sideeffects, all_records = TrainAndEvaluate(
            DSEtrain, DSEvalid, DSEtest, num_sideeffect, args,
            drug_d,
            drug_di, drug_pro, protein_p, protein_di, all_records)
        # 将每次返回的参数值添加到对应的列表中
        auc_list.append(auc)
        aupr_list.append(aupr)
        precision_list.append(precision_mean)
        recall_list.append(recall_mean)
        f1_score_list.append(f1_score_mean)
        mcc_list.append(mcc_mean)
        # 将当前折的训练记录信息添加到对应的列表中
        all_folds_all_records[fold_idx].extend(all_records)
    # 计算所有十折交叉验证结果的均值和标准差
    auc_mean = np.mean(auc_list)
    auc_std = np.std(auc_list)
    aupr_mean = np.mean(aupr_list)
    aupr_std = np.std(aupr_list)
    precision_mean = np.mean(precision_list)
    precision_std = np.std(precision_list)
    recall_mean = np.mean(recall_list)
    recall_std = np.std(recall_list)
    f1_score_mean = np.mean(f1_score_list)
    f1_score_std = np.std(f1_score_list)
    mcc_mean = np.mean(mcc_list)
    mcc_std = np.std(mcc_list)
    print("--------------------------------------------------------------")
    print('AUC Mean:', auc_mean, 'AUC Std:', auc_std)
    print('AUPR Mean:', aupr_mean, 'AUPR Std:', aupr_std)
    print('Precision Mean:', precision_mean, 'Precision Std:', precision_std)
    print('Recall Mean:', recall_mean, 'Recall Std:', recall_std)
    print('F1 Score Mean:', f1_score_mean, 'F1 Score Std:', f1_score_std)
    print('MCC Mean:', mcc_mean, 'MCC Std:', mcc_std)
    # print('predicted_sideeffects:', predicted_sideeffects)
    print("--------------------------------------------------------------")

    auc_aupr_precision_recall_f1_mcc = np.column_stack((auc_mean, auc_std, aupr_mean, aupr_std, precision_mean,
                                                        precision_std, recall_mean, recall_std, f1_score_mean,
                                                        f1_score_std, mcc_mean, mcc_std))

    np.savetxt('olddata_result/auc_aupr_precision_recall_f1_mcc.csv', auc_aupr_precision_recall_f1_mcc,
               fmt='%-.7f', delimiter=',',
               header='auc_mean,auc_std,aupr_mean,aupr_std,precision_mean,precision_std,recall_mean,recall_std,f1_score_mean,f1_score_std,mcc_mean,mcc_std',
               comments='')

    # 保存每一折的数据到单独的 Excel 文件
    for fold_idx, fold_records in enumerate(all_folds_all_records):
        all_epochs_data = []
        for epoch_record in fold_records:
            epoch = epoch_record["Epoch"]
            total_loss = epoch_record["Total Loss"]
            dse_loss = epoch_record["DSE Loss"]
            l2_loss = epoch_record["L2 Loss"]
            valid_auc = epoch_record["Valid AUC"]
            valid_aupr = epoch_record["Valid AUPR"]
            test_auc = epoch_record["Test AUC"]
            test_aupr = epoch_record["Test AUPR"]
            precision = epoch_record["Precision"]
            recall = epoch_record["Recall"]
            f1_score = epoch_record["F1 Score"]
            mcc = epoch_record["MCC"]
            predicted_sideeffects = epoch_record["predicted_sideeffects"]

            # 格式化 Sideeffect_Predict 列内容
            sideeffect_predict_str = ""
            for drug, side_effects in predicted_sideeffects.items():
                sideeffect_list = []
                for side_effect, probability in side_effects.items():
                    sideeffect_list.append(f"{side_effect}:{probability}%")
                sideeffect_predict_str += f"{drug}:[{','.join(sideeffect_list)}],"
            sideeffect_predict_str = sideeffect_predict_str[:-1]  # 移除末尾多余的逗号

            all_epochs_data.append(
                [epoch, total_loss, dse_loss, l2_loss, valid_auc, valid_aupr, test_auc, test_aupr,
                 precision, recall, f1_score, mcc, sideeffect_predict_str])

        columns = ['Epoch', 'Total Loss', 'DSE Loss', 'L2 Loss', 'Valid AUC', 'Valid AUPR', 'Test AUC', 'Test AUPR',
                   'Precision', 'Recall', 'F1 Score', 'MCC', 'Sideeffect_Predict']
        df = pd.DataFrame(all_epochs_data, columns=columns)
        output_file = f"./olddata_result/fold{fold_idx}_training_metrics.xlsx"
        df.to_excel(output_file, index=False)
        print(f"第 {fold_idx} 折的训练记录已保存到 {output_file}")

if __name__ == "__main__":
    args = parse_args()
    print(args)

    start = time.time()
    main(args)
    end = time.time()
    total_time_minutes = (end - start) / 60
    print(f"Total time: {total_time_minutes:.0f} minutes")
