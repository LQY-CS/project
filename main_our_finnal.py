import numpy as np
import torch
from utils import *
from train import *
from arguments_finnal import arg_parser
import os
import csv
import warnings as warnings
warnings.filterwarnings("ignore")
from torch_geometric.nn import GCNConv
import torch.nn as nn
from collections import defaultdict
from torch_geometric.data import Data
import yaml
from datetime import datetime

best_val_acc = 0.0
best_model_state = None
prototypes = None
mix_label = None
top_k_label = None
high_mask = None
mean_label = None
from torch_geometric.utils import degree



class encoder(torch.nn.Module):
    def __init__(self, in_feat, hidden, out_feat, dropout):
        super(encoder, self).__init__()
        self.conv1 = GCNConv(in_feat, hidden *2 )
        self.conv2 = GCNConv(hidden * 2 , hidden )
        self.cls = nn.Linear(hidden, out_feat)
        self.proj = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.p = dropout
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.p, training=self.training)
        x = self.conv2(x, edge_index)

        logits = self.cls(x)
        feat_out = self.proj(x)
        return feat_out, logits

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.cls.reset_parameters()
        for layer in self.proj:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


def construct_super_nodes(data):
    # 创建一个字典来存储每个类别对应的节点索引
    label_to_nodes = defaultdict(list)
    train_node_indices = torch.nonzero(data.train_mask, as_tuple=True)[0]
    for node_idx in train_node_indices:
        label = int(data.y[node_idx])
        label_to_nodes[label].append(node_idx)
    super_node_features = []
    super_node_edges = []
    super_node_labels = []
    num_nodes = data.num_nodes
    edge_index = data.edge_index
    super_node_indices = [] 
    for label, nodes in label_to_nodes.items():
        nodes_tensor = torch.tensor(nodes, dtype=torch.long)
        node_features = data.x[nodes_tensor]
        super_node_feature = node_features.mean(dim=0)
        super_node_features.append(super_node_feature)
        super_node_labels.append(label)
        super_node_idx = num_nodes
        num_nodes += 1
        super_node_indices.append(super_node_idx)
        for node_idx in nodes:
            super_node_edges.append([super_node_idx, node_idx])  # 超级节点 -> 原始节点
            super_node_edges.append([node_idx, super_node_idx])  # 原始节点 -> 超级节点

    super_node_edges = torch.tensor(super_node_edges, dtype=torch.long).t().contiguous()
    new_edge_index = torch.cat([edge_index, super_node_edges], dim=1)

    super_node_features = torch.stack(super_node_features)
    new_features = torch.cat([data.x, super_node_features], dim=0)
    super_node_labels = torch.tensor(super_node_labels, dtype=torch.long, device=data.y.device)
    new_labels = torch.cat([data.y, super_node_labels], dim=0)
    super_node_labels = torch.tensor(super_node_labels, dtype=torch.long, device=data.clean_y.device)
    new_clean_labels = torch.cat([data.clean_y, super_node_labels], dim=0)
    num_super_nodes = len(super_node_indices)
    new_train_mask = torch.cat([data.train_mask, torch.zeros(num_super_nodes, dtype=torch.bool)])
    new_val_mask = torch.cat([data.val_mask, torch.zeros(num_super_nodes, dtype=torch.bool)])
    new_test_mask = torch.cat([data.test_mask, torch.zeros(num_super_nodes, dtype=torch.bool)])
    sup_mask = torch.cat([torch.zeros(data.num_nodes, dtype=torch.bool), torch.ones(num_super_nodes, dtype=torch.bool)])
    new_data = Data(
        x=new_features,
        edge_index=new_edge_index,
        y=new_labels,
        clean_y = new_clean_labels,
        train_mask=new_train_mask,
        val_mask=new_val_mask,
        test_mask=new_test_mask,
        sup_mask=sup_mask,
        num_nodes= data.num_nodes + num_super_nodes,
        num_classes=data.num_classes,
        num_edges=new_edge_index.size(1),
    )
    new_data = new_data.to(data.x.device)
    return new_data


def custom_cross_entropy_loss(out, mixed_labels):

    loss = -(mixed_labels * F.log_softmax(out, dim=1)).sum(dim=1).mean()
    return loss


def neb_con(feat, src, dst, neg, temperature=0.5):
    """
    构造邻居对比损失函数，使用源节点和目标节点作为正样本，使用源节点和负样本节点作为负样本。

    参数：
    - feat: 节点的特征张量，形状为 [N, D]，N 是节点数量，D 是特征维度。
    - src: 正邻居关系的源节点索引，形状为 [E]，E 是正样本边的数量。
    - dst: 正邻居关系的目标节点索引，形状为 [E]。
    - neg: 负样本节点的索引，形状为 [E]。
    - temperature: 控制相似度的温度参数。

    返回：
    - loss: 计算得到的对比损失。
    """
    # 获取源节点、目标节点和负样本的特征
    src_feat = feat[src]  # [E, D]
    dst_feat = feat[dst]  # [E, D]
    neg_feat = feat[neg]  # [E, D]

    # 归一化特征，以便计算余弦相似度
    # src_feat = F.normalize(src_feat, p=2, dim=1)  # [E, D]
    # dst_feat = F.normalize(dst_feat, p=2, dim=1)  # [E, D]
    # neg_feat = F.normalize(neg_feat, p=2, dim=1)  # [E, D]

    
    src_norm = torch.norm(src_feat, p=2, dim=1, keepdim=True)
    dst_norm = torch.norm(dst_feat, p=2, dim=1, keepdim=True)
    neg_norm = torch.norm(neg_feat, p=2, dim=1, keepdim=True)
    
    # 计算正样本之间的相似度 (使用余弦相似度)
    # positive_sim = torch.sum(torch.mm(src_feat, dst_feat.T), dim=1, keepdim=True)  # [E, 1]

    # # 计算源节点与负样本之间的相似度
    # negative_sim_1 = torch.sum(torch.mm(src_feat, neg_feat.T), dim=1, keepdim=True)  # [E, 1]
    
    # negative_sim_2 = torch.sum(torch.mm(dst_feat, neg_feat.T), dim=1, keepdim=True)  # [E, 1]
    # 计算正样本之间的余弦相似度
    positive_sim = torch.sum(src_feat * dst_feat, dim=1, keepdim=True) / (src_norm * dst_norm)

    # 计算源节点与负样本之间的余弦相似度
    negative_sim_1 = torch.sum(src_feat * neg_feat, dim=1, keepdim=True) / (src_norm * neg_norm)
    negative_sim_2 = torch.sum(dst_feat * neg_feat, dim=1, keepdim=True) / (dst_norm * neg_norm)
    # 将正样本相似度与负样本相似度拼接
    # logits = torch.cat([positive_sim, negative_sim_1, negative_sim_2], dim=1)  # [E, 2]

    negative_sim_2 = torch.sum(dst_feat * neg_feat, dim=1, keepdim=True) / (dst_norm * neg_norm)
    # 将正样本相似度与负样本相似度拼接
    # logits = torch.cat([positive_sim, negative_sim_1], dim=1)  # [E, 2]
    logits = torch.cat([positive_sim, negative_sim_1, negative_sim_2], dim=1)  # [E, 2]
    # 应用温度缩放
    logits /= temperature

    # 创建标签，正样本的标签为0
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=feat.device)

    # 计算对比损失（交叉熵损失）
    loss = F.cross_entropy(logits, labels)

    return loss

def select_by_low_entropy_per_class(out_t_all, percentage=0.3):
    probabilities = F.softmax(out_t_all, dim=1)
    predicted_labels = torch.argmax(probabilities, dim=1)
    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-12), dim=1)
    num_classes = out_t_all.size(1)
    selected_mask = torch.zeros_like(entropy, dtype=torch.bool)
    for c in range(num_classes):
        class_indices = (predicted_labels == c).nonzero(as_tuple=True)[0]

        if len(class_indices) == 0:
            continue  
        class_entropy = entropy[class_indices]
        num_to_select = int(percentage * len(class_indices))
        num_to_select = max(num_to_select, 1)  # 至少选择一个样本，避免 num_to_select 为 0
        _, lowest_entropy_indices = torch.topk(-class_entropy, num_to_select)
        selected_class_indices = class_indices[lowest_entropy_indices]
        selected_mask[selected_class_indices] = True
    return selected_mask, predicted_labels

def create_mixed_labels_from_logits(logits, num_classes, top_k1, top_k2, train_mask):
    batch_size = logits.size(0)
    mixed_labels = torch.zeros((batch_size, num_classes), device=logits.device)
    _, topk_indices_train = torch.topk(logits[train_mask], top_k1, dim=1) if train_mask.sum() > 0 else (None, None)
    _, topk_indices_non_train = torch.topk(logits[~train_mask], top_k2, dim=1) if (~train_mask).sum() > 0 else (None, None)
    mixed_labels[train_mask] = mixed_labels[train_mask].scatter_(1, topk_indices_train, 1 / top_k1)
    mixed_labels[~train_mask] = mixed_labels[~train_mask].scatter_(1, topk_indices_non_train, 1 / top_k2)
    return mixed_labels


def compute_weighted_prototypes(node_features, logits, high_mask, num_classes):
    logits = F.softmax(logits, dim=1)
    high_confidence_features = node_features[high_mask]  # [N_high, D]
    high_confidence_logits = logits[high_mask]           # [N_high, C]
    predicted_labels = torch.argmax(high_confidence_logits, dim=1)  # [N_high]
    one_hot_labels = F.one_hot(predicted_labels, num_classes=num_classes)  # [N_high, C]
    weights = high_confidence_logits  # [N_high, C]
    weighted_features = high_confidence_features.unsqueeze(1) * weights.unsqueeze(2)  # [N_high, C, D]
    sum_weighted_features = torch.sum(weighted_features * one_hot_labels.unsqueeze(2), dim=0)  # [C, D]
    sum_weights = torch.sum(weights * one_hot_labels, dim=0).view(-1, 1)  # [C, 1]
    prototypes = sum_weighted_features / (sum_weights + 1e-10)  # [C, D]，添加小量以避免除零
    return prototypes

def update_pseudo_labels_with_prototypes(node_features, prototypes):
    node_features_norm = F.normalize(node_features, p=2, dim=1)
    prototypes_norm = F.normalize(prototypes, p=2, dim=1)
    similarities = torch.mm(node_features_norm, prototypes_norm.t())  # [N, C]
    return similarities



def train(model, data, odata, epoch, args, optimizer, alpha_M):
    global best_val_acc, best_model_state, prototypes, mix_label, top_k_label, high_mask, mean_label
    model.train()
    optimizer.zero_grad()
    out, logits = model(data)
    loss_ce_sup = F.nll_loss(F.log_softmax(logits[data.sup_mask], dim=-1), data.y[data.sup_mask])
    neg = torch.randint(0, odata.num_nodes, (odata.num_edges, ))
    con_loss = neb_con(out[~data.sup_mask], odata.edge_index[0], odata.edge_index[1], neg, args.temperature)
    if epoch < args.warm_up:
        
        logits_soft = F.softmax(logits, dim=1)
        topk_val, topk_indices = torch.topk(logits_soft, args.top_k)
        sparse_logits = torch.zeros_like(logits)
        batch_indices = torch.arange(logits.size(0)).unsqueeze(1)  # (batch_size, 1)
        sparse_logits[batch_indices, topk_indices] = topk_val
        coef_cosine = epoch / args.warm_up
        top_k_label +=  coef_cosine * sparse_logits
        loss = loss_ce_sup + con_loss * args.beta
    else:
        if mix_label == None:
            mix_label = create_mixed_labels_from_logits(top_k_label, data.num_classes, args.top_k, data.num_classes, data.train_mask)
            mix_label = mix_label.to(out.device)
        with torch.no_grad():
            if high_mask == None:
                high_mask = select_by_low_entropy_per_class(logits, args.sample_high_rate)[0]
            P = compute_weighted_prototypes(out, logits, high_mask, data.num_classes)
            P = P.to(out.device)
            alpha_P = args.alpha_P 
            prototypes = alpha_P * prototypes + (1 - alpha_P) * P
            similarities = update_pseudo_labels_with_prototypes(out, prototypes)
            combined_logits = alpha_M * logits + (1 - alpha_M) * similarities
            mix_label = mix_label * F.softmax(combined_logits, dim=1) 
            mix_label = mix_label / mix_label.sum(dim=1, keepdim=True)
        
        loss_ce = custom_cross_entropy_loss(logits, mix_label)
        loss = con_loss * args.beta  + loss_ce * args.gamma + loss_ce_sup  
    
    loss.backward()
    optimizer.step()
    model.eval()
    high_mask = None
    with torch.no_grad():
        _, val_logits = model(data)
        val_pred = val_logits[data.val_mask].max(1)[1]
        val_acc = val_pred.eq(data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
        test_pred = val_logits[data.test_mask].max(1)[1]
        test_acc = test_pred.eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()  # 保存最佳模型参数
    return val_acc, test_acc, out


def test(model, data):
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()
    _, logits = model(data)
    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.clean_y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


def load_data_our(path):
    """
    从指定路径加载PyG数据对象。
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def get_clean_data(args):
    if args.label_rate == 0.05:
        root = '/root/workspace/lqy/NPNC/test_data/' + args.dataset
    else:
        root = '/root/workspace/lqy/NPNC/test_data/small/' + args.dataset + '/label_rate_' + str(args.label_rate)
    clean_graph_path = os.path.join(root, 'clean_graph.pkl')
    loaded_clean_data = load_data_our(clean_graph_path)
    return loaded_clean_data

def get_run_data(args, run):
    if args.label_rate == 0.05:
        root = '/root/workspace/lqy/NPNC/test_data/' + args.dataset
    else:
        root = '/root/workspace/lqy/NPNC/test_data/small/' + args.dataset + '/label_rate_' + str(args.label_rate)
    run_data_path = os.path.join(os.path.join(root), str(args.noise), str(args.ptb_rate), 'run_' + str(run), 'processed_graph.pkl')
    print(f'run_data_path = {run_data_path}')
    loaded_run_data = load_data_our(run_data_path)
    return loaded_run_data

def conbine_data(loaded_clean_data, loaded_run_data):
    # 将干净图的数据和运行后的数据合并到一个Data对象中
    data_combined = Data(
        x=loaded_clean_data.x,
        edge_index=loaded_clean_data.edge_index,
        clean_y=loaded_clean_data.clean_y,
        num_classes=loaded_clean_data.num_classes,
        transform=loaded_clean_data.transform,
        y=loaded_run_data.y,
        train_mask=loaded_run_data.train_mask,
        val_mask=loaded_run_data.val_mask,
        test_mask=loaded_run_data.test_mask,
        idx_train=loaded_run_data.idx_train,
        idx_val=loaded_run_data.idx_val,
        idx_test=loaded_run_data.idx_test
    )
    return data_combined


def load_config(dataset, noise, ptb_rate):
    # 创建配置文件名称
    config_name = f"{dataset}_{noise}_{int(ptb_rate * 100)}.yaml"
    root_path = f'/root/workspace/lqy/NPNC/DND-NET-main/our_op/config/{dataset}/'
    
    # 使用 os.path.join() 来拼接路径
    config_path = os.path.join(root_path, config_name)

    # 加载配置文件
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No configuration file found at {config_path}.")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # 返回配置内容
    return config

def run_main():
    global best_val_acc, prototypes, top_k_label, mix_label, high_mask, mean_label
    args = arg_parser()
    config = load_config(args.dataset, args.noise, args.ptb_rate)
    for key, value in config.items():
        setattr(args, key, value)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    clean_data = get_clean_data(args)
    accs = []
    accs_e = []
    for run in range(1, args.runs+1):
        setup_seed(run)
        if args.ptb_rate == 0.0:
            args.ptb_rate = 0.2
            run_data = get_run_data(args, run)
            args.ptb_rate = 0.0
        else:
            run_data = get_run_data(args, run)
        data = conbine_data(clean_data, run_data)
        if args.ptb_rate == 0.0:
            data.y = data.clean_y
        odata = data
        data = construct_super_nodes(data)
        data = data.to(device)
        odata = odata.to(device)
        print(f'args = {args}')
        model = encoder(data.num_features, args.hidden, data.num_classes, args.dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr= args.lr, weight_decay= args.weight_decay)
        best_val_acc = 0.0
        acc_val_all = []
        acc_test_all = []
        mix_label = None
        high_mask = None
        stopping_args = Stop_args(patience=100, max_epochs=200)
        early_stopping = EarlyStopping(model, **stopping_args)
        top_k_label = torch.zeros(data.num_nodes, data.num_classes, device=device) 
        prototypes = torch.zeros(data.num_classes, args.hidden, device=device)
        mean_label = torch.ones(data.num_nodes, data.num_classes, device=device)
        alpha_M = torch.nn.Parameter(torch.tensor(args.alpha_M), requires_grad=True)
        for epoch in range(1, 200+1):
            
            val_acc, test_acc, out= train(model, data, odata, epoch, args, optimizer, alpha_M)
            acc_val_all.append(val_acc)
            acc_test_all.append(test_acc)
            if early_stopping.check([val_acc], epoch):
                break
        best_epoch = np.argmax(acc_val_all)
        print('[RUN{}] best epoch: {}, val acc: {:.4f}, test acc: {:.4f}'.format(run, (1+best_epoch * args.eval_freq),acc_val_all[best_epoch], acc_test_all[best_epoch]))
        accs_e.append(acc_test_all[best_epoch])
        data = odata
    print('[FINAL RESULT] test acc: {:.4f}+-{:.4f}'.format(np.mean(accs_e), np.std(accs_e)))
    if args.label_rate == 0.05:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        res = [{
        'timestamp': current_time,  # 加入系统时间
        'dataset': args.dataset,
        'noise': args.noise,
        'noise_ratio': args.ptb_rate,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'dropout': args.dropout,
        'hidden': args.hidden,
        'top_k': args.top_k,
        'warm_up': args.warm_up,
        'temperature': args.temperature,
        'gamma': args.gamma,
        'beta': args.beta,
        'alpha_P': args.alpha_P,
        'sample_high_rates': args.sample_high_rate,
        'top_k': args.top_k,
        'acc_mean': '{:.4f}'.format(np.mean(accs_e)),
        'std': '{:.4f}'.format(np.std(accs_e))
        }]
        args.outf = '/root/workspace/lqy/NPNC/DND-NET-main/our_op/v1/result/finnal result/new_f_result.csv'
        # 获取输出文件的目录部分并确保目录存在
        output_dir = os.path.dirname(args.outf)
        os.makedirs(output_dir, exist_ok=True)

        # 检查文件是否存在
        file_exists = os.path.isfile(args.outf)
        with open(args.outf, 'a', newline='') as file:
            csv_writer = csv.DictWriter(file, fieldnames=res[0].keys())
            if not file_exists:
                csv_writer.writeheader()  # 写入表头
            csv_writer.writerows(res)  # 写入数据
    else:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        res = [{
        'timestamp': current_time,  # 加入系统时间
        'dataset': args.dataset,
        'label_rate': args.label_rate,
        'noise': args.noise,
        'noise_ratio': args.ptb_rate,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'dropout': args.dropout,
        'hidden': args.hidden,
        'top_k': args.top_k,
        'warm_up': args.warm_up,
        'temperature': args.temperature,
        'gamma': args.gamma,
        'beta': args.beta,
        'alpha_P': args.alpha_P,
        'sample_high_rates': args.sample_high_rate,
        'top_k': args.top_k,
        'acc_mean': '{:.4f}'.format(np.mean(accs_e)),
        'std': '{:.4f}'.format(np.std(accs_e))
        }]
        args.outf = '/root/workspace/lqy/NPNC/DND-NET-main/our_op/v1/result/finnal result/f_samll_result.csv'
        # 获取输出文件的目录部分并确保目录存在
        output_dir = os.path.dirname(args.outf)
        os.makedirs(output_dir, exist_ok=True)

        # 检查文件是否存在
        file_exists = os.path.isfile(args.outf)
        with open(args.outf, 'a', newline='') as file:
            csv_writer = csv.DictWriter(file, fieldnames=res[0].keys())
            if not file_exists:
                csv_writer.writeheader()  # 写入表头
            csv_writer.writerows(res)  # 写入数据
    

if __name__ == '__main__':
    run_main()