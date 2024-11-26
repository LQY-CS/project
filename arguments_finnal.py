import argparse

def arg_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--optimizer', type=str, default="Adam",
                        choices=['Adam', 'SGD'],
                        help='Optimizer for training')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum of optimizer.')
    
    parser.add_argument('--epochs', type=int,  default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--eval_freq', type=int,  default=1,
                        help='Number of epochs to train.')
    parser.add_argument('--useunlabel', type=str,  default='yes', choices=['yes', 'no'],
                        help='train cr loss with all data or unlabeled data')

    
   
    parser.add_argument('--noisy_rate', type=int,  default=0.1,
                        help='max number of neighbors for local sim.')
    parser.add_argument('--dataset', type=str, default="citeseer",
                        choices=['citeseer','pubmed', 'cs', 'arxiv', 'cora_ml', 'computers', 'photo', 'dblp'], help='dataset')
    parser.add_argument('--ptb_rate', type=float, default=0.2,
                        help="noise ptb_rate")
    parser.add_argument("--label_rate", type=float, default=0.05,
                        help='rate of labeled data')
    parser.add_argument("--val_rate", type=float, default=0.20,
                        help='rate of labeled data')
    parser.add_argument('--noise', type=str, default='uniform', choices=['uniform', 'pair'],
                        help='type of noises')

    parser.add_argument('--verbose', choices=["True", "False"], default=False,
                        help='printing logs?')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--patience', type=int, default=100)

    parser.add_argument('--runs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1024)
    
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='Weight decay.')
    parser.add_argument('--dropout', type=float, default=0.9, help='dropout')
    parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units.')
    parser.add_argument('--top_k', type=int, default=2, help='top_k')
    parser.add_argument('--warm_up', type=int, default=30, help='warm_up.')
    parser.add_argument('--temperature', type=float, default=0.5, help='temperature')
    parser.add_argument('--sample_high_rate', type=float, default=0.05, help='sample_high_rate')
    parser.add_argument('--alpha_P', type=float, default=0.8, help='alpha_P')
    parser.add_argument('--alpha_M', type=float, default=0.2, help='alpha_M')
    parser.add_argument('--beta', type=float, default=0.2, help='lass_ce')
    parser.add_argument('--gamma', type=float, default=1, help='gamma loss_con')
    
    args = parser.parse_args()
    
    return args