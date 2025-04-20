import argparse
import warnings

import torch
import train
from utils import init_seeds, setting_lr_parameters
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser('HGAD: Hierarchical Gaussian Mixture Normalizing Flow Modeling for Unified Anomaly Detection')
    
    # Model parameters
    parser.add_argument('--backbone_arch', default='tf_efficientnet_b6', type=str, 
                        help='feature extractor: (default: efficientnet_b6)')
    parser.add_argument('--flow_arch', default='conditional_flow_model', type=str, 
                        help='normalizing flow model (default: cnflow)')
    parser.add_argument('--feature_levels', default=3, type=int, 
                        help='nudmber of feature layers (default: 3)')
    parser.add_argument('--coupling_layers', default=12, type=int, 
                        help='number of coupling layers used in normalizing flow (default: 8)')
    parser.add_argument('--clamp_alpha', default=1.9, type=float, 
                        help='clamp alpha hyperparameter in normalizing flow (default: 1.9)')
    parser.add_argument('--pos_embed_dim', default=256, type=int,
                        help='dimension of positional enconding (default: 128)')
    parser.add_argument('--lambda1', default=1.0, type=float, 
                        help='hyperparameter lambad_1 in the loss (default: 1.0)')
    parser.add_argument('--lambda2', default=100.0, type=float, 
                        help='hyperparameter lambad_2 in the loss (default: 100.0)')
    parser.add_argument('--label_smoothing', default=0.02, type=float, 
                        help='smoothing the class labels (default: 0.02)')
    parser.add_argument('--n_intra_centers', default=10, type=int, 
                        help='number of intra-class centers (default: 10)')
    
    # Data configures
    parser.add_argument('--img_size', default=1024, type=int, 
                        help='image size (default: 1024)')
    parser.add_argument('--msk_size', default=256, type=int, 
                        help='mask size (default: 256)')
    parser.add_argument('--batch_size', default=8, type=int, 
                        help='train batch size (default: 32)')
    
    # training configures
    parser.add_argument('--lr', type=float, default=2e-4, 
                        help='learning rate (default: 2e-4)')
    parser.add_argument('--lr_decay_epochs', nargs='+', default=[50, 75, 90],
                        help='learning rate decay epochs (default: [50, 75, 90])')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, 
                        help='learning rate decay rate (default: 0.1)')
    parser.add_argument('--lr_warm', type=bool, default=True, 
                        help='learning rate warm up (default: True)')
    parser.add_argument('--lr_warm_epochs', type=int, default=2, 
                        help='learning rate warm up epochs (default: 2)')
    parser.add_argument('--lr_cosine', type=bool, default=True, 
                        help='cosine learning rate schedular (default: True)')
    parser.add_argument('--temp', type=float, default=0.5, 
                        help='temp of cosine learning rate scheduler (default: 0.5)')                    
    parser.add_argument('--meta_epochs', type=int, default=25, 
                        help='number of meta epochs to train (default: 25)')
    parser.add_argument('--sub_epochs', type=int, default=1, 
                        help='number of sub epochs to train (default: 8)')
    
    # misc
    parser.add_argument("--gpu", default='0', type=str, 
                        help='GPU device number')
    parser.add_argument("--seed", default=0, type=int, 
                        help='Random seed')
    parser.add_argument('--print_freq', default=200, type=int, 
                        help='frequency to print information')
    parser.add_argument('--output_dir', default='./outputs', type=str, 
                        help='directory to save model weights')
    
    parser.add_argument('--json_path', default='base_classes', type=str,)
    parser.add_argument('--pretrained_path', type=str, default='outputs/HGAD_base_img.pt')

    
    args = parser.parse_args()
    
    # base training
    if args.json_path.startswith("base"):
        if args.json_path.endswith("classes"):
            scenario = "scenario_1"
        elif args.json_path.endswith("except_mvtec_visa"):
            scenario = "scenario_2"
        elif args.json_path.endswith("except_continual_ad"):
            scenario = "scenario_3"
        else:
            raise ValueError("Invalid json_path for base training.")
    # continual training
    else:
        if args.json_path.startswith("5classes"):
            scenario = "scenario_4"
            
    args.output_dir = f"./outputs/{scenario}/5classes"
    
    return args
    
    
if __name__ == '__main__':
    args = parse_args()
    init_seeds(args.seed)
    setting_lr_parameters(args)
    
    args.device = torch.device("cuda:" + args.gpu)
    
    train.train(args)
