import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='RemoteSAM training and testing')
    parser.add_argument('--amsgrad', action='store_true',
                        help='if true, set amsgrad to True in an Adam or AdamW optimizer.')
    parser.add_argument('-b', '--batch-size', default=2, type=int)
    parser.add_argument('--bert_tokenizer', default='./bert-base-uncased', help='HuggingFace BERT tokenizer name or path')
    parser.add_argument('--ck_bert', default='./bert-base-uncased', help='HuggingFace BERT model name or path')
    parser.add_argument('--modernbert_path', default='./gte-modernbert-base',
                        help='local ModernBERT model and tokenizer directory')
    parser.add_argument('--vision_backbone', default='dinov3_convnext_b',
                        choices=['swin_b', 'dinov3_convnext_b'],
                        help='visual backbone implementation')
    parser.add_argument('--text_encoder', default='modernbert',
                        choices=['bert', 'modernbert'],
                        help='text encoder implementation')
    parser.add_argument('--dinov3_convnext_path', default='./dinov3-convnext-B',
                        help='local DINOv3 ConvNeXt-B model directory')
    parser.add_argument('--dataset', default='rrsisd', help='dataset name')
    parser.add_argument('--ddp_trained_weights', action='store_true',
                        help='Only needs specified when testing,'
                             'whether the weights to be loaded are from a DDP-trained model')
    parser.add_argument('--dist_timeout', default=7200, type=int,
                        help='distributed process-group timeout in seconds')
    parser.add_argument('--device', default='cuda:0', help='device')  # only used when testing on a single machine
    parser.add_argument('--epochs', default=40, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--fusion_drop', default=0.0, type=float, help='dropout rate for PWAMs')
    parser.add_argument('--img_size', default=896, type=int, help='input image size')
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0,
                        help='local rank for DistributedDataParallel')
    parser.add_argument('--lr', default=0.00003, type=float,
                        help='base learning rate for newly initialized RemoteSAM modules')
    parser.add_argument('--backbone_lr', default=0.00001, type=float,
                        help='learning rate for pretrained visual backbone parameters')
    parser.add_argument('--fusion_lr', default=0.00005, type=float,
                        help='learning rate for RemoteSAM fusion/CIM parameters')
    parser.add_argument('--classifier_lr', default=0.00005, type=float,
                        help='learning rate for decoder/classifier parameters')
    parser.add_argument('--text_lr', default=0.000005, type=float,
                        help='learning rate for text encoder parameters')
    parser.add_argument('--text_adapter_lr', default=None, type=float,
                        help='learning rate for the text feature adapter; defaults to --fusion_lr or --lr')
    parser.add_argument('--text_adapter_dropout', default=0.1, type=float,
                        help='dropout rate inside the text feature adapter')
    parser.add_argument('--no_text_adapter', action='store_true',
                        help='disable the residual text feature adapter')
    parser.add_argument('--use_modal_lora', action='store_true',
                        help='enable hard-gated RGB/SAR LoRA branches inside the visual backbone')
    parser.add_argument('--modal', default='rgb', choices=['rgb', 'sar'],
                        help='input modality for task/inference scripts; controls modal LoRA routing')
    parser.add_argument('--modal_lora_r', '--lora_r', dest='modal_lora_r', default=4, type=int,
                        help='rank of each RGB/SAR modal LoRA branch')
    parser.add_argument('--modal_lora_alpha', '--lora_alpha', dest='modal_lora_alpha', default=8.0, type=float,
                        help='LoRA alpha; alpha/r is the residual scaling')
    parser.add_argument('--modal_lora_dropout', '--lora_dropout', dest='modal_lora_dropout', default=0.05, type=float,
                        help='dropout inside the modal LoRA residual branch')
    parser.add_argument('--modal_lora_lr', default=0.00005, type=float,
                        help='learning rate for modal LoRA parameters')
    parser.add_argument('--modal_lora_stages', default='2,3',
                        help='0-indexed backbone stages using modal LoRA; default keeps it on high-level stages')
    parser.add_argument('--modal_lora_targets', default='',
                        help='comma-separated target Linear names; default is backbone-specific')
    parser.add_argument('--modal_lora_freeze_base', action='store_true',
                        help='freeze wrapped base Linear weights and train only modal LoRA residuals')
    parser.add_argument('--mha', default='', help='If specified, should be in the format of a-b-c-d, e.g., 4-4-4-4,'
                                                  'where a, b, c, and d refer to the numbers of heads in stage-1,'
                                                  'stage-2, stage-3, and stage-4 PWAMs')
    parser.add_argument('--max_tokens', default=20, type=int, help='maximum text token length')
    parser.add_argument('--model', default='lavt_one', help='model: lavt, lavt_one')
    parser.add_argument('--model_id', default='RemoteSAM_DINOv3ConvNeXtB_ModernBERT',
                        help='name to identify the model')
    parser.add_argument('--output-dir', default='./checkpoints/DINOv3-convnext-B-modernbert/',
                        help='path where to save checkpoint weights')
    parser.add_argument('--pin_mem', action='store_true',
                        help='If true, pin memory when using the data loader.')
    parser.add_argument('--pretrained_swin_weights', default='./pretrained_weights/swin_base_patch4_window12_384_22k.pth',
                        help='path to pre-trained Swin backbone weights')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--refer_data_root', default='/data/dishimin/RemoteSAM-270K', help='REFER dataset root directory')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--split', default='test', help='only used when testing')
    parser.add_argument('--splitBy', default='unc', help='change to umd or google when the datasset is G-Ref (RefCOCOg)')
    parser.add_argument('--swin_type', default='base',
                        help='tiny, small, base, or large variants of the Swin Transformer')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float, metavar='W', help='weight decay',
                        dest='weight_decay')
    parser.add_argument('--window12', action='store_true',
                        help='only needs specified when testing,'
                             'when training, window size is inferred from pre-trained weights file name'
                             '(containing \'window12\'). Initialize Swin with window size 12 instead of the default 7.')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument("--imageFolder", default='/data/user2/zjj/RRSIS-D/images/rrsisd/JPEGImages', type=str, 
                        help="imageFolder path")
    parser.add_argument("--ref_file_path", default='/data/user2/zjj/RRSIS-D/rrsisd/refs(unc).p', type=str, 
                        help="ref_file_path")
    parser.add_argument("--instances_path", default='/data/user2/zjj/RRSIS-D/rrsisd/instances.json', type=str, 
                        help="instances.json's path")
    parser.add_argument("--eval", action="store_true", help="Only run evaluation")
    parser.add_argument('--EPOC', action='store_true', help='if true, use EPOC')
    parser.add_argument('--save_path', default='', help='save_path')
    parser.add_argument('--task', type=str, choices=['CNT', 'DET', 'CAP', 'VG', 'MCC', 'MLC','SEG','REF'], 
                        help='For tasks other than referring segmentation and semantic segmentation, specify the task')
    parser.add_argument('--annoFolder', default='/data/dishimin/iSAID/rrsisd/Annotations/', type=str, help="annoFolder path")
    parser.add_argument('--sentence', default='', help='sentence input')
    parser.add_argument('--_class', default='', help='class input')
    parser.add_argument('--det_prompt_class', default='',
                        help='optional DET text prompt class; evaluation category remains --_class')
    parser.add_argument('--skip_class_eval', action='store_true',
                        help='for DET evaluation, save per-class predictions without running per-class COCOeval')
    parser.add_argument('--use_swanlab', action='store_true', help='log training curves with swanlab')
    parser.add_argument('--swanlab_project', default='remotesam_rgb', help='swanlab project name')
    parser.add_argument('--swanlab_run_name', default='', help='optional swanlab run name')
    parser.add_argument('--swanlab_mode', default='cloud', choices=['cloud', 'local', 'disabled'],
                        help='swanlab logging mode')
    

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args_dict = parser.parse_args()
