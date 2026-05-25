
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='CrisisMMD Multimodal Classification with EJMACC Sentiment')

    # 基础配置（原有）
    parser.add_argument('--model_name', type=str, default='full_task2', help='Model name for saving')
    parser.add_argument('--mode', type=str, default='both', choices=['both', 'image_only', 'text_only'],
                        help='Modalities to use (both/image_only/text_only)')
    parser.add_argument('--task', type=str, default='task2', choices=['task1', 'task2', 'task3', 'task2_full'],
                        help='Task type (task1:informative, task2:humanitarian, task3:damage)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device (cuda:0/cpu)')
    parser.add_argument('--max_iter', type=int, default=40, help='Max training iterations')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--eval', action='store_true', help='Evaluation mode (no training)')
    parser.add_argument('--log_hard_examples', action='store_true')

    parser.add_argument('--hard_examples_output', type=str, default='')
    parser.add_argument('--hard_conf_threshold', type=float, default=0.6)
    parser.add_argument('--use_llm_distill', action='store_true')
    parser.add_argument('--llm_review_path', type=str, default='')
    parser.add_argument('--llm_lambda', type=float, default=0.3)
    parser.add_argument('--llm_min_conf', type=float, default=0.0)
    # 模型加载（原有）
    parser.add_argument('--model_to_load', type=str, default='', help='Path to full model checkpoint')
    parser.add_argument('--image_model_to_load', type=str, default='', help='Path to image-only model checkpoint')
    parser.add_argument('--text_model_to_load', type=str, default='', help='Path to text-only model checkpoint')

    # LLM sample weighting (method A1)
    parser.add_argument('--use_llm_weight', action='store_true',
                        help='Enable LLM-provided sample weighting (weight * CE(gold))')
    parser.add_argument('--llm_weight_path', type=str, default='',
                        help='Path to JSONL: {tweet_id, weight, ...} for sample weighting')
    parser.add_argument('--llm_weight_clip', type=float, default=2.0,
                        help='Clip sample weight to [0, llm_weight_clip]')
    parser.add_argument('--llm_weight_default', type=float, default=1.0,
                        help='Default weight when tweet_id not found')
    # 数据配置（原有）
    parser.add_argument('--max_dataset_size', type=int, default=1000000,
                        help='Max number of samples to load (设置为大整数)')
    parser.add_argument('--load_size', type=int, default=256, help='Image load size')
    parser.add_argument('--crop_size', type=int, default=224, help='Image crop size')
    parser.add_argument('--wiki_data_root', type=str,
                        default='/root/autodl-tmp/pycharm_project_376/CrisisKAN-main/output1/wiki_results',
                        help='Path to wiki enhanced text data')
    parser.add_argument('--wiki_knowledge_path', type=str, default='',
                        help='预编码 wiki 向量 .pt（与样本顺序对齐）；留空则模型内用 text CLS 代替 wiki')
    parser.add_argument('--setting', type=str, default='settingA', help='Dataset setting (settingA/settingB)')

    # 情感特征配置（原有）
    parser.add_argument('--use_sentiment_features', action='store_true',
                        help='Enable EJMACC 6-dimensional sentiment features')
    parser.add_argument('--sentiment_model', type=str,
                        default='/root/autodl-tmp/CrisisKAN-main/bert-base-uncased-emotion',
                        help='Path to sentiment analysis model')
    parser.add_argument('--debug', action='store_true', help='Debug mode (print sentiment details)')

    # 训练配置（原有）
    parser.add_argument('--save_dir', type=str, default='./output', help='Directory to save results')
    parser.add_argument('--display_freq', type=int, default=100, help='Display training stats frequency')
    parser.add_argument('--tensorboard', action='store_true', help='Use tensorboard for logging')
    parser.add_argument('--freeze_image_encoder', action='store_true', help='Freeze image encoder weights')
    parser.add_argument('--freeze_text_encoder', action='store_true', help='Freeze text encoder weights')

    # ===================== 新增：解决过拟合的核心参数 =====================
    # 正则化参数
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Global dropout rate (默认0.3，过拟合时调至0.5)')
    parser.add_argument('--attention_dropout', type=float, default=0.3, help='Attention layer dropout rate')
    parser.add_argument('--classifier_dropout', type=float, default=0.3, help='Classifier head dropout rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (L2正则，建议1e-4~1e-5)')

    # 学习率调度参数
    parser.add_argument('--warmup_epochs', type=int, default=2, help='Learning rate warmup epochs')
    parser.add_argument('--cosine_T0', type=int, default=5, help='Cosine annealing restart period')
    parser.add_argument('--cosine_eta_min', type=float, default=1e-6, help='Minimum learning rate for cosine annealing')

    # 早停参数
    parser.add_argument('--early_stop_patience', type=int, default=5, help='Patience for early stopping (0=disable)')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.001, help='Min loss change for early stopping')

    # 模型复杂度参数
    parser.add_argument('--num_attention_layers', type=int, default=2,
                        help='Number of cross attention layers (默认2，过拟合调1)')
    parser.add_argument('--enhance_mode', type=str, default='token_only',
                        choices=['token_only', 'patch_only'],
                        help='Single-side enhancement mode for task1/2/3')
    parser.add_argument('--use_task_gate', action='store_true',
                        help='Learnable task gate: run task1/2/3 branches; gate supervised by current --task')
    parser.add_argument('--task_gate_lambda', type=float, default=0.1,
                        help='Weight for task gate supervision loss (set 0 to disable aux loss)')
    parser.add_argument('--gate_hidden_dim', type=int, default=256, help='Hidden dim for task gate MLP')

    # 图像描述（BLIP）→ 文本向量，与推文向量融合
    parser.add_argument('--use_image_caption', action='store_true',
                        help='BLIP 生成图像描述，经 ELECTRA 编码后与推文 [CLS] 融合')
    parser.add_argument('--caption_model_name', type=str,
                        default='/root/autodl-tmp/pycharm_project_376/blip-image-captioning-base',
                        help='BLIP 本地目录或 HuggingFace 模型名')
    parser.add_argument('--caption_max_length', type=int, default=32, help='BLIP 生成最大 token 数')
    parser.add_argument('--train_caption_model', action='store_true',
                        help='若设置则 BLIP 参数可训练（默认冻结，显存开销大）')

    # 文本 → 图像（diffusers 文生图）→ ViT 与真实图 [CLS] 融合
    parser.add_argument('--use_text_to_image', action='store_true',
                        help='用推文解码为 prompt，SD 类模型生成图后再过 ViT，与真实图全局特征融合')
    parser.add_argument('--t2i_model_name', type=str, default='stabilityai/sd-turbo',
                        help='HuggingFace 文生图模型（默认 sd-turbo，需 diffusers）')
    parser.add_argument('--t2i_num_inference_steps', type=int, default=1,
                        help='文生图扩散步数（sd-turbo 常用 1～4）')
    parser.add_argument('--t2i_guidance_scale', type=float, default=0.0,
                        help='Classifier-free guidance（sd-turbo 常设为 0）')
    parser.add_argument('--save_text_to_image', action='store_true',
                        help='保存文生图生成结果到本地目录')
    parser.add_argument('--t2i_save_dir', type=str, default='./output/t2i_generated',
                        help='文生图保存目录（当 --save_text_to_image 启用时生效）')

    # 数据增强参数
    parser.add_argument('--use_image_aug', action='store_true',
                        help='Enable image data augmentation (随机裁剪/翻转/颜色抖动)')

    # 分层冻结参数（更精细的冻结控制，替代原有简单冻结）
    parser.add_argument('--freeze_image_layers', type=int, default=0,
                        help='Freeze first N layers of image encoder (0=不冻结)')
    parser.add_argument('--freeze_text_layers', type=int, default=0,
                        help='Freeze first N layers of text encoder (0=不冻结)')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print("Args parsed successfully!")
    print(f"Use sentiment features: {args.use_sentiment_features}")
    # 打印新增参数示例
    print(f"Dropout rate: {args.dropout_rate}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Use image augmentation: {args.use_image_aug}")
