

from os import path as osp
import os
import sys
import importlib.util
import logging
import json
import math
import time
import nltk
from tqdm import tqdm

# 将工程根目录加入路径（与 main.py 同目录，便于 args/trainer/crisismmd_dataset 等）
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# PyTorch 相关
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 自定义模块（models.py 按文件路径加载，避免 cwd/sys.path 找不到模块）
from args import get_args
from trainer import Trainer
from crisismmd_dataset import CrisisMMDataset

_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models.py")
if not os.path.isfile(_MODEL_PATH):
    raise FileNotFoundError(
        f"未找到模型文件: {_MODEL_PATH}\n"
        "请确认 models.py 与 main.py 在同一目录并已上传到服务器。"
    )
_spec = importlib.util.spec_from_file_location("crisis_mm_models", _MODEL_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"无法加载: {_MODEL_PATH}")
_mm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mm)
ViTElectraMMModel = _mm.ViTElectraMMModel
ImageOnlyModel = _mm.ImageOnlyModel
TextOnlyModel = _mm.TextOnlyModel

# 下载停用词（用于文本预处理）
try:
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logging.warning(f"停用词下载失败: {e}")

# ===================== 全局配置 =====================
# 设置日志（基础配置）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)


# 学习率日志打印函数（替代ReduceLROnPlateau的verbose参数）
def log_learning_rate(epoch, optimizer, scheduler=None):
    """打印当前学习率，可选打印调度器状态"""
    lr = optimizer.param_groups[0]['lr']
    log_msg = f"📚 Epoch {epoch + 1} | 当前学习率: {lr:.6f}"
    if scheduler:
        log_msg += f" | 调度器模式: {scheduler.mode} | 衰减因子: {scheduler.factor}"
    logging.info(log_msg)


# ===================== 主函数 =====================
if __name__ == '__main__':
    # 1. 解析命令行参数
    opt = get_args()

    # 2. 设备配置（单/多GPU处理）
    gpu_ids = []
    if opt.device.startswith('cuda:'):
        if ',' in opt.device:
            # 多GPU配置
            gpu_ids = [int(x.strip()) for x in opt.device.split(':')[1].split(',')]
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in gpu_ids)
            device = torch.device(f"cuda:{gpu_ids[0]}")
            logging.info(f"✅ 使用多GPU训练: {gpu_ids}, 主设备: {device}")
        else:
            # 单GPU配置
            single_gpu_id = int(opt.device.split(':')[1]) if len(opt.device.split(':')) > 1 else 0
            os.environ["CUDA_VISIBLE_DEVICES"] = str(single_gpu_id)
            device = torch.device(opt.device)
            logging.info(f"✅ 使用单GPU训练: {device}")
    else:
        # CPU配置
        device = torch.device('cpu')
        logging.info(f"✅ 使用CPU训练")

    # 数据加载器工作进程数（核心修复：强制设为0，避免多进程卡死）
    num_workers = 0  # 无论GPU/CPU都设为0，彻底解决多进程阻塞问题
    logging.info(f"📌 数据加载工作进程数: {num_workers} (强制单进程，避免卡死)")

    # 3. 核心训练配置
    # 梯度累积参数
    grad_accumulation_steps = 2
    effective_batch_size = opt.batch_size * grad_accumulation_steps
    logging.info(
        f"📌 梯度累积配置 | 步数: {grad_accumulation_steps} | 原始batch_size: {opt.batch_size} | 等效batch_size: {effective_batch_size}"
    )

    # 模型/任务核心配置
    model_to_load = opt.model_to_load
    image_model_to_load = opt.image_model_to_load
    text_model_to_load = opt.text_model_to_load
    EVAL = opt.eval  # 评估/训练模式
    USE_TENSORBOARD = opt.tensorboard
    SAVE_DIR = opt.save_dir
    MODEL_NAME = opt.model_name if opt.model_name else f"model_{int(time.time())}"
    MODE = opt.mode  # 'both'/'image_only'/'text_only'
    TASK = opt.task  # 'task1'/'task2'/'task2_full'/'task3'
    JOINT_MODE = getattr(opt, "joint_mode", "none")

    # Joint training 下，评估/输出维度主要依赖 TASK->OUTPUT_SIZE 映射。
    # 这里为了避免 task2 head 类别数初始化错误，要求 joint_mode 时 --task 固定为 task2（6类）。
    if JOINT_MODE in ("task12", "task23") and TASK != "task2":
        raise ValueError(f"Joint training requires --task task2 (6 classes) as the evaluation task, got: {TASK}")

    # 4. 任务输出维度配置（对齐CrisisMMD数据集定义）
    OUTPUT_SIZE = {
        'task1': 2,  # 信息性分类（informative/not）
        'task2': 6,  # 人道主义分类（6类合并版）
        'task2_full': 8,  # 人道主义分类（8类完整版）
        'task3': 3  # 损失程度分类（little/mild/severe）
    }.get(TASK, None)

    if OUTPUT_SIZE is None:
        raise ValueError(f"❌ 未知任务类型: {TASK} | 支持的任务: task1/task2/task2_full/task3")
    logging.info(f"📌 任务配置 | 类型: {TASK} | 输出维度: {OUTPUT_SIZE}")

    # 5. 结果保存文件夹创建
    save_dir = osp.join(SAVE_DIR, MODEL_NAME)
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    logging.info(f"📌 结果保存路径: {save_dir}")

    # 补充日志文件（按任务/时间区分）
    task_log_filename = osp.join(save_dir, f'{TASK}_{JOINT_MODE}_training_{int(time.time())}.log')
    task_file_handler = logging.FileHandler(task_log_filename)
    task_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(task_file_handler)

    # 打印实验配置汇总
    logging.info("\n" + "=" * 80)
    logging.info("📋 实验配置汇总")
    logging.info("=" * 80)
    logging.info(f"任务类型: {TASK} | 联合模式: {JOINT_MODE} | 模型模式: {MODE} | 评估模式: {EVAL}")
    logging.info(f"设备类型: {device} | GPU数量: {len(gpu_ids) if gpu_ids else 1}")
    logging.info(f"Batch配置: 原始={opt.batch_size} | 等效={effective_batch_size} | 梯度累积={grad_accumulation_steps}")
    logging.info(f"情感特征: {'✅ 启用' if opt.use_sentiment_features else '❌ 禁用'}")
    logging.info(f"最大迭代: {opt.max_iter} | 初始学习率: {opt.learning_rate}")
    logging.info(f"数据设置: {opt.setting} | 最大数据集大小: {opt.max_dataset_size}")
    logging.info("=" * 80 + "\n")

    # 6. 数据集加载（训练/验证/测试集）
    train_loader = None
    dev_loader = None
    test_loader = None
    train_loader_joint = None

    # 6.1 训练集加载（仅训练模式）
    if not EVAL:
        try:
            if JOINT_MODE == "none":
                train_set = CrisisMMDataset()
                train_set.initialize(
                    opt,
                    phase="train",
                    cat="all",
                    task=TASK,
                    setting=opt.setting,
                )
                train_loader = DataLoader(
                    train_set,
                    batch_size=opt.batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=True if device.type == "cuda" else False,
                    drop_last=True,  # 避免最后一个批次样本数不足
                )
            else:
                # 交替多任务训练：A/B 两个 train loader
                if JOINT_MODE == "task12":
                    task_a, task_b = "task1", "task2"
                elif JOINT_MODE == "task23":
                    task_a, task_b = "task2", "task3"
                else:
                    raise ValueError(f"Unknown joint_mode: {JOINT_MODE}")

                train_set_a = CrisisMMDataset()
                train_set_a.initialize(
                    opt,
                    phase="train",
                    cat="all",
                    task=task_a,
                    setting=opt.setting,
                )
                loader_a = DataLoader(
                    train_set_a,
                    batch_size=opt.batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=True if device.type == "cuda" else False,
                    drop_last=True,
                )

                train_set_b = CrisisMMDataset()
                train_set_b.initialize(
                    opt,
                    phase="train",
                    cat="all",
                    task=task_b,
                    setting=opt.setting,
                )
                loader_b = DataLoader(
                    train_set_b,
                    batch_size=opt.batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=True if device.type == "cuda" else False,
                    drop_last=True,
                )
                train_loader_joint = {"a": loader_a, "b": loader_b}

                # joint 模式：Trainer 需要两个 loader
                train_loader = train_loader_joint

            # 彻底删除预加载逻辑（核心修复：避免内存阻塞）
            if JOINT_MODE == "none":
                logging.info(
                    f"✅ 训练集加载完成 | 批次数量: {len(train_loader)} | 总样本数: {len(train_set)}"
                )
            else:
                logging.info(
                    "✅ 联合训练数据加载完成 | "
                    f"A={task_a} batches={len(loader_a)} | B={task_b} batches={len(loader_b)}"
                )

            # 验证情感特征加载
            # 情感特征形状日志（仅在非联合时可复用 train_set 变量；联合模式只做简要提示）
            if JOINT_MODE == "none" and opt.use_sentiment_features and len(train_set) > 0:
                # 取第一个样本验证（非预加载，仅验证字段）
                first_sample = train_set[0]
                if 'sentiment_features' in first_sample:
                    sent_shape = first_sample['sentiment_features'].shape
                    logging.info(f"✅ 训练集EJMACC情感特征已加载 | 特征形状: {sent_shape}")
                else:
                    logging.warning(f"⚠️ 启用了情感特征，但训练集未返回sentiment_features字段（不影响训练，会自动补0）")
        except Exception as e:
            logging.error(f"❌ 训练集加载失败: {e}")
            raise e

    # 6.2 验证集加载（必加载）
    try:
        dev_set = CrisisMMDataset()
        dev_set.initialize(
            opt,
            phase='dev',
            cat='all',
            task=TASK,
            setting=opt.setting
        )
        dev_loader = DataLoader(
            dev_set,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )
        # 删除预加载逻辑
        logging.info(f"✅ 验证集加载完成 | 批次数量: {len(dev_loader)} | 总样本数: {len(dev_set)}")
    except Exception as e:
        logging.error(f"❌ 验证集加载失败: {e}")
        raise e

    # 6.3 测试集加载（必加载）
    try:
        test_set = CrisisMMDataset()
        test_set.initialize(
            opt,
            phase='test',
            cat='all',
            task=TASK,
            setting=opt.setting
        )
        test_loader = DataLoader(
            test_set,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )
        # 删除预加载逻辑
        logging.info(f"✅ 测试集加载完成 | 批次数量: {len(test_loader)} | 总样本数: {len(test_set)}")

        # 验证测试集情感特征
        if opt.use_sentiment_features and len(test_set) > 0:
            first_sample = test_set[0]
            if 'sentiment_features' in first_sample:
                sent_shape = first_sample['sentiment_features'].shape
                logging.info(f"✅ 测试集EJMACC情感特征已加载 | 特征形状: {sent_shape}")
            else:
                logging.warning(f"⚠️ 启用了情感特征，但测试集未返回sentiment_features字段（不影响训练，会自动补0）")
    except Exception as e:
        logging.error(f"❌ 测试集加载失败: {e}")
        raise e

    # 7. 模型初始化
    # 数据里未映射标签可能为 -1；蒸馏分支也会对缺失样本填 -1，需与 trainer 中 ignore 一致
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    model = None

    try:
        if MODE == 'text_only':
            # 文本单模态模型
            model = TextOnlyModel(
                save_dir=save_dir,
                num_class=OUTPUT_SIZE
            ).to(device)
            logging.info(f"✅ 文本单模态模型初始化完成 | TextOnlyModel")

        elif MODE == 'image_only':
            # 图像单模态模型
            model = ImageOnlyModel(
                save_dir=save_dir,
                num_class=OUTPUT_SIZE
            ).to(device)
            logging.info(f"✅ 图像单模态模型初始化完成 | ImageOnlyModel")

        elif MODE == 'both':
            # 多模态模型（集成EJMACC情感特征）
            _wk = (getattr(opt, "wiki_knowledge_path", "") or "").strip()
            model = ViTElectraMMModel(
                save_dir=save_dir,
                dim_visual_repr=768,
                dim_text_repr=768,
                num_class=OUTPUT_SIZE,  # 兼容旧参数，自动映射到对应任务
                sentiment_weight=1.0,
                wiki_knowledge_path=_wk if _wk else None,
                use_image_caption=bool(getattr(opt, "use_image_caption", False)),
                caption_model_name=str(
                    getattr(opt, "caption_model_name", "/root/autodl-tmp/pycharm_project_376/blip-image-captioning-base")
                ),
                freeze_caption_model=not bool(getattr(opt, "train_caption_model", False)),
                caption_max_length=int(getattr(opt, "caption_max_length", 32)),
                use_text_to_image=bool(getattr(opt, "use_text_to_image", False)),
                t2i_model_name=str(getattr(opt, "t2i_model_name", "stabilityai/sd-turbo")),
                t2i_num_inference_steps=int(getattr(opt, "t2i_num_inference_steps", 1)),
                t2i_guidance_scale=float(getattr(opt, "t2i_guidance_scale", 0.0)),
                save_text_to_image=bool(getattr(opt, "save_text_to_image", False)),
                t2i_save_dir=str(getattr(opt, "t2i_save_dir", "./output/t2i_generated")),
            ).to(device)
            logging.info(f"✅ 多模态模型初始化完成 | ViTElectraMMModel | 任务: {TASK} | 启用EJMACC情感特征融合")

        else:
            raise ValueError(f"❌ 未知模型模式: {MODE} | 支持模式: both/image_only/text_only")
    except Exception as e:
        logging.error(f"❌ 模型初始化失败: {e}")
        raise e

    # 8. 多GPU包装（DataParallel）
    if device.type == 'cuda' and len(gpu_ids) > 1 and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
        logging.info(f"✅ 模型包装为DataParallel | 使用{torch.cuda.device_count()}个GPU")

    # 9. 优化器与学习率调度器配置
    # 9.1 优化器（AdamW，论文常用）
    optimizer = optim.AdamW(
        model.parameters(),
        lr=opt.learning_rate,
        eps=1e-8,
        weight_decay=1e-4
    )

    # 9.2 学习率调度器（修复verbose参数问题，改用手动打印）
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',  # 基于验证集损失最小化调整
        factor=0.1,  # 学习率衰减因子（乘以0.1）
        patience=5,  # 5个epoch无提升则衰减
        threshold=1e-4,  # 提升阈值（避免微小波动）
        cooldown=2,  # 衰减后冷却2个epoch
        min_lr=1e-6,  # 最小学习率（防止过小）
    )

    # 9.3 计算总优化步数（适配梯度累积）
    if not EVAL and train_loader:
        # joint_mode 下 train_loader 是 {"a": loader_a, "b": loader_b}；非 joint 下是 DataLoader
        if isinstance(train_loader, dict):
            loader_a = train_loader.get("a", None)
            loader_b = train_loader.get("b", None)
            if loader_a is None or loader_b is None:
                raise ValueError("joint_mode requires train_loader['a'] and train_loader['b']")
            total_train_samples = max(len(loader_a.dataset), len(loader_b.dataset))
        else:
            total_train_samples = len(train_loader.dataset)
        optim_steps_per_iter = math.ceil(total_train_samples / effective_batch_size)
        t_total = optim_steps_per_iter * opt.max_iter
    else:
        t_total = 1000  # 评估模式默认值
    logging.info(f"📌 训练配置 | 总优化步数: {t_total} | 最大迭代次数: {opt.max_iter}")

    # 10. 训练器初始化（核心修复：删除多余的log_lr_func参数）
    try:
        trainer = Trainer(
            train_loader=train_loader,
            dev_loader=dev_loader,
            test_loader=test_loader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            save_dir=save_dir,
            display=opt.display_freq,
            eval=EVAL,
            device=device,
            tensorboard=USE_TENSORBOARD,
            mode=MODE,
            grad_accumulation_steps=grad_accumulation_steps,
            task_type=TASK,
            joint_mode=JOINT_MODE,
            lambda_joint_a=getattr(opt, "lambda_joint_a", 1.0),
            lambda_joint_b=getattr(opt, "lambda_joint_b", 1.0),

            # hard examples
            log_hard_examples=True,
            hard_examples_output=(
                    getattr(opt, "hard_examples_output", "")
                    or os.path.join(save_dir, f"hard_examples_{TASK}.jsonl")
            ),

            # LLM distill：用命令行参数（不再写死）
            use_llm_distill=getattr(opt, "use_llm_distill", False),
            llm_review_path=(getattr(opt, "llm_review_path", "") or None),
            llm_lambda=float(getattr(opt, "llm_lambda", 0.3)),
        )
        logging.info(f"✅ 训练器初始化完成 | Trainer")
    except Exception as e:
        logging.error(f"❌ 训练器初始化失败: {e}")
        raise e

    # 11. 预训练模型加载
    # 11.1 加载完整模型（多模态/单模态）
    if model_to_load and os.path.exists(model_to_load):
        try:
            if hasattr(trainer, '_load_model'):
                trainer._load_model(model_to_load)
            else:
                # 兼容直接加载模型权重
                checkpoint = torch.load(model_to_load, map_location=device)
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            logging.info(f"✅ 加载完整预训练模型: {model_to_load}")
        except Exception as e:
            logging.warning(f"⚠️ 完整模型加载失败: {e} (将从头训练)")

    # 11.2 多模态模型加载预训练组件（图像/文本编码器）
    if MODE == 'both':
        # 加载图像编码器
        if image_model_to_load and os.path.exists(image_model_to_load):
            logging.info(f"\n📥 加载预训练图像编码器: {image_model_to_load}")
            try:
                model_module = model.module if hasattr(model, 'module') else model
                loaded_params = model_module.load_pretrained_components(
                    image_checkpoint_path=image_model_to_load,
                    text_checkpoint_path=None
                )
                logging.info(f"✅ 图像编码器加载完成 | 加载参数数: {loaded_params}")
            except Exception as e:
                logging.warning(f"⚠️ 图像编码器加载失败: {str(e)} (将使用随机初始化)")

        # 加载文本编码器
        if text_model_to_load and os.path.exists(text_model_to_load):
            logging.info(f"\n📥 加载预训练文本编码器: {text_model_to_load}")
            try:
                model_module = model.module if hasattr(model, 'module') else model
                loaded_params = model_module.load_pretrained_components(
                    image_checkpoint_path=None,
                    text_checkpoint_path=text_model_to_load
                )
                logging.info(f"✅ 文本编码器加载完成 | 加载参数数: {loaded_params}")
            except Exception as e:
                logging.warning(f"⚠️ 文本编码器加载失败: {str(e)} (将使用随机初始化)")

        # 冻结编码器（可选，兼容命令行参数）
        freeze_image = getattr(opt, 'freeze_image_encoder', False)
        freeze_text = getattr(opt, 'freeze_text_encoder', False)
        if freeze_image or freeze_text:
            logging.info(f"\n🔒 冻结模型组件 | 图像编码器: {freeze_image} | 文本编码器: {freeze_text}")
            try:
                model_module = model.module if hasattr(model, 'module') else model
                model_module.freeze_components(freeze_image=freeze_image, freeze_text=freeze_text)
                logging.info(f"✅ 模型组件冻结完成")
            except Exception as e:
                logging.warning(f"⚠️ 模型组件冻结失败: {str(e)} (将不冻结编码器)")

    # 11.3 单模态模型加载
    elif MODE == 'image_only' and image_model_to_load and os.path.exists(image_model_to_load):
        try:
            trainer._load_model(image_model_to_load)
            logging.info(f"✅ 加载图像单模态预训练模型: {image_model_to_load}")
        except Exception as e:
            logging.warning(f"⚠️ 图像单模态模型加载失败: {e} (将从头训练)")

    elif MODE == 'text_only' and text_model_to_load and os.path.exists(text_model_to_load):
        try:
            trainer._load_model(text_model_to_load)
            logging.info(f"✅ 加载文本单模态预训练模型: {text_model_to_load}")
        except Exception as e:
            logging.warning(f"⚠️ 文本单模态模型加载失败: {e} (将从头训练)")

    # 12. 执行训练/评估
    if not EVAL:
        # 训练模式
        logging.info("\n" + "=" * 80)
        logging.info("🚀 开始训练")
        logging.info("=" * 80)
        logging.info(f"模型结构: {model.__class__.__name__}")
        logging.info(f"初始学习率: {opt.learning_rate}")
        logging.info(f"最大迭代次数: {opt.max_iter}")
        logging.info(f"损失函数: CrossEntropyLoss")
        logging.info("=" * 80 + "\n")

        # 启动训练（核心逻辑完全保留，不影响结果）
        best_results = trainer.train(max_iter=opt.max_iter)

        # 打印最佳训练结果
        if best_results:
            logging.info("\n" + "=" * 80)
            logging.info("🎉 训练完成 - 最佳模型结果")
            logging.info("=" * 80)
            logging.info(f"最佳迭代轮次: {best_results.get('iteration', 'N/A')}")
            logging.info(f"验证集加权F1: {best_results.get('dev_f1', 0.0):.4f}")
            logging.info(f"测试集准确率: {best_results.get('test_acc', 0.0):.4f}")
            logging.info(f"测试集Micro F1: {best_results.get('test_micro_f1', 0.0):.4f}")
            logging.info(f"测试集Macro F1: {best_results.get('test_macro_f1', 0.0):.4f}")
            logging.info(f"测试集Weighted F1: {best_results.get('test_weighted_f1', 0.0):.4f}")
            logging.info(f"\n📁 所有结果保存路径: {save_dir}")
            logging.info("=" * 80)

            # 保存最佳结果到JSON
            best_results_path = osp.join(save_dir, f'best_results_{TASK}.json')
            with open(best_results_path, 'w', encoding='utf-8') as f:
                json.dump(best_results, f, indent=4, ensure_ascii=False)
            logging.info(f"✅ 最佳结果已保存至: {best_results_path}")

    else:
        # 评估模式
        logging.info("\n" + "=" * 80)
        logging.info("📊 开始评估")
        logging.info("=" * 80)

        # 执行评估
        eval_results = trainer.evaluate()

        # 打印并保存评估结果
        if eval_results:
            logging.info("\n" + "=" * 80)
            logging.info("📊 评估结果汇总")
            logging.info("=" * 80)

            # 验证集结果
            val_res = eval_results.get('validation', {})
            logging.info("验证集:")
            logging.info(f"  准确率: {val_res.get('accuracy', 0.0):.4f}")
            logging.info(f"  损失值: {val_res.get('loss', 0.0):.4f}")
            logging.info(f"  Micro F1: {val_res.get('micro_f1', 0.0):.4f}")
            logging.info(f"  Macro F1: {val_res.get('macro_f1', 0.0):.4f}")
            logging.info(f"  Weighted F1: {val_res.get('weighted_f1', 0.0):.4f}")

            # 测试集结果
            test_res = eval_results.get('test', {})
            logging.info("\n测试集:")
            logging.info(f"  准确率: {test_res.get('accuracy', 0.0):.4f}")
            logging.info(f"  损失值: {test_res.get('loss', 0.0):.4f}")
            logging.info(f"  Micro F1: {test_res.get('micro_f1', 0.0):.4f}")
            logging.info(f"  Macro F1: {test_res.get('macro_f1', 0.0):.4f}")
            logging.info(f"  Weighted F1: {test_res.get('weighted_f1', 0.0):.4f}")
            logging.info("=" * 80)

            # 保存评估结果
            eval_save_path = osp.join(save_dir, f'evaluation_results_{TASK}_{int(time.time())}.json')
            with open(eval_save_path, 'w', encoding='utf-8') as f:
                json.dump(eval_results, f, indent=4, ensure_ascii=False)
            logging.info(f"✅ 评估结果已保存至: {eval_save_path}")
        else:
            logging.error("❌ 评估结果为空！")

    logging.info("\n✨ 程序执行完成！")

