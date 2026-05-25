
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from transformers import ElectraConfig, ElectraModel, ElectraTokenizer

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
except ImportError:  # pragma: no cover
    BlipProcessor = None
    BlipForConditionalGeneration = None

try:
    from diffusers import AutoPipelineForText2Image
except ImportError:  # pragma: no cover
    AutoPipelineForText2Image = None
import os
import torch.nn.functional as F
from torchvision.models import vit_b_16
import logging
import numpy as np
import json
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

# ========== 全局配置与日志初始化 ==========
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# BLIP 默认本地目录（与 args 一致；避免把 Salesforce/blip-... 当成相对路径拼到 cwd）
DEFAULT_BLIP_LOCAL_DIR = "/root/autodl-tmp/pycharm_project_376/blip-image-captioning-base"


def _looks_like_hf_model_id(name: str) -> bool:
    """如 Salesforce/blip-image-captioning-base — 不是本地相对路径，不能用 abspath 拼 cwd。"""
    s = (name or "").strip()
    if not s or s.startswith(("/", os.sep, "~", ".")):
        return False
    if s.count("/") != 1:
        return False
    a, b = s.split("/", 1)
    return bool(a) and bool(b) and ".." not in s


def _resolve_blip_load_path(caption_model_name: str) -> tuple[str, bool]:
    """
    返回 (传给 from_pretrained 的路径, 是否仅本地)。
    Hub 形式且默认本地目录存在时，自动改用 DEFAULT_BLIP_LOCAL_DIR。
    """
    raw = str(caption_model_name).strip()
    expanded = os.path.abspath(os.path.expanduser(raw))
    if os.path.isdir(expanded):
        return expanded, True
    if _looks_like_hf_model_id(raw) and os.path.isdir(DEFAULT_BLIP_LOCAL_DIR):
        logging.warning(
            f"caption_model_name 为 Hub 标识 ({raw})，服务器上无联网或目录不在 cwd 下；"
            f"自动改用本地目录: {DEFAULT_BLIP_LOCAL_DIR}"
        )
        return DEFAULT_BLIP_LOCAL_DIR, True
    return raw, False


# ========== 图像预处理（适配ViT-B/16） ==========
def get_vit_transform():
    """ViT-B/16 标准预处理流程"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ========== 第二步：基础模型类（所有模型的父类） ==========
class BaseModel(nn.Module):
    def __init__(self, save_dir):
        super(BaseModel, self).__init__()
        self.save_dir = save_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 确保保存目录存在
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        self.to(self.device)

    def save(self, filename):
        """保存模型权重"""
        state_dict = self.state_dict()
        save_path = os.path.join(self.save_dir, f"{filename}.pt")
        torch.save(state_dict, save_path)
        logging.info(f"✅ 模型权重已保存至 {save_path}")

    def load(self, filepath):
        """加载模型权重（兼容跨设备）"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型权重文件不存在：{filepath}")
        state_dict = torch.load(filepath, map_location=self.device)
        self.load_state_dict(state_dict, strict=False)
        logging.info(f"✅ 模型权重已从 {filepath} 加载")

    def load_pretrained_components(self, image_checkpoint_path=None, text_checkpoint_path=None):
        """加载预训练的图像/文本编码器权重（适配所有多模态模型）"""
        total_loaded = 0
        # 加载图像编码器权重
        if isinstance(self, ImageOnlyModel) and image_checkpoint_path and os.path.exists(image_checkpoint_path):
            self.load(image_checkpoint_path)
            total_loaded = len(self.state_dict())
            logging.info(f"单图像模型加载权重成功，参数数: {total_loaded}")
        # 加载文本编码器权重
        elif isinstance(self, TextOnlyModel) and text_checkpoint_path and os.path.exists(text_checkpoint_path):
            self.load(text_checkpoint_path)
            total_loaded = len(self.state_dict())
            logging.info(f"单文本模型加载权重成功，参数数: {total_loaded}")
        return total_loaded

    def freeze_components(self, freeze_image=False, freeze_text=False):
        """冻结编码器参数（用于微调）"""
        if isinstance(self, ImageOnlyModel) and freeze_image:
            for param in self.imageEncoder.parameters():
                param.requires_grad = False
            logging.info("已冻结单图像模型的编码器参数")
        elif isinstance(self, TextOnlyModel) and freeze_text:
            for param in self.textEncoder.parameters():
                param.requires_grad = False
            logging.info("已冻结单文本模型的编码器参数")

    def extract_feature(self, x, task_type='task2'):
        """
        新增：提取特征向量（适配事件抽取训练代码格式）
        返回：float32 一维numpy数组
        """
        self.eval()
        with torch.no_grad():
            # 特征提取模式：强制返回元组（logits, feat）
            logits, feat = self.forward(x, task_type=task_type, return_feat=True)
            # 转换为numpy数组（float32，一维）
            feat_np = feat.cpu().numpy().astype(np.float32).squeeze()
            return feat_np


# ========== 第三步：单模态模型（图像/文本，用于预训练） ==========
class MMModel(BaseModel):
    """多模态模型父类（定义统一接口）"""

    def __init__(self, imageEncoder, textEncoder, save_dir):
        super(MMModel, self).__init__(save_dir=save_dir)
        self.imageEncoder = imageEncoder.to(self.device)
        self.textEncoder = textEncoder.to(self.device)

    def forward(self, x):
        raise NotImplementedError("多模态模型需实现forward方法")


class TextOnlyModel(BaseModel):
    """单文本模型（Electra-Base）"""

    def __init__(self, save_dir, dim_text_repr=768, num_class=2):
        super(TextOnlyModel, self).__init__(save_dir)
        self.dropout = nn.Dropout()
        self.config = ElectraConfig()
        # 加载预训练Electra编码器
        self.textEncoder = ElectraModel.from_pretrained(
            '/root/autodl-tmp/pycharm_project_376/electra-base-discriminator_model/google/electra-base-discriminator'
        ).to(self.device)
        # 分类头
        self.linear = nn.Linear(dim_text_repr, num_class)

    def forward(self, x):
        """前向传播（处理文本输入）"""
        if isinstance(x, tuple):
            _, text = x
        else:
            text = x
        text = {k: v.to(self.device) for k, v in text.items()}

        # 提取文本特征
        hidden_states = self.textEncoder(**text)
        cls_feat = self.dropout(hidden_states[0][:, 0, :])  # [CLS] token特征
        return self.linear(cls_feat)


class ImageOnlyModel(BaseModel):
    """单图像模型（ViT-B/16）"""

    def __init__(self, save_dir, dim_visual_repr=768, num_class=2):
        super(ImageOnlyModel, self).__init__(save_dir)
        # 加载预训练ViT模型
        self.imageEncoder = vit_b_16(pretrained=True)
        # 替换分类头（保留768维特征输出）
        self.imageEncoder.heads = nn.Sequential(
            nn.Linear(self.imageEncoder.hidden_dim, dim_visual_repr)
        ).to(self.device)
        # 分类层
        self.flatten_vis = nn.Flatten()
        self.linear = nn.Linear(dim_visual_repr, num_class)
        self.dropout = nn.Dropout()

    def forward(self, x):
        """前向传播（处理图像输入）"""
        if isinstance(x, tuple):
            image, _ = x
        else:
            image = x
        image = image.to(self.device)

        # 提取图像特征
        img_feat = self.imageEncoder(image)
        flat_feat = self.dropout(self.flatten_vis(img_feat))
        return self.linear(flat_feat)


# ========== 第四步：论文核心IDEA双注意力模块（所有任务共享） ==========
class IDEAModule(nn.Module):
    """IDEA双注意力模块（同时捕捉和谐/冲突跨模态信息）"""

    def __init__(self, dim_text, dim_img, shared_dim=1024, dropout=0.3):
        super().__init__()
        # 模态投影到共享空间（消除模态差异）
        self.text_proj = nn.Sequential(
            nn.Linear(dim_text, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        self.img_proj = nn.Sequential(
            nn.Linear(dim_img, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        # 相似度权重层（修正维度：输入1维，输出共享维度）
        self.sim_weight = nn.Linear(1, shared_dim)
        # 论文最优温度系数
        self.temp_ham = 1.65  # 和谐注意力温度
        self.temp_cam = 0.75  # 冲突注意力温度
        # 投影回原维度（提前初始化，避免动态定义导致梯度异常）
        self.proj_back = nn.Linear(shared_dim, dim_text)

    def forward(self, text_feat, img_feat):
        """
        前向传播：输入[B, 1, 768]，输出融合和谐/冲突信息的特征[B, 1, 768]
        """
        B = text_feat.shape[0]
        # 1. 投影到共享1024维空间
        text_shared = self.text_proj(text_feat.squeeze(1)).unsqueeze(1)  # [B, 1, 1024]
        img_shared = self.img_proj(img_feat.squeeze(1)).unsqueeze(1)  # [B, 1, 1024]

        # 2. 计算跨模态相似度矩阵
        sim_matrix = torch.matmul(text_shared, img_shared.transpose(1, 2))  # [B, 1, 1]
        # 修正维度：展平→加权→恢复维度
        sim_matrix_flat = sim_matrix.squeeze(-1)  # [B, 1]
        sim_matrix_weighted = self.sim_weight(sim_matrix_flat).unsqueeze(-1)  # [B, 1024, 1]
        sim_matrix = sim_matrix_weighted[:, :1, :1]  # 恢复[B, 1, 1]，适配注意力计算

        # 3. 和谐注意力（捕捉模态对齐信息）
        ham_weights = F.softmax(sim_matrix / self.temp_ham, dim=-1)
        text_ham = torch.matmul(ham_weights, img_shared)  # 图像引导文本增强
        img_ham = torch.matmul(ham_weights.transpose(1, 2), text_shared)  # 文本引导图像增强

        # 4. 冲突注意力（捕捉模态矛盾信息）
        cam_weights = F.softmax(-sim_matrix / self.temp_cam, dim=-1)
        text_cam = torch.matmul(cam_weights, img_shared)
        img_cam = torch.matmul(cam_weights.transpose(1, 2), text_shared)

        # 5. 残差融合（投影回768维，避免梯度消失）
        text_ham = self.proj_back(text_ham)
        text_cam = self.proj_back(text_cam)
        img_ham = self.proj_back(img_ham)
        img_cam = self.proj_back(img_cam)

        # 6. 最终融合（原始特征+和谐特征+冲突特征）
        text_fused = text_feat + text_ham + text_cam
        img_fused = img_feat + img_ham + img_cam

        return text_fused, img_fused


# ========== 第五步：三任务全适配多模态核心模型（最终版） ==========
class CrossAttnMMModel(MMModel):
    """
    三任务通用多模态模型：
    Task1/2/3 在 IDEA 后共用同一套 cross-attn 主干（wiki、图生文 caption、文生图 patch、模态间融合）；
    差异仅在分类头与 Task3 的拼接顺序（patch/文本支路位置）。
    """

    def __init__(self, save_dir, dim_visual_repr=768, dim_text_repr=768,
                 task1_num_classes=2, task2_num_classes=6, task3_num_classes=3,
                 num_class=None, sentiment_weight=1.0, num_attention_layers=2,
                 wiki_knowledge_path=None, use_task_gate: bool = False, gate_hidden_dim: int = 256,
                 use_image_caption: bool = False,
                 caption_model_name: str = "/root/autodl-tmp/pycharm_project_376/blip-image-captioning-base",
                 freeze_caption_model: bool = True,
                 caption_max_length: int = 32,
                 use_text_to_image: bool = False,
                 t2i_model_name: str = "stabilityai/sd-turbo",
                 t2i_num_inference_steps: int = 1,
                 t2i_guidance_scale: float = 0.0,
                 save_text_to_image: bool = False,
                 t2i_save_dir: str = "./output/t2i_generated"):
        # 兼容旧参数：num_class映射到Task2
        if num_class is not None:
            task2_num_classes = num_class

        # 1. 初始化编码器（ViT+Electra，保留3维特征输出）
        vit_model = vit_b_16(pretrained=True)
        vit_model.heads = nn.Identity()  # 移除分类头，保留编码器
        self.tokenizer = ElectraTokenizer.from_pretrained(
            '/root/autodl-tmp/pycharm_project_376/electra-base-discriminator_model/google/electra-base-discriminator'
        )
        textEncoder = ElectraModel.from_pretrained(
            '/root/autodl-tmp/pycharm_project_376/electra-base-discriminator_model/google/electra-base-discriminator'
        )

        # 2. 父类初始化
        super().__init__(vit_model, textEncoder, save_dir)
        self.vit_original = self.imageEncoder

        # 3. 基础配置（所有任务共享）
        self.dim_visual_repr = dim_visual_repr
        self.dim_text_repr = dim_text_repr
        self.dropout = nn.Dropout(0.3)
        self.sentiment_weight = sentiment_weight
        self.num_attention_layers = num_attention_layers
        self.wiki_knowledge_path = wiki_knowledge_path
        self.use_task_gate = bool(use_task_gate)
        self.use_image_caption = bool(use_image_caption)
        self.caption_max_length = int(caption_max_length)
        self.caption_processor = None
        self.caption_generator = None
        self._caption_frozen = True
        if self.use_image_caption:
            if BlipProcessor is None or BlipForConditionalGeneration is None:
                raise ImportError("use_image_caption 需要 transformers 中的 BlipProcessor / BlipForConditionalGeneration")
            _cap_load, _cap_local = _resolve_blip_load_path(caption_model_name)
            _cap_kw = {"local_files_only": True} if _cap_local else {}
            if _cap_local:
                logging.info(f"BLIP 使用本地目录（不访问 HuggingFace Hub）: {_cap_load}")
                for _need in ("config.json", "preprocessor_config.json"):
                    _np = os.path.join(_cap_load, _need)
                    if not os.path.isfile(_np):
                        logging.warning(
                            f"本地 BLIP 目录缺少 {_need}，from_pretrained 可能失败；"
                            f"请从完整 HF 仓库复制该文件到 {_cap_load}"
                        )
            else:
                logging.warning(
                    f"BLIP 将尝试从 Hub 联网加载（请确认 caption_model_name 或准备 {DEFAULT_BLIP_LOCAL_DIR}）: {_cap_load}"
                )
            self.caption_processor = BlipProcessor.from_pretrained(_cap_load, **_cap_kw)
            self.caption_generator = BlipForConditionalGeneration.from_pretrained(
                _cap_load, **_cap_kw
            ).to(self.device)
            self._caption_frozen = bool(freeze_caption_model)
            if freeze_caption_model:
                for p in self.caption_generator.parameters():
                    p.requires_grad = False
                self.caption_generator.eval()
            logging.info(
                f"✅ 已启用图生文(use_image_caption) | caption_model={_cap_load} | "
                f"local_only={_cap_local} | frozen={self._caption_frozen} | max_len={self.caption_max_length} | "
                f"注入方式=IDEA 后 text<-caption cross-attn（输入侧不与推文融合）"
            )

        self.use_text_to_image = bool(use_text_to_image)
        self.t2i_pipe = None
        self.t2i_num_inference_steps = int(t2i_num_inference_steps)
        self.t2i_guidance_scale = float(t2i_guidance_scale)
        self.save_text_to_image = bool(save_text_to_image)
        self.t2i_save_dir = os.path.abspath(os.path.expanduser(str(t2i_save_dir)))
        self._t2i_save_counter = 0
        if self.save_text_to_image:
            os.makedirs(self.t2i_save_dir, exist_ok=True)
            logging.info(f"✅ 文生图保存已启用 | 目录: {self.t2i_save_dir}")
        if self.use_text_to_image:
            if AutoPipelineForText2Image is None:
                raise ImportError("use_text_to_image 需要安装 diffusers（及对应依赖）")
            _dt = torch.float16 if torch.cuda.is_available() else torch.float32
            self.t2i_pipe = AutoPipelineForText2Image.from_pretrained(
                t2i_model_name,
                torch_dtype=_dt,
            )
            self.t2i_pipe.to(self.device)
            self.t2i_pipe.set_progress_bar_config(disable=True)
            logging.info(
                f"✅ 已启用文生图(use_text_to_image) | t2i_model={t2i_model_name} | "
                f"steps={self.t2i_num_inference_steps} | guidance={self.t2i_guidance_scale} | dtype={_dt} | "
                f"注入方式=IDEA 后 img<-syn_patch cross-attn（输入侧不融合）"
            )

        _gd = dim_visual_repr + dim_text_repr
        self.task_gate_mlp = (
            nn.Sequential(
                nn.Linear(_gd, gate_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(gate_hidden_dim, 3),
            )
            if self.use_task_gate
            else None
        )
        self.task_config = {
            'task1': task1_num_classes,
            'task2': task2_num_classes,
            'task3': task3_num_classes
        }

        # 4. 情感特征门控模块（所有任务共享，注入情感信息）
        self.sent_proj_img = nn.Sequential(
            nn.Linear(6, dim_visual_repr),
            nn.LayerNorm(dim_visual_repr),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.sent_proj_text = nn.Sequential(
            nn.Linear(6, dim_text_repr),
            nn.LayerNorm(dim_text_repr),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.sent_gate_img = nn.Sequential(
            nn.Linear(dim_visual_repr * 2, dim_visual_repr),
            nn.Sigmoid()
        )
        self.sent_gate_text = nn.Sequential(
            nn.Linear(dim_text_repr * 2, dim_text_repr),
            nn.Sigmoid()
        )

        # 5. 论文核心模块（所有任务共享）
        self.idea_module = IDEAModule(dim_text=dim_text_repr, dim_img=dim_visual_repr, shared_dim=1024)

        # 6. 注意力模块（所有任务共享，自注意力+交叉注意力）
        self.img_self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=dim_visual_repr, num_heads=8, batch_first=True, dropout=0.2)
            for _ in range(num_attention_layers)
        ])
        self.text_self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=dim_text_repr, num_heads=8, batch_first=True, dropout=0.2)
            for _ in range(num_attention_layers)
        ])
        self.cross_attn_img2text_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=dim_text_repr, num_heads=8, batch_first=True, dropout=0.2)
            for _ in range(num_attention_layers)
        ])
        self.cross_attn_text2img_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=dim_visual_repr, num_heads=8, batch_first=True, dropout=0.2)
            for _ in range(num_attention_layers)
        ])
        # IDEA 后专用：图生文 / 文生图 patch 各一路，不与 wiki、模态间层共享权重
        self.post_idea_caption_attn = nn.MultiheadAttention(
            embed_dim=dim_text_repr, num_heads=8, batch_first=True, dropout=0.2
        )
        self.ln_post_idea_caption = nn.LayerNorm(dim_text_repr)
        self.post_idea_syn_attn = nn.MultiheadAttention(
            embed_dim=dim_visual_repr, num_heads=8, batch_first=True, dropout=0.2
        )
        self.ln_post_idea_syn_img = nn.LayerNorm(dim_visual_repr)
        # 文本侧双注入门控：分别控制 wiki 与 caption 对文本残差注入强度
        # 使用 sigmoid 约束到 (0,1)，避免两路语义硬叠加导致相互干扰
        self.text_wiki_gate = nn.Parameter(torch.tensor(0.5))
        self.text_caption_gate = nn.Parameter(torch.tensor(0.5))
        # 保存最近一次 forward 里提取到的 attention 可视化信息（按需返回，不影响训练）
        self._last_attn_info = None

        # 6.1 IDEA 前置注意力（先做模态内自注意力）
        self.pre_idea_img_self_attn = nn.MultiheadAttention(
            embed_dim=dim_visual_repr, num_heads=8, batch_first=True, dropout=0.2
        )
        self.pre_idea_text_self_attn = nn.MultiheadAttention(
            embed_dim=dim_text_repr, num_heads=8, batch_first=True, dropout=0.2
        )
        self.pre_idea_ln_img = nn.LayerNorm(dim_visual_repr)
        self.pre_idea_ln_text = nn.LayerNorm(dim_text_repr)

        # 7. 任务专属轻量模块（仅10%参数，无核心冗余）
        # Task3专属：Patch注意力+量化文本投影
        self.task3_img_patch_self_attn = nn.MultiheadAttention(
            embed_dim=dim_visual_repr, num_heads=8, batch_first=True, dropout=0.3
        )
        self.task3_patch_feat_proj = nn.Sequential(
            nn.Linear(dim_visual_repr, dim_visual_repr),
            nn.LayerNorm(dim_visual_repr),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.task3_quant_text_proj = nn.Sequential(
            nn.Linear(dim_text_repr, dim_text_repr),
            nn.LayerNorm(dim_text_repr),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Task2专属：Wikipedia知识注入投影
        self.task2_wiki_proj = nn.Sequential(
            nn.Linear(dim_text_repr * 2, dim_text_repr),
            nn.LayerNorm(dim_text_repr),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Task1/2/3 分类前端拼接维：全局两支 + patch + token（均为 768）→ 3072
        _fuse_4 = dim_visual_repr * 2 + dim_text_repr * 2

        # Task1：融合后接分类（与 Task3 同维，含 patch/token 支路）
        self.task1_global_fusion = nn.Sequential(
            nn.Linear(_fuse_4, _fuse_4),
            nn.LayerNorm(_fuse_4),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        # 8. 层归一化（动态计算维度，避免硬编码，解决维度不匹配）
        self.task1_norm_dim = _fuse_4
        self.task2_norm_dim = _fuse_4
        self.task3_norm_dim = _fuse_4

        self.ln_task1 = nn.LayerNorm(self.task1_norm_dim)
        self.ln_task2 = nn.LayerNorm(self.task2_norm_dim)
        self.ln_task3 = nn.LayerNorm(self.task3_norm_dim)
        self.ln_img = nn.ModuleList([nn.LayerNorm(dim_visual_repr) for _ in range(num_attention_layers)])
        self.ln_text = nn.ModuleList([nn.LayerNorm(dim_text_repr) for _ in range(num_attention_layers)])
        self.ln_global = nn.LayerNorm(dim_visual_repr + dim_text_repr)

        # 9. 任务专属分类头（适配各任务输出类别）
        self.task1_classifier = nn.Sequential(
            nn.Linear(self.task1_norm_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, task1_num_classes)
        )
        self.task2_classifier = nn.Sequential(
            nn.Linear(self.task2_norm_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, task2_num_classes)
        )
        self.task3_classifier = nn.Sequential(
            nn.Linear(self.task3_norm_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, task3_num_classes)
        )

    @staticmethod
    def _mean_or_zero(seq: torch.Tensor, dim: int, ref: torch.Tensor) -> torch.Tensor:
        """seq: [B,L,D]，L=0 时返回与 ref 同设备维度的零向量"""
        if seq is None or seq.size(dim) == 0:
            return torch.zeros(ref.shape[0], ref.shape[-1], device=ref.device, dtype=ref.dtype)
        return seq.mean(dim=dim)

    def _vit_forward(self, image_bchw: torch.Tensor) -> torch.Tensor:
        """ViT 主干：归一化图像张量 [B,3,224,224] → 编码器输出 [B,197,D]。"""
        vit_model = self.imageEncoder
        batch_size = image_bchw.size(0)
        x_vit = vit_model._process_input(image_bchw)
        batch_class_token = vit_model.class_token.expand(batch_size, -1, -1)
        vit_output = torch.cat([batch_class_token, x_vit], dim=1)
        if vit_model.encoder.pos_embedding.shape[1] != vit_output.shape[1]:
            pos_emb = F.interpolate(
                vit_model.encoder.pos_embedding.unsqueeze(0),
                size=(vit_output.shape[1], vit_output.shape[2]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        else:
            pos_emb = vit_model.encoder.pos_embedding
        vit_output = vit_output + pos_emb
        vit_output = vit_model.encoder(vit_output)
        if len(vit_output.shape) == 2:
            vit_output = vit_output.reshape(batch_size, 197, vit_model.hidden_dim)
        return vit_output

    def _text_prompts_from_ids(self, text: dict) -> list:
        """从 batch 的 input_ids 解码推文字符串，供文生图提示。"""
        ids = text["input_ids"]
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0
        out = []
        for i in range(ids.size(0)):
            row = ids[i]
            row = row[row != pad_id]
            s = self.tokenizer.decode(row, skip_special_tokens=True).strip()
            out.append(s if s else ".")
        return out

    def _pil_list_to_vit_batch(self, pil_list: list) -> torch.Tensor:
        """PIL 列表 → 与训练一致的 ImageNet 归一化张量 [B,3,224,224]。"""
        tfm = get_vit_transform()
        xs = []
        for pil in pil_list:
            xs.append(tfm(pil.convert("RGB")))
        return torch.stack(xs, dim=0).to(self.device)

    def _t2i_image_bchw_from_text(self, text: dict, tweet_ids: torch.Tensor | None = None) -> torch.Tensor:
        """推文文本 → diffusers 文生图 → 224 输入张量（生成过程不可导）。"""
        prompts = self._text_prompts_from_ids(text)
        pipe = self.t2i_pipe
        with torch.no_grad():
            result = pipe(
                prompt=prompts,
                num_inference_steps=self.t2i_num_inference_steps,
                guidance_scale=self.t2i_guidance_scale,
            )
        pils = result.images if hasattr(result, "images") else result
        if self.save_text_to_image:
            for i, pil in enumerate(pils):
                # 仅保留文件名安全字符，避免 Windows/Unix 路径非法字符
                prompt_stub = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in prompts[i][:40]).strip("_")
                if not prompt_stub:
                    prompt_stub = "empty_prompt"
                tweet_id_str = None
                if tweet_ids is not None and i < tweet_ids.size(0):
                    tid = int(tweet_ids[i].item())
                    if tid >= 0:
                        tweet_id_str = str(tid)
                if tweet_id_str:
                    fname = f"{tweet_id_str}_{prompt_stub}.png"
                else:
                    fname = f"{self._t2i_save_counter:08d}_{prompt_stub}.png"
                save_path = os.path.join(self.t2i_save_dir, fname)
                pil.save(save_path)
                self._t2i_save_counter += 1
        return self._pil_list_to_vit_batch(pils)

    def _vit_batch_to_pil_list(self, image: torch.Tensor):
        """ViT 输入 [B,3,H,W]（ImageNet 归一化）→ PIL 列表，供 BLIP processor 使用。"""
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        x = (image * std + mean).clamp(0.0, 1.0)
        out = []
        for i in range(x.size(0)):
            arr = (x[i].detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            out.append(Image.fromarray(arr))
        return out

    def _caption_cls_from_images(self, image: torch.Tensor) -> torch.Tensor:
        """
        BLIP 生成描述 → 同一 ELECTRA 编码 → [B, dim_text_repr]（图生文语义向量）。
        生成与 ELECTRA 前向在 no_grad 中；与推文融合仅在 IDEA 后经 post_idea_caption_attn 完成。
        """
        pil_list = self._vit_batch_to_pil_list(image)
        proc = self.caption_processor
        gen = self.caption_generator
        with torch.no_grad():
            inputs = proc(images=pil_list, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            out_ids = gen.generate(
                pixel_values=inputs["pixel_values"],
                max_length=self.caption_max_length,
            )
        captions = proc.batch_decode(out_ids, skip_special_tokens=True)
        cap_tokens = self.tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        cap_tokens = {k: v.to(self.device) for k, v in cap_tokens.items()}
        cap_hidden = self.textEncoder(**cap_tokens)[0]
        return cap_hidden[:, 0, :]

    def train(self, mode: bool = True):
        super().train(mode)
        if self.use_image_caption and self.caption_generator is not None:
            if getattr(self, "_caption_frozen", True):
                self.caption_generator.eval()
            else:
                self.caption_generator.train(mode)
        if getattr(self, "t2i_pipe", None) is not None:
            try:
                self.t2i_pipe.eval()
            except Exception:
                pass
        return self

    def simple_graph_aggregation(self, feat, temperature=0.1):
        """图聚合模块（所有任务共享），缓解样本分布不均，强化同类样本特征关联"""
        B = feat.shape[0]
        # 计算余弦相似度矩阵
        sim_matrix = F.cosine_similarity(feat.unsqueeze(1), feat.unsqueeze(0), dim=-1)
        # 阈值过滤（仅保留高相似样本关联）
        sim_matrix = torch.where(sim_matrix > 0.75, sim_matrix, torch.tensor(-1e9).to(self.device))
        # 注意力加权聚合
        attn_weights = F.softmax(sim_matrix / temperature, dim=-1)
        aggregated_feat = feat + torch.matmul(attn_weights, feat)
        return aggregated_feat.unsqueeze(1)

    def _prepare_wiki_feat(self, batch_size: int, text_cls_feat: torch.Tensor) -> torch.Tensor:
        if self.wiki_knowledge_path and os.path.exists(self.wiki_knowledge_path):
            return torch.load(self.wiki_knowledge_path, map_location=self.device)[:batch_size].unsqueeze(1)
        return text_cls_feat

    def _compute_task1_logits(
        self,
        img_feat_idea: torch.Tensor,
        text_feat_idea: torch.Tensor,
        wiki_feat: torch.Tensor,
        img_patch_feat: torch.Tensor,
        text_token_feat: torch.Tensor,
        syn_img_patch_feat: torch.Tensor | None = None,
        caption_feat: torch.Tensor | None = None,
        return_attn_weights: bool = False,
    ) -> torch.Tensor:
        # Task1 与 Task2/3 共用 IDEA 后统一主干（wiki / caption / 文生图 patch + 跨模态）
        img_agg, text_cross_agg = self._compute_unified_post_idea_feats(
            img_feat_idea=img_feat_idea,
            text_feat_idea=text_feat_idea,
            wiki_feat=wiki_feat,
            syn_img_patch_feat=syn_img_patch_feat,
            caption_feat=caption_feat,
            return_attn_weights=return_attn_weights,
        )

        patch_self, _ = self.task3_img_patch_self_attn(img_patch_feat, img_patch_feat, img_patch_feat)
        tok = self._mean_or_zero(text_token_feat, dim=1, ref=img_feat_idea.squeeze(1))
        patch_feat = self.task3_patch_feat_proj(patch_self.mean(dim=1))
        token_feat = self.task3_quant_text_proj(tok)

        fused_feat = torch.cat([img_agg, text_cross_agg, patch_feat, token_feat], dim=-1)
        fused_feat = self.ln_task1(fused_feat)
        return self.task1_classifier(fused_feat)

    def _compute_task2_logits(
        self,
        img_feat_idea: torch.Tensor,
        text_feat_idea: torch.Tensor,
        wiki_feat: torch.Tensor,
        img_patch_feat: torch.Tensor,
        text_token_feat: torch.Tensor,
        syn_img_patch_feat: torch.Tensor | None = None,
        caption_feat: torch.Tensor | None = None,
        return_attn_weights: bool = False,
    ) -> torch.Tensor:
        img_agg, text_cross_agg = self._compute_unified_post_idea_feats(
            img_feat_idea=img_feat_idea,
            text_feat_idea=text_feat_idea,
            wiki_feat=wiki_feat,
            syn_img_patch_feat=syn_img_patch_feat,
            caption_feat=caption_feat,
            return_attn_weights=return_attn_weights,
        )

        patch_self, _ = self.task3_img_patch_self_attn(img_patch_feat, img_patch_feat, img_patch_feat)
        tok = self._mean_or_zero(text_token_feat, dim=1, ref=img_feat_idea.squeeze(1))
        patch_feat = self.task3_patch_feat_proj(patch_self.mean(dim=1))
        token_feat = self.task3_quant_text_proj(tok)

        fused_feat = torch.cat([img_agg, text_cross_agg, patch_feat, token_feat], dim=-1)
        fused_feat = self.ln_task2(fused_feat)
        return self.task2_classifier(fused_feat)

    def _compute_task3_logits(
        self,
        img_feat_idea: torch.Tensor,
        text_feat_idea: torch.Tensor,
        wiki_feat: torch.Tensor,
        img_patch_feat: torch.Tensor,
        text_token_feat: torch.Tensor,
        syn_img_patch_feat: torch.Tensor | None = None,
        caption_feat: torch.Tensor | None = None,
        return_attn_weights: bool = False,
    ) -> torch.Tensor:
        img_agg, text_agg = self._compute_unified_post_idea_feats(
            img_feat_idea=img_feat_idea,
            text_feat_idea=text_feat_idea,
            wiki_feat=wiki_feat,
            syn_img_patch_feat=syn_img_patch_feat,
            caption_feat=caption_feat,
            return_attn_weights=return_attn_weights,
        )
        patch_self, _ = self.task3_img_patch_self_attn(img_patch_feat, img_patch_feat, img_patch_feat)
        patch_feat = self.task3_patch_feat_proj(patch_self.mean(dim=1))
        quant_tok = self._mean_or_zero(text_token_feat, dim=1, ref=img_feat_idea.squeeze(1))
        quant_text_feat = self.task3_quant_text_proj(quant_tok)
        fused_feat = torch.cat([img_agg, patch_feat, text_agg, quant_text_feat], dim=-1)
        fused_feat = self.ln_task3(fused_feat)
        return self.task3_classifier(fused_feat)

    def _compute_unified_post_idea_feats(
        self,
        img_feat_idea: torch.Tensor,
        text_feat_idea: torch.Tensor,
        wiki_feat: torch.Tensor,
        syn_img_patch_feat: torch.Tensor | None = None,
        caption_feat: torch.Tensor | None = None,
        return_attn_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        IDEA 后统一主干（交叉注意力）：
        1) 文本侧增强：text <- wiki，再 text <- caption（图生文 [B,1,D]，与 wiki 同机制）
        2) 图像侧增强：img <- text_strengthened，再 img <- syn_patch（文生图 patch，可选）
        3) 跨模态：text <-> img
        4) 双侧图聚合
        输入侧不与推文/真实图 CLS 融合 caption 或文生图。
        """
        idx_intra = 0
        idx_inter = 1 if len(self.cross_attn_img2text_layers) > 1 else 0

        # -------- 1a) text <- wiki（带可学习门控）--------
        if return_attn_weights:
            text_w_delta, attn_wiki = self.cross_attn_img2text_layers[idx_intra](
                text_feat_idea,
                wiki_feat,
                wiki_feat,
                need_weights=True,
                average_attn_weights=False,
            )
        else:
            text_w_delta, _ = self.cross_attn_img2text_layers[idx_intra](text_feat_idea, wiki_feat, wiki_feat)
        g_wiki = torch.sigmoid(self.text_wiki_gate)
        text_w = self.ln_text[idx_intra](text_feat_idea + g_wiki * text_w_delta)

        # -------- 1b) text <- caption（IDEA 后；无 BLIP 时跳过，带可学习门控）--------
        if caption_feat is not None:
            if return_attn_weights:
                text_c_delta, attn_caption = self.post_idea_caption_attn(
                    text_w,
                    caption_feat,
                    caption_feat,
                    need_weights=True,
                    average_attn_weights=False,
                )
            else:
                text_c_delta, _ = self.post_idea_caption_attn(text_w, caption_feat, caption_feat)
            g_caption = torch.sigmoid(self.text_caption_gate)
            text_strengthened = self.ln_post_idea_caption(text_w + g_caption * text_c_delta)
        else:
            text_strengthened = text_w
            if return_attn_weights:
                attn_caption = None
                g_caption = None

        if return_attn_weights:
            # attn_* 形状（average_attn_weights=False）：(B, num_heads, tgt_len, src_len)
            # 当前实现里 tgt_len=1, src_len=1（CLS-CLS），热力图会是一点/每个head一条
            self._last_attn_info = {
                "g_wiki": float(g_wiki.detach().cpu().item()),
                "g_caption": None if g_caption is None else float(g_caption.detach().cpu().item()),
                "attn_wiki": attn_wiki.detach().cpu(),
                "attn_caption": None if attn_caption is None else attn_caption.detach().cpu(),
            }
        else:
            self._last_attn_info = None

        # -------- 2a) img <- text_strengthened --------
        img_w, _ = self.cross_attn_text2img_layers[idx_intra](img_feat_idea, text_strengthened, text_strengthened)
        img_w = self.ln_img[idx_intra](img_w + img_feat_idea)

        # -------- 2b) img <- 文生图 patch（IDEA 后；未启用 T2I 时跳过）--------
        if syn_img_patch_feat is not None:
            img_s, _ = self.post_idea_syn_attn(img_w, syn_img_patch_feat, syn_img_patch_feat)
            img_strengthened = self.ln_post_idea_syn_img(img_s + img_w)
        else:
            img_strengthened = img_w

        # -------- 3) 跨模态融合：text <-> img --------
        text_2, _ = self.cross_attn_img2text_layers[idx_inter](text_strengthened, img_strengthened, img_strengthened)
        text_2 = self.ln_text[idx_inter](text_2 + text_strengthened)

        img_2, _ = self.cross_attn_text2img_layers[idx_inter](img_strengthened, text_strengthened, text_strengthened)
        img_2 = self.ln_img[idx_inter](img_2 + img_strengthened)

        text_agg = self.simple_graph_aggregation(text_2.squeeze(1)).squeeze(1)
        img_agg = self.simple_graph_aggregation(img_2.squeeze(1)).squeeze(1)
        return img_agg, text_agg

    def forward(
        self,
        x,
        task_type=None,
        return_contrast_feat=False,
        return_feat=False,
        return_attn_weights: bool = False,
        return_gate_values: bool = False,
    ):
        """
        核心前向传播：支持三任务动态切换，输入(image, text, sentiment_features)
        参数新增：
        - return_feat: 是否返回特征向量（训练时False，特征提取时True）
        - return_contrast_feat: 是否返回对比学习特征
        """
        # 任务类型校验与默认值
        if task_type is None:
            task_type = 'task2'
        if task_type not in self.task_config:
            raise ValueError(f"无效任务类型：{task_type}，支持：task1/task2/task3")

        # 解析输入
        if isinstance(x, tuple) and len(x) == 4:
            image, text, sentiment_features, tweet_ids = x
        else:
            image, text, sentiment_features = x
            tweet_ids = None
        image = image.to(self.device)
        text = {k: v.to(self.device) for k, v in text.items()}
        sentiment_features = sentiment_features.to(self.device)
        batch_size = image.size(0)

        # ========== 1. 基础特征提取（所有任务共享，ViT强制3维输出） ==========
        self._last_attn_info = None
        self._last_gate_info = None
        vit_output = self._vit_forward(image)
        # IDEA 前：图像模态内自注意力（token-level）
        img_pre_self, _ = self.pre_idea_img_self_attn(vit_output, vit_output, vit_output)
        vit_output = self.pre_idea_ln_img(vit_output + img_pre_self)
        # 提取图像特征（全局+Patch）
        img_cls_feat = vit_output[:, 0, :].unsqueeze(1)  # [B, 1, 768]（全局特征）
        img_patch_feat = vit_output[:, 1:, :]  # [B, 196, 768]（局部Patch特征，Task3专用）

        # 可选：推文 → 文生图 → ViT patch，仅用于 IDEA 后 img<-syn cross-attn（输入侧不融合 CLS）
        syn_img_patch_feat = None
        if self.use_text_to_image and self.t2i_pipe is not None:
            gen_bchw = self._t2i_image_bchw_from_text(text, tweet_ids=tweet_ids)
            vit_syn = self._vit_forward(gen_bchw)
            syn_img_patch_feat = vit_syn[:, 1:, :]  # [B, 196, 768]

        # ========== 2. 文本特征提取 ==========
        text_hidden = self.textEncoder(**text)[0]  # [B, seq_len, 768]
        # IDEA 前：文本模态内自注意力（token-level）
        text_pre_self, _ = self.pre_idea_text_self_attn(text_hidden, text_hidden, text_hidden)
        text_hidden = self.pre_idea_ln_text(text_hidden + text_pre_self)
        text_cls_feat = text_hidden[:, 0, :].unsqueeze(1)  # [B, 1, 768]（全局特征）
        text_token_feat = text_hidden[:, 1:, :]  # [B, seq_len-1, 768]（token特征，Task3专用）

        # 图生文：仅算 caption 向量，IDEA 后经 text<-caption cross-attn 注入（此处不与 text_cls 融合）
        caption_feat = None
        if self.use_image_caption:
            caption_feat = self._caption_cls_from_images(image).unsqueeze(1)  # [B, 1, 768]

        # ========== 3. 情感特征门控融合（所有任务共享） ==========
        sent_feat_img = self.sent_proj_img(sentiment_features).unsqueeze(1)
        sent_feat_text = self.sent_proj_text(sentiment_features).unsqueeze(1)

        img_gate = self.sent_gate_img(torch.cat([img_cls_feat.squeeze(1), sent_feat_img.squeeze(1)], dim=-1)).unsqueeze(
            1)
        text_gate = self.sent_gate_text(
            torch.cat([text_cls_feat.squeeze(1), sent_feat_text.squeeze(1)], dim=-1)).unsqueeze(1)

        if return_gate_values:
            # 把逐维 gate(768维)压缩为每个样本一个标量（便于画图）
            img_gate_mean = img_gate.detach().mean(dim=-1).squeeze(1).cpu()  # [B]
            text_gate_mean = text_gate.detach().mean(dim=-1).squeeze(1).cpu()  # [B]
            self._last_gate_info = {
                "img_gate_mean": img_gate_mean,
                "text_gate_mean": text_gate_mean,
            }

        img_cls_feat = img_cls_feat * (1 - img_gate) + sent_feat_img * img_gate
        text_cls_feat = text_cls_feat * (1 - text_gate) + sent_feat_text * text_gate

        # ========== 4. IDEA双注意力融合（所有任务共享论文核心优化） ==========
        text_feat_idea, img_feat_idea = self.idea_module(text_cls_feat, img_cls_feat)

        # ========== 5. 任务专属轻量逻辑（无缝切换，无核心冗余） ==========
        self._last_task_gate_logits = None
        wiki_feat = self._prepare_wiki_feat(batch_size, text_cls_feat)
        if self.use_task_gate and not return_contrast_feat and not return_feat:
            l1 = self._compute_task1_logits(
                img_feat_idea,
                text_feat_idea,
                wiki_feat,
                img_patch_feat,
                text_token_feat,
                syn_img_patch_feat,
                caption_feat,
                    return_attn_weights=return_attn_weights,
            )
            l2 = self._compute_task2_logits(
                img_feat_idea,
                text_feat_idea,
                wiki_feat,
                img_patch_feat,
                text_token_feat,
                syn_img_patch_feat,
                caption_feat,
                    return_attn_weights=return_attn_weights,
            )
            l3 = self._compute_task3_logits(
                img_feat_idea,
                text_feat_idea,
                wiki_feat,
                img_patch_feat,
                text_token_feat,
                syn_img_patch_feat,
                caption_feat,
                    return_attn_weights=return_attn_weights,
            )
            gate_in = torch.cat([img_feat_idea.squeeze(1), text_feat_idea.squeeze(1)], dim=-1)
            self._last_task_gate_logits = self.task_gate_mlp(gate_in)
            if task_type == 'task1':
                event_logits = l1
            elif task_type == 'task2':
                event_logits = l2
            elif task_type == 'task3':
                event_logits = l3
            else:
                raise ValueError(f"无效任务类型：{task_type}，支持：task1/task2/task3")
        elif task_type == 'task1':
            event_logits = self._compute_task1_logits(
                img_feat_idea,
                text_feat_idea,
                wiki_feat,
                img_patch_feat,
                text_token_feat,
                syn_img_patch_feat,
                caption_feat,
                    return_attn_weights=return_attn_weights,
            )

        elif task_type == 'task2':
            event_logits = self._compute_task2_logits(
                img_feat_idea,
                text_feat_idea,
                wiki_feat,
                img_patch_feat,
                text_token_feat,
                syn_img_patch_feat,
                caption_feat,
                    return_attn_weights=return_attn_weights,
            )

        elif task_type == 'task3':
            event_logits = self._compute_task3_logits(
                img_feat_idea,
                text_feat_idea,
                wiki_feat,
                img_patch_feat,
                text_token_feat,
                syn_img_patch_feat,
                caption_feat,
                    return_attn_weights=return_attn_weights,
            )

        # ========== 6. 返回逻辑（核心修复） ==========
        # 对比学习模式：返回logits + 对比特征
        if return_contrast_feat:
            self.contrast_proj_img = nn.Sequential(
                nn.Linear(self.dim_visual_repr, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            ).to(self.device)
            self.contrast_proj_text = nn.Sequential(
                nn.Linear(self.dim_text_repr, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            ).to(self.device)

            img_contrast_feat = F.normalize(self.contrast_proj_img(img_cls_feat.squeeze(1)), dim=-1)
            text_contrast_feat = F.normalize(self.contrast_proj_text(text_cls_feat.squeeze(1)), dim=-1)
            return event_logits, img_contrast_feat, text_contrast_feat
        # 特征提取模式：返回logits + 融合特征
        elif return_feat:
            return event_logits, fused_feat
        # 训练模式（默认）：仅返回logits（损失函数需要的单个张量）
        else:
            if return_attn_weights or return_gate_values:
                out = {}
                if return_attn_weights:
                    out["attn"] = self._last_attn_info
                if return_gate_values:
                    out["gates"] = self._last_gate_info
                return event_logits, out
            return event_logits

    def load_pretrained_components(self, image_checkpoint_path=None, text_checkpoint_path=None):
        """重载预训练组件加载逻辑，适配多模态模型"""
        total_loaded = 0
        # 加载图像编码器权重
        if image_checkpoint_path and os.path.exists(image_checkpoint_path):
            try:
                checkpoint = torch.load(image_checkpoint_path, map_location=self.device)
                img_ckpt = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
                img_encoder_params = {}
                for k, v in img_ckpt.items():
                    if k.startswith('imageEncoder.'):
                        img_encoder_params[k.replace('imageEncoder.', '')] = v
                    elif k.startswith('heads.') or k.startswith('features.'):
                        img_encoder_params[k] = v
                self.imageEncoder.load_state_dict(img_encoder_params, strict=False)
                loaded_num = len(img_encoder_params)
                total_loaded += loaded_num
                logging.info(f"✅ 加载图像编码器参数 {loaded_num} 个")
            except Exception as e:
                raise RuntimeError(f"加载图像编码器失败: {str(e)}")

        # 加载文本编码器权重
        if text_checkpoint_path and os.path.exists(text_checkpoint_path):
            try:
                checkpoint = torch.load(text_checkpoint_path, map_location=self.device)
                text_ckpt = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
                text_encoder_params = {}
                for k, v in text_ckpt.items():
                    if k.startswith('textEncoder.'):
                        text_encoder_params[k.replace('textEncoder.', '')] = v
                    elif k.startswith('embeddings.') or k.startswith('encoder.'):
                        text_encoder_params[k] = v
                self.textEncoder.load_state_dict(text_encoder_params, strict=False)
                loaded_num = len(text_encoder_params)
                total_loaded += loaded_num
                logging.info(f"✅ 加载文本编码器参数 {loaded_num} 个")
            except Exception as e:
                raise RuntimeError(f"加载文本编码器失败: {str(e)}")

        return total_loaded


# ========== 特征提取工具函数（核心新增） ==========
def extract_features_for_event_extraction(
        model,
        text_list,
        image_path_list,
        sentiment_features_list,
        task_type='task2',
        feature_save_dir='./features',
        output_json_path='./event_extraction_data.json'
):
    """
    提取多模态特征并生成事件抽取训练所需的JSON数据
    Args:
        model: 初始化好的CrossAttnMMModel
        text_list: 文本列表
        image_path_list: 图像路径列表
        sentiment_features_list: 情感特征列表（每个样本6维）
        task_type: 任务类型（task1/task2/task3）
        feature_save_dir: 特征保存目录
        output_json_path: 输出JSON文件路径
    """
    # 创建特征保存目录
    os.makedirs(feature_save_dir, exist_ok=True)

    # 图像预处理
    transform = get_vit_transform()

    # 结果列表
    output_data = []

    # 批量提取特征
    for idx, (text, img_path, sent_feat) in enumerate(tqdm(zip(text_list, image_path_list, sentiment_features_list))):
        # 1. 处理文本
        text_encoding = model.tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=512
        )

        # 2. 处理图像
        image = Image.open(img_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)

        # 3. 处理情感特征
        sent_feat_tensor = torch.tensor(sent_feat, dtype=torch.float32).unsqueeze(0)

        # 4. 提取特征（指定return_feat=True）
        feat_np = model.extract_feature(
            (image_tensor, text_encoding, sent_feat_tensor),
            task_type=task_type
        )

        # 5. 保存特征为.npy文件
        feat_save_path = os.path.join(feature_save_dir, f"feat_{idx}.npy")
        np.save(feat_save_path, feat_np)

        # 6. 构造JSON条目（适配事件抽取训练格式）
        output_data.append({
            "text": text,
            "event": "",  # 需手动填充事件标注结果
            "feature_path": os.path.abspath(feat_save_path)
        })

    # 7. 保存JSON文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        for entry in output_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

    logging.info(f"✅ 特征提取完成！")
    logging.info(f"- 特征文件保存至：{feature_save_dir}")
    logging.info(f"- JSON数据保存至：{output_json_path}")


# ========== 模型别名（保持与原有代码兼容） ==========
ViTElectraMMModel = CrossAttnMMModel

