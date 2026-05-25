import os
import torch
import numpy as np
import pickle
import json
from PIL import Image
from termcolor import colored, cprint
from preprocess import clean_text
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import ElectraTokenizer, AutoModelForSequenceClassification, AutoTokenizer
from base_dataset import BaseDataset, expand2square
from paths import dataroot
from functools import lru_cache
import logging

# 任务标签映射
task_dict = {
    'task1': 'informative',
    'task2_full': 'humanitarian',
    'task2': 'humanitarian',
    'task3': 'damage'
}

# 标签映射字典
labels_task1 = {'informative': 1, 'not_informative': 0}
labels_task2_full = {
    'infrastructure_and_utility_damage': 0,
    'not_humanitarian': 1,
    'other_relevant_information': 2,
    'rescue_volunteering_or_donation_effort': 3,
    'vehicle_damage': 4,
    'affected_individuals': 5,
    'injured_or_dead_people': 6,
    'missing_or_found_people': 7,
}
labels_task2 = {
    'infrastructure_and_utility_damage': 0,
    'not_humanitarian': 1,
    'other_relevant_information': 2,
    'rescue_volunteering_or_donation_effort': 3,
    'vehicle_damage': 4,
    'affected_individuals': 5,
    'injured_or_dead_people': 5,
    'missing_or_found_people': 5,
}
labels_task3 = {'little_or_no_damage': 0, 'mild_damage': 1, 'severe_damage': 2}


class ProfessionalSentimentExtractor:
    """
    专业的情感特征提取器，使用预训练的Transformer模型
    """

    def __init__(self, model_name='/root/autodl-tmp/twitter-roberta-base-sentiment-latest',
                 device='cuda', max_length=128, use_cache=True):
        """
        Args:
            model_name: 预训练情感分析模型
                - 'cardiffnlp/twitter-roberta-base-sentiment-latest' (推荐，专门用于推特/社交媒体)
                - 'distilbert-base-uncased-finetuned-sst-2-english' (通用情感分析，较小较快)
                - 'nlptown/bert-base-multilingual-uncased-sentiment' (多语言支持)
            device: 运行设备 ('cuda' 或 'cpu')
            max_length: 最大文本长度
            use_cache: 是否使用缓存
        """
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.max_length = max_length
        self.use_cache = use_cache
        self.cache = {}
        self.stats = {'total': 0, 'cache_hits': 0}

        print(f"初始化专业情感分析模型: {model_name}")
        print(f"设备: {self.device}, 最大长度: {max_length}")

        try:
            # 尝试加载指定模型
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            # 移动到设备
            self.model = self.model.to(self.device)
            self.model.eval()  # 设置为评估模式

            # 获取模型信息
            self.num_labels = self.model.config.num_labels
            self.id2label = self.model.config.id2label
            self.label_names = [self.id2label[i] for i in range(self.num_labels)]

            print(f"✅ 情感模型加载成功!")
            print(f"   模型类别数: {self.num_labels}")
            print(f"   类别标签: {self.label_names}")
            print(f"   模型最大长度: {self.tokenizer.model_max_length}")

            # 设置合适的模型
            self.model_loaded = True

        except Exception as e:
            print(f"❌ 主模型加载失败: {e}")
            print("尝试加载备用模型...")
            self._load_fallback_model()

    def _load_fallback_model(self):
        """加载备用模型"""
        try:
            # 尝试加载较小的模型
            fallback_model = 'distilbert-base-uncased-finetuned-sst-2-english'
            print(f"加载备用模型: {fallback_model}")

            self.model = AutoModelForSequenceClassification.from_pretrained(fallback_model)
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            self.model = self.model.to(self.device)
            self.model.eval()

            self.num_labels = self.model.config.num_labels
            self.id2label = self.model.config.id2label
            self.label_names = [self.id2label[i] for i in range(self.num_labels)]

            print(f"✅ 备用模型加载成功!")
            print(f"   类别: {self.label_names}")
            self.model_loaded = True

        except Exception as e:
            print(f"❌ 备用模型也加载失败: {e}")
            self.model_loaded = False
            self.model = None
            self.tokenizer = None

    def _get_cache_key(self, text):
        """生成缓存键"""
        # 使用文本的简单hash作为缓存键
        import hashlib
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:16]
        return text_hash

    def _extract_sentiment_probs(self, text):
        """提取原始情感概率分布"""
        if not self.model_loaded:
            return None

        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

            # 移动到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 前向传播（禁用梯度计算）
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)

            return probabilities.cpu().squeeze(0)  # [num_labels]

        except Exception as e:
            print(f"情感分析失败: {e}, 文本: {text[:50]}...")
            return None

    def extract_features(self, text):
        """
        从文本提取3维情感特征

        Returns:
            torch.Tensor: [polarity, subjectivity, is_positive]
        """
        self.stats['total'] += 1

        # 检查缓存
        if self.use_cache:
            cache_key = self._get_cache_key(text)
            if cache_key in self.cache:
                self.stats['cache_hits'] += 1
                if self.stats['total'] % 1000 == 0:
                    hit_rate = self.stats['cache_hits'] / self.stats['total']
                    print(f"情感特征缓存命中率: {hit_rate:.2%}")
                return self.cache[cache_key]

        # 检查空文本
        if not text or not text.strip():
            features = torch.tensor([0.0, 0.5, 0.0], dtype=torch.float32)
        elif not self.model_loaded:
            # 模型未加载成功
            features = torch.tensor([0.0, 0.5, 0.0], dtype=torch.float32)
        else:
            # 提取情感概率
            probs = self._extract_sentiment_probs(text)

            if probs is None:
                # 提取失败
                features = torch.tensor([0.0, 0.5, 0.0], dtype=torch.float32)
            else:
                # 根据模型类型转换情感特征
                features = self._probs_to_features(probs)

        # 缓存结果
        if self.use_cache:
            cache_key = self._get_cache_key(text)
            self.cache[cache_key] = features

        return features

    def _probs_to_features(self, probs):
        """将概率分布转换为3维情感特征"""
        if self.num_labels == 2:
            # 二分类: [负面, 正面]
            negative = probs[0].item()
            positive = probs[1].item()
            polarity = positive - negative  # -1到1
            is_positive = 1.0 if positive > negative else 0.0
            subjectivity = max(positive, negative)  # 置信度作为主观性估计

        elif self.num_labels == 3:
            # 三分类: [负面, 中性, 正面]
            negative = probs[0].item()
            neutral = probs[1].item()
            positive = probs[2].item()
            polarity = positive - negative  # -1到1
            is_positive = 1.0 if positive > negative else 0.0
            subjectivity = 1.0 - neutral  # 中性概率越低，主观性越高

        elif self.num_labels == 5:
            # 五星评分: [1星, 2星, 3星, 4星, 5星]
            # 转换为情感极性
            star_weights = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=torch.float32)
            polarity = torch.sum(star_weights * probs).item()
            is_positive = 1.0 if polarity > 0 else 0.0
            subjectivity = torch.max(probs).item()

        else:
            # 通用处理：假设前半部分是负面，后半部分是正面
            half = self.num_labels // 2
            negative = torch.sum(probs[:half]).item()
            positive = torch.sum(probs[half:]).item()
            polarity = positive - negative
            is_positive = 1.0 if positive > negative else 0.0
            subjectivity = max(positive, negative)

        return torch.tensor([polarity, subjectivity, is_positive], dtype=torch.float32)

    def extract_batch(self, texts, batch_size=16):
        """批量提取情感特征"""
        if not texts:
            return []

        features = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            for text in batch_texts:
                feat = self.extract_features(text)
                features.append(feat)

        return features

    def get_detailed_analysis(self, text):
        """获取详细的情感分析结果"""
        if not self.model_loaded:
            return {
                'text': text[:100],
                'sentiment': 'unknown',
                'polarity': 0.0,
                'subjectivity': 0.5,
                'is_positive': 0.0,
                'confidence': 0.0,
                'probabilities': []
            }

        probs = self._extract_sentiment_probs(text)

        if probs is None:
            return {
                'text': text[:100],
                'sentiment': 'unknown',
                'polarity': 0.0,
                'subjectivity': 0.5,
                'is_positive': 0.0,
                'confidence': 0.0,
                'probabilities': []
            }

        # 获取预测结果
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()
        sentiment_label = self.id2label.get(pred_idx, f"LABEL_{pred_idx}")

        # 提取特征
        features = self._probs_to_features(probs)

        return {
            'text': text[:100] + '...' if len(text) > 100 else text,
            'sentiment': sentiment_label,
            'confidence': confidence,
            'polarity': features[0].item(),
            'subjectivity': features[1].item(),
            'is_positive': features[2].item(),
            'probabilities': probs.tolist(),
            'all_labels': self.label_names
        }

    def get_stats(self):
        """获取统计信息"""
        return {
            'total_processed': self.stats['total'],
            'cache_hits': self.stats['cache_hits'],
            'cache_size': len(self.cache),
            'cache_hit_rate': self.stats['cache_hits'] / max(1, self.stats['total']),
            'model_loaded': self.model_loaded,
            'num_labels': self.num_labels if self.model_loaded else 0
        }


class CrisisMMDataset(BaseDataset):
    def __init__(self):
        super(CrisisMMDataset, self).__init__()
        self.sentiment_extractor = None
        self.use_sentiment = False
        self.sentiment_cache = {}  # 数据集级别的缓存
        self.sentiment_stats = {'extracted': 0, 'cached': 0}
        # LLM 属性监督（多任务）
        self.aux_map = {}  # tweet_id -> {"aux_labels":[...], "aux_conf":[...]}
        self.aux_dim = 0
        self.use_llm_aux = False

    def initialize(self, opt, phase='train', cat='all', task='task2', shuffle=False,
                   consistent_only=False, setting='settingA'):
        self.opt = opt
        self.phase = phase
        self.task = task
        self.shuffle = shuffle
        self.consistent_only = consistent_only
        self.setting = setting

        # ===== LLM 属性监督（多任务）=====
        self.use_llm_aux = bool(getattr(opt, "use_llm_aux", False))
        self.aux_dim = int(getattr(opt, "aux_dim", 0) or 0)
        aux_path = getattr(opt, "aux_labels_path", "") or ""
        if self.use_llm_aux and aux_path and os.path.exists(aux_path):
            loaded = 0
            with open(aux_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    tid = str(obj.get("tweet_id", "")).strip()
                    aux_labels = obj.get("aux_labels", None)
                    aux_conf = obj.get("aux_conf", None)
                    if not tid or aux_labels is None or aux_conf is None:
                        continue
                    self.aux_map[tid] = {"aux_labels": aux_labels, "aux_conf": aux_conf}
                    loaded += 1
            print(f"✅ 已加载 LLM aux 标签: {loaded} 条 | aux_dim={self.aux_dim} | path={aux_path}")
        else:
            if self.use_llm_aux:
                print(f"⚠️ 启用了 LLM aux 但未找到文件: {aux_path}")

        # 检查是否启用情感特征
        self.use_sentiment = hasattr(opt, 'use_sentiment_features') and opt.use_sentiment_features

        if self.use_sentiment:
            print(f"\n{'=' * 50}")
            print("启用专业情感特征提取")
            print(f"{'=' * 50}")

            # 初始化专业情感提取器
            try:
                # 选择适合危机数据的模型
                # cardiffnlp/twitter-roberta-base-sentiment-latest 专门用于社交媒体数据
                model_name = getattr(opt, 'sentiment_model',
                                     '/root/autodl-tmp/twitter-roberta-base-sentiment-latest')

                self.sentiment_extractor = ProfessionalSentimentExtractor(
                    model_name=model_name,
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    max_length=128,
                    use_cache=True
                )

                if not self.sentiment_extractor.model_loaded:
                    print("⚠️  情感模型加载失败，将禁用情感特征")
                    self.use_sentiment = False
                else:
                    print(f"✅ 专业情感模型已加载: {model_name}")

            except Exception as e:
                print(f"❌ 情感提取器初始化失败: {e}")
                self.use_sentiment = False

        else:
            print(f"ℹ️  未启用情感特征提取")

        self.dataset_root = f'{dataroot}/CrisisMMD_v2.0'

        self.wiki_data_root = opt.wiki_data_root if hasattr(opt, 'wiki_data_root') else \
            r"/tmp/pycharm_project_376/CrisisKAN-main/output1/wiki_results"

        ann_file = os.path.join(
            self.wiki_data_root,
            f'task_{task_dict[task]}_text_img_{phase}_wiki_enhanced.pkl'
        )

        self.label_map = None
        if task == 'task1':
            self.label_map = labels_task1
        elif task == 'task2_full':
            self.label_map = labels_task2_full
        elif task == 'task2':
            self.label_map = labels_task2
        elif task == 'task3':
            self.label_map = labels_task3
        else:
            print(f"错误：未知任务 {task}")
            self.data_list = []
            return

        self.tokenizer = ElectraTokenizer.from_pretrained('/tmp/pycharm_project_376/electra-base-discriminator_model/google/electra-base-discriminator')

        print(f"\n📂 数据加载配置:")
        print(f"  任务: {task}")
        print(f"  阶段: {phase}")
        print(f"  使用情感特征: {self.use_sentiment}")
        print(f"  增强数据文件: {ann_file}")
        print(f"  文件是否存在: {os.path.exists(ann_file)}")

        self.read_data(ann_file)

        # 预计算所有文本的token
        print(f"\n计算文本tokens...")
        for i, data in enumerate(self.data_list):
            data['text_tokens'] = self.tokenize(data['enhanced_text'])

            # 显示进度
            if i % 100 == 0 and i > 0:
                print(f"  已处理 {i}/{len(self.data_list)} 个样本")

        if self.shuffle and self.data_list:
            np.random.default_rng(seed=0).shuffle(self.data_list)
        if self.data_list:
            self.data_list = self.data_list[:self.opt.max_dataset_size]
            cprint(f'\n[*] 成功加载 {len(self.data_list)} 个样本', 'yellow')
        else:
            cprint('[!] 未加载到任何样本，请检查增强数据路径', 'red')

        self.N = len(self.data_list)

        # 如果需要情感特征，预提取一部分以减少运行时开销
        if self.use_sentiment and self.data_list:
            print(f"\n预提取情感特征（前200个样本）...")
            pre_extract_count = min(200, len(self.data_list))

            for i in range(pre_extract_count):
                data = self.data_list[i]
                text = data['original_text']
                cache_key = f"{data['tweet_id']}_{data['image_id']}"

                # 提取情感特征
                features = self.sentiment_extractor.extract_features(text)
                self.sentiment_cache[cache_key] = features
                data['sentiment_features'] = features

                if i % 50 == 0 and i > 0:
                    print(f"  已预提取 {i}/{pre_extract_count} 个样本的情感特征")

            # 显示情感特征统计
            if pre_extract_count > 0:
                sample_feat = self.sentiment_cache[list(self.sentiment_cache.keys())[0]]
                print(f"  情感特征示例: {sample_feat.tolist()}")

        if self.use_sentiment:
            stats = self.sentiment_extractor.get_stats()
            print(f"\n📊 情感特征统计:")
            print(f"  模型加载: {'成功' if stats['model_loaded'] else '失败'}")
            print(f"  模型类别数: {stats['num_labels']}")
            print(f"  缓存大小: {stats['cache_size']}")
            print(f"  缓存命中率: {stats['cache_hit_rate']:.2%}")

        self.transforms = transforms.Compose([
            transforms.Lambda(lambda img: expand2square(img)),
            transforms.Resize((opt.load_size, opt.load_size)),
            *([transforms.RandomHorizontalFlip(0.2),
               transforms.RandomCrop((opt.crop_size, opt.crop_size))] if phase == 'train' else
              [transforms.CenterCrop((opt.crop_size, opt.crop_size))]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def read_data(self, ann_file):
        """读取数据（保持原有逻辑，添加情感特征字段）"""
        try:
            with open(ann_file, 'rb') as f:
                enhanced_texts = pickle.load(f)
        except FileNotFoundError:
            print(f"Error: 增强数据文件 {ann_file} 未找到")
            print(f"当前查找路径: {os.path.abspath(ann_file)}")
            self.data_list = []
            return
        except Exception as e:
            print(f"读取增强数据文件失败: {str(e)}")
            self.data_list = []
            return

        # 读取原始TSV文件
        original_tsv_path = os.path.join(
            dataroot, 'data_split', self.setting,
            f'task_{task_dict[self.task]}_text_img_{self.phase}.tsv'
        )
        try:
            with open(original_tsv_path, encoding='utf-8') as f:
                original_info = f.readlines()[1:]
        except FileNotFoundError:
            print(f"Error: 原始TSV文件 {original_tsv_path} 未找到")
            self.data_list = []
            return
        except Exception as e:
            print(f"读取原始TSV文件失败: {str(e)}")
            self.data_list = []
            return

        if len(enhanced_texts) != len(original_info):
            print(f"警告：增强文本数量({len(enhanced_texts)})与原始数据数量({len(original_info)})不匹配")
            self.data_list = []
            return

        self.data_list = []
        for idx, (l, enhanced_text) in enumerate(zip(original_info, enhanced_texts)):
            l = l.rstrip('\n')
            split_values = l.split('\t')
            if len(split_values) != 9:
                print(f"Error in line {idx + 2}: 预期9个字段，实际{len(split_values)}个. 内容: {l}")
                continue

            event_name, tweet_id, image_id, original_text, image, label, label_text, label_image, label_text_image = split_values

            if self.consistent_only and label_text != label_image:
                continue

            image_path = f'{self.dataset_root}/{image}' if image != '<unset>' else None
            if image_path and not os.path.exists(image_path):
                if idx < 10:  # 只显示前10个警告
                    print(f"警告：图像路径不存在 {image_path}")

            # 保存数据，添加情感特征占位符
            self.data_list.append({
                'path_image': image_path,
                'original_text': original_text,
                'enhanced_text': enhanced_text,
                'text_tokens': None,  # 将在initialize中填充
                'sentiment_features': None,  # 将在__getitem__中填充
                'label_str': label,
                'label': self.label_map[label],
                'label_image_str': label_image,
                'label_image': self.label_map.get(label_image, -1),
                'label_text_str': label_text,
                'label_text': self.label_map.get(label_text, -1),
                'event_name': event_name,
                'tweet_id': tweet_id,
                'image_id': image_id,
                'cache_key': f"{tweet_id}_{image_id}"  # 用于情感特征缓存
            })

    def tokenize(self, sentence):
        """文本分词"""
        try:
            if not sentence.strip():
                sentence = " "
            ids = self.tokenizer(clean_text(sentence),
                                 padding='max_length',
                                 max_length=512,
                                 truncation=True).items()
            return {k: torch.tensor(v) for k, v in ids}
        except Exception as e:
            print(f"分词失败: {str(e)}，句子内容: {sentence[:50]}...")
            return {
                'input_ids': torch.zeros(512, dtype=torch.long),
                'attention_mask': torch.zeros(512, dtype=torch.long)
            }

    def __getitem__(self, index):
        if index >= len(self.data_list):
            raise IndexError(f"索引 {index} 超出范围 (0-{len(self.data_list) - 1})")

        data = self.data_list[index]

        # 准备返回数据
        to_return = {
            'path_image': data['path_image'],
            'original_text': data['original_text'],
            'enhanced_text': data['enhanced_text'],
            'text_tokens': data['text_tokens'],
            'label_str': data['label_str'],
            'label': data['label'],
            'label_image_str': data['label_image_str'],
            'label_image': data['label_image'],
            'label_text_str': data['label_text_str'],
            'label_text': data['label_text'],
            'event_name': data['event_name'],
            'tweet_id': data['tweet_id'],
            'image_id': data['image_id']
        }

        # ===== LLM 属性监督字段 =====
        if self.use_llm_aux and self.aux_dim > 0:
            tid = str(data.get("tweet_id", "")).strip()
            rec = self.aux_map.get(tid, None)
            if rec is not None:
                lbl = rec.get("aux_labels", None)
                conf = rec.get("aux_conf", None)
                try:
                    aux_labels = torch.tensor(lbl, dtype=torch.float32)
                    aux_conf = torch.tensor(conf, dtype=torch.float32)
                except Exception:
                    aux_labels = torch.full((self.aux_dim,), -1.0, dtype=torch.float32)
                    aux_conf = torch.zeros((self.aux_dim,), dtype=torch.float32)
            else:
                aux_labels = torch.full((self.aux_dim,), -1.0, dtype=torch.float32)
                aux_conf = torch.zeros((self.aux_dim,), dtype=torch.float32)
            # 保证长度正确
            if aux_labels.numel() != self.aux_dim:
                aux_labels = torch.full((self.aux_dim,), -1.0, dtype=torch.float32)
            if aux_conf.numel() != self.aux_dim:
                aux_conf = torch.zeros((self.aux_dim,), dtype=torch.float32)
            to_return["aux_labels"] = aux_labels
            to_return["aux_conf"] = aux_conf

        # ========== 专业情感特征提取（核心部分） ==========
        if self.use_sentiment and self.sentiment_extractor is not None:
            cache_key = data['cache_key']

            # 检查数据集级别的缓存
            if cache_key in self.sentiment_cache:
                sentiment_features = self.sentiment_cache[cache_key]
                self.sentiment_stats['cached'] += 1
            else:
                # 使用专业模型提取情感特征
                text = data['original_text']
                sentiment_features = self.sentiment_extractor.extract_features(text)

                # 更新缓存
                self.sentiment_cache[cache_key] = sentiment_features
                data['sentiment_features'] = sentiment_features
                self.sentiment_stats['extracted'] += 1

            to_return['sentiment_features'] = sentiment_features

            # 调试输出：显示前几个样本的情感特征
            if index < 3 and self.opt.debug:
                print(f"\n📊 样本{index}情感分析:")
                print(f"  文本: {data['original_text'][:80]}...")
                print(f"  情感特征: {sentiment_features.tolist()}")

                # 获取详细分析
                detailed = self.sentiment_extractor.get_detailed_analysis(data['original_text'])
                print(f"  情感标签: {detailed['sentiment']}")
                print(f"  置信度: {detailed['confidence']:.3f}")
                print(f"  极性: {detailed['polarity']:.3f}")

        elif self.use_sentiment:
            # 情感提取器不可用
            to_return['sentiment_features'] = torch.zeros(3, dtype=torch.float32)
            if index == 0:
                print("⚠️  情感提取器不可用，使用零向量作为情感特征")
        else:
            # 未启用情感特征
            to_return['sentiment_features'] = torch.zeros(3, dtype=torch.float32)
        # ========== 情感特征提取结束 ==========

        # 处理图像
        if data['path_image'] and os.path.exists(data['path_image']):
            try:
                with Image.open(data['path_image']).convert('RGB') as img:
                    to_return['image'] = self.transforms(img)
            except Exception as e:
                print(f"⚠️ 处理图像 {data.get('path_image')} 时出错: {str(e)}")
                to_return['image'] = torch.zeros(3, self.opt.crop_size, self.opt.crop_size)
        else:
            to_return['image'] = torch.zeros(3, self.opt.crop_size, self.opt.crop_size)

        return to_return

    def __len__(self):
        return len(self.data_list)

    def name(self):
        return 'CrisisMMDataset'

    def get_sentiment_info(self):
        """获取情感特征相关信息"""
        if not self.use_sentiment or self.sentiment_extractor is None:
            return {
                'enabled': False,
                'model_loaded': False,
                'message': '情感特征未启用'
            }

        stats = self.sentiment_extractor.get_stats()
        stats.update({
            'enabled': True,
            'dataset_cache_size': len(self.sentiment_cache),
            'dataset_extracted': self.sentiment_stats['extracted'],
            'dataset_cached': self.sentiment_stats['cached'],
            'cache_hit_rate_dataset': self.sentiment_stats['cached'] / max(1, self.sentiment_stats['extracted'] +
                                                                           self.sentiment_stats['cached'])
        })

        # 如果有提取的特征，显示统计信息
        if self.sentiment_cache:
            all_features = torch.stack(list(self.sentiment_cache.values()))
            stats.update({
                'feature_mean': all_features.mean(dim=0).tolist(),
                'feature_std': all_features.std(dim=0).tolist(),
                'feature_range': [all_features.min().item(), all_features.max().item()]
            })

        return stats


if __name__ == '__main__':
    class Opt:
        debug = True
        max_dataset_size = 50  # 测试时只加载少量数据
        load_size = 256
        crop_size = 224
        use_sentiment_features = True  # 启用情感特征
        wiki_data_root = "./datasets"  # 根据实际情况修改


    opt = Opt()
    print("测试专业情感特征数据集...")
    print("=" * 60)

    dset = CrisisMMDataset()
    dset.initialize(opt, phase='train', task='task1', setting='settingA')

    print(f"\n数据集大小: {len(dset)}")

    if len(dset) > 0:
        # 测试情感特征提取
        print("\n测试情感特征提取:")
        for i in range(min(3, len(dset))):
            sample = dset[i]
            print(f"\n样本 {i}:")
            print(f"  原始文本: {sample['original_text'][:80]}...")
            print(f"  情感特征: {sample['sentiment_features'].tolist()}")

        # 获取情感特征统计
        sentiment_info = dset.get_sentiment_info()
        print(f"\n情感特征统计:")
        for key, value in sentiment_info.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            elif isinstance(value, list):
                print(f"  {key}: {[round(v, 4) for v in value]}")
            else:
                print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("测试完成!")
