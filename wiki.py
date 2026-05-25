import pandas as pd
import json
import requests
import tagme
import re
import wikipedia
import os
from tqdm import tqdm
import pickle
import sys

# --------------------------
# 1. 配置参数（修改这里适配你的路径）
1# 所有TSV文件所在的文件夹（替换为你的实际路径）
TSV_FOLDER = r"C:\Users\zyy\PycharmProjects\Classproject\CrisisKAN-main\datasets\data_split\settingB"  # 示例路径
# 输出pickle文件的保存文件夹（自动创建）
OUTPUT_FOLDER = r"C:\Users\zyy\PycharmProjects\Classproject\CrisisKAN-main\output\wiki_resultsB"
# TagMe Token（必须替换为你的有效Token，注册地址：https://tagme.d4science.org/tagme/）
TAGME_TOKEN = "6f14109b-c0b1-499b-83a8-0c153c2d0a3a-843339462"  # 示例："9876543210abcdef1234567890"


# --------------------------
# 2. 工具函数（文本预处理+维基增强）
# --------------------------
def preprocess(text):
    """文本预处理：移除链接、特殊符号、非ASCII字符"""
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"http[s]?://\S+", "", text)  # 移除链接
    text = re.sub(r"[:,\n\r;@#!]", "", text)  # 移除特殊符号
    text = re.sub(r"[^\x00-\x7F]+", "", text)  # 移除非ASCII字符
    return text.strip()


def get_wiki_enhanced_text(text):
    """获取维基百科增强文本（容错处理）"""
    if not text:
        return ""

    # 1. TagMe实体识别（提取高置信度实体）
    wiki_entities = []
    try:
        tagme.GCUBE_TOKEN = TAGME_TOKEN
        annotations = tagme.annotate(text)
        if annotations:
            # 过滤置信度>0.15的实体
            for ann in annotations.get_annotations(0.15):
                wiki_entities.append(ann.entity_title)
    except Exception as e:
        print(f"⚠️ TagMe识别警告：{str(e)}")

    # 2. 拼接维基百科摘要
    final_text = text
    for entity in wiki_entities:
        try:
            # 维基百科页面获取（超时10秒，避免卡住）
            page = wikipedia.page(entity, timeout=10)
            final_text += " " + page.summary  # 加空格避免文本粘连
        except wikipedia.exceptions.DisambiguationError as e:
            # 歧义页面（如"Apple"），取第一个候选
            if e.options:
                try:
                    page = wikipedia.page(e.options[0], timeout=10)
                    final_text += " " + page.summary
                except:
                    continue
        except (wikipedia.exceptions.PageError, TimeoutError, Exception) as e:
            # 页面不存在/超时/其他错误，跳过
            continue
    return final_text.strip()


# --------------------------
# 3. 批量处理核心逻辑
# --------------------------
def batch_process_tsv_folder(tsv_folder, output_folder):
    """批量处理文件夹内所有TSV文件"""
    # 检查TSV文件夹是否存在
    if not os.path.exists(tsv_folder):
        print(f"❌ 错误：TSV文件夹不存在 → {tsv_folder}")
        sys.exit(1)

    # 创建输出文件夹（不存在则自动创建）
    os.makedirs(output_folder, exist_ok=True)
    print(f"📂 输出文件夹：{output_folder}（自动创建）")

    # 遍历TSV文件夹内所有.tsv文件
    tsv_files = [f for f in os.listdir(tsv_folder) if f.endswith(".tsv")]
    if len(tsv_files) == 0:
        print(f"❌ 错误：TSV文件夹内无.tsv文件 → {tsv_folder}")
        sys.exit(1)

    print(f"\n✅ 找到 {len(tsv_files)} 个TSV文件，开始批量处理：")
    for idx, tsv_filename in enumerate(tsv_files, 1):
        tsv_path = os.path.join(tsv_folder, tsv_filename)
        print(f"\n=== 处理第{idx}/{len(tsv_files)}个文件：{tsv_filename} ===")

        # 1. 加载TSV文件（兼容UTF-8/GBK编码）
        try:
            df = pd.read_csv(tsv_path, sep="\t", encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(tsv_path, sep="\t", encoding="gbk")
        except Exception as e:
            print(f"❌ 跳过：加载{tsv_filename}失败 → {str(e)}")
            continue

        # 2. 校验必要列（确保有tweet_text和event_name）
        required_cols = ["tweet_text", "event_name"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ 跳过：{tsv_filename}缺少列 → {missing_cols}")
            continue

        # 3. 文本预处理+维基增强
        df["tweet_text"] = df["tweet_text"].apply(preprocess)
        df["combined_text"] = df["tweet_text"] + " " + df["event_name"]  # 拼接文本+事件名

        # 批量获取维基增强文本（带进度条）
        enhanced_texts = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="📄 增强文本生成"):
            try:
                enhanced_text = get_wiki_enhanced_text(row["combined_text"])
                enhanced_texts.append(enhanced_text)
            except Exception as e:
                print(f"⚠️ 单条数据警告：{str(e)}，用原始文本替代")
                enhanced_texts.append(row["combined_text"])

        # 4. 保存结果（输出文件名与TSV对应）
        output_filename = os.path.splitext(tsv_filename)[0] + "_wiki_enhanced.pkl"
        output_path = os.path.join(output_folder, output_filename)
        with open(output_path, "wb") as f:
            pickle.dump(enhanced_texts, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"✅ 完成：结果保存 → {output_filename}（共{len(enhanced_texts)}条）")

    print(f"\n🎉 批量处理结束！所有结果已保存到 → {output_folder}")


# --------------------------
# 4. 运行入口（无需传参，直接执行）
# --------------------------
if __name__ == "__main__":
    # 检查TagMe Token是否配置
    if TAGME_TOKEN == "<你的TagMe Token>":
        print("❌ 错误：请先配置TagMe Token！")
        print("步骤：1. 访问https://tagme.d4science.org/tagme/注册 → 2. 替换代码中的TAGME_TOKEN")
        sys.exit(1)

    # 启动批量处理
    batch_process_tsv_folder(TSV_FOLDER, OUTPUT_FOLDER)