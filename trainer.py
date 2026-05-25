
import os
import json
import time
import csv
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import f1_score


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)


def _f1_filtered(labels, predictions, average: str) -> float:
    """Skip ignore_index=-1 (and any negative) when computing F1."""
    lab = np.asarray(labels, dtype=np.int64)
    prd = np.asarray(predictions, dtype=np.int64)
    m = lab >= 0
    if not m.any() or len(np.unique(lab[m])) < 2:
        return 0.0
    return float(f1_score(lab[m], prd[m], average=average))


class Trainer:
    def __init__(
        self,
        train_loader,
        dev_loader,
        test_loader,
        model: nn.Module,
        loss_fn,
        optimizer,
        scheduler,
        save_dir=".",
        display=100,
        eval=False,
        device="cuda",
        tensorboard=False,
        mode="both",
        grad_accumulation_steps=1,
        task_type="task2",
        joint_mode: str = "none",
        lambda_joint_a: float = 1.0,
        lambda_joint_b: float = 1.0,
        log_hard_examples: bool = False,
        hard_examples_output: str = "./hard_examples_task2.jsonl",
        # ======= LLM 蒸馏相关 =======
        use_llm_distill: bool = False,
        llm_review_path: Optional[str] = None,
        llm_lambda: float = 0.3,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.display = display
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.eval = eval
        self.device = device
        self.tensorboard = tensorboard
        self.mode = mode
        self.grad_accumulation_steps = grad_accumulation_steps
        self.task_type = task_type
        # 与 main.py 参数对齐：即使当前不启用 joint 训练也保留这些属性
        self.joint_mode = joint_mode
        self.lambda_joint_a = float(lambda_joint_a)
        self.lambda_joint_b = float(lambda_joint_b)

        # 难样本导出
        self.log_hard_examples = log_hard_examples
        self.hard_examples_output = hard_examples_output

        # ======= LLM 蒸馏 =======
        self.use_llm_distill = use_llm_distill
        self.llm_lambda = float(llm_lambda)
        self.llm_map: Dict[str, Tuple[int, float]] = {}  # tweet_id -> (llm_label_id, confidence)

        if self.use_llm_distill and llm_review_path and os.path.exists(llm_review_path):
            logging.info(f"📥 加载 LLM 审阅结果用于蒸馏: {llm_review_path}")
            loaded = 0
            skipped = 0
            with open(llm_review_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        skipped += 1
                        continue

                    # 格式 A（旧）: {"original": {...}, "deepseek_review": {...}}
                    orig = obj.get("original", {}) or {}
                    review = obj.get("deepseek_review", {}) or {}
                    if isinstance(review, dict) and review:
                        # 旧逻辑：只用 agree_with_gold=True
                        if review.get("agree_with_gold") is not True:
                            skipped += 1
                            continue

                        tweet_id = str(orig.get("tweet_id", "")).strip()
                        llm_label_id = review.get("llm_label_id", None)
                        conf = review.get("confidence", None)
                        if tweet_id and llm_label_id is not None and conf is not None:
                            self.llm_map[tweet_id] = (int(llm_label_id), float(conf))
                            loaded += 1
                        else:
                            skipped += 1
                        continue

                    # 格式 B（Qwen baseline）:
                    # {"tweet_id": "...", "parsed":{"pred_label_id":..., "confidence":...}, ...}
                    tweet_id = str(obj.get("tweet_id", "")).strip()
                    parsed = obj.get("parsed", {}) or {}
                    llm_label_id = parsed.get("pred_label_id", obj.get("pred_label_id", None))
                    conf = parsed.get("confidence", obj.get("confidence", 1.0))

                    if tweet_id and llm_label_id is not None:
                        try:
                            conf_val = float(conf)
                        except Exception:
                            conf_val = 1.0
                        self.llm_map[tweet_id] = (int(llm_label_id), conf_val)
                        loaded += 1
                    else:
                        skipped += 1

            logging.info(f"✅ 已加载 LLM 蒸馏样本数: {loaded} | 跳过: {skipped}")
        else:
            if self.use_llm_distill:
                logging.warning(f"⚠️ 启用了 LLM 蒸馏但未找到审阅文件: {llm_review_path}")
            else:
                logging.info("ℹ️ 未启用 LLM 蒸馏")

        # label key
        if mode == "both":
            self.label_key = "label"
        elif mode == "image_only":
            self.label_key = "label_image"
        elif mode == "text_only":
            self.label_key = "label_text"
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if not eval and tensorboard:
            self.writer = SummaryWriter(log_dir=os.path.join(save_dir, "tensorboard"))

        self.is_data_parallel = isinstance(model, nn.DataParallel)
        os.makedirs(self.save_dir, exist_ok=True)

        logging.info(f"✅ Trainer初始化完成 | 任务: {task_type} | 模式: {mode} | 设备: {device}")

    def _prepare_input(self, data):
        if self.mode == "both":
            if "sentiment_features" in data and data["sentiment_features"] is not None:
                sentiment_features = data["sentiment_features"].to(self.device, non_blocking=True)
            else:
                batch_size = data["image"].size(0)
                sentiment_features = torch.zeros(batch_size, 6, device=self.device)

            # 将 tweet_id 转成 int64 张量，便于 DataParallel 正确切分；
            # 非纯数字 id 记为 -1，模型侧会回退到序号/prompt 命名。
            tweet_id_tensor = None
            tweet_ids = data.get("tweet_id", None)
            if tweet_ids is not None:
                norm_ids = []
                for tid in tweet_ids:
                    s = str(tid).strip()
                    norm_ids.append(int(s) if s.isdigit() else -1)
                tweet_id_tensor = torch.tensor(norm_ids, dtype=torch.long, device=self.device)

            x = (
                data["image"].to(self.device, non_blocking=True),
                {k: v.to(self.device, non_blocking=True) for k, v in data["text_tokens"].items()},
                sentiment_features,
                tweet_id_tensor,
            )
        elif self.mode == "image_only":
            x = data["image"].to(self.device, non_blocking=True)
        elif self.mode == "text_only":
            x = {k: v.to(self.device, non_blocking=True) for k, v in data["text_tokens"].items()}
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        return x

    def _save_model(self, checkpoint_name: str):
        save_path = os.path.join(self.save_dir, f"{checkpoint_name}.pt")
        model_state = self.model.module.state_dict() if self.is_data_parallel else self.model.state_dict()
        save_dict = {
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "task_type": self.task_type,
            "mode": self.mode,
            "epoch": checkpoint_name,
            "save_dir": self.save_dir,
        }
        torch.save(save_dict, save_path)
        logging.info(f"✅ 模型已保存至: {save_path}")

    def train(self, max_iter: int):
        if self.device != "cpu":
            self.scaler = torch.cuda.amp.GradScaler()

        best_dev_f1 = 0.0
        best_iteration = 0
        best_test_results = None

        for idx_iter in range(max_iter):
            logging.info(f"\n{'=' * 80}")
            logging.info(f"📌 训练迭代 {idx_iter + 1}/{max_iter} | 任务: {self.task_type}")
            logging.info(f"{'=' * 80}")

            self.model.train()
            correct = 0
            total = 0
            accum_loss = 0.0
            batch_count = 0
            all_predictions = []
            all_labels = []

            for data in tqdm(self.train_loader, total=len(self.train_loader), desc=f"Iter {idx_iter + 1}"):
                x = self._prepare_input(data)
                y = data[self.label_key].to(self.device, non_blocking=True)
                valid_y = y >= 0

                if self.device != "cpu":
                    with torch.cuda.amp.autocast():
                        logits = self.model(x, task_type=self.task_type)
                        loss = self.loss_fn(logits, y)

                        # ===== LLM 蒸馏（只在 llm_map 命中的样本上）=====
                        if self.use_llm_distill and self.llm_map:
                            tweet_ids = data.get("tweet_id", None)
                            if tweet_ids is not None:
                                llm_labels = []
                                llm_confs = []
                                for tid in tweet_ids:
                                    key = str(tid)
                                    if key in self.llm_map:
                                        y_llm, conf = self.llm_map[key]
                                        llm_labels.append(y_llm)
                                        llm_confs.append(conf)
                                    else:
                                        llm_labels.append(-1)
                                        llm_confs.append(0.0)

                                llm_labels = torch.tensor(llm_labels, device=self.device, dtype=torch.long)
                                llm_confs = torch.tensor(llm_confs, device=self.device, dtype=logits.dtype)
                                mask = llm_labels != -1
                                if mask.any():
                                    ce = nn.CrossEntropyLoss(reduction="none", ignore_index=-1)
                                    distill_per = ce(logits, llm_labels)  # [B]
                                    weighted = distill_per * llm_confs * mask.float()
                                    distill_loss = weighted.sum() / (mask.float().sum() + 1e-8)
                                    loss = loss + self.llm_lambda * distill_loss

                        loss = loss / self.grad_accumulation_steps
                else:
                    logits = self.model(x, task_type=self.task_type)
                    loss = self.loss_fn(logits, y)

                    if self.use_llm_distill and self.llm_map:
                        tweet_ids = data.get("tweet_id", None)
                        if tweet_ids is not None:
                            llm_labels = []
                            llm_confs = []
                            for tid in tweet_ids:
                                key = str(tid)
                                if key in self.llm_map:
                                    y_llm, conf = self.llm_map[key]
                                    llm_labels.append(y_llm)
                                    llm_confs.append(conf)
                                else:
                                    llm_labels.append(-1)
                                    llm_confs.append(0.0)

                            llm_labels = torch.tensor(llm_labels, device=self.device, dtype=torch.long)
                            llm_confs = torch.tensor(llm_confs, device=self.device, dtype=logits.dtype)
                            mask = llm_labels != -1
                            if mask.any():
                                ce = nn.CrossEntropyLoss(reduction="none", ignore_index=-1)
                                distill_per = ce(logits, llm_labels)
                                weighted = distill_per * llm_confs * mask.float()
                                distill_loss = weighted.sum() / (mask.float().sum() + 1e-8)
                                loss = loss + self.llm_lambda * distill_loss

                    loss = loss / self.grad_accumulation_steps

                accum_loss += loss.item() * self.grad_accumulation_steps

                if self.device != "cpu":
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                preds = torch.argmax(logits, dim=1)
                correct += ((preds == y) & valid_y).sum().item()
                total += int(valid_y.sum().item())
                all_predictions.extend(preds.detach().cpu().numpy())
                all_labels.extend(y.detach().cpu().numpy())

                batch_count += 1
                if batch_count % self.grad_accumulation_steps == 0:
                    if self.device != "cpu":
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()

                if batch_count % self.display == 0:
                    avg_loss = accum_loss / batch_count
                    acc = correct / total if total > 0 else 0.0
                    logging.info(
                        f"Batch {batch_count}/{len(self.train_loader)} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")
                    if self.tensorboard:
                        total_batch = idx_iter * len(self.train_loader) + batch_count
                        self.writer.add_scalar("Train/Loss", avg_loss, total_batch)
                        self.writer.add_scalar("Train/Accuracy", acc, total_batch)

            if batch_count % self.grad_accumulation_steps != 0:
                if self.device != "cpu":
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            train_acc = correct / total if total > 0 else 0.0
            train_loss = accum_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0.0
            logging.info(f"训练准确率: {train_acc:.4f} | 训练损失: {train_loss:.4f}")

            lab_np = np.asarray(all_labels, dtype=np.int64)
            if (lab_np >= 0).any() and len(np.unique(lab_np[lab_np >= 0])) > 1:
                micro = _f1_filtered(all_labels, all_predictions, "micro")
                macro = _f1_filtered(all_labels, all_predictions, "macro")
                weighted = _f1_filtered(all_labels, all_predictions, "weighted")
                logging.info(f"Train Micro F1: {micro:.4f} | Macro F1: {macro:.4f} | Weighted F1: {weighted:.4f}")

            self._save_model(f"checkpoint_{idx_iter + 1}")

            dev_loss, dev_f1, dev_metrics = self.validate(idx_iter)

            # ===== 关键：每轮都跑 test，并打印你想要的一行一个指标 =====
            logging.info(f"\n📊 迭代 {idx_iter + 1} 测试集结果:")
            test_results = self.predict(f"test_iter_{idx_iter + 1}.csv")
            logging.info(f"   测试准确率: {test_results['accuracy']:.4f}")
            logging.info(f"   测试损失: {test_results['loss']:.4f}")
            logging.info(f"   测试Micro F1: {test_results['micro_f1']:.4f}")
            logging.info(f"   测试Macro F1: {test_results['macro_f1']:.4f}")
            logging.info(f"   测试Weighted F1: {test_results['weighted_f1']:.4f}")

            if dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                best_iteration = idx_iter + 1
                best_test_results = {
                    "iteration": best_iteration,
                    "dev_f1": dev_f1,
                    "dev_acc": dev_metrics.get("accuracy", 0.0),
                    "dev_loss": dev_loss,
                    "test_acc": test_results["accuracy"],
                    "test_loss": test_results["loss"],
                    "test_micro_f1": test_results["micro_f1"],
                    "test_macro_f1": test_results["macro_f1"],
                    "test_weighted_f1": test_results["weighted_f1"],
                    "predictions_file": test_results["output_file"],
                }
                self._save_model("best")
                logging.info(f"🎯 NEW BEST @ iter {best_iteration} | dev_f1={dev_f1:.4f}")

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(dev_loss)
            else:
                self.scheduler.step()

        self._final_summary(best_test_results, best_iteration)
        if hasattr(self, "writer"):
            self.writer.close()
        return best_test_results

    def validate(self, idx_iter: int = 0):
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data in tqdm(self.dev_loader, desc="验证集评估", leave=False):
                x = self._prepare_input(data)
                y = data[self.label_key].to(self.device, non_blocking=True)
                valid_y = y >= 0
                logits = self.model(x, task_type=self.task_type)
                loss = self.loss_fn(logits, y)
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                correct += ((preds == y) & valid_y).sum().item()
                total += int(valid_y.sum().item())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        acc = correct / total if total > 0 else 0.0
        loss = total_loss / len(self.dev_loader) if len(self.dev_loader) > 0 else 0.0
        metrics = {"accuracy": acc, "loss": loss}

        dev_f1 = 0.0
        lab_np = np.asarray(all_labels, dtype=np.int64)
        if (lab_np >= 0).any() and len(np.unique(lab_np[lab_np >= 0])) > 1:
            micro = _f1_filtered(all_labels, all_preds, "micro")
            macro = _f1_filtered(all_labels, all_preds, "macro")
            weighted = _f1_filtered(all_labels, all_preds, "weighted")
            dev_f1 = weighted
            metrics.update({"micro_f1": micro, "macro_f1": macro, "weighted_f1": weighted})
            logging.info(f"Dev acc={acc:.4f} loss={loss:.4f} micro={micro:.4f} macro={macro:.4f} weighted={weighted:.4f}")

        if self.tensorboard and not self.eval:
            self.writer.add_scalar("Dev/Loss", loss, idx_iter)
            self.writer.add_scalar("Dev/Accuracy", acc, idx_iter)
            self.writer.add_scalar("Dev/Weighted_F1", dev_f1, idx_iter)

        return loss, dev_f1, metrics

    def predict(self, output_file: str = "prediction.csv"):
        self.model.eval()
        predictions = []
        correct = 0
        total = 0
        total_loss = 0.0
        all_preds = []
        all_labels = []
        hard_examples = []

        with torch.no_grad():
            for data in tqdm(self.test_loader, desc="测试集预测", leave=False):
                x = self._prepare_input(data)
                y = data[self.label_key].to(self.device, non_blocking=True)
                valid_y = y >= 0

                logits = self.model(x, task_type=self.task_type)
                loss = self.loss_fn(logits, y)
                total_loss += loss.item()

                probs = torch.softmax(logits, dim=1)
                conf, preds = torch.max(probs, dim=1)

                correct += ((preds == y) & valid_y).sum().item()
                total += int(valid_y.sum().item())

                preds_np = preds.cpu().numpy()
                predictions.extend(preds_np.tolist())
                all_preds.extend(preds_np)
                all_labels.extend(y.cpu().numpy())

                # hard examples：错分 或 低置信
                if self.log_hard_examples:
                    bs = y.size(0)
                    orig_texts = data.get("original_text", [""] * bs)
                    enh_texts = data.get("enhanced_text", [""] * bs)
                    paths = data.get("path_image", [None] * bs)
                    event_names = data.get("event_name", [""] * bs)
                    tweet_ids = data.get("tweet_id", [""] * bs)
                    image_ids = data.get("image_id", [""] * bs)
                    sent_feats = data.get("sentiment_features", None)

                    for i in range(bs):
                        g = int(y[i].item())
                        if g < 0:
                            continue
                        p = int(preds[i].item())
                        c = float(conf[i].item())
                        if (p != g) or (c < 0.8):
                            if isinstance(sent_feats, torch.Tensor):
                                sf = sent_feats[i].cpu().tolist()
                            else:
                                sf = None
                            hard_examples.append(
                                {
                                    "phase": "test",
                                    "gold_label": g,
                                    "pred_label": p,
                                    "prob": c,
                                    "original_text": orig_texts[i],
                                    "enhanced_text": enh_texts[i],
                                    "path_image": paths[i],
                                    "event_name": event_names[i],
                                    "tweet_id": str(tweet_ids[i]),
                                    "image_id": str(image_ids[i]),
                                    "sentiment_features": sf,
                                }
                            )

        output_path = os.path.join(self.save_dir, output_file)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["prediction"])
            for p in predictions:
                writer.writerow([p])

        acc = correct / total if total > 0 else 0.0
        loss = total_loss / len(self.test_loader) if len(self.test_loader) > 0 else 0.0

        micro = macro = weighted = 0.0
        lab_np = np.asarray(all_labels, dtype=np.int64)
        if (lab_np >= 0).any() and len(np.unique(lab_np[lab_np >= 0])) > 1:
            micro = _f1_filtered(all_labels, all_preds, "micro")
            macro = _f1_filtered(all_labels, all_preds, "macro")
            weighted = _f1_filtered(all_labels, all_preds, "weighted")

        # ===== 关键：恢复你之前那种“📊 测试集结果 | 任务: ...”的打印 =====
        logging.info(f"\n{'=' * 80}")
        logging.info(f"📊 测试集结果 | 任务: {self.task_type}")
        logging.info(f"{'=' * 80}")
        logging.info(f"准确率: {acc:.4f}")
        logging.info(f"损失: {loss:.4f}")
        logging.info(f"Micro F1: {micro:.4f}")
        logging.info(f"Macro F1: {macro:.4f}")
        logging.info(f"Weighted F1: {weighted:.4f}")
        logging.info(f"预测结果保存至: {output_path}")
        logging.info(f"{'=' * 80}")

        if self.log_hard_examples:
            out_path = self.hard_examples_output
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            logging.info(f"DEBUG hard_examples collected = {len(hard_examples)} | output = {out_path}")

            with open(out_path, "w", encoding="utf-8") as f:
                for ex in hard_examples:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")

            logging.info(f"✅ 难样本已导出到: {out_path} (共 {len(hard_examples)} 条)")

        return {
            "predictions": predictions,
            "accuracy": acc,
            "loss": loss,
            "micro_f1": micro,
            "macro_f1": macro,
            "weighted_f1": weighted,
            "output_file": output_path,
        }

    def _final_summary(self, best_test_results, best_iteration):
        logging.info(f"\n{'=' * 80}")
        logging.info(f"🏁 训练完成 | 任务: {self.task_type}")
        logging.info(f"{'=' * 80}")
        if best_test_results:
            logging.info(f"🎯 最佳迭代: {best_iteration} | dev_f1={best_test_results['dev_f1']:.4f} "
                         f"| test_acc={best_test_results['test_acc']:.4f} "
                         f"| test_weighted_f1={best_test_results['test_weighted_f1']:.4f}")
        else:
            logging.warning("⚠️ 训练过程中未找到最佳模型！")

    def evaluate(self):
        logging.info("📋 评估模式")
        dev_loss, dev_f1, dev_metrics = self.validate()
        test_results = self.predict("evaluation_predictions.csv")
        eval_results = {"validation": dev_metrics, "test": test_results}
        eval_path = os.path.join(self.save_dir, f"evaluation_results_{self.task_type}.json")
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=4, ensure_ascii=False)
        logging.info(f"✅ 评估结果已保存至: {eval_path}")
        return eval_results
