"""CPKG 训练、检索评估和 Chinese-CLIP 基线入口。"""

import argparse
import logging
from pathlib import Path
import random

from uer.utils.poetry_data import (
    encode_poem,
    find_knowledge_spans,
    load_retrieval_pairs,
    resolve_image_path,
)


ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)


def get_args(argv=None):
    """解析命令行并校验参数，返回配置；读取帮助时不加载模型或下载权重。"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", required=True, choices=["train", "evaluate", "baseline"],
                        help="训练、CPKG 检索评估或 Chinese-CLIP 基线")
    parser.add_argument("--vi_model", default="RN50",
                        choices=["RN50", "ViT-B-16", "ViT-L-14", "ViT-L-14-336", "ViT-H-14"],
                        help="训练与评估必须使用相同的 CLIP 型号")
    parser.add_argument("--model_dir", default=str(ROOT / "models"), help="CLIP 权重缓存目录")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="运行设备")
    parser.add_argument("--train_path", default=str(ROOT / "data/tran_val/train.json"))
    parser.add_argument("--val_path", default=str(ROOT / "data/tran_val/val.json"))
    parser.add_argument("--kg_node", default=str(ROOT.parent / "KG/imagery_text_image_align.json"))
    parser.add_argument("--image_root", default=str(ROOT / "data/kg/images"), help="意象图片根目录")
    parser.add_argument("--output_model_path", default=str(ROOT / "outputs/CPKG_checkpoint"),
                        help="检查点路径前缀，文件名会追加轮数")
    parser.add_argument("--resume", help="继续训练的检查点文件")
    parser.add_argument("--checkpoint", help="evaluate 模式使用的 CPKG 检查点")
    parser.add_argument("--retrieval_path",
                        default=str(ROOT / "data/PI_retrieval_2/results_500_V1_c2e.json"))
    parser.add_argument("--retrieval_images", default=str(ROOT / "data/PI_retrieval_2/images"))
    parser.add_argument("--batch_size", type=int, default=16, help="每批样本数")
    parser.add_argument("--seq_length", type=int, default=52, help="诗句长度，包含 CLS 和 SEP")
    parser.add_argument("--max_samples", type=int, help="调试用：每个数据集最多取多少条原诗")
    parser.add_argument("--epochs_num", type=int, default=100, help="训练的目标总轮数")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--eval_interval", type=int, default=3, help="验证间隔轮数")
    parser.add_argument("--early_stop", type=int, default=5, help="连续无改善的验证次数")
    parser.add_argument("--report_steps", type=int, default=100, help="训练日志间隔批数")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--top_k", type=int, default=20, help="检索指标的 K")
    args = parser.parse_args(argv)
    for name in ("batch_size", "epochs_num", "eval_interval", "early_stop", "report_steps", "top_k"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} 必须为正整数")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("max_samples 必须为正整数")
    if not 4 <= args.seq_length <= 512:
        parser.error("seq_length 必须在 4 到 512 之间")
    if args.learning_rate <= 0 or not 0 <= args.warmup < 1:
        parser.error("学习率必须为正数，warmup 必须在 [0, 1) 内")
    if args.mode == "train" and args.batch_size < 2:
        parser.error("对比学习需要 batch_size 至少为 2")
    if args.mode == "evaluate" and not args.checkpoint:
        parser.error("evaluate 模式需要 --checkpoint")
    if args.resume and args.mode != "train":
        parser.error("--resume 仅用于 train 模式")
    return args


def get_modal(args, device):
    """加载指定 CLIP 和同结构的 CPKG 文本模型；缺少 CLIP 权重时使用其下载器。"""
    import torch
    from Chinese_CLIP_master.cn_clip.clip import get_text_info, load_from_name
    from uer.encoders.clip_bert import BERT

    model_clip, preprocess = load_from_name(args.vi_model, device=device, download_root=args.model_dir)
    model_clip.eval()
    if args.mode == "baseline":
        return model_clip, preprocess, None
    model_info, state = get_text_info(args.vi_model, device=device, download_root=args.model_dir)
    model = BERT(**model_info)
    # 保留原训练脚本对新增参数的初始化方式和检查点参数名称。
    for name, parameter in model.named_parameters():
        if name not in state:
            state[name] = torch.empty_like(parameter).normal_(mean=0.0, std=0.02)
    model.load_state_dict(state)
    return model_clip, preprocess, model.to(device)


def read_dataset(poems, kg, model_clip, preprocess, args, device):
    """按原先每诗五次抽样生成 CPU 张量数据集；读取图片并提取冻结的 CLIP 特征。"""
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torch.utils.data import TensorDataset
    from Chinese_CLIP_master.cn_clip import clip

    records = list(poems.values())[:args.max_samples]
    columns = [[] for _ in range(5)]
    skipped = 0
    with torch.no_grad():
        for record in records:
            sentence = record["诗句"]
            candidates = find_knowledge_spans(
                sentence, kg.tokenizer.cut(sentence), kg.lookup_table, args.seq_length,
            )
            if not candidates:
                skipped += 1
                continue
            tokens = encode_poem(sentence, clip._tokenizer.vocab, args.seq_length)
            translation = model_clip.encode_text(clip.tokenize(record["译文"]).to(device))
            translation = F.normalize(translation.float(), dim=-1).cpu().squeeze(0)
            word_cache = {}
            for _ in range(5):
                word, span = random.choice(candidates)
                if word not in word_cache:
                    # 保留原实验选择首个释义的策略，不在整理过程中改变消歧算法。
                    node = kg.kg_nodes[word][0]
                    image_path = resolve_image_path(args.image_root, node)
                    with Image.open(image_path) as image:
                        image_input = preprocess(image).unsqueeze(0).to(device)
                    image_feature = F.normalize(model_clip.encode_image(image_input).float(), dim=-1)
                    meaning = model_clip.encode_text(clip.tokenize(node["Meaning"]).to(device))
                    word_cache[word] = (image_feature.cpu().squeeze(0), meaning.float().cpu().squeeze(0))
                image_feature, meaning = word_cache[word]
                for column, value in zip(columns, (tokens, span, image_feature, translation, meaning)):
                    column.append(value)
    if not columns[0]:
        raise ValueError("没有可训练样本：请检查诗句、双字意象和序列截断长度")
    LOGGER.info("原诗 %d 条，跳过无可见双字意象的诗句 %d 条，生成样本 %d 条",
                len(records), skipped, len(columns[0]))
    return TensorDataset(
        torch.tensor(columns[0], dtype=torch.long),
        torch.tensor(columns[1], dtype=torch.long),
        *(torch.stack(column) for column in columns[2:]),
    )


def run_epoch(model, loader, device, optimizer=None, report_steps=100):
    """执行一轮训练或验证，返回三项按批平均的损失；仅训练时更新参数。"""
    import torch

    training = optimizer is not None
    model.train(training)
    totals = [0.0, 0.0, 0.0]
    steps = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            src, spans, images, translations, meanings = batch
            if training:
                optimizer.zero_grad(set_to_none=True)
            losses = model(
                src.to(device), spans, images.to(device),
                translations.to(device), meanings.to(device),
            )
            if not all(torch.isfinite(loss).item() for loss in losses):
                raise ValueError("出现非有限损失，请检查数据和学习率")
            if training:
                sum(losses).backward()
                optimizer.step()
            steps += 1
            totals = [total + loss.item() for total, loss in zip(totals, losses)]
            if training and steps % report_steps == 0:
                LOGGER.info("已训练 %d 批，平均总损失 %.6f", steps, sum(totals) / steps)
    if steps == 0:
        raise ValueError("数据加载器没有有效批次")
    return tuple(total / steps for total in totals)


def load_checkpoint(model, path, args):
    """加载模型并返回训练状态；验证批大小改变或未知时重置返回的早停状态，不改原文件。"""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    for name in ("vi_model", "seq_length"):
        if name in checkpoint and checkpoint[name] != getattr(args, name):
            raise ValueError(f"检查点的 {name}={checkpoint[name]}，请使用相同参数")
    model.load_state_dict(checkpoint["model_state_dict"])
    if "vi_model" not in checkpoint:
        LOGGER.warning("旧检查点没有模型与序列长度记录，请核对原实验配置")
    # 对比损失随验证批大小改变，旧损失不能继续作为早停的比较基准。
    if args.mode == "train" and checkpoint.get("args", {}).get("batch_size") != args.batch_size:
        checkpoint["best_loss"] = float("inf")
        checkpoint["bad_epochs"] = 0
        LOGGER.warning("验证批大小改变或原值未记录，已重置最佳验证损失和早停计数")
    return checkpoint


def save_checkpoint(model, optimizer, args, epoch, best_loss, bad_epochs):
    """保存权重、优化器和训练进度到带轮数后缀的文件，返回实际路径。"""
    import torch

    path = Path(f"{args.output_model_path}-{epoch}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "vi_model": args.vi_model,
        "seq_length": args.seq_length,
        "best_loss": best_loss,
        "bad_epochs": bad_epochs,
        "args": vars(args),
    }, path)
    return path


def restore_optimizer(optimizer, checkpoint, remaining_steps):
    """恢复优化器状态并追加剩余步数，避免改变批大小后出现负学习率。"""
    if "optimizer_state_dict" not in checkpoint:
        LOGGER.warning("旧检查点不含优化器状态，将从已保存权重重新建立优化器")
        return
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    completed_steps = max((state.get("step", 0) for state in optimizer.state.values()), default=0)
    for group in optimizer.param_groups:
        group["t_total"] = completed_steps + remaining_steps


def pretrain(args, device):
    """准备图谱和特征，执行训练、定期验证及早停；每轮保存可继续训练的检查点。"""
    import torch
    from torch.utils.data import DataLoader
    from uer.kg.kg import ModalKnowledgeGraph
    from uer.utils.optimizers import BertAdam

    if not Path(args.image_root).is_dir():
        raise FileNotFoundError(f"意象图片目录不存在：{args.image_root}，请设置 --image_root")
    model_clip, preprocess, model = get_modal(args, device)
    kg = ModalKnowledgeGraph(
        node_file=args.kg_node, train_path=args.train_path, dev_path=args.val_path,
    )
    train_data = read_dataset(kg.train_data, kg, model_clip, preprocess, args, device)
    val_data = read_dataset(kg.val_data, kg, model_clip, preprocess, args, device)
    # 特征已移到 CPU，释放冻结编码器占用的显存后再开始更新 CPKG。
    del kg, model_clip, preprocess
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # 单样本没有对比负例，只舍弃最后恰好剩余一个样本的训练批次。
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        drop_last=len(train_data) % args.batch_size == 1,
    )
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    no_decay = ("bias", "gamma", "beta")
    parameters = list(model.named_parameters())
    optimizer = BertAdam([
        {"params": [p for n, p in parameters if not any(key in n for key in no_decay)],
         "weight_decay_rate": 0.01},
        {"params": [p for n, p in parameters if any(key in n for key in no_decay)],
         "weight_decay_rate": 0.0},
    ], lr=args.learning_rate, warmup=args.warmup, t_total=len(train_loader) * args.epochs_num)
    first_epoch, best_loss, bad_epochs = 1, float("inf"), 0
    if args.resume:
        checkpoint = load_checkpoint(model, args.resume, args)
        first_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint.get("best_loss", float("inf"))
        bad_epochs = checkpoint.get("bad_epochs", 0)
    if first_epoch > args.epochs_num:
        raise ValueError("检查点已达到目标轮数，请增大 --epochs_num")
    if args.resume:
        remaining_steps = (args.epochs_num - first_epoch + 1) * len(train_loader)
        restore_optimizer(optimizer, checkpoint, remaining_steps)
    for epoch in range(first_epoch, args.epochs_num + 1):
        losses = run_epoch(model, train_loader, device, optimizer, args.report_steps)
        LOGGER.info("第 %d 轮：训练总损失 %.6f，意象-图片 %.6f，意象-释义 %.6f，诗句-译文 %.6f",
                    epoch, sum(losses), *losses)
        if epoch % args.eval_interval == 0 or epoch == args.epochs_num:
            val_loss = sum(run_epoch(model, val_loader, device))
            LOGGER.info("第 %d 轮：验证总损失 %.6f", epoch, val_loss)
            if val_loss < best_loss:
                best_loss, bad_epochs = val_loss, 0
            else:
                bad_epochs += 1
        path = save_checkpoint(model, optimizer, args, epoch, best_loss, bad_epochs)
        LOGGER.info("检查点已保存：%s", path)
        if bad_epochs >= args.early_stop:
            LOGGER.info("连续 %d 次验证未改善，停止训练", bad_epochs)
            break


def get_metric(results, K=1):
    """对一一配对的方形相似度矩阵计算 mAP@K 和 Recall@K；K 自动限制到样本数。"""
    import torch

    if results.ndim != 2 or results.shape[0] != results.shape[1] or results.shape[0] == 0:
        raise ValueError("检索分数必须是非空方阵，且对角线为正确配对")
    if K <= 0:
        raise ValueError("K 必须为正整数")
    count = min(K, results.shape[0])
    indices = results.topk(count, dim=1).indices
    correct = indices.eq(torch.arange(results.shape[0], device=results.device).unsqueeze(1))
    ranks = torch.arange(1, count + 1, device=results.device)
    mean_ap = (correct.float() / ranks).sum(dim=1).mean().item()
    recall = correct.any(dim=1).float().mean().item()
    return mean_ap, recall


def evaluate(args, device):
    """按批提取原诗和图片特征并打印双向检索指标；evaluate 模式使用 CPKG 权重。"""
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from Chinese_CLIP_master.cn_clip import clip

    pairs = load_retrieval_pairs(
        args.retrieval_path, args.retrieval_images, max_samples=args.max_samples,
    )
    model_clip, preprocess, model = get_modal(args, device)
    if model is not None:
        load_checkpoint(model, args.checkpoint, args)
        model.eval()
    text_features, image_features = [], []
    with torch.no_grad():
        for start in range(0, len(pairs), args.batch_size):
            batch = pairs[start:start + args.batch_size]
            images = []
            for _, image_path in batch:
                with Image.open(image_path) as image:
                    images.append(preprocess(image))
            image = model_clip.encode_image(torch.stack(images).to(device))
            poems = [poem for poem, _ in batch]
            if model is None:
                text = model_clip.encode_text(clip.tokenize(poems, context_length=args.seq_length).to(device))
            else:
                tokens = torch.tensor([
                    encode_poem(poem, clip._tokenizer.vocab, args.seq_length) for poem in poems
                ], dtype=torch.long, device=device)
                text = model.encode_text(tokens)
            image_features.append(F.normalize(image.float(), dim=-1).cpu())
            text_features.append(F.normalize(text.float(), dim=-1).cpu())
    scores = torch.cat(text_features) @ torch.cat(image_features).t()
    count = min(args.top_k, len(pairs))
    LOGGER.info("评估模式 %s，CLIP 型号 %s，样本数 %d", args.mode, args.vi_model, len(pairs))
    for direction, matrix in (("诗句检索图片", scores), ("图片检索诗句", scores.t())):
        mean_ap, recall = get_metric(matrix, K=count)
        print(f"{direction}：mAP@{count}={mean_ap:.6f}，Recall@{count}={recall:.6f}")


def main(argv=None):
    """解析配置、设置随机种子并分派运行模式；运行错误以非零退出码报告。"""
    args = get_args(argv)
    import torch
    from uer.utils.seed import set_seed

    logging.basicConfig(level=logging.INFO, format="%(levelname)s：%(message)s")
    set_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("当前环境不可用 CUDA，请使用 --device cpu")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if args.mode == "train":
        pretrain(args, device)
    else:
        evaluate(args, device)


if __name__ == "__main__":
    main()
