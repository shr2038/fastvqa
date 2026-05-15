import yaml
import decord
from fastvqa.datasets import get_spatial_fragments, SampleFrames, FragmentSampleFrames
from fastvqa.models import DiViDeAddEvaluator
import torch
import numpy as np
import argparse

def sigmoid_rescale(score, model="FasterVQA"):
    mean, std = mean_stds[model]
    x = (score - mean) / std
    print(f"Inferring with model [{model}]:")
    score = 1 / (1 + np.exp(-x))
    return score

mean_stds = {
    "FasterVQA": (0.14759505, 0.03613452), 
    "FasterVQA-MS": (0.15218826, 0.03230298),
    "FasterVQA-MT": (0.14699507, 0.036453716),
    "FAST-VQA":  (-0.110198185, 0.04178565),
    "FAST-VQA-M": (0.023889644, 0.030781006), 
}

opts = {
    "FasterVQA": "./options/fast/f3dvqa-b.yml", 
    "FasterVQA-MS": "./options/fast/fastervqa-ms.yml", 
    "FasterVQA-MT": "./options/fast/fastervqa-mt.yml", 
    "FAST-VQA": "./options/fast/fast-b.yml", 
    "FAST-VQA-M": "./options/fast/fast-m.yml", 
}

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-m", "--model", type=str, 
        default="FAST-VQA",  # 默认用FAST-VQA，和你下载的权重匹配
        help="model type: can choose between FasterVQA, FasterVQA-MS, FasterVQA-MT, FAST-VQA, FAST-VQA-M",
    )
    
    parser.add_argument(
        "-v", "--video_path", type=str, 
        default="./demos/10053703034.mp4", 
        help="the input video path"
    )
    
    parser.add_argument(
        "-d", "--device", type=str, 
        default="cpu",  # 默认用CPU，避免CUDA问题
        help="the running device"
    )
    
    args = parser.parse_args()

    # 读取视频
    video_reader = decord.VideoReader(args.video_path)
    print(f"✅ 视频加载成功，总帧数: {len(video_reader)}")
    
    # 加载配置文件
    opt = opts.get(args.model, opts["FAST-VQA"])
    with open(opt, "r") as f:
        opt = yaml.safe_load(f)
    print(f"✅ 配置文件加载成功")

    # 加载模型
    print("正在加载模型...")
    evaluator = DiViDeAddEvaluator(**opt["model"]["args"]).to(args.device)
    
    # 加载权重（直接指定正确的文件名）
    load_path = "pretrained_weights/fast-vqa_v0_3.pth"
    state_dict = torch.load(load_path, map_location=args.device)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    
    # 修正键名：同时修正 backbone -> fragments_backbone 和 cls_head -> vqa_head
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        if "backbone" in k and "fragments_backbone" not in k:
            new_k = new_k.replace("backbone", "fragments_backbone")
        if "cls_head" in k:
            new_k = new_k.replace("cls_head", "vqa_head")
        new_state_dict[new_k] = v
    
    evaluator.load_state_dict(new_state_dict, strict=True)
    print("✅ 模型和权重加载成功！")

    # 数据预处理（官方原版流程）
    vsamples = {}
    t_data_opt = opt["data"]["val-kv1k"]["args"]
    s_data_opt = opt["data"]["val-kv1k"]["args"]["sample_types"]
    
    for sample_type, sample_args in s_data_opt.items():
        print(f"\n正在处理采样类型: {sample_type}")
        
        # 时间采样
        if t_data_opt.get("t_frag",1) > 1:
            sampler = FragmentSampleFrames(
                fsize_t=sample_args["clip_len"] // sample_args.get("t_frag",1),
                fragments_t=sample_args.get("t_frag",1),
                num_clips=sample_args.get("num_clips",1),
            )
        else:
            sampler = SampleFrames(
                clip_len=sample_args["clip_len"], 
                num_clips=sample_args.get("num_clips",1)
            )
        
        num_clips = sample_args.get("num_clips",1)
        frames = sampler(len(video_reader))
        print(f"采样帧数: {frames}")
        
        # 读取采样的帧
        frame_dict = {idx: video_reader[idx] for idx in np.unique(frames)}
        imgs = [frame_dict[idx] for idx in frames]
        video = torch.stack(imgs, 0)
        video = video.permute(3, 0, 1, 2)

        # 空间采样
        sampled_video = get_spatial_fragments(video, **sample_args)
        mean, std = torch.FloatTensor([123.675, 116.28, 103.53]), torch.FloatTensor([58.395, 57.12, 57.375])
        sampled_video = ((sampled_video.permute(1, 2, 3, 0) - mean) / std).permute(3, 0, 1, 2)
        
        sampled_video = sampled_video.reshape(
            sampled_video.shape[0], num_clips, -1, *sampled_video.shape[2:]
        ).transpose(0,1)
        
        vsamples[sample_type] = sampled_video.to(args.device)
        print(f"采样后视频形状: {sampled_video.shape}")

    # 推理
    print("\n正在评估视频质量...")
    with torch.no_grad():
        result = evaluator(vsamples)
    
    # 计算最终分数
    score = sigmoid_rescale(result.mean().item(), model=args.model)
    
    # 输出结果
    print("\n" + "="*60)
    print(f"🎉 视频画质评分（MOS，范围[0,1]）：{score:.5f}")
    print(f"   对应1-5分制：{score*4+1:.2f}")
    print("="*60)
    print("✅ 评估完成！")