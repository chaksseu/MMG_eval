import numpy as np
import torch
from torchmetrics.multimodal.clip_score import CLIPScore
from tqdm import tqdm
from math import inf
import cv2
'''

def calculate_clip(videos1, caps, calculate_per_frame, calculate_final, clip_model, device):
    print("calculate_CLIP Score...")

    metric = CLIPScore(model_name_or_path=clip_model).to(device)
    videos1 = videos1.to(device)
    # videos [batch_size, timestamps, channel, h, w]
    clip_results = []
    idx = 0
    for video_num in tqdm(range(videos1.shape[0])):
        # get a video
        # video [timestamps, channel, h, w]
        text  = caps[video_num]
        video1 = videos1[video_num]

        #print(text)

        clip_results_of_a_video = []

        video1 = video1 * 255
        text = [text for i in range(len(video1))] 
        score = metric(video1, text)
        clip_results_of_a_video.append(score.item())


        clip_results.append(clip_results_of_a_video)
        

    clip_results = np.array(clip_results)

    clip = {}
    clip_std = {}

    if calculate_final:
        clip['final'] = np.mean(clip_results)
        clip_std['final'] = np.std(clip_results)

    result = {
        "clip": clip,
        "clip_std": clip_std,
        "clip_video_setting": video1.shape,
        "clip_video_setting_name": "time, channel, heigth, width",
    }

    return result
'''


import numpy as np
import torch
from torchmetrics.multimodal.clip_score import CLIPScore
from tqdm import tqdm


def calculate_clip(
    videos1,
    caps,
    calculate_per_frame,
    calculate_final,
    clip_model,
    device,
    chunk_size=40,
):
    """
    한 번에 모든 프레임을 GPU로 올리지 않고, 
    chunk_size 단위로 나누어 CLIPScore를 계산하여 
    OOM(Out Of Memory) 문제를 해결하는 예시 코드.
    
    Args:
        videos1 (torch.Tensor): [batch_size, timestamps, channel, height, width]
        caps (List[str]): 각 비디오별 텍스트 캡션 리스트
        calculate_final (bool): True면 모든 video에 대해 최종 mean/std 계산
        clip_model (str): CLIP 모델 이름 (ex: 'openai/clip-vit-base-patch32')
        device (torch.device): GPU 장치
        chunk_size (int): 한 번에 처리할 프레임 수

    Returns:
        dict: {
            "clip": {"final": float}, 
            "clip_std": {"final": float}, 
            "clip_video_setting": tuple,
            "clip_video_setting_name": str
        }
    """
    print("calculate_CLIP Score...")

    # (1) CLIPScore 객체 생성
    metric = CLIPScore(model_name_or_path=clip_model).to(device)

    # (2) 전체 영상을 GPU로 옮기지 않고, CPU에서 필요한 부분만 GPU로
    #     넘길 것이므로 videos1.to(device)는 사용하지 않음.
    clip_results = []

    # (3) 비디오 단위로 계산
    for video_num in tqdm(range(videos1.shape[0])):
        text = caps[video_num]
        # [timestamps, channel, height, width]
        video_cpu = videos1[video_num]  # 아직 GPU로 안 옮김

        # 주의: CLIPScore는 입력 영상이 0~255 범위를 가정.
        # 따라서 모델 내부 전처리에 맞게 스케일 조정
        video_cpu = video_cpu * 255

        # 영상 전체에 대한 score를 담을 리스트
        scores_per_video = []

        # (4) chunk_size 단위로 잘라서 계산
        #     예: timestamps가 32이고 chunk_size=8이면
        #     총 4번에 나눠서 GPU에 올려 계산
        for start_idx in range(0, video_cpu.shape[0], chunk_size):
            end_idx = start_idx + chunk_size
            frames_chunk = video_cpu[start_idx:end_idx]

            # 이 chunk만 GPU로 올림
            frames_chunk = frames_chunk.to(device)

            # 해당 chunk의 frame 수만큼 caption도 똑같이 반복
            text_chunk = [text for _ in range(frames_chunk.shape[0])]

            # (5) CLIPScore 계산
            with torch.no_grad():  # 메모리 사용량 줄이기
                score = metric(frames_chunk, text_chunk)

            # chunk 단위 score를 합산(혹은 저장)
            scores_per_video.append(score.item())

            # GPU 메모리 절약을 위해 사용 끝난 텐서는 메모리에서 제거
            del frames_chunk
            torch.cuda.empty_cache()

        # (6) 비디오 한 개에 대한 최종 score: chunk별 score 평균
        video_clip_score = np.mean(scores_per_video)
        clip_results.append(video_clip_score)

    # (7) 모든 비디오에 대한 결과 정리
    clip_results = np.array(clip_results)

    clip_dict = {}
    clip_std_dict = {}

    if calculate_final:
        clip_dict['final'] = np.mean(clip_results)
        clip_std_dict['final'] = np.std(clip_results)

    result = {
        "clip": clip_dict,
        "clip_std": clip_std_dict,
        # 마지막 비디오의 shape를 예시로 기록
        "clip_video_setting": videos1[0].shape,
        "clip_video_setting_name": "time, channel, height, width",
    }

    return result
