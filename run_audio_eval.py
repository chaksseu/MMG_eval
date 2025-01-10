'''
Audio and Visual Evaluation Toolkit

Author: Lucas Goncalves
Date Created: 2023-08-16 16:34:44 PDT
Last Modified: 2023-08-24 9:27:30 PDT		

Description:
Audio Evaluation - run_audio_eval.py
This toolbox includes the following metrics:
- FAD: Frechet audio distance
- ISc: Inception score
- FD: Frechet distance, realized by PANNs, a state-of-the-art audio classification model
- KL: KL divergence (softmax over logits)
- KL_Sigmoid: KL divergence (sigmoid over logits)
- SI_SDR: Scale-Invariant Signal-to-Distortion Ratio
- SDR: Signal-to-Distortion Ratio
- SI_SNR: Scale-Invariant Signal-to-Noise Ratio
- SNR: Signal-to-Noise Ratio
- PESQ: Perceptual Evaluation of Speech Quality
- STOI: Short-Time Objective Intelligibility
- CLAP-Score: Implemented with LAION-AI/CLAP

### Running the metris
python run_audio_eval.py --preds_folder /path/to/generated/audios --target_folder /path/to/the/target_audios \
--metrics SI_SDR SDR SI_SNR SNR PESQ STOI CLAP FAD ISC FD KL --results NAME_YOUR_RESULTS_FILE.txt


Third-Party Snippets/Credits:

[1] - Taken from [https://github.com/haoheliu/audioldm_eval] - [MIT License]
    - Adapted code for FAD, ISC, FID, and KL computation

[2] - Taken from [https://github.com/LAION-AI/CLAP] - [CC0-1.0 license]
    - Snipped utilized for audio embeddings and text embeddings retrieval

'''
import argparse
import os
import numpy as np
import datetime
import torch
import torchaudio
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.audio import (ScaleInvariantSignalDistortionRatio, ScaleInvariantSignalNoiseRatio,
                                SignalDistortionRatio, SignalNoiseRatio)
from utils.load_mel import WaveDataset
import laion_clap
from audio_metrics.clap_score import calculate_clap
from audio_metrics.fad import FrechetAudioDistance
from audio_metrics.fid import calculate_fid
from audio_metrics.isc import calculate_isc
from audio_metrics.kl import calculate_kl
from feature_extractors.panns import Cnn14


def check_folders(preds_folder, target_folder):
    preds_files = [f for f in os.listdir(preds_folder) if f.endswith('.wav')]
    target_files = [f for f in os.listdir(target_folder) if f.endswith('.wav')]
    if len(preds_files) != len(target_files):
        print('Mismatch in number of files between preds and target folders.')
        return False
    return True

def evaluate_audio_metrics(preds_folder, target_folder, metrics, results_file, clap_model, device="cpu"):
    scores = {metric: [] for metric in metrics}
    
    if target_folder is None or not check_folders(preds_folder, target_folder):
        text = 'Running only reference-free metrics'
        same_name = False
    else:
        text = 'Running all metrics specified'
        same_name = True

    # Frechet Audio Distance 사용 여부
    if 'FAD' in metrics or 'KL' in metrics or 'ISC' in metrics or 'FD' in metrics:
        sampling_rate = 16000
        # 여기서 device를 전달하여 FAD 클래스가 올바른 디바이스를 사용하도록 함
        frechet = FrechetAudioDistance(device=device)
        torch.manual_seed(0)
        
        num_workers = 0
        outputloader = DataLoader(
            WaveDataset(
                preds_folder,
                sampling_rate, 
                limit_num=None,
            ),
            batch_size=1,
            sampler=None,
            num_workers=num_workers,
        )
        resultloader = DataLoader(
            WaveDataset(
                target_folder,
                sampling_rate, 
                limit_num=None,
            ),
            batch_size=1,
            sampler=None,
            num_workers=num_workers,
        )
        out = {}

        # FAD 계산
        if 'FAD' in metrics:
            fad_score = frechet.score(preds_folder, target_folder, limit_num=None)
            out['frechet_audio_distance'] = fad_score
        
        '''
        if check_folders(preds_folder, target_folder) and 'KL' in metrics:
            kl_sigmoid, kl_softmax, kl_ref, paths_1 = calculate_kl(
                featuresdict_1, featuresdict_2, 'logits', same_name
            )
            out['kullback_leibler_divergence_sigmoid'] = float(kl_sigmoid)
            out['kullback_leibler_divergence_softmax'] =  float(kl_softmax)

        '''

    # Loading Clap Model
    if 'CLAP' in metrics:
        if clap_model == 0 or clap_model == 1:
            model_clap = laion_clap.CLAP_Module(enable_fusion=False) 
        elif clap_model == 2 or clap_model == 3:
            model_clap = laion_clap.CLAP_Module(enable_fusion=True) 

        model_clap.load_ckpt(model_id=clap_model) # Download the default pretrained checkpoint.
        # Resampling rate
        new_freq = 48000
    else:
        model_clap = None


    # Get the list of filenames and set up the progress bar
    filenames = [f for f in os.listdir(preds_folder) if f.endswith('.wav')]
    progress_bar = tqdm(filenames, desc='Processing')

    print(text)
    for filename in progress_bar:
        if filename.endswith('.wav'):
            try:
                preds_audio, _ = torchaudio.load(os.path.join(preds_folder, filename), num_frames=160000)
                #target_audio, _ = torchaudio.load(os.path.join(target_folder, filename), num_frames=160000)
                #min_len = min(preds_audio.size(1), target_audio.size(1))
                #preds_audio, target_audio = preds_audio[:, :min_len], target_audio[:, :min_len]
                
                #if np.shape(target_audio)[0] == 2:
                #    target_audio = target_audio.mean(dim=0)
                if np.shape(preds_audio)[0] == 2:
                    preds_audio = preds_audio.mean(dim=0)

                # Compute and store the scores for the specified metrics
                if 'CLAP' in metrics: scores['CLAP'].append(calculate_clap(model_clap, preds_audio, filename, new_freq))
                '''
                if si_snr: scores['SI_SNR'].append(si_snr(preds_audio.squeeze(), target_audio.squeeze()).item())
                if snr_calculator: scores['SNR'].append(snr_calculator(preds_audio.squeeze(), target_audio.squeeze()).item())
                if sdr_calculator: scores['SDR'].append(sdr_calculator(preds_audio.squeeze(), target_audio.squeeze()).item())
                if si_sdr: scores['SI_SDR'].append(si_sdr(preds_audio.squeeze(), target_audio.squeeze()).item())
                if pesq_metric: scores['PESQ'].append(pesq_metric(preds_audio.squeeze(), target_audio.squeeze()).item())
                if stoi_metric: scores['STOI'].append(stoi_metric(preds_audio.squeeze(), target_audio.squeeze()).item())
                '''
            except Exception as e:
                print(f'Error processing {filename}: {e}')


    # Print and save the average and standard deviation for each metric
    with open(results_file, 'w') as file:
        for metric, values in scores.items():
            if str(metric.upper()) not in ['FAD', 'ISC', 'FD', 'KL']:
                avg = np.mean(values)
                std = np.std(values)
                print(f'{metric.upper()}: Average = {avg}, Std = {std}')
                file.write(f'{metric.upper()}: Average = {avg}, Std = {std}\n')
        if 'FAD' in metrics:
            print(f"FAD: {out['frechet_audio_distance']:.5f}")
            file.write(f"FAD: {out['frechet_audio_distance']:.5f}\n")
        if 'ISC' in metrics:
            print(f"ISc: Average = {out['inception_score_mean']:8.5f}, Std = {out['inception_score_std']:5f})")
            file.write(f"ISc: Average = {out['inception_score_mean']:8.5f}, Std = {out['inception_score_std']:5f}\n")
        if 'FD' in metrics:
            print(f"FD: {out['frechet_distance']:8.5f}")
            file.write(f"FD: {out['frechet_distance']:8.5f}\n")
        if check_folders(preds_folder, target_folder) and 'KL' in metrics:
            print(f"KL_Sigmoid: {out['kullback_leibler_divergence_sigmoid']:8.5f}")
            print(f"KL_Softmax: {out['kullback_leibler_divergence_softmax']:8.5f}")
            file.write(f"KL_Sigmoid: {out['kullback_leibler_divergence_sigmoid']:8.5f}\n")
            file.write(f"KL: {out['kullback_leibler_divergence_softmax']:8.5f}\n")

# Defining clap model descriptions
CLAP_MODEL_DESCRIPTIONS = {
    0: '630k non-fusion ckpt',
    1: '630k+audioset non-fusion ckpt',
    2: '630k fusion ckpt',
    3: '630k+audioset fusion ckpt'
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate audio on acoustic metrics.')

    parser.add_argument('--preds_folder', required=True, help='Path to the folder with predicted audio files.')
    parser.add_argument('--target_folder', required=False, default=None, help='Path to the folder with target audio files.')
    
    parser.add_argument('--metrics', nargs='+',
                        choices=['SI_SDR', 'SDR', 'SI_SNR', 'SNR', 'PESQ', 'STOI', 'CLAP', 'FAD', 'ISC', 'FD', 'KL'],
                        help='List of metrics to calculate.')
    
    parser.add_argument('--clap_model', type=int, default=1, help='CLAP model id for score computations.')
    parser.add_argument('--results_file', required=True, help='Path to the text file to save the results.')
    
    # device 인자 추가 (기본값 "cpu")
    parser.add_argument('--device', default="cpu", help='Device to use: "cpu" or "cuda"')

                        
    args = parser.parse_args()
    evaluate_audio_metrics(
        args.preds_folder,
        args.target_folder,
        args.metrics,
        args.results_file,
        args.clap_model,
        device=args.device
    )