
import torchaudio
import numpy as np
from scipy.spatial.distance import cosine
import re  # 정규 표현식 사용을 위해 import

def clean_sentence(filename):

    # 파일 이름에서 밑줄을 공백으로 바꾸고 확장자를 제거
    sentence = filename.replace('_', ' ').replace('.wav', '')
    
    # 제외할 단어 목록
    excluded_words = ['batch', 'proc', 'sample', 'audio']
    
    # 제외할 단어 패턴 생성 (단어 뒤에 숫자가 올 수 있도록 수정)
    pattern_words = r'\b(?:' + '|'.join(excluded_words) + r')\d*\b'
    # 숫자 패턴 (단독으로 있는 숫자 제거)
    pattern_numbers = r'\b\d+\b'
    
    # 제외할 단어 제거
    sentence = re.sub(pattern_words, '', sentence, flags=re.IGNORECASE)
    
    # 숫자 제거
    sentence = re.sub(pattern_numbers, '', sentence)
    
    # 불필요한 공백 제거
    sentence = ' '.join(sentence.split())
    
    return sentence

def calculate_clap(model_clap, preds_audio, filename, freq):
    resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=freq)
    preds_audio_clap = resampler(preds_audio)

    preds_audio_clap  = preds_audio_clap.squeeze().numpy()

    # Get audio embeddings from audio data
    audio_data = preds_audio_clap.reshape(1, -1) # Make it (1,T) or (N,T)
    audio_embed = model_clap.get_audio_embedding_from_data(x=audio_data, use_tensor=False)

    # Get text embeddings from texts
    sentence_clean = clean_sentence(filename)
    
    text_data = [sentence_clean, sentence_clean]

    print(text_data)
    text_embed = model_clap.get_text_embedding(text_data)

    E_cap = np.array(text_embed[0])
    E_aud = np.squeeze(np.array(audio_embed))

    E_aud = E_aud / np.linalg.norm(E_aud)
    E_cap = E_cap / np.linalg.norm(E_cap)

    # Compute the cosine similarity
    similarity = 1 - cosine(E_aud, E_cap)

    # Scale the similarity score and bound it between 0 and 100
    score = max(100 * similarity, 0)

    return score

def clean_sentence(filename):
    """
    Clean the filename to extract meaningful text by removing excluded words and numbers.

    Params:
    -- filename : String, name of the audio file.

    Returns:
    -- Cleaned sentence as a string.
    """
    sentence = filename.replace('_', ' ').replace('.wav', '')
    
    excluded_words = ['batch', 'proc', 'sample', 'audio']
    
    pattern_words = r'\b(?:' + '|'.join(excluded_words) + r')\b'
    pattern_numbers = r'\b\d+\b'
    
    sentence = re.sub(pattern_words, '', sentence, flags=re.IGNORECASE)
    
    sentence = re.sub(pattern_numbers, '', sentence)
    
    sentence = ' '.join(sentence.split())
    
    return sentence