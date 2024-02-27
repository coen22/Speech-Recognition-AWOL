# Import necessary libraries
import asyncio
import base64
import heapq
import math
from numpy import array
import numpy as np
import soundfile as sf
import torch.nn.functional
import torch
import os
import json
from math import log

from scipy.io import wavfile
from transformers import (
    HubertForCTC,  # Model for speech recognition
    Wav2Vec2Processor,  # Processor for handling audio files
    RobertaForMaskedLM,  # Language model for understanding context
    RobertaTokenizer  # Tokenizer for the language model
)
from split import segment  # Function to segment long audio files
from tempfile import NamedTemporaryFile  # For creating temporary files

# Predefined model names and paths
lm_name = "pdelobelle/robbert-v2-dutch-base"
tokenizer_name = "facebook/hubert-large-ls960-ft"
model_name = "coen22/Speech-Recognition-AWO-L"

# Directory setup for model loading
dirname = os.path.dirname(__file__)
model_path = os.path.join(dirname, model_name)

# Beam search parameters
beam_size = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

working = False  # Flag to prevent concurrent predictions

# Model and tokenizer declarations
processor: Wav2Vec2Processor = None
model: HubertForCTC = None
tokenizer: RobertaTokenizer = None
lm_model: RobertaForMaskedLM = None

# Initialize models and tokenizer
def init():
    global processor, model, tokenizer, lm_model
    print("Loading am")  # Automatic speech recognition model
    processor = Wav2Vec2Processor.from_pretrained(tokenizer_name)
    model = HubertForCTC.from_pretrained(model_path).to(device)

    print("Loading lm")  # Language model
    tokenizer = RobertaTokenizer.from_pretrained(lm_name)
    lm_model = RobertaForMaskedLM.from_pretrained(lm_name)
    print("Done")

# Calculate language model probability
def lm_prob(sentence):
    if tokenizer is None:
        init()
    tokenize_input = tokenizer(sentence, return_tensors='pt')
    output = lm_model(**tokenize_input)
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(output.logits.squeeze(), tokenize_input.input_ids.squeeze()).data
    return loss.item()

# Beam search decoder for improving prediction accuracy
def beam_search_decoder(data, k):
    sequences = [[[], 0.0]]
    for row in data:
        all_candidates = []
        k_largest = heapq.nlargest(k, range(len(row)), row.take)
        for seq, score in sequences:
            for j in k_largest:
                s = score - math.log(row[j])
                candidate = [seq + [j], s]
                all_candidates.append(candidate)
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        sequences = ordered[:k]
    return sequences

# Main prediction function
def predict(audio, sampling_rate=16000):
    if len(audio) > 160_000:
        return '<too long segment>'
    tokens = processor(audio, sampling_rate=sampling_rate, return_tensors='pt').to(device)
    logits = model(tokens.input_values, tokens.attention_mask).logits

    # Normalize logits for processing
    min_range = torch.min(logits)
    max_range = torch.max(logits)
    logits = (logits - min_range) / (max_range - min_range)
    logits = array(logits[0].cpu().detach().numpy())
    predicted_ids = beam_search_decoder(logits, beam_size)

    # Decode predictions
    with processor.as_target_processor():
        scores = []
        transcriptions = []
        for predicted_id in predicted_ids:
            transcription = processor.decode(predicted_id[0])
            lm_score = lm_prob(transcription)
            scores.append(0.5 * lm_score + 0.5 * predicted_id[1])
            transcriptions.append(transcription)
    best_idx = np.argmax(scores)
    if transcriptions[best_idx] != '':
        return transcriptions[best_idx]

# Asynchronous prediction for handling file input
async def predict_file(filename="test_data/seq_pauze.wav"):
    global working
    if working:
        print("Waiting for other task")
        await asyncio.sleep(5)
    working = True
    if tokenizer is None:
        init()
    output = []
    for seg in segment(filename):
        if len(seg) < 2_048:
            print("Too short")
            continue
        torch_arr = torch.FloatTensor(seg)
        label = predict(torch_arr)
        try:
            with NamedTemporaryFile(delete=True, suffix=".wav", mode="wb+") as f:
                sf.write(f, seg, 16_000)
                f.seek(0)
                output.append({
                    "audio": base64.b64encode(f.read()).decode("utf-8"),
                    "label": label
                })
        except:
            print("Failed to add file")
    working = False
    return output

# Asynchronous wrapper for streaming predictions
async def predict_file_async(filename="test_data/seq_pauze.wav"):
    for seg in segment(filename):
        if len(seg) < 2_048:
            print("Too short")
            continue
        torch_arr = torch.FloatTensor(seg)
        label = predict(torch_arr)
        with NamedTemporaryFile(delete=True, suffix=".wav", mode="wb+") as f:
            sf.write(f, seg, 16_000)
            f.seek(0)
            yield json.dumps({
                "audio": base64.b64encode(f.read()).decode("utf-8"),
                "label": label
            })

# Main async function to execute prediction
async def main():
    result = await predict_file()
    print(result)

# Entry point for script execution
if __name__ == "__main__":
    asyncio.run(main())
