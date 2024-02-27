# -----------------------------
# Imports and environment setup
# -----------------------------

# Essential imports for handling files, datasets, metrics, and neural network models
import os
import gc
import numpy as np
import json
from datasets import load_dataset, load_metric, concatenate_datasets, load_from_disk, Sequence, Audio, Features, Value
import torch
import pandas as pd
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor, HubertForCTC, \
    TrainingArguments, Trainer
import librosa
import random
from IPython.display import display, HTML
import re
import unidecode
from data_collator import DataCollatorCTCWithPadding
import wandb

# Setting directories for caching and temporary files
cache_dir = "/projects/0/einf2504/cache/"
temp_dir = "/projects/0/einf2504/temp/"
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_DATASETS_CACHE'] = cache_dir
os.environ['HF_HOME'] = cache_dir
os.environ['TMPDIR'] = temp_dir
os.environ['TEMP'] = temp_dir
os.environ['TMP'] = temp_dir

# Garbage collection and clearing CUDA cache to free memory
gc.collect()
torch.cuda.empty_cache()

# -----------------------------
# Data Preparation
# -----------------------------

# Loading custom interview dataset from disk
interview_data = load_from_disk("./interview_data")

# Function to convert json audio data to the required format
def convert_json(batch):
    # Processing the audio and metadata
    audio = json.loads(batch['audio'])
    audio['array'] = np.array(audio['array']).astype(np.float32)
    batch['audio'] = {"path": "test.wav", "array": audio['array'], "sampling_rate": audio['sampling_rate']}
    batch["sentence"] = batch["sentence"] or "[ERROR]"
    return batch

# Applying the conversion function to the interview dataset
interview_data = interview_data.map(convert_json, features=Features({
                                        "path": Value(dtype='string'),
                                        "audio": Audio(sampling_rate=16_000, decode=True),
                                        "sentence": Value(dtype='string')
                                    }))
interview_data = interview_data.filter(lambda example: example['sentence'] != "[ERROR]")

# Loading Common Voice dataset and removing unnecessary columns
common_voice_train = load_dataset("mozilla-foundation/common_voice_9_0", "nl", split="train+validation")
common_voice_test = load_dataset("mozilla-foundation/common_voice_9_0", "nl", split="test")
common_voice_train = common_voice_train.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
common_voice_test = common_voice_test.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
common_voice_train = common_voice_train.cast_column("audio", Audio(sampling_rate=16_000))

# Concatenating the interview data with Common Voice dataset
common_voice_train = concatenate_datasets([common_voice_train, interview_data])

# -------------------------------
# Data Cleaning and Tokenization
# -------------------------------

# Regular expression to remove special characters from sentences
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�=&\(\)]'

def remove_special_characters(batch):
    batch["sentence"] = unidecode.unidecode(batch["sentence"])
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).upper() + " "
    return batch

# Applying text cleaning to datasets
common_voice_train = common_voice_train.map(remove_special_characters)
common_voice_test = common_voice_test.map(remove_special_characters)

# Loading tokenizer for the Hubert model
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained('facebook/hubert-large-ls960-ft')

# Function to convert audio files to arrays
def speech_file_to_array_fn(batch):
    batch["speech"] = batch['audio']['array'].astype(np.float32)
    batch["sampling_rate"] = batch['audio']['sampling_rate']
    batch["target_text"] = batch["sentence"]
    return batch

# Applying the conversion function to datasets and removing original columns
common_voice_train = common_voice_train.map(speech_file_to_array_fn, remove_columns=common_voice_train.column_names)
common_voice_test = common_voice_test.map(speech_file_to_array_fn, remove_columns=common_voice_test.column_names)

# Resampling audio to 16kHz
def resample(batch):
    batch["speech"] = librosa.resample(np.asarray(batch["speech"]), orig_sr=batch["sampling_rate"], target_sr=16_000)
    batch["sampling_rate"] = 16_000
    return batch

common_voice_train = common_voice_train.map(resample)
common_voice_test = common_voice_test.map(resample)

# -----------------------------
# Feature Extraction
# -----------------------------

# Loading feature extractor for the Hubert model
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/hubert-large-ls960-ft')

# Creating processor by combining tokenizer and feature extractor
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# Function to prepare dataset for training
def prepare_dataset(batch):
    # Encoding the input values
    batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"]).input_values[0]
    # Encoding the labels
    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch

# Applying dataset preparation function
common_voice_train = common_voice_train.map(prepare_dataset, remove_columns=common_voice_train.column_names, batch_size=8, num_proc=4, batched=True)
common_voice_test = common_voice_test.map(prepare_dataset, remove_columns=common_voice_test.column_names, batch_size=8, num_proc=4, batched=True)

# -----------------------------
# Model Training
# -----------------------------

# Initializing the model for CTC tasks
model = HubertForCTC.from_pretrained('facebook/hubert-large-ls960-ft', gradient_checkpointing=True, ctc_loss_reduction="mean", pad_token_id=processor.tokenizer.pad_token_id)

# Setting up training arguments
training_args = TrainingArguments(
  output_dir="./hubert_finetuned_nl",
  group_by_length=True,
  per_device_train_batch_size=8,
  evaluation_strategy="steps",
  num_train_epochs=30,
  fp16=True,
  save_steps=500,
  eval_steps=500,
  logging_steps=500,
  learning_rate=1e-4,
  warmup_steps=500,
  save_total_limit=2,
)

# Custom data collator
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# Metric for evaluating model performance
wer_metric = load_metric("wer")

# Function to compute metrics
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# Initializing the trainer
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=common_voice_train,
    eval_dataset=common_voice_test,
    tokenizer=processor.feature_extractor,
)

# Training the model
trainer.train()
