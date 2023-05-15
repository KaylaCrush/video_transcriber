import whisper
import datetime

import subprocess

import torch
import pyannote.audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment

import wave
import contextlib

from sklearn.cluster import AgglomerativeClustering
import numpy as np
import os

video_files = os.listdir('video')

import moviepy.editor as mp
def convert(base_filename):
    clip = mp.VideoFileClip(rf'video/{base_filename}.mp4')
    clip.audio.write_audiofile(rf'audio/{base_filename}.wav')

for video_file in video_files:
    base_filename = video_file[:-4]

    convert(base_filename)

    path = f'audio/{base_filename}.wav'

    num_speakers = 2  # @param {type:"integer"}

    language = 'English'  # @param ['any', 'English']

    model_size = 'medium'  # @param ['tiny', 'base', 'small', 'medium', 'large']

    model_name = model_size
    if language == 'English' and model_size != 'large':
        model_name += '.en'

    embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb")

    model = whisper.load_model(model_size)

    result = model.transcribe(path)
    segments = result["segments"]

    with contextlib.closing(wave.open(path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    audio = Audio()

    def segment_embedding(segment):
        start = segment["start"]
        # Whisper overshoots the end timestamp in the last segment
        end = min(duration, segment["end"])
        clip = Segment(start, end)
        waveform, sample_rate = audio.crop(path, clip)
        return embedding_model(waveform[None])

    embeddings = np.zeros(shape=(len(segments), 192))
    for i, segment in enumerate(segments):
        embeddings[i] = segment_embedding(segment)

    embeddings = np.nan_to_num(embeddings)

    clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
    labels = clustering.labels_
    for i in range(len(segments)):
        segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

    def time(secs):
        return datetime.timedelta(seconds=round(secs))

    f = open(f"transcripts/{base_filename}.txt", "w")

    for (i, segment) in enumerate(segments):
        if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
            f.write("\n" + segment["speaker"] + ' ' +
                    str(time(segment["start"])) + '\n')
        f.write(segment["text"][1:] + ' ')
    f.close()
