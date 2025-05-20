# PDF Path
pdf_path = "Dataset/human-nutrition-text.pdf"

import os
import requests
import pandas as pd
import random
import fitz
import fitz
import re
import time
import torch
import textwrap
import random
import numpy as np
import matplotlib.pyplot as plt
import requests
import urllib.request as request
import pickle


# for progress bars, requires !pip install tqdm
from tqdm.auto import tqdm

# Requires !pip install sentence-transformers
from sentence_transformers import SentenceTransformer, util

# Importing Transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available

# To Calculate Runtimes
from time import perf_counter as timer

# For NLP and Cleanup and Data Prep
from spacy.lang.en import English

# HuggingFaceImports
from langchain_community.embeddings import HuggingFaceEmbeddings
from huggingface_hub import hf_hub_download

from pathlib import Path
from dataclasses import dataclass

from time import perf_counter as timer

# Result Limitations
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Pytorch and device Setup as GPU

# setting device for embedding as GPU
# Please make sure you install the correct version of torch which is for GPU and not CPU
# As i am using RTX 2080 i have cu118 this may defer depending on the hardware of the machine
# pip install torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
device = "cuda" if torch.cuda.is_available() else "cpu"
