# Nanjing Voice Cloning Project (av4_project)

## Project Overview

This project aims to clone a specific Nanjing dialect accent from an
elderly male speaker and transfer it to a young female voice using
advanced voice conversion techniques. The final output is a unique
\"Nanjing young female voice\" that preserves the authentic dialect
characteristics while maintaining a natural feminine timbre.

## Technical Architecture

### Core Technologies

- **OpenVoice V2**: Instant voice cloning framework with tone color
  conversion
- **MeloTTS**: High-quality multi-lingual text-to-speech engine
- **PyTorch**: Deep learning framework for model inference
- **Librosa**: Audio processing and analysis
- **SoundFile**: Audio file I/O operations

### Key Components

#### 1. Voice Feature Extraction (`extract_accent.py`)

    # Core function: Speaker Embedding Extraction
    se_extractor.get_se(audio_path, tone_color_converter, vad=True)

- **Purpose**: Extracts voiceprint features (speaker embedding) from
  reference audio
- **Method**: Uses VAD (Voice Activity Detection) to remove background
  noise
- **Output**: `.pt` file containing 256-dimensional speaker embedding
  vector
- **Deep Learning**: Leverages pre-trained speaker verification model

#### 2. Tone Color Conversion (`nanjing_female.py`)

    # Core class: ToneColorConverter
    tone_color_converter = ToneColorConverter(config_path, device)
    tone_color_converter.convert(audio_src_path, src_se, tgt_se, output_path)

- **Purpose**: Transfers voice characteristics from source to target
- **Mechanism**: Uses adversarial networks to disentangle and recombine
  voice attributes
- **Key Parameters**:
  - `src_se`: Source speaker embedding (young female base)
  - `tgt_se`: Target speaker embedding (Nanjing accent)
  - `accent_strength`: Blending ratio between source and target

#### 3. Voice Blending Algorithm

    # Voice feature mixing
    blended_se = base_se * (1 - accent_strength) + nanjing_accent * accent_strength

- **Mathematical Foundation**: Linear interpolation in high-dimensional
  voice space
- **Effect**: Controls how much of the target accent bleeds into the
  source voice
- **Range**: 0.0 (pure source) to 1.0 (pure target)

#### 4. Text-to-Speech Base Generation

    # MeloTTS engine
    tts = TTS(language='ZH', device=device)
    tts.tts_to_file(text, speaker_id, output_path, speed)

- **Model**: Multi-speaker Tacotron2 variant
- **Languages**: Supports Chinese, English, Japanese, Korean
- **Voice Options**: Multiple pre-trained speaker embeddings (ZH-女声1,
  ZH-少女, etc.)

## Dependency Stack

### Core Dependencies

    # Deep Learning Framework
    torch>=2.0.0
    torchaudio>=2.0.0

    # Voice Conversion
    openvoice==1.0.0 (local installation)
    git+https://github.com/myshell-ai/MeloTTS.git

    # Audio Processing
    librosa==0.9.1
    soundfile>=0.12.0
    pydub==0.25.1
    av>=10.0.0

    # Data Processing
    numpy>=1.23.0
    scipy>=1.9.0

    # Utilities
    tqdm>=4.65.0
    huggingface-hub>=0.16.0

### Secondary Dependencies (from OpenVoice requirements)

    wavmark==0.0.3          # Audio watermarking (optional)
    eng_to_ipa==0.0.2       # English to IPA conversion
    inflect==7.0.0          # English number inflection
    unidecode==1.3.7        # Unicode normalization
    pypinyin==0.50.0        # Chinese to pinyin
    cn2an==0.5.22           # Chinese number conversion
    jieba==0.42.1           # Chinese word segmentation
    gradio==3.48.0          # Web interface (optional)
    langid==1.1.6           # Language detection

## Model Architecture

### OpenVoice V2 Structure

    checkpoints_v2/
    ├── converter/
    │   ├── config.json           # Model configuration
    │   └── checkpoint.pth         # 131MB - Main conversion model
    └── base_speakers/
        └── ses/
            └── zh.pth             # 256-dim Chinese speaker embedding

### Key Deep Learning Concepts

#### 1. Speaker Embedding

- **Definition**: A fixed-length vector representation of a speaker\'s
  voice characteristics
- **Dimension**: 256 floats
- **Extraction**: Uses pre-trained speaker verification network
- **Property**: Similar speakers have similar embeddings (cosine
  distance)

#### 2. Tone Color Conversion

- **Input**: Source audio + source embedding + target embedding
- **Process**:
  1.  Encoder extracts content features from source audio
  2.  Decoder reconstructs audio conditioned on target embedding
  3.  Adversarial training ensures naturalness
- **Output**: Audio with source content + target voice characteristics

#### 3. Voice Blending Mathematics

    Let:
    - E_base = Base speaker embedding (young female)
    - E_nanjing = Nanjing accent embedding (elderly male)
    - α = Accent strength (0.3 typical)

    Blended embedding:
    E_blended = (1-α)·E_base + α·E_nanjing

    Result: Linear interpolation in voice space

## Installation & Usage

### Environment Setup

    # Create conda environment
    conda create -n av4_310 python=3.10 -y
    conda activate av4_310

    # Install dependencies
    conda install -c conda-forge numpy librosa pydub av -y
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install gradio==3.48.0
    pip install git+https://github.com/myshell-ai/MeloTTS.git
    pip install -e OpenVoice/

### Voice Extraction

    # Extract Nanjing accent from reference audio
    python extract_accent.py reference_speaker.mp3
    # Output: nanjing_accent.pt

### Voice Generation

    # Generate speech with blended voice
    python nanjing_female.py "Text to speak" -o output.wav -a 0.35 -v ZH-少女

    # Batch process text files
    python nanjing_girl_file.py input.txt -a 0.35 -o output.wav

## Performance Considerations

### Device Support

- **CPU**: Works but slower, \~5x real-time for generation
- **GPU**: CUDA-enabled devices recommended, \~0.5x real-time

### Memory Usage

- **RAM**: \~2-3GB during inference
- **Disk**: \~500MB for models + dependencies
- **GPU Memory**: \~1.5GB if using CUDA

### Latency

- **Feature Extraction**: 30-60 seconds per minute of audio
- **Voice Generation**: 10-20 seconds per 100 characters
- **First Run**: Additional 2-5 minutes for model downloads

## Troubleshooting

### Common Issues

#### 1. AV ImportError

    conda install -c conda-forge av -y

#### 2. PyTorch CUDA Issues

    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

#### 3. Model Download Failures

    export HF_ENDPOINT=https://hf-mirror.com

#### 4. Audio Volume Fluctuations

    # Normalize audio after generation
    ffmpeg -i input.wav -af loudnorm=I=-16:LRA=11:TP=-1.5 output_norm.wav

## References

- [OpenVoice GitHub](https://github.com/myshell-ai/OpenVoice)
- [MeloTTS GitHub](https://github.com/myshell-ai/MeloTTS)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Librosa Documentation](https://librosa.org/doc/latest/index.html)

## License

This project uses open-source components with various licenses. Please
refer to individual component licenses for usage terms.

## Acknowledgments

- MyShell AI for OpenVoice V2
- HuggingFace for model hosting
- Conda-Forge community for Windows binary support

::: note
**Note:** This project was built and tested on Windows 10/11 with MSYS2
and Anaconda. The accent strength parameter (α) was empirically
determined to be 0.35 for optimal balance between dialect preservation
and naturalness.
:::
