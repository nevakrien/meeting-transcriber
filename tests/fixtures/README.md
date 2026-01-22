# Test Data

This directory contains sample audio files for testing the meeting transcriber functionality.

## Yoruba Speech Samples

The `yoruba_samples/` directory contains 15 sample WAV files from the OpenSLR Yoruba speech corpus (SLR86).

### Source

- **Dataset**: [OpenSLR SLR86 - Crowdsourced high-quality Yoruba speech data set](https://openslr.org/86/)
- **License**: Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
- **Format**: 16 kHz WAV files
- **Language**: Yoruba (Yorùbá)

### Sample Structure

```
yoruba_samples/
├── yom_00295_00677272420.wav    # Male speaker
├── yom_00295_00700005909.wav    # Male speaker  
├── yom_00295_02006503027.wav    # Male speaker
├── yom_00610_01446001123.wav    # Male speaker
├── yom_00610_01451481032.wav    # Male speaker
├── yom_00610_01486871372.wav    # Male speaker
├── yom_01208_00531224427.wav    # Male speaker
├── yom_01208_00748619928.wav    # Male speaker
├── yom_01208_01378886596.wav    # Male speaker
├── yom_01523_00944464995.wav    # Male speaker
├── yom_01523_01689055566.wav    # Male speaker
├── yom_01523_02104085397.wav    # Male speaker
├── yom_02121_01252162815.wav    # Male speaker
├── yom_02436_01950161981.wav    # Male speaker
└── yom_03034_01418840084.wav    # Male speaker
```

### Transcripts

The corresponding transcripts are available in:
- `line_index.tsv` - Combined transcripts
- `line_index_female.tsv` - Female speaker transcripts

### Usage for Testing

These samples can be used to validate:
- Audio segmentation algorithms
- Speaker diarization (single vs multiple speakers)
- Transcription accuracy
- Handling of audio artifacts and disfluencies

### Example Usage

```bash
# Test audio segmentation
python meeting_transcriber.py split test_data/yoruba_samples/yom_00295_00677272420.wav

# Test diarization
python meeting_transcriber.py diarize test_data/yoruba_samples/yom_00295_00677272420.wav
```