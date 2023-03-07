# Prepare dataset

Prepare data using nemo script

```shell
python NeMo/scripts/dataset_processing/get_librispeech_data.py test_clean,test_other
```

Leave long enough samples:

```shell
python filter_by_duration.py \
    --manifest path/to/manifest \
    --filtered_manifest path/to/filtered/manifest \
    --max_duration 13.5
```

Prepare noise (and maybe leave only pointsource_noises):

```shell
python NeMo/scripts/dataset_processing/get_openslr_rir_data.py
```

Augment clean dataset:

```shell
mkdir LibriSpeech-noised/test-clean
for snr in seq 40 -5 -10; do
python NeMo/scripts/dataset_processing/add_noise.py \
    --input_manifest=path/to/filtered/manifest \
    --noise_manifest=path/to/noise/dataset \
    --out_dir=LibriSpeech-noised/test-clean
    --snrs=40 35 30 25 20 15 10 5 0 "-5" "-10"
    --seed=42
    --num_workers=1
done
```

# RUN NEMO

```shell
sudo docker run -it --rm --gpus all --network host --ulimit memlock=-1 --ulimit stack=67108864 \
-v $(pwd):/eval -w /eval nvcr.io/nvidia/nemo:22.12
```

In docker buffered ctc:
```shell
python NeMo/examples/asr/asr_chunked_inference/ctc/speech_to_text_buffered_infer_ctc.py \
    pretrained_name="stt_en_conformer_ctc_large_ls" \
    dataset_manifest="path/to/manifest" \
    output_filename="recognitions_ctc.jsonl" \
    total_buffer_in_secs=9.0 \
    chunk_len_in_secs=4.5 \
    model_stride=4 \
    batch_size=1
```

In docker buffered rnnt:
```shell
python NeMo/examples/asr/asr_chunked_inference/rnnt/speech_to_text_buffered_infer_rnnt.py \
    pretrained_name="stt_en_conformer_transducer_large_ls" \
    dataset_manifest="<remove or path to manifest>" \
    output_filename="<remove or specify output filename>" \
    total_buffer_in_secs=9.0 \
    chunk_len_in_secs=4.5 \
    model_stride=4 \
    batch_size=1
    
# Longer Common Subsequence (LCS) Merge algorithm
python NeMo/examples/asr/asr_chunked_inference/rnnt/speech_to_text_buffered_infer_rnnt.py \
    pretrained_name="stt_en_conformer_transducer_large_ls" \
    dataset_manifest="<remove or path to manifest>" \
    output_filename="<remove or specify output filename>" \
    total_buffer_in_secs=9.0 \
    chunk_len_in_secs=4.5 \
    model_stride=4 \
    batch_size=1 \
    merge_algo="lcs"
```

