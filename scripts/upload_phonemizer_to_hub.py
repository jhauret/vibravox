import torch
import transformers

SENSOR = "rigid_in_ear_microphone"
RUN = "2024-06-15_14-23-20"
CHECKPOINT = "last.ckpt"

PATH = f"/home/jhauret/Downloads/stp/{SENSOR}/{RUN}/checkpoints/{CHECKPOINT}"

checkpoint = torch.load(PATH)

# Clean state_dict
state_dict_wav2vec2_for_ctc = {k.replace('wav2vec2_for_ctc.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('wav2vec2_for_ctc')}


feature_extractor = transformers.Wav2Vec2FeatureExtractor(feature_size=1,
                                                          sampling_rate=16000,
                                                          padding_value=0.0,
                                                          do_normalize=True,
                                                          return_attention_mask=False)
tokenizer = transformers.Wav2Vec2CTCTokenizer.from_pretrained("Cnam-LMSSC/vibravox-phonemes-tokenizer")
processor = transformers.Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# Load model
wav2vec2_for_ctc = transformers.Wav2Vec2ForCTC.from_pretrained(pretrained_model_name_or_path= 'facebook/wav2vec2-base-fr-voxpopuli-v2', #'facebook/wav2vec2-base'
                                                    apply_spec_augment= True,
                                                    ctc_loss_reduction= "mean",
                                                    attention_dropout= 0.1,
                                                    hidden_dropout= 0.1,
                                                    feat_proj_dropout= 0.1,
                                                    final_dropout= 0,
                                                    mask_time_prob= 0.05,
                                                    layerdrop= 0.05,
                                                    mask_feature_prob= 0.1024,
                                                    mask_feature_length= 64,
                                                    pad_token_id=35,
                                                    vocab_size=38)

wav2vec2_for_ctc.load_state_dict(state_dict_wav2vec2_for_ctc)

wav2vec2_for_ctc.push_to_hub(f"Cnam-LMSSC/phonemizer_{SENSOR}",
                             commit_message=f"Upload Wav2Vec2ForCTC after 10 epochs")
processor.push_to_hub(f"Cnam-LMSSC/phonemizer_{SENSOR}", commit_message=f"Upload standard Wav2Vec2Processor")

