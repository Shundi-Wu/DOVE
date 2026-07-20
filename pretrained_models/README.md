Place pretrained models here.

Supported training backbones:

- `CogVideoX1.5-5B/`: `zai-org/CogVideoX1.5-5B`
- `Wan2.1-T2V-1.3B-Diffusers/`: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`

Download the Wan model from the repository root with:

```bash
hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --local-dir pretrained_models/Wan2.1-T2V-1.3B-Diffusers
```

The resulting directory must contain these Diffusers components:

```text
Wan2.1-T2V-1.3B-Diffusers/
├── model_index.json
├── scheduler/
├── text_encoder/
├── tokenizer/
├── transformer/
└── vae/
```

The training scripts keep model-specific empty-prompt embeddings under
`prompt_embeddings/<backbone-name>/`; these caches are generated automatically
on the first run and must not be shared between CogVideoX and Wan.
