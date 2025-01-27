# janus

original: <https://github.com/deepseek-ai/Janus/tree/main/janus/models>

changes:
- all imports should be relative
- remove hardcoded `dtype=bfloat16` and `device=cuda`
- remove dependency on obsolete `attrdict` package
- use `sdpa` as default attention instead `flash-attention-v1`
- create standard end-to-end txt2img pipeline
- add simple `generate.py` demo
