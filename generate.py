import sys
import time
import logging
import torch
from model import JanusPipeline


# repo_id = 'deepseek-ai/Janus-1.3B'
repo_id = 'deepseek-ai/Janus-Pro-7B'
cache_dir = '/mnt/models/huggingface'
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    prompt = sys.argv[1]
    log.info(f'prompt={prompt}')
    t0 = time.time()
    pipe = JanusPipeline(
        repo_id,
        cache_dir=cache_dir,
        dtype=torch.bfloat16,
        device='cuda',
    )
    t1 = time.time()
    log.info(f'load time={t1-t0}')
    images = pipe(
        prompt=prompt,
        num_images_per_prompt=1,
        temperature=1.0,
        guidance_scale=5.0,
    )
    t2 = time.time()
    log.info(f'generate time={t2-t1}')
    for i, image in enumerate(images):
        fn = f'/tmp/janus-{i}.png'
        log.info(f'image={i} file={fn}')
        image.save(fn)
