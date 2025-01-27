### original <https://github.com/deepseek-ai/Janus/tree/main/janus/models>
### changes:
### - all imports should be relative
### - remove hardocded dtype=bfloat16 and device=cuda
### - remove dependency on obsolete attrdict package
### - use sdpa as default attention
### - create standard end-to-end txt2img pipeline
### - allow non-fixed image resolution: returns nonsense if not 384


from PIL import Image
from tqdm.rich import trange
import torch
import numpy as np
from diffusers import DiffusionPipeline
from transformers import AutoModelForCausalLM
from .image_processing_vlm import VLMImageProcessor # pylint: disable=unused-import
from .modeling_vlm import MultiModalityCausalLM
from .processing_vlm import VLChatProcessor


vl_chat_processor: VLChatProcessor = None
vl_gpt: MultiModalityCausalLM = None


class JanusPipeline(DiffusionPipeline):
    def __init__(self, repo_id: str, cache_dir: str, dtype: torch.dtype = torch.float16, device: torch.device = 'cuda'):
        global vl_chat_processor, vl_gpt # pylint: disable=global-statement
        self.compute_device = device
        self.compute_dtype = dtype
        super().__init__()
        if vl_chat_processor is None:
            vl_chat_processor = VLChatProcessor.from_pretrained(repo_id)
        self.tokenizer = vl_chat_processor.tokenizer
        if vl_gpt is None:
            vl_gpt = AutoModelForCausalLM.from_pretrained(
                repo_id,
                trust_remote_code=True,
                cache_dir=cache_dir,
                torch_dtype=self.compute_dtype,
                low_cpu_mem_usage=True,
                device_map="auto",
                # attn_implementation='sdpa' # sdpa, eager, flex_attention, flash_attention_2 # modified in class constructor
            )
            vl_gpt = vl_gpt.to(device=self.compute_device, dtype=self.compute_dtype)
            vl_gpt = vl_gpt.eval()

    @torch.inference_mode()
    def __call__(self,
                 prompt: str,
                 temperature: float = 1.0,
                 num_images_per_prompt: int = 1,
                 guidance_scale: float = 5,
                 width: int = 384,
                 height: int = 384,
                ):
        # pretty much hardcoded values
        patch_size: int = 16
        image_token_num_per_image: int = (width // patch_size) * (height // patch_size)

        # encode prompt
        conversation = [
            {"role": "User", "content": prompt},
            {"role": "Assistant", "content": ""},
        ]
        sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=vl_chat_processor.sft_format,
            system_prompt="",
        )
        prompt = sft_format + vl_chat_processor.image_start_tag
        input_ids = vl_chat_processor.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids)
        tokens = torch.zeros((num_images_per_prompt*2, len(input_ids)), dtype=torch.int).cuda()
        for i in range(num_images_per_prompt*2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = vl_chat_processor.pad_id
        inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
        generated_tokens = torch.zeros((num_images_per_prompt, image_token_num_per_image), dtype=torch.int).cuda()
        outputs = None

        # generate loop
        vl_gpt.language_model.model.to(self.compute_device)
        for i in trange(image_token_num_per_image):
            outputs = vl_gpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
            hidden_states = outputs.last_hidden_state
            logits = vl_gpt.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits = logit_uncond + guidance_scale * (logit_cond-logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)
            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)
        vl_gpt.language_model.model.to('cpu')

        # decode images
        dec = vl_gpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[num_images_per_prompt, 8, width // patch_size, height // patch_size])
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255)
        visual_img = np.zeros((num_images_per_prompt, width, height, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec
        outputs = []
        for i in range(num_images_per_prompt):
            image = Image.fromarray(visual_img[i])
            outputs.append(image)
        return outputs
