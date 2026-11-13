"""The language model behind one small interface (an adapter over vLLM): the pipeline asks for chats with sampling params,
thinking on or off, a LoRA adapter (one of several) or the base weights, and a prefilled assistant turn. Nothing else in the
package imports vLLM."""
from pathlib import Path


class Engine:
    def __init__(self, model, *, tp=1, gpu_mem=0.9, context=32768, seed=0, quant=None, lora=None, engine_kwargs=None):
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        loras = [lora] if isinstance(lora, (str, Path)) else list(lora or [])  # one adapter or several (an ensemble)

        self.SamplingParams = SamplingParams
        self.llm = LLM(model=model, tensor_parallel_size=tp, gpu_memory_utilization=gpu_mem, max_model_len=context,
                       enable_prefix_caching=True, disable_log_stats=False, seed=seed, language_model_only=True,
                       **({"quantization": quant} if quant else {}),
                       **({"enable_lora": True, "max_lora_rank": 64, "max_loras": len(loras)} if loras else {}), **(engine_kwargs or {}))
        self.loras = [LoRARequest("sql" if i == 0 else f"sql{i + 1}", i + 1, str(p)) for i, p in enumerate(loras)]
        self.tokenizer = self.llm.get_tokenizer()

    def params(self, **kw):
        return self.SamplingParams(**kw)

    def chat(self, chats, params, *, thinking=False, lora=False, prefill=False):
        """One batched generation. lora: False = base weights, True = the first adapter, or one adapter index per chat (every
        adapter stays loaded, so one batch can mix them). params: one SamplingParams or one per chat. prefill: continue the
        last (assistant) message."""
        kw = {"chat_template_kwargs": {"enable_thinking": thinking}, "use_tqdm": False}
        if self.loras and lora is not False:
            kw["lora_request"] = self.loras[0] if lora is True else [self.loras[m] for m in lora]
        if prefill:
            kw.update(add_generation_prompt=False, continue_final_message=True)
        return self.llm.chat(chats, params, **kw)

    def n_tokens(self, chat):
        return len(self.tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_dict=False))
