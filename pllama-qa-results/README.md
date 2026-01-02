---
base_model: radlab/pLLama3.2-1B
library_name: peft
model_name: polka-short-qa-results
tags:
- base_model:adapter:radlab/pLLama3.2-1B
- lora
- sft
- transformers
- trl
licence: license
pipeline_tag: text-generation
---

# Model Card for polka-short-qa-results

This model is a fine-tuned version of [radlab/pLLama3.2-1B](https://huggingface.co/radlab/pLLama3.2-1B).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 


This model was trained with SFT.

### Framework versions

- PEFT 0.18.0
- TRL: 0.26.2
- Transformers: 4.57.3
- Pytorch: 2.11.0.dev20260102
- Datasets: 4.4.2
- Tokenizers: 0.22.1

## Citations



Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```