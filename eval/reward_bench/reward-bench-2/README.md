---
language:
- en
license: odc-by
size_categories:
- 1K<n<10K
task_categories:
- question-answering
dataset_info:
  features:
  - name: id
    dtype: string
  - name: prompt
    dtype: string
  - name: chosen
    sequence: string
  - name: rejected
    sequence: string
  - name: num_correct
    dtype: int64
  - name: num_incorrect
    dtype: int64
  - name: total_completions
    dtype: int64
  - name: models
    sequence: string
  - name: subset
    dtype: string
  - name: additional_metadata
    struct:
    - name: category
      dtype: string
    - name: correct
      dtype: string
    - name: index
      dtype: float64
    - name: instruction_id_list
      sequence: string
    - name: label
      dtype: string
    - name: method
      dtype: string
    - name: models
      sequence: string
    - name: prompt_norm
      dtype: string
    - name: subcategory
      dtype: string
    - name: valid
      dtype: float64
  splits:
  - name: test
    num_bytes: 13772499
    num_examples: 1865
  download_size: 6973189
  dataset_size: 13772499
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
---
<!-- <img src="https://huggingface.co/spaces/allenai/reward-bench/resolve/main/src/logo.png" alt="RewardBench Logo" width="800" style="margin-left:'auto' margin-right:'auto' display:'block'"/> -->

[Code](https://github.com/allenai/reward-bench) | [Leaderboard](https://huggingface.co/spaces/allenai/reward-bench) | [Results](https://huggingface.co/datasets/allenai/reward-bench-2-results) | [Paper](https://arxiv.org/abs/2506.01937)

# RewardBench 2 Evaluation Dataset Card

The RewardBench 2 evaluation dataset is the new version of RewardBench that is based on unseen human data and designed to be substantially more difficult! RewardBench 2 evaluates capabilities of reward models over the following categories:
1. **Factuality** (*NEW!*): Tests the ability of RMs to detect hallucinations and other basic errors in completions.
2. **Precise Instruction Following** (*NEW!*): Tests the ability of RMs to judge whether text follows precise instructions, such as "Answer without the letter u".
3. **Math**: Tests RMs' abilities at math, on open-ended human prompts ranging from middle school physics and geometry to college-level chemistry, calculus, combinatorics, and more.
4. **Safety**: Tests RMs' abilities to correctly comply with or refuse prompts related to harmful use cases as well as general compliance behaviors.
5. **Focus**: Tests RMs' ability to detect high-quality, on-topic answers to general user queries.
6. **Ties** (*NEW*!): This new type of subset tests the robustness of RMs in domains with many possible similar answers. For example, the question "Name a color of the rainbow" has seven possible correct answers and infinitely many incorrect ones.

The RewardBench 2 leaderboard averages over these six subsets.
For the first five categories, the scoring for RewardBench 2 evaluates success as whether the score of a prompt-chosen pair is greater than the score of *three* prompt-rejected pairs. 
The "Ties" score is a weighted score of accuracy (as measured by *all* valid correct answers being scored higher than *all* incorrect answers) and whether the reward margin between correct and incorrect answers exceeds that of the highest and lowest-scored correct responses. This metric rewards not only correctness, but also a model's ability to prioritize correct answers over incorrect ones more strongly than it distinguishes between equally valid correct responses.

<img src="https://huggingface.co/datasets/allenai/blog-images/resolve/main/reward-bench/main-fig-hor.png" alt="RewardBench 2 Flow" width="800" style="margin-left:'auto' margin-right:'auto' display:'block'"/>

## Dataset Construction Summary
| Domain | Count | Prompt Source | Method of generating completions | Completion Filtering |
|--------|-------|---------------|----------------------------------|---------------------|
| Factuality | 475 | Human | Both | Multi-LM-as-a-judge |
| Precise IF | 160 | Human | Natural | Verifier functions |
| Math | 183 | Human | Natural | Majority voting |
| Safety | 450 | CoCoNot | Both | LM-as-a-judge & rubrics |
| Focus | 495 | Human | System Prompt Variation | N/A |
| Ties | 102 | Manual | System Prompt Variation | Manual verification |

## Dataset Details

Each sample in the dataset has the following items.
Note, the dataset is single-turn:
* `prompt` (`str`): the instruction given in the various test sets.
* `chosen` (`list[str]`): the chosen response(s) (1 chosen response for all subsets but ties)
* `rejected` (`list[str]`): the rejected responses (3 chosen responses for all subsets but ties)
* `num_correct` (`int`): the number of chosen responses
* `num_rejected` (`int`): the number of rejected responses
* `total_completions` (`int`): the total number of responses
* `models` (`list[str]`): a list of models that the chosen and rejected responses are generated from, respectively
* `subset` (`str`): the subset the datapoint is part of.
* `id` (`int`): an incremented id for every prompt in the benchmark.

To select a specific subset use HuggingFace Datasets `.filter` functionality.
```
dataset = dataset.filter(lambda ex: ex["subset"] == "Factuality")
```

## Models Used
We generated completions from the following models:
- [Mistral 7B Instruct v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3) (Apache 2.0)
- [Tulu 3 8B](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B) (Llama 3.1 Community License Agreement)
- [Tulu 3 70B](https://huggingface.co/allenai/Llama-3.1-Tulu-3-70B) (Llama 3.1 Community License Agreement)
- [Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) (Llama 3.1 Community License Agreement)
- [Llama 3.1 70B Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct) (Llama 3.1 Community License Agreement)
- [Llama 3.2 1B Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) (Llama 3.2 Community License Agreement)
- [Llama 2 7B Chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) (Llama 2 Community License Agreement)
- [Tulu 2 70B](https://huggingface.co/allenai/tulu-2-dpo-70b) (Ai2 ImpACT Low Risk License)
- [Qwen2.5 72B Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) (Qwen License Agreement)
- [Qwen2.5 7B Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) (Apache 2.0)
- [Qwen2.5 14B Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) (Apache 2.0)
- [Qwen2.5 0.5B Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) (Apache 2.0)
- [Qwen2.5 Math 72B Instruct](https://huggingface.co/Qwen/Qwen2.5-Math-72B-Instruct) (Qwen License Agreement)
- [Qwen2.5 Math 7B Instruct](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct) (Apache 2.0)
- [Deepseek Math 7B RL](https://huggingface.co/deepseek-ai/deepseek-math-7b-rl) (This model is licensed under the Deepseek License. Any use of the outputs from this model must be in accordance with the use restrictions in the [Deepseek License](https://github.com/deepseek-ai/DeepSeek-Math/blob/main/LICENSE-MODEL).)
- [OLMoE 1B 7B 0924 Instruct](https://huggingface.co/allenai/OLMoE-1B-7B-0924) (Apache 2.0)
- [Dolphin 2.0 Mistral 7b](https://huggingface.co/cognitivecomputations/dolphin-2.0-mistral-7b) (Apache 2.0)
- [Zephyr 7b Beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) (MIT License)
- GPT-4o (Outputs produced by GPT-4 are subject to OpenAI's [terms of use](https://openai.com/policies/row-terms-of-use/))
- Claude 3.5 Sonnet (Outputs produced by Claude are subject to Anthropic [terms of service](https://www.anthropic.com/legal/consumer-terms) and [usage policy](https://www.anthropic.com/legal/aup))

## License
This dataset is licensed under ODC-BY. It is intended for research and educational use in accordance with Ai2's [Responsible Use Guidelines](https://allenai.org/responsible-use). This dataset includes output data generated from third party models that are subject to separate terms governing their use.

## Trained Reward Models
We also trained and released several reward models— check out the [RewardBench 2 Collection](https://huggingface.co/collections/allenai/reward-bench-2-683d2612a4b3e38a3e53bb51) to use them!

## Citation
```
@misc{malik2025rewardbench2advancingreward,
      title={RewardBench 2: Advancing Reward Model Evaluation}, 
      author={Saumya Malik and Valentina Pyatkin and Sander Land and Jacob Morrison and Noah A. Smith and Hannaneh Hajishirzi and Nathan Lambert},
      year={2025},
      eprint={2506.01937},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.01937}, 
}
```