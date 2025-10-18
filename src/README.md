# **`1_steering_locate_features.py`**
This script identifies and scores latent features within a SARM (Sparse Autoencoder Reward Model) that are correlated with a specific concept, such as safety.

Core Functionality:
The script computes a score for each latent feature by analyzing the model's activations on paired comparison data (e.g., safe vs. unsafe responses). This score quantifies the feature's tendency to activate on contexts representing the positive concept (e.g., "safety").

Methodology:
1.  **Data Processing**: The script processes a dataset containing paired examples, where each entry has a "chosen" (positive/safe) and a "rejected" (negative/unsafe) response.
2.  **Feature Extraction**: It uses a **forward hook** in PyTorch to capture intermediate latent activations from a specific layer of the model. This method is non-intrusive and does not require modifying the model's source code.
3.  **Score Computation**: The activations are averaged across the dataset to obtain a mean activation vector for the chosen concept, $c$, and one for the rejected concept, $j$. The feature score, $s$, is then calculated using the following formula:

    $$s = \frac{c - j}{c + j + C}$$

    Where:
    - $c, j, s \in \mathbb{R}^{M}$, and $M$ is the dimension of latent features (e.g., 65536).
    - $c$: The mean activation vector for features on "chosen" samples.
    - $j$: The mean activation vector for features on "rejected" samples.
    - $C$: A constant added for numerical stability.

Interpretation of Scores:
- A high positive score $s_i$ for the $i$-th feature indicates that it activates significantly more for "chosen" (safe) inputs than for "rejected" (unsafe) ones. Such a feature can be interpreted as a **safety-promoting feature**.

Example Usage:
```shell
python src/1_steering_locate_features.py \
    --data_path steering/train/rm_bench/safety.jsonl \
    --device cuda:1 \
    --model_path Schrieffer/Llama-SARM-4B \
    --output_file steering/scores/safety_scores.pt
```

# **`2_steering_test.py`**

This script tests the effectiveness of steering vectors by applying them to a SARM model's latent activations and measuring the resulting change in reward scores.

#### Core Functionality

The script conducts a controlled experiment to verify whether manually modifying specific latent features—a process called **"steering"**—produces a predictable and desired shift in the model's final reward output. For each input, it calculates a score *before* and *after* the intervention, allowing for a direct comparison of the steering vector's impact.

#### Methodology

1.  **Load Steering Vector**: The script first loads a steering configuration from a JSON file. This file specifies which latent features to target and what modification to apply (e.g., amplify a feature by 2x, suppress it by 0.5x).

2.  **Intervention with Hook**: It uses a **forward pre-hook** on the model's final scoring layer. This hook acts as an interception mechanism, modifying the latent feature activations (`z`) in-flight *before* they are used to calculate the final reward score. The intervention is based on the principle that in a linear model where the reward `r` is a weighted sum of activations ($r = \sum z_i \cdot w_i$), modifying the activation `z_i` is a simple and effective proxy for modifying the weight `w_i`.

3.  **A/B Comparison**: For every sample in the test dataset, the script performs two forward passes:

      * **Before**: Calculates the reward score with the original, unmodified latent activations.
      * **After**: Calculates the reward score using the latent activations that have been altered by the steering hook.

#### Interpretation of Output

  * A **significant shift** in the score distribution on the KDE plot indicates that the steering was effective. For example, if the goal was to reduce scores for unsafe content, the "after" distribution should be clearly shifted to the left (lower values) of the "before" distribution.
  * When run on a non-target dataset, **minimal change** between the two distributions demonstrates the **specificity** of the steered features, confirming they don't unduly affect unrelated inputs.

#### Example Usage

```shell
python src/2_steering_test.py \
    --steering_path steering_latents.json \
    --data_path steering/test/rewardbenchv2/safety_c.jsonl \
    --model_path Schrieffer/Llama-SARM-4B \
    --output_path scored.jsonl \
    --device cuda:0
```