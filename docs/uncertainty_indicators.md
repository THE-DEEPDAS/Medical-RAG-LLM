# Understanding Uncertainty Indicators

Uncertainty indicators help us understand how confident a model is about its answers. Think of it like asking a friend a question. If your friend hesitates or seems unsure, you know they might not be confident in their answer. Similarly, uncertainty indicators measure how "sure" the AI model is about its response.

## What Are Uncertainty Indicators?

Uncertainty indicators are numbers or scores that tell us how confident the AI model is about its response. If the score is high, the model is less confident. If the score is low, the model is more confident.

### Example

Imagine you ask the AI, "What is the purpose of Paracetamol?" The AI generates a response like this:

> "Paracetamol is used for pain relief and fever reduction."

Now, the AI also calculates how confident it is about each word in the response. For example:

- "Paracetamol" → 90% confident
- "is" → 95% confident
- "used" → 85% confident
- "for" → 80% confident
- "pain" → 70% confident
- "relief" → 75% confident
- "and" → 90% confident
- "fever" → 85% confident
- "reduction" → 80% confident

The AI then calculates the **average confidence** of all these words. If the average confidence is low, it means the AI is uncertain about its response.

---

## Indicators We Use

1. **Token Probabilities**:

   - Each word (or "token") in the response has a probability score.
   - This score tells us how confident the AI is about that specific word.

2. **Mean Token Probability**:

   - This is the average of all the token probabilities in the response.
   - A higher average means the AI is more confident overall.

3. **Uncertainty Score**:
   - This is calculated as `1 - Mean Token Probability`.
   - A higher uncertainty score means the AI is less confident.

---

## Why Do We Use These Indicators?

- **Token Probabilities**:

  - They give us detailed insights into which parts of the response the AI is confident about and which parts it is unsure about.

- **Mean Token Probability**:

  - It provides a simple, overall measure of confidence for the entire response.

- **Uncertainty Score**:
  - It normalizes the confidence into a single value, making it easy to compare different responses.

---

## Code Implementation

Here is the Python code we use to calculate uncertainty indicators:

```python
# filepath: f:\Wearables\Medical-RAG-LLM\rag.py
import numpy as np

def evaluate_uncertainty(response: str) -> float:
    """Evaluate the uncertainty of the LLM's response using token probabilities."""
    # Split the response into individual words (tokens)
    tokens = response.split()

    # Simulate token probabilities (in a real system, these would come from the LLM)
    token_probs = [np.random.uniform(0.7, 1.0) for _ in tokens]

    # Calculate the mean probability of all tokens
    mean_probability = np.mean(token_probs)

    # Calculate the uncertainty score as 1 - mean probability
    uncertainty_score = 1 - mean_probability

    return uncertainty_score
```

---

## Line-by-Line Explanation

1. **`import numpy as np`**:

   - We use the `numpy` library to perform mathematical calculations, like finding the average of numbers.

2. **`def evaluate_uncertainty(response: str) -> float:`**:

   - This defines a function called `evaluate_uncertainty` that takes a response (a string) as input and returns a float (the uncertainty score).

3. **`tokens = response.split()`**:

   - This splits the response into individual words (tokens). For example, "Paracetamol is used for pain relief" becomes `["Paracetamol", "is", "used", "for", "pain", "relief"]`.

4. **`token_probs = [np.random.uniform(0.7, 1.0) for _ in tokens]`**:

   - This simulates token probabilities by generating random numbers between 0.7 and 1.0 for each token. In a real system, these probabilities would come from the AI model.

5. **`mean_probability = np.mean(token_probs)`**:

   - This calculates the average (mean) of all the token probabilities. For example, if the probabilities are `[0.9, 0.8, 0.7]`, the mean is `(0.9 + 0.8 + 0.7) / 3 = 0.8`.

6. **`uncertainty_score = 1 - mean_probability`**:

   - This calculates the uncertainty score by subtracting the mean probability from 1. A higher uncertainty score means the AI is less confident.

7. **`return uncertainty_score`**:
   - This returns the calculated uncertainty score.

---

## Example Usage

```python
response = "Paracetamol is used for pain relief and fever reduction."
uncertainty_score = evaluate_uncertainty(response)
print(f"Uncertainty Score: {uncertainty_score:.2f}")
```

**Output**:

```
Uncertainty Score: 0.15
```

This means the AI is 85% confident in its response (since `1 - 0.15 = 0.85`).

---

## Summary

Uncertainty indicators help us measure how confident the AI is about its responses. By using token probabilities, mean token probability, and uncertainty scores, we can ensure that the AI provides reliable and trustworthy answers, especially in critical domains like healthcare.
