
# Zarvan: An Efficient Gated Architecture for Sequence Modeling

 Zarvan is a novel sequence modeling architecture implemented in PyTorch, designed as a high-performance, linearly scalable alternative to the standard Transformer. It addresses the quadratic complexity bottleneck of self-attention by replacing it with a unique dual-context gating mechanism, enabling efficient processing of extremely long sequences without sacrificing performance.

The experimental results demonstrate that Zarvan achieves accuracy competitive with the Transformer across a diverse range of benchmarks while being significantly more efficient.

Figure: Zarvan's linear time complexity (blue) vs. the Transformer's quadratic complexity (red) on the IMDb dataset. Zarvan achieves comparable accuracy with vastly superior scalability.

## Key Features
Linear Complexity (O(S)): Zarvan's computation and memory usage scale linearly with sequence length S, making it ideal for long-document analysis, high-resolution vision tasks, and time-series forecasting.

Dual-Context Gating: Instead of self-attention, Zarvan uses two global context vectors—a Holistic Context for the overall "gist" and an Associative Context for focused memory—to intelligently modulate information flow for each token.

High Performance: Achieves accuracy competitive with, and in some cases superior to, the standard Transformer on benchmarks like IMDb, MS MARCO, MNIST, and long-range synthetic tasks.

Versatile & Flexible: Proven to be effective across different domains, including NLP, information retrieval, and vision.

PyTorch-based: A clean, modular, and easy-to-understand implementation in modern PyTorch.

## Architecture Deep Dive
The core of the model is the Zarvan Block. This block processes an input sequence by first computing two parallel global context vectors and then using them to inform a gated update for each token.

Figure: The data flow within a single Zarvan Block.

Holistic Context Extractor: Captures a comprehensive summary of the entire sequence into a single vector.

Associative Context Extractor: Learns to assign importance scores to each token, creating a weighted average that focuses on the most salient information. This is crucial for long-range memory tasks.

Gated Update Mechanism: The two context vectors are used to parameterize an input gate and a forget gate for each token, allowing the model to dynamically filter and update its representation based on a global understanding of the sequence.

## Results & Benchmarks
Zarvan has been rigorously tested against a standard Transformer baseline on five distinct tasks. A summary of the results is below. For a full analysis, please see our paper.

Benchmark	Task Domain	Zarvan Performance	Key Takeaway
IMDb	NLP Classification	Comparable Accuracy, ~41% Faster (at S=512)	Demonstrates superior scalability.
MNIST	Vision as Sequence	Statistically Comparable Accuracy	Proves versatility across modalities.
MS MARCO	Information Retrieval	Statistically Comparable Accuracy, Faster	Effective for semantic embedding tasks.
Selective Copy	Synthetic Memory	100% Accuracy	Perfect long-range, precise memory recall.
Adding Problem	Synthetic Reasoning	~99% Accuracy	Capable of basic computation on stored memory.


Installation
Clone the repository:

```Bash

git clone https://github.com/systbs/zarvan.git
cd zarvan
```
Install the required dependencies. It is recommended to use a virtual environment. The experiments rely on libraries listed in requirements.txt.

```Bash

pip install -r requirements.txt
```
This will install PyTorch, datasets, transformers, and other necessary packages.

Quick Start
You can easily import and use the ZarvanModel in your projects. The example below shows how to instantiate the model for a standard sequence classification task.

```Python

import torch
from zarvan import ZarvanModel # Assuming the code is in a package or zarvan.py

# --- Model Configuration ---
VOCAB_SIZE = 10000
NUM_CLASSES = 10
SEQ_LEN = 512
EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 4
BATCH_SIZE = 16

# --- Instantiation ---
print("Instantiating the ZarvanModel...")
model = ZarvanModel(
    vocab_size=VOCAB_SIZE,
    num_classes=NUM_CLASSES,
    seq_len=SEQ_LEN,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS
)

# --- Dummy Data and Forward Pass ---
print("Performing a dummy forward pass...")
dummy_input = torch.randint(low=1, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN))
output_logits = model(dummy_input, pool='mean') # Use 'mean' pooling for classification

print(f"Input shape:  {dummy_input.shape}")
print(f"Output shape: {output_logits.shape}") # Expected: (BATCH_SIZE, NUM_CLASSES)
print("\n✅ Zarvan model is ready for use!")
```
## Citation
If you use **Zarvan** in your research, please cite:
```bibtex
@article{sajjadi2025zarvan,
  title={Zarvan: An Efficient Gated Architecture for Sequence Modeling with Linear Complexity},
  author={Sajjadi, Yasser},
  journal={Preprints.org},
  doi={10.20944/preprints202507.2512.v1},
  year={2025}
}
```
You can also access the preprint here: https://www.preprints.org/manuscript/202507.2512/v1

## Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## Acknowledgements
This work was developed by Yasser Sajjadi in collaboration with the AI assistants Gemini from Google and Grok from xAI. Their assistance was instrumental in architectural design, debugging, and analysis.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
