# B2 Level English Conversation Model

This project aims to create a language model capable of B2 level English conversation. The model, trained on the DailyDialog dataset, has natural and fluent English conversation abilities.

## Author

**Muhammed KÖSE**  
Software Engineer
Artifical Intelligence Developer  
Email: kosemuhammet545@gmail.com  
LinkedIn: [muhammedksee](https://www.linkedin.com/in/muhammedksee)

## About the Project

CEFR (Common European Framework of Reference for Languages) B2 level represents an upper-intermediate level of English proficiency. At this level, a speaker can:

- Understand the main ideas of complex texts on both concrete and abstract topics
- Interact with a degree of fluency and spontaneity that makes regular interaction with native speakers possible without strain for either party
- Produce clear, detailed text on a wide range of subjects and explain a viewpoint on a topical issue

This project uses the DailyDialog dataset to create a language model with these capabilities.

## Dataset

DailyDialog is a high-quality multi-turn open-domain English dialogue dataset. It contains 13,118 dialogues:
- 11,118 dialogues in the training set
- 1,000 dialogues in the validation set
- 1,000 dialogues in the test set

On average, there are 8 turns per dialogue and 15 tokens per turn.

## Project Structure

```
.
├── README.md                       # Project documentation
├── requirements.txt                # Required Python packages
├── tokenizer.py                    # Tokenizer creation and dataset preparation
├── train_model.py                  # Full model training
├── simple_train.py                 # Simple training test
├── test_tokenizer.py               # Tokenizer test script
├── test_model.py                   # Model test script
├── generate.py                     # Text generation script
├── check_cuda.py                   # CUDA availability check script
├── eng.py                          # Model definitions and helper functions
├── nlp-chat-interface.html         # Web-based chat interface
├── training_log.txt                # Training log file
├── test_output.txt                 # Test outputs
├── test_output_new.txt             # New test outputs
├── test_output_final.txt           # Final test outputs
├── best_english_language_model.pt  # Best model weights
├── english_language_model.pt       # Latest model weights
├── saved_tokenizer/                # Saved tokenizer files
├── tokenizer/                      # Alternative tokenizer files
└── tokenized_dailydialog/          # Tokenized dataset
```

## Installation

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Prepare the dataset and tokenizer:

```bash
python tokenizer.py
```

3. Train the model:

```bash
python train_model.py --epochs 3 --batch_size 8 --output_dir ./english_b2_model
```

## Usage

### CUDA Check

To check GPU availability:

```bash
python check_cuda.py
```

### Tokenizer Test

To test the tokenizer:

```bash
python test_tokenizer.py
```

### Simple Training Test

For a quick training test:

```bash
python simple_train.py
```

### Full Training

For a complete model training:

```bash
python train_model.py --epochs 10 --batch_size 32 --output_dir ./english_b2_model
```

Training parameters:

- `--epochs`: Number of training epochs
- `--batch_size`: Batch size
- `--learning_rate`: Learning rate (default: 5e-5)
- `--warmup_steps`: Number of warmup steps (default: 500)
- `--model_name`: Base model name (default: "gpt2")
- `--output_dir`: Model save directory
- `--tokenizer_dir`: Tokenizer directory (default: "./saved_tokenizer")
- `--dataset_dir`: Tokenized dataset directory (default: "./tokenized_dailydialog")

### Model Testing

To test the trained model:

```bash
python test_model.py
```

### Text Generation

To generate text using the model:

```bash
python generate.py
```

### Web Interface

To use the model through a web interface, open the `nlp-chat-interface.html` file in a web browser.

## Model Architecture

The project uses a GPT-2 based language model. The model has been fine-tuned on the DailyDialog dataset to achieve B2 level English conversation capabilities.

Additionally, an alternative model architecture using LSTM has been implemented. This model is defined in the `eng.py` and `test_model.py` files.

Special tokens:
- `[PAD]`: Padding token
- `[EOS]`: End of sequence token
- `[BOS]`: Beginning of sequence token
- `[SEP]`: Separator token
- `[USER]`: User role token
- `[ASSISTANT]`: Assistant role token

## Training Process

The training process is defined in the `train_model.py` file. During training, the following steps are performed:

1. The tokenizer and dataset are loaded
2. The model is created or a pre-trained model is loaded
3. Training parameters are set
4. The model is trained and saved at regular intervals
5. Training results and model weights are saved

Training outputs are logged to the `training_log.txt` file.

## Example Outputs

The model can respond to conversation starters like:

```
[USER] How was your weekend? [SEP]
[ASSISTANT] It was great! I went hiking with my friends. The weather was perfect for outdoor activities. [SEP]

[USER] Can you tell me about your favorite book? [SEP]
[ASSISTANT] My favorite book is "To Kill a Mockingbird" by Harper Lee. It's a powerful story about justice and moral growth. [SEP]
```

For more example outputs, see the `test_output.txt`, `test_output_new.txt`, and `test_output_final.txt` files.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributing

1. Fork this repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Create a Pull Request 