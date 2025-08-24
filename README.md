# Image Captioning with CNN-RNN on COCO Dataset

This project implements an image captioning system based on a Convolutional Neural Network (CNN) encoder and Recurrent Neural Network (RNN) decoder using the Microsoft COCO dataset. The model extracts image features via a pretrained ResNet-50 and generates descriptive captions using an LSTM.

---

## Project Structure
```
your_project/
├── encoder.py              # CNN encoder implementation (ResNet-50)
├── decoder.py              # LSTM decoder implementation
├── vocab.py                # Vocabulary class for word-token mapping
├── dataset.py              # COCO dataset class and data loader
├── train_and_test.ipynb    # Notebook for training and evaluation
├── infer.ipynb             # Notebook for inference with image upload widget
├── models/                 # Directory to save trained model weights
│   ├── encoder-1.pkl
│   ├── decoder-1.pkl
│   └── final_model.pth
├── vocab.pkl               # Saved vocabulary file
├── coco2017/               # COCO 2017 dataset folder (images & annotations)
│   ├── train2017/
│   ├── val2017/
│   └── annotations/
└── training_log.txt        # Training loss and perplexity log
```



---

## Features

- Modular Python implementation separating encoder, decoder, dataset, and vocabulary.
- Training pipeline with batch sampling by caption length.
- BLEU score evaluation on the validation set.
- Interactive Jupyter notebooks for training/evaluation and inference.
- Inference notebook includes a file upload widget for easy caption generation on user images.
- Saving and loading of model checkpoints and vocabulary.
- Supports CPU and GPU execution with PyTorch.

---

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- nltk
- pycocotools
- tqdm
- matplotlib
- ipywidgets (for inference notebook)

You can install dependencies via:
```
pip install torch torchvision nltk pycocotools tqdm matplotlib ipywidgets
```
Also download the COCO 2017 dataset and organize it as shown in the structure.

---

## Usage

### 1. Training and Testing

- Open and run `train_and_test.ipynb`.
- The notebook will load the COCO dataset, initialize models, and train for the configured epochs.
- Model weights and logs will be saved under `models/` and `training_log.txt`.
- BLEU score evaluation is performed on the validation set after training.

### 2. Inference

- Open and run `infer.ipynb`.
- Upload images using the upload widget.
- The notebook uses the saved model weights to generate captions for uploaded images.
- Captions are displayed alongside the input images interactively.

---

## Model Details

- **Encoder:** Pretrained ResNet-50, frozen weights except for a learnable embedding layer.
- **Decoder:** LSTM with embedding layer, hidden state size configurable (default 512).
- **Vocabulary:** Built from COCO captions with thresholding to filter rare words.
- **Training:** Cross entropy loss, Adam optimizer, batch size and learning rate configurable.

---

## Notes

- Ensure COCO dataset folders and annotation JSON files are correctly placed as per the project structure.
- Vocabulary is saved as a pickle file after creation—loading from this file speeds up training runs.
- The inference notebook supports GPU acceleration if available.
- The project was tested on COCO 2017 with a vocabulary threshold of 5 and embedding dimension of 256.

---

## References

- Microsoft COCO Dataset: http://cocodataset.org/
- PyTorch Tutorials and Documentation
- NLTK for NLP processing and BLEU scoring

---

## License

This project is open-source and available under the MIT License.

---

Feel free to raise issues or contribute improvements!



