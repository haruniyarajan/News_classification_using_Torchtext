# News Article Classification with PyTorch

A deep learning project that automatically classifies news articles into four categories: **World**, **Sports**, **Business**, and **Sci/Tech** using PyTorch and the AG_NEWS dataset.

## 📋 Project Overview

This project implements a text classification model using PyTorch's `torchtext` library and neural networks with embedding layers. The model processes news articles, converts them into numerical representations, and predicts their appropriate category with over 80% accuracy.

## 🎯 Features

- **Automated Text Classification**: Classify news articles into 4 categories
- **EmbeddingBag Neural Network**: Efficient text representation using embeddings
- **Data Pipeline**: Complete preprocessing pipeline with tokenization and vocabulary building
- **Visualization**: 3D t-SNE visualization of learned embeddings
- **Model Persistence**: Save and load trained models
- **Batch Prediction**: Classify multiple articles at once

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Google Colab (recommended) or local Jupyter environment

### Installation

1. **Clone or upload the notebook to Google Colab**

2. **Install dependencies** (first cell in notebook):
```python
%pip install pandas numpy==1.26.4 seaborn==0.9.0 matplotlib scikit-learn portalocker>=2.0.0 plotly
%pip install torch==2.3.0+cpu torchdata==0.9.0+cpu torchtext==0.18.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu
```

3. **Run the notebook cells sequentially**

## 📊 Dataset

The project uses the **AG_NEWS** dataset, which contains:
- **120,000 training samples**
- **7,600 test samples**
- **4 classes**: World, Sports, Business, Sci/Tech
- News articles from over 2000 news sources

## 🏗️ Model Architecture

```
Input Text → Tokenization → Vocabulary Indexing → EmbeddingBag → Linear Layer → Output (4 classes)
```

### Key Components:
- **Tokenizer**: Basic English tokenizer
- **Vocabulary Size**: ~95,000 tokens
- **Embedding Dimension**: 64
- **Output Classes**: 4
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: SGD with learning rate 0.1

## 📈 Performance

- **Training Accuracy**: ~90%
- **Validation Accuracy**: ~85%
- **Test Accuracy**: ~80-82%
- **Training Time**: ~20 minutes for 10 epochs (CPU)

## 🔧 Usage

### Train the Model

```python
# The notebook handles training automatically
# Key parameters:
BATCH_SIZE = 64
EPOCHS = 10
LR = 0.1
```

### Make Predictions

```python
# Single article prediction
article = "The team won the championship in an exciting final match"
prediction = predict(article, text_pipeline)
print(f"Category: {prediction}")  # Output: Sports
```

### Load Saved Model

```python
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
```

## 📁 Project Structure

```
news-classification/
│
├── news_classification.ipynb    # Main Google Colab notebook
├── README.md                     # This file
├── best_model.pth               # Saved model weights (after training)
└── requirements.txt             # Python dependencies
```

## 🎓 How It Works

1. **Data Loading**: Load AG_NEWS dataset and split into train/validation/test
2. **Tokenization**: Convert text into tokens using basic_english tokenizer
3. **Vocabulary Building**: Create vocabulary from training data with ~95K unique tokens
4. **Embedding**: Map tokens to dense vectors using EmbeddingBag layer
5. **Classification**: Pass embeddings through fully connected layer to predict class
6. **Training**: Optimize using SGD with CrossEntropy loss
7. **Evaluation**: Test on unseen data and visualize results

## 📊 Example Predictions

| Article | Predicted Category |
|---------|-------------------|
| "International climate accord signed by world leaders" | World |
| "Team wins championship in thrilling overtime victory" | Sports |
| "Tech startup IPO exceeds market expectations" | Business |
| "New drug shows promise in Alzheimer's treatment" | Sci/Tech |

## 🔍 Visualization

The notebook includes a 3D t-SNE visualization that shows how the model learns to cluster similar articles together in the embedding space. Articles from the same category appear closer together.

## 🛠️ Customization

### Adjust Hyperparameters

```python
# In the notebook, modify these values:
BATCH_SIZE = 64        # Batch size for training
EPOCHS = 10            # Number of training epochs
LR = 0.1              # Learning rate
emsize = 64           # Embedding dimension
```

### Use Different Dataset

You can adapt the code to work with other text classification datasets by:
1. Modifying the data loading section
2. Adjusting `num_class` for your number of categories
3. Updating the `label_mapping` dictionary

## 📝 Technical Details

### Model Parameters
- **Vocabulary Size**: ~95,000
- **Embedding Dimension**: 64
- **Total Parameters**: ~6 million
- **Model Size**: ~24 MB

### Data Split
- **Training**: 95% (114,000 samples)
- **Validation**: 5% (6,000 samples)
- **Testing**: 7,600 samples

## 🤝 Contributing

Feel free to fork this project and submit pull requests for:
- Performance improvements
- Additional features
- Bug fixes
- Documentation enhancements

## 📚 References

- [PyTorch Text Classification Tutorial](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html)
- [TorchText Documentation](https://pytorch.org/text/stable/index.html)
- [AG_NEWS Dataset](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)

## 📄 License

This project is open source and available for educational purposes.

## 👥 Authors

- Adapted for Google Colab from IBM Skills Network tutorial
- Enhanced with additional features and documentation

## 🐛 Troubleshooting

### Common Issues:

1. **Out of Memory Error**: Reduce `BATCH_SIZE`
2. **Slow Training**: Consider using GPU in Colab (Runtime > Change runtime type > GPU)
3. **Import Errors**: Restart kernel after installing packages

### Tips:
- Always restart the kernel after installing dependencies
- Use GPU runtime for faster training (5-10 minutes vs 20 minutes)
- Save your model regularly during training

## 📞 Support

For issues or questions:
- Check existing issues in the repository
- Create a new issue with detailed description
- Include error messages and environment details

---

**Happy Classifying! 🎉**
