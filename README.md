
<!--# PyTorch Model using Resnet50 Method

### Developing an AI application

Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications.

In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using **[this dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)** of 102 flower categories, you can see a few examples below.

![](https://github.com/RonySoliman/PyTorch-Model-using-Resnet50/blob/main/Flowers.png)
The project is broken down into multiple steps:

- Load and preprocess the image dataset
- Train the image classifier on your dataset
- Use the trained classifier to predict image content
  
*We'll lead you through each part which you'll implement in Python.*

When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.

First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.
-->
# 🌸 Flower Image Classifier — PyTorch & ResNet50

A deep learning image classification application that identifies **102 species of flowers** using transfer learning with a fine-tuned **ResNet50** backbone. Built as part of the Udacity AI Programming with Python Nanodegree.

![](https://github.com/RonySoliman/PyTorch-Model-using-Resnet50/blob/main/Flowers.png)

---

## 📌 Project Overview

This project trains a convolutional neural network to classify flower images into 102 categories. It leverages a **pretrained ResNet50** model from `torchvision`, replaces its fully connected head with a custom classifier, and fine-tunes it on the Oxford 102 Flower Dataset.

The project is split into two parts:
- **Jupyter Notebook** — interactive development, training, and sanity checks
- **Command-line scripts** (`train.py` / `predict.py`) — production-ready training and inference pipelines

---

## 📁 Project Structure

```
├── Image Classifier Project.ipynb   # Interactive notebook (development)
├── train.py                         # CLI training script
├── predict.py                       # CLI inference script
├── cat_to_name.json                 # Mapping of class labels → flower names (102 classes)
├── Flowers.png                      # Sample output visualization
└── README.md
```

---

## 🌼 Dataset

**Oxford 102 Flower Dataset** — 102 flower categories with train, validation, and test splits.

Download via:
```bash
wget 'https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz'
mkdir flowers && tar -xzf flower_data.tar.gz -C flowers
```

Expected directory structure:
```
flowers/
├── train/
├── valid/
└── test/
```

---

## 🏗️ Model Architecture

| Component | Detail |
|---|---|
| Base model | ResNet50 (pretrained on ImageNet) |
| Frozen layers | All convolutional layers |
| Custom head | Linear(2048→512) → ReLU → Dropout(0.2) → Linear(512→102) → LogSoftmax |
| Output classes | 102 flower species |

---

## ⚙️ Data Transforms

| Split | Transforms Applied |
|---|---|
| **Train** | RandomRotation(30°), RandomResizedCrop(224), RandomHorizontalFlip, Normalize |
| **Validation** | Resize(255), CenterCrop(224), Normalize |
| **Test** | Resize(255), CenterCrop(224), Normalize |

Normalization uses ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install torch torchvision pillow numpy matplotlib pandas
```

### 2. Train the model

```bash
python train.py \
  --data_dir flowers \
  --arch resnet50 \
  --epochs 4 \
  --lr 0.003 \
  --hidden_units 512 \
  --dropout 0.2 \
  --gpu \
  --checkpoint_final checkpoint_final.pth
```

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | `flowers` | Path to dataset |
| `--arch` | `resnet50` | Model architecture |
| `--epochs` | `4` | Number of training epochs |
| `--lr` | `0.003` | Learning rate |
| `--hidden_units` | `512` | Hidden layer size |
| `--dropout` | `0.2` | Dropout probability |
| `--gpu` | enabled | Use GPU if available |
| `--checkpoint_final` | `./checkpoint_final.pth` | Path to save checkpoint |

### 3. Run inference

```bash
python predict.py \
  --input_img flowers/test/1/image_04938.jpg \
  --checkpoint_final checkpoint_final.pth \
  --top_k 5 \
  --cat_names cat_to_name.json \
  --gpu
```

| Argument | Default | Description |
|---|---|---|
| `--input_img` | sample test image | Path to input image |
| `--checkpoint_final` | `./checkpoint_final.pth` | Saved model checkpoint |
| `--top_k` | `5` | Number of top predictions to return |
| `--cat_names` | `cat_to_name.json` | Label-to-name mapping file |
| `--gpu` | enabled | Use GPU if available |

---

## 🔄 Training Pipeline

1. Load and transform dataset using `torchvision.datasets.ImageFolder`
2. Freeze all ResNet50 convolutional weights
3. Attach custom fully connected classifier head
4. Train with **Adam optimizer** and **NLLLoss** criterion
5. Log training loss, validation loss, and validation accuracy every 10 steps
6. Evaluate final accuracy on held-out test set
7. Save full model checkpoint (architecture + weights + class mapping)

---

## 💾 Checkpoint

The saved checkpoint contains:

```python
{
  'input_size': 2048,
  'output_size': 102,
  'arch': model,
  'fc': classifier,
  'state_dict': model.state_dict(),
  'class_to_idx': model.class_to_idx
}
```

Load it back with:

```python
checkpoint = torch.load('checkpoint_final.pth')
model = checkpoint['arch']
model.fc = checkpoint['fc']
model.load_state_dict(checkpoint['state_dict'])
model.class_to_idx = checkpoint['class_to_idx']
```

---

## 🔍 Inference Pipeline

1. **`process_image()`** — resizes, center-crops to 224×224, normalizes to ImageNet stats, returns a NumPy array
2. **`predict()`** — runs the model in eval mode, returns top-K class probabilities and labels
3. **Sanity check** — bar chart of top-5 predictions displayed alongside the input flower image

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `PyTorch` | Model training and inference |
| `torchvision` | Pretrained ResNet50, transforms, ImageFolder |
| `PIL / Pillow` | Image loading and preprocessing |
| `NumPy` | Array operations |
| `matplotlib` | Visualization of predictions |
| `argparse` | CLI argument parsing |

---

## 📝 Notes & Limitations

- Only `resnet50` is wired up in `train.py`; `resnet101`/`resnet152` were tested but excluded due to longer training time
- GPU acceleration (CUDA) is automatically used when available and falls back to CPU
- The full model object is saved in the checkpoint (not just `state_dict`), which makes it portable but ties it to the original class structure
- For production use, consider exporting with `torch.jit.script` or ONNX for framework-agnostic deployment

---

## 👤 Author

Built as a submission for the **Udacity AI Programming with Python Nanodegree** — Neural Networks & PyTorch module.
