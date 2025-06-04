# 📝 Ingredients

- [Installation](#Installation)
- [Dataset](#dataset)

# Installation
- Install necessary libraries 

```bash
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install torchvision
```


# Dataset
 ## Caltech101

- Create a folder named `caltech-101/` under `$DATA`.

- Download `101_ObjectCategories.tar.gz` from http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz and extract the file under `$DATA/caltech-101`.

- Download `split_zhou_Caltech101.json` from this [link](https://drive.google.com/file/d/1hyarUivQE36mY6jSomru6Fjd-JzwcCzN/view) and put it under `$DATA/caltech-101`.

The directory structure should look like:

```bash
📁 caltech-101/

├── 📁 101_ObjectCategories/

├── 📄 split_zhou_Caltech101.json

```



