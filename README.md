#Fine Tuning using HuggingFace Bert

##Download Data
```buildoutcfg
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xf aclImdb_v1.tar.gz
```

##Install Dependencies in Conda Environment
```buildoutcfg
conda env create --file envname.yml
```

##Model Architecture
HuggingFace Bert transformer with linear layer on top of final [cls] token


