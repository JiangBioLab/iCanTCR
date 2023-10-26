# iCanTCR
A deep learning framework for early cancer detection using T cell receptor repertoire in peripheral blood.
<p float="left">
  <img src="Fig/Figure1fff.png"/>
</p>

### Installation

From Source:

```
 git clone https://github.com/JiangBioLab/iCanTCR.git
 cd iCanTCR
 pip install -r requirements.txt
```

### Quick Start
 Using the examples to perform iCanTCR. The first column is the amino acid sequence of CDR3, the second column is the cloning fraction,Each sample contains the sequences with the highest cloning abundance.
   
```
 python --I examples --O output --D cpu     # cpu only
 python --I examples --O output --D gpu     # if gpu is available
```

 If you want to perform binary classification tasks only, please run the following command.
   
```
 python --I examples --O output --D cpu --T binary  # cpu only
 python --I examples --O output --D gpu --T binary  # if gpu is available
```

Similarity, for multi-classification tasks only.

```
 python --I examples --O output --D cpu --T multi  # cpu only
 python --I examples --O output --D gpu --T multi  # if gpu is available
```

### Web
 The iCanTCR program is also provided at the online webserver(http://jianglab.org.cn/iCanTCR).

### Model Training
 You can train the model using the provided training data with the following command.
   
```
 python bina_training.py
 python multi_training.py
```

### Contact
 Feel free to submit an issue or contact us at cyd_charrick@163.com for problems about the package.

