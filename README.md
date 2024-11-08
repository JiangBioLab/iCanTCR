# iCanTCR
A deep learning framework for early cancer detection using T cell receptor repertoire in peripheral blood.
In brief, the framework contains two deep learning classifiers, including a binary classification module 
and a multi-category classification module, and their corresponding cancer scoring strategies. 
<p float="left">
  <img src="./Fig/icantcr_model.png"/>
</p>

### Installation

From Source:

```
 git clone https://github.com/JiangBioLab/iCanTCR.git
 cd iCanTCR
```
Running the iCanTCR requires python3.6, numpy version 1.19.2, torch version 1.6.0, torchvision version 0.7.0, 
pandas version 1.1.2 and scikit-learn version 0.24.2 to be installed. 

If they are not installed on your environment, please first install numpy, pandas, and scikit-learn
by running the following command:

```
 pip install -r requirements.txt
```
Next, Please select the most suitable command from the following options to install torch and torchvision 
based on your terminal’s specific situation:

For OSX:
```
 pip install torch==1.6.0 torchvision==0.7.0
```

For Linux and Windows:
```
# CUDA 10.2
pip install torch==1.6.0 torchvision==0.7.0

# CUDA 10.1
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 9.2
pip install torch==1.6.0+cu92 torchvision==0.7.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html

# CPU only
pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```


### Quick Start
 Using the examples to perform iCanTCR. 
 The examples folder contains 3 files. In each file, the first column is the amino acid sequence 
 of CDR3, the second column is the cloning fraction,Each sample contains the sequences with the highest 
 cloning abundance. Now, you can choose one of the following commands based on whether you want to use GPU
 or not, and input it in the terminal.

   
```
 python -u iCanTCR_run.py --I examples --O output --D cpu     # cpu only
 python -u iCanTCR_run.py --I examples --O output --D gpu     # if gpu is available
```

 If you want to perform binary classification tasks only, please run the following command.
   
```
 python -u iCanTCR_run.py --I examples --O output --D cpu --T binary  # cpu only
 python -u iCanTCR_run.py --I examples --O output --D gpu --T binary  # if gpu is available
```

Similarly, for multi-classification tasks only.

```
 python -u iCanTCR_run.py --I examples --O output --D cpu --T multi  # cpu only
 python -u iCanTCR_run.py --I examples --O output --D gpu --T multi  # if gpu is available
```

After running the command, an output file under the output folder will be created, which contains 
name of the input folder and the corresponding prediction result.

### Web
 The iCanTCR program is also provided at the online webserver(http://jianglab.org.cn/iCanTCR).

### Model Training
 You can train the model using the provided training data with the following command.
   
```
 python bina_training.py # if gpu is available you can add "--D gpu" in the end of this command
 python multi_training.py # if gpu is available you can add "--D gpu" in the end of this command
```

The dataset is first randomly divided into training data and testing data. Then, the training data 
is further divided into a training set and a validation set for model optimization.

### Running new data
  Run iCanTCR on benchmark datasets as an example.
   
```
 cd data
 unzip sample_data.rar
 cd ..
 python -u iCanTCR_run.py --I sample_data/bench_bina --O output --D cpu --T binary  # if gpu is available, set the --D parameter to “gpu”.
 python -u iCanTCR_run.py --I sample_data/bench_multi --O output --D cpu --T multi  # if gpu is available, set the --D parameter to “gpu”. 
```


### Contact
 Feel free to submit an issue or contact us at cyd_charrick@163.com for problems about the tool.

