## <font style="color:rgb(31, 35, 40);">Installation</font>

<font style="color:rgb(31, 35, 40);">To install all dependencies, use the following commands:</font>

```plain
conda create -n linchain python=3.10
source activate linchain
pip install -r requirements.txt
```

## <font style="color:rgb(31, 35, 40);">Dataset</font>

<font style="color:rgb(31, 35, 40);">Get the train data and benchmarks from </font>[here](https://github.com/AGI-Edgerunners/LLM-Adapters)

## Train and Evaluation

To run the train and evaluation script, use the following commands (modify the script to your own path):

```plain
# commonsense reasoning
cd commonsense_reasoning
bash train_linchain.sh

# arithmetic reasoning
cd arithmetic_reasoning
bash train_linchain.sh

# glue
cd glue
bash deberta-v3-base_xxx.sh
```

## Ackonwledge

<font style="color:rgb(31, 35, 40);">This code is modifed based on </font>the [MoSLoRA](https://github.com/wutaiqiang/MoSLoRA)<font style="color:rgb(31, 35, 40);">, we thank for their efforts.</font>
