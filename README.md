# CS_Model_Adaptation
This repository contains the code for the paper "Refining Pre-trained Models for Generalizable Commonsense Reasoning" (EMNLP 2021). See full paper [here]()

Note that our finetuning and inference code are adpated from [ProtoQA](https://github.com/iesl/ProtoQA_GPT2), 
prefix-tuning code is adapted from [PrefixTuning](https://github.com/XiangLi1999/PrefixTuning),
Autoprompt code is adapated from [autoprompt](https://github.com/ucinlp/autoprompt),
and protoqa evaluation code is adpated from [protoqa-evaluator](https://github.com/iesl/protoqa-evaluator)

## Enviroments
This code has been tested on Python 3.7.9, Pytorch 1.7.0 and Transformers 4.2.1, you can install the required packages by 
```
pip install -r requirements.txt
```
Then we need to install the protoqa-evaluator by cd to protoqa-evaluator 
```
pip install -e protoqa-evaluator
```

## Finetuning
To finetune BART model on ProtoQA task, cd to src/Finetune and run
```
bash finetune.sh 
```
This would also run inference on ProtoQA dev set, you can find the results in results.json under your specified output directory. 

For GPT2 model, you can simply update the --model_type and --model_name_or_path 

### CommonGen
For CommonGen experiments, the command can also be found in finetune.sh, you can just uncomment it and run. 
The results can be found in the eval_generation.txt in the output directory after training.

To evalutate the generated output, please follow the [CommonGen](https://github.com/INK-USC/CommonGen/tree/master/evaluation/Traditional) official repo to set up a new environment and evaluate. 

## Prefix-tuning
Under src/Prefix_tuning and run
```
bash prefixtune.sh 
```

## Autoprompt
Under src/Autoprompt and run 
```
bash autoprompt.sh 
```

## Manual Annotation
For the 30 selected questions and 30 newly annotated questions, as well as their model predictions can be found in Data/Manual_annotation

## Cite 
```
```