# Modeling Knowledge Graphs with Composite Reasoning

This is the code of paper 
**Modeling Knowledge Graphs with Composite Reasoning**. 

Wanyun Cui, Linqiu Zhang. AAAI 2024

### 1. Results
The results of **CompilE_D** and **CompilE_N** on **WN18RR**, **FB15k237**, **UMLS** and **KINSHIP** are as follows.

<p align="center">
  <img src="./table3.png">
</p>

<p align="center">
  <img src="./table4.png" style="width:700px;height:350px;">
</p>

### 2. Reproduce the Results 
To reproduce the above results, download the pkl files for the four datasets [here](https://drive.google.com/drive/folders/1V4z9FeunObC0IOvDcRNH5A5_uCWVxHsN).

Move those pkl files to the current directory, and run the following commands.

```shell script
#################################### WN18RR ####################################
# CompilE_N
python learn.py --dataset WN18RR --model ComplEx --rank 2000 \
--optimizer Adagrad --learning_rate 1e-1 --batch_size 100 --use_N3 --use_N3_weight 0.1 \
--save_path trained_models/wn/dura \
--max_epochs 100 --valid 2 --data_path data \
--regularizer composite --mode_list "hrt-t-hrt;hrt-h-hrt" \
--n_pos 10 --w3 0.6 --fully_train

# CompilE_D
python learn.py --dataset WN18RR --model ComplEx --rank 2000 \
--optimizer Adagrad --learning_rate 1e-1 --batch_size 100 --use_DURA_W --use_DURA_W_weight 0.1 \
--save_path trained_models/wn/dura \
--max_epochs 100 --valid 2 --data_path data \
--regularizer composite --mode_list "hrt-t-hrt;hrt-h-hrt" \
--n_pos 10 --w3 0.4 --fully_train --do_ce_weight

#################################### FB237 ####################################
# CompilE_N
python learn.py --dataset FB237 --model ComplEx --rank 2000 \
--optimizer Adagrad --learning_rate 1e-1 --batch_size 100 --use_N3 --use_N3_weight 0.05 \
--save_path trained_models/umls/dura \
--max_epochs 100 --valid 2 --data_path data \
--regularizer composite --mode_list "hrt-hr-rt;hrt-rt-hr" \
--n_pos 10 --w2 0.2 --fully_train


# CompilE_D
python learn.py --dataset FB237 --model ComplEx --rank 2000 \
--optimizer Adagrad --learning_rate 1e-1 --batch_size 200 --use_DURA_W --use_DURA_W_weight 0.05 \
--save_path trained_models/umls/dura \
--max_epochs 100 --valid 2 --data_path data \
--regularizer composite --mode_list "hrt-hr-rt;hrt-rt-hr" \
--n_pos 10 --w2 0.1 --fully_train


#################################### UMLS ####################################
# CompilE_N
python learn.py --dataset umls --model ComplEx --rank 2000 \
--optimizer Adagrad --learning_rate 1e-1 --batch_size 100 --use_N3 --use_N3_weight 0.005 \
--save_path trained_models/umls/N3 \
--max_epochs 100 --valid 2 --data_path data \
--regularizer composite --mode_list "hrt-hr-rt;hrt-rt-hr" \
--n_pos 10 --w2 1.0 --fully_train

# CompilE_D
python learn.py --dataset umls --model ComplEx --rank 2000 \
--optimizer Adagrad --learning_rate 5e-2 --batch_size 100 --use_DURA_W --use_DURA_W_weight 0.001 \
--save_path trained_models/umls/dura \
--max_epochs 100 --valid 2 --data_path data \
--regularizer composite --mode_list "hrt-hr-rt;hrt-rt-hr" \
--n_pos 10 --w2 0.8 --fully_train


#################################### KINSHIP ####################################
# CompilE_N
python learn.py --dataset kinship --model ComplEx --rank 2000 \
--optimizer Adagrad --learning_rate 1e-1 --batch_size 100 --use_N3 --use_N3_weight 0.05 \
--save_path trained_models/kinship/N3 \
--max_epochs 100 --valid 2 --data_path data \
--regularizer composite --mode_list "hrt-rt-hrt;hrt-hr-hrt;hrt-t-hrt;hrt-h-hrt;hrt-hr-t;hrt-rt-h" \
--n_pos 10 --w1 0.4 --w3 0.4 --fully_train

# CompilE_D
python learn.py --dataset kinship --model ComplEx --rank 2000 \
--optimizer Adagrad --learning_rate 5e-2 --batch_size 100 --use_DURA_W --use_DURA_W_weight 0.005 \
--save_path trained_models/kinship/dura \
--max_epochs 100 --valid 2 --data_path data \
--regularizer composite --mode_list "hrt-hr-rt;hrt-rt-hrt" \
--n_pos 10 --w2 1.0 --fully_train
```

## Acknowledgement
We refer to the code of [kbc](https://github.com/facebookresearch/kbc) and [DURA](https://github.com/MIRALab-USTC/KGE-DURA). Thanks for their contributions.
