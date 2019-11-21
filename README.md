# CPI_extraction



Chemical-protein Interaction Extraction via Gaussian Probability Distribution and External Biomedical Knowledge
===========================================================================================================================

Our proposed model aims to detect and classify the chemical-protein interaction in biomedical texts, which has been the state-of-the-art model for CPI extraction. 

In this project, we provide the implementation of our proposed model. It is organized as follows.

CPI Extraction
  -Resources
    -NCBI_BERT_pubmed_uncased_L-12_H-768_A-12
      -vocab.txt
      -pytorch_model.bin
      -bert_config.json
  -Records
    -CPI_all_epoch3
      -pytorch_model.bin
      -bert_config.json
      -eval_results.txt
      -test_results.txt
  -SourceCode
    -BioRE.py
    -BioRE_BG.py
    -modeling.py
    -modeling_BG.py
    -file_utils.py
  -ProcessedData
    -CHEMPROT
      -train.tsv
      -dev.tsv
      -test.tsv
      -test_overlapping.tsv
      -test_normal.tsv
    -DDIExtraction2013
      -train.tsv
      -dev.tsv
      -test.tsv
      -test_overlapping.tsv
      -test_normal.tsv


Resources: provide the BERT model pre-trained on PubMed.
Records: provide a record of our proposed model.
SourceCode: provide the source codes.  BioRE.py performs our proposed model; and BioRE.py performs the 'BERT+Gaussian' model.
ProcessedData: provide the CHEMPROT and DDIExtraction2013 datasets, including the overlapping and normal instances.


QUICKSTART 
================================================================================================================

Our proposed model can be performed in the following environment:
python                    3.6.9
pytorch-pretrained-bert   0.6.1
torch                     1.2.0
numpy                     1.16.4
pandas                    0.25.0
scipy                     1.3.1

 

Since the model contains multiple layer, it generally need some time to train. If the users have no time or GPU to train model, the saved model in the Records can be loaded to test.

Some examples of execution instructions are listed below.

#### Run our proposed model ####

python BioRE.py \
  --task_name cpi \
  --do_train \
  --do_eval \
  --do_predict \
  --do_lower_case \
  --data_dir /$YourPath/CPI_extraction/ProcessedData/CHEMPROT \
  --bert_model /$YourPath/CPI_extraction/Resources/NCBI_BERT_pubmed_uncased_L-12_H-768_A-12 \
  --max_seq_length 128 \
  --train_batch_size 16 \
  --eval_batch_size 8 \
  --predict_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /$YourOutputPath


#### Load the record of our proposed model ####

python BioRE.py \
  --task_name cpi \
  --do_eval \
  --do_predict \
  --do_lower_case \
  --data_dir /$YourPath/CPI_extraction/ProcessedData/CHEMPROT \
  --bert_model /$YourPath/CPI_extraction/Resources/NCBI_BERT_pubmed_uncased_L-12_H-768_A-12 \
  --max_seq_length 128 \
  --train_batch_size 16 \
  --eval_batch_size 8 \
  --predict_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 1.0 \
  --output_dir /$YourOutputPath \
  --bert_saved /$YourSavedmodelPath


#### Run the 'BERT+Gaussian' model on the CHEMPROT dataset ####

python BioRE_BG.py \
  --task_name cpi \
  --do_train \
  --do_eval \
  --do_predict \
  --do_lower_case \
  --data_dir /$YourPath/CPI_extraction/ProcessedData/CHEMPROT \
  --bert_model /$YourPath/CPI_extraction/Resources/NCBI_BERT_pubmed_uncased_L-12_H-768_A-12 \
  --max_seq_length 128 \
  --train_batch_size 16 \
  --eval_batch_size 8 \
  --predict_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /$YourOutputPath

#### Run the 'BERT+Gaussian' model on the DDIExtraction dataset ####

python BioRE_BG.py \
  --task_name ddi \
  --do_train \
  --do_eval \
  --do_predict \
  --do_lower_case \
  --data_dir /$YourPath/CPI_extraction/ProcessedData/DDIExtraction2013 \
  --bert_model /$YourPath/CPI_extraction/Resources/NCBI_BERT_pubmed_uncased_L-12_H-768_A-12 \
  --max_seq_length 128 \
  --train_batch_size 16 \
  --eval_batch_size 8 \
  --predict_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 5.0 \
  --output_dir /$YourOutputPath

