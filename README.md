# Recurrent Point Review Models


This is a companion code for the paper "Recurrent Point Review Models"


## Description


Deep neural network models represent the state-of-the-art methodologies for natural language processing. 
Here we build on top of these methodologies to incorporate temporal information and model how review data changes over time.
We use the dynamic representations of recurrent point process models, which encode the history of how reviews of businesses 
and services arrive in time, to generate instantaneous language models with improved prediction capabilities. 
Simultaneously,our methodologies improve the predictive power of our point process models by incorporating summarized review content representations. 
We provide recurrent network and temporal convolution solutions for modeling the review content. 
We deploy our methodologies in the context of recommender systems,effectively characterizing the change in preference and taste of users as time evolves..


## Note

For the purpose of the reproducibility of the results for this paper we have created self containing package with detailed
description of how to run the models.

### Data uploading and pre-processing

For storing the data we use MongoDB, therefore it is pre-request for running the code to have installed MongoDB. Detailed 
explanation of how to [install](https://docs.mongodb.com/manual/installation/) the MongoDB can be found on the official web page.

Once you have installed mongoDB please do the following steps:

1. Download the Yelp 2019 [dataset](https://www.yelp.com/dataset/download) and copy the data into
```scripts/preprocessing/yelp/``` folder. 
2. run the ```sh upload.sh``` script.


### Installation of the dpp library

In order to run the code please install dpp library.

#### Install Tyche
1. ``cd Tyche``
2. `pip install -r requirements.txt`
3. `pip install .`

#### Install GENTEXT

1. ``cd GENTEXT``
2. `pip install -r requirements.txt`
3. `pip install .`


#### Install dpp

1. ``cd deep_point_process``
2. `pip install -r requirements.txt`
3. `pip install .`


### Running training scripts 

For running the training please use the *train_script.py* and choose one of the experiments.

1. `cd deep_point_process`
2. `python scripts/train_script.py experiments/business_text_pp_cnn_emb16.yaml`


### Running inference scripts 

For running the inference please use the *inference_script.py* and choose one of the experiments.

1. `cd deep_point_process`
2. `python scripts/inference_script.py models/yelp19_shopping_business_text_pp_cnn_model_tpp_model_embedding_size_64_model_tpp_model_cell_type_hidden_size_16`



