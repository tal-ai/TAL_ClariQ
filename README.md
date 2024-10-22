# TAL_ClariQ

This repository contains source code(Pytorch) for CLariQ competition.

Team name: TAL ML Team

Model name: CMAN++

See more detail in paper: 
[Automatic Short Answer Grading via Multiway Attention Networks. AIED (2) 2019: 169-173](https://arxiv.org/abs/1909.10166#:~:text=Automatic%20Short%20Answer%20Grading%20via%20Multiway%20Attention%20Networks,-Tiaoqiao%20Liu%2C%20Wenbiao&text=Automatic%20short%20answer%20grading%20(ASAG,monotonous%20and%20tedious%20grading%20workloads.))


## Note
1, The approximate running time is 2 hours, since we only have the V100 GPU for testing, there may be a discrepancy in the total running time.

2, Document relevance score for dev set:
```
NDCG1: 0.22661954068764104
NDCG3: 0.20740563773647339
NDCG5: 0.20241540875297784
NDCG10: 0.18973210299160412
NDCG20: 0.17328878793208852
P1: 0.28142838178680474
P3: 0.24746670206646312
P5: 0.23346608257002524
P10: 0.2015531660692951
P20: 0.15672043010752687
MRR100: 0.36629474494004755
```

## Running steps

### docker utilization

1, get docker container

```
docker pull ky941122/tal_ml_clariq:v1
```

2, Then activate conda env

```
conda activate clariq
```

3, If docker runs into any problems, set up the following environment.

cuda=10.1

numpy==1.19.3

pandas==1.1.4

torch==1.7.0+cu101


### run python code

1,run the following command get dev/test set result
```
cd ./TAL_ClariQ/ClariQ/

python write_test_file -multi_turn_request_file_path={absolut path for dev file} -output_run_file={path for output file} -topk=100 -batch_size=100

for example:
python write_test_file -multi_turn_request_file_path=./processed_data/little_dev.pkl -output_run_file=./processed_data/dev_run_files -topk=100 -batch_size=100
```

You'll get the ranking results on the validation set under the given folder. If you meet OOM error, you can control the "-batch_size" parameter, make it smaller, default batch_size is set to 300.


2, Callable .py file for one input_dict

In order to satisfy the human-in-the-loop style testing method, we provide the callable .py file. We have shown a specific case in run.py file.

Call query_one_dict function in run.py file, and you will get one output question for the input query dictionary.

```
from run import query_one_dict
case_0 = {'topic_id': 293,
        'facet_id': 'F0729',
        'initial_request': 'Tell me about the educational advantages of social networking sites.',
        'question': 'which social networking sites would you like information on',
        'answer': 'i don have a specific one in mind just overall educational benefits to social media sites',
        'conversation_context': [
            {'question': 'what level of schooling are you interested in gaining the advantages to social networking sites',
            'answer': 'all levels'},
            {'question': 'what type of educational advantages are you seeking from social networking',
            'answer': 'i just want to know if there are any'}
        ]
    }
print(query_one_dict(case_0, 100))
```