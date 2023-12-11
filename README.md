# Titanic dataset CPU-GPU-TPU evaluation
This GitHub repository assesses machine learning models on the Titanic dataset using CPU, GPU, and TPU accelerators. 
The primary objective is to provide a comprehensive comparison of the models' performance across different hardware configurations.

## Results
### CPU
```bash
Training and evaluating on CPU:
Training Accuracy: 0.7609841823577881
Training Loss: 0.5211043357849121
5/5 [==============================] - 0s 3ms/step - loss: 0.4578 - accuracy: 0.7902
Validation Accuracy: 0.7902097702026367
Validation Loss: 0.4577822983264923
------------------------- 

CPU Training Time: 21.56378149986267
5/5 [==============================] - 0s 3ms/step
CPU Inference Time: 0.15933847427368164
CPU Memory Usage: 1132.87109375 MB
14/14 [==============================] - 0s 3ms/step
Predictions: 
 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1]
```
Training and Evaluation:
- Training Accuracy: 0.7609841823577881
- Training Loss: 0.5211043357849121
- Validation Accuracy: 0.7902097702026367
- Validation Loss: 0.4577822983264923

Timing and Memory Usage:
- Training Time: 21.56378149986267 seconds
- Inference Time: 0.15933847427368164 seconds
- Memory Usage: 1132.87109375 MB

### GPU
```bash
Training and evaluating on GPU:
Training Accuracy: 0.7451669573783875
Training Loss: 0.5282307863235474
5/5 [==============================] - 0s 4ms/step - loss: 0.4281 - accuracy: 0.8182
Validation Accuracy: 0.8181818127632141
Validation Loss: 0.42806223034858704
------------------------- 

GPU Training Time: 38.89291977882385
5/5 [==============================] - 0s 2ms/step
GPU Inference Time: 0.12972807884216309
GPU Memory Usage: 2282.2734375 MB
14/14 [==============================] - 0s 2ms/step
Predictions: 
 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1
 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 0 1 1 1 1 1 1 1 1 1 1 1
 0 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1]
```
Training and Evaluation:
- Training Accuracy: 0.7451669573783875
- Training Loss: 0.5282307863235474
- Validation Accuracy: 0.8181818127632141
- Validation Loss: 0.42806223034858704

Timing and Memory Usage:
- Training Time: 38.89291977882385 seconds
- Inference Time: 0.12972807884216309 seconds
- Memory Usage: 2282.2734375 MB

### TPU
```bash
Training and evaluating on TPU:
Training Accuracy: 0.7715290188789368
Training Loss: 0.49968013167381287
5/5 [==============================] - 2s 77ms/step - loss: 0.4193 - accuracy: 0.8531
Validation Accuracy: 0.8531468510627747
Validation Loss: 0.41925570368766785
------------------------- 

TPU Training Time: 123.10009574890137
5/5 [==============================] - 1s 85ms/step
TPU Inference Time: 1.3607728481292725
TPU Memory Usage: 1056.53515625 MB
14/14 [==============================] - 1s 12ms/step
Predictions: 
 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1]
```
Training and Evaluation:
- Training Accuracy: 0.7715290188789368
- Training Loss: 0.49968013167381287
- Validation Accuracy: 0.8531468510627747
- Validation Loss: 0.41925570368766785

Timing and Memory Usage:
- Training Time: 123.10009574890137 seconds
- Inference Time: 1.3607728481292725 seconds
- Memory Usage: 1056.53515625 MB

## Analysis
### Training Accuracy and Loss:
TPU achieves the highest training accuracy (0.7715), outperforming both CPU (0.7610) and GPU (0.7452).
CPU exhibits the highest validation accuracy (0.7902) among the three.

### Inference Time:
GPU demonstrates the lowest inference time (0.1297 seconds), indicating faster predictions compared to CPU (0.1593 seconds) and TPU (1.3608 seconds).

### Memory Usage:
CPU exhibits the lowest memory usage (1132.87 MB), while GPU and TPU utilize more memory (2282.27 MB and 1056.54 MB, respectively).

## Conclusion
This detailed evaluation provides a nuanced understanding of model performance on CPU, GPU, and TPU. 
Users can leverage these insights to make informed decisions based on specific requirements, balancing factors such as accuracy, speed, and resource consumption across different hardware configurations.
 
