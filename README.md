# Application of deep learning algorithms in flow field pattern prediction

## Instructions
Data Set: HS/, LS/ (These data are not publicly accessible due to lab policy)

1. If there are no data in any one of catagories data/velocity, 
data/magnitude, data/direction, use 
```
python3 data_process.py 
```
to first  process raw data stored in HS/ and LS/.

2. Use 
```
python3 training_data_process.py --input_size=9
```
to transfer datain data/velocity, data/magnitude, data/direction to corresponding training 
data sequenced by time.

3. If there is no corresponding checkpoints for the model you want to use,
training process is needed.
	(i). Use 
  ```
  python3 train_rnn.py --ff_type='velocity' --input_size=9 --bi='false' 
  ```
  to train a normal rnn 
	(ii). Use 
  ```
  python3 train_rnn.py --ff_type='velocity' --input_size=9 --bi='true'
  ```
  to train a biRnn

Note: options for ff_type are {'velocity', 'magnitude', 'direction'}, options
for input_size are odd numbers including 5,7,9,11. Required training data is
in data/{ff_type} and ended with 'input_size'. For example, if input_size=9,
required training data is like 'x_y_9' such as '8_9_9'. If there are no data
of this type, please go back to step 1&2 to generate training data.

4. "test_rnn.py" is used to evaluate the performance of the trained model on a
set of unseen test data.   
	(i). Use 
  ```
  python3 test_rnn.py --ff_type='velocity' --input_size=9 
  ```
  to show results of a normal rnn and a brnn

5. "evaluate_rnn.py" is used to evaluate the performance of the trained model on a
specific flow field sequence. The default test data is randomly chosed from
the 100 cycles. You can also specify the cycle you want to predict by changing
parameters "num_cycle" and "start_ff" in the .py file.   
	(i). Use 
  ```
  python3 evaluate_rnn.py --ff_type='velocity' --input_size=9 --bi='false'
  ```
  to train a normal rnn 
	(ii). Use 
  ```
  python3 evaluate_rnn.py --ff_type='velocity' --input_size=9 --bi='true'
  ```
  to train a biRnn

6. Other .py files:

---------------------
|visualize_entire.py| 
---------------------
Visualize the entire flow field for one of velocity map, direction map, or
magnitude map. Usage: python3 visualize_entire.py --ff_type='velocity' --filename=9000

-----------------
|train_common.py| 
-----------------
Define placeholder, optimizer, accuracy, and loss function

----------
|utils.py| 
----------
Define some useful function used to save/load checkpoint, plot, get
parameters. etc

--------------------
|data/RNNDataSet.py| 
--------------------
Training, validation and test data for rnn model

--------------------
|model/build_rnn.py| 
--------------------
Rnn model

----------------------
|model/build_birnn.py| 
----------------------
BiRnn model

-------------
|config.json| 
-------------
Define some important parameters for the model and training process



