from __future__ import print_function

import argparse
import os
import pandas as pd

from sklearn.externals import joblib

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
## TODO: Import any additional libraries you need to define a model


# Provided model load function
def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    print("Loading model.")
    
    # load using joblib
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Done loading model.")
    
    return model


## TODO: Complete the main code
if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--train-file', type=str, default='train_c1_c4_lcs.csv')
    parser.add_argument('--estimator', type=str, default='naive-bayes', 
                       help='estimator rype. options: naive-bayes(default), svm-linear, svm-rbf')
    
    ## TODO: Add any additional arguments that you will need to pass into your model
    
    # args holds all passed-in arguments
    args = parser.parse_args()

    # Read in csv training file
    training_dir = args.data_dir
    train_file = args.train_file
    train_data = pd.read_csv(os.path.join(training_dir, train_file), header=None, names=None)

    # Labels are in the first column
    train_y = train_data.iloc[:,0]
    train_x = train_data.iloc[:,1:]
    
    
    ## --- Your code here --- ##
    
    # Defining and training the model: 
    if args.estimator == 'naive-bayes':
        model = GaussianNB()
        model.fit(train_x, train_y)
        
    elif args.estimator == 'svm-linear':
        model = LinearSVC(tol=1e-5)
        model.fit(train_x, train_y)
        
    elif args.estimator == 'svm-rbf':
        model = SVC(kernel='rbf')
        model.fit(train_x, train_y)
    
    else:
        raise 'estimator type is not correct. expected \'naive-bayes\' or \'svm-linear\' or \'svm-rbf\' received: {}'.format(args.estimator)
    
    
    
    ## --- End of your code  --- ##
    

    # Save the trained model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))