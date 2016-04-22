__author__ = 'Varun'

import os
import sys
import time

import numpy as np
from common import config
from learner import naive_bayes
from learner import svm
from feature_parser import feature_extractor
from vis import classifier_vis

def drive():
    """Driver to generate the deduced ML"""
    data_config = config.Section('data')
    root = data_config.get('data-small')
    s_dir = os.path.join(root, 'spam')
    h_dir = os.path.join(root, 'ham')

    assert os.path.exists(s_dir)
    assert os.path.exists(h_dir)

    try:

        #Extract features
        h_vec, s_vec = feature_extractor.feature_parser(h_dir, s_dir, 100)

        # print "Ham====================="
        # print '\n'.join(map(str, h_vec))
        # print "Spam===================="
        # print '\n'.join(map(str, s_vec))        
        # #Initialize Classifiers for Original and Deduced ML
        #original_ml_driver = naive_bayes.NaiveBayes()
        #deduced_ml_driver = naive_bayes.NaiveBayes()
        original_ml_driver = svm.SVM()
        deduced_ml_driver = svm.SVM()

        #Learn training data for Original ML - Train
        original_ml_driver.train(h_vec[:100] + s_vec[:100],
            [0 for i in range(100)] + [1 for i in range(100)])
        #original_ml_driver.train(h_vec[:500] + s_vec[:500],
        #    [0 for i in range(500)] + [1 for i in range(500)])
        #original_ml_driver.train(h_vec[:8300] + s_vec[:8300],
        #    [0 for i in range(8300)] + [1 for i in range(8300)])
        
        #Learn training data for Deduced ML - Train Hat
        train_hat = h_vec[100:200] + s_vec[100:200]
        train_labs = [0 for i in range(100)] + [1 for i in range(100)]
        deduced_ml_driver.train(train_hat, train_labs)
        #deduced_ml_driver.train(h_vec[8300:12500] + s_vec[8300:12500],
        #    [0 for i in range(4200)] + [1 for i in range(4200)])
        
        ham_score_orig = []
        ham_score_ded = []
        spam_score_orig = []
        spam_score_ded = []

        #Active Probing starts here - call activeProbing iteratively
        for i in range(201, 300):
        #for i in range(12501, 16545):
            print i

            ham_prob_orig_ml = list(original_ml_driver.predict_proba([h_vec[i]]))[0]
            ham_prob_ded_ml = list(deduced_ml_driver.predict_proba([h_vec[i]]))[0]

            spam_prob_orig_ml = list(original_ml_driver.predict_proba([s_vec[i]]))[0]
            spam_prob_ded_ml = list(deduced_ml_driver.predict_proba([s_vec[i]]))[0]
            
            score_x = np.array([ham_prob_orig_ml[0], spam_prob_orig_ml[0]])
            score_y = np.array([ham_prob_ded_ml[0], spam_prob_ded_ml[0]])
            
            ham_score_orig = []
            ham_score_ded = []
            spam_score_orig = []
            spam_score_ded = []

            for j in range(201, 300):
                ham_prob_orig_ml = list(original_ml_driver.predict_proba([h_vec[j]]))[0]
                ham_prob_ded_ml = list(deduced_ml_driver.predict_proba([h_vec[j]]))[0]

                spam_prob_orig_ml = list(original_ml_driver.predict_proba([s_vec[j]]))[0]
                spam_prob_ded_ml = list(deduced_ml_driver.predict_proba([s_vec[j]]))[0]

                ham_score_orig.append(ham_prob_orig_ml[0])
                ham_score_ded.append(ham_prob_ded_ml[0])

                spam_score_orig.append(spam_prob_orig_ml[0])
                spam_score_ded.append(spam_prob_ded_ml[0])

            classifier_vis.classifier_vis(ham_score_orig+spam_score_orig, 
            ham_score_ded+spam_score_ded, out_file='gif_folder/correlation%s.png'%(str(i).zfill(3)))

            argmax_val = np.argmax(np.abs(score_x - score_y))
            target_class = max(1, int(round(score_y[argmax_val])))
            
            #0.25 is the threshold, which is going to be parametrized later
            if np.max(np.abs(score_x - score_y)) > 0.25:
                if argmax_val == 0:
                    train_hat += [h_vec[i]]
                else:
                    train_hat += [s_vec[i]]
                train_labs += [target_class]
                #print train_hat, train_labs
                deduced_ml_driver.train(train_hat, train_labs)

            # ham_score_orig.append(ham_prob_orig_ml[0])
            # ham_score_ded.append(ham_prob_ded_ml[0])

            # spam_score_orig.append(spam_prob_orig_ml[0])
            # spam_score_ded.append(spam_prob_ded_ml[0])
            
            # print "**********************************************" 
            # print ham_prob_orig_ml
            # print ham_prob_ded_ml
            # print spam_prob_orig_ml
            # print spam_prob_ded_ml
            # print "**********************************************"

        # classifier_vis.classifier_vis(ham_score_orig+spam_score_orig, 
        #     ham_score_ded+spam_score_ded, out_file='correlation1.png')

        ham_score_orig = []
        ham_score_ded = []
        spam_score_orig = []
        spam_score_ded = []

        #reloop to get the correlation scores
        for i in range(201, 300):
            ham_prob_orig_ml = list(original_ml_driver.predict_proba([h_vec[i]]))[0]
            ham_prob_ded_ml = list(deduced_ml_driver.predict_proba([h_vec[i]]))[0]

            spam_prob_orig_ml = list(original_ml_driver.predict_proba([s_vec[i]]))[0]
            spam_prob_ded_ml = list(deduced_ml_driver.predict_proba([s_vec[i]]))[0]

            ham_score_orig.append(ham_prob_orig_ml[0])
            ham_score_ded.append(ham_prob_ded_ml[0])

            spam_score_orig.append(spam_prob_orig_ml[0])
            spam_score_ded.append(spam_prob_ded_ml[0])


        classifier_vis.classifier_vis(ham_score_orig+spam_score_orig, 
            ham_score_ded+spam_score_ded, out_file='gif_folder/correlation300.png')

    except KeyboardInterrupt:
        sys.exit()

def main():
    #options to be added later
    drive()

if __name__ == "__main__":
    main()