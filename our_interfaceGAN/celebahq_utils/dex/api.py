import sys
import os

sys.path.append("./")
sys.path.append("../")

import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
sys.path.append(os.path.join(parentdir, "dex"))

from networks import domain_classifier



dataset_name = 'celebahq'

def _eval(classifier_name):
    classifier = domain_classifier.define_classifier(dataset_name, classifier_name)
    return classifier

def estimate_score(classifier, imgs, no_soft=False):
    res = classifier(imgs,no_soft=no_soft)
    return res