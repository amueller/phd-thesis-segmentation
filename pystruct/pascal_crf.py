import numpy as np

from pystruct import learners
import pystruct.models as crfs

from datasets.pascal import PascalSegmentation
from pascal_helpers import load_pascal
from latent_crf_experiments.utils import (discard_void, add_edges,
                                          add_edge_features)


from IPython.core.debugger import Tracer
tracer = Tracer()


def main():
    ds = PascalSegmentation()
    # load training data
    edge_type = "pairwise"
    which = "kTrain"
    data_train = load_pascal(which=which, sp_type="cpmc")

    data_train = add_edges(data_train, edge_type)
    data_train = add_edge_features(ds, data_train)
    data_train = discard_void(ds, data_train, ds.void_label)

    X, Y = data_train.X, data_train.Y

    class_weights = 1. / np.bincount(np.hstack(Y))
    class_weights *= 21. / np.sum(class_weights)

    model = crfs.EdgeFeatureGraphCRF(
        class_weight=class_weights, symmetric_edge_features=[0, 1],
        antisymmetric_edge_features=[2], inference_method='qpbo')

    ssvm = learners.NSlackSSVM(model, C=0.01, n_jobs=-1)
    ssvm.fit(X, Y)

if __name__ == "__main__":
    main()
