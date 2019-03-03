# TrajOpt

Automatic Differentiation is pretty sweet. It makes it easy to make optimizers.

You can use an out of the box optimizer. Adam seemed to work pretty well.

Or you can write your own Newton step solve.r Pytorch does have the ability to get you hessians, but it is somewhat involved. The structure of trajectory optimization problems is reflected in the bandededness of the hessian. This makes it fast to invert and fast to get.

Enclosed is some experiments in using PyTorch to these ends.

See these blog posts for more.

http://www.philipzucker.com/deducing-row-sparse-matrices-matrix-vector-product-samples/

http://www.philipzucker.com/pytorch-trajectory-optimization-part-5-cleaner-code-50hz/

http://www.philipzucker.com/pytorch-trajectory-optimization-3-plugging-hessian/

http://www.philipzucker.com/pytorch-trajectory-optimization-part-2-work-progress/

http://www.philipzucker.com/pytorch-trajectory-optimization/

