# TrajOpt

Automatic Differentiation is pretty sweet. It makes it easy to make optimizers.

You can use an out of the box optimizer. Adam seemed to work pretty well.

Or you can write your own Newton step solve.r Pytorch does have the ability to get you hessians, but it is somewhat involved. The structure of trajectory optimization problems is reflected in the bandededness of the hessian. This makes it fast to invert and fast to get.

Enclosed is some experiments in using PyTorch to these ends.

See these blog posts for more.

