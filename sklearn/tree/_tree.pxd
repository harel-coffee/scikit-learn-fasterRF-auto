# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Joel Nothman <joel.nothman@gmail.com>
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#
# Licence: BSD 3 clause

# See _tree.pyx for details.

import numpy as np
cimport numpy as np

#ctypedef np.npy_float32 DTYPE_t          # Type of X
# Mike code
ctypedef np.npy_int8 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
# Mike code
#ctypedef np.npy_float32 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

# Mike code: to replace the use of double
#ctypedef np.npy_float32 DOUBLE_f
ctypedef double DOUBLE_f



# Mike code
cdef extern from "time.h": 

    # # Declare only what is used from `tm` structure. 
    # struct tm: 
    #     int tm_mday # Day of the month: 1-31 
    #     int tm_mon # Months *since* january: 0-11 
    #     int tm_year # Years since 1900 
 
    ctypedef long time_t 
#    tm* localtime(time_t *timer) 
    time_t time(time_t *tloc)  nogil


# Mike notes:
# functions whose nogil were removed: update_mse_mike, sort, node_split, and a call on line 2878 to node_split sets a 'with gil'

# =============================================================================
# Criterion
# =============================================================================

cdef class Criterion:
    # The criterion computes the impurity of a node and the reduction of
    # impurity of a split on that node. It also computes the output statistics
    # such as the mean in regression and class probabilities in classification.

    # Internal structures
    cdef DOUBLE_t* y                     # Values of y

    # Mike code
    cdef DOUBLE_t* y_sq                     # Values of y_sq

    cdef SIZE_t y_stride                 # Stride in y (since n_outputs >= 1)
    cdef DOUBLE_t* sample_weight         # Sample weights

    cdef SIZE_t* samples                 # Sample indices in X, y
    cdef SIZE_t start                    # samples[start:pos] are the samples in the left node
    cdef SIZE_t pos                      # samples[pos:end] are the samples in the right node
    cdef SIZE_t end

    cdef SIZE_t n_outputs                # Number of outputs
    cdef SIZE_t n_node_samples           # Number of samples in the node (end-start)
    cdef DOUBLE_f weighted_n_samples       # Weighted number of samples (in total)
    cdef DOUBLE_f weighted_n_node_samples  # Weighted number of samples in the node
    cdef DOUBLE_f weighted_n_left          # Weighted number of samples in the left node
    cdef DOUBLE_f weighted_n_right         # Weighted number of samples in the right node

    # Mike code
    cdef DOUBLE_f* y_node
    cdef DOUBLE_f* y_sq_node
    cdef DOUBLE_f* sample_weight_node
    cdef SIZE_t* samples_fx

    
    # The criterion object is maintained such that left and right collected
    # statistics correspond to samples[start:pos] and samples[pos:end].

    # Methods
    # cdef void init(self, DOUBLE_t* y, SIZE_t y_stride, DOUBLE_t* sample_weight,
    #                DOUBLE_f weighted_n_samples, SIZE_t* samples, SIZE_t start,
    #                SIZE_t end) nogil

    # Mike code
    cdef void init(self, DOUBLE_t* y, SIZE_t y_stride, DOUBLE_t* sample_weight,
                   DOUBLE_f weighted_n_samples, SIZE_t* samples, SIZE_t start,
                   SIZE_t end, DOUBLE_t* y_sq) nogil

    # Mike code
    cdef void init_mse_mike(self, DOUBLE_t* y, SIZE_t y_stride, DOUBLE_t* sample_weight,
                   DOUBLE_f weighted_n_samples, SIZE_t* samples, SIZE_t start,
                   SIZE_t end, 
                   DOUBLE_f weighted_n_node_samples,
                   DOUBLE_f* sum_total,
                   DOUBLE_f* sq_sum_total,
                   DOUBLE_t* y_sq) nogil

    cdef void reset(self) nogil
    cdef void update(self, SIZE_t new_pos) nogil

    # Mike code
    cdef void update_mse_mike(self, SIZE_t new_pos, int identity_weight) nogil

    cdef DOUBLE_f node_impurity(self) nogil
    cdef void children_impurity(self, DOUBLE_f* impurity_left,
                                DOUBLE_f* impurity_right) nogil

    # MSE_MIKE, I apparently access fields of fields or that would require the Python gil
    cdef void children_sums(self, DOUBLE_f* weighted_n_left,
                                  DOUBLE_f* weighted_n_right,
                                  DOUBLE_f* sum_left,
                                  DOUBLE_f* sum_right,
                                  DOUBLE_f* sq_sum_left,
                                  DOUBLE_f* sq_sum_right) nogil

    cdef void node_value(self, DOUBLE_f* dest) nogil
    cdef DOUBLE_f impurity_improvement(self, DOUBLE_f impurity) nogil


# =============================================================================
# Splitter
# =============================================================================

cdef class Splitter:
    # The splitter searches in the input space for a feature and a threshold
    # to split the samples samples[start:end].
    #
    # The impurity computations are delegated to a criterion object.

    # Internal structures
    cdef public Criterion criterion      # Impurity criterion
    cdef public SIZE_t max_features      # Number of features to test
    cdef public SIZE_t min_samples_leaf  # Min samples in a leaf

    cdef object random_state             # Random state
    cdef UINT32_t rand_r_state           # sklearn_rand_r random number state

    cdef SIZE_t* samples                 # Sample indices in X, y
    cdef SIZE_t n_samples                # X.shape[0]
    cdef DOUBLE_f weighted_n_samples       # Weighted number of samples
    cdef SIZE_t* features                # Feature indices in X
    cdef SIZE_t* constant_features       # Constant features indices
    cdef SIZE_t n_features               # X.shape[1]
    cdef DTYPE_t* feature_values         # temp. array holding feature values
    cdef SIZE_t start                    # Start position for the current node
    cdef SIZE_t end                      # End position for the current node

    cdef DTYPE_t* X
    cdef SIZE_t X_sample_stride
    cdef SIZE_t X_fx_stride
    cdef DOUBLE_t* y

    # Mike code
    cdef DOUBLE_t* y_sq
    cdef SIZE_t y_stride
    cdef DOUBLE_t* sample_weight

    # Mike code
    cdef int identity_weight

    # Mike code
    cdef time_t copy_time
    cdef time_t count_sort_time
    cdef time_t search_time
    cdef time_t sort_time
    cdef time_t update_time

    # Mike code
    cdef SIZE_t n_outputs                # Number of outputs

    # The samples vector `samples` is maintained by the Splitter object such
    # that the samples contained in a node are contiguous. With this setting,
    # `node_split` reorganizes the node samples `samples[start:end]` in two
    # subsets `samples[start:pos]` and `samples[pos:end]`.

    # The 1-d  `features` array of size n_features contains the features
    # indices and allows fast sampling without replacement of features.

    # The 1-d `constant_features` array of size n_features holds in
    # `constant_features[:n_constant_features]` the feature ids with
    # constant values for all the samples that reached a specific node.
    # The value `n_constant_features` is given by the the parent node to its
    # child nodes.  The content of the range `[n_constant_features:]` is left
    # undefined, but preallocated for performance reasons
    # This allows optimization with depth-based tree building.

    # Methods
    # cdef void init(self, np.ndarray X, np.ndarray y, DOUBLE_t* sample_weight)

    # Mike code
    cdef void init(self, np.ndarray X, np.ndarray y, np.ndarray y_sq, DOUBLE_t* sample_weight)

    cdef void node_reset(self, SIZE_t start, SIZE_t end,
                         DOUBLE_f* weighted_n_node_samples) nogil

    # Mike code, a special node reset that memoizes computation of sums.
    cdef void node_reset_mse_mike(self, SIZE_t start, SIZE_t end,
                         DOUBLE_f* weighted_n_node_samples,
                         DOUBLE_f* sum_total,
                         DOUBLE_f* sq_sum_total) nogil

    # cdef void node_split(self,
    #                      DOUBLE_f impurity,   # Impurity of the node
    #                      SIZE_t* pos,       # Set to >= end if the node is a leaf
    #                      SIZE_t* feature,
    #                      DOUBLE_f* threshold,
    #                      DOUBLE_f* impurity_left,
    #                      DOUBLE_f* impurity_right,
    #                      DOUBLE_f* impurity_improvement,
    #                      SIZE_t* n_constant_features) nogil

    # Mike code
    cdef void node_split(self,
                         DOUBLE_f impurity,   # Impurity of the node
                         SIZE_t* pos,       # Set to >= end if the node is a leaf
                         SIZE_t* feature,
                         DOUBLE_f* threshold,
                         DOUBLE_f* impurity_left,
                         DOUBLE_f* impurity_right,
                         DOUBLE_f* weighted_n_left,
                         DOUBLE_f* weighted_n_right,
                         DOUBLE_f* sum_left,
                         DOUBLE_f* sum_right,
                         DOUBLE_f* sq_sum_left,
                         DOUBLE_f* sq_sum_right,
                         DOUBLE_f* impurity_improvement,
                         SIZE_t* n_constant_features) nogil

    cdef void node_value(self, DOUBLE_f* dest) nogil

    cdef DOUBLE_f node_impurity(self) nogil


# =============================================================================
# Tree
# =============================================================================

cdef struct Node:
    # Base storage structure for the nodes in a Tree object

    SIZE_t left_child                    # id of the left child of the node
    SIZE_t right_child                   # id of the right child of the node
    SIZE_t feature                       # Feature used for splitting the node
    DOUBLE_t threshold                   # Threshold value at the node
    DOUBLE_t impurity                    # Impurity of the node (i.e., the value of the criterion)
    SIZE_t n_node_samples                # Number of samples at the node
    DOUBLE_t weighted_n_node_samples     # Weighted number of samples at the node

cdef class Tree:
    # The Tree object is a binary tree structure constructed by the
    # TreeBuilder. The tree structure is used for predictions and
    # feature importances.

    # Input/Output layout
    cdef public SIZE_t n_features        # Number of features in X
    cdef SIZE_t* n_classes               # Number of classes in y[:, k]
    cdef public SIZE_t n_outputs         # Number of outputs in y
    cdef public SIZE_t max_n_classes     # max(n_classes)

    # Inner structures: values are stored separately from node structure,
    # since size is determined at runtime.
    cdef public SIZE_t max_depth         # Max depth of the tree
    cdef public SIZE_t node_count        # Counter for node IDs
    cdef public SIZE_t capacity          # Capacity of tree, in terms of nodes
    cdef Node* nodes                     # Array of nodes
    cdef DOUBLE_f* value                   # (capacity, n_outputs, max_n_classes) array of values
    cdef SIZE_t value_stride             # = n_outputs * max_n_classes

    # Methods
    cdef SIZE_t _add_node(self, SIZE_t parent, bint is_left, bint is_leaf,
                          SIZE_t feature, DOUBLE_f threshold, DOUBLE_f impurity,
                          SIZE_t n_node_samples,
                          DOUBLE_f weighted_n_samples) nogil
    cdef void _resize(self, SIZE_t capacity)
    cdef int _resize_c(self, SIZE_t capacity=*) nogil

    cdef np.ndarray _get_value_ndarray(self)
    cdef np.ndarray _get_node_ndarray(self)

    cpdef np.ndarray predict(self, np.ndarray[DTYPE_t, ndim=2] X)
    cpdef np.ndarray apply(self, np.ndarray[DTYPE_t, ndim=2] X)
#    cpdef compute_feature_importances(self, normalize=*)
    cpdef compute_feature_importances(self, normalize=*, weighted=*)


# =============================================================================
# Tree builder
# =============================================================================

cdef class TreeBuilder:
    # The TreeBuilder recursively builds a Tree object from training samples,
    # using a Splitter object for splitting internal nodes and assigning
    # values to leaves.
    #
    # This class controls the various stopping criteria and the node splitting
    # evaluation order, e.g. depth-first or best-first.

    cdef Splitter splitter          # Splitting algorithm

    cdef SIZE_t min_samples_split   # Minimum number of samples in an internal node
    cdef SIZE_t min_samples_leaf    # Minimum number of samples in a leaf
    cdef SIZE_t max_depth           # Maximal tree depth

    # cpdef build(self, Tree tree, np.ndarray X, np.ndarray y,
    #             np.ndarray sample_weight=*)

    # Mike code
    cpdef build(self, Tree tree, np.ndarray X, np.ndarray y, np.ndarray y_sq, np.ndarray sample_weight=*)
