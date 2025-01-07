import tensorflow as tf
import keras

# from tensorflow import keras
# from tensorflow.keras import backend as K
from typing import Any, Dict, Optional


class FBetaScore(keras.metrics.Metric):
    """Computes F-beta score.

    This is the weighted harmonic mean of precision and recall.
    Output range is [0, 1]. Works for both multi-class and multi-label classification.
    """

    def __init__(
        self,
        num_classes: int,
        average: Optional[str] = None,
        beta: float = 1.0,
        threshold: Optional[float] = None,
        name: str = "fbeta_score",
        dtype: Any = None,
        **kwargs: Any,
    ) -> None:
        """Creates an F-Beta score instance.

        Args:
            num_classes: Number of unique classes in the dataset.
            average: Type of averaging to be performed on data.
                    Acceptable values are `None`, `micro`, `macro` and
                    `weighted`. Default value is None.
            beta: Determines the weight of precision and recall in harmonic mean.
                 Determines the weight given to the precision and recall.
                 Default value is 1.0.
            threshold: Elements of `y_pred` greater than threshold are
                converted to be 1, and the rest 0. If threshold is
                None, the argmax is converted to 1, and the rest 0.
            name: (Optional) String name of the metric instance.
            dtype: (Optional) Data type of the metric result.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)

        if average not in (None, "micro", "macro", "weighted"):
            raise ValueError(
                "Unknown average type. Acceptable values "
                "are: [None, 'micro', 'macro', 'weighted']"
            )

        if beta <= 0.0:
            raise ValueError("The value of beta should be greater than zero")

        if threshold is not None:
            if threshold > 1.0 or threshold <= 0.0:
                raise ValueError("threshold should be between 0 and 1")

        self.num_classes = num_classes
        self.average = average
        self.beta = beta
        self.threshold = threshold
        self.axis = None
        self.init_shape = []

        if self.average != "micro":
            self.axis = 0
            self.init_shape = [self.num_classes]

        def _zero_wt_init(name):
            return self.add_weight(
                name, shape=self.init_shape, initializer="zeros", dtype=self.dtype
            )

        self.true_positives = _zero_wt_init("true_positives")
        self.false_positives = _zero_wt_init("false_positives")
        self.false_negatives = _zero_wt_init("false_negatives")
        self.weights_intermediate = _zero_wt_init("weights_intermediate")

    def update_state(
        self,
        y_true: tf.types.experimental.TensorLike,
        y_pred: tf.types.experimental.TensorLike,
        sample_weight: Optional[tf.types.experimental.TensorLike] = None,
    ) -> None:
        """Accumulates confusion matrix statistics.

        Args:
            y_true: The ground truth values.
            y_pred: The predicted values.
            sample_weight: Optional weighting of each example. Defaults to 1.
                        Can be a `Tensor` whose rank is either 0, or the same rank as
                        `y_true`, and must be broadcastable to `y_true`.
        """
        if self.threshold is None:
            threshold = tf.reduce_max(y_pred, axis=-1, keepdims=True)
            # make sure [0, 0, 0] doesn't become [1, 1, 1]
            # Use abs(x) > eps, instead of x != 0 to check for zero
            y_pred = tf.logical_and(y_pred >= threshold, tf.abs(y_pred) > 1e-12)
        else:
            y_pred = y_pred > self.threshold

        y_true = tf.cast(y_true, self.dtype)
        y_pred = tf.cast(y_pred, self.dtype)

        def _weighted_sum(
            val: tf.types.experimental.TensorLike,
            sample_weight: Optional[tf.types.experimental.TensorLike],
        ) -> tf.types.experimental.TensorLike:
            if sample_weight is not None:
                val = tf.math.multiply(val, tf.expand_dims(sample_weight, 1))
            return tf.reduce_sum(val, axis=self.axis)

        self.true_positives.assign_add(_weighted_sum(y_pred * y_true, sample_weight))
        self.false_positives.assign_add(
            _weighted_sum(y_pred * (1 - y_true), sample_weight)
        )
        self.false_negatives.assign_add(
            _weighted_sum((1 - y_pred) * y_true, sample_weight)
        )
        self.weights_intermediate.assign_add(_weighted_sum(y_true, sample_weight))

    def result(self) -> tf.types.experimental.TensorLike:
        precision = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_positives
        )
        recall = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_negatives
        )

        mul_value = precision * recall
        add_value = (tf.math.square(self.beta) * precision) + recall
        mean = tf.math.divide_no_nan(mul_value, add_value)
        f1_score = mean * (1 + tf.math.square(self.beta))

        if self.average == "weighted":
            weights = tf.math.divide_no_nan(
                self.weights_intermediate, tf.reduce_sum(self.weights_intermediate)
            )
            f1_score = tf.reduce_sum(f1_score * weights)

        elif self.average is not None:  # [micro, macro]
            f1_score = tf.reduce_mean(f1_score)

        return f1_score

    def get_config(self) -> Dict[str, Any]:
        """Returns the serializable config of the metric."""

        config = {
            "num_classes": self.num_classes,
            "average": self.average,
            "beta": self.beta,
            "threshold": self.threshold,
        }

        base_config = super().get_config()
        return {**base_config, **config}

    def reset_state(self) -> None:
        """Resets all of the metric state variables."""
        self.true_positives.assign(tf.zeros(self.init_shape, self.dtype))
        self.false_positives.assign(tf.zeros(self.init_shape, self.dtype))
        self.false_negatives.assign(tf.zeros(self.init_shape, self.dtype))
        self.weights_intermediate.assign(tf.zeros(self.init_shape, self.dtype))


class F1Score(FBetaScore):
    """Computes F-1 Score.

    This is the harmonic mean of precision and recall.
    Output range is [0, 1]. Works for both multi-class and multi-label classification.
    """

    def __init__(
        self,
        num_classes: int,
        average: Optional[str] = None,
        threshold: Optional[float] = None,
        name: str = "f1_score",
        dtype: Any = None,
    ):
        """Creates an F-1 score instance.

        Args:
            num_classes: Number of unique classes in the dataset.
            average: Type of averaging to be performed on data.
                    Acceptable values are `None`, `micro`, `macro` and
                    `weighted`. Default value is None.
            threshold: Elements of `y_pred` greater than threshold are
                converted to be 1, and the rest 0. If threshold is
                None, the argmax is converted to 1, and the rest 0.
            name: (Optional) String name of the metric instance.
            dtype: (Optional) Data type of the metric result.
        """
        super().__init__(
            num_classes=num_classes,
            average=average,
            beta=1.0,
            threshold=threshold,
            name=name,
            dtype=dtype,
        )
