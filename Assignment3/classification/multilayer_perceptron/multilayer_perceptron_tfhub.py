import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow_hub as hub


def get_predictions(estimator, input_fn):
    return [x["class_ids"][0] for x in estimator.predict(input_fn=input_fn)]


def classify(train_df: pd.DataFrame, test_df: pd.DataFrame):
    # Training input on the whole training set with no limit on training epochs.
    train_input_fn = tf.estimator.inputs.pandas_input_fn(
        train_df, train_df["class"], num_epochs=50, shuffle=True)

    # Prediction on the whole training set.
    predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
        train_df, train_df["class"], shuffle=False)
    # Prediction on the test set.
    predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
        test_df, test_df["class"], shuffle=False)

    print("Download pretrained model")
    embed = hub.text_embedding_column(
        key="sentence",
        module_spec="https://tfhub.dev/google/nnlm-de-dim128/1",
        trainable=True)

    print("Train estimator")
    estimator = tf.estimator.DNNClassifier(
        hidden_units=[500, 100],
        feature_columns=[embed],
        n_classes=3,
        dropout=0.6,
        optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))

    estimator.train(input_fn=train_input_fn)

    print("Evaluate estimator")
    train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
    test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)

    training_set_accuracy = train_eval_result["accuracy"]
    test_set_accuracy = test_eval_result["accuracy"]

    print("Training set accuracy: ")
    print(training_set_accuracy)
    print("Test set accuracy: ")
    print(test_set_accuracy)

    LABELS = [
        "positive", "negative", "neutral"
    ]

    # Create a confusion matrix on training data.
    with tf.Graph().as_default():
        cm = tf.confusion_matrix(test_df["class"],
                                 get_predictions(estimator, predict_test_input_fn))
        with tf.Session() as session:
            cm_out = session.run(cm)

    # Normalize the confusion matrix so that each row sums to 1.
    cm_out = cm_out.astype(float) / cm_out.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_out, annot=True, xticklabels=LABELS, yticklabels=LABELS);
    plt.xlabel("Predicted");
    plt.ylabel("True");
    plt.show()
