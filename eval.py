#! /usr/bin/env python

import numpy as np
import tensorflow as tf
import data_helpers

# Define Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_string("sentence", "the movie was bad", "sentence to classify")

FLAGS = tf.flags.FLAGS

#######################################################################################################################
# process the raw sentence
new_review = data_helpers.clean_senetnce(FLAGS.sentence)

# load vocabulary
sentences, _ = data_helpers.load_data_and_labels()
sequence_length = max(len(x) for x in sentences)
sentences_padded = data_helpers.pad_sentences(sentences)
vocabulary, vocabulary_inv = data_helpers.build_vocab(sentences_padded)

num_padding = sequence_length - len(new_review)
new_sentence = new_review + ["<PAD/>"] * num_padding

# convert sentence to input matrix
array = []
for word in new_sentence:
    try:
        word_vector=vocabulary[word]
    except KeyError:
        word_vector=vocabulary["<PAD/>"]
    array.append(word_vector)
x=np.array([array])

#######################################################################################################################

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto()
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(x, FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, feed_dict={input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy
print('entered text: {}'.format(FLAGS.sentence))
print('predictions: {}'.format('negative' if all_predictions[0] == 0.0 else 'positive'))
