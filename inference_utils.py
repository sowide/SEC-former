import numpy as np
import tensorflow as tf
from transformers import TFXLNetModel, XLNetTokenizer


def get_model(path: str) -> tf.keras.Model:
    """
    This method allows you to load and get the full model, ready for inference.

    :param path: The path to the .h5 fine-tuned model.
    :return: The loaded model from given path.
    """
    return tf.keras.models.load_model(path, custom_objects={'TFXLNetModel': TFXLNetModel})


def inference(text: str or list, model: tf.keras.Model):
    """
    This method is used to make inference, i.e. to classify one or more paragraph based on
    standard 10-Ks items provided by SEC.

    :param text: A string representing the paragraph to classify.
    :param model: The instance of the model with which you want to make inference.
    """

    labels = ['Item 1', 'Item 1a', 'Item 2', 'Item 3', 'Item 4', 'Item 5',
              'Item 6', 'Item 7', 'Item 7a', 'Item 8', 'Item 9', 'Item 9a',
              'Item 9b', 'Item 10', 'Item 11', 'Item 12', 'Item 13', 'Item 15']
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
    encoded_input = tokenizer([text], padding='max_length', max_length=200, truncation=True)
    # Get score for each class.
    scores = model.predict(
        x=[np.array(encoded_input['input_ids']), np.array(encoded_input['attention_mask'])],
        batch_size=1, verbose=0
    )
    # Associate each score with correspondant class.
    scores = dict(zip(labels, scores.reshape(-1)))
    # Sort dict by values descending order (i.e. higher score first)
    scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    print(f'\nPredicted item: {list(scores.keys())[0]}.\nScores on each item:')
    for item, score in scores.items():
        print(f'{item}: {score:.4f}')
