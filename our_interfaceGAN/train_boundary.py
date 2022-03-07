import numpy as np
from sklearn import svm


def train_boundary(latent_codes,
                   scores,
                   chosen_num_or_ratio=0.02,
                   split_ratio=0.7,
                   invalid_value=None):
    """Trains boundary in latent space with offline predicted attribute scores.
    Given a collection of latent codes and the attribute scores predicted from the
    corresponding images, this function will train a linear SVM by treating it as
    a bi-classification problem. Basically, the samples with highest attribute
    scores are treated as positive samples, while those with lowest scores as
    negative. For now, the latent code can ONLY be with 1 dimension.
    NOTE: The returned boundary is with shape (1, latent_space_dim), and also
    normalized with unit norm.
    Args:
      latent_codes: Input latent codes as training data.
      scores: Input attribute scores used to generate training labels.
      chosen_num_or_ratio: How many samples will be chosen as positive (negative)
        samples. If this field lies in range (0, 0.5], `chosen_num_or_ratio *
        latent_codes_num` will be used. Otherwise, `min(chosen_num_or_ratio,
        0.5 * latent_codes_num)` will be used. (default: 0.02)
      split_ratio: Ratio to split training and validation sets. (default: 0.7)
      invalid_value: This field is used to filter out data. (default: None)
    Returns:
      A decision boundary with type `numpy.ndarray`.
    Raises:
      ValueError: If the input `latent_codes` or `scores` are with invalid format.
    """

    if (not isinstance(latent_codes, np.ndarray) or
            not len(latent_codes.shape) == 2):
        raise ValueError(f'Input `latent_codes` should be with type'
                         f'`numpy.ndarray`, and shape [num_samples, '
                         f'latent_space_dim]!')
    num_samples = latent_codes.shape[0]
    latent_space_dim = latent_codes.shape[1]
    if (not isinstance(scores, np.ndarray) or not len(scores.shape) == 2 or
            not scores.shape[0] == num_samples or not scores.shape[1] == 1):
        raise ValueError(f'Input `scores` should be with type `numpy.ndarray`, and '
                         f'shape [num_samples, 1], where `num_samples` should be '
                         f'exactly same as that of input `latent_codes`!')
    if chosen_num_or_ratio <= 0:
        raise ValueError(f'Input `chosen_num_or_ratio` should be positive, '
                         f'but {chosen_num_or_ratio} received!')

    print('Filtering training data.')
    if invalid_value is not None:
        latent_codes = latent_codes[scores[:, 0] != invalid_value]
        scores = scores[scores[:, 0] != invalid_value]

    print('Sorting scores to get positive and negative samples.')
    sorted_idx = np.argsort(scores, axis=0)[::-1, 0]
    latent_codes = latent_codes[sorted_idx]
    scores = scores[sorted_idx]
    num_samples = latent_codes.shape[0]
    if 0 < chosen_num_or_ratio <= 1:
        chosen_num = int(num_samples * chosen_num_or_ratio) # chosen_num个样本
    else:
        chosen_num = int(chosen_num_or_ratio)
    chosen_num = min(chosen_num, num_samples // 2)
    print(f"sample range: >{scores[chosen_num]}, <{scores[-chosen_num]}")

    print('Spliting training and validation sets:')
    train_num = int(chosen_num * split_ratio)
    val_num = chosen_num - train_num
    # Positive samples.
    positive_idx = np.arange(chosen_num)
    np.random.shuffle(positive_idx)
    positive_train = latent_codes[:chosen_num][positive_idx[:train_num]]
    positive_val = latent_codes[:chosen_num][positive_idx[train_num:]]
    # Negative samples.
    negative_idx = np.arange(chosen_num)
    np.random.shuffle(negative_idx)
    negative_train = latent_codes[-chosen_num:][negative_idx[:train_num]]
    negative_val = latent_codes[-chosen_num:][negative_idx[train_num:]]
    # Training set.
    train_data = np.concatenate([positive_train, negative_train], axis=0)
    train_label = np.concatenate([np.ones(train_num, dtype=np.int),
                                  np.zeros(train_num, dtype=np.int)], axis=0)
    print(f'  Training: {train_num} positive, {train_num} negative.') # Training: 1400 positive, 14 negative.
    # Validation set.
    val_data = np.concatenate([positive_val, negative_val], axis=0)
    val_label = np.concatenate([np.ones(val_num, dtype=np.int),
                                np.zeros(val_num, dtype=np.int)], axis=0)
    print(f'  Validation: {val_num} positive, {val_num} negative.')
    # Remaining set.
    remaining_num = num_samples - chosen_num * 2
    remaining_data = latent_codes[chosen_num:-chosen_num]
    remaining_scores = scores[chosen_num:-chosen_num]
    decision_value = (scores[0] + scores[-1]) / 2
    remaining_label = np.ones(remaining_num, dtype=np.int)
    remaining_label[remaining_scores.ravel() < decision_value] = 0
    remaining_positive_num = np.sum(remaining_label == 1)
    remaining_negative_num = np.sum(remaining_label == 0)
    print(f'  Remaining: {remaining_positive_num} positive, '
          f'{remaining_negative_num} negative.')

    print(f'Training boundary.')
    # scaler = StandardScaler()
    # scaler.fit(train_data)
    #
    # train_transformed = scaler.transform(train_data)
    # val_transformed = scaler.transform(val_data)
    # # remaining_transformed = scaler.transform(remaining_data)
    #
    # clf = SGDClassifier()
    # classifier = clf.fit(train_transformed, train_label)
    # print(f"Actual number of iterations: {classifier.n_iter_}")

    clf = svm.SVC(kernel='linear')
    classifier = clf.fit(train_data, train_label)
    print(f'Finish training.')

    if train_num:
        train_prediction = classifier.predict(train_data)
        correct_num = np.sum(train_label == train_prediction)
        print(f'Accuracy for training set: '
              f'{correct_num} / {train_num * 2} = '
              f'{correct_num / (train_num * 2):.6f}')

    if val_num:
        val_prediction = classifier.predict(val_data)
        correct_num = np.sum(val_label == val_prediction)
        print(f'Accuracy for validation set: '
              f'{correct_num} / {val_num * 2} = '
              f'{correct_num / (val_num * 2):.6f}')

    # if remaining_num:
    #     remaining_prediction = classifier.predict(remaining_data)
    #     correct_num = np.sum(remaining_label == remaining_prediction)
    #     print(f'Accuracy for remaining set: '
    #           f'{correct_num} / {remaining_num} = '
    #           f'{correct_num / remaining_num:.6f}')

    a = classifier.coef_.reshape(1, latent_space_dim).astype(np.float32)
    return a / np.linalg.norm(a)

