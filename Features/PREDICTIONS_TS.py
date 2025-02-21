import pandas as pd
import numpy as np


# ========================== Predictions Features ==========================
def average_predictions_features(
    predictions_series: pd.Series,
    window: int,
):
    """
    This function computes the rolling average of the predictions.

    Args:
        predictions_series (pd.Series): The series of predictions.
        window (int): The window size for the rolling average.

    Returns:
        pd.Series: The rolling average of the predictions.
    """
    # ======= I. Compute the rolling average =======
    rolling_avg = predictions_series.rolling(window=window + 1).mean()

    # ======= II. Convert to pd.Series and Normalize =======
    rolling_avg = pd.Series(rolling_avg, index=predictions_series.index)

    return rolling_avg


# -----------------------------------------------------------------------------
def volatility_predictions_features(
    predictions_series: pd.Series,
    window: int,
):
    """
    This function computes the rolling volatility of the predictions.

    Args:
        predictions_series (pd.Series): The series of predictions.
        window (int): The window size for the rolling volatility.

    Returns:
        pd.Series: The rolling volatility of the predictions.
    """
    # ======= I. Compute the rolling volatility =======
    rolling_volatility = predictions_series.rolling(window=window + 1).std()

    # ======= II. Convert to pd.Series and Normalize =======
    rolling_volatility = pd.Series(rolling_volatility, index=predictions_series.index)

    return rolling_volatility


# -----------------------------------------------------------------------------
def predictions_changes_features(
    predictions_series: pd.Series,
    window: int,
):
    # ======= 0. Auxiliary function =======
    def compute_changes(series: pd.Series):
        diff_series = series.diff() ** 2
        changes_count = diff_series[diff_series > 0].count()

        return changes_count

    # ======= I. Compute the rolling changes =======
    rolling_changes = predictions_series.rolling(window=window + 1).apply(compute_changes, raw=False)

    # ======= II. Convert to pd.Series and Normalize =======
    rolling_changes = pd.Series(rolling_changes, index=predictions_series.index)

    return rolling_changes


# -----------------------------------------------------------------------------
def entropy_predictions_features(
    predictions_series: pd.Series,
    window: int,
):
    # ======= II. Compute the rolling entropy features =======
    rolling_shannon = predictions_series.rolling(window=window + 1).apply(get_shannon_entropy, raw=False)
    rolling_plugin = predictions_series.rolling(window=window + 1).apply(get_plugin_entropy, raw=False)
    rolling_lempel_ziv = predictions_series.rolling(window=window + 1).apply(get_lempel_ziv_entropy, raw=False)
    rolling_kontoyiannis = predictions_series.rolling(window=window + 1).apply(get_kontoyiannis_entropy, raw=False)

    # ======= III. Convert to pd.Series and Center =======
    rolling_shannon = pd.Series(rolling_shannon, index=predictions_series.index)
    rolling_plugin = pd.Series(rolling_plugin, index=predictions_series.index)
    rolling_lempel_ziv = pd.Series(rolling_lempel_ziv, index=predictions_series.index)
    rolling_kontoyiannis = pd.Series(rolling_kontoyiannis, index=predictions_series.index)

    return rolling_shannon, rolling_plugin, rolling_lempel_ziv, rolling_kontoyiannis


# =========================================================================
# ========================== Auxiliary functions ==========================
def get_shannon_entropy(signs_series: pd.Series):
    """
    This function computes the Shannon entropy of a series of signs.

    Args:
        signs_series (pd.Series): The series of signs (must be a series of 1s and -1s).

    Returns:
        float: The Shannon entropy of the series of signs.
    """
    # ======= I. Count the frequency of each symbol in the data =======
    _, counts = np.unique(signs_series, return_counts=True)

    # ======= II. Compute frequentist probabilities =======
    probabilities = counts / len(signs_series)

    # ======= III. Compute Shannon entropy =======
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy


# -----------------------------------------------------------------------------
def get_plugin_entropy(signs_series: pd.Series, word_length: int = 1):
    """
    This function computes the plug-in entropy estimator.

    Args:
        signs_series (pd.Series): The series of signs (must be a series of 1s and -1s).
        word_length (int): The approximate word length.

    Returns:
        float: The plug-in entropy.
    """

    # ======= 0. Auxiliary Function =======
    def compute_pmf(message: str, word_length: int):
        """
        This function computes the probability mass function for a one-dimensional discrete random variable.

        Args:
            message (str or array): Encoded message.
            word_length (int): Approximate word length.

        Returns:
            dict: Dictionary of the probability mass function for each word from the message.
        """
        # ======= I. Initialize the dictionary that will store the different words and their indexes of appearance =======
        unique_words_indexes = {}

        # ======= II. Iterate through the message to store the indexes of appareance of each different word =======
        for i in range(word_length, len(message)):
            word = message[i - word_length : i]
            word_index = i - word_length

            if word not in unique_words_indexes:
                # If the word is not in the dictionary, add it and store the index of appearance
                unique_words_indexes[word] = [word_index]

            else:
                # If the word is already in the dictionary, append the index of appearance
                unique_words_indexes[word] = unique_words_indexes[word] + [word_index]

        # ======= III. Compute the probability mass function =======
        total_count = float(len(message) - word_length)
        pmf = {word: len(unique_words_indexes[word]) / total_count for word in unique_words_indexes}

        return pmf

    # ======= I. Convert the signs series to a string =======
    sequence = "".join(map(str, signs_series))

    # ======= II. Compute the probability mass function =======
    pmf = compute_pmf(sequence, word_length)

    # ======= III. Compute the plug-in entropy =======
    entropy = -sum([pmf[i] * np.log2(pmf[i]) for i in pmf]) / word_length

    return entropy


# -----------------------------------------------------------------------------
def get_lempel_ziv_entropy(signs_series: pd.Series):
    """
    This function computes the Lempel-Ziv entropy of a series of signs.

    Args:
        signs_series (pd.Series): The series of signs (must be a series of 1s and -1s).

    Returns:
        float: The Lempel-Ziv entropy of the series of signs.
    """

    # ======= 0. Convert the signs series to a list =======
    signs_series_list = list(signs_series)
    series_size = len(signs_series_list)

    # ======= I. Initialize the variables =======
    complexity = 0
    patterns = []

    # ======= II. Iterate through the list to extract the different patterns =======
    index = 0
    index_extension = 1
    while index + index_extension <= series_size:
        if signs_series_list[index : index + index_extension] not in patterns:
            # II.1 Found new pattern : increment the complexity and add the pattern to the list
            patterns.append(signs_series_list[index : index + index_extension])
            complexity += 1

            # updates the index and the extension
            index += index_extension
            index_extension = 1
        else:
            # II.2 The current pattern was already seen : update the extension
            index_extension += 1

    # ======= III. Compute the Lempel-Ziv entropy =======
    entropy = complexity / series_size

    return entropy


# -----------------------------------------------------------------------------
def get_kontoyiannis_entropy(signs_series: pd.Series, window=None):
    """
    This function computes the Kontoyiannis entropy of a series of signs.

    Args:
        signs_series (pd.Series): The series of signs (must be a series of 1s and -1s).
        window (int): The window size for the rolling entropy.

    Returns:
        float: The Kontoyiannis entropy of the series of signs
    """

    # ======= 0. Auxiliary Functions =======
    def matchLength(message: str, starting_index: int, maximum_length: int):
        """
        This function computes the length of the longest substring that appears at least twice in the message.

        Args:
            message (str): The message to analyze.
            starting_index (int): The starting index of the substring.
            maximum_length (int): The maximum length of the substring.

        Returns:
            int: The length of the longest substring.
            str: The longest substring.
        """
        # ======= I. Initialize the longest sub string =======
        longest_substring = ""

        # ======= II. Iterate through the message to find the longest substring =======
        for possible_length in range(maximum_length):
            # II.1. Extract the maximum substring
            maximum_substring = message[starting_index : starting_index + possible_length + 1]

            # II.2. Iterate through the message to find the longest substring
            for index in range(starting_index - maximum_length, starting_index):
                # II.2.1. Extract the substring to compare
                substring = message[index : index + possible_length + 1]

                # II.2.2. Check if the substring is the longest
                if maximum_substring == substring:
                    longest_substring = maximum_substring
                    break

        # ======= III. Compute the length of the longest substring =======
        longest_substring_length = len(longest_substring) + 1

        return longest_substring_length, longest_substring

    # ======= I. Convert the signs series to a string =======
    out = {"nb_patterns": 0, "sum": 0, "patterns": []}
    message = signs_series.copy()
    message = "".join(map(str, message))

    # ======= II. Extract the starting indexes (no need to iterate after half the series as a pattern would be found before if existing) =======
    if window is None:
        starting_indexes = range(1, len(message) // 2 + 1)
    else:
        window = min(window, len(message) // 2)
        starting_indexes = range(window, len(message) - window + 1)

    # ======= III. Compute the Kontoyiannis entropy =======
    for index in starting_indexes:
        # III.1. Compute the longest substring and its length
        if window is None:
            longest_pattern_length, longest_pattern = matchLength(message=message, starting_index=index, maximum_length=index)
            out["sum"] += np.log2(index + 1) / longest_pattern_length
        else:
            longest_pattern_length, longest_pattern = matchLength(message=message, starting_index=index, maximum_length=window)
            out["sum"] += np.log2(window + 1) / longest_pattern_length

        # III.2. Update the number of patterns and the list of patterns
        out["patterns"].append(longest_pattern)
        out["nb_patterns"] += 1

    # ======= IV. Compute the entropy and redundancy =======
    out["entropy"] = out["sum"] / out["nb_patterns"]
    out["redundancy"] = 1 - out["entropy"] / np.log2(len(message))

    entropy = out["entropy"] if out["entropy"] < 1 else 1

    return entropy
