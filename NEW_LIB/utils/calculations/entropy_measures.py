import numpy as np 
import pandas as pd



#! ==================================================================================== #
#! ============================= Preprocessing Function =============================== #
def get_movements_signs(
    series: pd.Series
) -> pd.Series:
    """
    Transform a series of values into a series of signs.
    
    Parameters:
        - series (pd.Series): Series of values.
    
    Returns:
        - signs_series (pd.Series): Series of signs.
    """
    # ======= I. Compute the sign of the returns =======
    diff_series = series.diff()
    signs_series = np.sign(diff_series)

    # ======= II. Adjust for incorrect value =======
    signs_series.iloc[0] = 1
    signs_series[signs_series == 0] = np.nan
    signs_series = signs_series.ffill()

    return signs_series



#! ==================================================================================== #
#! =========================== Series Entropy Functions =============================== #
def get_shannon_entropy(
    signs_series: pd.Series
) -> float:
    """
    Compute Shannon entropy.
    
    Parameters:
        - signs_series (pd.Series): Series of signs.
    
    Returns:
        - entropy (float): Shannon entropy.
    """
    # ======= I. Count the frequency of each symbol in the data =======
    _, counts = np.unique(signs_series, return_counts=True)

    # ======= II. Compute frequentist probabilities =======
    probabilities = counts / len(signs_series)

    # ======= III. Compute Shannon entropy =======
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy

#*____________________________________________________________________________________ #
def get_plugin_entropy(
    signs_series: pd.Series, 
    word_length: int = 1
) -> float:
    """
    Compute plug-in entropy.
    
    Parameters:
        - signs_series (pd.Series): Series of signs.
        - word_length (int): Length of the words to consider.
    
    Returns:
        - entropy (float): Plug-in entropy.
    """
    # ======= 0. Auxiliary Function =======
    def compute_pmf(
        message: str, 
        word_length: int
    ) -> dict:
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
    message = signs_series.copy()
    message -= message.min()
    message = [int(x) for x in message]

    message = "".join(map(str, message))

    # ======= II. Compute the probability mass function =======
    pmf = compute_pmf(message, word_length)

    # ======= III. Compute the plug-in entropy =======
    entropy = -sum([pmf[i] * np.log2(pmf[i]) for i in pmf]) / word_length

    return entropy

#*____________________________________________________________________________________ #
def get_lempel_ziv_entropy(
    signs_series: pd.Series
) -> float:
    """
    Compute Lempel-Ziv entropy.
    
    Parameters:
        - signs_series (pd.Series): Series of signs.
    
    Returns:
        - entropy (float): Lempel-Ziv entropy.
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

#*____________________________________________________________________________________ #
def get_kontoyiannis_entropy(
    signs_series: pd.Series, 
    window=None
) -> float:
    """
    Compute Kontoyiannis entropy.
    
    Parameters:
        - signs_series (pd.Series): Series of signs.
        - window (int): Window size for the analysis.
    
    Returns:
        - entropy (float): Kontoyiannis entropy.
    """
    # ======= 0. Auxiliary Functions =======
    def matchLength(
        message: str, 
        starting_index: int, 
        maximum_length: int
    ) -> tuple:
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
    message -= message.min()
    message = [int(x) for x in message]

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

#*____________________________________________________________________________________ #
def get_gini_impurity(
    signs_series: pd.Series
) -> float:
    """
    Computes Gini Impurity.
    
    Parameters:
        - signs_series (pd.Series): Series of signs.
    
    Returns:
        - gini_impurity (float): Gini impurity.
    """
    # ======= I. Count the frequency of each symbol in the data =======
    classes, counts = np.unique(signs_series, return_counts=True)
    probs = counts / counts.sum()
    
    # ======= II. Compute Gini impurity =======
    gini_impurity = 1 - np.sum(probs**2)
    
    return gini_impurity

#*____________________________________________________________________________________ #
def calculate_sample_entropy(
    series: pd.Series, 
    sub_vector_size: int = 2, 
    threshold_distance: float = 0.2
) -> float:
    """
    Compute the sample entropy of a time series.
    
    Parameters:
        - series (pd.Series): Time series data.
        - sub_vector_size (int): Size of the sub-vectors to consider.
        - threshold_distance (float): Threshold distance for similarity.
    
    Returns:
        - sample_entropy (float): Sample entropy of the time series.
    """
    #?____________________________________________________________________________________ #
    def get_maximum_distance(
        vector_1: list, 
        vector_2: list
    ) -> float:
        """
        Compute the maximum distance between two vectors.
        
        Parameters:
            - vector_1 (list): First vector.
            - vector_2 (list): Second vector.
        
        Returns:
            - max_distance (float): Maximum distance between the two vectors.
        """
        distances = [abs(ua - va) for ua, va in zip(vector_1, vector_2)]
        max_distance = max(distances)
        
        return max_distance
    
    #?____________________________________________________________________________________ #
    def get_phi(
        series: pd.Series, 
        sub_vector_size: int, 
        r_abs: float
        ) -> int:
        """
        Compute the phi value for a given series and sub-vector size.
        
        Parameters:
            - series (pd.Series): Time series data.
            - sub_vector_size (int): Size of the sub-vectors to consider.
            - r_abs (float): Threshold distance for similarity.
        
        Returns:
            - total_similarity_count (int): Total similarity count.
        """
        # ======= I. Create rolling sub-vectors =======
        series_size = len(series)
        rolling_sub_vectors = [
            [series.iloc[i + j] for j in range(sub_vector_size)]
            for i in range(series_size - sub_vector_size + 1)
        ]


        # ======= II. Count the number of similar sub-vectors =======
        similarity_counts = []
        for i, vector_i in enumerate(rolling_sub_vectors):
            count = sum(1 for j, vector_j in enumerate(rolling_sub_vectors) if i != j and get_maximum_distance(vector_i, vector_j) <= r_abs)
            similarity_counts.append(count)

        # ======= III. Compute the total similarity count =======
        total_similarity_count = sum(similarity_counts)
        
        return total_similarity_count
    
    #?____________________________________________________________________________________ #
    try:
        r_abs = threshold_distance * np.std(series)
        phi = get_phi(series, sub_vector_size, r_abs)
        phi_plus_1 = get_phi(series, sub_vector_size + 1, r_abs)

        sample_entropy = -np.log(phi_plus_1 / phi)
    
    except Exception:
        sample_entropy = np.nan
    
    return sample_entropy

