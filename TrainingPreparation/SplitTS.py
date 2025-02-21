def proportional_split(
    training_proportion: float,
    validation_proportion: float,
    dfs_list: list,
):
    # ======= I. Initialization of input and output =======
    training_data_dfs_list = []
    validation_data_dfs_list = []
    testing_data_dfs_list = []

    # ======= II. Compute the index to split the data =======
    # II.1 Use the biggest asset to compute the size of the data
    biggest_size = 0
    for asset_df in dfs_list:
        size_data = len(asset_df)
        if size_data > biggest_size:
            biggest_size = size_data
            biggest_asset = asset_df

    # II.1 Training data indexes
    training_start_index = biggest_asset.index[0]
    training_end_index = biggest_asset.index[int(training_proportion * size_data)]

    # II.2 Validation data indexes
    validation_start_index = biggest_asset.index[int(training_proportion * size_data + 1)]
    validation_end_index = biggest_asset.index[int((validation_proportion + training_proportion) * size_data - 1)]

    # II.3 Testing data indexes
    if validation_proportion + training_proportion != 1:
        testing_start_index = biggest_asset.index[int((validation_proportion + training_proportion) * size_data + 1)]
        testing_end_index = biggest_asset.index[-1]
    else:
        testing_data_dfs_list = None

    # ======= III. Split the data =======
    for asset_df in dfs_list:
        # III.1 Training data
        training_data = asset_df.loc[training_start_index:training_end_index]
        training_data.dropna(axis=0)
        if len(training_data) > 50:
            training_data_dfs_list.append(training_data)

        # III.2 Validation data
        validation_data = asset_df.loc[validation_start_index:validation_end_index]
        validation_data.dropna(axis=0)
        if len(validation_data) > 50:
            validation_data_dfs_list.append(validation_data)

        # III.3 Testing data
        if validation_proportion + training_proportion < 1:
            testing_data = asset_df.loc[testing_start_index:testing_end_index]
            testing_data.dropna(axis=0)
            if len(testing_data) > 50:
                testing_data_dfs_list.append(testing_data)

    return training_data_dfs_list, validation_data_dfs_list, testing_data_dfs_list


# -----------------------------------------------------------------------------
def cartesius_split(
    dfs_list: list,
):
    # ======= I. Initialization of input and output =======
    training_data_dfs_list = []
    validation_data_dfs_list = []
    testing_data_dfs_list = []
    production_data_dfs_list = []
    covid_data_dfs_list = []

    # ======= II. Compute the index to split the data =======
    # II.1 Training data indexes
    training_start_index = "2011-01-01"
    training_end_index = "2017-01-01"

    # II.2 Validation data indexes
    validation_start_index = "2017-01-01"
    validation_end_index = "2021-07-01"

    # II.3 Testing data indexes
    testing_start_index = "2021-07-01"
    testing_end_index = "2023-04-01"

    # II.4 Production data indexes
    production_start_index = "2023-04-01"
    production_end_index = "2024-10-31"

    # II.5 Covid data indexes
    covid_start_index = "2020-02-01"
    covid_end_index = "2020-06-30"

    # ======= III. Split the data =======
    for asset_df in dfs_list:
        # III.1 Training data
        training_data = asset_df.loc[training_start_index:training_end_index]
        training_data.dropna(axis=0)
        if len(training_data) > 50:
            training_data_dfs_list.append(training_data)

        # III.2 Validation data
        validation_data = asset_df.loc[validation_start_index:validation_end_index]
        validation_data.dropna(axis=0)
        if len(validation_data) > 50:
            validation_data_dfs_list.append(validation_data)

        # III.3 Testing data
        testing_data = asset_df.loc[testing_start_index:testing_end_index]
        testing_data.dropna(axis=0)
        if len(testing_data) > 50:
            testing_data_dfs_list.append(testing_data)

        # III.4 Production data
        production_data = asset_df.loc[production_start_index:production_end_index]
        production_data.dropna(axis=0)
        if len(production_data) > 50:
            production_data_dfs_list.append(production_data)

        # III.5 Covid data
        covid_data = asset_df.loc[covid_start_index:covid_end_index]
        covid_data.dropna(axis=0)
        if len(covid_data) > 50:
            covid_data_dfs_list.append(covid_data)

    return training_data_dfs_list, validation_data_dfs_list, testing_data_dfs_list, production_data_dfs_list, covid_data_dfs_list