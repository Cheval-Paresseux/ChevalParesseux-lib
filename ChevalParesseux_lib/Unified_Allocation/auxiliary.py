import pandas as pd
import numpy as np


# ==================================================================================== #
# ========================== Perspective changing Functions ========================== #
def equivalence_rates(
    input_rate: float, 
    input_frequence: str, 
    desired_compounding_frequence: str
):
    # ======= O. Define the number of compounding periods per year for each frequency =======
    frequencies = {
        "Yearly": 1,
        "Semi-annual": 2,
        "Quarterly": 4,
        "Monthly": 12,
        "Continuous": "Continuous",
    }

    # ======= I. Check if the input and desired compounding frequencies are valid =======
    if input_frequence not in frequencies or desired_compounding_frequence not in frequencies:
        raise ValueError("Invalid compounding frequency. Choose from 'Yearly', 'Semi-annual', 'Quarterly', 'Monthly', or 'Continuous'.")

    # ======= II. Calculate the equivalent rate for the desired compounding frequency =======
    if frequencies[input_frequence] == "Continuous":
        input_effective_rate = np.exp(input_rate) - 1
    else:
        input_effective_rate = (1 + input_rate / frequencies[input_frequence]) ** frequencies[input_frequence] - 1

    if frequencies[desired_compounding_frequence] == "Continuous":
        equivalent_rate = np.log(1 + input_effective_rate)
    else:
        equivalent_rate = frequencies[desired_compounding_frequence] * ((1 + input_effective_rate) ** (1 / frequencies[desired_compounding_frequence]) - 1)

    return equivalent_rate

# ____________________________________________________________________________________ #
def days_count_30_360(start_date: str, end_date: str):
    # ======= I. Convert the dates to datetime objects =======
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # ======= II. Adjust the day component =======
    # The day component of each date is adjusted so that any day greater than 30 is set to 30
    start_day = min(30, start_date.day)
    end_day = min(30, end_date.day)

    # ======= III. Compute the difference in days =======
    year_diff = end_date.year - start_date.year
    month_diff = end_date.month - start_date.month
    day_diff = end_day - start_day

    total_diff = 360 * year_diff + 30 * month_diff + day_diff

    return total_diff

# ____________________________________________________________________________________ #
def quotation_to_price(quotation: str):

    price = float(quotation.split("-")[0]) + float(quotation.split("-")[1]) / 32
    
    return price

# ____________________________________________________________________________________ #