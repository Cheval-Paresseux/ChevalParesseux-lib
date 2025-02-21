import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


def equivalence_rates(
    input_rate: float, input_frequence: str, desired_compounding_frequence: str
):
    """
    This function computes the equivalent rate for a given compounding frequency.

    Args:
        input_rate (float): The annual interest rate for which we want an equivalent rate.
        input_frequence (str): The compounding frequency of the input rate, expressed in terms such as
                            'Yearly', 'Semi-annual', 'Quarterly', 'Monthly', or 'Continuous'.
        desired_compounding_frequence (str): The desired compounding frequency to calculate the equivalent rate, expressed in terms such as
                            'Yearly', 'Semi-annual', 'Quarterly', 'Monthly', or 'Continuous'.


    Returns:
        equivalent_rate (float): The equivalent rate for the desired compounding frequency.
    """

    # ======= O. Define the number of compounding periods per year for each frequency =======
    frequencies = {
        "Yearly": 1,
        "Semi-annual": 2,
        "Quarterly": 4,
        "Monthly": 12,
        "Continuous": "Continuous",
    }

    # ======= I. Check if the input and desired compounding frequencies are valid =======
    if (
        input_frequence not in frequencies
        or desired_compounding_frequence not in frequencies
    ):
        raise ValueError(
            "Invalid compounding frequency. Choose from 'Yearly', 'Semi-annual', 'Quarterly', 'Monthly', or 'Continuous'."
        )

    # ======= II. Calculate the equivalent rate for the desired compounding frequency =======
    if frequencies[input_frequence] == "Continuous":
        input_effective_rate = np.exp(input_rate) - 1
    else:
        input_effective_rate = (
            1 + input_rate / frequencies[input_frequence]
        ) ** frequencies[input_frequence] - 1

    if frequencies[desired_compounding_frequence] == "Continuous":
        equivalent_rate = np.log(1 + input_effective_rate)
    else:
        equivalent_rate = frequencies[desired_compounding_frequence] * (
            (1 + input_effective_rate)
            ** (1 / frequencies[desired_compounding_frequence])
            - 1
        )

    return equivalent_rate


def stock_equivalent_forward_price(
    spot_price: float,
    risk_free_rate: float,
    maturity: float,
    dividend_rate: float = 0,
    storing_cost: float = 0,
):
    """
    This function computes the forward price of an asset.

    Args:
        spot_price (float): The spot price of the underlying asset.
        risk_free_rate (float): The annual risk-free rate for the given maturity.
        maturity (float): The time to maturity, expressed in years.

    Returns:
        forward_price (float): The calculated forward price.
    """
    forward_price = (spot_price + storing_cost) * np.exp(
        (risk_free_rate - dividend_rate) * maturity
    )

    return forward_price


def currency_forward_price(
    spot_price_AB: float,
    rf_A: float,
    rf_B: float,
    maturity: float,
):
    """
    This function computes the forward price of a currency.

    Args:
        spot_price_AB (float): The spot price of currency A in terms of currency B.
        rf_A (float): The annual risk-free rate for currency A.
        rf_B (float): The annual risk-free rate for currency B.
        maturity (float): The time to maturity, expressed in years.

    Returns:
        forward_price (float): The calculated forward price in the same unit as spot_price_AB.
    """
    forward_price = spot_price_AB * np.exp((rf_A - rf_B) * maturity)

    return forward_price


def futures_hedging(
    position_to_hedge_size: float,
    futures_contracts_size: float,
    correlation: float,
    subjacent_standard_deviation: float,
    futures_standard_deviation: float,
):
    """
    This function computes the hedge ratio and the number of futures contracts needed to hedge a position.

    Args:
        position_to_hedge_size (float): The size of the position to hedge (e.g., number of units of the asset).
        futures_contracts_size (float): The size of a single futures contract (e.g., number of units per contract).
        correlation (float): The correlation between the asset to hedge and the futures contract.
        subjacent_standard_deviation (float): The standard deviation of the underlying asset's returns.
        futures_standard_deviation (float): The standard deviation of the futures contract's returns.

    Returns:
        hedge_ratio (float): The optimal hedge ratio (also known as the hedge effectiveness).
        number_of_contracts_needed (float): The number of futures contracts needed to hedge the position.
    """
    # ======= I. Calculate the hedge ratio =======
    hedge_ratio = (
        correlation * subjacent_standard_deviation / futures_standard_deviation
    )

    # ======= II. Calculate the number of futures contracts needed =======
    number_of_contracts_needed = (
        hedge_ratio * position_to_hedge_size / futures_contracts_size
    )

    return hedge_ratio, number_of_contracts_needed


def days_count_30_360(start_date: str, end_date: str):
    """
    Compute the number of days between two dates using 30/360 day count convention.

    Args:
        start_date (str): The start date.
        end_date (str): The end date.

    Returns:
        total_diff (int): The number of days between the two dates according
    """
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


def quotation_to_price(quotation: str):
    """
    Convert an american quotation to a price.

    Args:
        quotation (str): The quotation, it should be input as a string with the following format: "100-16".

    Returns:
        price (float): The price.
    """
    price = float(quotation.split("-")[0]) + float(quotation.split("-")[1]) / 32
    return price


class Bond:
    """
    This class provides tools to price a bond, and compute its sensitivity to different parameters.
    It is fitted to "standard" bonds with fixed coupon rates, coupon frequencies, and face values.
    """

    def __init__(
        self,
        face_value: float,
        coupon_rate: float,
        maturity: float,
        coupon_frequency: str,
        risk_free_rate: float,
        yield_to_maturity: float = None,
        market_price: float = None,
    ):
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.maturity = maturity
        self.coupon_frequency = coupon_frequency
        self.rf = risk_free_rate
        self.yield_to_maturity = yield_to_maturity
        self.market_price = market_price

        # Deciding which rate we take into account
        if self.yield_to_maturity:
            self.discount_rate = self.yield_to_maturity
        else:
            self.discount_rate = self.rf

        # Changing type of frequency
        possible_frequencies = {
            "Monthly": 1 / 12,
            "Bi-monthly": 1 / 6,
            "Trimestrial": 1 / 4,
            "Semestral": 1 / 2,
            "Annual": 1,
            "2years": 2,
        }
        self.frequency = possible_frequencies[self.coupon_frequency]

    def coupon_value(self):
        """
        This function aims to get information about the coupons.

        Returns:
            number_of_coupons (int) -> the number of coupons
            coupon (float) -> the amount paid by a coupon
            coupons_pv (float) -> the present value of coupons
        """

        # ====== I. Calculate the number of coupons and the coupon value ======
        number_of_coupons = int(self.maturity / self.frequency)
        coupon = self.face_value * self.coupon_rate * self.frequency

        # ====== II. Calculate the present value of the coupons ======
        coupons_pv = sum(
            [
                coupon * np.exp(-self.discount_rate * t * self.frequency)
                for t in range(1, number_of_coupons + 1)
            ]
        )

        return number_of_coupons, coupon, coupons_pv

    def market_ytm(self):
        """
        Compute the market yield_to_maturity.

        Returns:
            yield_to_maturity (float) -> the market yield to maturity
        """

        # ====== 0. Check if a market price is given ======
        if self.market_price:
            # ====== I. Define the bond price function ======
            number_of_coupons, coupon, coupons_pv = self.coupon_value()

            def bond_pricing(r):
                coupons = sum(
                    [
                        coupon * np.exp(-r * t * self.frequency)
                        for t in range(1, number_of_coupons + 1)
                    ]
                )  # coupons present value
                face_value_discounted = self.face_value * np.exp(
                    -r * self.maturity
                )  # face_value present value
                return (
                    coupons + face_value_discounted - self.market_price
                )  # price = coupons + face_value discounted => we solve for =0

            # ====== II. Solve for the yield to maturity ======
            initial_guess = 0.05  # wanna take a bet ?
            ytm = fsolve(bond_pricing, initial_guess)
            yield_to_maturity = ytm[0]
        else:
            yield_to_maturity = None
            print("You did not give a market price")

        return yield_to_maturity

    def use_market_yield(self):
        """
        Let the user change the yield used to market rate.
        """

        if self.market_price:
            self.discount_rate = self.market_ytm()
        else:
            print("You did not give a market price")

    def bond_price(self):
        """
        Gets the bond_price.

        Returns:
            bond_price (float) -> the discounted price of the bond
        """

        _, _, coupons_pv = self.coupon_value()
        face_value_pv = self.face_value * np.exp(-self.discount_rate * self.maturity)

        bond_price = coupons_pv + face_value_pv

        return bond_price

    def duration(self):
        """
        Compute the duration of the bond.

        Returns:
            duration (float) -> duration of the bond
        """

        # ====== I. Calculate the number of coupons and the coupon value ======
        number_of_coupons, coupon, _ = self.coupon_value()

        # ====== II. Calculate the duration of the bond ======
        duration = sum(
            [
                t
                * self.frequency
                * coupon
                * np.exp(-self.discount_rate * t * self.frequency)
                for t in range(1, number_of_coupons + 1)
            ]
        ) + self.maturity * self.face_value * np.exp(
            -self.discount_rate * self.maturity
        )

        # ====== III. Normalize the duration ======
        if self.market_price:
            duration = duration / self.market_price
        else:
            duration = duration / self.bond_price()

        return duration

    def convexity(self):
        """
        Compute the duration of the bond.

        Returns:
            duration (float) -> duration of the bond
        """

        # ====== I. Calculate the number of coupons and the coupon value ======
        number_of_coupons, coupon, _ = self.coupon_value()

        # ====== II. Calculate the convexity of the bond ======
        convexity = sum(
            [
                (t**2)
                * self.frequency
                * coupon
                * np.exp(-self.discount_rate * t * self.frequency)
                for t in range(1, number_of_coupons + 1)
            ]
        ) + (self.maturity**2) * self.face_value * np.exp(
            -self.discount_rate * self.maturity
        )

        # ====== III. Normalize the convexity ======
        if self.market_price:
            convexity = convexity / self.market_price
        else:
            convexity = convexity / self.bond_price()

        return convexity

    def yield_change(self, delta_yield: float, method: str):
        """
        Compute the change in price of the bond according to a change in yield.

        Args:
            delta_yield (float) -> change in the yield.
            method (str) -> "duration" for small changes / "convexity" for larger changes.

        Returns:
            change_return (float) -> change of the bond price expressed in a return
            bond_change (float) -> change of the bond price expressed in value
            new_bond_price (float) -> price of the bond after the change in yield
        """

        if method == "duration":
            bond_change = -self.bond_price() * self.duration() * delta_yield
            change_return = bond_change / self.bond_price()
        elif method == "convexity":
            bond_change = (
                -self.bond_price() * self.duration() * delta_yield
                + self.bond_price() * (1 / 2) * self.convexity() * delta_yield**2
            )
            change_return = bond_change / self.bond_price()
        else:
            print("Invalid method : it should be -duration- or -convexity-")

        new_bond_price = self.bond_price() + bond_change

        return change_return, bond_change, new_bond_price

    def exposition(self):
        """
        Used to visualize the exposition of the bond.
        """
        # ====== I. Create lists to store the changes in return ======
        convex_list = []
        duration_list = []

        # ====== II. Create a range of yield changes ======
        yield_changes = np.arange(-0.1, 0.1, 0.00001)

        # ====== III. Compute the changes in return for both methods ======
        for i in yield_changes:
            convex_change, bond_change, new_bond_price = self.yield_change(
                delta_yield=i, method="convexity"
            )
            duration_change, bond_change, new_bond_price = self.yield_change(
                delta_yield=i, method="duration"
            )
            convex_list.append(convex_change)
            duration_list.append(duration_change)

        # ====== IV. Plot the exposition to a change in yield ======
        plt.figure(figsize=(10, 6))
        plt.plot(yield_changes, convex_list, label="Convexity Change", color="blue")
        plt.plot(yield_changes, duration_list, label="Duration Change", color="red")
        plt.title("Exposition to a Change in Yield")
        plt.xlabel("Yield Change")
        plt.ylabel("Return Change")
        plt.legend()
        plt.grid(True)
        plt.show()


class ABS:
    def __init__(self):
        self.loan_pool = None
        self.defaulted_pool = None
        self.recovered_pool = None

    def simulate_loan_pool(
        self,
        num_loans: int = 100,
        principal_lower_bound: float = 10000,
        principal_upper_bound: float = 50000,
        default_prob_lower_bound: float = 0.01,
        default_prob_upper_bound: float = 0.1,
    ):
        """
        Generates a random loan pool.

        Args:
            num_loans (int): Number of loans to generate.

        Returns:
            loan_pool (pd.DataFrame): A DataFrame containing the loan pool.
        """

        loan_pool = pd.DataFrame(
            {
                "loan_id": np.arange(1, num_loans + 1),
                "principal": np.random.uniform(
                    principal_lower_bound, principal_upper_bound, num_loans
                ),
                "default_prob": np.random.uniform(
                    default_prob_lower_bound, default_prob_upper_bound, num_loans
                ),
            }
        )

        self.loan_pool = loan_pool

        return loan_pool

    def simulate_defaults(self):
        """
        Simulate loan defaults based on default probability.

        Returns:
            defaulted_pool (pd.DataFrame): DataFrame with 'defaulted' column indicating if a loan has defaulted.
        """
        if self.loan_pool is None:
            print(
                "No loan pool found. You need to simulate or provide a loan pool first."
            )
            defaulted_pool = None
        else:
            defaulted_pool = self.loan_pool.copy()
            for index, row in self.loan_pool.iterrows():
                default = np.random.rand() < row["default_prob"]
                defaulted_pool.at[index, "defaulted"] = default

            self.defaulted_pool = defaulted_pool

        return defaulted_pool

    def simulate_recovery(self, recovery_rate: float = 0.5):
        """
        Simulate loan recovery based on recovery rate.
        Args:
            recovery_rate (float): Recovery rate for defaulted loans.

        Returns:
            recovered_pool (pd.DataFrame): DataFrame with 'recovered_amount' column indicating the recovered amount.
        """
        if self.defaulted_pool is None:
            print(
                "No defaulted pool found. You need to simulate defaults or provide a defaulted pool first."
            )
            recovered_pool = None
        else:
            recovered_pool = self.defaulted_pool.copy()
            recovered_pool["recovered_amount"] = np.where(
                recovered_pool["defaulted"],
                recovered_pool["principal"] * recovery_rate,
                recovered_pool["principal"],
            )

        self.recovered_pool = recovered_pool

        return recovered_pool

    def create_abs(
        self,
        senior_tranche: float = 0.8,
        mezzanine_tranche: float = 0.15,
        equity_tranche: float = 0.05,
    ):
        """
        Simulate the Asset-Backed Security (ABS) cash flows with a waterfall structure.

        Args:
            loan_pool (pd.DataFrame): DataFrame containing loan information.
            num_loans (int): Number of loans in the ABS.
            recovery_rate (float): Recovery rate for defaulted loans.

        Returns:
            abs_cash_flows (pd.DataFrame): DataFrame with ABS tranche cash flows under a waterfall structure.
        """
        # ------- 1. Check if we have the necessary data -------
        if self.recovered_pool is None:
            print(
                "No recovered pool found. You need to simulate recovery or provide a recovered pool first."
            )
            return None

        recovered_pool = self.recovered_pool

        # ------- 2. Define ABS Structure -------
        expected_cash_flows = recovered_pool["principal"].sum()
        total_cash_flow = recovered_pool["recovered_amount"].sum()

        tranches = {
            "Senior": senior_tranche,  # Senior tranche gets 80% of expected cash flow
            "Mezzanine": mezzanine_tranche,  # Mezzanine tranche gets 15%
            "Equity": equity_tranche,  # Equity tranche gets 5%
        }

        tranche_targets = {
            "Senior": expected_cash_flows * tranches["Senior"],
            "Mezzanine": expected_cash_flows * tranches["Mezzanine"],
            "Equity": expected_cash_flows * tranches["Equity"],
        }

        # ------- 3. Allocate Cash Flows -------
        tranche_cash_flows = {}
        remaining_cash_flow = total_cash_flow

        for tranche in ["Senior", "Mezzanine", "Equity"]:
            tranche_target = tranche_targets[tranche]

            if remaining_cash_flow >= tranche_target:
                tranche_cash_flows[tranche] = tranche_target
                remaining_cash_flow -= tranche_target
            else:
                tranche_cash_flows[tranche] = remaining_cash_flow
                remaining_cash_flow = 0

        # ------- 4. Create ABS Payoff DataFrame -------
        tranche_structure = pd.DataFrame(
            {
                "Tranche": ["Senior", "Mezzanine", "Equity"],
                "Target Payment ($)": [
                    tranche_targets["Senior"],
                    tranche_targets["Mezzanine"],
                    tranche_targets["Equity"],
                ],
                "Allocated Cash ($)": [
                    tranche_cash_flows["Senior"],
                    tranche_cash_flows["Mezzanine"],
                    tranche_cash_flows["Equity"],
                ],
            }
        )

        return tranche_structure


def create_cdo(
    abs_list: list = None,
    cdo_senior_tranche: float = 0.65,
    cdo_mezzanine_tranche: float = 0.25,
    cdo_equity_tranche: float = 0.1,
    num_abs: int = 5,
    num_loans: int = 100,
    principal_lower_bound: float = 10000,
    principal_upper_bound: float = 50000,
    default_prob_lower_bound: float = 0.01,
    default_prob_upper_bound: float = 0.1,
    recovery_rate: float = 0.5,
    abs_senior_tranche: float = 0.8,
    abs_mezzanine_tranche: float = 0.15,
    abs_equity_tranche: float = 0.05,
):
    """
    Simulate an ABS CDO by pooling the Mezzanine tranches from multiple ABS structures and applying a waterfall structure.


    """
    # ======= 0. Initialize total cash flows for the ABS Mezzanine tranche =======
    total_mezzanine_target_cash_flow = 0
    total_mezzanine_cash_flow = 0

    # ======= 1. Simulate each ABS structure =======
    if abs_list is None:
        abs_list = []
        for i in range(num_abs):
            abs_generator = ABS()
            _ = abs_generator.simulate_loan_pool(
                num_loans=num_loans,
                principal_lower_bound=principal_lower_bound,
                principal_upper_bound=principal_upper_bound,
                default_prob_lower_bound=default_prob_lower_bound,
                default_prob_upper_bound=default_prob_upper_bound,
            )
            _ = abs_generator.simulate_defaults()
            _ = abs_generator.simulate_recovery(recovery_rate=recovery_rate)
            abs_tranche_structure = abs_generator.create_abs(
                senior_tranche=abs_senior_tranche,
                mezzanine_tranche=abs_mezzanine_tranche,
                equity_tranche=abs_equity_tranche,
            )
            
            abs_list.append(abs_tranche_structure)

    for abs_tranche_structure in abs_list:
        total_mezzanine_target_cash_flow += abs_tranche_structure.loc[
            abs_tranche_structure["Tranche"] == "Mezzanine", "Target Payment ($)"
        ].values[0]
        total_mezzanine_cash_flow += abs_tranche_structure.loc[
            abs_tranche_structure["Tranche"] == "Mezzanine", "Allocated Cash ($)"
        ].values[0]

    # ======= 2. Define the CDO tranches structure =======
    cdo_tranches = {
        "Senior": cdo_senior_tranche,  # CDO Senior tranche gets 65% of the Mezzanine cash flow
        "Mezzanine": cdo_mezzanine_tranche,  # CDO Mezzanine tranche gets 25%
        "Equity": cdo_equity_tranche,  # CDO Equity tranche gets 10%
    }

    # Define target payments for each CDO tranche based on Mezzanine cash flows
    cdo_tranche_targets = {
        "Senior": total_mezzanine_target_cash_flow * cdo_tranches["Senior"],
        "Mezzanine": total_mezzanine_target_cash_flow * cdo_tranches["Mezzanine"],
        "Equity": total_mezzanine_target_cash_flow * cdo_tranches["Equity"],
    }

    # ======= 3. Apply waterfall structure to allocate cash flows =======
    remaining_cdo_cash_flow = total_mezzanine_cash_flow
    cdo_tranche_cash_flows = {}

    for tranche in ["Senior", "Mezzanine", "Equity"]:
        tranche_target = cdo_tranche_targets[tranche]

        if remaining_cdo_cash_flow >= tranche_target:
            cdo_tranche_cash_flows[tranche] = tranche_target
            remaining_cdo_cash_flow -= tranche_target
        else:
            cdo_tranche_cash_flows[tranche] = remaining_cdo_cash_flow
            remaining_cdo_cash_flow = 0

    # ======= 4. Create the CDO tranche structure DataFrame =======
    cdo_tranche_structure = pd.DataFrame(
        {
            "Tranche": ["Senior", "Mezzanine", "Equity"],
            "Target Payment ($)": [
                cdo_tranche_targets["Senior"],
                cdo_tranche_targets["Mezzanine"],
                cdo_tranche_targets["Equity"],
            ],
            "Allocated Cash ($)": [
                cdo_tranche_cash_flows["Senior"],
                cdo_tranche_cash_flows["Mezzanine"],
                cdo_tranche_cash_flows["Equity"],
            ],
        }
    )

    return cdo_tranche_structure


############################################################################################################
############################################################################################################
############################################################################################################

"""
Now we will go through some examples on how to use those functions. 
"""
# ======= I. equivalence_rates =======
input_rate = 0.05
input_frequence = "Yearly"
desired_compounding_frequence = "Quarterly"

equiv_rate = equivalence_rates(
    input_rate, input_frequence, desired_compounding_frequence
)
print(f"Equivalent rate for quarterly compounding: {equiv_rate:.4f}")


# ======= II. stock_equivalent_forward_price =======
spot_price = 100  # Spot price of the asset
risk_free_rate = 0.03  # 3% annual risk-free rate
maturity = 1  # 1 year

forward_price = stock_equivalent_forward_price(spot_price, risk_free_rate, maturity)
print(f"Forward price: {forward_price:.2f}")

# ======= III. currency_forward_price =======
spot_price_AB = 1.2  # Spot price of currency A in terms of currency B
rf_A = 0.02  # 2% annual risk-free rate for currency A
rf_B = 0.03  # 3% annual risk-free rate for currency B
maturity = 1  # 1 year

forward_price = currency_forward_price(spot_price_AB, rf_A, rf_B, maturity)
print(f"Forward price: {forward_price:.4f}")

# ======= IV. futures_hedging =======
position_to_hedge_size = 1000  # Size of the position to hedge
futures_contracts_size = 100  # Size of a single futures contract
correlation = 0.8  # Correlation between the asset and futures contract
subjacent_standard_deviation = 0.2  # Standard deviation of the asset's returns
futures_standard_deviation = (
    0.15  # Standard deviation of the futures contract's returns
)

hedge_ratio, number_of_contracts_needed = futures_hedging(
    position_to_hedge_size,
    futures_contracts_size,
    correlation,
    subjacent_standard_deviation,
    futures_standard_deviation,
)
print(f"Hedge ratio: {hedge_ratio:.2f}")
print(f"Number of futures contracts needed: {number_of_contracts_needed:.2f}")

# ======= V. days_count_30_360 =======
start_date = "2021-01-15"
end_date = "2021-03-15"

total_diff = days_count_30_360(start_date, end_date)
print(f"Number of days between the two dates: {total_diff}")

# ======= VI. quotation_to_price =======
quotation = "100-16"
price = quotation_to_price(quotation)
print(f"Price: {price:.2f}")

# ======= VII. class Bond =======
bond = Bond(
    face_value=1000,
    coupon_rate=0.06,  # 6% coupon rate
    maturity=5,  # 5 years to maturity
    coupon_frequency="Annual",
    risk_free_rate=0.04,  # 4% risk-free rate
    market_price=980,  # Market price of the bond
)

number_of_coupons, coupon, coupons_pv = bond.coupon_value()
print(f"Number of coupons: {number_of_coupons}")
print(f"Coupon value: {coupon:.2f}")
print(f"Present value of coupons: {coupons_pv:.2f}")

ytm = bond.market_ytm()
print(f"Yield to Maturity: {ytm:.4f}")

bond.use_market_yield()
print(f"Discount rate after using market yield: {bond.discount_rate:.4f}")

bond_price = bond.bond_price()
print(f"Bond price: {bond_price:.2f}")

duration = bond.duration()
print(f"Duration: {duration:.2f}")

convexity = bond.convexity()
print(f"Convexity: {convexity:.2f}")

delta_yield = 0.01  # 1% change in yield
change_return, bond_change, new_bond_price = bond.yield_change(
    delta_yield, method="convexity"
)
print(f"Change in bond price: {bond_change:.2f}")
print(f"Change in return: {change_return:.2f}")
print(f"New bond price: {new_bond_price:.2f}")

bond.exposition()

# ======= VIII. class ABS =======
abs_generator = ABS()
loan_pool = abs_generator.simulate_loan_pool()
print(loan_pool.head())

defaulted_pool = abs_generator.simulate_defaults()
print(defaulted_pool.head())

recovered_pool = abs_generator.simulate_recovery()
print(recovered_pool.head())

abs_tranche_structure = abs_generator.create_abs()
print(abs_tranche_structure)

# ======= IX. create_cdo =======
default_prob_lower_bound: float = 0.01
default_prob_upper_bound: float = 0.3 # Change this value to see the impact on the CDO structure

cdo_tranche_structure = create_cdo(
    default_prob_lower_bound=default_prob_lower_bound, 
    default_prob_upper_bound=default_prob_upper_bound
    )
print(cdo_tranche_structure)
