



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