



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