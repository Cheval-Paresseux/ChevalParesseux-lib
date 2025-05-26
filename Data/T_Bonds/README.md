# U.S. Treasury Futures – Maturity Reference Guide

This directory contains historical data for U.S. Treasury Futures with four different maturities:

- `ZT=F` – 2-Year Treasury Note Futures  
- `ZF=F` – 5-Year Treasury Note Futures  
- `ZN=F` – 10-Year Treasury Note Futures  
- `ZB=F` – 30-Year Treasury Bond Futures  

These are standardized contracts traded on the Chicago Board of Trade (CBOT) and used primarily to hedge or speculate on U.S. interest rate movements.

---

## 📘 Overview of Each Ticker

### `ZT=F` — 2-Year Treasury Note Futures

- **Maturity**: 2 years  
- **Contract Size**: $200,000 
- **Use Case**: Short-term interest rate hedging/speculation  
- **Interest Rate Sensitivity**: Low  
- **Typical Users**: Money market participants, short-duration bond traders

---

### `ZF=F` — 5-Year Treasury Note Futures

- **Maturity**: 5 years  
- **Contract Size**: $100,000  
- **Use Case**: Intermediate-term rate exposure  
- **Interest Rate Sensitivity**: Moderate  
- **Typical Users**: Asset managers adjusting portfolio duration, macro traders

---

### `ZN=F` — 10-Year Treasury Note Futures

- **Maturity**: 10 years  
- **Contract Size**: $100,000  
- **Use Case**: Benchmark for medium/long-term rates  
- **Interest Rate Sensitivity**: High  
- **Typical Users**: Hedge funds, fixed-income portfolio managers, macro investors

---

### `ZB=F` — 30-Year Treasury Bond Futures

- **Maturity**: 30 years (technically, bonds with >15 years to maturity)  
- **Contract Size**: $100,000  
- **Use Case**: Long-duration risk management  
- **Interest Rate Sensitivity**: Very High  
- **Typical Users**: Pension funds, insurers, duration traders

---

## 📈 Data Notes

- **Frequency**: Daily (market trading days)  
- **Source**: Yahoo Finance  
- **File Format**: CSV

Each ticker represents the **front-month continuous futures contract**, meaning it rolls over to the next available contract as expiration approaches.

---

## 🧠 Interpreting the Prices

- The futures price is quoted in **points** and **32nds of a point** (e.g., `124'16` = 124.5).
- Futures prices **move inversely** with interest rates:
  - If market expects **rates to fall** → prices **rise**
  - If market expects **rates to rise** → prices **fall**

---

## 🔗 Additional Resources

- [CME Group – U.S. Treasury Futures](https://www.cmegroup.com/markets/interest-rates.html)
- [FRED – Yield Curve Data](https://fred.stlouisfed.org/)
- [Treasury.gov – Daily Treasury Yield Curve Rates](https://home.treasury.gov/)
