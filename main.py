import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


norm = scipy.stats.norm

# derivative of Black Scholes
def vega(S, d1, t):
    v = S * norm.cdf(d1) * (t ** .5)
    return v


# d1 d2 formulas
def d(sigma, S, K, r, t):
    d1 = (1 / (sigma * (t ** .5))) * (np.log(S / K) + ((r + (sigma ** 2) / 2) * t))
    d2 = d1 - (sigma * (t ** .5))
    return d1, d2


# catch number input
def numCatch(ans):
    try:
        float(ans)
        return True
    except ValueError:
        print("PLease Enter numerical value")
        return False


def IV():
    # this here below to prevent squiggles
    strike_price_ans = 0
    stock_price_ans = 0
    sigma_guess_ans = 0
    days_to_exp_ans = 0
    tolerance_ans = 0
    
    # get input
    while True:
        call_price_ans = input(" Call Price: ")
        check = numCatch(call_price_ans)
        if not check:
            continue
        strike_price_ans = input(" Strike Price: ")
        check = numCatch(strike_price_ans)
        if not check:
            continue
        stock_price_ans = input(" Stock Price: ")
        check = numCatch(stock_price_ans)
        if not check:
            continue
        days_to_exp_ans = input(" Days to expiration(less than 365): ")
        check = numCatch(days_to_exp_ans)
        if not check:
            continue
        tolerance_ans = input(' Enter desired tolerance(ex: .001): ')
        check = numCatch(tolerance_ans)
        if not check:
            continue
        elif float(tolerance_ans) >= 1:
            print("Tolerance needs to be less than 1")
            continue
        sigma_guess_ans = input(" Guess Sigma(.5 is a good start): ")
        check = numCatch(sigma_guess_ans)
        if not check:
            continue

        break

    # convert to nums
    call_price = float(call_price_ans)
    strike_price = float(strike_price_ans)
    stock_price = float(stock_price_ans)
    sigma_guess = float(sigma_guess_ans)
    days_to_exp = (float(days_to_exp_ans) / 365)
    tolerance = float(tolerance_ans)

    # current risk free rate
    risk_free_rate = .0147

    # beginning error to measure against tolerance in while loop
    error = 1

    # make list to graph performance with list of error changes
    error_list = []

    # while error is greater than our tolerance
    while error > tolerance:

        # create previous volatility value
        prev_vol = sigma_guess

        # solve our d1 and d2 equations
        d1, d2 = d(sigma_guess, stock_price, strike_price, risk_free_rate, days_to_exp)

        # set our expected call price value which we want to be within our tolerance to zero
        # using black scholes formula
        call_expect = (stock_price * norm.cdf(d1) - (strike_price * np.exp(-risk_free_rate * days_to_exp)) * norm.cdf(d2)) - call_price

        # find our derivative of black scholes to adjust the next sigma guess
        vega_n = stock_price * norm.cdf(d1) * (days_to_exp ** .5)

        # alter next sigma guess by derivative
        sigma_guess = (-call_expect) / vega_n + sigma_guess

        # appending the error list for graphing purposes
        error_list.append(error)

        # adjusting error value based on difference of previous and current sigma guesses
        error = abs((sigma_guess - prev_vol) / prev_vol)

        print(f"Error:{error} Tolerance:{tolerance}")
        print(f"Call Price Estimate: {call_price}")
        print('\n')
    error_count = []
    for i in range(len(error_list)):
        error_count.append(i)


    print("\n")
    print("\n")
    print(f"IV: {sigma_guess}")

    plt.plot(error_list, error_count)
    plt.xlabel('Change in Error')
    plt.ylabel('Iteration progress')
    plt.title('Error performance on Black Scholes Equation using Vega')
    plt.show()


IV()
