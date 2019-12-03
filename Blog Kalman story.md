# Using Kalman filters to derive predictive factors from limit order book data
 
This post is based on the experience I have got while taking part in a very interesting forecasting competition hosted by [XTX](https://xtxforum.correlation1.com/t/welcome-to-the-xtx-markets-global-forecasting-challenge/7). Participants were challenged by the task to forecast future return of a (presumably) Forex asset based on the limit order book (LOB) data. No details of the asset or limit order book dates were disclosed as part of the competition.

As part of the competition, XTX provided their proprietary model development data. The data included 3 million tick records, with each record containing multiple levels of bid and ask prices. The competition was designed in such a way that the full previous history of accumulated tick data was available at every point in time, but no historical returns can be used for the future forecast. This eliminated classical time series models and required participants to fully utilise predictive patterns of the limit order book.

In this post I will focus on the application of Kalman filter to derive implicit state of the LOB. Detailed description of the theory behind Kalman filters can be found in a broad range of academic resources ([example](http://web.mit.edu/kirtley/kirtley/binlustuff/literature/control/Kalman%20filter.pdf)).

## Model framework
I followed a relatively standard approach to building a regression model:

- XGBoost regression model as a core
- Grid search of parameters based on walk forward cross-validation
- Generation of additional dynamic factors based on historical tick data

I will omit details of XGBoost and cross-validation, all details can be found on [my github](https://github.com/alexbotsula/XTX_Challenge/blob/master/Model%20scripts/2_1_XTX_xgboost.ipynb). Instead, I will focus on the feature engineering technique on Kalman filter. 

## Using Kalman filter to infer implicit flows of LOB
The LOB data comes in a form of arrays of bid/ask size and price, split into 15 buckets: _bidSize[0..14]_, _askSize[0..14]_, _bidPrice[0..14]_, _askPrice[0..14]_. Refer to the plot below for cumulative Bid and Ask volumes in the LOB over a short period of time:

![](https://raw.githubusercontent.com/alexbotsula/XTX_Challenge/master/cumm_bid_ask.png)

To track the dynamics of LOB over time, implied hidden cash flows between the LOB buckets are introduced. To simplify, all buckets _[1..14]_ are merged into a single one. Hence the observations are described as vector _['bidSize1_14', 'bidSize0', 'askSize0', 'askSize1_14']_

The hidden states of the system is defined based on the following eleven parameters:

- Four parameters defining the current volume in each of the buckets _bidSize1_14, bidSize0, askSize0, askSize1_14_; Even though these parameters are directly observed in the data, Kalman filter treat them as being affected by measurement and process error;
- Four parameters defining an external cash flow in/out of the corresponding buckets;
- Three parameters defining the following cash-flows between the buckets:
	1. ask1_14 &lrarr;  ask_0
	2. bid1_14 &lrarr;  bid_0
	3. ask_0 &lrarr;  bid_0

The hidden states assume the existence of an implicit flow between the buckets, in addition to the flow from the external world. As we see further, the estimates of these flows are used as factors in the regression model.

Based on the states above, the $11 \times 11$ Kalman state-transition matrix is defined as follows:

$$\mathbf{A} = \left[\begin{array}
{rrr}
1 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & -1 & 0\\
0 & 1 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 1 & -1\\
0 & 0 & 1 & 0 & 0 & 0 & 1 & 0 & 1 & 0 & 1\\
0 & 0 & 0 & 1 & 0 & 0 & 0 & 1 & -1 & 0 & 0\\
0 & 0 & 0 & 0 & 1 & 0 & & & \dots & & 0\\
\vdots & & & & & \ddots & \\
0 & 0 & & & & & \dots & & & 0 & 1
\end{array}\right]
$$
In the state-transition matrix, rows and columns are ordered in accordance with the list of hidden states above. From row 5 onwards it is an identity matrix, representing that no changes in flows are expected. Rows 1 to 4 represent the transition of current bid/ask volume. For example, transition of $BidSize^{1-14}$ over time $t$ is as follows: $$BidSize^{1-14}_{t+1} = BidSize^{1-14}_{t} + BidFlow^{1-14}_{t} - Flow^{bid1\_14 <-> bid\_0}_{t} + \omega_{t}$$ with $\omega_{t}$ representing white noise process.

The Kalman filter algorithm is implemented in the code below:
```python
process_data = np.array(data[['bidSize1_14', 'bidSize0', 'askSize0', 'askSize1_14']])
N_states = 11                                                                       # number of states
xhat = np.zeros((process_data.shape[0], N_states))                                  # a posteriori estimate of x
P = np.identity(N_states)                                                           # a posteriori error estimate
xhatminus = np.zeros((process_data.shape[0], N_states))                             # a priori estimate of x
Pminus = np.identity(N_states)                                                      # a priori error estimate
K = np.zeros((N_states, process_data.shape[1]))                                     # gain or blending factor
Q = np.identity(N_states) * 1e-3                                                    # estimate of process variance
R = 1                                                                               # estimate of measurement variance

A = np.identity(N_states)
A[0:4, 4:11] = [
    [1, 0, 0, 0, 0, -1, 0],
    [0, 1, 0, 0, 0, 1, -1],
    [0, 0, 1, 0, 1, 0,  1],
    [0, 0, 0, 1, -1, 0, 0]
]

H = np.zeros((process_data.shape[1], N_states))
H[0:4, 0:4] = np.identity(4)

xhat[0, 0:4] = process_data[0, :]

for k in range(1, process_data.shape[0]):
    # time update
    xhatminus[k, :] = A @ xhat[k-1, :]
    Pminus = A @ P @ np.transpose(A) + Q

    # measurement update
    K = Pminus @ np.transpose(H) @ np.linalg.inv(H @ Pminus @ np.transpose(H) + R)
    xhat[k, :] = xhatminus[k, :] + (K @ (process_data[k, :] - H @ xhatminus[k, :]))
    P = (np.identity(N_states) - K @ H) @ Pminus
```
The code returns the values of implicit flows that are further used as predictive factors in the core regression model. As an example, the first 20000 observations of the  flow (ask_0 &lrarr;  bid_0) is represented on the time series below: 
![](https://raw.githubusercontent.com/alexbotsula/XTX_Challenge/master/flow3_20000.png)

The new predictive factors are used in the XGBoost model and their relative importance is estimated based on the number of times they are used in the XGBoost decision trees. 

The extract of the variable importance statistics is represented on the chart below, with the first feature representing the flow ask_0 &lrarr;  bid_0:
![](https://raw.githubusercontent.com/alexbotsula/XTX_Challenge/master/variable_importance.png)

##Conclusion
We observed an example of factor that is generated based on implicit dynamic cash flow of asset's LOB. Even though the flow is implicit and unobserved in the data, the use of Kalman filter algorithm allowed to generate the factor with significant predictive power when used as an input into the core XGBoost algorithm.
 
> Disclaimer: the technique described in this post was used as part of a broader range of predictive factors. In combination, the model submission was ranked 13th in the final competition ranking.
<img src="https://raw.githubusercontent.com/alexbotsula/XTX_Challenge/master/final_rating.jpg" width="200"/>
