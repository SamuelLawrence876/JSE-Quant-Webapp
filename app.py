import datetime
import pandas as pd
from urllib.error import HTTPError
import streamlit as st
from time import sleep
import ffn
st.set_option('deprecation.showPyplotGlobalUse', False)

# Idea tab:
# LSTM preditction
# RNN Prediction
# Impliment other quant libraries
# Link Articles and possibly summarise them as recent news
# Moving Averages

# Custom Back ground
# st.markdown('<style>body{background-color: #93AFC0;}</style>', unsafe_allow_html=True)
# st.markdown(
#     """
# <style>
# .sidebar .sidebar-content {
#     background-image: linear-gradient(#7B9AB8,#7B9AB8);
#     color: white;
# }
# </style>
# """,
#     unsafe_allow_html=True,
# )
#

def app():
    selection = ["Stock Market Analysis", "Portfolio Assessment"]  # Selections
    choice = st.sidebar.selectbox("Dashboard Selection", selection)
    if choice == 'Stock Market Analysis':

        # Building both our selectbox and URL from the JSE's website
        st.title('Jamaica Stock Exchange Quantitative DashBoard 📈')
        st.write('**Contact the author: Samuel Lawrence - ** http://www.samuel-lawrence.co.uk')
        st.write("More About the Jamaica Stock Exchange: https://www.jamstockex.com/ ")
        Index = 'https://www.jamstockex.com/market-data/download-data/price-history/'
        df_index = pd.read_html(Index)
        df_index = df_index[0]
        Stock_Select = df_index['Instrument']

        # Creating Select Box
        Stock_Selection = st.multiselect('Select multiple stocks', Stock_Select)
        try:
            # Date selection for Analysis
            date = st.date_input("Analysis Dates", (datetime.date(2019, 1, 3), datetime.date(2020, 1, 10)))

            # We need these datse to pass into our URl
            since_date = str(date[0])
            until_date = str(date[1])

            # We need dates in datetime format for our calculations
            # until_date_calender  = date[1]
            # since_date_calender = date[0]

        except (IndexError, NameError, UnboundLocalError):
            pass

        for x in range(0, 10):  # Multiple attempts because the JSE website gives us HTTP errors randomly
            try:
                try:
                    # Extracting Data from the JSE based on stock chosen
                    JSE_URL = 'https://www.jamstockex.com/market-data/download-data/price-history/{}/' + since_date + '/' + until_date

                    stock_dfs = []
                    for stock in Stock_Selection:
                        dfs = pd.read_html(JSE_URL.format(stock))
                        if not dfs:
                            st.write(f'No tables found for {stock}')
                            continue
                        stock_dfs.append(dfs[0])

                    full_df = pd.concat(stock_dfs)
                    display_columns = [
                        "Date",
                        "Instrument",
                        # "Volume  (non block) ($)",
                        "Close  Price ($)"
                    ]

                    # Creating Dataframe for stocks
                    sub_df = full_df[display_columns]
                    sub_pivot = sub_df.pivot(index='Date', columns='Instrument')
                    # st.write(sub_pivot)

                    data = pd.DataFrame(
                        sub_pivot.to_records())  # Turning our pivot table back into a dataframe

                    # Cleaning Data for readability
                    data.columns = [
                        hdr.replace("('Close  Price ($)',", "").replace(")", "").replace("'", "").replace("'",
                                                                                                          "").replace(
                            " ", "") \
                        for hdr in data.columns]  # Turning columns back into ticker names

                    # Readable format for FFN
                    data['Date'] = pd.to_datetime(data.Date, infer_datetime_format=True)
                    data = data.set_index('Date')

                    # Prepping for analysis
                    GS = ffn.GroupStats(data)
                    GS.set_riskfree_rate(.03)
                    perf = data.calc_stats()
                    returns = data.to_log_returns().dropna()

                    st.subheader(" Prices History: ")
                    st.write(data.tail())

                    st.subheader("Graph of stocks: ")
                    st.line_chart(data)

                    # Breaking up calculations for easier analysis
                    Analyzer_choice = st.selectbox("Analysis Type",
                                                   ["Ratios",
                                                    "Returns", "Look Back Returns", "Portfolio Weights",
                                                    "Change Analysis",
                                                    "Machine Learning - Clustering", "Correlation", "Stocks Summary"])

                    # Analysis Choices start here
                    if Analyzer_choice == "Correlation":
                        st.subheader("Correlation of returns:")
                        st.write("**More about this analysis:** https://en.wikipedia.org/wiki/Heat_map ")
                        st.pyplot(ffn.plot_corr_heatmap(returns))

                    if Analyzer_choice == "Stocks Summary":
                        st.subheader('Stocks Summary:')
                        General_stats = perf.stats
                        st.write(General_stats)

                    if Analyzer_choice == "Ratios":
                        st.subheader(
                            'Calmar Ratio')
                        st.write("**More about this ratio:** https://www.investopedia.com/terms/c/calmarratio.asp")
                        st.write(ffn.calc_calmar_ratio(data))

                        st.subheader("Risk / Return Ratio ")
                        st.write(ffn.calc_risk_return_ratio(data))
                        st.write("**More about this ratio:** https://www.investopedia.com/terms/r/riskrewardratio.asp"
                                 "#:~:text=The%20risk%2Freward%20ratio%20marks,"
                                 "undertake%20to%20earn%20these%20returns.")

                        st.subheader("Sortino ratio")  # Experimental
                        # Number of periods
                        # num_years = (until_date_calender.year - since_date_calender.year)

                        st.write(ffn.calc_sortino_ratio(returns, rf=0.0, annualize=True))
                        st.write("**More about this ratio:** https://www.investopedia.com/terms/s/sortinoratio.asp ")

                        st.subheader("Sharpe Ratio")  # Experimental
                        st.write(ffn.calc_sharpe(data, rf=0.0, annualize=True))
                        st.write("**More about this ratio:** https://www.investopedia.com/terms/s/sharperatio.asp ")

                        st.subheader('Max Drawdown')
                        st.write("**More about this ratio:** https://www.investopedia.com/terms/m/maximum-drawdown-mdd"
                                 ".asp#:~:text=A%20maximum%20drawdown%20(MDD)%20is,"
                                 "over%20a%20specified%20time%20period. ")
                        st.write(ffn.calc_max_drawdown(data))

                    if Analyzer_choice == "Portfolio Weights":
                        st.subheader("ERC Risk Parity Portfolio Weights")
                        st.write(ffn.calc_erc_weights(returns=returns).as_format('.2%'))
                        st.write('**About this ratio:** A calculation of the equal risk contribution / risk parity '
                                 'weights given the portfolio returns.')

                        st.subheader("Inverse Volatility Weights")
                        st.write(ffn.calc_inv_vol_weights(returns).as_format('.2%'))
                        st.write(
                            "**About this ratio:** A calculation of weights proportional to the inverse volatility of "
                            "each stock ")

                        st.subheader("Mean Variance Weights")
                        st.write(returns.calc_mean_var_weights().as_format('.2%'))
                        st.write("**About this ratio:** optimal portolio based on classic Markowitz Mean/Variance "
                                 "Optimisation methods")
                        st.write("**More information at:** https://www.investopedia.com/terms/m/meanvariance-analysis"
                                 ".asp")

                    if Analyzer_choice == "Returns":
                        st.subheader('Total returns over the period')
                        st.write(ffn.calc_total_return(data).as_format('.2%'))

                        st.subheader('CAGR - compound annual growth rate.')
                        st.write("**More about this ratio:** https://www.investopedia.com/terms/c/cagr.asp")
                        st.write(ffn.calc_cagr(data).as_format('.2%'))

                        st.subheader("Distribution of returns")
                        st.pyplot(ax=returns.hist(figsize=(20, 10), bins=30), clear_figure=True)

                    if Analyzer_choice == "Look Back Returns":
                        st.subheader('Look Back Returns over the period:')
                        st.write(GS.display_lookback_returns())

                    if Analyzer_choice == "Machine Learning - Clustering":  ##
                        st.subheader("Threshold Clustering Algorithm (FTCA)")
                        st.write("Grouping Stocks based on similar characteristics")
                        thresh = returns.calc_ftca(threshold=0.1)
                        thresh_df = pd.DataFrame.from_dict(thresh, orient='index')

                        st.write(thresh_df)
                        st.write(
                            "**More info about this :** http://cssanalytics.wordpress.com/2013/11/26/fast-threshold"
                            "-clustering-algorithm-ftca/")

                    # if Analyzer_choice == "Change Analysis":
                            # Counting the amout of changes in dataframe
                    #     def Change(Data):
                    #         if x > -0.5 and x <= 0.5:
                    #             return 'Slight or No change'
                    #         elif x > 0.5 and x <= 1:
                    #             return 'Slight Positive'
                    #         elif x > -1 and x <= -0.5:
                    #             return 'Slight Negative'
                    #         elif x > 1 and x <= 3:
                    #             return 'Positive'
                    #         elif x > -3 and x <= -1:
                    #             return 'Negative'
                    #         elif x > 3 and x <= 7:
                    #             return 'Among top gainers'
                    #         elif x > -7 and x <= -3:
                    #             return 'Among top losers'
                    #         elif x > 7:
                    #             return 'Bull run'
                    #         elif x <= -7:
                    #             return
                    #
                    #     test = data
                    #     for stock in test.columns:
                    # #         test[stock] = test[stock].apply(Change)
                    #
                    #     for Categorical_Values in test.columns:
                    #         cat_num = test[Categorical_Values].value_counts().plot(kind='pie', figsize=(10, 5))
                    #         st.write(cat_num)

                        #st.write(test)




                    # if Analyzer_choice == "Volume Analysis":
                    #    st.subheader("Coming Soon")
                    # https://www.investopedia.com/terms/v/volume-analysis.asp#:~:text=What%20is%20Volume%20Analysis,
                    # in%20a%20given%20time%20period.&text=By%20analyzing%20trends%20in%20volume,
                    # changes%20in%20a%20security's%20price.

                except HTTPError:
                    sleep(1)
                else:

                    break
                # st.error('Try again later')

            except (ValueError,UnboundLocalError):
                pass
    st.write("Github repo: https://github.com/SamuelLawrence876/JSE-Quant-Webapp")

    if choice == 'Portfolio Assessment':
        st.title('Under Construction!')
    

if __name__ == "__main__":
    app()
