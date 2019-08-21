#For data handling
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
#For getting data from the internet
from bs4 import BeautifulSoup
import urllib.request
import yfinance as yf
#For clustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import SpectralClustering
from sklearn import decomposition

def get_stock_names():
    #Getting S&P 100 company names from Wikipedia
    #Modified (2016-01-19) https://adesquared.wordpress.com/2013/06/16/using-python-beautifulsoup-to-scrape-a-wikipedia-table/
    wiki = "https://en.wikipedia.org/wiki/S%26P_100"
    header = {'User-Agent': 'Mozilla/5.0'} #Needed to prevent 403 error on Wikipedia
    try:
        req = urllib.request.Request(wiki,headers=header)
        page = urllib.request.urlopen(req)
        soup = BeautifulSoup(page)
        #Finding a table on Wikipedia page
        table = soup.find("table", { "class" : "wikitable sortable" })
        #Finding stock names from the table
        stock_name = []
        for row in table.findAll("tr"):
            cells = row.findAll("td")
            if len(cells) == 2:
                stock_name.append(cells[0].find(text=True))
        print("Retrieved stock names.")
        return stock_name
    except: 
        stock_name = []
        print("Something went wrong. Check your internet connection.")
        return stock_name
    
def get_stock_data(stock_name, start_date, end_date):
    #Reading data from Quandal API, note that free anonymous requests are limited
    #my_api_key = 'your own api key'
    #my_api_key = None
    #Stock data is limited to certain range
    #get_names = ['YAHOO/' + x + '.6' for x in stock_name]
    data = yf.download(stock_name, start_date, end_date)
    #data = data.sort_index()
    return data

def stock_summary(stock_names, start_date, end_date):
    #Creating summary with the stock data
    data = pd.DataFrame()
    for stock_name in stock_names:
        #Get stock data
        sdata = get_stock_data(stock_name, start_date, end_date)
        sdata = sdata['Close']
        data = pd.concat([data, sdata], axis=1, sort=False)
    sdata = data
    #Filling possible NaNs
    sdata=sdata.fillna(method='ffill',axis=0)
    sdata=sdata.fillna(method='bfill',axis=0)
    #Normalising to the first item
    sdata = sdata / sdata.iloc[0]
    #Creating return data summarising each stock
    r_data = pd.DataFrame(index=stock_names)
    #Creating daily returns from the data
    daily = sdata.iloc[1:-1]-sdata.iloc[0:-2].values
    #Creating data summary
    r_data['Mean'] = sdata.mean().values
    r_data['Std'] = sdata.std().values
    r_data['Skew'] = sdata.skew().values
    r_data['Kurt'] = sdata.kurt().values
    r_data['Dailyr_mean'] = daily.mean().values
    r_data['Dailyr_std'] = daily.std().values
    r_data['Dailyr_skew'] = daily.skew().values
    r_data['Dailyr_kurt'] = daily.kurt().values
    r_data = r_data.dropna(axis=0)
    print('Data collected.')
    return r_data



def cluster(X, n, plot):
    #Creating clusters from the stock data
    #Setting random seed for consistency
    np.random.seed(1234)
    label=X.index
    xname = X.columns
    #Perform PCA to reduce dimension
    pca = decomposition.PCA(n_components=2)
    #Scale the data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    #Perform PCA
    pca.fit(X)
    X = pca.transform(X)
    #Transforming data
    X=np.log(abs(X)+1)
    #Defining clusters
    clu=SpectralClustering(n_clusters=n)
    clu.fit(X)
    #Giving labels to the data
    pred = clu.fit_predict(X)
    #Finding the stock names for portfolio
    c_name , centers = mean_stock(X ,label, pred, n)
    #Plotting
    if plot==1:
        #Plotting PCA visualization
        index = np.arange(8)
        plt.figure(figsize=(8,6))
        plt.bar(index,pca.components_[0,:],0.5,label='1. dimension')
        plt.bar(index+0.5,pca.components_[1,:],0.5,color='r',label='2. dimension')
        plt.xticks(index+0.5,xname, rotation=15)
        plt.grid()
        plt.legend()
        plt.title("Bar plot of 1. and 2. PCA component")
        #plt.show()
        plt.savefig("bar.png")
        #Plotting clusters and centers
        cn = clu.fit_predict(X)
        fig = plt.figure(facecolor='white',figsize=(10,6))
        plt.scatter(centers[:,0], centers[:,1], c='green', s=300)
        plt.scatter(X[:,0], X[:,1], c=cn, s=100)
        k=0
        for i in label:
            plt.annotate(i, xy = (X[k,0], X[k,1]))
            k=k+1
        k=0
        for i in c_name:
            plt.annotate(i, xy = (centers[k,0], centers[k,1]), weight='bold',color='red')
            k=k+1
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.title('Clustered PCA data')
        plt.grid()
        #plt.show()
        plt.savefig("pca.png")
    return c_name

def mean_stock(X ,label, pred, n):
    #Finding the mean stock of the cluster
    c_names = [] 
    centers = np.empty((0,2), float)
    length = X.shape[0]
    for i in range(0,n):
        names = label[pred==i]
        center = X[pred==i].mean(axis=0)
        #Finding the id closest to the center (mean)
        min_id=np.argmin(((X[pred==i]-center)**2).sum(axis=1))
        c_names.append(names[min_id])
        centers = np.append(centers,[X[pred==i][min_id]], axis=0)
    return c_names, centers

def q_learn(portfolio, start_train_date, start_test_date, end_date, invest, plot, goal):
    #Q-learning algorithm to trade daily stocks
    sdata = pd.DataFrame()
    for sname in portfolio:
        data = get_stock_data(sname, start_train_date, end_date)
        data = data['Close']
        sdata = pd.concat([sdata, data], axis=1, sort=False)
    data = sdata
    data=data.fillna(method='ffill',axis=0)
    data=data.fillna(method='bfill',axis=0)
    #Selecting history range to study in days
    past = 180
    pl_end = len(data) - len(data[(data.index > pd.to_datetime(start_test_date)) & (data.index < pd.to_datetime(end_date))])
    #Creating the empty Q-matrix
    pn = len(portfolio) # Size of the portfolio
    pa = 3              # Possible actions ['BUY,'SELL','WAIT']
    ps = 3*3*2*2        # Number of states
    Q=np.zeros((pn,ps,pa))
    #Create states
    state_dict = create_states()
    #Q-learn parameters
    gamma = 0.30     # Learning parameter
    alpha = 0.15     # Learning parameter
    state0 = 0       # Initial state
    action0 = 'WAIT' # Initial action
    p_start = 0.55   # Random action probability at the begining of training
    p_end = 0.01     # Random action probability at the end of training
    #Probability decay
    p = get_prob(past, pl_end, len(data), p_start, p_end)
    #Starting the learning loop
    pv=[]
    port_value = np.ones(len(portfolio))*invest/float(len(portfolio))
    for k in range(past,len(data)):
        #Subsetting data
        sub_data = data.iloc[(k-past):k]
        sub_data = sub_data / sub_data.iloc[0]
        #Change since yesterday
        yrt = data.iloc[k].values/data.iloc[k-1].values-1
        #Calculating daily returns
        daily_rt = sub_data.iloc[1:past] - sub_data.iloc[0:(past-1)].values
        #Calculating daily returns indicators
        #ma = pd.rolling_mean(sub_data, 15)
        ma = sub_data.rolling(15).mean()
        #mstd = pd.rolling_std(sub_data, 15)
        mstd = sub_data.rolling(15).std()
        ma_l=ma-2*mstd
        ma_h=ma+2*mstd
        #Sharpe ratio
        sharpe = np.sqrt(252)*daily_rt.mean(axis=0)/daily_rt.std(axis=0)
        #Momentum
        momentum = sub_data.iloc[0] / sub_data.iloc[14] - 1
        #Inputs for states
        sharpe = sharpe.values
        mome = momentum.values
        over_ma_h  = ma.iloc[-1,:].values>ma_h.iloc[-1,:].values
        under_ma_l = ma.iloc[-1,:].values<ma_l.iloc[-1,:].values
        for i in range(0,pn):
            #Get the current state
            state = i_state(sharpe[i], mome[i], over_ma_h[i], under_ma_l[i], state_dict)
            #Update the Q-matrix for one stock
            Q[i,:,:], state0, action0 ,reward = Q_update(Q[i,:,:], state, state0, action0, alpha, gamma, p[k], yrt[i], goal)
            #Updating portfolio value
            port_value[i] = port_value[i] + reward * port_value[i]
        #Saving portfolio value
        pv.append(sum(port_value))
    H=data.iloc[past:len(data)]
    H['value'] = pv
    #Plotting one example stock
    if plot == 1:
        plt.figure(facecolor='white',figsize=(10,6))
        sub_data.iloc[:,0].plot(title=portfolio[0])
        ma.iloc[:,0].plot(color='red')
        ma_l.iloc[:,0].plot(color='grey')
        ma_h.iloc[:,0].plot(color='grey')
        sr = 'Sharpe ratio %.2f' % sharpe[0]
        plt.text(50, 1.0, sr ,color='green', fontsize=15,
        bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})
        plt.legend(['Adj close', 'Rolling ave.','Two std dev'], loc=2)
        plt.xticks(rotation=15)
        plt.ylabel('Value')
        plt.tight_layout()
        #plt.show()
        plt.savefig("stock.png")
    return H['value']

def get_prob(l_start, l_level, l_end, p_start, p_end):
    #Probability decay for the Q update
    x = np.arange(l_end)
    #Linear decay
    y0 = float(p_start)
    y1 = float(p_end)
    x0 = float(l_start)
    x1 = float(l_level)
    #Solving linear equation
    slope = (y0-y1)/(x0-x1)
    inter = slope * (-1)*x1 + y1
    y = slope*x + inter
    #Flat out the end
    y[l_level:l_end] = p_end
    return y


def i_state(sharpe, mome, over_ma_h, under_ma_l, state_dict):
    #This gets states from the state dictionary
    #Sharpe    <= 0, 0-1.5 , 1.5 =>  
    if sharpe < 0.0:
        i0 = 'sh0'
    if sharpe >= 0.0 and sharpe <= 0.80:
        i0 = 'sh1'
    if sharpe > 0.80:
        i0 = 'sh2'
    if np.isnan(sharpe):
        i0 = 'sh0'
    #Momentum  <= 0, 0-0.1 , 0.1 =>   
    if mome < 0.0:
        i1 = 'mo0'
    if mome >= 0.0 and mome <= 0.1:
        i1 = 'mo1'
    if mome > 0.1:
        i1 = 'mo2'
    if np.isnan(mome):
        i1 = 'mo0'
    #over ma_h    False, True    => 
    if over_ma_h == True:
        i2 = 'om1'
    else:
        i2 = 'om0'
    #under ma_l   False, True    =>  
    if under_ma_l == True:
        i3 = 'um1'
    else:
        i3 = 'um0'
    #Combine inputs into a key
    all = i0 + i1 + i2 + i3
    #Return the item from the dictionary
    return state_dict[all]

def i_action( act ):
    #Fuction converts action into an integer
    valid_a = ('BUY','SELL','WAIT')
    i = 0
    for i1 in valid_a:
        if i1 == act:
            return i
        i = i+1

def create_states():
    #This function converts inputs into unique integer
    #and creates a dictionary
    #
    #Vector with sharpe
    valid_sh=('sh0','sh1','sh2')
    #Vector with momentum
    valid_mo=('mo0','mo1','mo2')
    #Vectors with boolean values
    valid_mh = ('om0', 'om1')
    valid_ml = ('um0', 'um1')
    i=0
    state_dict = {}
    #Loops create dictionary with all possible items
    for i1 in valid_sh:
        for i2 in valid_mo:
            for i3 in valid_mh:
                for i4 in valid_ml:
                    all = i1 + i2 + i3 + i4 
                    state_dict[all] = i
                    i=i+1
    return state_dict

def Q_update(Q, state, state0, action0, alpha, gamma, p, yrt, goal):
    #Q-update function
    #Defining valid actions
    v_action = ('BUY','SELL','WAIT')
    #Taking a random action to improve learning
    if random.random() <= p:
        action = v_action[np.argmax(Q[state, :])]
    else:
        action = random.choice(v_action)
    #Calculating reward
    reward = get_reward(action, yrt, goal)
    #Updating the Q-matrix with reinforced learning 
    Q[state0, i_action(action0)] = Q[state0, i_action(action0)] + alpha*(reward + gamma * max(Q[state, :])-Q[state0, i_action(action0)]) 
    #Saving parameters for next time
    state0 = state
    action0 = action
    return Q, state0, action0, reward

def get_reward(action, yrt, goal):
    #Return the reward according the action
    if action == 'BUY':
        reward = yrt
    if action == 'SELL':
        reward = yrt
    if action == 'WAIT':
        reward = 0.0
    if goal == 'lose':
        return reward*(-1.0)
    else:
        return reward


'''
The main function of the code is the following:
1. Get stock names for S&P100 companies
2. Summarize the stock data for these companies
3. Using clustering, selecting 4 stocks for portfolio
4. Use Q-learning to do daily trading for the portfolio
5. Estimate how well the Q-learning worked

*** NOTE ***
get_stock_data() function uses Quandl API key
You can get free API key from www.quandl.com which is
enough to run this code fully. If no API key is provided
number of stocks will be limited
*** NOTE ***

'''

if __name__ == '__main__':
    #If you want plots set plot to 1
    plot = 1
    #Defining short stock names (format FB, GOOG)
    stock_names = get_stock_names()
	#Getting summary information for the stocks
    #Setting range for 2010 - 2014
    start_date = '2013-01-01'
    end_date = '2017-12-31'
    s_data = stock_summary(stock_names, start_date, end_date)
    #Selecting 4 stocks from the summary with PCA and clustering
    portfolio = cluster(s_data, 4, plot)
    #portfolio = ['LOW', 'ALL', 'MSFT', 'AAPL']
    start_train_date = '2013-01-01'
    start_test_date = '2017-12-31'
    end_date = '2018-12-31'
    #Launch Q-learning agent to manage your portfolio
    investment = 100000 #For example in Euro or Dollar
    #Trying to maximise your portfolio
    pv_max = q_learn(portfolio, start_train_date, start_test_date, end_date, investment, plot, 'win')
    #Trying to minimize your portfolio (to test learning)
    pv_min = q_learn(portfolio, start_train_date, start_test_date, end_date, investment, plot, 'lose')
    #Combining results
    pv=pd.concat([pv_max, pv_min], axis=1)
    pv.columns=['To win','To lose']
    #Plotting
    if plot == 1:
       #Plotting the scaled stock summary data
       scaler = MinMaxScaler()
       X = scaler.fit_transform(s_data)
       X = pd.DataFrame(X)
       X.columns = s_data.columns
       pd.plotting.scatter_matrix(X, alpha=0.2, color='green')
       #plt.show()
       plt.savefig("Scatter.png")
       #Plotting the Q-learning results
       pv.plot(title='Q-learning testing', figsize=(10, 6))
       plt.grid()
       plt.ylabel('Value [Euro]')
       plt.tight_layout()
       #plt.show()
       plt.savefig("Value.png")
       #Reference to S&P 500
       ref_data = get_stock_data(['SPY'],start_train_date,end_date)
       ref_data = ref_data['Close']
       hmm=pd.concat([pv,ref_data],axis=1)
       hmm=hmm.iloc[180:-1]/hmm.iloc[180] #Normalise and limit to use range
       #Do nothing data
       temp = pd.DataFrame()
       for stock_name in portfolio:
           donoth = get_stock_data(stock_name, start_train_date, end_date)
           #data = get_stock_data(sname, start_train_date, end_date)
           donoth = donoth['Close']
           temp = pd.concat([temp, donoth], axis=1, sort=False)
       donoth=temp
       donoth=donoth.iloc[180:-1]/donoth.iloc[180]
       donoth=donoth*0.25
       donoth=donoth.sum(axis=1)
       hmm=pd.concat([hmm,donoth],axis=1)
       hmm.columns=['To win','To lose','S&P 500','Do nothing']
       hmm.plot()
       plt.grid()
       plt.ylabel('Normalized value')
       plt.tight_layout()
       #plt.show()
       plt.savefig("Result.png")
    print('Done.')
 



