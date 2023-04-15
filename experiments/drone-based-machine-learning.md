# Our Drone based Machine Learning for Climate Change
```python
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import folium
import imageio
from tqdm import tqdm_notebook
from folium.plugins import MarkerCluster
import geoplot as gplt
import geopandas as gpd
import geoplot.crs as gcrs
import imageio
import mapclassify as mc
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
import scipy
from itertools import product
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf

plt.style.use('ggplot')
plt.rcParams['font.family'] = 'sans-serif' 
plt.rcParams['font.serif'] = 'Ubuntu' 
plt.rcParams['font.monospace'] = 'Ubuntu Mono' 
plt.rcParams['font.size'] = 14 
plt.rcParams['axes.labelsize'] = 12 
plt.rcParams['axes.labelweight'] = 'bold' 
plt.rcParams['axes.titlesize'] = 12 
plt.rcParams['xtick.labelsize'] = 12 
plt.rcParams['ytick.labelsize'] = 12 
plt.rcParams['legend.fontsize'] = 12 
plt.rcParams['figure.titlesize'] = 12 
plt.rcParams['image.cmap'] = 'jet' 
plt.rcParams['image.interpolation'] = 'none' 
plt.rcParams['figure.figsize'] = (12, 10) 
plt.rcParams['axes.grid']=True
plt.rcParams['lines.linewidth'] = 2 
plt.rcParams['lines.markersize'] = 8
colors = ['xkcd:pale orange', 'xkcd:sea blue', 'xkcd:pale red', 'xkcd:sage green', 'xkcd:terra cotta', 'xkcd:dull purple', 'xkcd:teal', 'xkcd: goldenrod', 'xkcd:cadet blue',
'xkcd:scarlet']
data = pd.read_csv('GlobalLandTemperaturesByMajorCity.csv'
```


## Data Visualization
```python
city_data = data.drop_duplicates(['City'])
city_data.head()
LAT = []
LONG = []
for city in city_data.City.tolist():
    locator = Nominatim(user_agent="myGeocoder")
    location = locator.geocode(city)
    LAT.append(location.latitude)
    LONG.append(location.longitude)
    
city_data['Latitude'] = LAT
city_data['Longitude'] = LONG
<ipython-input-413-7a073d0319c2>:1: SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame. Try using .loc[row_indexer,col_indexer] = value instead city_data['Latitude'] = LAT <ipython-input-413-7a073d0319c2>:2: SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame. Try using .loc[row_indexer,col_indexer] = value instead city_data['Longitude'] = LONG 
from geopy.geocoders import Nominatim

world_map= folium.Map()
geolocator = Nominatim(user_agent="Piero")
marker_cluster = MarkerCluster().add_to(world_map)

for i in range(len(city_data)):
        lat = city_data.iloc[i]['Latitude']
        long = city_data.iloc[i]['Longitude']
        radius=5
        folium.CircleMarker(location = [lat, long], radius=radius,fill =True, color='darkred',fill_color='darkred').add_to(marker_cluster)
world_map
explodes = (0,0.3)
plt.pie(data[data['City']=='Chicago'].AverageTemperature.isna().value_counts(),explode=explodes,startangle=0,colors=['firebrick','indianred'],
   labels=['Non NaN elements','NaN elements'], textprops={'fontsize': 20})
([<matplotlib.patches.Wedge at 0x7fe7c169b8e0>,
  <matplotlib.patches.Wedge at 0x7fe7c169bdc0>],
 [Text(-1.0950344628649946, 0.10440079088767877, 'Non NaN elements'),
  Text(1.3936802176892007, -0.132873815410646, 'NaN elements')])

chicago_data = data[data['City']=='Chicago']
chicago_data['AverageTemperature']=chicago_data.AverageTemperature.fillna(method='bfill')
<ipython-input-415-84f2fdaf2630>:1: SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame. Try using .loc[row_indexer,col_indexer] = value instead chicago_data['AverageTemperature']=chicago_data.AverageTemperature.fillna(method='bfill') 
chicago_data['AverageTemperatureUncertainty']=chicago_data.AverageTemperatureUncertainty.fillna(method='bfill')
<ipython-input-416-78f8cd0627c9>:1: SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame. Try using .loc[row_indexer,col_indexer] = value instead chicago_data['AverageTemperatureUncertainty']=chicago_data.AverageTemperatureUncertainty.fillna(method='bfill') 
chicago_data = chicago_data.reset_index()
chicago_data = chicago_data.drop(columns=['index'])
chicago_data.dt = pd.to_datetime(chicago_data.dt)
YEAR = []
MONTH = []
DAY = []
WEEKDAY = []
for i in range(len(chicago_data)):
    WEEKDAY.append(chicago_data.dt[i].weekday())
    DAY.append(chicago_data.dt[i].day)
    MONTH.append(chicago_data.dt[i].month)
    YEAR.append(chicago_data.dt[i].year)
chicago_data['Year'] = YEAR
chicago_data['Month'] = MONTH
chicago_data['Day'] = DAY 
chicago_data['Weekday'] = WEEKDAY
change_year_index = []
change_year = []
year_list = chicago_data['Year'].tolist()
for y in range(0,len(year_list)-1):
    if year_list[y]!=year_list[y+1]:
        change_year.append(year_list[y+1])
        change_year_index.append(y+1)
chicago_data.loc[change_year_index].head()
x_ticks_year_list=np.linspace(min(year_list),max(year_list),10).astype(int)
change_year_index = np.array(change_year_index)
x_ticks_year_index = []
for i in range(1,len(x_ticks_year_list)):
    x_ticks_year_index.append(change_year_index[np.where(np.array(change_year)==x_ticks_year_list[i])][0])
sns.scatterplot(x=chicago_data.index,y=chicago_data.AverageTemperature,s=25,color='firebrick')
plt.xticks(x_ticks_year_index,x_ticks_year_list)
plt.title('Temperature vs Year Scatter plot',color='firebrick',fontsize=40)
plt.xlabel('Year')
plt.ylabel('Average Temperature')
Text(0, 0.5, 'Average Temperature')

last_year_data = chicago_data[chicago_data.Year>=2010].reset_index().drop(columns=['index'])
P = np.linspace(0,len(last_year_data)-1,5).astype(int)
def get_timeseries(start_year,end_year):
    last_year_data = chicago_data[(chicago_data.Year>=start_year) & (chicago_data.Year<=end_year)].reset_index().drop(columns=['index'])
    return last_year_data
def plot_timeseries(start_year,end_year):
    last_year_data = get_timeseries(start_year,end_year)
    P = np.linspace(0,len(last_year_data)-1,5).astype(int)
    plt.plot(last_year_data.AverageTemperature,marker='.',color='firebrick')
    plt.xticks(np.arange(0,len(last_year_data),1)[P],last_year_data.dt.loc[P],rotation=60)
    plt.xlabel('Date (Y/M/D)')
    plt.ylabel('Average Temperature')
def plot_from_data(data,time,c='firebrick',with_ticks=True,label=None):
    time = time.tolist()
    data = np.array(data.tolist())
    P = np.linspace(0,len(data)-1,5).astype(int)
    time = np.array(time)
    if label==None:
        plt.plot(data,marker='.',color=c)
    else:
        plt.plot(data,marker='.',color=c,label=label)
    if with_ticks==True:
        plt.xticks(np.arange(0,len(data),1)[P],time[P],rotation=60)
    plt.xlabel('Date (Y/M/D)')
    plt.ylabel('Average Temperature')
plt.figure(figsize=(20,20))
plt.suptitle('Plotting 4 decades',fontsize=40,color='firebrick')

plt.subplot(2,2,1)
plt.title('Starting year: 1800, Ending Year: 1810',fontsize=15)
plot_timeseries(1800,1810)
plt.subplot(2,2,2)
plt.title('Starting year: 1900, Ending Year: 1910',fontsize=15)
plot_timeseries(1900,1910)
plt.subplot(2,2,3)
plt.title('Starting year: 1950, Ending Year: 1960',fontsize=15)
plot_timeseries(1900,1910)
plt.subplot(2,2,4)
plt.title('Starting year: 2000, Ending Year: 2010',fontsize=15)
plot_timeseries(1900,1910)
plt.tight_layout()

FFT = np.fft.fft(chicago_data.AverageTemperature)
FFT_abs = np.abs(FFT)
new_N=int(len(FFT)/2) 
f_nat=1
new_X = np.linspace(0, f_nat/2, new_N, endpoint=True)
new_X = 1/new_X
plt.plot(new_X,2*FFT_abs[0:int(len(FFT)/2.)]/len(new_X),color='firebrick')
plt.xlabel('Period ($Month$)',fontsize=20)
plt.ylabel('Amplitude',fontsize=20)
plt.title('(Fast) Fourier Transform Method Algorithm',fontsize=30,color='firebrick')
plt.grid(True)
plt.xlim(2,22)
<ipython-input-404-feddc0648f82>:6: RuntimeWarning: divide by zero encountered in true_divide new_X = 1/new_X 
(2.0, 22.0)
```


## Checking on Stationarity
```python
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(chicago_data.AverageTemperature, ax=ax1,color ='firebrick')
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(chicago_data.AverageTemperature, ax=ax2,color='firebrick')

result = adfuller(chicago_data.AverageTemperature)
print('ADF Statistic on the entire dataset: {}'.format(result[0]))
print('p-value: {}'.format(result[1]))
print('Critical Values:')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value))
ADF Statistic on the entire dataset: -6.029493829973582 p-value: 1.429253097986233e-07 Critical Values: 1%: -3.4323875260668344 5%: -2.862440255934873 10%: -2.5672492261933377 
result = adfuller(chicago_data.AverageTemperature[0:120])
print('ADF Statistic on the first decade: {}'.format(result[0]))
print('p-value: {}'.format(result[1]))
print('Critical Values:')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value))
ADF Statistic on the first decade: -2.096122487386037 p-value: 0.2460766247103494 Critical Values: 1%: -3.4901313156261384 5%: -2.8877122815688776 10%: -2.5807296460459184 
plt.title('The dataset used for prediction', fontsize=30,color='firebrick')
plot_timeseries(1992,2013)

temp = get_timeseries(1992,2013)
N = len(temp.AverageTemperature)
split = 0.95
training_size = round(split*N)
test_size = round((1-split)*N)
series = temp.AverageTemperature[:training_size]
date = temp.dt[:training_size]
test_series = temp.AverageTemperature[len(date)-1:len(temp)]
test_date = temp.dt[len(date)-1:len(temp)]
#test_date = test_date.reset_index().dt
#test_series = test_series.reset_index().AverageTemperature
test_date
247   2012-08-01
248   2012-09-01
249   2012-10-01
250   2012-11-01
251   2012-12-01
252   2013-01-01
253   2013-02-01
254   2013-03-01
255   2013-04-01
256   2013-05-01
257   2013-06-01
258   2013-07-01
259   2013-08-01
260   2013-09-01
Name: dt, dtype: datetime64[ns]
plot_from_data(series,date,label='Training Set')
plot_from_data(test_series,test_date,'navy',with_ticks=False,label='Test Set')
plt.legend()
<matplotlib.legend.Legend at 0x7fe78baf1550>

def optimize_ARIMA(order_list, exog):
    """
        Return dataframe with parameters and corresponding AIC
        
        order_list - list with (p, d, q) tuples
        exog - the exogenous variable
    """
    
    results = []
    
    for order in tqdm_notebook(order_list):
        #try: 
        model = SARIMAX(exog, order=order).fit(disp=-1)
    #except:
    #        continue
            
        aic = model.aic
        results.append([order, model.aic])
    #print(results)
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p, d, q)', 'AIC']
    #Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    
    return result_df
ps = range(0, 10, 1)
d = 0
qs = range(0, 10, 1)
```

## Create a list with all possible combination of parameters
```python
parameters = product(ps, qs)
parameters_list = list(parameters)

order_list = []

for each in parameters_list:
    each = list(each)
    each.insert(1, d)
    each = tuple(each)
    order_list.append(each)
    
result_d_0 = optimize_ARIMA(order_list, exog = series)
<ipython-input-558-69a81fa0ddc3>:11: TqdmDeprecationWarning: This function will be removed in tqdm==5.0.0 Please use `tqdm.notebook.tqdm` instead of `tqdm.tqdm_notebook` for order in tqdm_notebook(order_list): 
HBox(children=(FloatProgress(value=0.0), HTML(value='')))
/Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/tsa/statespace/sarimax.py:975: UserWarning: Non-invertible starting MA parameters found. Using zeros as starting parameters. warn('Non-invertible starting MA parameters found.' /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/tsa/statespace/sarimax.py:963: UserWarning: Non-stationary starting autoregressive parameters found. Using zeros as starting parameters. warn('Non-stationary starting autoregressive parameters' /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " 
/Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " 
/Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " 
[[(0, 0, 0), 2006.5955754200622], [(0, 0, 1), 1744.0622691268422], [(0, 0, 2), 1584.824071971162], [(0, 0, 3), 1490.4667002371614], [(0, 0, 4), 1439.0132560058169], [(0, 0, 5), 1376.3401954432268], [(0, 0, 6), 1364.6587575038036], [(0, 0, 7), 1375.7076153006046], [(0, 0, 8), 1348.572544534084], [(0, 0, 9), 1329.0317837444618], [(1, 0, 0), 1480.1803556701348], [(1, 0, 1), 1427.732652623732], [(1, 0, 2), 1392.3271349346603], [(1, 0, 3), 1373.9406061936177], [(1, 0, 4), 1353.7162031999849], [(1, 0, 5), 1354.6425255272015], [(1, 0, 6), 1322.4678200524272], [(1, 0, 7), 1354.614554467216], [(1, 0, 8), 1309.9134199151172], [(1, 0, 9), 1295.759064714729], [(2, 0, 0), 1387.3660760539801], [(2, 0, 1), 1383.8594232075206], [(2, 0, 2), 1362.4706111480345], [(2, 0, 3), 1353.5457870432053], [(2, 0, 4), 1357.5511761554078], [(2, 0, 5), 1331.174708423257], [(2, 0, 6), 1346.242079983238], [(2, 0, 7), 1305.482089115721], [(2, 0, 8), 1307.2306849196007], [(2, 0, 9), 1313.3830222952272], [(3, 0, 0), 1378.2657613889996], [(3, 0, 1), 1377.3176612796165], [(3, 0, 2), 1175.7074244711857], [(3, 0, 3), 1127.1776823795933], [(3, 0, 4), 1102.619700805783], [(3, 0, 5), 1097.6935770081977], [(3, 0, 6), 1111.6024335574048], [(3, 0, 7), 1111.744202144364], [(3, 0, 8), 1123.5742088647676], [(3, 0, 9), 1101.9429630884645], [(4, 0, 0), 1369.61084494553], [(4, 0, 1), 1247.5767863609503], [(4, 0, 2), 1377.863565136418], [(4, 0, 3), 1181.0874978631746], [(4, 0, 4), 1352.0423315210787], [(4, 0, 5), 1097.9207774505664], [(4, 0, 6), 1097.294457575149], [(4, 0, 7), 1100.44361399836], [(4, 0, 8), 1105.3981157441099], [(4, 0, 9), 1102.4663397705003], [(5, 0, 0), 1339.2947908105275], [(5, 0, 1), 1203.491090411266], [(5, 0, 2), 1251.0437385291198], [(5, 0, 3), 1151.3818500162347], [(5, 0, 4), 1190.1060141761982], [(5, 0, 5), 1355.399475520113], [(5, 0, 6), 1098.0956758442296], [(5, 0, 7), 1095.9978437342154], [(5, 0, 8), 1137.506792487029], [(5, 0, 9), 1103.7157117517677], [(6, 0, 0), 1299.398175306333], [(6, 0, 1), 1176.9280248432685], [(6, 0, 2), 1207.1028737651504], [(6, 0, 3), 1145.3381294791898], [(6, 0, 4), 1360.1363962011437], [(6, 0, 5), 1204.4996477074856], [(6, 0, 6), 1141.3547812827105], [(6, 0, 7), 1101.9772545776723], [(6, 0, 8), 1114.6605714076572], [(6, 0, 9), 1110.2968912974145], [(7, 0, 0), 1249.8721005139332], [(7, 0, 1), 1159.173822708619], [(7, 0, 2), 1140.9272473847368], [(7, 0, 3), 1203.0003397295022], [(7, 0, 4), 1177.3112947799164], [(7, 0, 5), 1156.8849786410237], [(7, 0, 6), 1102.8681117111362], [(7, 0, 7), 1106.9034601197975], [(7, 0, 8), 1098.5522497211518], [(7, 0, 9), 1103.3873245625352], [(8, 0, 0), 1234.7578099259813], [(8, 0, 1), 1158.7238117151655], [(8, 0, 2), 1162.6892692636206], [(8, 0, 3), 1171.4768452904568], [(8, 0, 4), 1196.054851013345], [(8, 0, 5), 1168.5761633444101], [(8, 0, 6), 1104.6752371487432], [(8, 0, 7), 1103.9565500400286], [(8, 0, 8), 1108.4151128603585], [(8, 0, 9), 1101.9381127028985], [(9, 0, 0), 1205.5071894735945], [(9, 0, 1), 1141.9379154542216], [(9, 0, 2), 1162.4819423556783], [(9, 0, 3), 1166.9515804006833], [(9, 0, 4), 1179.1293464130922], [(9, 0, 5), 1159.4158868057434], [(9, 0, 6), 1153.924264397198], [(9, 0, 7), 1100.6000173982238], [(9, 0, 8), 1105.429984528222], [(9, 0, 9), 1123.2335918607264]] 
/Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " 
result_d_0.head()
ps = range(0, 10, 1)
d = 1
qs = range(0, 10, 1)
```

## Create a list with all possible combination of parameters
```python
parameters = product(ps, qs)
parameters_list = list(parameters)

order_list = []

for each in parameters_list:
    each = list(each)
    each.insert(1, d)
    each = tuple(each)
    order_list.append(each)
    
result_d_1 = optimize_ARIMA(order_list, exog = series)

result_d_1
<ipython-input-558-69a81fa0ddc3>:11: TqdmDeprecationWarning: This function will be removed in tqdm==5.0.0 Please use `tqdm.notebook.tqdm` instead of `tqdm.tqdm_notebook` for order in tqdm_notebook(order_list): 
HBox(children=(FloatProgress(value=0.0), HTML(value='')))
/Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/tsa/statespace/sarimax.py:975: UserWarning: Non-invertible starting MA parameters found. Using zeros as starting parameters. warn('Non-invertible starting MA parameters found.' /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/tsa/statespace/sarimax.py:963: UserWarning: Non-stationary starting autoregressive parameters found. Using zeros as starting parameters. warn('Non-stationary starting autoregressive parameters' /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " 
/Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " 
/Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " 
[[(0, 1, 0), 1477.334658915959], [(0, 1, 1), 1428.9620234603017], [(0, 1, 2), 1398.1422356996677], [(0, 1, 3), 1430.9458156623677], [(0, 1, 4), 1372.8832879650986], [(0, 1, 5), 1344.976585814904], [(0, 1, 6), 1312.7854324905447], [(0, 1, 7), 1344.9970511259708], [(0, 1, 8), 1300.321023243617], [(0, 1, 9), 1286.1698713322098], [(1, 1, 0), 1403.014190941968], [(1, 1, 1), 1404.1319324020792], [(1, 1, 2), 1405.2477375277963], [(1, 1, 3), 1378.6644622435874], [(1, 1, 4), 1347.7635457400759], [(1, 1, 5), 1321.3832396856492], [(1, 1, 6), 1331.3046923235556], [(1, 1, 7), 1315.6207760475038], [(1, 1, 8), 1282.0725267480918], [(1, 1, 9), 1316.51899015957], [(2, 1, 0), 1402.963621436826], [(2, 1, 1), 1401.045508703059], [(2, 1, 2), 1167.0987273821659], [(2, 1, 3), 1100.6919342172819], [(2, 1, 4), 1099.1845514163329], [(2, 1, 5), 1085.9168877172992], [(2, 1, 6), 1089.0595767405812], [(2, 1, 7), 1102.4245454369202], [(2, 1, 8), 1099.6047050692143], [(2, 1, 9), 1106.09803200949], [(3, 1, 0), 1380.2125869665272], [(3, 1, 1), 1237.3076164910917], [(3, 1, 2), 1136.562033639523], [(3, 1, 3), 1114.3937352037192], [(3, 1, 4), 1106.5286439895817], [(3, 1, 5), 1090.4043158610739], [(3, 1, 6), 1112.0162224090623], [(3, 1, 7), 1093.439369568022], [(3, 1, 8), 1125.5400626794021], [(3, 1, 9), 1092.3894279045317], [(4, 1, 0), 1338.7945205095798], [(4, 1, 1), 1193.4882602752587], [(4, 1, 2), 1128.7879688415405], [(4, 1, 3), 1140.1151656648128], [(4, 1, 4), 1155.423634988642], [(4, 1, 5), 1104.239259810182], [(4, 1, 6), 1131.920351378175], [(4, 1, 7), 1093.0292125129156], [(4, 1, 8), 1127.001507802086], [(4, 1, 9), 1096.7132634894697], [(5, 1, 0), 1293.1980845686644], [(5, 1, 1), 1166.030529139805], [(5, 1, 2), 1158.6239179118475], [(5, 1, 3), 1131.6310791711544], [(5, 1, 4), 1140.023448371518], [(5, 1, 5), 1094.5366387921463], [(5, 1, 6), 1104.9394468648547], [(5, 1, 7), 1099.6327643545535], [(5, 1, 8), 1097.7777845868843], [(5, 1, 9), 1104.2399996341442], [(6, 1, 0), 1241.3727099862915], [(6, 1, 1), 1149.3011102186597], [(6, 1, 2), 1114.0667501362611], [(6, 1, 3), 1123.2663169472935], [(6, 1, 4), 1128.915470743259], [(6, 1, 5), 1093.4905767761852], [(6, 1, 6), 1099.9226690534615], [(6, 1, 7), 1105.4670909541599], [(6, 1, 8), 1092.3762072178683], [(6, 1, 9), 1094.177596512547], [(7, 1, 0), 1225.683565318736], [(7, 1, 1), 1148.1815689482676], [(7, 1, 2), 1153.0308145933063], [(7, 1, 3), 1132.4960167369627], [(7, 1, 4), 1119.4626035974609], [(7, 1, 5), 1128.1335419661061], [(7, 1, 6), 1103.8461563322653], [(7, 1, 7), 1095.6580353427116], [(7, 1, 8), 1097.955793275964], [(7, 1, 9), 1095.5318810409303], [(8, 1, 0), 1196.0364619295715], [(8, 1, 1), 1131.875486971361], [(8, 1, 2), 1116.8602174421835], [(8, 1, 3), 1154.333894054842], [(8, 1, 4), 1115.5050482692984], [(8, 1, 5), 1100.392827943417], [(8, 1, 6), 1095.0800630343747], [(8, 1, 7), 1093.1787881774417], [(8, 1, 8), 1117.5142792262805], [(8, 1, 9), 1090.4951530980036], [(9, 1, 0), 1157.2746886782002], [(9, 1, 1), 1117.573635566392], [(9, 1, 2), 1115.1331279671285], [(9, 1, 3), 1094.66612273789], [(9, 1, 4), 1111.0844014584163], [(9, 1, 5), 1091.7632476193603], [(9, 1, 6), 1094.9684942087833], [(9, 1, 7), 1091.6161197811134], [(9, 1, 8), 1096.376092979901], [(9, 1, 9), 1091.2026851160963]] 
/Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " 
result_d_1.head()
final_result = result_d_0.append(result_d_1)
best_models = final_result.sort_values(by='AIC', ascending=True).reset_index(drop=True).head()
best_model_params_0 = best_models[best_models.columns[0]][0]
best_model_params_1 = best_models[best_models.columns[0]][1]
best_model_0 = SARIMAX(series, order=best_model_params_0).fit()
print(best_model_0.summary())
best_model_1 = SARIMAX(series, order=best_model_params_1).fit()
print(best_model_1.summary())
/Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/tsa/statespace/sarimax.py:975: UserWarning: Non-invertible starting MA parameters found. Using zeros as starting parameters. warn('Non-invertible starting MA parameters found.' /Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " 
SARIMAX Results ============================================================================== Dep. Variable: AverageTemperature No. Observations: 248 Model: SARIMAX(2, 1, 5) Log Likelihood -534.958 Date: Sat, 17 Apr 2021 AIC 1085.917 Time: 11:42:44 BIC 1113.992 Sample: 0 HQIC 1097.220 - 248 Covariance Type: opg ============================================================================== coef std err z P>|z| [0.025 0.975] ------------------------------------------------------------------------------ ar.L1 1.7320 0.001 3440.332 0.000 1.731 1.733 ar.L2 -1.0000 0.000 -6492.021 0.000 -1.000 -1.000 ma.L1 -2.4616 0.449 -5.478 0.000 -3.342 -1.581 ma.L2 2.0863 0.543 3.841 0.000 1.022 3.151 ma.L3 -0.5374 0.302 -1.779 0.075 -1.130 0.055 ma.L4 0.0227 0.232 0.098 0.922 -0.432 0.477 ma.L5 -0.1099 0.078 -1.403 0.161 -0.263 0.044 sigma2 4.2305 0.114 37.023 0.000 4.007 4.454 =================================================================================== Ljung-Box (Q): 68.29 Jarque-Bera (JB): 8.90 Prob(Q): 0.00 Prob(JB): 0.01 Heteroskedasticity (H): 1.36 Skew: 0.12 Prob(H) (two-sided): 0.17 Kurtosis: 3.90 =================================================================================== Warnings: [1] Covariance matrix calculated using the outer product of gradients (complex-step). [2] Covariance matrix is singular or near-singular, with condition number 1.48e+19. Standard errors may be unstable. SARIMAX Results ============================================================================== Dep. Variable: AverageTemperature No. Observations: 248 Model: SARIMAX(2, 1, 6) Log Likelihood -535.530 Date: Sat, 17 Apr 2021 AIC 1089.060 Time: 11:42:45 BIC 1120.644 Sample: 0 HQIC 1101.776 - 248 Covariance Type: opg ============================================================================== coef std err z P>|z| [0.025 0.975] ------------------------------------------------------------------------------ ar.L1 1.7320 0.001 3339.099 0.000 1.731 1.733 ar.L2 -0.9999 0.000 -3122.916 0.000 -1.001 -0.999 ma.L1 -2.4409 0.783 -3.116 0.002 -3.976 -0.906 ma.L2 1.9998 1.250 1.599 0.110 -0.451 4.451 ma.L3 -0.4058 0.523 -0.777 0.437 -1.430 0.618 ma.L4 -0.0739 0.271 -0.273 0.785 -0.605 0.457 ma.L5 -0.0742 0.192 -0.387 0.698 -0.450 0.301 ma.L6 -0.0049 0.077 -0.064 0.949 -0.155 0.145 sigma2 4.1416 3.348 1.237 0.216 -2.420 10.703 =================================================================================== Ljung-Box (Q): 65.25 Jarque-Bera (JB): 9.13 Prob(Q): 0.01 Prob(JB): 0.01 Heteroskedasticity (H): 1.34 Skew: 0.13 Prob(H) (two-sided): 0.18 Kurtosis: 3.90 =================================================================================== Warnings: [1] Covariance matrix calculated using the outer product of gradients (complex-step). 
/Users/pierohmd/opt/anaconda3/lib/python3.8/site-packages/statsmodels/base/model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals warn("Maximum Likelihood optimization failed to converge. " 
Model (2,1,5) results:
best_model_0.plot_diagnostics(figsize=(15,12))
plt.show()

Model (2,1,6) results:
best_model_1.plot_diagnostics(figsize=(15,12))
plt.show()
```
## Forecasting

```python
fore_l= test_size-1
forecast = best_model_0.get_prediction(start=training_size, end=training_size+fore_l)
forec = forecast.predicted_mean
ci = forecast.conf_int(alpha=0.05)

s_forecast = best_model_1.get_prediction(start=training_size, end=training_size+fore_l)
s_forec = s_forecast.predicted_mean
s_ci = forecast.conf_int(alpha=0.05)
error_test=chicago_data.loc[test_date[1:].index.tolist()].AverageTemperatureUncertainty
index_test = test_date[1:].index.tolist()
test_set = test_series[1:]
lower_test = test_set-error_test
upper_test = test_set+error_test
fig, ax = plt.subplots(figsize=(16,8), dpi=300)
x0 = chicago_data.AverageTemperature.index[0:training_size]
x1=chicago_data.AverageTemperature.index[training_size:training_size+fore_l+1]
#ax.fill_between(forec, ci['lower Load'], ci['upper Load'])
plt.plot(x0, chicago_data.AverageTemperature[0:training_size],'k', label = 'Average Temperature')

plt.plot(chicago_data.AverageTemperature[training_size:training_size+fore_l], '.k', label = 'Actual')

forec = pd.DataFrame(forec, columns=['f'], index = x1)
#forec.f.plot(ax=ax,color = 'Darkorange',label = 'Forecast (d = 2)')
#ax.fill_between(x1, ci['lower AverageTemperature'], ci['upper AverageTemperature'],alpha=0.2, label = 'Confidence inerval (95%)',color='grey')

s_forec = pd.DataFrame(s_forec, columns=['f'], index = x1)
s_forec.f.plot(ax=ax,color = 'firebrick',label = 'Forecast  (2,1,6) model')
ax.fill_between(x1, s_ci['lower AverageTemperature'], s_ci['upper AverageTemperature'],alpha=0.2, label = 'Confidence inerval (95%)',color='grey')


plt.legend(loc = 'upper left')
plt.xlim(80,)
plt.xlabel('Index Datapoint')
plt.ylabel('Temperature')
plt.show()

#plt.plot(forec)
plt.figure(figsize=(12,12))
plt.subplot(2,1,1)
plt.fill_between(x1, lower_test, upper_test,alpha=0.2, label = 'Test set error range',color='navy')
plt.plot(test_set,marker='.',label="Actual",color='navy')
plt.plot(forec,marker='d',label="Forecast",color='firebrick')
plt.xlabel('Index Datapoint')
plt.ylabel('Temperature')
#plt.fill_between(x1, s_ci['lower AverageTemperature'], s_ci['upper AverageTemperature'],alpha=0.3, label = 'Confidence inerval (95%)',color='firebrick')
plt.legend()
plt.subplot(2,1,2)
#plt.fill_between(x1, lower_test, upper_test,alpha=0.2, label = 'Test set error range',color='navy')
plt.plot(test_set,marker='.',label="Actual",color='navy')
plt.plot(s_forec,marker='d',label="Forecast",color='firebrick')
plt.fill_between(x1, ci['lower AverageTemperature'], ci['upper AverageTemperature'],alpha=0.3, label = 'Confidence inerval (95%)',color='firebrick')
plt.legend()
plt.xlabel('Index Datapoint')
plt.ylabel('Temperature')
Text(0, 0.5, 'Temperature')

plt.fill_between(np.arange(0,len(test_set),1), lower_test, upper_test,alpha=0.2, label = 'Test set error range',color='navy')
plot_from_data(test_set,test_date,c='navy',label='Actual')
plot_from_data(forec['f'],test_date,c='firebrick',label='Forecast')
plt.legend(loc=2)
<matplotlib.legend.Legend at 0x7fe76325a700>

import jovian 
jovian.commit()
[jovian] 
 ```
