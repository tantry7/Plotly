#!/usr/bin/env python
# coding: utf-8

# In[3]:


pip install plotly pip install cufflinks pip install chart_studio


# In[4]:


pip install seaborn


# In[5]:


pip install matplotlib


# In[6]:


pip install pandas


# In[7]:


pip install numpy


# In[8]:


import pandas as pd
import numpy as np
import chart_studio.plotly as py
import cufflinks as cf
import seaborn as sns
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')

from plotly.offline import download_plotlyjs, init_notebook_mode,plot ,plot, iplot
init_notebook_mode(connected=True)

cf.go_offline()


# In[9]:


#BASICS


# In[10]:


arr = np.random.rand(50,4)
df_1 = pd.DataFrame(arr,columns=['A','B','C','D'])
df_1.head()
df_1.iplot()


# In[11]:


#lINE PLOTS


# In[12]:


import plotly.graph_objects as go
df_stocks = px.data.stocks()
px.line(df_stocks, x ='date',y = 'GOOG',labels = {'x':'Date','y':'Price'})

px.line(df_stocks, x = 'date',y = ['GOOG','AAPL'], labels ={'x':'Date','y':'Price'},title = "Apple Vs Google")

fig = go.Figure()
fig.add_trace(go.Scatter(x = df_stocks.date, y = df_stocks.AAPL,mode="lines", name="Apple"))
fig.add_trace(go.Scatter(x = df_stocks.date, y = df_stocks.AMZN,mode="lines+markers", name="amazon"))
fig.add_trace(go.Scatter(x = df_stocks.date, y = df_stocks.GOOG,mode="lines", name="Google", line = dict(color='Firebrick',width=2,dash="dashdot")))


# In[13]:



fig.update_layout(title="STOCK price data 2018-20",xaxis_title="Price",yaxis_title="Date")
fig.update_layout(
xaxis = dict(
showline=True,showgrid=False,showticklabel=True,
linecolor = 'rgb(204,204,204)',
linewidht=2, tickfont = dict(
family="Arial",size=12,color='rgb(82,82,82)',
),
),
yaxis= dict(showgrid= False,zeroline=False, showline =False, showticklabels = False),
autosize =False,
margin=dict(
autoexpand = False,1=100,r=20,t=110,),
showlegend =False,plot_bgcolor='white')


# In[14]:


#bar chart


# In[15]:


df_us = px.data.gapminder().query("country =='India'")
px.bar(df_us, x ='year',y='pop')


# In[16]:


df_tips = px.data.tips()


# In[17]:


px.bar(df_tips, x='day',y='tip',color='sex',title='tips by sex on each day',labels={'tip':'Tip ammount','day':'Day of the Week'})
px.bar(df_tips, x='sex',y='total_bill',color='smoker',barmode='group')


# In[18]:


df_europe = px.data.gapminder().query("continent == 'Europe' and year == 2007 and pop>2.e6")
fig = px.bar(df_europe , y='pop', x='country', text ='pop', color = 'country')

fig


# In[19]:


fig.update_traces(texttemplate = '%{text:.2s}',textposition='outside')
fig.update_layout(uniformtext_minsize=8)
fig.update_layout(xaxis_tickangle=-45)


# In[20]:


#SCATTER PLOT


# In[21]:


df_iris = px.data.iris()
px.scatter(df_iris, x='sepal_width',y='sepal_length',color='species',size='petal_length',hover_data=['petal_width'])
fig = go.Figure()
fig.add_trace(go.Scatter(x = df_iris.sepal_width, y=df_iris.sepal_length,mode='markers',
                         marker_color= df_iris.sepal_width,
                         text = df_iris.species,marker=dict(showscale=True)))
fig.update_traces(marker_line_width=2,marker_size=10)


# In[22]:


fig = go.Figure(data=go.Scattergl(
x = np.random.randn(100000),
y = np.random.randn(100000),
mode ='markers',
marker = dict(
color = np.random.randn(100000),
colorscale='Viridis',
line_width=1)))
fig


# In[23]:


#pie Charts


# In[24]:


df_asia= px.data.gapminder().query("continent == 'Asia' and year == 2007 ")
px.pie(df_asia,values='pop',names='country',title='population of asian continent',color_discrete_sequence =px.colors.sequential.RdBu)


# In[25]:


colors= ['blue','green','black','purple','red','brown']
fig = go.Figure(data=[go.Pie(labels= ['Water','Grass','Normal','Pyschic','Fire','Ground'],
                             values =[110,90,80,70,60])])
fig.update_traces(hoverinfo="label+percent",textfont_size=20,textinfo='label+percent',pull=[0.1,0.2,0,0,0,0],marker=dict(colors=colors, line= dict(color="#FFFFFF",width=2)))
fig


# In[26]:


#histograms


# In[27]:


dice_1 = np.random.randint(1,7,5000)
dice_2 = np.random.randint(1,7,5000)
dice_sum = dice_1 + dice_2
fig = px.histogram(dice_sum,nbins=11,labels={'value':'Dice roll'},title='5000 Dice Roll Histogram',marginal='violin',color_discrete_sequence=['green'])
fig


# In[28]:


fig.update_layout(
xaxis_title_text = "Dice ROll",
yaxis_title_text = "Dice Sum ",
bargap =0.2, showlegend =False)
fig


# In[29]:


df_tips = px.data.tips()
px.histogram(df_tips,x='total_bill',color = 'sex')


# In[30]:


#BOX PLOTS


# In[31]:


df_tips = px.data.tips()


# In[34]:


px.box(df_tips,x ='sex',y = 'tip',points = 'all')
px.box(df_tips,x = 'day',y ='tip',color='sex')


# In[43]:


fig = go.Figure()
fig.add_trace(go.Box(x=df_tips.sex,y =df_tips.tip,marker_color='blue',boxmean='sd'))


# In[46]:



df_stocks = px.data.stocks()
fig = go.Figure()
fig.add_trace(go.Box(y=df_stocks.GOOG,boxpoints='all',fillcolor='blue',jitter=0.5,whiskerwidth=0.2))
fig.add_trace(go.Box(y=df_stocks.AAPL,boxpoints='all',fillcolor='red',jitter=0.5,whiskerwidth=0.2))
fig.update_layout(title='Google vs Apple',yaxis = dict(gridcolor='rgb(255,255,255)',gridwidth =3),paper_bgcolor='rgb(243,243,243)',plot_bgcolor ='rgb(243,243,243)')


# In[47]:


#voilin plots


# In[49]:


df_tips = px.data.tips()
px.violin(df_tips,y='total_bill',box=True,points='all')


# In[52]:


px.violin(df_tips,y='tip',x='smoker',color='sex',box=True,points='all',hover_data=df_tips.columns)


# In[58]:


px.violin(df_tips,y='tip',x='smoker',color='sex',box=True,points='all',hover_data=df_tips.columns)
fig = go.Figure()
fig.add_trace(go.Violin(x =df_tips['day'][df_tips['smoker']=='Yes'],y=df_tips['total_bill'][df_tips['smoker']=='Yes'],legendgroup='Yes',scalegroup='Yes',name='Yes',side='positive',line_color='Blue'))
fig.add_trace(go.Violin(x =df_tips['day'][df_tips['smoker']=='No'],y=df_tips['total_bill'][df_tips['smoker']=='No'],legendgroup='Yes',scalegroup='Yes',name='No',side='positive',line_color='Red'))


# In[59]:


flights = sns.load_dataset("flights")
flights 


# In[63]:


fig = px.density_heatmap(flights, x ="year",y ="month",z ='passengers',color_continuous_scale='Viridis')
fig


# In[64]:


fig = px.density_heatmap(flights, x ="year",y ="month",z ='passengers',marginal_x ='histogram',marginal_y ='histogram')
fig


# In[66]:


#3D scatter plot


# In[68]:


fig = px.scatter_3d(flights, x ="year",y ="month",z ='passengers',color='year',opacity=0.7)
fig


# In[70]:


fig = px.line_3d(flights, x ="year",y ="month",z ='passengers',color='year')
fig


# In[71]:


#scatter matrices


# In[72]:


fig = px.scatter_matrix(flights,color='month')
fig


# In[73]:


#map scatter plots


# In[74]:


df = px.data.gapminder().query('year == 2007')
fig =px.scatter_geo(df,locations ='iso_alpha',hover_name='country',size='pop',projection='orthographic')
fig


# In[76]:


#polar charts


# In[82]:


df_wind = px.data.wind()
px.scatter_polar(df_wind, r= 'frequency',theta = 'direction',color='strength',size='frequency',symbol='strength')


# In[84]:


df_wind = px.data.wind()
px.line_polar(df_wind, r= 'frequency',theta = 'direction',color='strength',line_close=True,template="plotly_dark")


# In[85]:


#ternary plot 


# In[87]:


df_exp = px.data.experiment()
px.scatter_ternary(df_exp, a='experiment_1',b ='experiment_2',c='experiment_3',hover_name='group',color="gender")


# In[88]:


#facets


# In[91]:


df_tips = px.data.tips()
px.scatter(df_tips,x='total_bill',y='tip',color='smoker',facet_col='sex')
px.histogram(df_tips,x = 'total_bill', y ='tip',color='sex',facet_row='time',facet_col='day',category_orders={'day':["Thur","fri","Sat","Sun"],"time":["Lunch","Dinner"]})


# In[94]:


att_df = sns.load_dataset("attention")
fig = px.line(att_df,x="solutions", y = "score",facet_col="subject",facet_col_wrap=5,title="Scores Based on attention")
fig


# In[95]:


#animated Plot 


# In[97]:


df_cnt = px.data.gapminder()
px.scatter(df_cnt,x ="gdpPercap",y ="lifeExp",animation_frame="year",animation_group="country",size="pop",color='continent',hover_name='country',log_x=True,size_max=55,range_x=[100,10000],range_y=[25,90])


# In[101]:


px.bar(df_cnt,x='continent',y ='pop',color='continent',animation_frame='year',animation_group='country',range_y=[0,4000000000])


# In[ ]:




