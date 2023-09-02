
import numpy as np
from flask import Flask,request,render_template
import pickle
import requests, json
import csv
import numpy as np
from flask import Flask,request,render_template
import pickle
import requests, json
import csv
import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver


import urllib.request
from pprint import pprint
from html_table_parser.parser import HTMLTableParser
import pandas as pd

from geopy.geocoders import Nominatim

from flask import Flask,render_template,request,redirect,flash
from werkzeug.utils import secure_filename
from main1 import getPrediction
from main import getPrediction1
import os
from geopy.geocoders import Nominatim
import numpy as np
from flask import Flask,request,render_template
from flask import Flask,render_template,request,redirect,flash
import pickle
import requests, json
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing



UPLOAD_FOLDER = 'static/images/'

app = Flask(__name__,static_folder="static")

app.secret_key = "secret key"

app.config['UPLOAD_FOLDER']= UPLOAD_FOLDER


des = pickle.load(open('models\DS.pkl','rb'))

svm = pickle.load(open('models\SVM.pkl','rb'))

rf = pickle.load(open('models\RF.pkl','rb'))

desfull = pickle.load(open('models\dsfull.pkl','rb'))

svmfull = pickle.load(open('models\svmfull.pkl','rb'))

rff = pickle.load(open("models\s1.pkl",'rb'))


@app.route('/')
def home():
    return render_template('home1.html')

@app.route('/crop',methods=['get'])
def crop():
    return render_template("crop.html")

@app.route('/wnpk',methods=['get'])
def wnpk():
    return render_template('index.html')

@app.route('/npk',methods=['get'])
def npk():
    return render_template('index2.html')

@app.route('/pest',methods=['get'])
def pest():
    return render_template('pest.html')

@app.route('/cost',methods=['get'])
def cost():
    return render_template('cost.html')

@app.route('/weed',methods=['get'])
def weed():
    return render_template('weed.html')

@app.route('/predict',methods=['post'])
def predict():
    
    #int_features=[float(x) for x in request.form.values()]
    #features= [np.array(int_features)]
     
   # temp=float(request.form['temp'])
   # hum=float(request.form['hum'])
    city_name =request.form['city']
    
    
     
    # initialize Nominatim API
    geolocator = Nominatim(user_agent="Your_Project")
    from selenium.webdriver.chrome.options import Options

# Configure Chrome options
    chrome_options = Options()
    chrome_options.headless = True

    PATH_TO_DRIVER = "C:\Program Files (x86)\chromedriver.exe"
    driver = webdriver.Chrome(PATH_TO_DRIVER)
    driver.get("https://en.climate-data.org/search/?q="+city_name)

    search_results = driver.find_element("class name", "search_results")
    link = search_results.find_element("tag name", "a").get_attribute("href")
    print(link)

    driver.quit()
    # Make a request to the website
    url = link


    # Define headers to make the request look like it is coming from a web browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

    # Make a request to the website with the headers included
    req = urllib.request.Request(url, headers=headers)
    f = urllib.request.urlopen(req)

    # Read the contents of the website
    xhtml = f.read().decode('utf-8')

    # Define the HTMLTableParser object and feed the HTML contents into it
    p = HTMLTableParser()
    p.feed(xhtml)

    # Print the table data
    pprint(p.tables[1])

    # Convert the parsed data to a Pandas DataFrame
    df = pd.DataFrame(p.tables[1])
    print("\n\nPANDAS DATAFRAME\n")
    print(df)
    df.to_excel('D:\Mini project\scrap\output.xlsx', index=False)


    df = pd.read_excel('D:\Mini project\scrap\output.xlsx')

    i=int(request.form['mon'])

    hum=df.iloc[5][i]
    temp=df.iloc[1][i]
    temp = float(temp.split(' ')[0])




    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    
    complete_url = base_url + "appid=" + "Your_Api_Key" + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()
    if x["cod"] != "404": 
     y = x["main"]
     current_temperature = y["temp"]
     c = current_temperature - 273.15
     print(c)
     ch = y["humidity"]
     print(ch) 
     z=x["coord"]
     print(type(z['lat']))
     print(type(z['lon']))
     
     # Latitude & Longitude input
     Latitude = str(z['lat'])
     Longitude =str(z['lon'])
     location = geolocator.reverse(Latitude +","+ Longitude)
     address = location.raw['address']
     # traverse the data
     city = address.get('city', '')
     state = address.get('state', '')
     country = address.get('country', '')
     code = address.get('country_code')
     zipcode = address.get('postcode')
     print('City : ', city)
     print('State : ', state)
     print('Country : ', country)
     print('Zip Code : ', zipcode)
     
     with open('pin.csv','r') as csvfile:
         reader=csv.reader(csvfile)
         for row in reader:
             if zipcode in row:
                 
                 dist=row[7]
            
     print("District : " + dist)            
     
     with open('rain.csv','r') as csvfile:
         reader=csv.reader(csvfile)
         
         for row in reader: 
             if dist.upper() in row:
                avgrf=float(row[1])/12
             
     print("Average rainfall is = " + str(avgrf))
        
    else:
     print(" City Not Found ")
    
    
    ph=request.form['ph']
    start, end = map(float, ph.split("-"))
     
  #  features=np.array([[temp,hum,ph,avgrf]])
    
    #prediction=des.predict(features)
    #prediction1=svm.predict(features)
    
    #prediction3=rf.predict(features)
    
    for i in range(int(start*10), int(end*10)+1):
        features=np.array([[temp,hum,(i/10),avgrf]])
        prediction=des.predict(features)
        prediction1=svm.predict(features)
        prediction3=rf.predict(features)

        all_predictions = np.concatenate((prediction, prediction1, prediction3))

# Convert the array into a set of strings
        prediction_set = set(map(str, all_predictions))

# Convert the set into a string
        prediction_string = ', '.join(prediction_set)

# Print the resulting string
    print(prediction_string)
    return render_template('index.html',prediction_text1=' The set of crops recommened are :{} '.format(prediction_string))
 
    
    #return render_template('index.html',prediction_text1=' Support Vector Machine {}'.format(prediction1),prediction_text='Decision Tree {} '.format(prediction),prediction_text3=' Random Forest(Highest Accuracy){} '.format(prediction3))
    

@app.route('/predict1',methods=['post'])
def predict1():
    city_name =request.form['city']
    
    
     
    # initialize Nominatim API
    geolocator = Nominatim(user_agent="project1")

    PATH_TO_DRIVER = "C:\Program Files (x86)\chromedriver.exe"
    driver = webdriver.Chrome(PATH_TO_DRIVER)
    driver.get("https://en.climate-data.org/search/?q="+city_name)

    search_results = driver.find_element("class name", "search_results")
    link = search_results.find_element("tag name", "a").get_attribute("href")
    print(link)

    driver.quit()
    # Make a request to the website
    url = link


    # Define headers to make the request look like it is coming from a web browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

    # Make a request to the website with the headers included
    req = urllib.request.Request(url, headers=headers)
    f = urllib.request.urlopen(req)

    # Read the contents of the website
    xhtml = f.read().decode('utf-8')

    # Define the HTMLTableParser object and feed the HTML contents into it
    p = HTMLTableParser()
    p.feed(xhtml)

    # Print the table data
    pprint(p.tables[1])

    # Convert the parsed data to a Pandas DataFrame
    df = pd.DataFrame(p.tables[1])
    print("\n\nPANDAS DATAFRAME\n")
    print(df)
    df.to_excel('D:\Mini project\scrap\output.xlsx', index=False)


    df = pd.read_excel('D:\Mini project\scrap\output.xlsx')

    i=int(request.form['mon'])

    hum=df.iloc[5][i]
    temp=df.iloc[1][i]
    temp = float(temp.split(' ')[0])




    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    
    complete_url = base_url + "appid=" + "98a25fbf985ed7da9eb47274c9d2d495" + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()
    if x["cod"] != "404": 
     y = x["main"]
     current_temperature = y["temp"]
     c = current_temperature - 273.15
     print(c)
     ch = y["humidity"]
     print(ch) 
     z=x["coord"]
     print(type(z['lat']))
     print(type(z['lon']))
     
     # Latitude & Longitude input
     Latitude = str(z['lat'])
     Longitude =str(z['lon'])
     location = geolocator.reverse(Latitude +","+ Longitude)
     address = location.raw['address']
     # traverse the data
     city = address.get('city', '')
     state = address.get('state', '')
     country = address.get('country', '')
     code = address.get('country_code')
     zipcode = address.get('postcode')
     print('City : ', city)
     print('State : ', state)
     print('Country : ', country)
     print('Zip Code : ', zipcode)
     
     with open('pin.csv','r') as csvfile:
         reader=csv.reader(csvfile)
         for row in reader:
             if zipcode in row:
                 
                 dist=row[7]
            
     print("District : " + dist)            
     
     with open('rain.csv','r') as csvfile:
         reader=csv.reader(csvfile)
         
         for row in reader: 
             if dist.upper() in row:
                avgrf=float(row[1])/12
             
     print("Average rainfall is = " + str(avgrf))
        
    else:
     print(" City Not Found ")
    
    
    n=float(request.form['n'])
    p=float(request.form['p'])
    k=float(request.form['k'])
    print(n,p,k)
    
    
    ph=request.form['ph']
    start, end = map(float, ph.split("-"))
     
  #  features=np.array([[temp,hum,ph,avgrf]])
    
    #prediction=des.predict(features)
    #prediction1=svm.predict(features)
    
    #prediction3=rf.predict(features)
    
    for i in range(int(start*10), int(end*10)+1):
        feat=np.array([[n,p,k,temp,hum,(i/10),avgrf]])
        predi=desfull.predict(feat)
        predi1=svmfull.predict(feat)
        predi3=rff.predict(feat)
        

        all_predictions = np.concatenate((predi, predi1, predi3))

# Convert the array into a set of strings
        prediction_set = set(map(str, all_predictions))

# Convert the set into a string
        prediction_string = ', '.join(prediction_set)

# Print the resulting string
    print(prediction_string)
    return render_template('index2.html',prediction_text1=' The set of crops recommened are :{} '.format(prediction_string))
    
    
@app.route('/wee' ,methods=['POST'])
def wee():
        if request.method=='POST':
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file=request.files['file']
            if file.filename=='':
                flash('No file selected for uploading')
                return redirect(request.url)
            if file:
                filename = secure_filename(file.filename)  #Use this werkzeug method to secure filename. 
                file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
                #getPrediction(filename)
                label = getPrediction1(filename)
                print(label) 
                name=label
                rec=""
                
            if name == "Black-grass":
                  rec="use Flufenacet and pendimethalin"
            elif name == "Small-flowered Cranesbill":
                 rec=" use clomazone+napropamide"
            elif name == "Sugar beet":
                 rec=" use paraquat (Gramoxone SL 2.0)"
            elif name == "Shepherd’s Purse":
               rec= " use  2, 4-D, MCPP (mecoprop), Dicamba*, or Triclopyr"
            elif name == "Scentless Mayweed":
                    rec=" use  bromoxynil (Buctril®) and dicambda (Clarity®) "
            elif name == "Maize":
                   rec=" use Sempra"
            elif name == "Loose Silky-bent":
                    rec=" use Apera spica-venti (L.) P.B"
            elif name == "Fat Hen":
                    rec=" use Roundup Fast Action or Weedol Fast Acting Weedkiller."
            elif name == "Common wheat":
                    rec=" use Isoproturon 800 g/ha "
            elif name == "Common Chickweed":
                    rec=" use  41% Glyphosate "
            elif name == "Cleavers":
                    rec=" use amidosulfuron, florasulam, fluroxypyr, mecoprop"
            else:
                    rec="use Sulfonylurea herbicides (e.g. metsulfuron), triazolopyrimidines "
               
                       
                
            flash("The Identified Weed is " + label )
            flash("The possible weedicide : " + rec)
            full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            flash(full_filename)
            return redirect('/weed')
    
    
@app.route('/pes',methods=['POST'])
def submit_file():
    if request.method=='POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file=request.files['file']
        if file.filename=='':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)  #Use this werkzeug method to secure filename. 
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            #getPrediction(filename)
            label = getPrediction(filename)
            print(label) 
            name=label
            rec=""
            if name == "aphids":
                rec="insecticidal soaps and oils"
            elif name == "armyworm":
                rec="spinosad, bifenthrin, cyfluthrin, and cypermethrin"
            elif name == "beetle":
                rec="Pyrethrin "
            elif name == "bollworm":
                rec=" chlorpyrifos, methomyl, and lambda-cyhalothrin. "
            elif name == "grasshopper":
                rec=" carbaryl"
            elif name == "mites":
                rec="Azobenzene, dicofol, ovex, and tetradifon  "
            elif name == "mosquito":
                rec="   Bifen IT "
            elif name == "sawfly":
                rec="  permethrin, bifenthrin, lambda cyhalothrin, and carbaryl "
            else:
                rec="Neem seed kernel extract 5% or Azadirachtin 0.03% 400 ml/ac"
                   
            
            flash("The Identified Pest is " + label )
            flash("The possible pesticide : " + rec)
            full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            flash(full_filename)
            return redirect('/pest')
@app.route('/calc',methods=['post'])
def calc():
    crop =request.form['cropname']
    n=float(request.form['years'])
    if crop=='wheat':
      df=pd.read_csv('wheat.csv', parse_dates=['Time'], index_col='Time')
    elif crop=='maize':
      df=pd.read_csv('maize.csv', parse_dates=['Time'], index_col='Time')
    elif crop=='rice':
        df=pd.read_csv('rice.csv', parse_dates=['Time'], index_col='Time')

    
    # define alpha and beta values
    alpha = 0.8
    beta = 0.2

    # apply exponential smoothing
    fit = ExponentialSmoothing(df['Value'], trend='add', seasonal=None, initialization_method="estimated").fit(smoothing_level=alpha, smoothing_slope=beta)

    # predict future values
    future_years = pd.date_range(start=df.index[-1], periods=n, freq='Y')
    forecast = fit.forecast(len(future_years))

    # print forecast values
    print(forecast)
    f=str(forecast)
    print(f)
    print("after")
    p=0
    p1=24
    op=""
    n=int(n)
    for i in range(n):
     c=f[p:p1]
     op=c
     p=p+28
     p1=p1+28
     flash(op)
    print(op)
    
    df=pd.read_csv('data.csv',parse_dates=True)
    df=df.dropna()
    
    df
    df_m=df.set_index('date')
    df_m
    df=df_m.drop(['yield'], axis=1)
    df
    df.corr()

    print("\nMissing values : ", df.isnull().any())

    import pmdarima as pm

    model = pm.auto_arima(df['production'],seasonal=False,
    start_p=0, start_q=0, max_p=3,max_d=2,max_q=3, test='adf',error_action='ignore',suppress_warnings=True,stepwise=True, trace=True)

    train=df[(df.index.get_level_values(0) >= '2016-01-01') & (df.index.get_level_values(0)
    <= '2022-01-01')]

    test=df[(df.index.get_level_values(0) > '2022-01-01')]

    model.fit(train['production'])

    forecast=model.predict(n_periods=n, return_conf_int=True)

    forecast
    
    flash("VALUES PREDICTED USING MULTIVARIATE ARIMA MODEL")
    flash(" ")
    f=str(forecast)
    p=1
    p1=24
    n=int(n)
    op=""
    for i in range(n):
     c=f[p:p1]
     op=c
     p=p+28
     p1=p1+28
     flash(op)
    
    
    return render_template('cost.html')
    
        
if __name__=="__main__":
    app.run()