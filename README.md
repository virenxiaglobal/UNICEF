
# 🚁 UNICEF Drone Weather Data Analysis

We are using AI algorithms to analyse drone data and identify trends and patterns related to climate change. Our project imports libraries for data processing and machine learning and uses a large variety of Machine Learning and Deep Learning models to fairly distinguish the best approach when predicting the weather. I specifically used two datasets as described below:

## 📊 Dataset
The GlobalLandTemperatures dataset contains historical temperature data for various cities around the world. It includes the following fields:

- dt: the date of the temperature measurement
- AverageTemperature: the average temperature for that day
- AverageTemperatureUncertainty: the uncertainty of the average temperature measurement
- City: the name of the city where the temperature measurement was taken
- Country: the country where the city is located
- Latitude: the latitude of the city
- Longitude: the longitude of the city

The WeatherHistory dataset contains historical weather data for a specific location. It includes the following fields:

- Formatted Date: the date of the weather measurement
- Summary: a summary of the weather conditions for that day
- Precip Type: the type of precipitation, if any
- Temperature (C): the temperature in degrees Celsius
- Apparent Temperature (C): the perceived temperature in degrees Celsius
- Humidity: the humidity level
- Wind Speed (km/h): the wind speed in kilometers per hour
- Wind Bearing (degrees): the direction of the wind in degrees
- Visibility (km): the visibility in kilometers
- Loud Cover: the amount of cloud cover
- Pressure (millibars): the air pressure in millibars
- Daily Summary: a summary of the weather conditions for that day

The data in these datasets can be used for various analyses such as studying climate change, understanding temperature patterns in different cities, or predicting weather conditions based on historical data.


## 🏄🏻‍♂️ General Methodology

1. Collect weather data using drones, including measurements of temperature, humidity, pressure, wind speed and direction, and precipitation.
2. Clean and format the data for training and testing a machine learning model by removing outliers and missing data, and converting it into a compatible format for use with a specific machine learning library.
3. Split the data into training and testing sets for training and evaluating the performance of the model.
4. Choose a machine learning model algorithm, ranging from simple linear regression to complex deep learning neural networks, that can predict weather based on the drone-gathered data.
5. Use the trained model to make predictions on the test data and evaluate its performance by comparing the predicted results with the actual results.
6. Fine-tune the model and repeat steps 4 and 5 until a satisfactory level of accuracy is achieved.
7. Use the final model to make predictions on new, unseen data.

## 💻 Setup
To run the files in this repository, you will need to have the following dependencies installed:
- Python 3
- Virtualenv
First, create a virtual environment using the following command:

```bash
python -m venv .venv
```
Activate the virtual environment on Windows:

```bash
.venv\Scripts\activate.bat
```
Activate the virtual environment on Linux and macOS:

```bash
source .venv/bin/activate
```
Once the virtual environment is activated, install the required libraries by running:

```bash
pip install -r requirements.txt
```
This will install all the libraries listed in the requirements.txt file.

Once the dependencies are installed, you will need to set the virtual environment as the interpreter for your Jupyter notebook by running the following command:

```bash
python -m ipykernel install --user --name=.venv
```
This will add the virtual environment as an option in the Jupyter notebook's kernel list. You will now be able to select the virtual environment as the interpreter for your notebook.
Please note that the images provided in this README are not relevant to the project and are not necessary for running the files in the repository.

[<img src="https://user-images.githubusercontent.com/122212474/213563600-38a5b587-f780-44af-a803-5c840a137829.webm" width="50%">](https://user-images.githubusercontent.com/122212474/213563600-38a5b587-f780-44af-a803-5c840a137829.webm "Now in Android: 55")


[<img src="https://user-images.githubusercontent.com/122212474/213566858-c0747379-6ffb-40c0-9799-556bf1cb363b.webm" width="50%">](https://user-images.githubusercontent.com/122212474/213566858-c0747379-6ffb-40c0-9799-556bf1cb363b.webm "Now in Android: 55")


[<img src="https://user-images.githubusercontent.com/122212474/213567519-b6565105-de1f-4c60-b872-b06163fdd283.webm" width="50%">](https://user-images.githubusercontent.com/122212474/213567519-b6565105-de1f-4c60-b872-b06163fdd283.webm "Now in Android: 55")


[<img src="https://user-images.githubusercontent.com/122212474/213567639-dc7e7c77-b30c-4b1f-9685-d1d3dcf7e972.webm" width="50%">](https://user-images.githubusercontent.com/122212474/213567639-dc7e7c77-b30c-4b1f-9685-d1d3dcf7e972.webm "Now in Android: 55")



