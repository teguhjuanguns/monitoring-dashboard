import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import ARIMA as SF_ARIMA   # atau AutoARIMA kalau pakai itu
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import streamlit as st
import sys
import plotly.express as px
import plotly.graph_objects as go


# (opsional) matikan warning di awal
warnings.filterwarnings("ignore")

import BPTK_Py
from BPTK_Py import Model
from BPTK_Py import sd_functions as sd


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://i.ibb.co.com/nMFgVLF1/Desain-tanpa-judul.png");
             background-attachment: scroll;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 

# -----------------------------------------------------------------------------
# Section: Read Parameter Tables from Google Sheets
# -----------------------------------------------------------------------------

# -*- coding: utf-8 -*-

SHEET_ID = "1Q3PkFQwx3yoROVaDN5MkXv_lndTo9mRkV-2TyB6EhtE"
GID = 0  # tab Eksogen Variable

csv_url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"

df1 = pd.read_csv(csv_url)

df1["Variable"] = df1["Variable"].str.strip()
values = df1.set_index("Variable")["Value"].astype(float)

starttime=values["Start Time"]
stoptime=values["Stop Time"]

model = Model(starttime=starttime,stoptime=stoptime,dt=values["Time Step"],name='EMML vs EMMN')

def power(base_val, exponent_val):
    """Custom power function for BPTK"""
    t = sd.time()
    return lambda t: pow(base_val(t), exponent_val(t))

warnings.filterwarnings("ignore")

GID2 = 1250430400
csv_url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID2}"

dfforecast1 = pd.read_csv(csv_url)
dfforecast1 = dfforecast1[["Year", "Qty"]].dropna(how="any")

dfforecast1["Year"] = dfforecast1["Year"].astype(float)
dfforecast1["Qty"]  = dfforecast1["Qty"].astype(float)
Total_Motorcycle_Ownership = dfforecast1.values.tolist()
print(Total_Motorcycle_Ownership)

#####------------------------------------------------------------------------------------------------------------------------

GID3 = 1824036385
csv_url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID3}"

dfforecast2 = pd.read_csv(csv_url)
dfforecast2 = dfforecast2[["Year", "Qty"]].dropna(how="any")

dfforecast2["Year"] = dfforecast2["Year"].astype(float)
dfforecast2["Qty"]  = dfforecast2["Qty"].astype(float)
EMMN_EMML_Ownership = dfforecast2.values.tolist()
print(EMMN_EMML_Ownership)

#####------------------------------------------------------------------------------------------------------------------------

GID4 = 1545911132
csv_url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID4}"

dfforecast3 = pd.read_csv(csv_url)
dfforecast3 = dfforecast3[["Year", "Qty"]].dropna(how="any")

dfforecast3["Year"] = dfforecast3["Year"].astype(float)
dfforecast3["Qty"]  = dfforecast3["Qty"].astype(float)
EMML_Ownership = dfforecast3.values.tolist()
print(EMML_Ownership)

#####------------------------------------------------------------------------------------------------------------------------

model.points["Total_Motorcycle_Ownership"] = Total_Motorcycle_Ownership

Total_Motorcycle_Ownership = model.converter("Total_Motorcycle_Ownership")
Total_Motorcycle_Ownership.equation = sd.lookup(sd.time(), "Total_Motorcycle_Ownership")

model.points["EMMN_EMML_Ownership"] = EMMN_EMML_Ownership

EMMN_EMML_Ownership = model.converter("EMMN_EMML_Ownership")
EMMN_EMML_Ownership.equation = sd.lookup(sd.time(), "EMMN_EMML_Ownership")

model.points["EMML_Ownership"] = EMML_Ownership

EMML_Ownership = model.converter("EMML_Ownership")
EMML_Ownership.equation = sd.lookup(sd.time(), "EMML_Ownership")

#####------------------------------------------------------------------------------------------------------------------------

EMML_Ownership_Proportion = model.converter("EMML_Ownership_Proportion")
EMML_Ownership_Proportion.equation = EMML_Ownership / Total_Motorcycle_Ownership

EMMN_Ownership_Proportion = model.converter("EMMN_Ownership_Proportion")
EMMN_Ownership_Proportion.equation = (EMMN_EMML_Ownership-EMML_Ownership) / Total_Motorcycle_Ownership

ICE_Ownership_Proportion = model.converter("ICE_Ownership_Proportion")
ICE_Ownership_Proportion.equation = 1 - EMML_Ownership_Proportion - EMMN_Ownership_Proportion

##"""**Authorities**"""

CSOx = model.constant("CSOx")
CSOx.equation = values["Total Cost of Emissions SOx"]

CPM10 = model.constant("CPM10")
CPM10.equation = values["Total Cost of Emissions PM10"]

CNOx = model.constant("CNOx")
CNOx.equation = values["Total Cost of Emissions NOx"]

AQ_Price = model.constant("AQ_Price")
AQ_Price.equation = values["Air Quality Emission Cost"]

GHG_Price = model.constant("GHG_Price")
GHG_Price.equation = values["Greenhouse Gas Emission Cost"]

#####------------------------------------------------------------------------------------------------------------------------

NPUrbanICE = model.constant("NPUrbanICE")
NPUrbanICE.equation = values["Urban Noise Cost ICE"]
NPSuburbICE = model.constant("NPSuburbICE")
NPSuburbICE.equation = values["Suburban Noise Cost ICE"]
Noise_Cost_ICE_ = model.converter("Noise_Cost_ICE_")
Noise_Cost_ICE_.equation = NPUrbanICE + NPSuburbICE

NPUrbanEM = model.constant("NPUrbanEM")
NPUrbanEM.equation = values["Urban Noise Cost EM"]
NPSuburbEM = model.constant("NPSuburbEM")
NPSuburbEM.equation = values["Suburban Noise Cost EM"]
Noise_Cost_EM = model.converter("Noise_Cost_EM")
Noise_Cost_EM.equation = NPUrbanEM + NPSuburbEM

#####------------------------------------------------------------------------------------------------------------------------

EM_Price_Subsidy = model.converter("EM_Price_Subsidy")
EM_Price_Subsidy._function_string = (
    "lambda model, t: 0.0 if t < 2023.0 else (7e+06 if t < 2025.0 else 0.0)"
)
EM_Price_Subsidy.generate_function()

#####------------------------------------------------------------------------------------------------------------------------

SWDKLLJ = model.constant("SWDKLLJ")
SWDKLLJ.equation = values["SWDKLLJ"]

#####------------------------------------------------------------------------------------------------------------------------

Discount_Rate = model.constant("Discount_Rate")
Discount_Rate.equation = values["Discount Rate"]

#####------------------------------------------------------------------------------------------------------------------------

Tax_Cost_per_5_Years = model.constant("Tax_Cost_per_5_Years");
Tax_Cost_per_5_Years.equation = values["Five-Year Administrative Tax Cost"]

#####------------------------------------------------------------------------------------------------------------------------

Purchase_Tax = model.constant("Purchase_Tax")
Purchase_Tax.equation = values["Purchase Tax"]

Purchase_Tax_Subsidy_pct = model.converter("Purchase_Tax_Subsidy_pct")
Purchase_Tax_Subsidy_pct._function_string = (
    "lambda model, t: 0.11 if t < 2023.0 else 0.01"
)

Purchase_Tax_Subsidy_pct.generate_function()

#####------------------------------------------------------------------------------------------------------------------------

PKB_pct = model.constant("PKB_pct")
PKB_pct.equation = values["Motorcycle PKB Rate"]

PKB_pct_Subsidy = model.converter("PKB_pct_Subsidy")

PKB_pct_Subsidy._function_string = (
    "lambda model, t: 0.02 if t < 2021.0 else (0.002 if t < 2023.0 else 0.0)"
)

PKB_pct_Subsidy.generate_function()

#####------------------------------------------------------------------------------------------------------------------------

EPIRate = model.constant("EPIRate")
EPIRate.equation = values["Electricity Price Annual Increase Rate"]
Electricity_Price_Increase = model.flow("Electricity_Price_Increase")
Electricity_Price = model.stock("Electricity_Price")
Electricity_Price.initial_value = values["Initial Electricity Tariff for EM Charging"]
Electricity_Price.equation = Electricity_Price_Increase
Electricity_Price_Increase.equation = EPIRate * Electricity_Price

#####------------------------------------------------------------------------------------------------------------------------

FPIRate = model.constant("FPIRate")
FPIRate.equation = values["Fuel Price Annual Increase Rate"]
Fuel_Price_Increase = model.flow("Fuel_Price_Increase")
Fuel_Price = model.stock("Fuel_Price")
Fuel_Price.initial_value = values["Initial Fuel Tariff for ICE"]
Fuel_Price.equation = Fuel_Price_Increase
Fuel_Price_Increase.equation = FPIRate * Fuel_Price

#"""**User (GDP)**"""

Increase_Rate_in_Asia_GDP_per_Capita = model.constant("Increase_Rate_in_Asia_GDP_per_Capita")
Increase_Rate_in_Asia_GDP_per_Capita.equation = values["Increase Rate in Asia GDP per Capita"]
Average_GDP_per_Capita_in_Asia = model.stock("Average_GDP_per_Capita_in_Asia")
Average_GDP_per_Capita_in_Asia.initial_value = values["Initial Average GDP per Capita in Asia"]
Average_GDP_per_Capita_in_Asia.equation = (Increase_Rate_in_Asia_GDP_per_Capita * Average_GDP_per_Capita_in_Asia)

#####------------------------------------------------------------------------------------------------------------------------

Birth_Rate = model.constant("Birth_Rate")
Birth_Rate.equation = values["Birth Rate"]

Birth = model.flow("Birth")

Death_Rate = model.constant("Death_Rate")
Death_Rate.equation = values["Death Rate"]

Death = model.flow("Death")

Population = model.stock("Population")
Population.initial_value = values["Initial Population"]
Population.equation = Birth - Death

Birth.equation = Birth_Rate * Population
Death.equation = Death_Rate * Population

#####------------------------------------------------------------------------------------------------------------------------

GDP_Increase_Rate = model.constant("GDP_Increase_Rate")
GDP_Increase_Rate.equation = values["GDP Increase Rate"]
GDP_Increase = model.flow("GDP_Increase")
GDP = model.stock("GDP")
GDP.initial_value = values["Initial GDP"]
GDP.equation = GDP_Increase
GDP_Increase.equation = GDP * GDP_Increase_Rate

#####------------------------------------------------------------------------------------------------------------------------

GDP_per_Capita = model.converter("GDP_per_Capita")
GDP_per_Capita.equation = (GDP / Population)

GDP_per_capita_ratio = model.converter("GDP_per_capita_ratio")
GDP_per_capita_ratio.equation = GDP_per_Capita / Average_GDP_per_Capita_in_Asia

#"""**Manufacturer (Pricing)**"""

ICEPIncRate = model.constant("ICEPIncRate")
ICEPIncRate.equation = values["ICE Price Increase Rate"]
ICEPInc = model.flow("ICEPInc")
ICE_Price = model.stock("ICE_Price")
ICE_Price.initial_value = values["Initial ICE Price"]
ICE_Price.equation = ICEPInc
ICEPInc.equation = ICE_Price * ICEPIncRate

#####------------------------------------------------------------------------------------------------------------------------

emml_price_increase_rate_value = values["EMML Price Increase Rate"]
EMMLPIncRate = model.converter("EMMLPIncRate")
EMMLPIncRate._function_string = (
    f"lambda model, t: 0.0 if t < 2024.0 else {emml_price_increase_rate_value}"
)
EMMLPIncRate.generate_function()
EMMLPInc = model.flow("EMMLPInc")
EMML_Price_ = model.stock("EMML_Price_")
EMML_Price_.initial_value = values["Initial EMML Price"]
EMML_Price_.equation = EMMLPInc
EMMLPInc.equation = EMML_Price_ * EMMLPIncRate

EMML_Price = model.converter("EMML_Price")
EMML_Price._function_string = (
    "lambda model, t: 0.0 if t < 2024.0 else model.memoize('EMML_Price_', t)"
)
EMML_Price.generate_function()

Distance_Travelled_EMML = model.constant("Distance_Travelled_EMML")
Distance_Travelled_EMML.equation = values["EMML Annual Distance Travelled"]

Distance_Traveled_per_Charge_EMML = model.constant("Distance_Traveled_per_Charge_EMML")
Distance_Traveled_per_Charge_EMML.equation = values["Distance Traveled per Charge EMML"]

Charging_Frequency_EMML = model.converter("Charging_Frequency_EMML")
Charging_Frequency_EMML._function_string = (
    "lambda model, t: "
    "model.memoize('Distance_Travelled_EMML', t) / "
    "model.memoize('Distance_Traveled_per_Charge_EMML', t)"
)
Charging_Frequency_EMML.generate_function()

Charging_Time_EMML = model.constant("Charging_Time_EMML")
Charging_Time_EMML.equation = values["EMML Charging Time"]

Battery_Charging_Cycle_EMML = model.constant("Battery_Charging_Cycle_EMML")
Battery_Charging_Cycle_EMML.equation = values["Battery Charging Cycles EMML"]

Average_Expected_Years_Kept_Battery_EMML = model.converter("Average_Expected_Years_Kept_Battery_EMML")
Average_Expected_Years_Kept_Battery_EMML._function_string = (
    "lambda model, t: "
    "model.memoize('Battery_Charging_Cycle_EMML', t) / "
    "model.memoize('Charging_Frequency_EMML', t)"
)
Average_Expected_Years_Kept_Battery_EMML.generate_function()

Depreciation_Battery_EMML = model.constant("Depreciation_Battery_EMML")
Depreciation_Battery_EMML.equation = values["EMML Battery Depreciation Rate"]

Battery_Price_EMML = model.converter("Battery_Price_EMML")
Battery_Price_EMML.equation = 0.25 * EMML_Price

#####------------------------------------------------------------------------------------------------------------------------

emmn_price_increase_rate_value = values["EMMN Price Increase Rate"]
EMMNPIncRate = model.converter("EMMNPIncRate")
EMMNPIncRate._function_string = (
    f"lambda model, t: 0.0 if t < 2017.0 else {emmn_price_increase_rate_value}"
)
EMMNPIncRate.generate_function()
EMMNPInc = model.flow("EMMNPInc")
EMMN_Price_ = model.stock("EMMN_Price_")
EMMN_Price_.initial_value = values["Initial EMMN Price"]
EMMN_Price_.equation = EMMNPInc
EMMNPInc.equation = EMMN_Price_ * EMMNPIncRate

EMMN_Price = model.converter("EMMN_Price")
EMMN_Price._function_string = (
    "lambda model, t: 0.0 if t < 2017.0 else model.memoize('EMMN_Price_', t)"
)
EMMN_Price.generate_function()

Distance_Travelled_EMMN = model.constant("Distance_Travelled_EMMN")
Distance_Travelled_EMMN.equation = values["EMMN Annual Distance Travelled"]

Distance_Traveled_per_Charge_EMMN = model.constant("Distance_Traveled_per_Charge_EMMN")
Distance_Traveled_per_Charge_EMMN.equation = values["Distance Traveled per Charge EMMN"]

Charging_Frequency_EMMN = model.converter("Charging_Frequency_EMMN")
Charging_Frequency_EMMN._function_string = (
    "lambda model, t: "
    "model.memoize('Distance_Travelled_EMMN', t) / "
    "model.memoize('Distance_Traveled_per_Charge_EMMN', t)"
)
Charging_Frequency_EMMN.generate_function()

Charging_Time_EMMN = model.constant("Charging_Time_EMMN")
Charging_Time_EMMN.equation = values["EMMN Charging Time"]

Battery_Charging_Cycle_EMMN = model.constant("Battery_Charging_Cycle_EMMN")
Battery_Charging_Cycle_EMMN.equation = values["Battery Charging Cycles EMMN"]

Average_Expected_Years_Kept_Battery_EMMN = model.converter("Average_Expected_Years_Kept_Battery_EMMN")
Average_Expected_Years_Kept_Battery_EMMN._function_string = (
    "lambda model, t: "
    "model.memoize('Battery_Charging_Cycle_EMMN', t) / "
    "model.memoize('Charging_Frequency_EMMN', t)"
)
Average_Expected_Years_Kept_Battery_EMMN.generate_function()

Depreciation_Battery_EMMN = model.constant("Depreciation_Battery_EMMN")
Depreciation_Battery_EMMN.equation = values["EMMN Battery Depreciation Rate"]

Battery_Price_EMMN = model.converter("Battery_Price_EMMN")
Battery_Price_EMMN.equation = 0.25 * EMMN_Price

#"""**Manufacturer (Energy Consumption)**"""

ECIncRICE = model.constant("ECIncRICE")
ECIncRICE.equation = values["ICE Energy Consumption Increase Rate"]
ECIncICE = model.flow("ECIncICE")
Energy_Consumption_Rate_ICE = model.stock("Energy_Consumption_Rate_ICE")
Energy_Consumption_Rate_ICE.initial_value = values["Initial ICE Energy Consumption Rate"]
Energy_Consumption_Rate_ICE.equation = ECIncICE
ECIncICE.equation = Energy_Consumption_Rate_ICE * ECIncRICE

#####------------------------------------------------------------------------------------------------------------------------

emml_energy_consumption_increase_rate_value = values["EMML Energy Consumption Increase Rate"]
ECIncREMML = model.converter("ECIncREMML")
ECIncREMML._function_string = (
    f"lambda model, t: 0.0 if t < 2024.0 else (pow(1+{emml_energy_consumption_increase_rate_value}, model.memoize('Charging_Frequency_EMML',t))-1)"
)
ECIncREMML.generate_function()
ECIncEMML = model.flow("ECIncEMML")
Energy_Consumption_Rate_EMML_ = model.stock("Energy_Consumption_Rate_EMML_")
Energy_Consumption_Rate_EMML_.initial_value = values["Initial EMML Energy Consumption Rate"]
Energy_Consumption_Rate_EMML_.equation = ECIncEMML
ECIncEMML.equation = Energy_Consumption_Rate_EMML_ * ECIncREMML

Energy_Consumption_Rate_EMML = model.converter("Energy_Consumption_Rate_EMML")
Energy_Consumption_Rate_EMML._function_string = (
    "lambda model, t: 0.0 if t < 2024.0 else model.memoize('Energy_Consumption_Rate_EMML_', t)"
)
Energy_Consumption_Rate_EMML.generate_function()

#####------------------------------------------------------------------------------------------------------------------------

emmn_energy_consumption_increase_rate_value = values["EMMN Energy Consumption Increase Rate"]
ECIncREMMN = model.converter("ECIncREMMN")
ECIncREMMN._function_string = (
    f"lambda model, t: 0.0 if t < 2017.0 else (pow(1+{emmn_energy_consumption_increase_rate_value}, model.memoize('Charging_Frequency_EMMN',t))-1)"
)
ECIncREMMN.generate_function()
ECIncEMMN = model.flow("ECIncEMMN")
Energy_Consumption_Rate_EMMN_ = model.stock("Energy_Consumption_Rate_EMMN_")
Energy_Consumption_Rate_EMMN_.initial_value = values["Initial EMMN Energy Consumption Rate"]
Energy_Consumption_Rate_EMMN_.equation = ECIncEMMN
ECIncEMMN.equation = Energy_Consumption_Rate_EMMN_ * ECIncREMMN

Energy_Consumption_Rate_EMMN = model.converter("Energy_Consumption_Rate_EMMN")
Energy_Consumption_Rate_EMMN._function_string = (
    "lambda model, t: 0.0 if t < 2017.0 else model.memoize('Energy_Consumption_Rate_EMMN_', t)"
)
Energy_Consumption_Rate_EMMN.generate_function()

#"""**User (TCO of ICE)**"""

SOxICEInc_Rate = model.constant("SOxICEInc_Rate")
SOxICEInc_Rate.equation = values["SOxICE Increase Rate"]
SOxICEIncrease = model.flow("SOxICEIncrease")
SOxICE = model.stock("SOxICE")
SOxICE.initial_value = values["Initial Total Emissions SOxICE"]
SOxICE.equation = SOxICEIncrease
SOxICEIncrease.equation = SOxICE * SOxICEInc_Rate

PM10ICEInc_Rate = model.constant("PM10ICEInc_Rate")
PM10ICEInc_Rate.equation = values["PM10ICE Increase Rate"]
PM10ICEIncrease = model.flow("PM10ICEIncrease")
PM10ICE = model.stock("PM10ICE")
PM10ICE.initial_value = values["Initial Total Emissions PM10ICE"]
PM10ICE.equation = PM10ICEIncrease
PM10ICEIncrease.equation = PM10ICE * PM10ICEInc_Rate

NOxICEInc_Rate = model.constant("NOxICEInc_Rate")
NOxICEInc_Rate.equation = values["NOxICE Increase Rate"]
NOxICEIncrease = model.flow("NOxICEIncrease")
NOxICE = model.stock("NOxICE")
NOxICE.initial_value = values["Initial Total Emissions NOxICE"]
NOxICE.equation = NOxICEIncrease
NOxICEIncrease.equation = NOxICE * NOxICEInc_Rate

External_Cost_Health_ImpactsICE = model.converter("External_Cost_Health_ImpactsICE")
External_Cost_Health_ImpactsICE.equation = (CNOx * NOxICE + CPM10 * PM10ICE + CSOx * SOxICE)

#####------------------------------------------------------------------------------------------------------------------------

SEInc_Rate = model.constant("SEInc_Rate")
SEInc_Rate.equation = values["Social External Cost Increase Rate"]
SEIncrease = model.flow("SEIncrease")
Social_External_Cost = model.stock("Social_External_Cost")
Social_External_Cost.initial_value = values["Initial Social External Cost"]
Social_External_Cost.equation = SEIncrease
SEIncrease.equation = Social_External_Cost * SEInc_Rate

#####------------------------------------------------------------------------------------------------------------------------

Maintenance_Time_ICE = model.constant("Maintenance_Time_ICE")
Maintenance_Time_ICE.equation = values["ICE Maintenance Time per Visit"]

Maintenance_Frequency_ICE = model.constant("Maintenance_Frequency_ICE")
Maintenance_Frequency_ICE.equation = values["ICE Maintenance Frequency"]

#####------------------------------------------------------------------------------------------------------------------------

Distance_Travelled_ICE = model.constant("Distance_Travelled_ICE")
Distance_Travelled_ICE.equation = values["ICE Annual Distance Travelled"]

#####------------------------------------------------------------------------------------------------------------------------

Health_Costs_of_Emissions_ICE = model.converter("Health_Costs_of_Emissions_ICE")
Health_Costs_of_Emissions_ICE.equation = Distance_Travelled_ICE * External_Cost_Health_ImpactsICE

#####------------------------------------------------------------------------------------------------------------------------

Energy_Consumption_ICE = model.converter("Energy_Consumption_ICE")
Energy_Consumption_ICE.equation = Distance_Travelled_ICE * Energy_Consumption_Rate_ICE * Fuel_Price

#####------------------------------------------------------------------------------------------------------------------------

GHG_Rate_ICEInc_Rate = model.constant("GHG_Rate_ICEInc_Rate")
GHG_Rate_ICEInc_Rate.equation = values["ICE Greenhouse Gas Emission Increase Rate"]
GHG_Rate_ICEIncrease = model.flow("GHG_Rate_ICEIncrease")
GHG_Rate_ICE = model.stock("GHG_Rate_ICE")
GHG_Rate_ICE.initial_value = values["Initial ICE Greenhouse Gas Emission Rate"]
GHG_Rate_ICE.equation = GHG_Rate_ICEIncrease
GHG_Rate_ICEIncrease.equation = GHG_Rate_ICE * GHG_Rate_ICEInc_Rate

#####------------------------------------------------------------------------------------------------------------------------

AQ_Rate_ICEInc_Rate = model.constant("AQ_Rate_ICEInc_Rate")
AQ_Rate_ICEInc_Rate.equation = values["ICE Greenhouse Gas Emission Increase Rate"]
AQ_Rate_ICEIncrease = model.flow("AQ_Rate_ICEIncrease")
AQ_Rate_ICE = model.stock("AQ_Rate_ICE")
AQ_Rate_ICE.initial_value = values["Initial ICE Air Quality Emission Rate"]
AQ_Rate_ICE.equation = AQ_Rate_ICEIncrease
AQ_Rate_ICEIncrease.equation = AQ_Rate_ICE * AQ_Rate_ICEInc_Rate

#####------------------------------------------------------------------------------------------------------------------------

Air_Quality_Cost_ICE = model.converter("Air_Quality_Cost_ICE")
Air_Quality_Cost_ICE.equation = (Distance_Travelled_ICE * AQ_Rate_ICE) * AQ_Price

#####------------------------------------------------------------------------------------------------------------------------

Noise_Cost_ICE = model.converter("Noise_Cost_ICE")
Noise_Cost_ICE.equation = Distance_Travelled_ICE* Noise_Cost_ICE_

#####------------------------------------------------------------------------------------------------------------------------

Greenhouse_Gas_Cost_ICE = model.converter("Greenhouse_Gas_Cost_ICE")
Greenhouse_Gas_Cost_ICE.equation = (Distance_Travelled_ICE * GHG_Rate_ICE) * GHG_Price

#####------------------------------------------------------------------------------------------------------------------------

Refueling_Frequency_ICE = model.constant("Refueling_Frequency_ICE")
Refueling_Frequency_ICE.equation = values["ICE Refueling Frequency"]

Refueling_Time_ICE = model.constant("Refueling_Time_ICE")
Refueling_Time_ICE.equation = values["ICE Refueling Time"]

Refueling_Loss_Cost_ICE = model.converter("Refueling_Loss_Cost_ICE")
Refueling_Loss_Cost_ICE.equation = (Refueling_Frequency_ICE * Refueling_Time_ICE) * Social_External_Cost

#####------------------------------------------------------------------------------------------------------------------------

Maintenance_Loss_Cost_ICE = model.converter("Maintenance_Loss_Cost_ICE")
Maintenance_Loss_Cost_ICE.equation = (Maintenance_Frequency_ICE * Maintenance_Time_ICE) * Social_External_Cost

#####------------------------------------------------------------------------------------------------------------------------

Average_Expected_Years_Kept_ICE = model.constant("Average_Expected_Years_Kept_ICE")
Average_Expected_Years_Kept_ICE.equation = values["Average Expected Vehicle Lifetime"]

#####------------------------------------------------------------------------------------------------------------------------

Social_Impact_Cost_ICE = model.converter("Social_Impact_Cost_ICE")
Social_Impact_Cost_ICE.equation = (Refueling_Loss_Cost_ICE + Health_Costs_of_Emissions_ICE+ Maintenance_Loss_Cost_ICE) * Average_Expected_Years_Kept_ICE

#####------------------------------------------------------------------------------------------------------------------------

Environmental_Impact_Cost_ICE = model.converter("Environmental_Impact_Cost_ICE")
Environmental_Impact_Cost_ICE.equation = (Air_Quality_Cost_ICE + Greenhouse_Gas_Cost_ICE + Noise_Cost_ICE) * Average_Expected_Years_Kept_ICE

#####------------------------------------------------------------------------------------------------------------------------

Depreciation_ICE = model.constant("Depreciation_ICE")
Depreciation_ICE.equation = values["ICE Depreciation Rate"]

#####------------------------------------------------------------------------------------------------------------------------

Resale_Value_ICE = model.converter("Resale_Value_ICE")
Resale_Value_ICE._function_string = (
    "lambda model, t: ("
    "pow(1.0 - model.memoize('Depreciation_ICE', t), "
    "model.memoize('Average_Expected_Years_Kept_ICE', t))"
    "* model.memoize('ICE_Price', t)"
    ") / "
    "pow(1.0 + model.memoize('Discount_Rate', t), "
    "(model.memoize('Average_Expected_Years_Kept_ICE', t) - 1.0))"
)

Resale_Value_ICE.generate_function()

#"""**User (TCO of EMML)**"""

SOxEM = model.constant("SOxEM")
SOxEM.equation = values["Total Emissions SOxEM"]

PM10EM = model.constant("PM10EM")
PM10EM.equation = values["Total Emissions PM10EM"]

NOxEM = model.constant("NOxEM")
NOxEM.equation = values["Total Emissions NOxEM"]

External_Cost_Health_ImpactsEM = model.converter("External_Cost_Health_ImpactsEM")
External_Cost_Health_ImpactsEM.equation = (CNOx * NOxEM + CPM10 * PM10EM + CSOx * SOxEM)

#####------------------------------------------------------------------------------------------------------------------------

Maintenance_Time_EMML = model.constant("Maintenance_Time_EMML")
Maintenance_Time_EMML.equation = values["EMML Maintenance Time per Visit"]

Maintenance_Frequency_EMML = model.constant("Maintenance_Frequency_EMML")
Maintenance_Frequency_EMML.equation = values["EMML Maintenance Frequency"]

#####------------------------------------------------------------------------------------------------------------------------

Distance_Travelled_EMML = model.constant("Distance_Travelled_EMML")
Distance_Travelled_EMML.equation = values["EMML Annual Distance Travelled"]

#####------------------------------------------------------------------------------------------------------------------------

Health_Costs_of_Emissions_EMML = model.converter("Health_Costs_of_Emissions_EMML")
Health_Costs_of_Emissions_EMML.equation = Distance_Travelled_EMML * External_Cost_Health_ImpactsEM

#####------------------------------------------------------------------------------------------------------------------------

Energy_Consumption_EMML = model.converter("Energy_Consumption_EMML")
Energy_Consumption_EMML._function_string = (
    "lambda model, t: "
    "("
        "((model.memoize('Distance_Travelled_EMML', t) * model.memoize('Energy_Consumption_Rate_EMML', t)) "
        "* model.memoize('Electricity_Price', t)) "
        "if model.memoize('Energy_Consumption_Rate_EMML', t) > 0.0 else 0.0"
    ")"
)
Energy_Consumption_EMML.generate_function()

#####------------------------------------------------------------------------------------------------------------------------

GHG_Rate_EMML = model.constant("GHG_Rate_EMML")
GHG_Rate_EMML.equation = values["EMML Greenhouse Gas Emission Rate"]

#####------------------------------------------------------------------------------------------------------------------------

AQ_Rate_EMML = model.constant("AQ_Rate_EMML")
AQ_Rate_EMML.equation = values["EMML Air Quality Emission Rate"]

#####------------------------------------------------------------------------------------------------------------------------

Air_Quality_Cost_EMML = model.converter("Air_Quality_Cost_EMML")
Air_Quality_Cost_EMML.equation = (Distance_Travelled_EMML * AQ_Rate_EMML) * AQ_Price

#####------------------------------------------------------------------------------------------------------------------------

Noise_Cost_EMML = model.converter("Noise_Cost_EMML")
Noise_Cost_EMML.equation = Distance_Travelled_EMML* Noise_Cost_EM

#####------------------------------------------------------------------------------------------------------------------------

Greenhouse_Gas_Cost_EMML = model.converter("Greenhouse_Gas_Cost_EMML")
Greenhouse_Gas_Cost_EMML.equation = (Distance_Travelled_EMML * GHG_Rate_EMML) * GHG_Price

#####------------------------------------------------------------------------------------------------------------------------

Charging_Loss_Cost_EMML = model.converter("Charging_Loss_Cost_EMML")
Charging_Loss_Cost_EMML.equation = ((Charging_Frequency_EMML * 0.25) * Charging_Time_EMML) * Social_External_Cost

#####------------------------------------------------------------------------------------------------------------------------

Maintenance_Loss_Cost_EMML = model.converter("Maintenance_Loss_Cost_EMML")
Maintenance_Loss_Cost_EMML.equation = (Maintenance_Frequency_EMML * Maintenance_Time_EMML) * Social_External_Cost

#####------------------------------------------------------------------------------------------------------------------------

Average_Expected_Years_Kept_EMML = model.constant("Average_Expected_Years_Kept_EMML")
Average_Expected_Years_Kept_EMML.equation = values["Average Expected Vehicle Lifetime"]

#####------------------------------------------------------------------------------------------------------------------------

Social_Impact_Cost_EMML = model.converter("Social_Impact_Cost_EMML")
Social_Impact_Cost_EMML.equation = (Charging_Loss_Cost_EMML + Health_Costs_of_Emissions_EMML+ Maintenance_Loss_Cost_EMML) * Average_Expected_Years_Kept_EMML

#####------------------------------------------------------------------------------------------------------------------------

Environmental_Impact_Cost_EMML = model.converter("Environmental_Impact_Cost_EMML")
Environmental_Impact_Cost_EMML.equation = ( Air_Quality_Cost_EMML + Greenhouse_Gas_Cost_EMML + Noise_Cost_EMML) * Average_Expected_Years_Kept_EMML

#####------------------------------------------------------------------------------------------------------------------------

Depreciation_EMML = model.constant("Depreciation_EMML")
Depreciation_EMML.equation = values["EMML Depreciation Rate"]

#####------------------------------------------------------------------------------------------------------------------------

Battery_Resale_EMML = model.converter("Battery_Resale_EMML")
Battery_Resale_EMML.equation = (Average_Expected_Years_Kept_Battery_EMML * Depreciation_Battery_EMML) * Battery_Price_EMML

#####------------------------------------------------------------------------------------------------------------------------

Resale_Value_EMML = model.converter("Resale_Value_EMML")
Resale_Value_EMML._function_string = (
    "lambda model, t: ("
    "pow(1.0 - model.memoize('Depreciation_EMML', t), "
    "model.memoize('Average_Expected_Years_Kept_EMML', t))"
    "* model.memoize('EMML_Price', t)"
    "+ model.memoize('Battery_Resale_EMML', t)"
    ") / "
    "pow(1.0 + model.memoize('Discount_Rate', t), "
    "(model.memoize('Average_Expected_Years_Kept_EMML', t) - 1.0))"
)

Resale_Value_EMML.generate_function()

#"""**User (TCO of EMMN)**"""

Maintenance_Time_EMMN = model.constant("Maintenance_Time_EMMN")
Maintenance_Time_EMMN.equation = values["EMMN Maintenance Time per Visit"]

#####------------------------------------------------------------------------------------------------------------------------

Maintenance_Frequency_EMMN = model.constant("Maintenance_Frequency_EMMN")
Maintenance_Frequency_EMMN.equation = values["EMMN Maintenance Frequency"]

#####------------------------------------------------------------------------------------------------------------------------

Health_Costs_of_Emissions_EMMN = model.converter("Health_Costs_of_Emissions_EMMN")
Health_Costs_of_Emissions_EMMN.equation = Distance_Travelled_EMMN * External_Cost_Health_ImpactsEM

#####------------------------------------------------------------------------------------------------------------------------

Energy_Consumption_EMMN = model.converter("Energy_Consumption_EMMN")
Energy_Consumption_EMMN._function_string = (
    "lambda model, t: "
    "("
        "((model.memoize('Distance_Travelled_EMMN', t) * model.memoize('Energy_Consumption_Rate_EMMN', t)) "
        "* model.memoize('Electricity_Price', t)) "
        "if model.memoize('Energy_Consumption_Rate_EMMN', t) > 0.0 else 0.0"
    ")"
)
Energy_Consumption_EMMN.generate_function()

#####------------------------------------------------------------------------------------------------------------------------

GHG_Rate_EMMN = model.constant("GHG_Rate_EMMN")
GHG_Rate_EMMN.equation = values["EMMN Greenhouse Gas Emission Rate"]

#####------------------------------------------------------------------------------------------------------------------------

AQ_Rate_EMMN = model.constant("AQ_Rate_EMMN")
AQ_Rate_EMMN.equation = values["EMMN Air Quality Emission Rate"]

#####------------------------------------------------------------------------------------------------------------------------

Air_Quality_Cost_EMMN = model.converter("Air_Quality_Cost_EMMN")
Air_Quality_Cost_EMMN.equation = (Distance_Travelled_EMMN * AQ_Rate_EMMN) * AQ_Price

#####------------------------------------------------------------------------------------------------------------------------

Noise_Cost_EMMN = model.converter("Noise_Cost_EMMN")
Noise_Cost_EMMN.equation = Distance_Travelled_EMMN * Noise_Cost_EM

#####------------------------------------------------------------------------------------------------------------------------

Greenhouse_Gas_Cost_EMMN = model.converter("Greenhouse_Gas_Cost_EMMN")
Greenhouse_Gas_Cost_EMMN.equation = (Distance_Travelled_EMMN * GHG_Rate_EMMN) * GHG_Price

#####------------------------------------------------------------------------------------------------------------------------

Charging_Loss_Cost_EMMN = model.converter("Charging_Loss_Cost_EMMN")
Charging_Loss_Cost_EMMN.equation = ((Charging_Frequency_EMMN * 0.25) * Charging_Time_EMMN) * Social_External_Cost

#####------------------------------------------------------------------------------------------------------------------------

Maintenance_Loss_Cost_EMMN = model.converter("Maintenance_Loss_Cost_EMMN")
Maintenance_Loss_Cost_EMMN.equation = (Maintenance_Frequency_EMMN * Maintenance_Time_EMMN) * Social_External_Cost

#####------------------------------------------------------------------------------------------------------------------------

Average_Expected_Years_Kept_EMMN = model.constant("Average_Expected_Years_Kept_EMMN")
Average_Expected_Years_Kept_EMMN.equation = values["Average Expected Vehicle Lifetime"]

#####------------------------------------------------------------------------------------------------------------------------

Social_Impact_Cost_EMMN = model.converter("Social_Impact_Cost_EMMN")
Social_Impact_Cost_EMMN.equation = (Charging_Loss_Cost_EMMN + Health_Costs_of_Emissions_EMMN + Maintenance_Loss_Cost_EMMN) * Average_Expected_Years_Kept_EMMN

#####------------------------------------------------------------------------------------------------------------------------

Environmental_Impact_Cost_EMMN = model.converter("Environmental_Impact_Cost_EMMN")
Environmental_Impact_Cost_EMMN.equation = ( Air_Quality_Cost_EMMN + Greenhouse_Gas_Cost_EMMN + Noise_Cost_EMMN) * Average_Expected_Years_Kept_EMMN

#####------------------------------------------------------------------------------------------------------------------------

Depreciation_EMMN = model.constant("Depreciation_EMMN")
Depreciation_EMMN.equation = values["EMMN Depreciation Rate"]

#####------------------------------------------------------------------------------------------------------------------------

Battery_Resale_EMMN = model.converter("Battery_Resale_EMMN")
Battery_Resale_EMMN.equation = (Average_Expected_Years_Kept_Battery_EMMN * Depreciation_Battery_EMMN) * Battery_Price_EMMN

#####------------------------------------------------------------------------------------------------------------------------

Resale_Value_EMMN = model.converter("Resale_Value_EMMN")
Resale_Value_EMMN._function_string = (
    "lambda model, t: ("
    "pow(1.0 - model.memoize('Depreciation_EMMN', t), "
    "model.memoize('Average_Expected_Years_Kept_EMMN', t))"
    "* model.memoize('EMMN_Price', t)"
    "+ model.memoize('Battery_Resale_EMMN', t)"
    ") / "
    "pow(1.0 + model.memoize('Discount_Rate', t), "
    "(model.memoize('Average_Expected_Years_Kept_EMMN', t) - 1.0))"
)

Resale_Value_EMMN.generate_function()

#"""**Infrastructurer (Maintenance Network Support)**"""

Min_Powertrain_Prop_For_Non_OEM_Maintenance_Support_ICE = model.constant("Min_Powertrain_Prop_For_Non_OEM_Maintenance_Support_ICE")
Min_Powertrain_Prop_For_Non_OEM_Maintenance_Support_ICE.equation = values["ICE Minimum Powertrain Proportion for Non OEM Maintenance Support"]

Max_Powertrain_Prop_For_Non_OEM_Maintenance_Support_ICE = model.constant("Max_Powertrain_Prop_For_Non_OEM_Maintenance_Support_ICE")
Max_Powertrain_Prop_For_Non_OEM_Maintenance_Support_ICE.equation = values["ICE Maximum Powertrain Proportion for Non OEM Maintenance Support"]

Fraction_Attainment_of_Powertrain_Popularity_for_Non_OEM_Maintenance_Support_ICE = model.converter("Fraction_Attainment_of_Powertrain_Popularity_for_Non_OEM_Maintenance_Support_ICE")
Fraction_Attainment_of_Powertrain_Popularity_for_Non_OEM_Maintenance_Support_ICE._function_string = (
    "lambda model, t: ("
    "0.0 if model.memoize('ICE_Ownership_Proportion', t) <= "
    "model.memoize('Min_Powertrain_Prop_For_Non_OEM_Maintenance_Support_ICE', t) "
    "else ("
    "1.0 if model.memoize('ICE_Ownership_Proportion', t) > "
    "model.memoize('Max_Powertrain_Prop_For_Non_OEM_Maintenance_Support_ICE', t) "
    "else ("
    "("
    "model.memoize('ICE_Ownership_Proportion', t) - "
    "model.memoize('Min_Powertrain_Prop_For_Non_OEM_Maintenance_Support_ICE', t)"
    ") / ("
    "model.memoize('Max_Powertrain_Prop_For_Non_OEM_Maintenance_Support_ICE', t) - "
    "model.memoize('Min_Powertrain_Prop_For_Non_OEM_Maintenance_Support_ICE', t)"
    ")"
    ")"
    ")"
    ")"
)

Fraction_Attainment_of_Powertrain_Popularity_for_Non_OEM_Maintenance_Support_ICE.generate_function()

Total_Desired_or_Expected_pct_Non_OEM_Maintenance_Network_Serving_Powertrain_ICE = model.converter("Total_Desired_or_Expected_pct_Non_OEM_Maintenance_Network_Serving_Powertrain_ICE")
Total_Desired_or_Expected_pct_Non_OEM_Maintenance_Network_Serving_Powertrain_ICE.equation = sd.lookup(Fraction_Attainment_of_Powertrain_Popularity_for_Non_OEM_Maintenance_Support_ICE, "Powertrain_Proportion_to_Non_OEM_Maintenance_Infrastructure")*100

Initial_pct_Non_OEM_Maintenance_Network_Serving_Powertrain_ICE = \
    model.converter("Initial_pct_Non_OEM_Maintenance_Network_Serving_Powertrain_ICE")
Initial_pct_Non_OEM_Maintenance_Network_Serving_Powertrain_ICE._function_string = (
    "lambda model, t: "
    "model.memoize("
    "'Total_Desired_or_Expected_pct_Non_OEM_Maintenance_Network_Serving_Powertrain_ICE', "
    "2016.0)"
)
Initial_pct_Non_OEM_Maintenance_Network_Serving_Powertrain_ICE.generate_function()

Non_OEM_Maintenance_Infrastructure_Investment_Planning_Development_Delay = model.constant("Non_OEM_Maintenance_Infrastructure_Investment_Planning_Development_Delay")
Non_OEM_Maintenance_Infrastructure_Investment_Planning_Development_Delay.equation = values["Non OEM Maintenance Infrastructure Investment Planning Development Delay"]

Increase_in_pct_Non_OEM_ICE = model.flow("Increase_in_pct_Non_OEM_ICE")
Actual_pct_Non_OEM_ICE = model.stock("Actual_pct_Non_OEM_ICE")
Actual_pct_Non_OEM_ICE.initial_value = Initial_pct_Non_OEM_Maintenance_Network_Serving_Powertrain_ICE
Actual_pct_Non_OEM_ICE.equation = Increase_in_pct_Non_OEM_ICE

Increase_in_pct_Non_OEM_ICE.equation = (Total_Desired_or_Expected_pct_Non_OEM_Maintenance_Network_Serving_Powertrain_ICE - Actual_pct_Non_OEM_ICE) / Non_OEM_Maintenance_Infrastructure_Investment_Planning_Development_Delay

#####------------------------------------------------------------------------------------------------------------------------

Min_Powertrain_Prop_For_Non_OEM_Maintenance_Support_EMML = model.constant("Min_Powertrain_Prop_For_Non_OEM_Maintenance_Support_EMML")
Min_Powertrain_Prop_For_Non_OEM_Maintenance_Support_EMML.equation = values["EMML Minimum Powertrain Proportion for Non OEM Maintenance Support"]

Max_Powertrain_Prop_For_Non_OEM_Maintenance_Support_EMML = model.constant("Max_Powertrain_Prop_For_Non_OEM_Maintenance_Support_EMML")
Max_Powertrain_Prop_For_Non_OEM_Maintenance_Support_EMML.equation = values["EMML Maximum Powertrain Proportion for Non OEM Maintenance Support"]

Fraction_Attainment_of_Powertrain_Popularity_for_Non_OEM_Maintenance_Support_EMML = model.converter("Fraction_Attainment_of_Powertrain_Popularity_for_Non_OEM_Maintenance_Support_EMML")
Fraction_Attainment_of_Powertrain_Popularity_for_Non_OEM_Maintenance_Support_EMML._function_string = (
    "lambda model, t: ("
    "0.0 if model.memoize('EMML_Ownership_Proportion', t) <= "
    "model.memoize('Min_Powertrain_Prop_For_Non_OEM_Maintenance_Support_EMML', t) "
    "else ("
    "1.0 if model.memoize('EMML_Ownership_Proportion', t) > "
    "model.memoize('Max_Powertrain_Prop_For_Non_OEM_Maintenance_Support_EMML', t) "
    "else ("
    "("
    "model.memoize('EMML_Ownership_Proportion', t) - "
    "model.memoize('Min_Powertrain_Prop_For_Non_OEM_Maintenance_Support_EMML', t)"
    ") / ("
    "model.memoize('Max_Powertrain_Prop_For_Non_OEM_Maintenance_Support_EMML', t) - "
    "model.memoize('Min_Powertrain_Prop_For_Non_OEM_Maintenance_Support_EMML', t)"
    ")"
    ")"
    ")"
    ")"
)

Fraction_Attainment_of_Powertrain_Popularity_for_Non_OEM_Maintenance_Support_EMML.generate_function()

Total_Desired_or_Expected_pct_Non_OEM_Maintenance_Network_Serving_Powertrain_EMML = model.converter("Total_Desired_or_Expected_pct_Non_OEM_Maintenance_Network_Serving_Powertrain_EMML")
Total_Desired_or_Expected_pct_Non_OEM_Maintenance_Network_Serving_Powertrain_EMML.equation = sd.lookup(Fraction_Attainment_of_Powertrain_Popularity_for_Non_OEM_Maintenance_Support_EMML, "Powertrain_Proportion_to_Non_OEM_Maintenance_Infrastructure")*100

Initial_pct_Non_OEM_Maintenance_Network_Serving_Powertrain_EMML = \
    model.converter("Initial_pct_Non_OEM_Maintenance_Network_Serving_Powertrain_EMML")
Initial_pct_Non_OEM_Maintenance_Network_Serving_Powertrain_EMML._function_string = (
    "lambda model, t: "
    "model.memoize("
    "'Total_Desired_or_Expected_pct_Non_OEM_Maintenance_Network_Serving_Powertrain_EMML', "
    "2016.0)"
)
Initial_pct_Non_OEM_Maintenance_Network_Serving_Powertrain_EMML.generate_function()

Increase_in_pct_Non_OEM_EMML = model.flow("Increase_in_pct_Non_OEM_EMML")
Actual_pct_Non_OEM_EMML = model.stock("Actual_pct_Non_OEM_EMML")
Actual_pct_Non_OEM_EMML.initial_value = Initial_pct_Non_OEM_Maintenance_Network_Serving_Powertrain_EMML
Actual_pct_Non_OEM_EMML.equation = Increase_in_pct_Non_OEM_EMML

Increase_in_pct_Non_OEM_EMML.equation = (Total_Desired_or_Expected_pct_Non_OEM_Maintenance_Network_Serving_Powertrain_EMML - Actual_pct_Non_OEM_EMML) / Non_OEM_Maintenance_Infrastructure_Investment_Planning_Development_Delay

#####------------------------------------------------------------------------------------------------------------------------

Min_Powertrain_Prop_For_Non_OEM_Maintenance_Support_EMMN = model.constant("Min_Powertrain_Prop_For_Non_OEM_Maintenance_Support_EMMN")
Min_Powertrain_Prop_For_Non_OEM_Maintenance_Support_EMMN.equation = values["EMMN Minimum Powertrain Proportion for Non OEM Maintenance Support"]

Max_Powertrain_Prop_For_Non_OEM_Maintenance_Support_EMMN = model.constant("Max_Powertrain_Prop_For_Non_OEM_Maintenance_Support_EMMN")
Max_Powertrain_Prop_For_Non_OEM_Maintenance_Support_EMMN.equation = values["EMMN Maximum Powertrain Proportion for Non OEM Maintenance Support"]

Fraction_Attainment_of_Powertrain_Popularity_for_Non_OEM_Maintenance_Support_EMMN = model.converter("Fraction_Attainment_of_Powertrain_Popularity_for_Non_OEM_Maintenance_Support_EMMN")
Fraction_Attainment_of_Powertrain_Popularity_for_Non_OEM_Maintenance_Support_EMMN._function_string = (
    "lambda model, t: ("
    "0.0 if model.memoize('EMMN_Ownership_Proportion', t) <= "
    "model.memoize('Min_Powertrain_Prop_For_Non_OEM_Maintenance_Support_EMMN', t) "
    "else ("
    "1.0 if model.memoize('EMMN_Ownership_Proportion', t) > "
    "model.memoize('Max_Powertrain_Prop_For_Non_OEM_Maintenance_Support_EMMN', t) "
    "else ("
    "("
    "model.memoize('EMMN_Ownership_Proportion', t) - "
    "model.memoize('Min_Powertrain_Prop_For_Non_OEM_Maintenance_Support_EMMN', t)"
    ") / ("
    "model.memoize('Max_Powertrain_Prop_For_Non_OEM_Maintenance_Support_EMMN', t) - "
    "model.memoize('Min_Powertrain_Prop_For_Non_OEM_Maintenance_Support_EMMN', t)"
    ")"
    ")"
    ")"
    ")"
)

Fraction_Attainment_of_Powertrain_Popularity_for_Non_OEM_Maintenance_Support_EMMN.generate_function()

model.points["Powertrain_Proportion_to_Non_OEM_Maintenance_Infrastructure"] = [
    [0.0, 0.0],
    [0.114286, 0.022556],
    [0.218182, 0.067669],
    [0.298701, 0.12782],
    [0.363636, 0.214286],
    [0.415584, 0.315789],
    [0.54026, 0.680451],
    [0.649351, 0.879699],
    [0.755844, 0.951128],
    [0.841558, 0.981203],
    [1.0, 1.0],
]

Total_Desired_or_Expected_pct_Non_OEM_Maintenance_Network_Serving_Powertrain_EMMN = model.converter("Total_Desired_or_Expected_pct_Non_OEM_Maintenance_Network_Serving_Powertrain_EMMN")
Total_Desired_or_Expected_pct_Non_OEM_Maintenance_Network_Serving_Powertrain_EMMN.equation = sd.lookup(Fraction_Attainment_of_Powertrain_Popularity_for_Non_OEM_Maintenance_Support_EMMN, "Powertrain_Proportion_to_Non_OEM_Maintenance_Infrastructure")*100

Initial_pct_Non_OEM_Maintenance_Network_Serving_Powertrain_EMMN = \
    model.converter("Initial_pct_Non_OEM_Maintenance_Network_Serving_Powertrain_EMMN")
Initial_pct_Non_OEM_Maintenance_Network_Serving_Powertrain_EMMN._function_string = (
    "lambda model, t: "
    "model.memoize("
    "'Total_Desired_or_Expected_pct_Non_OEM_Maintenance_Network_Serving_Powertrain_EMMN', "
    "2016.0)"
)
Initial_pct_Non_OEM_Maintenance_Network_Serving_Powertrain_EMMN.generate_function()

Increase_in_pct_Non_OEM_EMMN = model.flow("Increase_in_pct_Non_OEM_EMMN")
Actual_pct_Non_OEM_EMMN = model.stock("Actual_pct_Non_OEM_EMMN")
Actual_pct_Non_OEM_EMMN.initial_value = Initial_pct_Non_OEM_Maintenance_Network_Serving_Powertrain_EMMN
Actual_pct_Non_OEM_EMMN.equation = Increase_in_pct_Non_OEM_EMMN

Increase_in_pct_Non_OEM_EMMN.equation = ((Total_Desired_or_Expected_pct_Non_OEM_Maintenance_Network_Serving_Powertrain_EMMN - Actual_pct_Non_OEM_EMMN) / Non_OEM_Maintenance_Infrastructure_Investment_Planning_Development_Delay)

#####------------------------------------------------------------------------------------------------------------------------

Sensitivity_of_maintenance_repair_costs_to_competition_from_Non_OEM_providers = model.constant("Sensitivity_of_maintenance_repair_costs_to_competition_from_Non_OEM_providers")
Sensitivity_of_maintenance_repair_costs_to_competition_from_Non_OEM_providers.equation = values["Sensitivity of Maintenance Repair Costs to Competition from Non OEM Providers"]

Max_Maintenance_Repair_Cost_Reduction_from_Competition = model.constant("Max_Maintenance_Repair_Cost_Reduction_from_Competition")
Max_Maintenance_Repair_Cost_Reduction_from_Competition.equation = values["Max Maintenance Repair Cost Reduction from Competition"]

Base_Maintenance_Repair_CostsICE = model.constant("Base_Maintenance_Repair_CostsICE")
Base_Maintenance_Repair_CostsICE.equation = values["Base Maintenance Repair Costs ICE"]

Base_Maintenance_Repair_CostsEM = model.constant("Base_Maintenance_Repair_CostsEM")
Base_Maintenance_Repair_CostsEM.equation = values["Base Maintenance Repair Costs EM"]

#####------------------------------------------------------------------------------------------------------------------------

Maintenance_Repair_Costs_Reduction_from_Competition_ICE = model.converter(
    "Maintenance_Repair_Costs_Reduction_from_Competition_ICE"
)

Maintenance_Repair_Costs_Reduction_from_Competition_ICE._function_string = (
    "lambda model, t: "
    "model.memoize('Max_Maintenance_Repair_Cost_Reduction_from_Competition', t) "
    "* pow("
    "model.memoize('Actual_pct_Non_OEM_ICE', t) / 100.0, "
    "model.memoize('Sensitivity_of_maintenance_repair_costs_to_competition_from_Non_OEM_providers', t)"
    ")"
)

Maintenance_Repair_Costs_Reduction_from_Competition_ICE.generate_function()

#####------------------------------------------------------------------------------------------------------------------------

Maintenance_Repair_Costs_Reduction_from_Competition_EMML = model.converter(
    "Maintenance_Repair_Costs_Reduction_from_Competition_EMML"
)

Maintenance_Repair_Costs_Reduction_from_Competition_EMML._function_string = (
    "lambda model, t: "
    "model.memoize('Max_Maintenance_Repair_Cost_Reduction_from_Competition', t) "
    "* pow("
    "model.memoize('Actual_pct_Non_OEM_EMML', t) / 100.0, "
    "model.memoize('Sensitivity_of_maintenance_repair_costs_to_competition_from_Non_OEM_providers', t)"
    ")"
)

Maintenance_Repair_Costs_Reduction_from_Competition_EMML.generate_function()

#####------------------------------------------------------------------------------------------------------------------------

Maintenance_Repair_Costs_Reduction_from_Competition_EMMN = model.converter(
    "Maintenance_Repair_Costs_Reduction_from_Competition_EMMN"
)

Maintenance_Repair_Costs_Reduction_from_Competition_EMMN._function_string = (
    "lambda model, t: "
    "model.memoize('Max_Maintenance_Repair_Cost_Reduction_from_Competition', t) "
    "* pow("
    "model.memoize('Actual_pct_Non_OEM_EMMN', t) / 100.0, "
    "model.memoize('Sensitivity_of_maintenance_repair_costs_to_competition_from_Non_OEM_providers', t)"
    ")"
)

Maintenance_Repair_Costs_Reduction_from_Competition_EMMN.generate_function()

Yearly_Tax_Cost_ICE = model.converter("Yearly_Tax_Cost_ICE")
Yearly_Tax_Cost_ICE._function_string = (
    "lambda model, t: ("
    "model.memoize('PKB_pct', t) * "
    "pow(1.0 - model.memoize('Depreciation_ICE', t), (t - 2016.0 - 1.0)) * "
    "model.memoize('Resale_Value_ICE', t)"
    ") + model.memoize('SWDKLLJ', t)"
)

Yearly_Tax_Cost_ICE.generate_function()

Insurance_Cost_ICE = model.converter("Insurance_Cost_ICE")
Insurance_Cost_ICE.equation = 0.035 * ICE_Price

Maintenance_Cost_ICE = model.converter("Maintenance_Cost_ICE")
Maintenance_Cost_ICE.equation = Base_Maintenance_Repair_CostsICE * (1 - Maintenance_Repair_Costs_Reduction_from_Competition_ICE)

Operational_Cost_ICE = model.converter("Operational_Cost_ICE")
Operational_Cost_ICE.equation = ((Energy_Consumption_ICE + Insurance_Cost_ICE + Maintenance_Cost_ICE + Yearly_Tax_Cost_ICE) * Average_Expected_Years_Kept_ICE) + Tax_Cost_per_5_Years

Purchase_Cost_ICE = model.converter("Purchase_Cost_ICE")
Purchase_Cost_ICE.equation = ICE_Price + (Purchase_Tax*ICE_Price)

Life_Cycle_Cost_ICE = model.converter("Life_Cycle_Cost_ICE")
Life_Cycle_Cost_ICE.equation = (Purchase_Cost_ICE + Operational_Cost_ICE - Resale_Value_ICE)

ICE_Ownership_Costs = model.converter("ICE_Ownership_Costs")
ICE_Ownership_Costs._function_string = (
    "lambda model, t: "
    "("
        "model.memoize('Life_Cycle_Cost_ICE', t) + "
        "model.memoize('Social_Impact_Cost_ICE', t) + "
        "model.memoize('Environmental_Impact_Cost_ICE', t)"
    ") / 1e+06"
)

ICE_Ownership_Costs.generate_function()

#####------------------------------------------------------------------------------------------------------------------------

Purchase_Cost_After_Subsidies_EMML = model.converter("Purchase_Cost_After_Subsidies_EMML")
Purchase_Cost_After_Subsidies_EMML.equation = ((EMML_Price - EM_Price_Subsidy) + (Purchase_Tax_Subsidy_pct * EMML_Price))

Yearly_Tax_Cost_EMML_After_Subsidies = model.converter("Yearly_Tax_Cost_EMML_After_Subsidies")
Yearly_Tax_Cost_EMML_After_Subsidies._function_string = (
    "lambda model, t: ("
    "model.memoize('PKB_pct_Subsidy', t) * "
    "pow(1.0 - model.memoize('Depreciation_EMML', t), (t - 2023.0 - 1.0)) * "
    "model.memoize('Resale_Value_EMML', t)"
    ") + model.memoize('SWDKLLJ', t)"
)

Yearly_Tax_Cost_EMML_After_Subsidies.generate_function()

Insurance_Cost_EMML = model.converter("Insurance_Cost_EMML")
Insurance_Cost_EMML.equation = 0.015 * EMML_Price

Maintenance_Cost_EMML = model.converter("Maintenance_Cost_EMML")
Maintenance_Cost_EMML.equation = Base_Maintenance_Repair_CostsEM * (1 - Maintenance_Repair_Costs_Reduction_from_Competition_EMML)

Operational_Cost_EMML = model.converter("Operational_Cost_EMML")
Operational_Cost_EMML.equation = ((Energy_Consumption_EMML + Insurance_Cost_EMML + Maintenance_Cost_EMML + Yearly_Tax_Cost_EMML_After_Subsidies) * Average_Expected_Years_Kept_EMML) + Tax_Cost_per_5_Years

Life_Cycle_Cost_EMML = model.converter("Life_Cycle_Cost_EMML")
Life_Cycle_Cost_EMML.equation = (Purchase_Cost_After_Subsidies_EMML + Operational_Cost_EMML - Resale_Value_EMML)

EMML_Ownership_Costs = model.converter("EMML_Ownership_Costs")
EMML_Ownership_Costs._function_string = (
    "lambda model, t: 0.0 if t < 2024.0 else "
    "((model.memoize('Life_Cycle_Cost_EMML', t) + "
    "model.memoize('Social_Impact_Cost_EMML', t) + "
    "model.memoize('Environmental_Impact_Cost_EMML', t)) / 1e+06)"
)
EMML_Ownership_Costs.generate_function()

#####------------------------------------------------------------------------------------------------------------------------

Purchase_Cost_After_Subsidies_EMMN = model.converter("Purchase_Cost_After_Subsidies_EMMN")
Purchase_Cost_After_Subsidies_EMMN.equation = ((EMMN_Price - EM_Price_Subsidy) + (Purchase_Tax_Subsidy_pct * EMMN_Price))

Yearly_Tax_Cost_EMMN_After_Subsidies = model.converter("Yearly_Tax_Cost_EMMN_After_Subsidies")
Yearly_Tax_Cost_EMMN_After_Subsidies._function_string = (
    "lambda model, t: ("
    "model.memoize('PKB_pct_Subsidy', t) * "
    "pow(1.0 - model.memoize('Depreciation_EMMN', t), (t- 2016.0 - 1.0)) * "
    "model.memoize('Resale_Value_EMMN', t)"
    ") + model.memoize('SWDKLLJ', t)"
)

Yearly_Tax_Cost_EMMN_After_Subsidies.generate_function()

Insurance_Cost_EMMN = model.converter("Insurance_Cost_EMMN")
Insurance_Cost_EMMN.equation = 0.015 * EMMN_Price

Maintenance_Cost_EMMN = model.converter("Maintenance_Cost_EMMN")
Maintenance_Cost_EMMN.equation = Base_Maintenance_Repair_CostsEM * (1 - Maintenance_Repair_Costs_Reduction_from_Competition_EMMN)

Operational_Cost_EMMN = model.converter("Operational_Cost_EMMN")
Operational_Cost_EMMN.equation = ((Energy_Consumption_EMMN + Insurance_Cost_EMMN + Maintenance_Cost_EMMN+ Yearly_Tax_Cost_EMMN_After_Subsidies) * Average_Expected_Years_Kept_EMMN) + Tax_Cost_per_5_Years

Life_Cycle_Cost_EMMN = model.converter("Life_Cycle_Cost_EMMN")
Life_Cycle_Cost_EMMN.equation = (Purchase_Cost_After_Subsidies_EMMN + Operational_Cost_EMMN - Resale_Value_EMMN)

EMMN_Ownership_Costs = model.converter("EMMN_Ownership_Costs")
EMMN_Ownership_Costs._function_string = (
    "lambda model, t: 0.0 if t < 2017.0 else "
    "((model.memoize('Life_Cycle_Cost_EMMN', t) + "
    "model.memoize('Social_Impact_Cost_EMMN', t) + "
    "model.memoize('Environmental_Impact_Cost_EMMN', t)) / 1e+06)"
)
EMMN_Ownership_Costs.generate_function()

#####------------------------------------------------------------------------------------------------------------------------

Yearly_Tax_Cost_EMML_Without_Subsidies = model.converter("Yearly_Tax_Cost_EMML_Without_Subsidies")
Yearly_Tax_Cost_EMML_Without_Subsidies._function_string = (
    "lambda model, t: ("
    "model.memoize('PKB_pct', t) * "
    "pow(1.0 - model.memoize('Depreciation_EMML', t), (t - 2023.0 - 1.0)) * "
    "model.memoize('Resale_Value_EMML', t)"
    ") + model.memoize('SWDKLLJ', t)"
)

Yearly_Tax_Cost_EMML_Without_Subsidies.generate_function()

Yearly_Tax_Cost_Gap_EMML = model.converter("Yearly_Tax_Cost_Gap_EMML")
Yearly_Tax_Cost_Gap_EMML.equation = (Yearly_Tax_Cost_EMML_Without_Subsidies - Yearly_Tax_Cost_EMML_After_Subsidies)

Purchase_Cost_Without_Subsidies_EMML = model.converter("Purchase_Cost_Without_Subsidies_EMML")
Purchase_Cost_Without_Subsidies_EMML.equation = (EMML_Price * Purchase_Tax) + EMML_Price

Purchase_Cost_Gap_EMML = model.converter("Purchase_Cost_Gap_EMML")
Purchase_Cost_Gap_EMML.equation = (Purchase_Cost_Without_Subsidies_EMML - Purchase_Cost_After_Subsidies_EMML)

Total_Subsidy_EMML = model.converter("Total_Subsidy_EMML")
Total_Subsidy_EMML._function_string = (
    "lambda model, t: "
    "("
        "(model.memoize('Purchase_Cost_Gap_EMML', t) + model.memoize('Yearly_Tax_Cost_Gap_EMML', t)) / "
        "(model.memoize('Purchase_Cost_Without_Subsidies_EMML', t) + model.memoize('Yearly_Tax_Cost_EMML_Without_Subsidies', t))"
    ") "
    "if (model.memoize('Purchase_Cost_Without_Subsidies_EMML', t) + model.memoize('Yearly_Tax_Cost_EMML_Without_Subsidies', t)) > 0.000001 else 0.0"
)

Total_Subsidy_EMML.generate_function()

#####------------------------------------------------------------------------------------------------------------------------

Yearly_Tax_Cost_EMMN_Without_Subsidies = model.converter("Yearly_Tax_Cost_EMMN_Without_Subsidies")
Yearly_Tax_Cost_EMMN_Without_Subsidies._function_string = (
    "lambda model, t: ("
    "model.memoize('PKB_pct', t) * "
    "pow(1.0 - model.memoize('Depreciation_EMMN', t), (t - 2016.0 - 1.0)) * "
    "model.memoize('Resale_Value_EMMN', t)"
    ") + model.memoize('SWDKLLJ', t)"
)

Yearly_Tax_Cost_EMMN_Without_Subsidies.generate_function()

Yearly_Tax_Cost_Gap_EMMN = model.converter("Yearly_Tax_Cost_Gap_EMMN")
Yearly_Tax_Cost_Gap_EMMN.equation = (Yearly_Tax_Cost_EMMN_Without_Subsidies - Yearly_Tax_Cost_EMMN_After_Subsidies)

Purchase_Cost_Without_Subsidies_EMMN = model.converter("Purchase_Cost_Without_Subsidies_EMMN")
Purchase_Cost_Without_Subsidies_EMMN.equation = (EMMN_Price * Purchase_Tax) + EMMN_Price

Purchase_Cost_Gap_EMMN = model.converter("Purchase_Cost_Gap_EMMN")
Purchase_Cost_Gap_EMMN.equation = (Purchase_Cost_Without_Subsidies_EMMN - Purchase_Cost_After_Subsidies_EMMN)

Total_Subsidy_EMMN = model.converter("Total_Subsidy_EMMN")
Total_Subsidy_EMMN._function_string = (
    "lambda model, t: "
    "("
        "(model.memoize('Purchase_Cost_Gap_EMMN', t) + model.memoize('Yearly_Tax_Cost_Gap_EMMN', t)) / "
        "(model.memoize('Purchase_Cost_Without_Subsidies_EMMN', t) + model.memoize('Yearly_Tax_Cost_EMMN_Without_Subsidies', t))"
    ") "
    "if (model.memoize('Purchase_Cost_Without_Subsidies_EMMN', t) + model.memoize('Yearly_Tax_Cost_EMMN_Without_Subsidies', t)) > 0.0 else 0.0"
)

Total_Subsidy_EMMN.generate_function()

#"""**Manufacrturer (Marketing)**"""

Marketing_Effort_From_Subsidies_EMML = model.converter("Marketing_Effort_From_Subsidies_EMML")
Marketing_Effort_From_Subsidies_EMML._function_string = (
    "lambda model, t: min(1.0, max(0.0, "
    "model.memoize('Base_Marketing_Modifier_for_Subsidies', t) * "
    "pow("
    "(model.memoize('Total_Subsidy_EMML', t) / "
    " model.memoize('Base_Subsidies_for_Marketing', t)), "
    "model.memoize('Sensitivity_of_Marketing_Effort_to_Subsidies', t)"
    ")))"
)

Marketing_Effort_From_Subsidies_EMML.generate_function()

model.points["Marketing_Efforts_for_EMML_Launch"] = [
        [0.0, 0.0],
        [2016.0, 0.0],
        [2017.0, 0.0],
        [2018, 0.0],
        [2019, 0.0],
        [2020, 0.0],
        [2021, 0.0],
        [2022, 0.0],
        [2023, 1.0],
        [2024, 1.0],
        [2025, 1.0],
        [2026, 1.0],
        [2027, 1.0],
        [2028, 0.0],
        [2029, 0.0],
        [2030, 0.0],
]

Marketing_Efforts_for_EMML_Launch = model.converter("Marketing_Efforts_for_EMML_Launch")
Marketing_Efforts_for_EMML_Launch.equation = sd.lookup(sd.time(), "Marketing_Efforts_for_EMML_Launch")

Marketing_Effort_EMML = model.converter("Marketing_Effort_EMML")
Marketing_Effort_EMML._function_string = (
    "lambda model, t: ("
    "model.memoize('Marketing_Efforts_for_EMML_Launch', t)"
    " if model.memoize('Marketing_Efforts_for_EMML_Launch', t) > 0.0 "
    " else model.memoize('Marketing_Effort_From_Subsidies_EMML', t)"
    ")"
)

Marketing_Effort_EMML.generate_function()

#####------------------------------------------------------------------------------------------------------------------------

Base_Marketing_Modifier_for_Subsidies = model.constant("Base_Marketing_Modifier_for_Subsidies")
Base_Marketing_Modifier_for_Subsidies.equation = values["Base Marketing Modifier for Subsidies"]

Sensitivity_of_Marketing_Effort_to_Subsidies = model.constant("Sensitivity_of_Marketing_Effort_to_Subsidies")
Sensitivity_of_Marketing_Effort_to_Subsidies.equation = values["Sensitivity of Marketing Effort to Subsidies"]

Base_Subsidies_for_Marketing = model.constant("Base_Subsidies_for_Marketing")
Base_Subsidies_for_Marketing.equation = values["Base Subsidies for Marketing"]

Marketing_Effort_From_Subsidies_EMMN = model.converter("Marketing_Effort_From_Subsidies_EMMN")
Marketing_Effort_From_Subsidies_EMMN._function_string = (
    "lambda model, t: min(1.0, max(0.0, "
    "model.memoize('Base_Marketing_Modifier_for_Subsidies', t) * "
    "pow("
    "(model.memoize('Total_Subsidy_EMMN', t) / "
    " model.memoize('Base_Subsidies_for_Marketing', t)), "
    "model.memoize('Sensitivity_of_Marketing_Effort_to_Subsidies', t)"
    ")))"
)

Marketing_Effort_From_Subsidies_EMMN.generate_function()

model.points["Marketing_Efforts_for_EMMN_Launch"] = [
        [0.0, 0.0],
        [2016.0, 1.0],
        [2017.0, 1.0],
        [2018, 1.0],
        [2019, 1.0],
        [2020, 1.0],
        [2021, 0.0],
        [2022, 0.0],
        [2023, 0.0],
        [2024, 0.0],
        [2025, 0.0],
        [2026, 0.0],
        [2027, 0.0],
        [2028, 0.0],
        [2029, 0.0],
        [2030, 0.0],
]

Marketing_Efforts_for_EMMN_Launch = model.converter("Marketing_Efforts_for_EMMN_Launch")
Marketing_Efforts_for_EMMN_Launch.equation = sd.lookup(sd.time(), "Marketing_Efforts_for_EMMN_Launch")

Marketing_Effort_EMMN = model.converter("Marketing_Effort_EMMN")
Marketing_Effort_EMMN._function_string = (
    "lambda model, t: ("
    "model.memoize('Marketing_Efforts_for_EMMN_Launch', t)"
    " if model.memoize('Marketing_Efforts_for_EMMN_Launch', t) > 0.0 "
    " else model.memoize('Marketing_Effort_From_Subsidies_EMMN', t)"
    ")"
)

Marketing_Effort_EMMN.generate_function()

#"""**User (Cost Impact)**"""

Maximum_Cost_Difference_Reference = model.constant("Maximum_Cost_Difference_Reference")
Maximum_Cost_Difference_Reference.equation = values["Maximum Cost Difference Reference"]

Affordability_Sensitivity_to_GDP = model.constant("Affordability_Sensitivity_to_GDP")
Affordability_Sensitivity_to_GDP.equation = values["Affordability Sensitivity to GDP"]

Maximum_Cost_Difference = model.converter("Maximum_Cost_Difference")
Maximum_Cost_Difference._function_string = (
    "lambda model, t: 1.0 + ("
    "model.memoize('Maximum_Cost_Difference_Reference', t) * "
    "pow("
    "model.memoize('GDP_per_capita_ratio', t), "
    "model.memoize('Affordability_Sensitivity_to_GDP', t)"
    "))"
)

Maximum_Cost_Difference.generate_function()

Minimum_Cost_Difference_Reference = model.constant("Minimum_Cost_Difference_Reference")
Minimum_Cost_Difference_Reference.equation = values["Minimum Cost Difference Reference"]


Minimum_Cost_Difference = model.converter("Minimum_Cost_Difference")
Minimum_Cost_Difference._function_string = (
    "lambda model, t: 1.0 + ("
    "model.memoize('Minimum_Cost_Difference_Reference', t) * "
    "pow("
    "model.memoize('GDP_per_capita_ratio', t), "
    "model.memoize('Affordability_Sensitivity_to_GDP', t)"
    "))"
)

Minimum_Cost_Difference.generate_function()

model.points["Cost_to_WtC_Ratio"] = [
        [0.0, 1.0],
        [0.1, 0.99],
        [0.2, 0.90],
        [0.3, 0.70],
        [0.45, 0.50],
        [0.6, 0.30],
        [0.7, 0.20],
        [0.8, 0.131579],
        [0.9, 0.052632],
        [0.95, 0.022556],
        [1.0, 0.0],
]

Value_Assigned_to_EM_Environmental_Impact = model.constant("Value_Assigned_to_EM_Environmental_Impact")
Value_Assigned_to_EM_Environmental_Impact.equation = values["Value Assigned to EM Environmental Impact"]

#####------------------------------------------------------------------------------------------------------------------------

Average_Cost_of_Ownership_with_EMML_Existence = model.converter("Average_Cost_of_Ownership_with_EMML_Existence")

Average_Cost_of_Ownership_with_EMML_Existence._function_string = (
    "lambda model, t: "
    "((model.memoize('ICE_Ownership_Costs', t) + model.memoize('EMMN_Ownership_Costs', t)) / 2) "
    "if t < 2024.0 else "
    "((model.memoize('ICE_Ownership_Costs', t) + model.memoize('EMML_Ownership_Costs', t) + model.memoize('EMMN_Ownership_Costs', t)) / 3)"
)

Average_Cost_of_Ownership_with_EMML_Existence.generate_function()

Proportion_of_EMML_Ownership_Cost_to_Average_Ownership_Cost_with_EMML_Existence = model.converter("Proportion_of_EMML_Ownership_Cost_to_Average_Ownership_Cost_with_EMML_Existence")
Proportion_of_EMML_Ownership_Cost_to_Average_Ownership_Cost_with_EMML_Existence._function_string = (
    "lambda model, t: "
    "((model.memoize('EMML_Ownership_Costs', t) / model.memoize('Average_Cost_of_Ownership_with_EMML_Existence', t)) "
    "if model.memoize('Average_Cost_of_Ownership_with_EMML_Existence', t) != 0.0 else 0.0) "
    "* (1 + model.memoize('Value_Assigned_to_EM_Environmental_Impact', t))"
)
Proportion_of_EMML_Ownership_Cost_to_Average_Ownership_Cost_with_EMML_Existence.generate_function()

Cost_Impact_EMML = model.converter("Cost_Impact_EMML")
normalized_ratio3 = sd.max(0.0,
                       (Proportion_of_EMML_Ownership_Cost_to_Average_Ownership_Cost_with_EMML_Existence - Minimum_Cost_Difference) / (Maximum_Cost_Difference - Minimum_Cost_Difference) if (Maximum_Cost_Difference - Minimum_Cost_Difference) != 0 else 0.0
                      )
Cost_Impact_EMML.equation = sd.lookup(normalized_ratio3, "Cost_to_WtC_Ratio")

#####------------------------------------------------------------------------------------------------------------------------

Proportion_of_EMMN_Ownership_Cost_to_Average_Ownership_Cost_with_EMML_Existence = model.converter("Proportion_of_EMMN_Ownership_Cost_to_Average_Ownership_Cost_with_EMML_Existence")
Proportion_of_EMMN_Ownership_Cost_to_Average_Ownership_Cost_with_EMML_Existence._function_string = (
    "lambda model, t: "
    "((model.memoize('EMMN_Ownership_Costs', t) / model.memoize('Average_Cost_of_Ownership_with_EMML_Existence', t)) "
    "if model.memoize('Average_Cost_of_Ownership_with_EMML_Existence', t) != 0.0 else 0.0) "
    "* (1 + model.memoize('Value_Assigned_to_EM_Environmental_Impact', t))"
)
Proportion_of_EMMN_Ownership_Cost_to_Average_Ownership_Cost_with_EMML_Existence.generate_function()

Cost_Impact_EMMN_with_EMML_Existence = model.converter("Cost_Impact_EMMN_with_EMML_Existence")

normalized_ratio2 = sd.max(0.0,
                       (Proportion_of_EMMN_Ownership_Cost_to_Average_Ownership_Cost_with_EMML_Existence - Minimum_Cost_Difference) / (Maximum_Cost_Difference - Minimum_Cost_Difference) if (Maximum_Cost_Difference - Minimum_Cost_Difference) != 0 else 0.0
                      )
Cost_Impact_EMMN_with_EMML_Existence.equation = sd.lookup(normalized_ratio2, "Cost_to_WtC_Ratio")

#####------------------------------------------------------------------------------------------------------------------------

Average_Cost_of_Ownership_without_EMML_Existence = model.converter("Average_Cost_of_Ownership_without_EMML_Existence")

Average_Cost_of_Ownership_without_EMML_Existence._function_string = (
    "lambda model, t: (model.memoize('EMMN_Ownership_Costs', t) + model.memoize('ICE_Ownership_Costs', t)) / 2"
)

Average_Cost_of_Ownership_without_EMML_Existence.generate_function()

Proportion_of_EMMN_Ownership_Cost_to_Average_Ownership_Cost_without_EMML_Existence = model.converter("Proportion_of_EMMN_Ownership_Cost_to_Average_Ownership_Cost_without_EMML_Existence")
Proportion_of_EMMN_Ownership_Cost_to_Average_Ownership_Cost_without_EMML_Existence._function_string = (
    "lambda model, t: "
    "((model.memoize('EMMN_Ownership_Costs', t) / model.memoize('Average_Cost_of_Ownership_without_EMML_Existence', t)) "
    "if model.memoize('Average_Cost_of_Ownership_without_EMML_Existence', t) != 0.0 else 0.0) "
    "* (1 + model.memoize('Value_Assigned_to_EM_Environmental_Impact', t))"
)
Proportion_of_EMMN_Ownership_Cost_to_Average_Ownership_Cost_without_EMML_Existence.generate_function()

Cost_Impact_EMMN_without_EMML_Existence = model.converter("Cost_Impact_EMMN_without_EMML_Existence")

normalized_ratio = sd.max(0.0,
                      (Proportion_of_EMMN_Ownership_Cost_to_Average_Ownership_Cost_without_EMML_Existence - Minimum_Cost_Difference) / (Maximum_Cost_Difference - Minimum_Cost_Difference) if (Maximum_Cost_Difference - Minimum_Cost_Difference) != 0 else 0.0
                     )
Cost_Impact_EMMN_without_EMML_Existence.equation = sd.lookup(normalized_ratio, "Cost_to_WtC_Ratio")

#####------------------------------------------------------------------------------------------------------------------------

Base_Marketing_Response = model.constant("Base_Marketing_Response")
Base_Marketing_Response.equation = values["Base Marketing Response"]

model.points["Marketing_Effort_Marketing_Effect"] = [
        [0.0, 0.0],
        [0.124, 0.022],
        [0.235, 0.061],
        [0.367, 0.154],
        [0.5, 0.3],
        [0.629, 0.509],
        [0.75, 0.75],
        [0.871, 0.939],
        [0.915, 0.974],
        [1.0, 1.0],
        [1.1, 1.0],
]

Marketing_Effort_Marketing_Effect_EMML = model.converter("Marketing_Effort_Marketing_Effect_EMML")
Marketing_Effort_Marketing_Effect_EMML.equation = sd.lookup(Marketing_Effort_EMML, "Marketing_Effort_Marketing_Effect")

Effectiveness_of_Marketing_EMML = model.converter("Effectiveness_of_Marketing_EMML")
Effectiveness_of_Marketing_EMML.equation = (Base_Marketing_Response * Marketing_Effort_Marketing_Effect_EMML * Cost_Impact_EMML)

#####------------------------------------------------------------------------------------------------------------------------

Marketing_Effort_Marketing_Effect_EMMN = model.converter("Marketing_Effort_Marketing_Effect_EMMN")
Marketing_Effort_Marketing_Effect_EMMN.equation = sd.lookup(Marketing_Effort_EMMN, "Marketing_Effort_Marketing_Effect")

#####------------------------------------------------------------------------------------------------------------------------

Effectiveness_of_Marketing_EMMN_with_EMML_Existence = model.converter("Effectiveness_of_Marketing_EMMN_with_EMML_Existence")
Effectiveness_of_Marketing_EMMN_with_EMML_Existence.equation = (Base_Marketing_Response * Marketing_Effort_Marketing_Effect_EMMN * Cost_Impact_EMMN_with_EMML_Existence)

#####------------------------------------------------------------------------------------------------------------------------

Effectiveness_of_Marketing_EMMN_without_EMML_Existence = model.converter("Effectiveness_of_Marketing_EMMN_without_EMML_Existence")
Effectiveness_of_Marketing_EMMN_without_EMML_Existence.equation = (Base_Marketing_Response * Marketing_Effort_Marketing_Effect_EMMN * Cost_Impact_EMMN_without_EMML_Existence)

#"""**User (WTC)**"""

Frequency_and_Effectiveness_of_Contact_between_User_and_Non_User = model.constant("Frequency_and_Effectiveness_of_Contact_between_User_and_Non_User")
Frequency_and_Effectiveness_of_Contact_between_User_and_Non_User.equation = values["Frequency and Effectiveness of Contact between User and Non User"]

Frequency_and_Effectiveness_of_Contact_between_Non_User = model.constant("Frequency_and_Effectiveness_of_Contact_between_Non_User")
Frequency_and_Effectiveness_of_Contact_between_Non_User.equation = values["Frequency and Effectiveness of Contact between Non User"]

#####------------------------------------------------------------------------------------------------------------------------

Direct_Exposure_of_Cost_EMMN_with_EMML_Existence = model.converter("Direct_Exposure_of_Cost_EMMN_with_EMML_Existence")
Indirect_Exposure_of_Cost_EMMN_with_EMML_Existence = model.converter("Indirect_Exposure_of_Cost_EMMN_with_EMML_Existence")

#####------------------------------------------------------------------------------------------------------------------------

Initial_Willingness_to_Consider_EMMN_with_EMML_Existence = model.constant("Initial_Willingness_to_Consider_EMMN_with_EMML_Existence")
Initial_Willingness_to_Consider_EMMN_with_EMML_Existence.equation = values["Initial Willingness to Consider EMMN"]

#####------------------------------------------------------------------------------------------------------------------------

Social_Exposure_Reference_Level = model.constant("Social_Exposure_Reference_Level")
Social_Exposure_Reference_Level.equation = values["Social Exposure Reference Level"]

Slope_of_Decay_Rate_WtC_EMMN = model.converter("Slope_of_Decay_Rate_WtC_EMMN")
Slope_of_Decay_Rate_WtC_EMMN.equation = (1.0 / (2.0 *Social_Exposure_Reference_Level) if Social_Exposure_Reference_Level != 0.0 else 0.0)

WtC_Basic_Decay_with_EMML_Existence = model.constant("WtC_Basic_Decay_with_EMML_Existence")
WtC_Basic_Decay_with_EMML_Existence.equation = values["WtC Basic Decay"]

Average_Fractional_Decay_of_EMMN = model.converter("Average_Fractional_Decay_of_EMMN")

#####------------------------------------------------------------------------------------------------------------------------

Increase_in_WtC_EMMN_with_EMML_Existence = model.flow("Increase_in_WtC_EMMN_with_EMML_Existence")

WtC_EMMN_Decay_with_EMML_Existence = model.flow("WtC_EMMN_Decay_with_EMML_Existence")

#####------------------------------------------------------------------------------------------------------------------------

Willingness_to_Consider_EMMN_with_EMML_Existence = model.stock("Willingness_to_Consider_EMMN_with_EMML_Existence")

Willingness_to_Consider_EMMN_with_EMML_Existence.initial_value = Initial_Willingness_to_Consider_EMMN_with_EMML_Existence

#####------------------------------------------------------------------------------------------------------------------------

Direct_Exposure_of_Cost_EMMN_with_EMML_Existence.equation = Cost_Impact_EMMN_with_EMML_Existence * EMMN_Ownership_Proportion * Frequency_and_Effectiveness_of_Contact_between_User_and_Non_User * Willingness_to_Consider_EMMN_with_EMML_Existence
Indirect_Exposure_of_Cost_EMMN_with_EMML_Existence.equation = Cost_Impact_EMMN_with_EMML_Existence * (1 - EMMN_Ownership_Proportion) * Frequency_and_Effectiveness_of_Contact_between_Non_User * Willingness_to_Consider_EMMN_with_EMML_Existence

#####------------------------------------------------------------------------------------------------------------------------

Impact_of_Total_Social_Exposure_EMMN_with_EMML_Existence = model.converter("Impact_of_Total_Social_Exposure_EMMN_with_EMML_Existence")
Impact_of_Total_Social_Exposure_EMMN_with_EMML_Existence.equation = (Effectiveness_of_Marketing_EMMN_with_EMML_Existence + Direct_Exposure_of_Cost_EMMN_with_EMML_Existence + Indirect_Exposure_of_Cost_EMMN_with_EMML_Existence)

#####------------------------------------------------------------------------------------------------------------------------

Increase_in_WtC_EMMN_with_EMML_Existence.equation = (Impact_of_Total_Social_Exposure_EMMN_with_EMML_Existence * sd.max(1.0 - Willingness_to_Consider_EMMN_with_EMML_Existence, 0.0))

#####------------------------------------------------------------------------------------------------------------------------

Average_Fractional_Decay_of_EMMN_with_EMML_Existence = model.converter("Average_Fractional_Decay_of_EMMN_with_EMML_Existence")
Average_Fractional_Decay_of_EMMN_with_EMML_Existence._function_string = (
    "lambda model, t: "
    "model.memoize('WtC_Basic_Decay_with_EMML_Existence', t) * ("
        "pow(2.718281828459045, "
            "-4.0 * model.memoize('Slope_of_Decay_Rate_WtC_EMMN', t) * "
            "(model.memoize('Impact_of_Total_Social_Exposure_EMMN_with_EMML_Existence', t) - "
             "model.memoize('Social_Exposure_Reference_Level', t)))"
        " / "
        "(1.0 + pow(2.718281828459045, "
            "-4.0 * model.memoize('Slope_of_Decay_Rate_WtC_EMMN', t) * "
            "(model.memoize('Impact_of_Total_Social_Exposure_EMMN_with_EMML_Existence', t) - "
             "model.memoize('Social_Exposure_Reference_Level', t)))"
        ")"
    ")"
)

Average_Fractional_Decay_of_EMMN_with_EMML_Existence.generate_function()

WtC_EMMN_Decay_with_EMML_Existence.equation = Willingness_to_Consider_EMMN_with_EMML_Existence*(Average_Fractional_Decay_of_EMMN_with_EMML_Existence)

#####------------------------------------------------------------------------------------------------------------------------

Willingness_to_Consider_EMMN_with_EMML_Existence.equation = (Increase_in_WtC_EMMN_with_EMML_Existence - WtC_EMMN_Decay_with_EMML_Existence)
Willingness_to_Consider_EMMN_with_EMML_Existence_in_Percent = model.converter("Willingness_to_Consider_EMMN_with_EMML_Existence_in_Percent")
Willingness_to_Consider_EMMN_with_EMML_Existence_in_Percent.equation = Willingness_to_Consider_EMMN_with_EMML_Existence * 100

Frequency_and_Effectiveness_of_Contact_between_User_and_Non_User = model.constant("Frequency_and_Effectiveness_of_Contact_between_User_and_Non_User")
Frequency_and_Effectiveness_of_Contact_between_User_and_Non_User.equation = values["Frequency and Effectiveness of Contact between User and Non User"]

Frequency_and_Effectiveness_of_Contact_between_Non_User = model.constant("Frequency_and_Effectiveness_of_Contact_between_Non_User")
Frequency_and_Effectiveness_of_Contact_between_Non_User.equation = values["Frequency and Effectiveness of Contact between Non User"]

#####------------------------------------------------------------------------------------------------------------------------

Direct_Exposure_of_Cost_EMML = model.converter("Direct_Exposure_of_Cost_EMML")
Indirect_Exposure_of_Cost_EMML = model.converter("Indirect_Exposure_of_Cost_EMML")

#####------------------------------------------------------------------------------------------------------------------------

Initial_Willingness_to_Consider_EMML = model.constant("Initial_Willingness_to_Consider_EMML")
Initial_Willingness_to_Consider_EMML.equation = values["Initial Willingness to Consider EMML"]

#####------------------------------------------------------------------------------------------------------------------------

Social_Exposure_Reference_Level = model.constant("Social_Exposure_Reference_Level")
Social_Exposure_Reference_Level.equation = values["Social Exposure Reference Level"]

Slope_of_Decay_Rate_WtC_EMML = model.converter("Slope_of_Decay_Rate_WtC_EMML")
Slope_of_Decay_Rate_WtC_EMML.equation = (1.0 / (2.0 *Social_Exposure_Reference_Level) if Social_Exposure_Reference_Level != 0.0 else 0.0)

WtC_Basic_Decay_EMML = model.constant("WtC_Basic_Decay_EMML")
WtC_Basic_Decay_EMML.equation = values["WtC Basic Decay"]

Average_Fractional_Decay_of_EMML = model.converter("Average_Fractional_Decay_of_EMML")

#####------------------------------------------------------------------------------------------------------------------------

Increase_in_WtC_EMML = model.flow("Increase_in_WtC_EMML")

WtC_EMML_Decay = model.flow("WtC_EMML_Decay")

#####------------------------------------------------------------------------------------------------------------------------

Willingness_to_Consider_EMML = model.stock("Willingness_to_Consider_EMML")

Willingness_to_Consider_EMML.initial_value = Initial_Willingness_to_Consider_EMML

#####------------------------------------------------------------------------------------------------------------------------

Direct_Exposure_of_Cost_EMML.equation = Cost_Impact_EMML * EMML_Ownership_Proportion * Frequency_and_Effectiveness_of_Contact_between_User_and_Non_User * Willingness_to_Consider_EMML
Indirect_Exposure_of_Cost_EMML.equation = Cost_Impact_EMML * (1 - EMML_Ownership_Proportion) * Frequency_and_Effectiveness_of_Contact_between_Non_User * Willingness_to_Consider_EMML

#####------------------------------------------------------------------------------------------------------------------------

Impact_of_Total_Social_Exposure_EMML = model.converter("Impact_of_Total_Social_Exposure_EMML")
Impact_of_Total_Social_Exposure_EMML.equation = (Effectiveness_of_Marketing_EMML + Direct_Exposure_of_Cost_EMML + Indirect_Exposure_of_Cost_EMML)

#####------------------------------------------------------------------------------------------------------------------------

Increase_in_WtC_EMML.equation = (Impact_of_Total_Social_Exposure_EMML* sd.max(1.0 - Willingness_to_Consider_EMML, 0.0))

#####------------------------------------------------------------------------------------------------------------------------

Average_Fractional_Decay_of_EMML = model.converter("Average_Fractional_Decay_of_EMML")
Average_Fractional_Decay_of_EMML._function_string = (
    "lambda model, t: "
    "model.memoize('WtC_Basic_Decay_EMML', t) * ("
        "pow(2.718281828459045, "
            "-4.0 * model.memoize('Slope_of_Decay_Rate_WtC_EMML', t) * "
            "(model.memoize('Impact_of_Total_Social_Exposure_EMML', t) - "
             "model.memoize('Social_Exposure_Reference_Level', t)))"
        " / "
        "(1.0 + pow(2.718281828459045, "
            "-4.0 * model.memoize('Slope_of_Decay_Rate_WtC_EMML', t) * "
            "(model.memoize('Impact_of_Total_Social_Exposure_EMML', t) - "
             "model.memoize('Social_Exposure_Reference_Level', t)))"
        ")"
    ")"
)

Average_Fractional_Decay_of_EMML.generate_function()

WtC_EMML_Decay.equation = Willingness_to_Consider_EMML * (Average_Fractional_Decay_of_EMML)

#####------------------------------------------------------------------------------------------------------------------------

Willingness_to_Consider_EMML.equation = (Increase_in_WtC_EMML - WtC_EMML_Decay)
Willingness_to_Consider_EMML_in_Percent = model.converter("Willingness_to_Consider_EMML_in_Percent")
Willingness_to_Consider_EMML_in_Percent.equation = Willingness_to_Consider_EMML * 100

Frequency_and_Effectiveness_of_Contact_between_User_and_Non_User = model.constant("Frequency_and_Effectiveness_of_Contact_between_User_and_Non_User")
Frequency_and_Effectiveness_of_Contact_between_User_and_Non_User.equation = values["Frequency and Effectiveness of Contact between User and Non User"]

Frequency_and_Effectiveness_of_Contact_between_Non_User = model.constant("Frequency_and_Effectiveness_of_Contact_between_Non_User")
Frequency_and_Effectiveness_of_Contact_between_Non_User.equation = values["Frequency and Effectiveness of Contact between Non User"]

#####------------------------------------------------------------------------------------------------------------------------

Direct_Exposure_of_Cost_EMMN_without_EMML_Existence = model.converter("Direct_Exposure_of_Cost_EMMN_without_EMML_Existence")
Indirect_Exposure_of_Cost_EMMN_without_EMML_Existence = model.converter("Indirect_Exposure_of_Cost_EMMN_without_EMML_Existence")

#####------------------------------------------------------------------------------------------------------------------------

Initial_Willingness_to_Consider_EMMN_without_EMML_Existence = model.constant("Initial_Willingness_to_Consider_EMMN_without_EMML_Existence")
Initial_Willingness_to_Consider_EMMN_without_EMML_Existence.equation = values["Initial Willingness to Consider EMMN"]

#####------------------------------------------------------------------------------------------------------------------------

Social_Exposure_Reference_Level = model.constant("Social_Exposure_Reference_Level")
Social_Exposure_Reference_Level.equation = values["Social Exposure Reference Level"]

Slope_of_Decay_Rate_WtC_EMMN = model.converter("Slope_of_Decay_Rate_WtC_EMMN")
Slope_of_Decay_Rate_WtC_EMMN.equation = (1.0 / (2.0 *Social_Exposure_Reference_Level) if Social_Exposure_Reference_Level != 0.0 else 0.0)

WtC_Basic_Decay_without_EMML_Existence = model.constant("WtC_Basic_Decay_without_EMML_Existence")
WtC_Basic_Decay_without_EMML_Existence.equation = values["WtC Basic Decay"]

Average_Fractional_Decay_of_EMMN = model.converter("Average_Fractional_Decay_of_EMMN")

#####------------------------------------------------------------------------------------------------------------------------

Increase_in_WtC_EMMN_without_EMML_Existence = model.flow("Increase_in_WtC_EMMN_without_EMML_Existence")

WtC_EMMN_Decay_without_EMML_Existence = model.flow("WtC_EMMN_Decay_without_EMML_Existence")

#####------------------------------------------------------------------------------------------------------------------------

Willingness_to_Consider_EMMN_without_EMML_Existence = model.stock("Willingness_to_Consider_EMMN_without_EMML_Existence")

Willingness_to_Consider_EMMN_without_EMML_Existence.initial_value = Initial_Willingness_to_Consider_EMMN_without_EMML_Existence

#####------------------------------------------------------------------------------------------------------------------------

Direct_Exposure_of_Cost_EMMN_without_EMML_Existence.equation = Cost_Impact_EMMN_without_EMML_Existence * EMMN_Ownership_Proportion * Frequency_and_Effectiveness_of_Contact_between_User_and_Non_User * Willingness_to_Consider_EMMN_without_EMML_Existence
Indirect_Exposure_of_Cost_EMMN_without_EMML_Existence.equation = Cost_Impact_EMMN_without_EMML_Existence * (1 - EMMN_Ownership_Proportion) * Frequency_and_Effectiveness_of_Contact_between_Non_User * Willingness_to_Consider_EMMN_without_EMML_Existence

#####------------------------------------------------------------------------------------------------------------------------

Impact_of_Total_Social_Exposure_EMMN_without_EMML_Existence = model.converter("Impact_of_Total_Social_Exposure_EMMN_without_EMML_Existence")
Impact_of_Total_Social_Exposure_EMMN_without_EMML_Existence.equation = (Effectiveness_of_Marketing_EMMN_without_EMML_Existence + Direct_Exposure_of_Cost_EMMN_without_EMML_Existence + Indirect_Exposure_of_Cost_EMMN_without_EMML_Existence)

#####------------------------------------------------------------------------------------------------------------------------

Increase_in_WtC_EMMN_without_EMML_Existence.equation = (Impact_of_Total_Social_Exposure_EMMN_without_EMML_Existence* sd.max(1.0 - Willingness_to_Consider_EMMN_without_EMML_Existence, 0.0))

#####------------------------------------------------------------------------------------------------------------------------

Average_Fractional_Decay_of_EMMN_without_EMML_Existence = model.converter("Average_Fractional_Decay_of_EMMN_without_EMML_Existence")
Average_Fractional_Decay_of_EMMN_without_EMML_Existence._function_string = (
    "lambda model, t: "
    "model.memoize('WtC_Basic_Decay_without_EMML_Existence', t) * ("
        "pow(2.718281828459045, "
            "-4.0 * model.memoize('Slope_of_Decay_Rate_WtC_EMMN', t) * "
            "(model.memoize('Impact_of_Total_Social_Exposure_EMMN_without_EMML_Existence', t) - "
             "model.memoize('Social_Exposure_Reference_Level', t)))"
        " / "
        "(1.0 + pow(2.718281828459045, "
            "-4.0 * model.memoize('Slope_of_Decay_Rate_WtC_EMMN', t) * "
            "(model.memoize('Impact_of_Total_Social_Exposure_EMMN_without_EMML_Existence', t) - "
             "model.memoize('Social_Exposure_Reference_Level', t)))"
        ")"
    ")"
)

Average_Fractional_Decay_of_EMMN_without_EMML_Existence.generate_function()

WtC_EMMN_Decay_without_EMML_Existence.equation = Willingness_to_Consider_EMMN_without_EMML_Existence * (Average_Fractional_Decay_of_EMMN_without_EMML_Existence)

#####------------------------------------------------------------------------------------------------------------------------

Willingness_to_Consider_EMMN_without_EMML_Existence.equation = (Increase_in_WtC_EMMN_without_EMML_Existence - WtC_EMMN_Decay_without_EMML_Existence)
Willingness_to_Consider_EMMN_without_EMML_Existence_in_Percent = model.converter("Willingness_to_Consider_EMMN_without_EMML_Existence_in_Percent")
Willingness_to_Consider_EMMN_without_EMML_Existence_in_Percent.equation = Willingness_to_Consider_EMMN_without_EMML_Existence * 100

Agregat_of_Willingness_to_Consider_EM_without_EMML_Existence = model.converter("Agregat_of_Willingness_to_Consider_EM_without_EMML_Existence")
Agregat_of_Willingness_to_Consider_EM_without_EMML_Existence.equation = Willingness_to_Consider_EMMN_without_EMML_Existence_in_Percent

Agregat_of_Willingness_to_Consider_EM_with_EMML_Existence = model.converter("Agregat_of_Willingness_to_Consider_EM_with_EMML_Existence")
Agregat_of_Willingness_to_Consider_EM_with_EMML_Existence.equation = Willingness_to_Consider_EMMN_with_EMML_Existence_in_Percent + Willingness_to_Consider_EMML_in_Percent


# -----------------------------------------------------------------------------
# Section: Register the model with BPTK
# -----------------------------------------------------------------------------

import BPTK_Py as _BPTK
bptk = _BPTK.bptk()
bptk.register_model(model)

scenario_manager_name = "smEmml vs emmn"

# -----------------------------------------------------------------------------
# Section: Pre‑compute data series for each variable
# -----------------------------------------------------------------------------

try:
    # Compute and store the DataFrame for each variable once.
    # Each call returns a DataFrame indexed by time.  We keep the raw
    # DataFrame; it will be tidied when displayed in the UI.
    o1 = bptk.plot_scenarios(
        scenarios="base",
        scenario_managers=scenario_manager_name,
        equations=["Willingness_to_Consider_EMMN_without_EMML_Existence_in_Percent"],
        series_names={},
        return_df=True,
    )
    o2 = bptk.plot_scenarios(
        scenarios="base",
        scenario_managers=scenario_manager_name,
        equations=["Willingness_to_Consider_EMMN_with_EMML_Existence_in_Percent"],
        series_names={},
        return_df=True,
    )
    o3 = bptk.plot_scenarios(
        scenarios="base",
        scenario_managers=scenario_manager_name,
        equations=["Willingness_to_Consider_EMML_in_Percent"],
        series_names={},
        return_df=True,
    )
    o4 = bptk.plot_scenarios(
        scenarios="base",
        scenario_managers=scenario_manager_name,
        equations=["Agregat_of_Willingness_to_Consider_EM_without_EMML_Existence"],
        series_names={},
        return_df=True,
    )
    o5 = bptk.plot_scenarios(
        scenarios="base",
        scenario_managers=scenario_manager_name,
        equations=["Agregat_of_Willingness_to_Consider_EM_with_EMML_Existence"],
        series_names={},
        return_df=True,
    )


    # Map variable names to their precomputed DataFrames
    precomputed_series = {
        "Willingness_to_Consider_EMMN_without_EMML_Existence": o1,
        "Willingness_to_Consider_EMMN_with_EMML_Existence": o2,
        "Willingness_to_Consider_EMML_in_Percent": o3,
        "Agregat_of_Willingness_to_Consider_EM_without_EMML_Existence": o4,
        "Agregat_of_Willingness_to_Consider_EM_with_EMML_Existence": o5,
    }
except Exception as _exc_precompute:
    # If any error occurs during precomputation, fall back to an empty mapping.
    precomputed_series = {}


# -----------------------------------------------------------------------------
# Section: Streamlit UI
# -----------------------------------------------------------------------------

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #BBDEFB;    /* warna background sidebar */
    }
    [data-testid="stSidebar"] * {
        color: #000000;               /* warna teks di sidebar */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("Parameter Options")

# The intruction text in the below of title, justify
st.sidebar.write("Select a variable to view its simulation results.")


# Variable list for the sidebar.  Users can select any of these variables to
# visualize.  This list corresponds to converters or stocks defined above.
VARIABLE_OPTIONS = [
    "Willingness to Consider EMMN without EMML Existence",
    "Willingness to Consider EMMN with EMML Existence",
    "Willingness to Consider EMML",
    "Agregat of Willingness to Consider EM without EMML Existence",
    "Agregat of Willingness to Consider EM with EMML Existence",
]

# Dropdown for select variabel
selected_var = st.sidebar.selectbox(
    "Select a variable",          # label tidak kosong
    VARIABLE_OPTIONS,
    index=0,
    label_visibility="collapsed"  # label disembunyikan di UI
)


# Button linking to the parameter database
if st.sidebar.button("Update Database"):
    st.sidebar.markdown(
        "[Open Parameter Database](https://docs.google.com/spreadsheets/d/"
        "1Q3PkFQwx3yoROVaDN5MkXv_lndTo9mRkV-2TyB6EhtE/edit?usp=sharing)"
    )

st.markdown(
    """
    <h1 style="
        text-align: center;
        text-justify: inter-word;
        font-size: 2.5rem;
        font-weight: 600;
        margin-top: 0;
        margin-bottom: 0.5rem;
    ">
        Data-Driven Monitor Dashboard of Electric Motorcycle Market Competition (EMML vs EMMN) using a Python-Based System Dynamics Model
    </h1>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    .justify-text {
        text-align: justify;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p class="justify-text">
        This dashboard shows the simulated competition between the electric motorcycle Market
        Leader (EMML), which is just entering the electric motorcycle
        niche, and the electric motorcycle Market Nicher (EMMN), which entered this
        niche earlier. Use the sidebar to select a variable and see how its value
        changes over time.
    </p>
    """,
    unsafe_allow_html=True,
)


df = None
used_custom_plot = False
if selected_var == "Willingness to Consider EMMN without EMML Existence":
    df = o1
elif selected_var == "Willingness to Consider EMMN with EMML Existence":
    df = o2
elif selected_var == "Willingness to Consider EMML":
    df = o3
elif selected_var == "Agregat of Willingness to Consider EM without EMML Existence":
    df = o4
elif selected_var == "Agregat of Willingness to Consider EM with EMML Existence":
    df = o5

if not used_custom_plot:
    if df is None or df.empty:
        st.warning(
            f"No data available for the variable '{selected_var}'. "
            "It may not be defined or there was an error during simulation."
        )
    else:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(col).strip() for col in df.columns.values]

        match_cols = [col for col in df.columns if selected_var in col]
        var_col = match_cols[0] if match_cols else df.columns[0]

        table_df = df[[var_col]].copy().reset_index()
        table_df.columns = ["Time", selected_var]

        # Layout: table & chart side by side
        col_table, col_chart = st.columns(2)

        table_df_rounded = table_df.copy()
        if selected_var in table_df_rounded.columns:
            table_df_rounded[selected_var] = table_df_rounded[selected_var].round(6)
        with col_table:
            st.subheader("Data Table")
            st.dataframe(
                table_df_rounded,
                height = 300,
		use_container_width=True,
            )

        with col_chart:
            st.subheader("Line Chart")
            chart_df = table_df.set_index("Time")
            st.line_chart(chart_df,height=300,)