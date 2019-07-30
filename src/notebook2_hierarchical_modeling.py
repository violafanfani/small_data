
# coding: utf-8

# # Hierarchical modeling
# 
# As we have covered in this morning's lecture, hierarchical models are one of the most attractive applications of Statistical Learning. We are first going to check a very complete example on how to predict the exposure to Radon in Americal households. Try then to solve some exercises yourself.
# 
# To begin with, check the this notebook: https://docs.pymc.io/notebooks/multilevel_modeling.html, if you want to try it out, this github repo https://github.com/fonnesbeck/multilevel_modeling hosts both the code and the data.
# 
# 
# You can now try to solve the following problems. I have got the exercises and data from the practical from http://www.bias-project.org.uk/WB2011Man/BHM-2011-practical.pdf. 
# 
# 
# 

# ## Mortality rates
# 
# In this question, you will be modelling data on mortality rates following surgery in each of 12
# hospitals. The data file surgical-dat.txt contains the following columns: hospital_code, alphanumerical code for the hospital, surgeries, the number of operations carried out in each hospital in a 1 year period, deaths, the
# number of deaths within 30 days of surgery in each hospital.
# 
# The aim of the analysis is to use surgical mortality rates as an indicator of each hospital’s
# performance and to identify whether any hospitals appear to be performing unusually well or
# poorly.

# In[5]:


import pandas as pd
import pymc3 as pm
import numpy as np
import seaborn as sns
# Open the table

with open('surgical-dat.txt', 'r') as f:
    data = pd.read_csv(f)

data


# In[6]:


# Build the model


# In[7]:


# Check the results


# In[8]:


# Which hospital is more reliable? To what extent?


# ## Patient treatment
# 
# This example uses (simulated) data from a clinical trial comparing two alternative treatments
# for HIV-infected individuals. 80 patients with HIV infection were randomly assigned to one of
# 2 treatment groups (drug = 0 (didanosine, ddI) and drug = 1 (zalcitabine, ddC)). CD4 counts
# were recorded at study entry (time t = 0) and again at 2, 6 and 12 months. An indicator of
# whether the patient had already been diagnosed with AIDS at study entry was also recorded
# (AIDS = 1 if patient diagnosed with AIDS, and 0 otherwise).

# In[9]:


with open('cd4-dat.csv','r') as f:
    cd4=pd.read_csv(f)

cd4.head()


# In[10]:


# Run the model and monitor the slope and intercept parameters, the regression
# coefficients for the effects of treatment and AIDS, and the residual error variance.

# HINT: Use the unpooled estimator with a GLM


# In[11]:


# Modify your code for the previous model to include a random intercept and a random
# slope (i.e. time coefficient) for each patient. Treat the coefficients for the effects of drug
# treatment and AIDS as fixed (i.e. not random effects) as before.

