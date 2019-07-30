
# coding: utf-8

# # Harder problems 

# The previous notebooks were addressing simple problems, that in most cases would not require a full Bayesian model to be fitted. We are now going to explore some more complex scenarios. 
# 
# First here is a list of the most interesting notebooks and exercises I've found. The PyMC documentation website is a goldmine itself, since they increasingly include a variety of examples all correlated by a notebook. Not all of them are relevant for the next problems, but you can check them out to have an idea of what one can do with Probabilistic modeling. 
# 
# 
# **Rugby Example**
# https://docs.pymc.io/notebooks/rugby_analytics.html
# 
# **Survival Analysis**
# 
# https://docs.pymc.io/notebooks/bayes_param_survival_pymc3.html
# 
# **CO2 Levels Prediction**
# 
# https://docs.pymc.io/notebooks/GP-MaunaLoa.html
# 
# **Dependent Density Regression**
# 
# https://docs.pymc.io/notebooks/dependent_density_regression.html
# 
# **Dirichlet Processes**
# 
# https://docs.pymc.io/notebooks/dp_mix.html

# # AB Testing
# 
# An example of AB Testing can be found in Chapter 2 of the Bayesian Methods for hackers (https://nbviewer.jupyter.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter2_MorePyMC/Ch2_MorePyMC_PyMC3.ipynb)
# 
# 
# A/B testing is a statistical design pattern for determining the difference of effectiveness between two different treatments, approaches, strategies. In the example above they model the case of web-developers interested in knowing which design of their website yields more sales or some other metric of interest. They will route some fraction of visitors to site A, and the other fraction to site B, and record if the visit yielded a sale or not. 
# 
# Here we go back to the IADS summer school example in the slides. Shaaba was still sending out invitations, as fliers and emails. After 1 month, she wanted to know which strategy was more effective and which one was more cost-effective. ( imagine a world where each flier/ad corresponds to a single signup).  
# 
# Idea: For AB testing, you should model both datasets with the same likelihood and check which one has the best values for the parameters. For cost-effectiveness, remember that you can use deterministic variables to evaluate the cost of each strategy. 

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import pymc3 as pm
import matplotlib.pyplot as plt


# In[28]:


# Data: for each flier/ad we have a binary value 
# that corresponds to whether it was effective or not (the person signed)

fliers = [1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0]
emails = [1, 0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,0,0]

# Cost of each flier/email (Shaaba's time isn't for free)
cost_flier = 10
cost_email = 2

# Revenue for each sign up
fee = 100


# In[29]:


# How many times did someone sign up? What's the observed frequency for the sign up? 
#What was the cost of the two campaigns? 


# In[30]:


# What distribution is appropriate to describe the likelihood of the data?
# Build the model


# In[31]:


# Show results. Which approach is more effective? cost effective? What happens if the cost of fliers drops?


# In[32]:


# How many times was the sign up ratio of fliers higher than the one of emails


# In[33]:





# # Image segmentation
# 
# Bayesian learning allows to fit mixture models and apply data clustering. A good starting point for clustering with PyMC3 is the notebook https://nbviewer.jupyter.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter3_MCMC/Ch3_IntroMCMC_PyMC3.ipynb.
# 
# There they apply a clustering algorithm to a mixture model. 
# 
# ## Remove Background
# 
# Try here to remove the background from this image. Instead of the 2D image, imagine the histogram of the values of all pixels...
# At the beginning don't worry about the 2d structure of the image, there are filtering algorithms that can take care of that at the end. 
# 
# 

# In[62]:


with open('dummy_image.txt', 'r') as f:
    data=np.loadtxt(f,delimiter =',')


# In[61]:


# Plot the histogram of the data


# In[64]:


# Build a clustering model for the mixture distribution using categorical variables


# In[65]:


# Show Assignment


# In[66]:


# Use the predictions to segment the image.


# In[67]:





# In[14]:





# In[70]:





# In[72]:





# # PET imaging counts
# 
# PET ( positron emission tomography) is a functional medical imaging technique relying on radioactive decay. In a usual PET test, a radioactive substance is introduced in the human body as a molecule whose destiny is known. For example, usually radioactive F18 molecules are used, so that metabolic activity, requiring glucose, can be traced.
# 
# The raw data recorded by a PET are counts for each pixel for each time interval, but what is most interesting would be the average value of counts, so that area with stronger activity can be detected.
# 
# In the figure below, that reports the total number of counts for each pixel of a dummy PET scan, you can spot a yellow area of high activity.

# In[162]:


# Ignore this
with open('PET.txt', 'r') as f:
    pet=np.loadtxt(f,delimiter =',')
pet_signal =np.array([np.random.poisson(pet) for i in range(1000)])

# Total number of counts
plt.imshow(np.sum(pet_signal,axis=0))


# ### Exercise
# Try to obtain the average counts for each pixel and try to segment the area of high activity using the techniques seen so far

# In[112]:


print(signal)
plt.imshow(np.average((signal),axis=0))

