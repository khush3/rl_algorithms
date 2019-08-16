#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Solution |to custom environment


# In[2]:


from custom_environment import ENVIRONMENT
from helpers import get_q_table, improve_q_table, Parameters
import numpy as np
import cv2
import pickle


# In[ ]:





# In[3]:


env = ENVIRONMENT(diagonal=True, size=10, num_enemy = 3, num_food = 1)
q = get_q_table(size=10)
parameters = Parameters()

# Test Environment
print(env.startover())

for i in range(10):
    print(env.step(np.random.randint(0,4)))
    env.render()

cv2.destroyAllWindows()

print(env.startover())


# In[4]:


# Improve the Q-value table
q = improve_q_table(env, q, parameters=parameters)


# In[6]:


# save the q table
import time
with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q, f)


# In[ ]:


np.random.randn()


# In[ ]:


parameters.HM_EPISODES


# In[ ]:





# In[ ]:




