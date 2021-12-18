#!/usr/bin/env python
# coding: utf-8

# ### Berra Karayel 0054477 CSSM502 Second Homework-Test


# In[16]:


import unittest
import numpy as np
import pandas as pd
import numpy.testing as npt


# In[22]:


from linear import lin_regress

# In[15]:


class Linear_Regression_Test(unittest.TestCase):
    
    def setting(self):
        self.indep = pd.DataFrame(
            [[1.0, np.nan, 2.0], [12.0, 24.0, 36.0], [3.0, 6.0, 8.0], [8.0, 12.0, 9.0], [8.0, 10.0, 4.0]])
        self.dep = pd.DataFrame([10.0, np.nan, 20.0, 40.0, 20.0])
        np.seterr(divide='ignore')
        coefficient, Std_error, T1, T2, self.indep_clean, self.dep_clean = linear_regression(self.indep, self.dep)

    def test_missing(self):
        indep = self.indep.dropna().to_numpy()
        dep = self.dep.dropna().to_numpy()
        npt.assert_array_equal(indep, self.indep_clean)
        npt.assert_array_equal(dep, self.dep_clean)

    def test_empty(self):
        indep = self.indep.dropna()
        dep = self.dep.dropna()
        shape_indep = np.shape(indep)
        shape_dep = np.shape(dep)
        control_indep = np.zeros(shape_indep, dtype=bool)
        control_dep = np.zeros(shape_dep, dtype=bool)
        npt.assert_array_equal(control_indep, pd.isnull(self.indep_clean))
        npt.assert_array_equal(control_dep, pd.isnull(self.dep_clean))


if __name__ == "__main__":
    unittest.main()

