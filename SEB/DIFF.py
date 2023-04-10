--- ./assignment2.py	2023-03-30 22:04:03.000000000 -0400
+++ assignment2/assignment2.py	2023-03-29 22:31:37.000000000 -0400
@@ -98,9 +98,6 @@
     def uncond_var(self):
         return self.omega/(1 - self.persistence())
 
-    def corr_ret_var(self, h_t):
-        return -2*self.gamma / np.sqrt(2 + 4*self.gamma**2)
-    
     def simulateP(self, S_t0, n_days, n_paths, h_tp1, z=None):
         '''Simulate excess returns and their variance under the P measure
         
@@ -142,12 +139,6 @@
 
         return ex_r, h, z
 
-    def simulateQ(self, S_t0, n_days, n_paths, h_tp1, z=None):
-        raise RuntimeError('Implement for assignment 2')
-    
-    def option_price(self, cum_ex_r, F_t0_T, K, rf, dtm, is_call):
-        raise RuntimeError('Implement for assignment 2')
-
     
 def plot_excess_return_forecasts(horizon, P, Q, annualized=False):
     ann = [1.0]
Only in .: assignment2_solution.py
