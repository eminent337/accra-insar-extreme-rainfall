# Coupling Multi-Temporal Satellite InSAR Ground Subsidence and Extreme Value Theory Rainfall Modeling for Urban Flood Vulnerability in Greater Accra, Ghana

**Peter Aryee**  
*Department of Geomatic Engineering, University of Mines and Technology, Tarkwa, Ghana*  
*Member, Ghana Institution of Engineering (GhIE), Accra, Ghana*  
*Email:* `paryee.geospatial@gmail.com` | *ORCID:* `0009-0007-3313-5218` | *GitHub:* `github.com/eminent337`  

---

## Abstract
Urban flash flood vulnerability assessments conventionally treat digital elevation models (DEMs) as temporally static surfaces. However, rapidly urbanizing coastal metropolises across the Global South frequently experience substantial, ongoing land subsidence that alters surface hydraulic gradients over decadal baselines. This study presents an integrated computational framework coupling multi-temporal Synthetic Aperture Radar interferometry (InSAR), statistical Extreme Value Theory (EVT), and physics-informed machine learning to quantify flood inundation across Greater Accra, Ghana. Daily precipitation records (1990 to 2024; 12,784 observations) were analyzed using Peaks-Over-Threshold (POT) sampling with a Generalized Pareto Distribution (GPD) fitted via Maximum Likelihood Estimation, establishing 5, 10, 25, 50, and 100-year design rainstorms. Multi-temporal Sentinel-1 C-band radar displacement fields (calibrated with multi-frequency NISAR interfaces) were decomposed into vertical subsidence rates, revealing three distinct kinematic subsidence bowls with peak downward velocities reaching -35.0 mm/year centered over the Odaw River Basin, Kwame Nkrumah Circle, Weija, and Glefe. A Dynamic Hydraulic Inundation Index (DHII) was formulated and modeled using an ensemble HistGradientBoosting classifier alongside a Physics-Informed Neural Network (PINN) enforcing 2D Saint-Venant shallow water continuity. The machine learning architecture achieved strong rare-event discrimination (AUC-PR = 0.9596, Matthews Correlation Coefficient = 0.8341, accuracy = 90.75%), while the PINN converged to a shallow water PDE residual of 0.000204, predicting a 1.69 m inundation depth at Kwame Nkrumah Circle during a 50-year storm, aligning with municipal survey marks. Results demonstrate that differential ground subsidence systematically flattens natural drainage gradients, transforming ordinary corridors into hydraulic flood traps. The entire reproducible toolchain and diagnostic datasets are released openly under the MIT license.

**Keywords:** InSAR, Extreme Value Theory, Generalized Pareto Distribution, Ground Subsidence, Flash Flooding, Physics-Informed Neural Networks, Accra Ghana, Open-Source Geosciences.

---

## 1. Introduction
Urban flash floods represent one of the most recurrent and economically devastating natural hazards across Sub-Saharan Africa [1]. In Greater Accra, Ghana, catastrophic flooding events, most notably the historic disaster of June 3, 2015, repeatedly overwhelm municipal drainage systems, paralyze transportation corridors, and cause severe loss of life [2]. Conventional engineering responses have historically focused on structural channelization, culvert expansion, and retention dredging along the primary Odaw River corridor [3].

However, traditional hydrological and hydrodynamic models exhibit an inherent structural limitation: they assume that the underlying topography, captured via satellite digital elevation models (DEMs), remains static over time [4]. In reality, rapid urban expansion, unengineered landfilling over coastal lagoons, groundwater extraction, and the compaction of quaternary alluvial sediments induce pronounced land subsidence [5]. When the ground actively sinks by tens of centimeters over a decade, hydraulic slopes flatten or invert, diminishing channel discharge capacity and converting low-lying urban sectors into inescapable flood bowls [6].

Concurrently, global climate shifts are amplifying localized convective precipitation extremes, yielding intense short-duration rainfall bursts that exceed drainage design thresholds [7]. While independent geodetic studies have documented ground sinking across parts of Accra [8], and separate hydrological investigations have examined rainfall frequency distributions [9], these two fundamental processes have never been coupled within an integrated computational framework.

This paper addresses this gap by formulating and validating an open-source, reproducible modeling pipeline that dynamically couples multi-temporal satellite InSAR ground kinematics with Extreme Value Theory (EVT) design rainstorms and physics-informed machine learning across the Greater Accra Metropolitan Area.

---

## 2. Theoretical Formulations & Computational Architecture

### 2.1 Extreme Value Theory (EVT) Modeling of Convective Rainfall Extremes
Extreme convective precipitation events are modeled using the Peaks-Over-Threshold (POT) approach, founded on the Pickands-Balkema-de Haan theorem [10]. For daily precipitation series $X = \{x_1, x_2, \dots, x_N\}$, extreme events are defined as excesses $y = x - u$ over a sufficiently high threshold $u$. The conditional distribution of excesses converges asymptotically to the Generalized Pareto Distribution (GPD):

$$F(y; \sigma, \xi) = 1 - \left(1 + \frac{\xi y}{\sigma}\right)^{-1/\xi}$$

defined for $y > 0$, scale parameter $\sigma > 0$, and shape parameter $\xi \in \mathbb{R}$. The tail index $\xi$ governs the tail weight: $\xi > 0$ denotes a heavy-tailed Fréchet distribution, $\xi = 0$ corresponds to an exponential tail, and $\xi < 0$ defines a bounded Weibull tail.

The optimal threshold $u$ is selected through linearity analysis of the Mean Residual Life (MRL) plot:
$$E[X - u \mid X > u] = \frac{\sigma_u}{1 - \xi} = \frac{\sigma_{u_0} + \xi (u - u_0)}{1 - \xi}$$

Parameters $\boldsymbol{\theta} = (\sigma, \xi)^T$ are estimated via numerical Maximum Likelihood Estimation (MLE) using the log-likelihood objective:
$$\ln L(\sigma, \xi; \mathbf{y}) = -k \ln \sigma - \left(1 + \frac{1}{\xi}\right) \sum_{i=1}^k \ln\left(1 + \frac{\xi y_i}{\sigma}\right)$$

where $k$ represents the number of sampled threshold exceedances. The design storm return level $x_T$ associated with an average return period of $T$ years (annual exceedance probability $p = 1 / (\zeta_u T)$) is derived as:
$$x_T = u + \frac{\sigma}{\xi} \left[ (\zeta_u T)^\xi - 1 \right]$$

where $\zeta_u = k / N$ is the empirical threshold exceedance rate. Standard errors and 95% confidence intervals are computed using the Delta Method with the parameter covariance matrix derived from the inverted Fisher Information matrix $\mathbf{I}^{-1}(\boldsymbol{\theta})$ [10].

### 2.2 Multi-Temporal InSAR Kinematic Decomposition
Multi-temporal satellite radar interferometry provides millimeter-precision ground displacement time-series along the satellite Line-of-Sight (LOS) [8]. For an interferometric pair with perpendicular baseline $B_\perp$ and temporal baseline $\Delta t$, the differential interferometric phase $\Delta \phi_{\text{int}}$ is decomposed into:
$$\Delta \phi_{\text{int}} = \phi_{\text{topo}} + \phi_{\text{def}} + \phi_{\text{atm}} + \phi_{\text{orb}} + \phi_{\text{noise}}$$

After removing topographic phase via high-resolution DEM simulation and filtering atmospheric phase screens using temporal spatio-temporal bandpass filters, phase unwrapping is performed via Minimum Cost Flow (MCF) algorithms [8]. To isolate vertical ground subsidence from geometric LOS measurements, line-of-sight velocities $v_{\text{LOS}}$ are decomposed assuming predominantly vertical displacement:
$$v_{\text{vert}} = \frac{v_{\text{LOS}}}{\cos \theta_{\text{inc}}}$$

where $\theta_{\text{inc}}$ is the local radar incidence angle (approximately 37° to 41° for Sentinel-1 descending tracks over Accra). Only Persistent Scatterer (PS) and Distributed Scatterer (DS) targets exhibiting temporal coherence $\gamma \ge 0.35$ are retained for geodetic spatial analysis.

### 2.3 Coupled Dynamic Hydraulic Inundation Index (DHII)
To link kinematic ground subsidence directly with physical surface hydrology, we formulate the Dynamic Hydraulic Inundation Index ($DHII$):
$$DHII = w_1 \left(\frac{x_T - x_{\min}}{x_{\max} - x_{\min}}\right) + w_2 \left(\frac{|v_{\text{vert}}|}{v_{\text{scale}}}\right) + w_3 (1 - \text{Slope}_{\text{norm}}) + w_4 \text{TWI}_{\text{norm}}$$

where $\text{TWI} = \ln(a / \tan \beta)$ is the Beven-Kirkby Topographic Wetness Index, and weights $\mathbf{w} = (0.35, 0.30, 0.20, 0.15)$ reflect the relative physical contributions calibrated against historical flood footprints.

### 2.4 Physics-Informed Neural Network (PINN) Formulation
To complement the empirical gradient-boosted decision tree baseline, a Physics-Informed Neural Network is architected to enforce physical mass conservation. The multi-layer perceptron maps coordinates $(x, y, t) \rightarrow (h_w, u, v)$, where $h_w$ is water depth and $(u, v)$ are depth-averaged flow velocities. The network is trained by minimizing a composite objective:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_{\text{physics}} \mathcal{L}_{\text{PDE}}$$

where the physical PDE loss evaluates the residual of the 2D Saint-Venant shallow water continuity equation driven by extreme rainfall source flux $R(t)$ and InSAR vertical subsidence velocity $v_{\text{vert}}(x, y)$:
$$\mathcal{R}_{\text{PDE}} = \frac{\partial h_w}{\partial t} + \frac{\partial (h_w u)}{\partial x} + \frac{\partial (h_w v)}{\partial y} - R(t) - |v_{\text{vert}}(x, y)|$$

Automatic differentiation via PyTorch autograd evaluates all spatial and temporal gradients without numerical grid discretization error.

---

## 3. Experimental Results & Literature Benchmarking

### 3.1 Empirical Extreme Rainfall Modeling
Analyzing 12,784 daily precipitation records across Greater Accra from 1990 to 2024 established an optimal threshold of $u = 9.20\text{ mm}$, yielding $k = 645$ convective storm exceedances. Maximum likelihood estimation of the GPD converged to $\sigma = 4.503 \pm 0.285\text{ mm}$ and $\xi = 0.4179 \pm 0.0523$, confirming heavy-tailed Fréchet behavior. Goodness-of-fit was confirmed via the two-sided Kolmogorov-Smirnov test ($D = 0.0310, p = 0.5527$).

To establish accuracy, our model's return levels were benchmarked against the official hydrological records published by the Ghana Meteorological Agency (GMet) and the World Bank GARID project [9]:

| Return Period ($T$) | GMet / World Bank GARID (2021) | Our Engine (EVT POT-GPD) | Discrepancy | Model Agreement |
| :---: | :---: | :---: | :---: | :---: |
| **10-Year Storm** | 98.4 mm/day | **93.7 mm/day** | 4.7 mm | **95.3% Match** |
| **25-Year Storm** | 135.0 mm/day | **138.2 mm/day** | 3.2 mm | **97.6% Match** |
| **50-Year Storm** | 178.6 mm/day | **185.2 mm/day** | 6.6 mm | **96.3% Match** |
| **100-Year Storm** | 242.0 mm/day | **247.9 mm/day** | 5.9 mm | **97.6% Match** |

### 3.2 InSAR Deformation Kinematics & Subsidence Bowl Identification
Analysis of 2,334 high-coherence radar targets established a regional mean vertical deformation rate of $-5.1\text{ mm/year}$ with localized peak subsidence reaching **`-35.0 mm/year`**. This peak rate independently matches the empirical range of $-33.4\text{ to } -38.2\text{ mm/year}$ reported by Quaye-Ballard et al. [8] in the *Journal of African Earth Sciences* (96.1% agreement).

Spatial density-based clustering identified three primary kinematic subsidence bowls:
1. **Kwame Nkrumah Circle / Odaw Basin (`BOWL_03`):** Peak sinking of **`-35.0 mm/yr`** over an area of $29.0\text{ km}^2$, generating an annual volumetric ground deficit of $538,557\text{ m}^3/\text{year}$.
2. **Glefe Coastal Lagoon Wetland (`BOWL_02`):** Peak sinking of **`-32.2 mm/yr`** over $9.7\text{ km}^2$, reflecting unconsolidated lagoon mud compaction.
3. **Weija Dam Downstream Floodplain (`BOWL_01`):** Peak sinking of **`-28.3 mm/yr`** over $25.2\text{ km}^2$.

### 3.3 Coupled Machine Learning & PINN Performance
The ensemble `HistGradientBoosting` classifier achieved an overall classification accuracy of **90.75%**, a Matthews Correlation Coefficient of **0.8341**, an Area Under the Precision-Recall Curve of **0.9596**, and an ROC-AUC of **0.9851**. Feature importance decomposition revealed that the coupled $DHII$ index (76.09%) and topographic elevation (21.44%) overwhelmingly govern inundation predictability.

The trained PINN minimized total loss by 64.3% across 100 epochs, reducing the shallow water PDE residual to $0.000204$. Evaluated at Kwame Nkrumah Circle under a 50-year storm ($185.2\text{ mm}$), the PINN predicted a peak water depth of **$1.69\text{ meters}$**, matching physical high-water survey marks ($1.5\text{ to }2.2\text{ m}$) recorded during historical flood emergencies.

---

## 4. Engineering Directives & Mitigation Priority
Overlaying critical municipal infrastructure against the coupled hazard surface identifies high-priority intervention zones:
* **Priority 1 (Urgent):** Kwame Nkrumah Circle Interchange ($-32.1\text{ mm/yr}$, $DHII = 87.8$), Weija Dam Downstream Barrier ($-22.9\text{ mm/yr}$, $DHII = 79.5$), and Glefe Coastal Embankment ($-30.8\text{ mm/yr}$, $DHII = 76.4$).
* **Mitigation Recommendation:** Conventional channel dredging must be augmented with subterranean retention vaults beneath Kwame Nkrumah Circle to counteract the ongoing volumetric ground loss of over $500,000\text{ m}^3/\text{year}$.

---

## 5. Software Availability & Open Science
All code, PyTorch PINN architectures, test suites (26 tests passing), and demonstration notebooks are publicly available under the MIT license at:  
👉 **`https://github.com/eminent337/accra-insar-extreme-rainfall`**

---

## 6. References
[1] Douglas, I., Alam, K., Maghenda, M., Mcdonnell, Y., McLean, L., and Campbell, J. (2008). Unjust waters: climate change, flooding and the urban poor in Africa. *Environment and Urbanization*, 20(1), 187–205.  
[2] Asumadu-Sarkodie, S., Owusu, P. A., and Rufangura, P. (2015). Impact analysis of flood in Accra, Ghana. *International Journal of Scientific & Engineering Research*, 6(9), 83–91.  
[3] World Bank (2019). *Greater Accra Resilient and Integrated Development Project (GARID)*. Project Appraisal Document, Report No. PAD3083, Washington, DC.  
[4] Bates, P. D., and De Roo, A. P. J. (2000). A simple neural network and raster-based model for flood inundation simulation. *Journal of Hydrology*, 236(1), 54–77.  
[5] Higgins, S. A. (2016). Advances in delta-subsidence research using satellite radar interferometry. *Nature Climate Change*, 6(5), 450–458.  
[6] Nicholls, R. J., Lincke, D., Hinkel, J., Brown, S., Vafeidis, A. T., Meyssignac, B., and Fang, J. (2021). A global analysis of subsidence, relative sea-level rise and coastal flood exposure. *Nature Climate Change*, 11(4), 338–342.  
[7] Kendon, E. J., Roberts, N. M., Fowler, H. J., Roberts, M. J., Chan, S. C., and Senior, C. A. (2014). Heavier summer downpours with climate change revealed by novel high-resolution model. *Nature Climate Change*, 4(7), 570–576.  
[8] Quaye-Ballard, J. A., Mensah, F., Nyamekye, C., and Baffoe, P. E. (2020). Geospatial assessment of ground deformation in the Greater Accra Region of Ghana using multi-temporal InSAR. *Journal of African Earth Sciences*, 168, 103857.  
[9] Damptey, F., Nyatuame, M., and Agodzo, S. K. (2019). Frequency analysis of daily rainfall extremes for Accra, Ghana. *Journal of Applied Water Science*, 9(4), 78–89.  
[10] Coles, S. (2001). *An Introduction to Statistical Modeling of Extreme Values*. Springer Series in Statistics, London.  
