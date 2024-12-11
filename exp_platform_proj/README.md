## README

This project has multiple goals.  
(1) generate synthetic data where the ground-truth causal structure is known.  
(2) show data dictionary usage  
(3) estimate interventions and counterfactuals from observational data by hypothesizing an appropriate DAG and fitting data to SEM.  
(4) compare using VAEs on behavioral constructs with outside sources to manually defining a score.  
(5) demonstrate experimental platform, and how to choose optimal intervention.  

Potentially:  
(6) compare prediction and general capability with ML prediciton model.  
(7) find ground truth statement by making high correlation features independent when stratifying by right variables, (maybe when startifiyng by behavioral constructs or other VAE found functions?) this help back up stategic business directions/investments.  
(8) add to synthetic data's behavioral construct more complexity to be estimated by VAE,  
    i.e. for trust: make neighbourhoods code with different trusts that actually depend on other factors related to neighbourhood, show that  
    when using autoencoder on neghibourhood stats you can make the neighbourhood itself independent.  
(9) something about decision trees?  
(10) have loss function defined to be able to add constraints such as occam's razor bias, fal-pos weighting, etc   
