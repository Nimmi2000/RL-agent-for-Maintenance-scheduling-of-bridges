# RL-agent-for-Maintenance-scheduling-of-bridges
This repository contains a working flow model for training RL agent capable of autonomous maintenance decision making prioritising both age of bridges and captial involved in renovations

![image](https://github.com/user-attachments/assets/a296c30c-249e-4547-92e2-bce4299a458e)

![image](https://github.com/user-attachments/assets/be2d6327-aa8c-4a4b-b792-a0bd3f9a9779)

States :
	1) Traffic intensity [4000,20000]
	2) Latitude
	3) Longitude
	4) Age
	5) Aux_age
	6) Failure Probability [Calculated based on Survival function]
	7) Survival function (Reliability) [Calculated based on aux_age]
  8) Capital

Actions :
	1) Nothing
	2) Monitoring
	3) Minor Intervention
  4) Medium Intervention
  5) Major Intervention
  6) Replace

At each step :

Age is incremented by +1 (Assuming maintenance action is taken every 1 years)

Aux age is modified based on the actions (Check deterioration_model.py file)



