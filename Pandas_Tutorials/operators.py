# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 09:49:49 2018

@author: Richie
"""
## operators

# Define variables
my_kitchen = 18.0
your_kitchen = 14.0

# my_kitchen bigger than 10 and smaller than 18?

print(my_kitchen>10 and my_kitchen < 18)
# my_kitchen smaller than 14 or bigger than 17?
print(my_kitchen<14 or my_kitchen>17)

# Double my_kitchen smaller than triple your_kitchen?
print(2*my_kitchen<3*your_kitchen)

## operators for numpy arrays
# Create arrays
import numpy as np
my_house = np.array([18.0, 20.0, 10.75, 9.50])
your_house = np.array([14.0, 24.0, 14.25, 9.0])

# my_house greater than 18.5 or smaller than 10

print(np.logical_or(my_house>18.5, my_house<10))
# Both my_house and your_house smaller than 11
print(np.logical_and(my_house<11, your_house<11))

## if, elif, else

# Define variables
room = "kit"
area = 14.0

# Define variables
room = "kit"
area = 14.0

# if-else construct for room
if room == "kit" :
    print("looking around in the kitchen.")
else :
    print("looking around elsewhere.")

# if-else construct for area
if area > 15 :
    print("big place!")
else : 
    print("pretty small.")

# if-elif-else construct for room
if room == "kit" :
    print("looking around in the kitchen.")
elif room == "bed":
    print("looking around in the bedroom.")
else :
    print("looking around elsewhere.")

# if-elif-else construct for area
if area > 15 :
    print("big place!")
elif area >10 :
    print ("medium size, nice!")
else :
    print("pretty small.")    
    
## Logical operators  with pandas dataframe
    
    # Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Extract drives_right column as Series: dr
dr = cars["drives_right"]

# Use dr to subset cars: sel

sel = cars[dr] 

# Print sel
print(sel)

# Alternative way to subset
# Convert code to a one-liner
sel = cars[cars['drives_right']]

# Print sel
print(sel)

# Create car_maniac: observations that have a cars_per_cap over 500
cpc = cars['cars_per_cap']
many_cars = cpc>500
car_maniac = cars[many_cars]
# Print car_maniac
print(car_maniac)

# Create medium: observations with cars_per_cap between 100 and 500
cpc = cars['cars_per_cap']
between = np.logical_and(cpc > 100 , cpc < 500)
medium = cars[between]


# Print medium
print(medium)































































    