In this power monitoring system, the power of the bulb is monitored using a machine learning technique. This system consists of raspberry pi3 b+, ADC, 6volt transformer, bulb holder and current sensor. In which raspberry pi plays as mini-computer to this system, the ADC (MCP 3008) IC is used to convert the analog value to digital value, the 6-volt transformer is used to reduce the 230 volts to 6 volts and last the bulb holder is used to hold the bulb. 

Existing system
In the existing system, the classification is done based on manually calculating the analog value and do the calculation based upon it.

Proposed System
In this proposed system, machine learning is used with Sci-kit learn, in which linear regression is used to classify the power of the bulb.

In this project, the data set is created for the testing bulbs, the first data set consists of four classes such as NOLOAD, MEDIUM, NORMAL and RISK. The second data set consist of three class such as NOLOAD, NORMAL, RISK. From there the class and the parameter are split separately and send them in Logistic Regression then the prediction is done based on the trained model the class is printed as output in a display.

